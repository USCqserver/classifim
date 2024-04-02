import classifim_utils
import numpy as np
import os.path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import classifim.pipeline
from classifim.data_loading import enumerate_distinct_rows
from classifim_utils import load_tensor

class MnistZsTransform:
    def __init__(self):
        self.transform_matrix = None

    def fit_transform(self, zs_probs, zs_ids, mnist_labels, num_classes=10):
        """
        Prepare the data for ClassiFIM BC neural network.

        Args:
            zs_probs: np.ndarray of shape (num_samples, num_classes) of
                output probabilities of MNIST_CNN models.
            zs_ids: np.ndarray of shape (num_samples,) of indices of
                MNIST images corresponding to zs_probs.
            mnist_labels: np.ndarray with labels of MNIST images.
            num_classes: Number of classes in the dataset (10 for MNIST).

        Returns:
            scaled_zs: np.ndarray of shape (num_samples, 2, num_classes)
                of scaled output and target probabilities.
        """
        num_samples = zs_probs.shape[0]
        assert zs_probs.shape == (num_samples, num_classes)
        assert np.all(0 <= zs_probs)
        assert np.all(zs_probs <= 1)
        assert len(mnist_labels.shape) == 1
        num_labels = mnist_labels.shape[0]
        assert num_labels > np.max(zs_ids)
        assert np.allclose(np.unique(mnist_labels), np.arange(num_classes))

        zs_onehot = np.zeros((num_samples, num_classes))
        zs_onehot[np.arange(num_samples), mnist_labels[zs_ids]] = 1
        # Temporarily increase precision:
        zs_probs = zs_probs.astype(np.float64)
        probs_mean = np.mean(zs_probs)
        probs_std = np.std(zs_probs)
        onehot_mean = 1 / num_classes
        onehot_std = np.sqrt(onehot_mean * (1 - onehot_mean))

        # Scale:
        zs_probs = (zs_probs - probs_mean) / probs_std
        zs_onehot = (zs_onehot - onehot_mean) / onehot_std
        corr = np.mean(zs_probs * zs_onehot)
        zs_probs = zs_probs - corr * zs_onehot
        assert np.allclose(np.std(zs_probs)**2, 1 - corr**2)
        zs_probs_scale = (1 - corr**2)**0.5
        zs_probs = zs_probs / zs_probs_scale
        # transform_matrix @ (1, orig_probs, orig_onehot) = (zs_probs, zs_onehot)
        self.transform_matrix = np.array([
            [(-probs_mean / probs_std + corr * onehot_mean / onehot_std)
                 / zs_probs_scale,
             1 / (probs_std * zs_probs_scale),
             -corr / (onehot_std * zs_probs_scale)],
            [-onehot_mean / onehot_std, 0, 1 / onehot_std]])
        return np.stack([zs_probs, zs_onehot], axis=1).astype(np.float32)

    def transform(self, zs_probs, zs_ids, mnist_labels, num_classes=10):
        """
        Apply learned transform.

        Args:
            zs_probs: np.ndarray of shape (num_samples, num_classes) of
                output probabilities of MNIST_CNN models.
            zs_ids: np.ndarray of shape (num_samples,) of indices of
                MNIST images corresponding to zs_probs.
            mnist_labels: np.ndarray with labels of MNIST images.
            num_classes: Number of classes in the dataset (10 for MNIST).

        Returns:
            scaled_zs: np.ndarray of shape (num_samples, 2, num_classes)
                of scaled output and target probabilities.
        """
        num_samples = zs_probs.shape[0]
        zs_onehot = np.zeros((num_samples, num_classes))
        zs_onehot[np.arange(num_samples), mnist_labels[zs_ids]] = 1
        zs = np.stack([zs_probs, zs_onehot], axis=1)
        offset = self.transform_matrix[np.newaxis, :, 0, np.newaxis]
        m = self.transform_matrix[np.newaxis, :, 1:, np.newaxis]
        return (offset + np.sum(m * zs[:, np.newaxis, :, :], axis=2)).astype(np.float32)


# ClassiFIM BC models. All models satisfy the following properties:
# - Model inputs are
#   - lambda0s: (batch_size, num_lambdas) distribution parameters
#       for the midpoint.
#   - dlambdas: (batch_size, num_lambdas) distance between two points
#       in the parameter space.
#   - zs: (batch_size, 2, num_classes) samples.
#   - zxs: (batch_size, 1, img_width, img_height) images.
# - Output is a tensor of shape (batch_size, num_lambdas). For convenience,
#   the second, postprocessed, output is provided, which is used in the
#   cross-entropy loss:
#   - output1: tensor of shape (*batch_size, num_lambdas).
#         Represents the output of NN before sigmoid.
#         The purpose of the model is to make this output available.
#   - output2: tensor of shape (*batch_size,).
#         Represents the final output of NN.
#         This output is used for cross-entropy loss.
# - Defaults:
#   - num_lambdas = 2
#   - num_classes = 10

def preprocess_lambdas(lambda0s, dlambdas):
    """
    Common preprocessing for most of the models.

    Only support num_lambdas = 2.
    """
    return torch.cat([
            dlambdas[..., 0, np.newaxis]**2 - 2/3,
            dlambdas[..., 1, np.newaxis]**2 - 2/3,
            dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
            3.5 * lambda0s
        ], dim=-1)

class Model0(torch.nn.Module):
    def __init__(self, num_lambdas=2, num_classes=10):
        super().__init__()
        assert num_lambdas == 2, (
            f"num_lambdas = {num_lambdas} != 2 is not yet supported")
        self.num_lambdas = num_lambdas
        self.num_classes = num_classes

        # Process lambda0s and dlambdas.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # Process zs.
        self.zs_fc = nn.Sequential(
            nn.Linear(2 * num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU())

        # Process zxs (MNIST images).
        # Start shape (per row): (1, 28, 28)
        self.zxs_cnn = nn.Sequential(
            nn.Conv2d(1, 24, 5, 1), # (24, 24, 24)
            nn.MaxPool2d(2), # (24, 12, 12)
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, 1), # (32, 10, 10)
            nn.MaxPool2d(2), # (32, 5, 5)
            nn.ReLU(),
            nn.Flatten(), # (800,)
            nn.Linear(800, 128),
            nn.ReLU())

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_lambdas))

    def forward(self, lambda0s, dlambdas, zs, zxs):
        lambdas_in = preprocess_lambdas(lambda0s, dlambdas)
        lambdas_out = self.lambdas_fc(lambdas_in)
        zs_in = zs.view(*zs.shape[:-2], -1)
        zs_out = self.zs_fc(zs_in)
        zxs_out = self.zxs_cnn(zxs)
        combined = torch.cat([lambdas_out, zs_out, zxs_out], dim=-1)
        output1 = self.fc(combined)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        return output1, output2


class MnistDataLoader:
    """
    When used with GPU, this loads the data to GPU memory in its entirety
    and generates shuffled pairs on GPU.

    Inspired by TwelveSitesDataLoader5.
    """
    def __init__(self, data, batch_size, device):
        """
        Args:
            data: a dictionary with the following keys:
            - scaled_lambdas: (num_orig_samples, num_lambdas) array of
                distribution parameters.
            - scaled_zs: (num_orig_samples, ...) array of
                samples.
            - zs_ids: (num_orig_samples,) array of indices to `xs`
            - xs: additional data to index on the fly.
        """
        lambdas = torch.from_numpy(data["scaled_lambdas"].astype(np.float32))
        self.lambdas = lambdas.to(device=device)
        zs = torch.from_numpy(data["scaled_zs"].astype(np.float32))
        assert zs.shape[0] == lambdas.shape[0]
        self.zs = zs.to(device=device)
        # PyTorch: tensors used as indices must be long, byte or bool tensors:
        self.zs_ids = torch.from_numpy(data["zs_ids"]).to(
            dtype=torch.long, device=device)
        self.xs = torch.from_numpy(data["xs"]).to(
            dtype=torch.float32, device=device)
        category = enumerate_distinct_rows(data["scaled_lambdas"])
        self.category = torch.from_numpy(category).to(device=device)
        assert batch_size % 2 == 0, f"batch_size must be even, not {batch_size}"
        self.pair_batch_size = batch_size // 2
        self.device = device
        self.max_sample_pairs = self.lambdas.shape[0]
        self.num_sample_pairs = None
        self.reshuffle()
        self.pos = 0

    def reshuffle(self):
        ii0 = torch.randperm(self.max_sample_pairs, device=self.device)
        ii1 = torch.randperm(self.max_sample_pairs, device=self.device)
        i_valid = (self.category[ii0] != self.category[ii1]).nonzero().view(-1)
        self.ii0 = ii0[i_valid]
        self.ii1 = ii1[i_valid]
        self.num_sample_pairs = len(i_valid)

    def __iter__(self):
        self.pos = 0
        return self

    def _retrieve_samples(self, i0, i1):
        """
        Retrieve 2 * (i1 - i0) samples from the dataset.

        Note: this exploits the model structure that the model is symmetric
            with respect to simultaneously changing
            dlambda -> -dlambda and label -> -label.
            We always set label to 1 and change the sign of dlambda if needed.

        Returns: tuple with the following 4 tensors:
            lambda0
            dlambda
            zs
            zxs
            label
        """
        ii0 = self.ii0[i0:i1]
        ii1 = self.ii1[i0:i1]
        lambda_m = self.lambdas[ii0] # lambda0 - dlambda / 2
        lambda_p = self.lambdas[ii1] # lambda0 + dlambda / 2
        lambda0 = (lambda_p + lambda_m) / 2
        dlambda = lambda_p - lambda_m
        zs_m = self.zs[ii0]
        zs_p = self.zs[ii1]
        zxs_m = self.xs[self.zs_ids[ii0]]
        zxs_p = self.xs[self.zs_ids[ii1]]
        label = torch.ones(
            2 * (i1 - i0), dtype=torch.float32, device=self.device)
        return (
            torch.cat((lambda0, lambda0), dim=0),
            torch.cat((-dlambda, dlambda), dim=0),
            torch.cat((zs_m, zs_p), dim=0),
            torch.cat((zxs_m, zxs_p), dim=0),
            label)

    def __next__(self):
        if self.pos >= self.num_sample_pairs:
            self.pos = 0
            raise StopIteration()
        i0 = self.pos
        i1 = min(self.pos + self.pair_batch_size, self.num_sample_pairs)
        self.pos = i1
        return self._retrieve_samples(i0, i1)

class Pipeline(classifim.pipeline.Pipeline):
    """
    Pipeline for MNIST-CNN.
    """
    def fit_transform(self, dataset):
        self.fit_transform_lambdas(dataset)
        self.zs_transform = MnistZsTransform()
        dataset["scaled_zs"] = self.zs_transform.fit_transform(
            zs_probs=dataset["zs_probs"],
            zs_ids=dataset["zs_ids"],
            mnist_labels=dataset["mnist_labels"])
        dataset["xs"] = dataset["mnist_inputs"] # aka images

    def transform(self, dataset):
        self.transform_lambdas(dataset)
        dataset["scaled_zs"] = self.zs_transform.transform(
            zs_probs=dataset["zs_probs"],
            zs_ids=dataset["zs_ids"],
            mnist_labels=dataset["mnist_labels"])
        dataset["xs"] = dataset["mnist_inputs"]

    def _get_data_loader(self, dataset, is_train, batch_size=None, device=None):
        """
        Construct a DataLoader for the training or test set.

        Note that the returned object is not torch.utils.data.DataLoader
        for efficiency reasons.
        """
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        # Data we feed to ClassiFIM dataset construction:
        # (scaled_lambdas, scaled_zs) from self.dataset_train.
        return MnistDataLoader(
            data=dataset,
            batch_size=batch_size,
            device=device)

    def construct_model(self, device=None):
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = Model0(**model_init_kwargs)
        if device is not None:
            model = model.to(device)
        self.model = model

    def get_fim_loader(self, dataset, batch_size=None, device=None):
        """
        Returns a generator that yields batches of data for FIM estimation.
        """
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        zs = torch.from_numpy(dataset["scaled_zs"]).to(device)
        zs_ids = torch.from_numpy(dataset["zs_ids"]).to(
            dtype=torch.long, device=device)
        xs = torch.from_numpy(dataset["xs"]).to(device)
        num_samples = zs.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_samples)
            yield (
                slice(batch_start, batch_end),
                (zs[batch_start:batch_end], xs[zs_ids[batch_start:batch_end]]))

def load_mnist_data(config, verbose=True):
    dataset_filename = config["dataset_filename"]
    if verbose:
        print(f"Loading dataset from '{dataset_filename}'")
    npz_dataset = np.load(dataset_filename)
    dataset_dirname = os.path.dirname(dataset_filename)
    mnist_labels_filename = os.path.join(
        dataset_dirname,
        npz_dataset["mnist_labels_filename"].item())
    if verbose:
        print(f"Loading MNIST labels from '{mnist_labels_filename}'")
    mnist_labels = load_tensor(
        mnist_labels_filename,
        tensor_name="labels")
    assert len(mnist_labels.shape) == 1, (
        f"Expected 1D labels, got {mnist_labels.shape}.")
    num_mnist_rows = mnist_labels.shape[0]
    mnist_inputs_filename = os.path.join(
        dataset_dirname,
        npz_dataset["mnist_inputs_filename"].item())
    if verbose:
        print(f"Loading MNIST inputs from '{mnist_inputs_filename}'")
    mnist_inputs = load_tensor(
        mnist_inputs_filename,
        tensor_name="inputs")
    assert mnist_inputs.shape == (num_mnist_rows, 1, 28, 28), (
        f"mnist_inputs.shape = {mnist_inputs.shape} "
        + f"!= ({num_mnist_rows}, 1, 28, 28)")
    return npz_dataset, mnist_inputs, mnist_labels

def flatten_dict_for_npz(
        d, key_prefix="", sep=".", permissive=False, target=None):
    """
    Flattens a nested dictionary to prepare for saving to npz file.

    Args:
        d: dictionary to flatten.
        key_prefix: prefix to prepend to all keys. Should include trailing
            separator if desired.
        sep: separator to use between keys.
        permissive: if True, ignore object imperfect for npz format.
        target: target dictionary to write to. If None, a new dictionary is
            created.
    """
    res = target if target is not None else {}
    if not permissive:
        assert isinstance(key_prefix, str), (
            f"key_prefix must be a string, got {type(key_prefix)}")
        assert isinstance(sep, str), (
            f"sep must be a string, got {type(sep)}")
    for k, v in d.items():
        new_key = key_prefix + k
        if isinstance(v, dict):
            flatten_dict_for_npz(v, new_key + sep, sep, res)
        else:
            if isinstance(v, set):
                v = list(v)
            assert new_key not in res, (
                f"Key {new_key} already exists in res.")
            try:
                v_np = np.array(v)
            except Exception:
                if not permissive:
                    raise
                v_np = None
            if not permissive:
                assert v_np.dtype != object, (
                    f"Key {new_key} has dtype object, which is not supported "
                    + "by npz format (without pickle).")
            if v_np is not None and v_np.dtype != object:
                res[new_key] = v_np
            else:
                res[new_key] = v
    return res
