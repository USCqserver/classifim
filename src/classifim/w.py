"""
Utils for van Nieuwenburg's W.
"""

import classifim.pipeline
import datetime
import numpy as np
import torch
import torch.nn as nn

def transform_dataset(dataset, sweep_lambda_index, scalar_keys=None):
    """
    Transforms dataset to convert `lambdas` parameterization
    to (lambda_fixed, lambda_sweep).

    Args:
        dataset (dict): dataset to transform (in-place)
        sweep_lambda_index (int): index of the lambda to sweep (0 or 1)
        scalar_keys (list): list of keys to treat as scalars

    Returns:
        None
    """
    assert sweep_lambda_index in [0, 1]
    assert dataset["lambdas"].shape[-1] == 2
    fixed_lambda_index = 1 - sweep_lambda_index
    dataset["sweep_lambda_index"] = sweep_lambda_index
    lambda_fixed = dataset["lambdas"][:, fixed_lambda_index]
    dataset["lambda_fixed"] = lambda_fixed
    lambda_sweep = dataset["lambdas"][:, sweep_lambda_index]
    dataset["lambda_sweep"] = lambda_sweep
    ii = np.argsort(lambda_fixed)
    scalar_keys = set(scalar_keys or [])
    scalar_keys.add("sweep_lambda_index")
    for key in dataset.keys():
        if key in scalar_keys:
            continue
        dataset[key] = dataset[key][ii]
    lambda_sweep_thresholds = np.unique(lambda_sweep)
    lambda_sweep_thresholds = (
            lambda_sweep_thresholds[:-1] + lambda_sweep_thresholds[1:]) / 2
    dataset["lambda_sweep_thresholds"] = lambda_sweep_thresholds
    lambda_fixed_unique, lambda_fixed_idx = np.unique(
        dataset["lambda_fixed"], return_index=True)
    lambda_fixed_idx = np.concatenate(
        [lambda_fixed_idx, [len(lambda_fixed)]])
    dataset["lambda_fixed_unique"] = lambda_fixed_unique
    dataset["lambda_fixed_idx"] = lambda_fixed_idx

class DataLoader:
    """
    In-memory data loader for W.
    """
    def __init__(self, data, i_start, i_end, batch_size, device, **kwargs):
        """
        Args:
            data: a dictionary with the following keys:
            - lambda_sweep: array of shape (num_orig_samples,)
                (to be used for labels).
            - lambda_sweep_thresholds: thresholds to use for labels.
            - unpacked_zs: (num_orig_samples, n_sites, 2) array of
                samples.
        """
        num_samples = i_end - i_start
        self.num_samples = num_samples

        lambda_sweep = data["lambda_sweep"][
                i_start:i_end, np.newaxis].astype(np.float32)
        self.lambda_sweep = torch.from_numpy(lambda_sweep).to(device)
        assert self.lambda_sweep.shape == (num_samples, 1)

        lambda_sweep_thresholds = data["lambda_sweep_thresholds"]
        self.lambda_sweep_thresholds = torch.from_numpy(
                lambda_sweep_thresholds[np.newaxis, :].astype(np.float32)
            ).to(device)
        self.layer_bs = self.lambda_sweep_thresholds.shape[1]
        assert self.lambda_sweep_thresholds.shape == (
                1, self.layer_bs)

        self._init_zs(
            data, device, idx=range(i_start, i_end), **kwargs)

        self.ii = torch.randperm(num_samples, device=device)
        self.batch_size = batch_size
        assert self.ii.shape == (num_samples,)

    def reshuffle(self):
        torch.randperm(self.num_samples, out=self.ii)

    def __iter__(self):
        self.pos = 0
        return self

    def _retrieve_zs(self, ii):
        """
        Retrieve zs for indices in ii.

        Args:
            ii: 1D tensor of indices.

        Returns: tuple, each element of which is a tensor of shape
            (len(ii), ...)
        """
        raise NotImplementedError()

    def _retrieve_samples(self, i0, i1):
        labels = (self.lambda_sweep[i0:i1] > self.lambda_sweep_thresholds)
        return (
            *self._retrieve_zs(range(i0, i1)),
            labels.to(torch.float32))

    def __next__(self):
        if self.pos >= self.num_samples:
            self.pos = 0
            raise StopIteration()
        i0 = self.pos
        i1 = min(self.pos + self.batch_size, self.num_samples)
        self.pos = i1
        return self._retrieve_samples(i0, i1)

    @classmethod
    def from_lambda_fixed_i(
            cls, dataset, batch_size, device, lambda_fixed_i, **kwargs):
        i_start = dataset["lambda_fixed_idx"][lambda_fixed_i]
        i_end = dataset["lambda_fixed_idx"][lambda_fixed_i + 1]
        return cls(
            data=dataset,
            i_start=i_start,
            i_end=i_end,
            batch_size=batch_size,
            device=device,
            **kwargs)


class BatchLinear(nn.Module):
    def __init__(self, layer_bs, in_features, out_features, bias=True):
        super().__init__()
        self.layer_bs = layer_bs
        self.in_features = in_features
        self.out_features = out_features

        # Create weight tensors for each group
        self.weights = nn.Parameter(torch.Tensor(
            layer_bs, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(layer_bs, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=0)
        if self.bias is not None:
            bound = 1 / (self.in_features)**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x shape: (*batch_size, layer_bs, in_features)
        # weights shape: (layer_bs, in_features, out_features)

        # Apply linear transformation
        x_out = torch.matmul(x.unsqueeze(-2), self.weights).squeeze(-2)

        # Add bias if it's available
        if self.bias is not None:
            x_out += self.bias

        return x_out

class Pipeline(classifim.pipeline.Pipeline):
    def __init__(self, config):
        assert config["hold_out_test"], (
            "hold_out_test=False is not supported: "
            + "self.dataset_test slot is used for validation set, "
            + "which is used for computing the W.")
        super().__init__(config)

    def _transform(self, dataset):
        sweep_lambda_index = self.config["sweep_lambda_index"]
        transform_dataset(
            dataset, sweep_lambda_index, self.config["scalar_keys"])

    def fit_transform(self, dataset):
        self._transform(dataset)
        self.lambda_sweep_thresholds = dataset["lambda_sweep_thresholds"]
        model_kwargs = self.config.get("model_init_kwargs", {})
        if "layer_bs" not in model_kwargs:
            model_kwargs["layer_bs"] = len(self.lambda_sweep_thresholds)
        self.config["model_init_kwargs"] = model_kwargs

    def transform(self, dataset):
        self._transform(dataset)
        np.testing.assert_allclose(
            self.lambda_sweep_thresholds,
            dataset["lambda_sweep_thresholds"])

    def _get_data_loader(
            self, dataset, is_train, batch_size=None, device=None,
            lambda_fixed_i=None, cls=DataLoader, **kwargs):
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        if lambda_fixed_i is None:
            lambda_fixed_i = self.config["lambda_fixed_i"]
        return cls.from_lambda_fixed_i(
            dataset, batch_size, device, lambda_fixed_i, **kwargs)

    @staticmethod
    @torch.no_grad()
    def _eval_w(data_loader, model, loss_fn, device="cuda:0", verbose=True):
        model.eval()
        model.to(device)
        num_points = 0
        sum_loss = None
        sum_accuracy = None
        num_thresholds = None
        for data in data_loader:
            torch_data = [d.to(device, non_blocking=True) for d in data]
            labels = torch_data[-1]
            torch_data = torch_data[:-1]
            output = model(*torch_data)
            cur_loss = loss_fn(output, labels).sum(dim=0)
            cur_accuracy = ((output > 0) == labels).sum(dim=0)
            if sum_accuracy is None:
                num_thresholds = labels.shape[1]
                sum_loss = torch.zeros(num_thresholds, device=device)
                sum_accuracy = torch.zeros(num_thresholds, device=device)
            assert cur_loss.shape == (num_thresholds,)
            assert cur_accuracy.shape == (num_thresholds,)
            num_points += labels.shape[0]
            sum_loss += cur_loss
            sum_accuracy += cur_accuracy
        mean_loss = sum_loss.cpu().numpy() / num_points
        mean_accuracy = sum_accuracy.cpu().numpy() / num_points
        if verbose:
            avg_loss = np.mean(mean_loss)
            avg_accuracy = np.mean(mean_accuracy)
            print(
                f"loss: {avg_loss:>8f}, accuracy: {avg_accuracy * 100:.2f}%, "
                f"num_points: {num_points}, num_thresholds: {num_thresholds}.")
        return {
            "loss": mean_loss,
            "accuracy": mean_accuracy,
            "num_points": num_points,
            "num_thresholds": num_thresholds}

    def eval_w(self):
        torch.manual_seed(self.prng.get_int_seed("eval_w"))
        test_loader = self.get_test_loader()
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        if not hasattr(self, "w") or self.w is None:
            self.w = {}
        self.w[self.config["lambda_fixed_i"]] = self._eval_w(
            test_loader, self.model, loss_fn, device=self.config["device"])

    def get_reshaped_w(self):
        indices = sorted(self.w.keys())
        num_indices = len(indices)
        assert num_indices > 0
        assert indices == list(range(num_indices)), (
            "Can only reshape w if w.keys() can be interpreted as list indices."
            + f" Got w.keys={indices}.")
        keys = list(self.w[0].keys())
        res = {}
        for key in keys:
            example_value = self.w[0][key]
            if isinstance(example_value, np.ndarray):
                res[key] = np.stack(
                    [self.w[i][key] for i in range(num_indices)])
            elif (isinstance(example_value, int)
                  or isinstance(example_value, float)):
                res[key] = np.array([
                    self.w[i][key] for i in range(num_indices)])
        return res

    @staticmethod
    def train_nn(
            train_loader, model, loss_fn, optimizer, scheduler,
            num_epochs, device="cuda:0", verbose=1):
        """
        Train BC neural network.

        Args:
            verbose: integer >= 0, larger means more verbose:
                0: no output
                1: short output (a few lines)
                2: one line per epoch
        """
        model.train()
        model.to(device)
        res = {"epoch": [], "loss": [], "learning_rate": []}

        if verbose >= 1:
            print(f"{datetime.datetime.now()}: Training started")
        if verbose == 1:
            print(f"{num_epochs=}: ", end="")
        for epoch in range(num_epochs):
            num_points = 0
            running_loss = 0.0
            for data in train_loader:
                torch_data = [d.to(device, non_blocking=True) for d in data]
                labels = torch_data[-1]
                torch_data = torch_data[:-1]

                # Forward pass
                output = model(*torch_data)
                loss = loss_fn(output, labels)
                # Backprop:
                # set_to_none optimization described at
                # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Collect statistics
                cur_num_points = torch_data[0].shape[0]
                running_loss += loss.item() * cur_num_points
                num_points += cur_num_points
            last_lr = scheduler.get_last_lr()[0]
            if verbose >= 2:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    + f"Loss: {running_loss / num_points}, LR: {last_lr:.4g}")
            elif verbose == 1:
                print(f"{epoch + 1}:{running_loss / num_points:.4g}", end=" ")
            scheduler.step()
            train_loader.reshuffle()
            res["epoch"].append(epoch)
            res["loss"].append(running_loss / num_points)
            res["learning_rate"].append(last_lr)
            res["num_points"] = num_points
        if verbose == 1:
            print()
        return res

    @staticmethod
    @torch.no_grad()
    def test_nn(data_loader, model, loss_fn, num_epochs=1, device="cuda:0"):
        return NotImplementedError("Use eval_w instead.")

    def save_w(self):
        res = {}
        v0 = next(iter(self.w.values()))
        res["num_points"] = v0["num_points"]
        res["num_thresholds"] = v0["num_thresholds"]
        array_keys = ["loss", "accuracy"]
        for array_key in array_keys:
            res[array_key] = np.stack([
                self.w[i][array_key] for i in range(len(self.w))])
        res["lambda_fixed_i"] = np.array(list(self.w.keys()))
        res["lambda_fixed"] = self.dataset_train["lambda_fixed_unique"][
            res["lambda_fixed_i"]]
        res["lambda_sweep_thresholds"] = self.lambda_sweep_thresholds
        file_name = self.config["w_filename"]
        np.savez_compressed(file_name, **res)
        print(f"{datetime.datetime.now()}: Saved W to {file_name}.")

