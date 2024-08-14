import classifim.data_loading
import classifim.pipeline
import classifim.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoaderMixin:
    def _init_zs(self, data, device):
        zs = data["samples"]
        n_bits = data["width"]
        if isinstance(n_bits, np.ndarray):
            n_bits = n_bits.item()
        assert isinstance(n_bits, int)
        # 32 is not supported because torch does not support uint32, only int32:
        assert 1 <= n_bits <= 31
        self.n_bits = n_bits
        assert len(zs.shape) == 2
        self.zs = torch.from_numpy(zs.astype(np.int32)).to(device=device)

    def _retrieve_zs(self, ii):
        zs = classifim.utils.unpackbits32_torch(
                self.zs[ii], num_bits=self.n_bits, device=self.device
            ) * 2.0 - 1.0
        return (zs, )

class DataLoader(DataLoaderMixin, classifim.data_loading.InMemoryLoader):
    pass

class FimLoader(DataLoaderMixin, classifim.data_loading.FimLoader):
    pass

class ModelNaiveCNN(nn.Module):
    """
    Very small model for 2D systems with PBC.

    Not recommended for more complex systems.
    """
    def __init__(self, height, width, num_lambdas=2):
        super().__init__()
        # We do not actually use width and height: the same CNN layer
        # can be applied to PBC lattice of any size.
        self.height = height
        self.width = width
        self.num_lambdas = num_lambdas
        if num_lambdas == 1:
            self.preprocess_lambdas = self._preprocess_lambdas_1d
        elif num_lambdas == 2:
            self.preprocess_lambdas = self._preprocess_lambdas_2d
        else:
            raise NotImplementedError(
                f"Only num_lambdas=1,2 (not {num_lambdas}) are supported.")
        self.lambdas_fc = nn.Sequential(
            nn.Linear(num_lambdas * (num_lambdas + 3) // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.zs_cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=0),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_lambdas)
        )

    def _preprocess_lambdas_1d(self, lambdas, dlambdas):
        assert lambdas.shape[-1] == 1
        lambdas_in = torch.cat([
                dlambdas**2,
                lambdas
            ], dim=-1)
        return lambdas_in

    def _preprocess_lambdas_2d(self, lambdas, dlambdas):
        assert lambdas.shape[-1] == 2
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2,
                dlambdas[..., 1, np.newaxis]**2,
                dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
                lambdas
            ], dim=-1)
        return lambdas_in

    def forward(self, lambdas, dlambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 1).
            dlambdas: tensor of shape (*batch_size, 1).
            zs: tensor of shape (*batch_size, height, width).

        Returns:
            output1: tensor of shape (*batch_size, 1).
                Represents the output of NN before sigmoid.
                The purpose of the model is to make this output available.
            output2: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        batch_size = dlambdas.shape[:-1]
        lambdas_in = self.preprocess_lambdas(lambdas, dlambdas)
        lambdas_out = self.lambdas_fc(lambdas_in)
        assert zs.shape == (*batch_size, self.height, self.width)
        assert zs.dtype == torch.float32
        zs = F.pad(zs.unsqueeze(1), (0, 2, 0, 2), mode='circular')
        zs_out = self.zs_cnn(zs)
        zs_out = F.adaptive_avg_pool2d(zs_out, 1).squeeze(dim=(-1, -2))
        assert zs_out.shape == (*batch_size, 32), (
            f"{zs_out.shape} != {(*batch_size, 32)}")

        combined = torch.cat([lambdas_out, zs_out], dim=1)
        output1 = self.fc(combined)
        assert output1.shape == dlambdas.shape
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

class Pipeline(classifim.pipeline.Pipeline):
    def _get_data_loader(self, dataset, is_train, batch_size=None, device=None):
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        return DataLoader(dataset, batch_size=batch_size, device=device)

    def construct_model(self, device=None):
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = ModelNaiveCNN(
            height=self.dataset_train["height"],
            width=self.dataset_train["width"],
            **model_init_kwargs)
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
        return FimLoader(dataset, batch_size=batch_size, device=device)
