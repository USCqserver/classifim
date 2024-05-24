import classifim.data_loading
import classifim.pipeline
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoader(classifim.data_loading.InMemoryLoader):
    def _init_zs(self, data, device):
        zs = data["zs"]
        assert len(zs.shape) == 2
        self.zs = torch.from_numpy(zs).to(device=device, dtype=torch.uint8)

    def _retrieve_zs(self, ii):
        return (self.zs[ii], )

class Model1DSimple(nn.Module):
    """
    Model for 1D systems
    """
    def __init__(
            self, n_sites=300, in_classes=6, lambda_coeff=1.0,
            dlambda_coeff=1.0):
        super().__init__()
        self.n_sites = n_sites
        self.in_classes = in_classes
        self.lambda_coeff = lambda_coeff
        self.dlambda_coeff2 = dlambda_coeff**2
        # The formula in `forward` is hard-coded so
        # other values are not supported for now:
        num_lambdas = 2

        # Process lambdas and dlambdas**2.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(num_lambdas * (num_lambdas + 3) // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Process zs
        self.zs_cnn = nn.Sequential(
            # in_classes = 6; in_width = 300
            nn.Conv1d(in_classes, 48, kernel_size=8, padding=2, stride=2),
            # width = (in_width + 2 * padding - kernel_size) // stride + 1 = 150
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=4, padding=1),
            # width = 149
            nn.ReLU()
        )

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_lambdas)
        )

    # def _pos_encoding(self):
    #     n_sites = self.n_sites
    #     x = torch.arange(n_sites, dtype=torch.float32)
    #     pos_encoding = torch.empty(
    #         (1, 2, n_sites), dtype=torch.float32)
    #     torch.cos(torch.pi * x / n_sites, out=pos_encoding[0, 0, :])
    #     torch.cos(2 * torch.pi * x / n_sites, out=pos_encoding[0, 1, :])
    #     return pos_encoding

    def forward(self, lambdas, dlambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 2).
            dlambdas: tensor of shape (*batch_size, 2).
            zs: tensor of shape (*batch_size, n_sites, 2).

        Returns:
            output1: tensor of shape (*batch_size, 2).
                Represents the output of NN before sigmoid.
                The purpose of the model is to make this output available.
            output2: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        batch_size = dlambdas.shape[:-1]
        assert lambdas.shape[-1] == 2
        dl0 = dlambdas[..., 0, None]
        dl1 = dlambdas[..., 1, None]
        lambdas_in = torch.cat([
                self.dlambda_coeff2 * (dl0**2 - 2/3),
                self.dlambda_coeff2 * (dl0 * dl1),
                self.dlambda_coeff2 * (dl1**2 - 2/3),
                self.lambda_coeff * lambdas
            ], dim=-1)

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas_in)

        # Process zs
        assert zs.shape == (*batch_size, self.n_sites), (
            f"{zs.shape} != {(*batch_size, 1, self.n_sites)}")
        assert zs.dtype == torch.uint8
        zs = (F.one_hot(zs.long(), num_classes=self.in_classes)
              .transpose(-1, -2).to(dtype=torch.float32))
        # zs = torch.cat([zs, self.zs_pos.expand(*batch_size, -1, -1)], dim=-2)
        zs_out = self.zs_cnn(zs)
        zs_out = F.adaptive_avg_pool1d(zs_out, 1).squeeze(dim=-1)
        assert zs_out.shape == (*batch_size, 64), (
            f"{zs_out.shape} != {(*batch_size, 64)}")

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output1 = self.fc(combined)
        assert output1.shape == dlambdas.shape
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

class Pipeline(classifim.pipeline.Pipeline):
    """
    Pipeline for 1D systems
    """
    def fit_transform(self, dataset):
        self.fit_transform_lambdas(dataset)
        n_sites = dataset["zs"].shape[-1]
        if "n_sites" in self.config:
            assert self.config["n_sites"] == n_sites
        else:
            self.config["n_sites"] = n_sites

    def transform(self, dataset):
        self.transform_lambdas(dataset)

    def _get_data_loader(
            self, dataset, is_train, batch_size=None, device=None):
        """
        Construct a DataLoader for the training or test set.

        Note that the returned object is not torch.utils.data.DataLoader
        for efficiency reasons.
        """
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        return DataLoader(
            dataset, batch_size=batch_size, device=device,
            subsample=self.config.get("subsample", None))

    def construct_model(self, device=None):
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = Model1DSimple(**model_init_kwargs)
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
        zs = torch.from_numpy(dataset["zs"][:, :]).to(
            device=device, dtype=torch.uint8)
        num_samples = zs.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_samples)
            yield (
                slice(batch_start, batch_end),
                (zs[batch_start:batch_end],))

    def init_model(self):
        """
        Initializes:
            self.model
            self.loss_fn
            self.optimizer
            self.scheduler
            self.train_loader
        Returns: None
        """
        torch.manual_seed(self.prng.get_int_seed("init_model"))
        device = self.config["device"]
        self.train_loader = self.get_train_loader(device=device)

        # Initialize the network, loss function, and optimizer
        self.construct_model(device=device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        max_lr = self.config.get("max_lr", 1e-2)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=max_lr/2,
            betas=self.config.get("adam_betas", (0.9, 0.99)),
            weight_decay=self.config.get("weight_decay", 1e-4))
        pct_start = self.config.get("one_cycle_lr.pct_start", 0.1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lr, pct_start=pct_start,
            epochs=self.config["num_epochs"], steps_per_epoch=1)

