import classifim.layers
import classifim.pipeline
import classifim.w
import classifim_utils
import datetime
import numpy as np
import sys
import torch
import torch.nn as nn

class TwelveSitesNN2(nn.Module):
    """
    This is similar to classifim.layers.TwelveSitesNN2, but designed
    for simple binary classification, not ClassiFIM.
    """
    def __init__(self, n_sites=12, cnn_layer=classifim.layers.Cnn12Sites):
        super(TwelveSitesNN2, self).__init__()
        self.n_sites = n_sites
        self.width_scale = (n_sites / 12)**0.5
        scale_width = lambda w: int(w * self.width_scale + 0.5)

        # Process lambda
        self.lambdas_fc = nn.Sequential(
            nn.Linear(1, scale_width(32)),
            nn.ReLU(),
            nn.Linear(scale_width(32), scale_width(64)),
            nn.ReLU()
        )

        # Process zs
        self.zs_cnn = nn.Sequential(
            cnn_layer(2, 32),
            nn.ReLU(),
            cnn_layer(32, 64),
            nn.ReLU()
        )

        self.zs_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(64 + scale_width(64), scale_width(128)),
            nn.ReLU(),
            nn.Linear(scale_width(128), scale_width(64)),
            nn.ReLU(),
            nn.Linear(scale_width(64), 1)
        )

    def forward(self, lambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 1).
            zs: tensor of shape (*batch_size, n_sites, 2).

        Returns:
            output: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        batch_size = lambdas.shape[:-1]

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas)

        # Process zs
        zs_out = self.zs_cnn(zs)
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output = self.fc(combined).squeeze(dim=-1)

        return output

class BCDataLoader:
    def __init__(self, data, batch_size, device):
        """
        Args:
            data: a dictionary with the following keys:
            - sweep_lambda_index: int, index of the column in
                `scaled_lambdas` to ignore.
            - labels: (num_orig_samples,) array of labels.
            - scaled_lambdas: (num_orig_samples, num_lambdas) array of
                distribution parameters.
            - unpacked_zs: (num_orig_samples, ...) array of
                samples.
        """
        sweep_lambda_index = data["sweep_lambda_index"]
        ii = (data["labels"] != 0)
        lambdas = data["scaled_lambdas"][ii].astype(np.float32)
        lambdas = np.delete(lambdas, sweep_lambda_index, axis=1)
        self.lambdas = torch.from_numpy(lambdas).to(device)
        self.zs = torch.from_numpy(
                data["unpacked_zs"][ii].swapaxes(1, 2)
            ).to(device)
        labels = (data["labels"][ii] + 1) / 2
        assert np.all((labels == 0) | (labels == 1))
        self.labels = torch.from_numpy(labels).to(device)
        self.ii = torch.randperm(self.lambdas.shape[0], device=device)
        self.num_samples = self.lambdas.shape[0]
        self.batch_size = batch_size
        assert self.ii.shape == (self.num_samples,)
        assert self.lambdas.shape == (self.num_samples, 1)
        n_sites = self.zs.shape[1]
        assert self.zs.shape == (self.num_samples, n_sites, 2)
        assert self.labels.shape == (self.num_samples,)

    def reshuffle(self):
        torch.randperm(self.num_samples, out=self.ii)

    def __iter__(self):
        self.pos = 0
        return self

    def _retrieve_samples(self, i0, i1):
        return self.lambdas[i0:i1], self.zs[i0:i1], self.labels[i0:i1]

    def __next__(self):
        if self.pos >= self.num_samples:
            self.pos = 0
            raise StopIteration()
        i0 = self.pos
        i1 = min(self.pos + self.batch_size, self.num_samples)
        self.pos = i1
        return self._retrieve_samples(i0, i1)

class BCPipeline(classifim.pipeline.Pipeline):
    def __init__(self, config):
        self.n_sites = config["n_sites"]
        super().__init__(config)

    def unpack_zs(self, zs):
        """
        Converts an array of (2 * self.n_sites)-bit integers into an array of Z
            values in {-1, 1}.

        Input format: 0bTT...TTBB...BB
        Bit Bj of zs is (zs & (1 << j)) > 0.
        Bit Tj of zs is (zs & (1 << (self.nsites + j))) > 0.

        Output format: [[Z_{B0}, Z_{B1}, ..., Z_{B11}], [Z_{T0}, ..., Z_{T11}]]
        Z_{Bj} of zs_out is zs_out[..., 0, j].
        Z_{Tj} of zs_out is zs_out[..., 1, j].
        """
        zs_bin = classifim_utils.unpackbits(zs, 2 * self.n_sites)
        zs_bin = (1 - 2 * zs_bin).reshape(*zs.shape, 2, self.n_sites)
        return zs_bin.astype(np.float32)

    def _transform(self, dataset):
        dataset["unpacked_zs"] = self.unpack_zs(dataset["zs"])
        sweep_lambda_index = self.config["sweep_lambda_index"]
        dataset["sweep_lambda_index"] = sweep_lambda_index
        label_lambdas = dataset["scaled_lambdas"][:, sweep_lambda_index]

        # 1.0 if label_lambdas > 0.5
        # 0.0 if label_lambdas between -0.5 and 0.5
        # -1.0 if label_lambdas < -0.5
        dataset["labels"] = np.where(
            label_lambdas > 0.5,
            1.0,
            np.where(
                label_lambdas < -0.5,
                -1.0,
                0.0)).astype(np.float32)

    def fit_transform(self, dataset):
        """
        Fits the scaling parameters to the dataset and transforms it.
        """
        self.fit_transform_lambdas(dataset)
        self._transform(dataset)

    def transform(self, dataset):
        self.transform_lambdas(dataset)
        self._transform(dataset)

    def _get_data_loader(self, dataset, is_train, batch_size=None, device=None):
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        return BCDataLoader(
            data=dataset,
            batch_size=batch_size,
            device=device)

    def construct_model(self, device=None):
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = TwelveSitesNN2(**model_init_kwargs)
        if device is not None:
            model = model.to(device)
        self.model = model

    # train_nn is the same as in classifim.w.Pipeline:
    train_nn = classifim.w.Pipeline.train_nn
    test_nn = classifim.w.Pipeline.test_nn

