import classifim.pipeline
import classifim.w
import classifim.xxz1d as xxz1d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WDataLoader(classifim.w.DataLoader):
    """
    Produces batches of data for training W of the following form:
        - zs: (batch_size, n_sites) array of samples.
        - labels: (batch_size, layer_bs) array of labels.
    """
    def _init_zs(self, data, device, idx):
        zs = data["zs"][idx]
        assert len(zs.shape) == 2
        self.zs = torch.from_numpy(zs).to(device=device, dtype=torch.uint8)

    def _retrieve_zs(self, idx):
        return (self.zs[idx], )

class Model1DSimple(nn.Module):
    """
    Similar to classifim.xxz1d.Model1DSimple, but for van Nieuwenburg's W:
    - Separate model for each value of lambda_fixed and lambda_critical.
    - Label is for binary classification of presumably two phases.
    """
    def __init__(self, n_sites=300, in_classes=6, layer_bs=1):
        super().__init__()
        self.n_sites = n_sites
        self.in_classes = in_classes
        self.layer_bs = layer_bs

        self.zs_cnn = nn.Sequential(
            nn.Conv1d(
                layer_bs * in_classes, layer_bs * 24,
                kernel_size=8, padding=2, stride=2, groups=layer_bs),
            nn.ReLU(),
            nn.Conv1d(
                layer_bs * 24, layer_bs * 48,
                kernel_size=4, padding=1, groups=layer_bs),
            nn.ReLU()
        )
        self.zs_out_nclasses = 48
        self.fc = nn.Sequential(
            classifim.w.BatchLinear(layer_bs, self.zs_out_nclasses, 64),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, 64, 64),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, 64, 1)
        )

    def forward(self, zs):
        assert zs.dtype == torch.uint8
        (batch_size, n_sites) = zs.shape
        assert n_sites == self.n_sites
        zs = (F.one_hot(zs.long(), num_classes=self.in_classes)
              .transpose(-1, -2).to(dtype=torch.float32))
        # Reshape zs to include layer_bs dimension
        zs = zs.unsqueeze(1).repeat(1, self.layer_bs, 1, 1)
        assert zs.shape == (batch_size, self.layer_bs, self.in_classes, n_sites)
        zs = zs.reshape(-1, self.layer_bs * self.in_classes, self.n_sites)
        zs_out = self.zs_cnn(zs)
        nc = self.zs_out_nclasses
        assert zs_out.shape[:-1] == (batch_size, self.layer_bs * nc), (
            f"{zs_out.shape} != ({batch_size}, {self.layer_bs} * {nc}, ?)")
        zs_out = F.adaptive_avg_pool1d(zs_out, 1).squeeze(dim=-1)
        assert zs_out.shape == (batch_size, self.layer_bs * nc)
        zs_out = zs_out.reshape(batch_size, self.layer_bs, nc)
        output = self.fc(zs_out).squeeze(dim=-1)
        assert output.shape == (batch_size, self.layer_bs)
        return output

class WPipeline(classifim.w.Pipeline, xxz1d.Pipeline):
    def _get_data_loader(
            self, dataset, is_train, **kwargs):
        return classifim.w.Pipeline._get_data_loader(
            self, dataset, is_train, cls=WDataLoader, **kwargs)

    def construct_model(self, device=None):
        return classifim.w.Pipeline.construct_model(
            self, device=device, cls=Model1DSimple)
