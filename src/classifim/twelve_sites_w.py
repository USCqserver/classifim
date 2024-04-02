import classifim.layers
import classifim.pipeline
import classifim.twelve_sites_bc
import classifim_utils
import datetime
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchCnn12Sites(nn.Module):
    """
    Convolutional layer for 12-site lattice.

    The lattice sites can be indexed by a single
    integer i in the range 0 to 11. There are 2
    types of edges: nn (nearest neighbor) and
    nnn (next nearest neighbor). The sites i and j are
    - connected by an nn edge if |i - j| % 12 = 1 or 3,
    - connected by an nnn edge if |i - j| % 12 = 2 or 4.

    Correspondinly, in this layer information can propagate in 3 ways:
    - self: information propagates within each site,
    - nn: information propagates along nn edges,
    - nnn: information propagates along nnn edges.
    """
    def __init__(self, layer_batch_size, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.Tensor(
            layer_batch_size, 3 * in_channels, out_channels))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weights, a=0)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (*batch_size, layer_batch_size, 12, in_channels).

        Returns:
            output: tensor of shape
                (*batch_size, layer_batch_size, 12, out_channels).
        """
        # Message passing part
        # nn edges add {-3, -1, 1, 3} to site index.
        # nnn edges addd {-4, -2, 2, 4} to site index.
        # We use the following identities to optimize the computation:
        # nn: {-3, -1, 1, 3} = {-1, 3} + {0, -2}
        # nnn: {-4, -2, 2, 4} = {-2, 4} + {0, -2}
        nn_contrib0 = torch.roll(x, shifts=(-1,), dims=(-2,)) + \
                torch.roll(x, shifts=(3,), dims=(-2,))
        nnn_contrib0 = torch.roll(x, shifts=(-2,), dims=(-2,)) + \
                torch.roll(x, shifts=(4,), dims=(-2,))
        nn_nnn_contrib0 = torch.cat((nn_contrib0, nnn_contrib0), dim=-1)
        nn_nnn_contrib1 = nn_nnn_contrib0 + \
                torch.roll(nn_nnn_contrib0, shifts=(-2,), dims=(-2,))

        # Combine contributions
        all_contribs1 = torch.cat((x, nn_nnn_contrib1), dim=-1)
        output = torch.matmul(all_contribs1, self.weights)

        return output

class TwelveSitesNN2(nn.Module):
    """
    This is similar to classifim.layers.TwelveSitesNN2, but designed
    for van Nieuwenburg's W, training a separate model for each
    value of lambda_fixed and lambda_sweep_critical.
    """
    def __init__(self, n_sites=12, layer_bs=63, cnn_layer=BatchCnn12Sites):
        super(TwelveSitesNN2, self).__init__()
        self.n_sites = n_sites
        self.width_scale = (n_sites / 12)**0.5
        scale_width = lambda w: int(w * self.width_scale + 0.5)

        self.zs_cnn = nn.Sequential(
            cnn_layer(layer_bs, 2, 32),
            nn.ReLU(),
            cnn_layer(layer_bs, 32, 64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            classifim.w.BatchLinear(layer_bs, 64, scale_width(64)),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, scale_width(64), scale_width(64)),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, scale_width(64), 1)
        )

    def zs_pool(self, x):
        """Remove dimension -2 of x by applying AvgPool1d."""
        x = x.transpose(-1, -2)
        batch_size = x.shape[:-1]
        x = F.adaptive_avg_pool1d(
            x.view(-1, batch_size[-1], x.shape[-1]), 1)
        return x.view(*batch_size)

    def forward(self, zs):
        """
        Args:
            zs: tensor of shape (*batch_size, n_sites, 2).

        Returns:
            output: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        # Process zs
        zs_out = self.zs_cnn(zs)
        zs_out = self.zs_pool(zs_out)

        # Final classification
        output = self.fc(zs_out).squeeze(dim=-1)

        return output

class WDataLoader(classifim.w.DataLoader):
    """
    Produces batches of data for training W of the following form:
        - zs: (batch_size, 1, n_sites, 2) array of samples.
        - labels: (batch_size, layer_bs) array of labels.
    """
    def _init_zs(self, data, device, idx):
        zs = data["unpacked_zs"][idx].astype(np.float32)
        zs = zs.swapaxes(1, 2)[:, np.newaxis, :, :]
        n_sites = zs.shape[2]
        assert zs.shape == (num_samples, 1, n_sites, 2)
        self.zs = torch.from_numpy(zs).to(device)

    def _retrieve_zs(self, idx):
        return (self.zs[idx], )

class WPipeline(classifim.w.Pipeline, classifim.twelve_sites_bc.BCPipeline):
    def _transform(self, dataset):
        dataset["unpacked_zs"] = self.unpack_zs(dataset["zs"])
        classifim.w.Pipeline._transform(self, dataset)

    def _get_data_loader(
            self, dataset, is_train, **kwargs):
        return classifim.w.Pipeline._get_data_loader(
            self, dataset, is_train, cls=WDataLoader, **kwargs)

    def construct_model(self, device=None):
        return classifim.w.Pipeline.construct_model(
            self, device=device, cls=TwelveSitesNN2)

