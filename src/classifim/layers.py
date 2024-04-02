import torch
import torch.nn as nn
import numpy as np

class Cnn12Sites(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super(Cnn12Sites, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.Tensor(3 * in_channels, out_channels))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weights, a=0)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (*batch_size, 12, in_channels).

        Returns:
            output: tensor of shape (*batch_size, 12, out_channels).
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

class PeriodicCnn(nn.Module):
    """
    Convolutional layer for 1d-like lattice with periodic boundary conditions.

    The lattice sites can be indexed by a single
    integer i in the range 0 to (n_sites - 1). The edges connect nearest
    neighbors, i.e. the sites i and j are connected iff |i - j| = 1.

    Correspondinly, in this layer information can propagate in 2 ways:
    - self: information propagates within each site,
    - nn: information propagates to the neighboring sites.
    """
    def __init__(self, in_channels, out_channels):
        super(PeriodicCnn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = nn.Parameter(torch.Tensor(2 * in_channels, out_channels))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weights, a=0)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (*batch_size, n_sites, in_channels).

        Returns:
            output: tensor of shape (*batch_size, n_sites, out_channels).
        """
        # Message passing part
        # nn edges add {-1, 1} to site index.
        nn_contrib = torch.roll(x, shifts=(-1,), dims=(-2,)) + \
                torch.roll(x, shifts=(1,), dims=(-2,))

        # Combine contributions
        all_contribs = torch.cat((x, nn_contrib), dim=-1)
        output = torch.matmul(all_contribs, self.weights)

        return output

class Cnn12SitesT(Cnn12Sites):
    """
    This layer is equivalent to Cnn12Sites, but it uses
    an order of operations which is more efficient
    for in_channels > out_channels.
    """
    def __init__(self, in_channels, out_channels):
        super(Cnn12SitesT, self).__init__(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (*batch_size, 12, in_channels).

        Returns:
            output: tensor of shape (*batch_size, 12, out_channels).
        """
        # Propagate information within each site
        self_contrib = torch.matmul(x, self.self_weights)

        # Propagate information along nn edges
        nn_x = torch.matmul(x, self.nn_weights)
        nn_contrib = torch.roll(nn_x, shifts=(1,), dims=(-2,)) + \
                     torch.roll(nn_x, shifts=(-1,), dims=(-2,)) + \
                     torch.roll(nn_x, shifts=(3,), dims=(-2,)) + \
                     torch.roll(nn_x, shifts=(-3,), dims=(-2,))

        # Propagate information along nnn edges
        nnn_x = torch.matmul(x, self.nnn_weights)
        nnn_contrib = torch.roll(nnn_x, shifts=(2,), dims=(-2,)) + \
                      torch.roll(nnn_x, shifts=(-2,), dims=(-2,)) + \
                      torch.roll(nnn_x, shifts=(4,), dims=(-2,)) + \
                      torch.roll(nnn_x, shifts=(-4,), dims=(-2,))

        # Combine contributions
        output = self_contrib + nn_contrib + nnn_contrib

        return output

def cnn_12_sites(in_channels, out_channels):
    if in_channels <= out_channels:
        return Cnn12Sites(in_channels, out_channels)
    return Cnn12SitesT(in_channels, out_channels)

# TODO: consider the alternatives:
# - use constant planes for lambda inputs.
# - use a deeper, ResNet-like architecture.
class TwelveSitesNN(nn.Module):
    def __init__(self):
        super(TwelveSitesNN, self).__init__()

        # Process lambdas and dlambdas**2.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Process zs
        self.zs_cnn = nn.Sequential(
            Cnn12Sites(2, 32),
            nn.ReLU(),
            Cnn12Sites(32, 64),
            nn.ReLU()
        )

        self.zs_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, lambdas, dlambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 2).
            dlambdas: tensor of shape (*batch_size, 2).
            zs: tensor of shape (*batch_size, 12, 2).

        Returns:
            output1: tensor of shape (*batch_size, 2).
                Represents the output of NN before sigmoid.
                The purpose of the model is to make this output available.
            output2: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        # TODO:9: Fuse:
        # (use torch.jit)
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        batch_size = dlambdas.shape[:-1]
        # Compute all 3 monomials of degree 2 of dlambdas.
        # Add 2 lambdas.
        # dlambdas2.shape == (*batch_size, 5)
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2,
                dlambdas[..., 1, np.newaxis]**2,
                dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
                lambdas
            ], dim=-1)
        assert lambdas_in.shape == (*batch_size, 5)

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas_in)
        assert lambdas_out.shape == (*batch_size, 64)

        # Process zs
        zs_out = self.zs_cnn(zs)
        assert zs_out.shape == (*batch_size, 12, 64), (
            f"{zs_out.shape = } != {(*batch_size, 12, 64)}")
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)
        assert zs_out.shape == (*batch_size, 64), (
            f"{zs_out.shape = } != {(*batch_size, 64)}")

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output1 = self.fc(combined)
        assert output1.shape == (*batch_size, 2)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

class TwelveSitesNN2(nn.Module):
    """
    Same as TwelveSitesNN, but with debiased dlambda**2.
    """
    def __init__(self, n_sites=12, cnn_layer=Cnn12Sites):
        super(TwelveSitesNN2, self).__init__()
        self.n_sites = n_sites
        self.width_scale = (n_sites / 12)**0.5
        scale_width = lambda w: int(w * self.width_scale + 0.5)

        # Process lambdas and dlambdas**2.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(5, scale_width(32)),
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
            nn.Linear(scale_width(64), 2)
        )

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
        # TODO:9: Fuse:
        # (use torch.jit)
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        batch_size = dlambdas.shape[:-1]
        # Compute all 3 monomials of degree 2 of dlambdas.
        # Add 2 lambdas.
        # dlambdas2.shape == (*batch_size, 5)
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2 - 2/3,
                dlambdas[..., 1, np.newaxis]**2 - 2/3,
                dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
                lambdas
            ], dim=-1)

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas_in)

        # Process zs
        zs_out = self.zs_cnn(zs)
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output1 = self.fc(combined)
        assert output1.shape == (*batch_size, 2)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

class TwelveSitesNN3(nn.Module):
    """
    Same as TwelveSitesNN, but with constant planes for
    lambda inputs instead of separate branch.
    """
    def __init__(self):
        super(TwelveSitesNN3, self).__init__()

        # Process zs + constant lambda planes
        # num_input_channels = 7 =
        #   2 (zs)
        #   + 2 (lambdas)
        #   + 3 (dlambdas**2)
        self.zs_cnn = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            Cnn12Sites(32, 64),
            nn.ReLU(),
            Cnn12Sites(64, 64),
            nn.ReLU()
        )

        self.zs_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, lambdas, dlambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 2).
            dlambdas: tensor of shape (*batch_size, 2).
            zs: tensor of shape (*batch_size, 12, 2).

        Returns:
            output1: tensor of shape (*batch_size, 2).
                Represents the output of NN before sigmoid.
                The purpose of the model is to make this output available.
            output2: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        batch_size = dlambdas.shape[:-1]
        # Compute all 3 monomials of degree 2 of dlambdas.
        # Add 2 lambdas.
        # dlambdas2.shape == (*batch_size, 5)
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2 - 2/3,
                dlambdas[..., 1, np.newaxis]**2 - 2/3,
                dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
                lambdas
            ], dim=-1)
        assert lambdas_in.shape == (*batch_size, 5)

        # Combine lambdas_in and zs:
        zs_combined = torch.cat([
                zs,
                lambdas_in.unsqueeze(dim=-2).expand(*batch_size, 12, 5)
            ], dim=-1)
        assert zs_combined.shape == (*batch_size, 12, 7)

        # Process zs
        zs_out = self.zs_cnn(zs_combined)
        assert zs_out.shape == (*batch_size, 12, 64), (
            f"{zs_out.shape = } != {(*batch_size, 12, 64)}")
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)
        assert zs_out.shape == (*batch_size, 64), (
            f"{zs_out.shape = } != {(*batch_size, 64)}")

        # Final classification
        output1 = self.fc(zs_out)
        assert output1.shape == (*batch_size, 2)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

class ResNetLinearBlock(nn.Module):
    def __init__(self, in_out_channels, hidden_channels):
        super(ResNetLinearBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetCnn12SitesBlock(nn.Module):
    def __init__(self, in_out_channels, hidden_channels):
        super(ResNetCnn12SitesBlock, self).__init__()

        self.block = nn.Sequential(
            cnn_12_sites(in_out_channels, hidden_channels),
            nn.ReLU(),
            cnn_12_sites(hidden_channels, in_out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)

class TwelveSitesNN4(nn.Module):
    """
    Same as TwelveSitesNN2, but with ResNet-like skip connections.
    """
    def __init__(self):
        super(TwelveSitesNN4, self).__init__()

        # Process lambdas and dlambdas**2.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            ResNetLinearBlock(32, 64)
        )

        # Process zs
        self.zs_cnn = nn.Sequential(
            Cnn12Sites(2, 32),
            nn.ReLU(),
            ResNetCnn12SitesBlock(32, 64)
        )

        self.zs_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification
        self.fc = nn.Sequential(
            ResNetLinearBlock(64, 128),
            nn.Linear(64, 2)
        )
        self.fc[1].apply(self._custom_weight_init)

    @staticmethod
    def _custom_weight_init(m):
        assert isinstance(m, nn.Linear)
        m.weight.data *= 0.06
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    def forward(self, lambdas, dlambdas, zs):
        """
        Args:
            lambdas: tensor of shape (*batch_size, 2).
            dlambdas: tensor of shape (*batch_size, 2).
            zs: tensor of shape (*batch_size, 12, 2).

        Returns:
            output1: tensor of shape (*batch_size, 2).
                Represents the output of NN before sigmoid.
                The purpose of the model is to make this output available.
            output2: tensor of shape (*batch_size,).
                Represents the final output of NN.
                This output is used for cross-entropy loss.
        """
        # TODO:9: Fuse:
        # (use torch.jit)
        # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
        batch_size = dlambdas.shape[:-1]
        # Compute all 3 monomials of degree 2 of dlambdas.
        # Add 2 lambdas.
        # dlambdas2.shape == (*batch_size, 5)
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2 - 2/3,
                dlambdas[..., 1, np.newaxis]**2 - 2/3,
                dlambdas[..., 0, np.newaxis] * dlambdas[..., 1, np.newaxis],
                lambdas
            ], dim=-1)
        assert lambdas_in.shape == (*batch_size, 5)

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas_in)
        assert lambdas_out.shape == (*batch_size, 32)

        # Process zs
        zs_out = self.zs_cnn(zs)
        assert zs_out.shape == (*batch_size, 12, 32), (
            f"{zs_out.shape = } != {(*batch_size, 12, 32)}")
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)
        assert zs_out.shape == (*batch_size, 32), (
            f"{zs_out.shape = } != {(*batch_size, 32)}")

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output1 = self.fc(combined)
        assert output1.shape == (*batch_size, 2)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2

@torch.no_grad()
def eval_chifc_estimate(dataset, model, device="cuda:0"):
    """
    Args:
        dataset: dict with the following keys:
        - lambdas: numpy array of shape (num_samples, 2) on [-1, 1] scale.
        - zs: numpy array of shape (num_samples, 2, 12).

    Returns:
        lambdas: numpy array of shape (num_samples, 2).
            (same as dataset['lambdas'] but scaled to [0, 1] instead of [-1, 1]).
        output1: numpy array of shape (num_samples, 2). Output of the model
        used to estimate fidelity susceptibility.
    """
    lambdas = dataset["lambdas"]
    zs = dataset["zs"]

    model.eval()
    model.to(device)
    lambdas = torch.from_numpy(lambdas).to(device)
    zs = torch.from_numpy(zs).transpose(1, 2).to(device)
    dlambdas = torch.zeros(lambdas.shape, device=device)
    output1, _ = model(lambdas, dlambdas, zs)
    return ((lambdas.cpu().numpy() + 1) / 2, output1.cpu().numpy())

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DDihedral(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device=None):
        super(Conv1DDihedral, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if device is not None:
            self.device = device
            device_arg = {'device': device}
        else:
            device_arg = {}

        # Initializing the symmetric weight
        half_kernel_size = (self.kernel_size + 1) // 2  # Compute the half size of kernel
        self.weight = nn.Parameter(torch.empty(
            size=(out_channels, in_channels, half_kernel_size), **device_arg))
        nn.init.kaiming_uniform_(self.weight, a=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Handle the PBC padding
        padding_size = self.kernel_size - 1
        x_padded = torch.cat([x, x[..., :padding_size]], dim=-1)

        # Construct the symmetric kernel
        if self.kernel_size % 2 == 0:  # Even kernel size
            symmetric_weight = torch.cat(
                [self.weight, self.weight.flip(dims=(-1,))], dim=-1)
        else:  # Odd kernel size
            symmetric_weight = torch.cat(
                [self.weight, self.weight[..., :-1].flip(dims=(-1,))], dim=-1)

        return F.conv1d(
            x_padded, symmetric_weight,
            bias=None, stride=1, padding=0, dilation=1, groups=1)

# Example usage:
class Model1DDihedral(nn.Module):
    """
    Model for linear systems with Dihedral symmetry
    """
    def __init__(self, n_sites=20):
        super().__init__()
        self.n_sites = n_sites

        # Process lambdas and dlambdas**2.
        self.lambdas_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        conv = classifim.layers.Conv1DDihedral
        # Process zs
        self.zs_cnn = nn.Sequential(
            conv(1, 32, kernel_size=5),
            nn.ReLU(),
            conv(32, 64, kernel_size=5),
            nn.ReLU()
        )

        self.zs_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

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
        assert lambdas.shape[-1] == 1
        lambdas_in = torch.cat([
                dlambdas[..., 0, np.newaxis]**2 - 2/3,
                lambdas
            ], dim=-1)

        # Process lambdas
        lambdas_out = self.lambdas_fc(lambdas_in)

        # Process zs
        assert zs.shape == (*batch_size, 1, self.n_sites)
        zs_out = self.zs_cnn(zs)
        assert zs_out.shape == (*batch_size, 64, self.n_sites)
        zs_out = self.zs_pool(zs_out.transpose(-1, -2)).squeeze(dim=-1)
        assert zs_out.shape == (*batch_size, 64)

        # Concatenate
        combined = torch.cat([lambdas_out, zs_out], dim=1)

        # Final classification
        output1 = self.fc(combined)
        assert output1.shape == (*batch_size, 2)
        output2 = torch.sum(output1 * dlambdas, dim=-1)
        assert output2.shape == (*batch_size,)

        return output1, output2
