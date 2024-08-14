import classifim.pipeline
import classifim.w
import classifim.ising2d as ising2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WDataLoader(classifim.w.DataLoader):
    """
    The implementation is almost identical to ising2d.DataLoaderMixin,
    but with the additional argument idx in `_init_zs`.
    """
    def _init_zs(self, data, device, idx):
        zs = data["samples"][idx]
        n_bits = data["width"]
        if isinstance(n_bits, np.ndarray):
            n_bits = n_bits.item()
        assert isinstance(n_bits, int)
        # 32 is not supported because torch does not support uint32, only int32:
        assert 1 <= n_bits <= 31
        self.n_bits = n_bits
        assert len(zs.shape) == 2
        self.zs = torch.from_numpy(zs.astype(np.int32)).to(device=device)

    def _retrieve_zs(self, idx):
        zs = classifim.utils.unpackbits32_torch(
                self.zs[idx], num_bits=self.n_bits, device=self.zs.device
            ) * 2.0 - 1.0
        return (zs, )

class ModelNaiveCNN(nn.Module):
    """
    Similar to ising2d.ModelNaiveCNN, but for van Nieuwenburg's W:
    - Separate model for each value of lambda_fixed and lambda_critical.
    - Models for different values of lambda_critical are combined into a single
        model.
    - Label is for binary classification of presumably two phases.
    """
    def __init__(self, height=None, width=None, layer_bs=1):
        """
        layer_bs (layer batch size) is the number of parallelly trained models
        (for different lambda_critical).
        """
        super().__init__()
        # We do not actually use width and height: the same CNN layer
        # can be applied to PBC lattice of any size.
        self.height = height
        self.width = width
        self.layer_bs = layer_bs
        self.zs_cnn = nn.Sequential(
            nn.Conv2d(layer_bs * 1, layer_bs * 18,
                kernel_size=(3, 3), padding=0, groups=layer_bs),
            nn.ReLU())
        self.fc = nn.Sequential(
            classifim.w.BatchLinear(layer_bs, 18, 64),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, 64, 32),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, 32, 32),
            nn.ReLU(),
            classifim.w.BatchLinear(layer_bs, 32, 1))

    def forward(self, zs):
        batch_size, height, width = zs.shape
        assert height == self.height, f"{zs.shape=}, {self.height=}"
        assert width == self.width
        zs = F.pad(zs.unsqueeze(1), (0, 2, 0, 2), mode='circular')
        zs = zs.repeat(1, self.layer_bs, 1, 1)
        assert zs.shape == (batch_size, self.layer_bs, height + 2, width + 2)
        zs_out = self.zs_cnn(zs)
        zs_out = F.adaptive_avg_pool2d(zs_out, 1).squeeze(dim=(-1, -2))
        assert zs_out.shape == (batch_size, self.layer_bs * 18)
        zs_out = zs_out.reshape(batch_size, self.layer_bs, 18)
        output = self.fc(zs_out).squeeze(dim=-1)
        assert output.shape == (batch_size, self.layer_bs)
        return output

class WPipeline(classifim.w.Pipeline, ising2d.Pipeline):
    def _get_data_loader(self, dataset, is_train, **kwargs):
        return classifim.w.Pipeline._get_data_loader(
            self, dataset, is_train, cls=WDataLoader, **kwargs)

    def construct_model(self, device=None):
        model_init_kwargs = {
            'height': self.dataset_train["height"],
            'width': self.dataset_train["width"],
            'layer_bs': self.preprocessor.layer_bs
        }
        model_init_kwargs.update(self.config.get("model_init_kwargs", {}))
        return classifim.w.Pipeline.construct_model(
            self, device=device, cls=ModelNaiveCNN,
            model_init_kwargs=model_init_kwargs)
