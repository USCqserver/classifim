import classifim.utils
import classifim.input
import numba
import numpy as np
import os
import sklearn.decomposition
from classifim_gen.bits import popcount64

@numba.njit(numba.uint64[:,:](numba.int8[:,:], numba.int8))
def compact_zs_jit(zs, offset):
    num_samples, nbits = zs.shape
    n_uint64 = (nbits + 63) // 64
    num_last_block_bits = 1 + ((nbits + 63) % 64)
    res = np.empty((num_samples, n_uint64), dtype=np.uint64)

    for i in range(num_samples):
        bit_offset = numba.uint32(0)
        j = numba.int32(0)
        for block in range(n_uint64 - 1):
            v = numba.uint64(0)
            j0 = numba.uint64(0)
            while j0 < 64:
                v += numba.uint64(zs[i][j] - offset) << j0
                j += numba.int32(1)
                j0 += numba.uint64(1)
            res[i][block] = v
        v = numba.uint64(0)
        j0 = numba.uint64(0)
        while j0 < num_last_block_bits:
            v += numba.uint64(zs[i][j] - offset) << j0
            j += numba.int32(1)
            j0 += numba.uint64(1)
        res[i][n_uint64 - 1] = v
    return res

@numba.njit(
    numba.float32(
        numba.uint64[:, :], numba.uint64[:, :], numba.uint64,
        numba.float32, numba.float32),
    nogil=True)
def _get_k_shadow_jit(za, zb, nbits, gamma, tau):
    gamma1 = gamma / nbits
    total = 0.0
    n_blocks = (nbits + numba.uint64(63)) // numba.uint64(64)

    for i in range(len(za)):
        for j in range(len(zb)):
            sum_tr1 = numba.uint64(nbits)
            for k in range(n_blocks):
                sum_tr1 -= popcount64(za[i][k] ^ zb[j][k])
            total += np.exp(gamma1 * numba.float32(sum_tr1))

    count = za.shape[0] * zb.shape[0]
    return np.exp(tau * total / count)

@numba.njit(
    numba.float32(
        numba.uint64[:], numba.uint64[:], numba.uint64,
        numba.float32, numba.float32),
    nogil=True)
def _get_k_shadow_jit64(za, zb, nbits, gamma, tau):
    gamma1 = gamma / nbits
    total = 0.0
    for i in range(len(za)):
        for j in range(len(zb)):
            sum_tr1 = numba.uint64(nbits) - popcount64(za[i] ^ zb[j])
            total += np.exp(gamma1 * numba.float32(sum_tr1))
    count = len(za) * len(zb)
    return np.exp(tau * total / count)

def get_k_shadow_jit(za, zb, nbits=64, gamma=1.0, tau=1.0):
    assert nbits > 0
    n_blocks = (nbits + 63) // 64
    assert za.shape[1] == n_blocks
    assert zb.shape[1] == n_blocks
    return _get_k_shadow_jit(za, zb, nbits, gamma, tau)

def get_k_shadow_jit64(za, zb, nbits=64, gamma=1.0, tau=1.0):
    """
    Same as get_k_shadow_jit but without extra dimension for blocks.
    """
    assert za.ndim == 1
    assert zb.ndim == 1
    assert nbits <= 64
    return _get_k_shadow_jit64(za, zb, nbits, gamma, tau)

def get_k_shadow_matrix(grouped_zs, nbits=64, single_block=False):
    """
    Args:
        grouped_zs: list of 1D np.ndarrays of dtype np.int32.
            (arrays may have different lengths)

    Returns:
        Matrix of np.float64 values
    """
    if single_block:
        assert nbits <= 64
        for zs in grouped_zs:
            assert zs.ndim == 1
            assert zs.dtype == np.uint64
        shadow_func = _get_k_shadow_jit64
    else:
        shadow_func = get_k_shadow_jit
    num_groups = len(grouped_zs)
    res = np.empty((num_groups, num_groups), dtype=np.float32)
    for i in range(num_groups):
        zi = grouped_zs[i]
        res[i, i] = shadow_func(zi, zi, nbits, 1.0, 1.0)
        for j in range(i):
            zj = grouped_zs[j]
            cur_res = shadow_func(zi, zj, nbits, 1.0, 1.0)
            res[i, j] = cur_res
            res[j, i] = cur_res
    return res

class ComputeKShadowMatrixPipeline:
    def __init__(
            self, data_dir, suffix, load_dataset=True, grid_shape=(64, 64),
            nbits=None, zs_offset=0, test_size=0.1, scalar_keys=None):
        self.data_dir = data_dir
        self.suffix = suffix
        self.grid_shape = grid_shape
        self.nbits = nbits
        self.test_size = test_size
        self.zs_offset = zs_offset
        if scalar_keys is None:
            self.scalar_keys = ["seed"]
        else:
            self.scalar_keys = scalar_keys
        if load_dataset:
            self.load_dataset()

    def load_dataset(self):
        dataset_filename = os.path.join(
            self.data_dir, "classifim_datasets", f"dataset_{self.suffix}.npz")
        with np.load(dataset_filename) as f:
            npz_dataset = dict(f)

        prng = classifim.utils.DeterministicPrng(self.suffix)
        dataset_train, dataset_test = classifim.input.split_train_test(
            npz_dataset, test_size=0.1,
            seed=prng.get_seed("test"),
            scalar_keys=self.scalar_keys)
        self.dataset_train = dataset_train
        zs = dataset_train["zs"]
        if zs.ndim == 2:
            nbits = zs.shape[1]
            assert np.all(self.zs_offset <= zs)
            assert np.all(zs <= self.zs_offset + 1)
            if self.nbits is None:
                self.nbits = nbits
            else:
                assert self.nbits == nbits
            self.zs_is_packed = False
            zs_compact = compact_zs_jit(zs_compact)
            if self.nbits <= 64:
                assert zs_compact.shape[1] == 1
                zs_compact = zs_compact[:, 0]
            dataset_train["zs"] = zs_compact
        elif zs.ndim == 1:
            assert self.nbits <= 64
            self.zs_is_packed = True
            dataset_train["zs"] = zs.astype(np.uint64)

    def group_zs(self):
        lambdas = self.dataset_train["lambdas"]
        _, groups = np.unique(lambdas, axis=0, return_inverse=True)
        num_lambdas = np.prod(self.grid_shape)
        num_groups = np.max(groups) + 1
        assert num_groups == num_lambdas, f"{num_groups} != {num_lambdas}"
        zs_compact = self.dataset_train["zs"]

        grouped_zs = [[] for _ in range(num_lambdas)]
        for g, z in zip(groups, zs_compact):
            grouped_zs[g].append(z)
        self.grouped_zs = [np.array(zs, dtype=np.uint64) for zs in grouped_zs]

    def compute_k_shadow(self):
        self.k_shadow_matrix = get_k_shadow_matrix(
            self.grouped_zs, nbits=self.nbits, single_block=self.nbits <= 64)

    def compute_pca(self, top_components=10):
        kpca = sklearn.decomposition.KernelPCA(
            n_components=top_components, kernel="precomputed")
        pca = kpca.fit_transform(self.k_shadow_matrix)
        assert pca.shape == (np.prod(self.grid_shape), top_components)
        self.pca = pca.reshape(*self.grid_shape, top_components).swapaxes(0, 1)

    def save_k_shadow(self):
        filename = os.path.join(
            self.data_dir, "models", f"pcaz_k_shadow_{self.suffix}.npz")
        assert os.path.isdir(os.path.dirname(filename))
        res = dict(
            suffix=self.suffix,
            k_shadow_matrix=self.k_shadow_matrix,
            nbits=self.nbits,
            **{key: self.dataset_train[key] for key in self.scalar_keys})
        if (pca := getattr(self, "pca", None)) is not None:
            res["pca"] = pca
        np.savez_compressed(filename, **res)

    def load_k_shadow(self):
        filename = os.path.join(
            self.data_dir, "models", f"pcaz_k_shadow_{self.suffix}.npz")
        assert os.path.isdir(os.path.dirname(filename))
        with np.load(filename) as f:
            self.k_shadow_matrix = f["k_shadow_matrix"]
            self.nbits = f["nbits"]
            if "pca" in f:
                self.pca = f["pca"]
            if not hasattr(self, "dataset_train"):
                self.dataset_train = {}
            for key in self.scalar_keys:
                self.dataset_train[key] = f[key]
