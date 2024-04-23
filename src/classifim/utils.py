import hashlib
import numpy as np
import os
import os.path
import torch

def find_data_dir(config_name="data_dir", code_dir=None):
    """
    Find the directory with datasets.

    Look for it in the following order:
    * Look for CONFIG_TOML_PATH and read the data_dir field.
    * Try f"{code_dir}/../data".
    * FAIL (raise RuntimeError).
    """
    try:
        import toml
        CONFIG_TOML_PATH = os.path.expanduser("~/.config/classifim/config.toml")
        if os.path.isfile(CONFIG_TOML_PATH):
            config = toml.load(CONFIG_TOML_PATH)
            if "main" in config:
                config = config["main"]
            data_dir = config[config_name]
            data_dir = os.path.realpath(os.path.expanduser(data_dir))
            if os.path.isdir(data_dir):
                return data_dir
            else:
                print(f"WARNING: data_dir='{data_dir}' is specified in "
                    f"'{CONFIG_TOML_PATH}' but it does not exist.")
    except:
        # Loading the data_dir path from config.toml is optional.
        # Ignore any errors silently.
        pass
    if code_dir is None:
        code_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.realpath(os.path.join(code_dir, os.path.join("..", "data")))
    if os.path.isdir(data_dir):
        return data_dir
    raise RuntimeError(f"Could not find data_dir (tried '{data_dir}').")

def maybe_create_subdir(parent, subdir):
    """
    Create a subdirectory if it does not exist.

    Return os.path.join(parent, subdir).
    """
    path = os.path.join(parent, subdir)
    assert parent == os.path.dirname(path), (
        f"subdir='{subdir}' should be a direct subdir of parent='{parent}'.")
    if not os.path.exists(path):
        os.mkdir(path)
    assert os.path.isdir(path)
    return path

class DeterministicPrng:
    """
    A simple class to generate deterministic but different seeds for specific
    operations.
    """
    def __init__(self, seed):
        m = hashlib.sha256()
        m.update(str(seed).encode('utf-8'))
        self.prefix = m.digest()

    def get_seed(self, key):
        m = hashlib.sha256()
        m.update(self.prefix)
        m.update(str(key).encode('utf-8'))
        return np.random.SeedSequence(int(m.hexdigest(), 16), pool_size=8)

    def get_int_seed(self, key):
        return self.get_seed(key).generate_state(1)[0]

    def get_int64_seed(self, key):
        """
        Return a random positive int64 seed.
        """
        res = 0
        seed_seq = self.get_seed(key)
        while res == 0:
            res = seed_seq.generate_state(1, dtype=np.uint64)[0]
            # res is numpy.uint64. Convert to int:
            res = int(res) % 2**63
        return res

def unpackbits(zs, num_bits):
    """
    Converts an array of integers into an array of bits.

    Least significant bits are listed first.

    Args:
        zs: 1D array of integers.
        num_bits: number of bits to extract from each integer.
    """
    # Implementation based on
    # https://stackoverflow.com/a/22227898/301644
    return ((zs[..., np.newaxis] & (1 << np.arange(num_bits)))) > 0

def unpackbits32_torch(zs, num_bits, device):
    """
    Same as unpackbits but for torch tensors.

    Args:
        zs: 1D array of 32-bit integers.
        num_bits: number of bits to extract from each integer.
    """
    shift = torch.arange(num_bits, device=device)
    return (zs[..., None] & (1 << shift)) > 0

def packbits(zs):
    """
    Converts an array of bits into an array of integers.

    Least significant bits are listed first.

    Args:
        zs: 2D array of bits.
    """
    return np.sum(zs * (1 << np.arange(zs.shape[-1])), axis=-1)

def hash_base36(v, length=13):
    """
    Stable alphanumeric hash of a value.
    """
    hash_obj = hashlib.sha256(repr(v).encode())
    # byteorder='big' makes sure this is equivalent to
    # hash_value = int(hash_obj.hexdigest(), 16)
    hash_value = int.from_bytes(hash_obj.digest(), byteorder='big')
    return np.base_repr(hash_value, 36, length)[-length:].lower()

def load_tensor(filename, tensor_name=None):
    """
    Loads a tensor from a file.

    Uses either torch.load or np.load depending on the file extension.

    Args:
        filename (str): The filename to load from.
        tensor_name (str): The name of the tensor to load. Used for .npz files.

    Returns:
        The loaded tensor (numpy array).
    """
    if filename.endswith(".pt"):
        return torch.load(filename).numpy()
    elif filename.endswith(".npy"):
        return np.load(filename)
    elif filename.endswith(".npz"):
        return np.load(filename)[tensor_name]
    else:
        raise ValueError(f"Unknown file extension for '{filename}'.")

