import hashlib
import numpy as np
import pyarrow as pa
import sys

def fil24_unpack_zs(zs):
    """
    Converts an array of 24-bit integers into an array of Z values in {-1, 1}.

    Input format: 0bTT...TTBB...BB
    Bit Bj of zs is (zs & (1 << j)) > 0.
    Bit Tj of zs is (zs & (1 << (self.nsites + j))) > 0.

    Output format: [[Z_{B0}, Z_{B1}, ..., Z_{B11}], [Z_{T0}, ..., Z_{T11}]]
    Z_{Bj} of zs_out is zs_out[..., 0, j].
    Z_{Tj} of zs_out is zs_out[..., 1, j].
    """
    # TODO:9: remove in favor of classifim.twelve_sites_pipeline.Pipeline.unpack_zs
    zs_bin = classifim.io.unpackbits(zs, 24)
    return (1 - 2 * zs_bin).reshape(*zs.shape, 2, 12).astype(np.float32)

def split_train_test(dataset, test_size=0.1, seed=0, scalar_keys=()):
    """
    Splits a dataset into a training set and a test set.

    Args:
        dataset: a dictionary, each value of which is an np.ndarray
          of shape (N, ...) where N is the number of samples.
          Dictionary-like objects like numpy.lib.npyio.NpzFile
          are also supported.
        test_size: fraction of the dataset to use for the test set.
        seed: random seed.
        scalar_keys: keys of dataset whose values should be copied
            as is instead of splitting.

    Returns:
        dataset_train, dataset_test: dictionaries with the same keys as dataset.
    """
    assert 0 < test_size < 1
    if scalar_keys is None:
        scalar_keys = ()
    dataset_train = {}
    dataset_test = {}
    # Delay initialization until we hit non-scalar key:
    num_samples = None
    for key, value in dataset.items():
        if key in scalar_keys:
            dataset_train[key] = value
            dataset_test[key] = value
            continue
        if num_samples is None:
            rng = np.random.default_rng(seed)
            num_samples = value.shape[0]
            test_ii0 = rng.choice(
                    num_samples, size=int(test_size * num_samples), replace=False)
            test_ii = np.zeros(num_samples, dtype=bool)
            test_ii[test_ii0] = True
        assert len(value.shape) > 0 and value.shape[0] == num_samples, (
            f"dataset['{key}'].shape={value.shape}, shape[0] != {num_samples}.\n"
            + f"dataset.keys()={list(dataset.keys())}\n"
            + "Did you forget to specify it in scalar_keys?")
        dataset_train[key] = value[~test_ii]
        dataset_test[key] = value[test_ii]
    return dataset_train, dataset_test

def scale_lambdas(dataset, inplace=False, dtype=None):
    """
    Scales lambdas from [0, 1) to [-1, 1).

    Args:
        dataset: a dictionary with keys (lambdas, zs).
        inplace: whether to modify dataset in place.
        dtype: dtype of the resulting lambdas.

    Returns:
        dataset with the same keys as the original dataset.
    """
    if not inplace:
        dataset = dataset.copy()
    lambdas = 2 * dataset["lambdas"] - 1
    if dtype is not None:
        lambdas = lambdas.astype(dtype)
    dataset["lambdas"] = lambdas
    return dataset

def get_classifim_train_dataset(dataset, num_passes=1, seed=0):
    """
    Converts a dataset with keys (lambdas, zs) using BitstringChiFc method.

    The resulting dataset has keys
    - lambda0: midpoint between 2 points in the parameter space,
        i.e. (lambda_p + lambda_m) / 2.
    - dlambda: lambda_p - lambda_m = (lambda_p - lambda_0) * 2.
    - zs: bitstring sampled either from lambda_p or lambda_m (with equal
        prior probability 1/2 each).
    - label: 0 if zs was sampled from lambda_m, 1 if zs was sampled from lambda_p.

    Note that lambda_p = lambda0 + dlambda / 2 and
    lambda_m = lambda0 - dlambda / 2.

    The size of the resulting dataset is num_input_samples * 2 * num_passes, where
    num_input_samples is the number of samples in the original dataset.

    Args:
        dataset: a dictionary with keys (lambdas, zs).
        num_passes: number of passes over the dataset to make. The resulting
            dataset will contain num_input_samples * 2 * num_passes samples.
        seed: random seed.

    Returns:
        dictionary with keys (lambda0, dlambda, zs, label).
    """
    rng = np.random.default_rng(seed)
    num_input_samples = dataset["lambdas"].shape[0]
    ii0 = np.tile(np.arange(num_input_samples), num_passes)
    ii1 = rng.choice(ii0, size=ii0.size, replace=False)

    # Ensure that for every j lambdas[ii0[j]] != lambdas[ii1[j]].
    ii = np.arange(ii0.size)
    lambdas = dataset["lambdas"]
    extra_lambdas_axes = tuple(range(1, lambdas.ndim))
    max_iterations = 20
    for iteration_i in range(max_iterations + 1):
        i_same = ii[np.all(
            lambdas[ii0] == lambdas[ii1], axis=extra_lambdas_axes)]
        if i_same.size == 0:
            break
        elif iteration_i == max_iterations:
            raise RuntimeError(
                "Failed to fix lambdas[ii0] == lambdas[ii1] problem in "
                f"{max_iterations} iterations.")
        elif np.unique(lambdas[ii0[i_same]]).size > 1:
            ii1[i_same] = ii1[rng.choice(
                i_same.size, size=i_same.size, replace=False)]
            continue
        # np.unique(lambdas[ii0[i_same]]).size == 1
        # In this case we can't fix the problem by shuffling ii1[i_same].
        common_lambda = lambdas[ii0[i_same[0]]]
        i_new_size = min(ii0.size, 8 + 2 * i_same.size)
        i_new = rng.choice(ii0.size, size=i_new_size, replace=False)
        i_new = i_new[~np.all(
            lambdas[ii0[i_new]] == common_lambda,
            axis=extra_lambdas_axes)]
        i_new = i_new[~np.all(
            lambdas[ii1[i_new]] == common_lambda,
            axis=extra_lambdas_axes)]
        i_new = i_new[:i_same.size]
        ii1[np.concatenate([i_same, i_new])] = (
            ii1[np.concatenate([i_new, i_same])])

    lambda_m = lambdas[ii0]
    lambda_p = lambdas[ii1]
    lambda0 = (lambda_p + lambda_m) / 2
    dlambda = lambda_p - lambda_m
    zs_m = dataset["zs"][ii0]
    zs_p = dataset["zs"][ii1]
    res = {
        "lambda0": np.tile(lambda0, (2, *[1] * (lambda0.ndim - 1))),
        "dlambda": np.tile(dlambda, (2, *[1] * (dlambda.ndim - 1))),
        "zs": np.concatenate([zs_m, zs_p]),
        "label": np.concatenate([
            np.zeros(ii0.size, dtype=np.float32),
            np.ones(ii0.size, dtype=np.float32)]),
        "ii0": ii0,
        "ii1": ii1}
    num_output_samples = num_input_samples * 2 * num_passes
    # i_shuffle = rng.choice(
    #     num_output_samples, size=num_output_samples, replace=False)
    for key, value in res.items():
        if key in ("ii0", "ii1"):
            continue
        assert value.shape[0] == num_output_samples, (
            f"{key} {value.shape} {num_output_samples}")
        # res[key] = value[i_shuffle]
    return res

def flatten_dict_for_npz(
        d, key_prefix="", sep=".", permissive=False, target=None):
    """
    Flattens a nested dictionary to prepare for saving to npz file.

    Args:
        d: dictionary to flatten.
        key_prefix: prefix to prepend to all keys. Should include trailing
            separator if desired.
        sep: separator to use between keys.
        permissive: if True, ignore object imperfect for npz format.
        target: target dictionary to write to. If None, a new dictionary is
            created.
    """
    res = target if target is not None else {}
    if not permissive:
        assert isinstance(key_prefix, str), (
            f"key_prefix must be a string, got {type(key_prefix)}")
        assert isinstance(sep, str), (
            f"sep must be a string, got {type(sep)}")
    for k, v in d.items():
        new_key = key_prefix + k
        if isinstance(v, dict):
            flatten_dict_for_npz(v, new_key + sep, sep, target=res)
        else:
            if isinstance(v, set):
                v = list(v)
            assert new_key not in res, (
                f"Key {new_key} already exists in res.")
            try:
                v_np = np.array(v)
            except Exception:
                if not permissive:
                    raise
                v_np = None
            if not permissive:
                assert v_np.dtype != object, (
                    f"Key {new_key} has dtype object, which is not supported "
                    + "by npz format (without pickle).")
            if v_np is not None and v_np.dtype != object:
                res[new_key] = v_np
            else:
                res[new_key] = v
    return res

def unflatten_dict_from_npz(d, sep="."):
    """
    Unflattens a dictionary that was saved to an npz file.

    Args:
        d: flattened dictionary.
        sep: separator used between keys.

    Returns:
        unflattened dictionary.
    """
    res = {}
    for k, v in d.items():
        keys = k.split(sep)
        cur = res
        for key in keys[:-1]:
            if key not in cur:
                cur[key] = {}
            cur = cur[key]
        key = keys[-1]
        assert key not in cur, f"Duplicate key '{k}'(?)"
        cur[key] = v
    return res

def samples1d_uint8_to_bytes(samples):
    """
    Convert (n_samples, width) array of individual bytes to a 1D array of bytes.

    Args:
        samples: numpy array of shape (n_samples, width) of dtype with itemsize 1.
    """
    assert samples.itemsize == 1
    assert samples.dtype in [np.int8, np.uint8, np.dtype((np.bytes_, 1))]
    num_samples, width = samples.shape
    # Convert to np.uint8 to ensure consistent behavior.
    samples = samples.astype(np.uint8)
    samples = np.frombuffer(
        samples.tobytes(),
        dtype=np.dtype((np.bytes_, width)))
    assert samples.shape == (num_samples, )
    return samples

def samples1d_bytes_to_uint8(samples, width=None):
    """
    Convert a 1D array of bytes to an array of individual bytes.

    Args:
        samples: numpy array of shape (n_samples, ) of dtype
            (np.bytes_, width).
        width: width of the grid.

    Returns:
        numpy array of shape (n_samples, width) of dtype np.uint8.
    """
    num_samples, = samples.shape
    num_bytes = samples.itemsize
    if width is not None:
        assert num_bytes == width
    samples = np.frombuffer(samples.tobytes(), dtype=np.uint8)
    samples = samples.reshape((num_samples, num_bytes))
    return samples

def samples2d_to_bytes(samples, width):
    """
    Convert (n_samples, height, width) array of bits stored as
    2D array `samples` of shape `(num_samples, height)` and dtype `np.uint64`
    to a 2D np.ndarray of bytes.

    Each sample is a bit array of shape `(height, width)`.

    Args:
        samples: numpy array of shape (n_samples, height) of dtype np.uint64.
        width: width of the grid.

    Returns:
        np.ndarray `output_bytes` of shape `(num_samples, )`
        and dtype `np.dtype(np.bytes_, ceil(width * height / 8))`.
        bit #l of `output_bytes[i]`, i.e.
        `output_bytes[i][l_high] >> l_low & 1` (where `l = l_high * 8 + l_low`),
        is set to `(samples[i, j] >> k) & 1` where
        `l = j * width + k` when `l < width * height`, and to 0 otherwise.
    """
    assert 1 <= width <= 64
    n_samples, height = samples.shape
    assert 1 <= height
    n_bits = width * height
    n_words = (n_bits + 63) // 64
    out_buf = np.zeros((n_samples, n_words), dtype=np.uint64)
    bit_i = np.uint64(0)
    word_i = 0
    np_width = np.uint64(width)
    np_64 = np.uint64(64)
    np_0 = np.uint64(0)
    for j in range(height):
        in_bits = samples[:, j]
        out_buf[:, word_i] |= in_bits << bit_i
        bit_i += np_width
        if bit_i >= np_64:
            bit_i -= np_64
            word_i += 1
            if bit_i > np_0:
                out_buf[:, word_i] |= in_bits >> (np_width - bit_i)
    n_bytes = (n_bits + 7) // 8
    output_bytes = out_buf.view(np.uint8).reshape((n_samples, n_words * 8))[
            :, :n_bytes]
    return samples1d_uint8_to_bytes(output_bytes)

def bytes_to_pa(in_bytes):
    """
    Convert a numpy array of bytes to a PyArrow array.

    Specifically, we assume that all entries in `in_bytes` have the same length
    `in_bytes.itemsize` produce the output array of type `pa.binary()`
    (not `pa.binary(in_bytes.itemsize)` because the latter is not supported
    by HuggingFace datasets).

    References:
    * https://github.com/apache/arrow/issues/41388
    * https://stackoverflow.com/questions/78359858

    Args:
        in_bytes: numpy array of bytes.

    Returns:
        PyArrow array.
    """
    fixed_len_array = pa.array(in_bytes, type=pa.binary(in_bytes.itemsize))
    return fixed_len_array.cast(pa.binary())

def bytes_to_samples2d(in_bytes, height, width):
    """
    Convert a 2D np.ndarray of bytes back to an array of bits stored as
    2D array of shape `(n_samples, height)` and dtype `np.uint64`.
    This function implements the inverse of `samples2d_to_bytes`.

    Args:
        in_bytes: either
            * np.ndarray of shape `(n_samples, ceil(width * height / 8))`
                and dtype `np.uint8`, or
            * np.ndarray of shape `(n_samples, )`
                and dtype `np.dtype(np.bytes_, ceil(width * height / 8))`.
        width: width of the grid (number of bits per row in the output).
        height: number of rows in each sample.

    Returns:
        samples: numpy array of shape (n_samples, height) of dtype `np.uint64`.
    """
    if in_bytes.dtype.kind == "S":
        n_samples, = in_bytes.shape
        n_bytes = in_bytes.itemsize
        in_bytes = np.frombuffer(in_bytes.tobytes(), dtype=np.uint8).reshape(
            (n_samples, n_bytes))
    else:
        assert in_bytes.dtype == np.uint8
        n_samples, n_bytes = in_bytes.shape
    n_bits = width * height
    assert n_bytes * 8 >= n_bits
    n_words = (n_bits + 63) // 64

    if n_bytes < n_words * 8:
        in_bytes = np.pad(in_bytes, ((0, 0), (0, n_words * 8 - n_bytes)))
    in_words = in_bytes.view(np.uint64)
    assert in_words.shape == (n_samples, n_words)

    samples = np.empty((n_samples, height), dtype=np.uint64)
    bit_i = np.uint64(0)
    word_i = 0
    np_width = np.uint64(width)
    np_64 = np.uint64(64)
    np_0 = np.uint64(0)
    width_mask = np.uint64((1 << width) - 1)
    for j in range(height):
        samples[:, j] = width_mask & (in_words[:, word_i] >> bit_i)
        bit_i += np_width
        if bit_i >= np_64:
            bit_i -= np_64
            word_i += 1
            if bit_i > np_0:
                samples[:, j] |= width_mask & (
                        in_words[:, word_i] << (np_width - bit_i))
    return samples

def pa_to_np(pa_array):
    """
    Convert a PyArrow array to a numpy array.

    Most of the time, this is equivalent to pa_array.to_numpy(),
    but for binary arrays a conversion to np.dtype((np.bytes_, n_bytes))
    is attempted. This assumes that all binary array fields have fixed
    length.

    Args:
        pa_array: PyArrow array.

    Returns:
        numpy array.
    """
    if pa_array.type == pa.binary() and len(pa_array) > 0:
        n_bytes = len(pa_array[0].as_py())
        try:
            return pa_array.to_numpy().astype(np.dtype((np.bytes_, n_bytes)))
        except Exception:
            pass
    return pa_array.to_numpy()

def prepare_for_json(obj):
    """
    Prepare an object for saving to a JSON file.

    This loses some information (e.g. {'a'} becomes ['a']).

    Args:
        obj: object to prepare.

    Returns:
        Data structure that can be saved to a JSON file.
    """
    if isinstance(obj, dict):
        return {str(k): prepare_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [prepare_for_json(v) for v in obj]
    return obj
