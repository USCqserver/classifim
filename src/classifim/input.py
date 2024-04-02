import hashlib
import numpy as np

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
    zs_bin = classifim.input.unpackbits(zs, 24)
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
            num_samples = next(iter(dataset.values())).shape[0]
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
