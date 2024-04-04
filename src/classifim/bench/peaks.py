"""
Computing peak accuracy for comparison with prior work.
"""

import classifim_gen.plot_tools
import functools
import numpy as np
import scipy.signal
import sklearn.cluster

def rename_keys(d, key_map):
    """
    Renames the keys of the dictionary `d` according to `key_map`.

    Args:
        d: The original dictionary.
        key_map: A dictionary mapping old keys to new keys.

    Returns: A new dictionary with keys renamed.
    """
    return {key_map.get(key, key): value for key, value in d.items()}

def extract_gs_meshgrid(fim_df, sweep_lambda_index):
    """
    Extract gs FIM in meshgrid format along a single sweep direction.

    Note: This function is different from meshgrid_transform_2D_fim
    in two main ways:
    - Only a single component of the FIM is extracted.
    - The output corresponds to the same lambda_fixed values as in the
        original grid. meshgrid_transform_2D_fim's corresponds to midpoints.
    """
    df = fim_df[fim_df["dir"] == str(sweep_lambda_index)]
    sweep_lambda_name = "lambda" + str(sweep_lambda_index)
    fixed_lambda_name = "lambda" + str(1 - sweep_lambda_index)
    mg = classifim_gen.plot_tools.df_to_meshgrid(
            df, sweep_lambda_name, fixed_lambda_name)
    # axis=0 means 'lambda_fixed', axis=1 means 'lambda_sweep':
    mg = rename_keys(mg, {
        sweep_lambda_name: "lambda_sweep",
        fixed_lambda_name: "lambda_fixed"})
    return mg

def extract_ml_meshgrid(ml_meshgrid, sweep_lambda_index):
    assert sweep_lambda_index in [0, 1]
    fim = ml_meshgrid["fim_" + str(sweep_lambda_index) * 2]
    # axis=0 means 'lambda1', axis=1 means 'lambda0'
    if sweep_lambda_index == 1:
        fim = fim.T
    # axis=0 means 'lambda_fixed', axis=1 means 'lambda_sweep'
    return {
        "lambda_sweep": ml_meshgrid["lambda" + str(sweep_lambda_index)],
        "lambda_fixed": ml_meshgrid["lambda" + str(1 - sweep_lambda_index)],
        "fim": fim}

def w_smoothing(x, y, axis=0, extra_padding=1):
    """
    Smoothens the w.

    Args:
        x: 1-d np.ndarray representing x-coordinates of the w plot
          (unchanged by this function).
        y: y-coordinates to smoothen.
        axis: axis along which to apply the smoothing.
        extra_padding: number of additional values for padding.
    """
    y = np.moveaxis(y, axis, 0)
    n = y.shape[0]
    batch_size = y.shape[1:]
    assert x.shape == (n,)
    padding_size = 1 + extra_padding
    y_padding = np.ones(shape=(padding_size, *batch_size), dtype=y.dtype)
    y0 = np.concatenate([y_padding, y, y_padding], axis=0)
    dxl = x[1] - x[0]
    dxr = x[-1] - x[-2]
    x0 = np.concatenate([
        x[0] - np.arange(1, padding_size + 1) * dxl,
        x,
        x[-1] + np.arange(1, padding_size + 1) * dxr])
    x0 = x0.reshape(n + 2 * padding_size, *[1] * len(batch_size))
    y0left = np.maximum.accumulate(y0 + x0) - x0
    y0right = (np.maximum.accumulate(y0[::-1] + x0) - x0)[::-1]
    y1 = np.maximum(np.maximum(y0left, y0), y0right)
    y2 = np.maximum(y1[1:-1], (y1[2:] + y1[:-2]) / 2)
    y2 = np.moveaxis(y2, 0, axis)
    return y2

def get_gs_peaks(
        gs_mg, xmin=None, xmax=None, margin1=3/64, margin2=6/64,
        min_prominence=1.0):
    """
    Get the peaks of the ground state FIM.

    Args:
        gs_mg: dict describing the ground state FIM (in the meshgrid format).
            Keys: "lambda_fixed", "lambda_sweep", "fim".
            Note that gs_mg cannot typically be obtained by reshaping
            gs_fim_mgrid:
            - gs_fim_mgrid values correspond to lambda_fixed between the
                values in the original grid.
            - other methods produce FIM estimates for lambda_fixed on the
                original grid.
        xmin: beginning of the range of lambda_sweep.
        xmax: end of the range of lambda_sweep.
        margin1: size of the outer margin. If there is a peak within this
            margin, we ensure there is a peak within margin2 for neighboring
            points (by inserting artificial peak if necessary).
        margin2: size of the inner margin. Peaks within this margin are
            not used in the accuracy calculation.
        min_prominence: ignore peaks with prominence less than this value.
    """
    fim = gs_mg["fim"] # axis 0: lambda_fixed, axis 1: lambda_sweep
    lambda_sweep = gs_mg["lambda_sweep"]
    lambda_fixed = gs_mg["lambda_fixed"]

    assert np.all(lambda_sweep[:-1] < lambda_sweep[1:]), (
            f"lambda_sweep is not sorted: {lambda_sweep}")

    if xmin is None:
        xmin = 1.5 * lambda_sweep[0] - 0.5 * lambda_sweep[1]

    if xmax is None:
        xmax = 1.5 * lambda_sweep[-1] - 0.5 * lambda_sweep[-2]

    assert margin2 >= margin1
    assert 2 * margin2 < xmax - xmin

    mean_fim = np.mean(fim)
    fim = np.pad(
        fim, [(0, 0), (1, 1)], mode='constant', constant_values=mean_fim)
    peaks = [
        scipy.signal.find_peaks(v, prominence=min_prominence)[0] - 1
        for v in fim]
    peak_xs = [lambda_sweep[p] for p in peaks]

    min_peak = np.array([np.min(p, initial=xmax) for p in peak_xs])
    min_neighbour_peak = np.pad(
            min_peak, [(1, 1)], mode='constant', constant_values=xmax)
    min_neighbour_peak = np.minimum(
            min_neighbour_peak[:-2], min_neighbour_peak[2:])
    add_left = np.arange(len(min_peak))[
            (min_peak > xmin + margin2)
            & (min_neighbour_peak <= xmin + margin1)]

    max_peak = np.array([np.max(p, initial=xmin) for p in peak_xs])
    max_neighbour_peak = np.pad(
            max_peak, [(1, 1)], mode='constant', constant_values=xmin)
    max_neighbour_peak = np.maximum(
            max_neighbour_peak[:-2], max_neighbour_peak[2:])
    add_right = np.arange(len(max_peak))[
            (max_peak < xmax - margin2)
            & (max_neighbour_peak >= xmax - margin1)]

    for i in add_left:
        peak_xs[i] = np.concatenate([[xmin], peak_xs[i]])
    for i in add_right:
        peak_xs[i] = np.concatenate([peak_xs[i], [xmax]])
    is_inner = [
        ((xmin + margin2 < p) & (p < xmax - margin2)).astype(bool)
        for p in peak_xs]
    lambda_fixed_ii = np.concatenate([
        np.full(len(p), i, dtype=int) for i, p in enumerate(peak_xs)])
    return {
        "lambda_fixed_ii": lambda_fixed_ii,
        "lambda_fixed": lambda_fixed[lambda_fixed_ii],
        "lambda_sweep": np.concatenate(peak_xs),
        "is_inner": np.concatenate(is_inner),
        "is_single": np.concatenate([
            np.full(len(p), len(p) == 1, dtype=bool) for p in peak_xs]),
        "num_peaks": np.array([len(p) for p in peak_xs]),
        "num_inner_peaks": np.array([np.sum(flags) for flags in is_inner])}

def get_w_peaks(
        lambda_fixed, lambda_sweep, w_accuracy, num_peaks, postprocess=False,
        lambda_fixed_expected=None):
    """
    Extract peaks from van Nieuwenburg's W.

    Args:
        lambda_fixed: 1D np.ndarray
        lambda_sweep: 1D np.ndarray
        w_accuracy: 2D array with axis 0 corresponding to lambda_fixed
            and axis 1 corresponding to lambda_sweep.
        num_peaks: number of peaks (per each lambda_fixed) to extract.
        postprocess: whether to postprocess w_accuracy before extracting peaks.
        lambda_fixed_expected: if not None, verify that lambda_fixed values
            of the output are matching the expected values.
    """
    assert w_accuracy.shape == (len(lambda_fixed), len(lambda_sweep))
    assert num_peaks.shape == (len(lambda_fixed),)
    if postprocess:
        w_accuracy = w_smoothing(
            lambda_sweep, w_accuracy, axis=1, extra_padding=1)
    else:
        w_accuracy = np.pad(
            w_accuracy, [(0, 0), (1, 1)], mode='constant', constant_values=1.0)
    x = lambda_sweep
    x = np.concatenate([[2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]])
    (peak_ifixed,), peak_x = find_peaks_v(x, w_accuracy, num_peaks, axis=1)
    res_lambda_fixed = lambda_fixed[peak_ifixed]
    if lambda_fixed_expected is not None:
        assert np.array_equal(res_lambda_fixed, lambda_fixed_expected)
    return {
        "lambda_fixed_ii": peak_ifixed,
        "lambda_fixed": res_lambda_fixed,
        "lambda_sweep": peak_x}

def get_pca_peak(x, num_peaks):
    """
    Returns:
        Array of length `num_peaks` with the number of elements of `x`
        to count until each peak. E.g. if res[0] = 3, then the peak
        is assumed to be between x[2] and x[3].
    """
    num_points, num_features = x.shape
    scale = np.sum(np.std(x, axis=0)**2)**0.5
    dx0 = np.sum((x[1:] - x[:-1])**2, axis=1)**0.5
    x0 = np.empty(shape=(num_points, ), dtype=x.dtype)
    x0[0] = 0.0
    np.cumsum(dx0, out=x0[1:])
    x0 = x0.reshape((num_points, 1)) * 0.5
    x = np.hstack([x0, x])
    n_clusters = num_peaks+1
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(x)
    cluster_labels = kmeans.labels_
    cluster_xmeans = []
    xrange = np.arange(len(x))
    for i in range(n_clusters):
        cluster_xmeans.append(np.mean(xrange[cluster_labels == i]))
    ii = np.argsort(cluster_xmeans)
    label_counts = np.bincount(cluster_labels)
    return np.cumsum(label_counts[ii][:-1])

def get_pca_peaks(
        lambda_fixed, lambda_sweep, pca, num_peaks, postprocess=False):
    if postprocess:
        lambda_sweep, pca = smoothen_classifim_1d(lambda_sweep, pca, axis=1)
    peak_ii = [get_pca_peak(x, n) for x, n in zip(pca, num_peaks)]
    lambda_fixed_ii = np.concatenate([
        np.full(len(p), i, dtype=int) for i, p in enumerate(peak_ii)])
    peak_ii = np.concatenate(peak_ii)
    return {
        "lambda_fixed_ii": lambda_fixed_ii,
        "lambda_fixed": lambda_fixed[lambda_fixed_ii],
        "lambda_sweep_ii": peak_ii,
        "lambda_sweep": (lambda_sweep[peak_ii] + lambda_sweep[peak_ii - 1]) / 2}

def find_peaks(x, y, num_peaks):
    peaks, properties = scipy.signal.find_peaks(y, prominence=0)
    ii = np.argsort(properties["prominences"])[::-1]
    ii = ii[:num_peaks]
    peaks = peaks[ii]
    x_left = x[peaks]
    has_tie = (y[peaks] == y[peaks + 1])
    while np.sum(has_tie) > 0:
        peaks += has_tie
        has_tie = (y[peaks] == y[peaks + 1])
    x_right = x[peaks]
    return np.sort((x_left + x_right) / 2)

def find_peaks_v(x, y, num_peaks, axis=0):
    """
    Computes the locations of the peaks of `y`. As find_peaks, but vectorized.

    Args:
        x: x-coordinates corresponding to values of y along the given axis.
        y: array peaks of which we are trying to find.
        num_peaks: how many peaks are we trying to find (for each value of all
            indices of y except the index along axis).
        axis: along which axis are we looking for peaks.

    Returns: pair with the following components:
        idx: A tuple of (len(y.shape) - 1) coordinate arrays corresponding
            to other coordinates of the peaks.
        x_vals: x coordinates of the peaks.
    """
    y = np.moveaxis(y, axis, -1)
    y_shape = y.shape
    batch_size = y_shape[:-1]
    n = y_shape[-1]
    assert x.shape == (n,), f"{x.shape} != {(n,)} ({y_shape=})"
    assert num_peaks.shape == batch_size
    prod_bs = np.prod(batch_size)
    y = y.reshape(prod_bs, n)
    num_peaks = num_peaks.reshape(prod_bs)
    res_idx = []
    res_x = []
    for i in range(prod_bs):
        cur_num_peaks = num_peaks[i]
        cur_res_x = find_peaks(x, y[i], cur_num_peaks)
        if cur_num_peaks > cur_res_x.shape[0]:
            cur_res_x = np.append(cur_res_x, 0.5)
        assert len(cur_res_x) == num_peaks[i], (
            f"len({cur_res_x}) != num_peaks[{i}] == {cur_num_peaks}")
        res_idx.append(np.full(fill_value=i, shape=cur_res_x.shape))
        res_x.append(cur_res_x)
    res_idx = np.concatenate(res_idx)
    res_x = np.concatenate(res_x)
    res_idx = np.unravel_index(res_idx, batch_size)
    return res_idx, res_x

def _smoothen_classifim_1d(x, y, kernel, axis, cut=0):
    y = np.moveaxis(y, axis, 0)
    y_shape = y.shape
    n = y_shape[0]
    batch_size = y_shape[1:]
    assert x.shape == (n,)
    assert len(kernel.shape) == 1
    p = np.ones_like(x)
    conv_p = scipy.signal.convolve(p, kernel)
    conv_x = scipy.signal.convolve(x, kernel)
    conv_y = scipy.signal.convolve(y, kernel.reshape((kernel.shape[0], *[1] * len(batch_size))))
    res_x = conv_x / conv_p
    res_y = conv_y / conv_p.reshape((len(conv_p), *[1] * len(batch_size)))
    if cut > 0:
        res_x = res_x[cut:-cut]
        res_y = res_y[cut:-cut]
    return res_x, np.moveaxis(res_y, 0, axis)

def interweave(a1, a2):
    assert a1.shape[1:] == a2.shape[1:]
    n1 = a1.shape[0]
    n2 = a2.shape[0]
    assert n1 - 1 <= n2
    assert n2 <= n1
    res = np.empty((n1 + n2, *a1.shape[1:]), dtype=a1.dtype)
    res[0::2] = a1
    res[1::2] = a2
    return res

@functools.lru_cache(maxsize=32)
def _variance_correction(size0, sigma0, size1):
    """
    Compute correction for kernel1 to get the same variance as kernel0
    if convolved with i.i.d. random variables.
    """
    kernel0 = np.exp(-(np.arange(size0) - (size0 - 1) / 2)**2 / sigma0**2)
    target_variance = np.sum(kernel0**2) / np.sum(kernel0)**2
    expr1 = (np.arange(size1) - (size1 - 1) / 2)**2 / sigma0**2
    variance_correction = 1.0
    for _ in range(10):
        kernel1 = np.exp(-expr1 * variance_correction)
        d_kernel1 = -expr1 * kernel1
        s1 = np.sum(kernel1)
        ds1 = np.sum(d_kernel1)
        s2 = np.sum(kernel1**2)
        ds2 = 2 * np.sum(kernel1 * d_kernel1)
        cur_variance = s2 / s1**2
        error = target_variance / cur_variance - 1
        variance_correction += error / (ds2 / s2 - 2 * ds1 / s1)
    assert np.abs(error) < 1e-13
    return variance_correction

def smoothen_classifim_1d(x, y, axis, kernel0_size=5, kernel0_sigma=1.0, cut=1):
    range0 = np.arange(kernel0_size) - (kernel0_size - 1)/2
    kernel0 = np.exp(-range0**2 / kernel0_sigma**2)
    range1 = np.arange(kernel0_size + 1) - (kernel0_size) / 2
    variance_correction = _variance_correction(
        kernel0_size, kernel0_sigma, kernel0_size + 1)
    sigma1 = kernel0_sigma / variance_correction**0.5
    kernel1 = np.exp(-range1**2 / sigma1**2)
    x0, y0 = _smoothen_classifim_1d(x, y, kernel0, axis, cut=1)
    x1, y1 = _smoothen_classifim_1d(x, y, kernel1, axis, cut=1)
    assert x0.shape[0] + 1 == x1.shape[0]
    res_x = interweave(x1, x0)
    y0 = np.moveaxis(y0, axis, 0)
    y1 = np.moveaxis(y1, axis, 0)
    res_y = interweave(y1, y0)
    res_y = np.moveaxis(res_y, 0, axis)
    return res_x, res_y

def get_classifim_peaks(
        ml_mg, num_peaks, postprocess=False, lambda_fixed_expected=None,
        lambda_fixed_tolerance=None):
    """
    Extract peaks from ClassiFIM predictions.

    Args:
        ml_mg: dict describing the ClassiFIM predictions. Keys:
            "lambda_fixed", "lambda_sweep", "fim"
        num_peaks: number of peaks (per each lambda_fixed) to extract.
        postprocess: whether to postprocess fim before extracting peaks.
        lambda_fixed_expected: if not None, verify that lambda_fixed values
            of the output are matching the expected values.
        lambda_fixed_tolerance: if not None, adjust lambda_fixed values
            by at most this value to match lambda_fixed_expected.
    """
    x = ml_mg["lambda_sweep"]
    fim = ml_mg["fim"]
    assert fim.shape == (len(ml_mg["lambda_fixed"]), len(x))
    assert num_peaks.shape == (len(ml_mg["lambda_fixed"]),)
    if postprocess:
        x, fim = smoothen_classifim_1d(x, fim, axis=1)
    (peak_ifixed,), peak_x = find_peaks_v(x, fim, num_peaks, axis=1)
    res_lambda_fixed = ml_mg["lambda_fixed"][peak_ifixed]
    if lambda_fixed_expected is not None:
        if lambda_fixed_tolerance is not None:
            np.testing.assert_allclose(
                res_lambda_fixed, lambda_fixed_expected)
            res_lambda_fixed = lambda_fixed_expected
        np.testing.assert_array_equal(res_lambda_fixed, lambda_fixed_expected)
    return {
        "lambda_fixed_ii": peak_ifixed,
        "lambda_fixed": res_lambda_fixed,
        "lambda_sweep": peak_x}

def set_error(group, x_gt, x_pred, x_gt_ii=None):
    """
    Args:
        group is a sorted array describing how x_gt and x_pred indices are split into groups:
            indices i and j are in the same group iff group[i] == group[j]
        x_gt describes a ground truth set for each of the groups. Its values within each group are sorted in ascending order.
        x_gt_ii: If specified, should be bool array: use only selected values of x_gt.
        x_pred describes predicted sets. Its values within each group are sorted in ascending order.
    Returns:
        Error is computed as follows:
            * For each j let x = x_gt[j]. Find y closest to x in values x_pred corresponding to the same group. Then compute the error_j = (x - y)**2.
            * Return average over j of error_j.
    """
    n = len(group)
    assert group.shape == (n,)
    assert x_gt.shape == (n,)
    assert x_pred.shape == (n,)
    if x_gt_ii is None:
        x_gt_ii = np.full(fill_value=True, shape=x_gt.shape)
    _, group_idx = np.unique(group, return_index=True)
    num_groups = len(group_idx)
    group_idx = np.concatenate([group_idx, [len(group)]])
    res = 0
    for g in range(num_groups):
        i0 = group_idx[g]
        i1 = group_idx[g+1]
        cur_x_gt = x_gt[i0:i1]
        cur_x_gt_ii = x_gt_ii[i0:i1]
        cur_x_gt = cur_x_gt[cur_x_gt_ii]
        if cur_x_gt.size == 0:
            continue
        cur_x_pred = x_pred[i0:i1]
        cur_error = (cur_x_gt[:, np.newaxis] - cur_x_pred[np.newaxis, :])**2
        cur_error = np.min(cur_error, axis=1)
        res += np.sum(cur_error)
    return res / np.sum(x_gt_ii)

# TODO:9: move to the test file.
def test_set_error():
    # Test
    group = np.array([1, 1, 1, 2, 2, 2, 3])
    x_gt = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 3.0])
    x_pred = np.array([2.0, 3.0, 5.0, 0.5, 2.5, 3.5, 4.0])

    # Error =
    #   mean([(1.0 - 2.0)**2, (2.0 - 2.0)**2, (3.0 - 3.0)**2, (1.5 - 0.5)**2, (1.5 - 2.5)**2, (3.5 - 3.5)**2, (3.0 - 4.0)**2]) =
    #   mean([1, 0, 0, 1, 0, 0, 1]) = 3 / 7
    res = set_error(group, x_gt, x_pred)
    assert np.allclose(res, 3/7), f"{res}"

test_set_error()

def compute_peak_rmses(gs_peaks, model_peaks_dict):
    """
    Args:
        gs_peaks: dict with keys
            'lambda_fixed', 'lambda_sweep', 'is_inner', 'is_single'
        model_peaks_dict: dict with elements key: value where key is model name
            and value is a dict.

    Returns:
        RMSE stats of the peak locations predicted by the models.
    """
    gs_peak_y = gs_peaks["lambda_fixed"]
    gs_peak_x = gs_peaks["lambda_sweep"]
    assert gs_peak_y.shape == gs_peak_x.shape
    assert len(gs_peak_y.shape) == 1
    ii = gs_peaks["is_inner"]
    assert ii.shape == gs_peak_y.shape
    ii_single = ii & gs_peaks["is_single"]
    res = {
        "num_peaks": len(gs_peak_y),
        "num_acc_peaks": np.sum(ii),
        "num_single_peaks": np.sum(ii_single)}
    for key, value in model_peaks_dict.items():
        res[key] = set_error(
            gs_peak_y, x_gt=gs_peak_x, x_gt_ii=ii,
            x_pred=value["lambda_sweep"])**0.5

    for key, value in model_peaks_dict.items():
        res[key + "_single"] = set_error(
            gs_peak_y, x_gt=gs_peak_x, x_gt_ii=ii_single,
            x_pred=value["lambda_sweep"])**0.5
    return res
