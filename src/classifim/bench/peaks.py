"""
Computing peak accuracy for comparison with prior work.
"""

import classifim.bench.plot_tools
import functools
import itertools
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import sklearn.cluster
import sys
import warnings

def rename_keys(d, key_map):
    """
    Renames the keys of the dictionary `d` according to `key_map`.

    Args:
        d: The original dictionary.
        key_map: A dictionary mapping old keys to new keys.

    Returns: A new dictionary with keys renamed.
    """
    return {key_map.get(key, key): value for key, value in d.items()}

def extract_gt_meshgrid(fim_df, sweep_lambda_index):
    """
    Extract gt FIM in meshgrid format along a single sweep direction.

    Note: This function is different from meshgrid_transform_2D_fim
    in two main ways:
    - Only a single component of the FIM is extracted.
    - The output corresponds to the same lambda_fixed values as in the
        original grid. meshgrid_transform_2D_fim's corresponds to midpoints.
    """
    df = fim_df[fim_df["dir"] == str(sweep_lambda_index)]
    sweep_lambda_name = "lambda" + str(sweep_lambda_index)
    fixed_lambda_name = "lambda" + str(1 - sweep_lambda_index)
    mg = classifim.bench.plot_tools.df_to_meshgrid(
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

def get_gt_peaks(
        gt_mg, xmin=None, xmax=None, margin1=3/64, margin2=6/64,
        min_prominence=1.0):
    """
    Get the peaks of the ground state FIM.

    Args:
        gt_mg: dict describing the ground state FIM (in the meshgrid format).
            Keys: "lambda_fixed", "lambda_sweep", "fim".
            Note that gt_mg cannot typically be obtained by reshaping
            gt_fim_mgrid:
            - gt_fim_mgrid values correspond to lambda_fixed between the
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
            Note: peaks with prominence between min_prominence / 2 and
            min_prominence are still returned, but are not considered
            for the accuracy calculation.
    """
    fim = gt_mg["fim"] # axis 0: lambda_fixed, axis 1: lambda_sweep
    lambda_sweep = gt_mg["lambda_sweep"]
    lambda_fixed = gt_mg["lambda_fixed"]

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
    raw_peaks = [
        scipy.signal.find_peaks(v, prominence=min_prominence / 2)
        for v in fim]
    prominences = [p[1]["prominences"] for p in raw_peaks]
    peaks = [p[0] - 1 for p in raw_peaks]
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
        prominences[i] = np.concatenate([[0], prominences[i]])
    for i in add_right:
        peak_xs[i] = np.concatenate([peak_xs[i], [xmax]])
        prominences[i] = np.concatenate([prominences[i], [0]])
    try:
        is_inner = [
            ((xmin + margin2 < p)
                & (p < xmax - margin2)
                & (min_prominence <= pr)
            ).astype(bool)
            for p, pr in zip(peak_xs, prominences)]
    except ValueError as e:
        e.info = list(zip(peak_xs, prominences))
        raise e
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
        "num_inner_peaks": np.array([np.sum(flags) for flags in is_inner]),
        "xmin": xmin,
        "xmax": xmax}

def get_w_peaks(
        lambda_fixed, lambda_sweep, w_accuracy, num_peaks, postprocess=False,
        lambda_fixed_expected=None, xrange=None, return_pp=False):
    """
    Extract peaks from van Nieuwenburg's W.

    Args:
        lambda_fixed, lambda_sweep: 1D np.ndarrays with grid coordinates.
        w_accuracy: 2D array with axis 0 corresponding to lambda_fixed
            and axis 1 corresponding to lambda_sweep.
        num_peaks: number of peaks (per each lambda_fixed) to extract.
        postprocess: whether to postprocess w_accuracy before extracting peaks.
        lambda_fixed_expected: if not None, verify that lambda_fixed values
            of the output are matching the expected values.
        xrange: Used to blindly guess peaks if W predicts less peaks than
            needed.
        return_pp: whether to return the postprocessed w_accuracy.
    """
    assert w_accuracy.shape == (len(lambda_fixed), len(lambda_sweep))
    assert num_peaks.shape == (len(lambda_fixed),)
    res = {}
    if postprocess:
        w_accuracy = w_smoothing(
            lambda_sweep, w_accuracy, axis=1, extra_padding=1)
    else:
        assert not return_pp
        w_accuracy = np.pad(
            w_accuracy, [(0, 0), (1, 1)], mode='constant', constant_values=1.0)
    x = lambda_sweep
    x = np.concatenate([[2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]])
    if return_pp:
        res["pp"] = {
            "accuracy": w_accuracy,
            "lambda_sweep": x,
            "lambda_fixed": lambda_fixed}
    (peak_ifixed,), peak_x = find_peaks_v(
        x, w_accuracy, num_peaks, axis=1, xrange=xrange)
    res_lambda_fixed = lambda_fixed[peak_ifixed]
    if lambda_fixed_expected is not None:
        assert np.array_equal(res_lambda_fixed, lambda_fixed_expected)
    res["lambda_fixed_ii"] = peak_ifixed
    res["lambda_fixed"] = res_lambda_fixed
    res["lambda_sweep"] = peak_x
    return res

def get_pca_peak(x, num_peaks):
    """
    Args:
        x: 2D array with axis 0 corresponding to lambda_sweep
            and axis 1 corresponding to the PCA components.
        num_peaks: number of peaks to extract.

    Returns:
        Array of length `num_peaks` with the number of elements of `x`
        to count until each peak. E.g. if res[0] = 3, then the peak
        is assumed to be between x[2] and x[3].
    """
    num_points, num_features = x.shape
    # Suppress high-frequency noise by scaling down features
    # with high total variation:
    total_variation = np.sum(np.abs(x[1:] - x[:-1]), axis=0)
    avg_tv = np.mean(total_variation)
    x = x * avg_tv / (0.1 * avg_tv + 0.9 * total_variation[None, :])
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
        grid, pca, num_peaks, postprocess=False,
        sweep_lambda_index=0):
    """
    Compute peak locations using SPCA data.

    Args:
        grid: a list of 1D arrays with grid coordinates
            or tuples (start, stop, num_points).
        pca: 3D array with
            - axis 0 corresponding to lambda0
            - axis 1 corresponding to lambda1
            - axis 2 corresponding to the PCA components.
        num_peaks: 1D array, number of peaks (per each lambda_fixed) to extract.
        postprocess: whether to smoothen pca before extracting peaks.
        sweep_lambda_index: index of the lambda_sweep axis in pca.
    """
    assert sweep_lambda_index in [0, 1]
    assert len(grid) == 2
    grid = grid[:]
    for i in range(len(grid)):
        if isinstance(grid[i], tuple):
            grid[i] = np.linspace(*grid[i])
    # Ensure that axis 1 corresponds to lambda_sweep:
    if sweep_lambda_index == 0:
        lambda_sweep = grid[0]
        lambda_fixed = grid[1]
        pca = pca.swapaxes(0, 1)
    else:
        lambda_sweep = grid[1]
        lambda_fixed = grid[0]
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

def blindly_guess_peaks(x, num_peaks):
    """
    Adds an uninformed guess of additional `num_peaks` peak locations.

    Consider an algorithm predicting the peak locations. Say we know
    there are $n$ peaks, but the algorithm only predicted $k < n$ peaks.
    Which $n$ peak locations do we submit then? The best guess is to fill
    the gaps between the predicted peak locations as 'equidistantly` as
    possible. This function tries to do that. It is not guaranteed to
    provide an optimal solution.

    Args:
        x: sorted 1D array of peak locations including the boundaries.
        num_peaks: number of peaks to add.

    Returns:
        x with `num_peaks` additional peak locations inserted.
    """
    if num_peaks == 0:
        return x
    dx = np.diff(x)
    assert np.all(dx > 0)
    k = len(dx) # number of intervals
    eps = np.nextafter(x.dtype.type(1), np.inf) - x.dtype.type(1)
    lambda_lb = (1 - 4 * eps) * 1.5 / np.max(dx)
    x_len = x[-1] - x[0]
    lambda_ub = (1 + 4 * eps) * min(
        (num_peaks + 1) / np.max(dx),
        (num_peaks + 1.5 * k) / x_len)
    cur_lambda = min(0.9 * lambda_ub, (num_peaks + k) / x_len)
    ns_lb = None
    ns_ub = None
    while True:
        ns = np.maximum(0, (cur_lambda * dx - 0.5).astype(int))
        cur_num_peaks = np.sum(ns)
        if cur_num_peaks == num_peaks:
            break
        if cur_num_peaks < num_peaks:
            lambda_lb = cur_lambda
            ns_lb = ns
            cur_lambda = min(
                lambda_lb + 0.9 * (lambda_ub - lambda_lb),
                cur_lambda * (1 + num_peaks) / (1 + cur_num_peaks))
        if cur_num_peaks > num_peaks:
            lambda_ub = cur_lambda
            ns_ub = ns
            cur_lambda = max(
                lambda_lb + 0.1 * (lambda_ub - lambda_lb),
                cur_lambda * (1 + num_peaks) / (1 + cur_num_peaks))
        if cur_lambda == lambda_lb or cur_lambda == lambda_ub:
            ns = None
            break
    if ns is None:
        assert ns_lb is not None and ns_ub is not None
        dns = ns_ub - ns_lb
        assert np.all((dns == 0) | (dns == 1))
        ns = ns_lb
        cur_num_peaks = np.sum(ns)
        dns_idx = np.nonzero(dns)[0]
        dns_idx = dns_idx[:num_peaks - cur_num_peaks]
        ns[dns_idx] += 1
    res = [
        x[i] + (x[i + 1] - x[i]) * np.arange(n + 1) / (n + 1)
        for i, n in enumerate(ns)]
    res.append([x[-1]])
    return np.concatenate(res)


def find_peaks_v(x, y, num_peaks, xrange, axis=0):
    """
    Computes the locations of the peaks of `y`. As find_peaks, but vectorized.

    Args:
        x: x-coordinates corresponding to values of y along the given axis.
        y: array peaks of which we are trying to find.
        num_peaks: how many peaks are we trying to find (for each value of all
            indices of y except the index along axis).
        xrange: pair (xmin, xmax) describing the interval for x values.
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
            if xrange is None:
                raise ValueError(
                    "`y` didn't have enough peaks and xrange "
                    "for blind guessing was not provided: "
                    f"{cur_res_x.shape[0]} < {cur_num_peaks} "
                    f"for idx={np.unravel_index(i, batch_size)}.")
            xmin, xmax = xrange
            cur_res_x = blindly_guess_peaks(
                np.concatenate([[xmin], cur_res_x, [xmax]]),
                cur_num_peaks - cur_res_x.shape[0])
            cur_res_x = cur_res_x[1:-1]
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
        lambda_fixed_tolerance=None, xrange=None, return_pp=False):
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
        xrange: Interval for peaks, used if guessing is needed.
        return_pp: whether to return the postprocessed fim.
    """
    x = ml_mg["lambda_sweep"]
    fim = ml_mg["fim"]
    assert fim.shape == (len(ml_mg["lambda_fixed"]), len(x))
    assert num_peaks.shape == (len(ml_mg["lambda_fixed"]),)
    res = {}
    if postprocess:
        x, fim = smoothen_classifim_1d(x, fim, axis=1)
        res["pp"] = {
            "lambda_sweep": x,
            "lambda_fixed": ml_mg["lambda_fixed"],
            "fim": fim}
    else:
        assert not return_pp
    (peak_ifixed,), peak_x = find_peaks_v(x, fim, num_peaks, axis=1, xrange=xrange)
    res_lambda_fixed = ml_mg["lambda_fixed"][peak_ifixed]
    if lambda_fixed_expected is not None:
        if lambda_fixed_tolerance is not None:
            np.testing.assert_allclose(
                res_lambda_fixed, lambda_fixed_expected,
                atol=lambda_fixed_tolerance)
            res_lambda_fixed = lambda_fixed_expected
        np.testing.assert_array_equal(res_lambda_fixed, lambda_fixed_expected)
    res["lambda_fixed_ii"] = peak_ifixed
    res["lambda_fixed"] = res_lambda_fixed
    res["lambda_sweep"] = peak_x
    return res

def set_error(
        group, x_gt, x_pred, x_gt_ii=None, xmin=None, xmax=None, verbosity=1):
    """
    Args:
        group is a sorted array describing how x_gt and x_pred indices are split
            into groups: indices i and j are in the same group
            iff group[i] == group[j]
        x_gt describes a ground truth set for each of the groups. Its values
            within each group are sorted in ascending order.
        x_gt_ii: If specified, should be bool array: use only selected values of
            x_gt.
        x_pred describes predicted sets. Its values within each group are sorted
            in ascending order.
        xmin, xmax: boundary of the region. If specified, error is bounded by the
            distance to the nearest boundary.
        verbosity: 0: no warnings, 1: warnings,
            2+: debug info may be added in the future.

    Error is computed as follows:
        * For each j let x = x_gt[j]. Find y closest to x in values x_pred
            corresponding to the same group.
            Then compute the error_j = (x - y)**2.
    Returns: mean_j(error_j)
    """
    n = len(group)
    assert group.shape == (n,)
    assert x_gt.shape == (n,)
    assert x_pred.shape == (n,)
    if x_gt_ii is None:
        x_gt_ii = np.full(fill_value=True, shape=x_gt.shape)
    if np.sum(x_gt_ii) == 0:
        if verbosity >= 1:
            warnings.warn("No ground truth values selected.")
        return np.nan
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
        if xmin is not None:
            cur_error = np.minimum(cur_error, (cur_x_gt - xmin)**2)
        if xmax is not None:
            cur_error = np.minimum(cur_error, (cur_x_gt - xmax)**2)
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

def compute_peak_rmses(gt_peaks, model_peaks_dict, verbosity=1):
    """
    Args:
        gt_peaks: dict with keys
            'lambda_fixed', 'lambda_sweep', 'is_inner', 'is_single'
        model_peaks_dict: dict with elements key: value where key is model name
            and value is a dict.
        verbosity: 0: no warnings, 1: warnings, 2+: possibly additional info.

    Returns:
        RMSE stats of the peak locations predicted by the models.
    """
    gt_peak_y = gt_peaks["lambda_fixed"]
    gt_peak_x = gt_peaks["lambda_sweep"]
    assert gt_peak_y.shape == gt_peak_x.shape
    assert len(gt_peak_y.shape) == 1
    ii = gt_peaks["is_inner"]
    assert ii.shape == gt_peak_y.shape
    ii_single = ii & gt_peaks["is_single"]
    res = {
        "num_peaks": len(gt_peak_y),
        "num_acc_peaks": np.sum(ii),
        "num_single_peaks": np.sum(ii_single)}
    boundary_kwargs = {
        key: gt_peaks[key] for key in ["xmin", "xmax"] if key in gt_peaks}
    for key, value in model_peaks_dict.items():
        res[key] = set_error(
            gt_peak_y, x_gt=gt_peak_x, x_gt_ii=ii,
            x_pred=value["lambda_sweep"], **boundary_kwargs,
            verbosity=verbosity)**0.5

    for key, value in model_peaks_dict.items():
        res[key + "_single"] = set_error(
            gt_peak_y, x_gt=gt_peak_x, x_gt_ii=ii_single,
            x_pred=value["lambda_sweep"],
            verbosity=verbosity)**0.5
    return res

def paired_t_tests(
        df, verbose=False, metric_name="smooth_peaks", key_name="seed",
        pval_format=".3f"):
    grouped = df.groupby("method_name")

    method_dfs = {}
    for method, group in grouped:
        metrics = group.set_index(key_name)[metric_name]
        v = np.mean(metrics)
        method_dfs[method] = (v, metrics)
    method_names = list(method_dfs.keys())
    results = {}
    for method1, method2 in itertools.combinations(method_names, 2):
        v1, metrics1 = method_dfs[method1]
        v2, metrics2 = method_dfs[method2]
        aligned = pd.concat(
            [metrics1, metrics2], axis=1, keys=[method1, method2]).dropna()
        t_stat, p_val = scipy.stats.ttest_rel(
                aligned[f"{method1}"],
                aligned[f"{method2}"])
        results[(method1, method2)] = {
            "t_stat": t_stat, "p_val": p_val, "mean1": v1, "mean2": v2}
        if verbose:
            sign = "≈"
            if p_val < 0.05:
                sign = "<" if v1 < v2 else ">"
            print(
                f"{method1} vs {method2}: "
                f"{v1:.3f} {sign} {v2:.3f} "
                f"(t-stat={t_stat:.3f}, p={p_val:{pval_format}})")
    return results

DEFAULT_PEAK_ERROR_METRICS = {
    "num_peaks": ["peaks", "smooth_peaks"],
    "num_peaks_single": ["peaks_single", "smooth_peaks_single"]}

def peak_errors2d_to_df(peak_errors, top_keys=None, metric_names=None):
    """
    Convert peak errors to a DataFrame.

    Args:
        peak_errors: list, with each element being a dict with keys
            - seed, sweep_lambda_index
            - values for count columns
            - methods: list of method names. Should be the same for all elements.
            - Values with keys "{method_name}_{metric_name}".
        top_keys: List of top level keys.
        metric_names: dict with count columns as keys and lists of metric names
            as values.

    Returns:
        DataFrame with columns:
            - "seed", "sweep_lambda_index", "method_name",
            - columns corresponding to count and metric names.
    """
    if top_keys is None:
        top_keys = ["seed", "sweep_lambda_index"]
    dfs = []
    metric_names = metric_names or DEFAULT_PEAK_ERROR_METRICS
    count_names = list(metric_names.keys())
    metric_names_flat = list(itertools.chain(*metric_names.values()))
    for cur_peak_errors in peak_errors:
        cur_df = {
            key: cur_peak_errors[key]
            for key in top_keys + count_names}
        cur_df.update(**{k: [] for k in metric_names_flat + ["method_name"]})
        for method_name in cur_peak_errors["methods"]:
            cur_df["method_name"].append(method_name)
            for metric_name in metric_names_flat:
                cur_df[metric_name].append(
                    cur_peak_errors[method_name + "_" + metric_name])
        dfs.append(pd.DataFrame(cur_df))
    peak_metrics_df_raw = pd.concat(dfs, ignore_index=True)
    return peak_metrics_df_raw

def strict_series_sum(x):
    return x.sum(skipna=False)

def peak_errors2d_agg_sweeps(
        peak_metrics_df_raw, metric_names=None, keys=None):
    """
    Aggregate peak errors over sweeps.

    Metrics are assumed to be RMSEs, i.e. they are aggregated using the formula
    $x = \sqrt{\sum_j n_j x_j^2 / \sum_j n_j}$.

    Args:
        peak_metrics_df_raw: DataFrame with columns
            - "seed", "sweep_lambda_index", "method_name",
            - columns corresponding to count and metric names.
        metric_names: dict with count columns as keys and lists of metric names
            as values,
        keys: list of keys to group by.

    Returns:
        DataFrame with columns:
            - "seed", "method_name", "num_rows"
            - columns corresponding to count and metric names.
    """
    if keys is None:
        keys = ["seed", "method_name"]
    df = peak_metrics_df_raw.copy()
    metric_names = metric_names or DEFAULT_PEAK_ERROR_METRICS
    count_names = list(metric_names.keys())
    metric_names_flat = list(itertools.chain(*metric_names.values()))
    for count_name, cur_metric_names in metric_names.items():
        count_s = df[count_name]
        for metric_name in cur_metric_names:
            metric_s = df[metric_name]
            df[metric_name] = np.where(count_s == 0., 0., metric_s**2 * count_s)
    df = df.groupby(keys).agg(
        num_rows=("sweep_lambda_index", "size"),
        **{k: (k, strict_series_sum) for k in count_names + metric_names_flat})
    for count_name, cur_metric_names in metric_names.items():
        count_s = df[count_name]
        for metric_name in cur_metric_names:
            metric_s = df[metric_name]
            df[metric_name] = np.sqrt(metric_s / count_s)
    df.reset_index(inplace=True)
    return df

