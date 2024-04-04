import numpy as np
import pandas as pd
from tqdm import tqdm

from .plot_tools import as_data_frame

FIDELITY_DIRECTIONS_2D = {
    "0": np.array([1, 0]),
    "1": np.array([0, 1]),
    "+": np.array([1, 1]),
    "-": np.array([1, -1])}
# Note that '+' and '-' are not normalized.
# To get normalized fidelity susceptibility along these directions, divide by 2.

def classical_fidelity_from_probs(probs0, probs1):
    return ((probs0 * probs1)**0.5).sum()

def compute_2d_fim(
        lambdas_to_params_f, probs_cache=None, probs_f=None, resolution=64,
        fidelity_f=classical_fidelity_from_probs,
        verbose=False):
    """
    Compute fidelity susceptibilities for a 2D grid of lambda values.

    The grid is expected to be a square grid of resolution x resolution size:
    it consists of points (lambda0, lambda1) where lambda0 and lambda1 are in
    np.arange(resolution) / resolution.

    This function computes chi_fc (estimate of classical fidelity
    susceptibility) depending on lambda0, lambda1, and dir (direction),
    where dir is one of the following:
    - '0' corresponds to v = (1, 0) / resolution
    - '1' corresponds to v = (0, 1) / resolution
    - '+' corresponds to v = (1, 1) / resolution
    - '-' corresponds to v = (1, -1) / resolution
    You can use FIDELITY_DIRECTIONS_2D to convert between dir character
    and the direction $\vec{v}$.

    Fidelity susceptibility is defined as
    $$\chi_{F_c}(\vec{\lambda}, \vec{v}) =
        \vec{v}^T \chi_{F_c}(\vec{\lambda}) \vec{v} =
        \frac{\partial^2}{\partial \delta^2}
        F(\vec{\lambda}, \vec{\lambda} + \delta \vec{v}) \bigg|_{\delta = 0}$$
    where $F(\vec{\lambda}, \vec{\lambda}')$ is the fidelity between the states
    at $\vec{\lambda}$ and $\vec{\lambda}'$.

    Args:
        lambdas_to_params_f: function that maps lambda values to parameters.
        probs_cache: cache of probability distributions.
        probs_f: function that returns probability distribution for given
            parameters. Exactly one of probs_cache and probs_f must be
            specified (not None).
        resolution: number of points in each dimension of lambda grid.
        fidelity_f: function that computes fidelity between two states.
        verbose: if True, print progress.

    Returns:
        pd.DataFrame with columns:
            lambda0, lambda1, dir, chi_fc
    """
    if probs_f is None:
        assert probs_cache is not None
        probs_f = lambda params: probs_cache.get_ground_state(params)["probs"]
    else:
        assert probs_cache is None
    # Create a grid of lambda_int values, where
    # lambda_int[i] = int(lambda[i] * resolution).
    lambda_ints = np.mgrid[0:resolution, 0:resolution].reshape(2, -1).T
    res = {
        "lambda0": [],
        "lambda1": [],
        "dir": [],
        "chi_fc": []}
    if verbose:
        pbar = tqdm(total=4 * resolution**2)
    for direction, v in FIDELITY_DIRECTIONS_2D.items():
        dir2 = np.dot(v, v)
        # i_v is v rotated 90 degrees counterclockwise, i.e.
        # 1j * v if v is interpreted as a complex number.
        i_v = np.array([-v[1], v[0]])
        ii = np.lexsort((
            lambda_ints[:, 1],
            lambda_ints[:, 0],
            lambda_ints @ i_v))
        prev_lambda_int = (-2, -2)
        prev_probs = None
        for i in ii:
            cur_lambda_int = lambda_ints[i]
            delta_lambda_int = cur_lambda_int - prev_lambda_int
            delta_along_v = np.dot(delta_lambda_int, v)
            delta_perp_v = np.dot(delta_lambda_int, i_v)
            if delta_perp_v != 0:
                prev_probs = None
            else:
                assert delta_along_v == dir2
            params = lambdas_to_params_f(*(cur_lambda_int / resolution))
            cur_probs = probs_f(params)
            if prev_probs is not None:
                mid_lambda = (cur_lambda_int + prev_lambda_int) / (2 * resolution)
                fidelity = fidelity_f(prev_probs, cur_probs)
                fim = (1 - fidelity) * 8 * resolution**2
                res["lambda0"].append(mid_lambda[0])
                res["lambda1"].append(mid_lambda[1])
                res["dir"].append(direction)
                res["fim"].append(fim)
            if verbose:
                pbar.update()
            prev_lambda_int = cur_lambda_int
            prev_probs = cur_probs
    if verbose:
        pbar.close()
    return pd.DataFrame(res)


def compute_2d_fim_from_fidelities(
        fidelities, resolution=64, scaling_resolution=None, verbose=False):
    """
    Compute FIM from pre-computed fidelities.

    Args:
        fidelities: dict containing numpy arrays of fidelity values for each
            direction. fidelities.keys() are tuples corresponding to values
            of FIDELITY_DIRECTIONS_2D.
            Fidelity between lambda_int_a and lambda_int_b is stored in
            fidelities[tuple_v][tuple(np.minimum(lambda_int_a, lambda_int_b))],
            where tuple_v is tuple(lambda_int_b - lambda_int_a).
        resolution: size of the grid (64x64).
        scaling_resolution: resolution to be used for scaling lambda values
            (default: resolution).
        verbose: if True, print progress.

    Returns:
        pd.DataFrame with columns:
            lambda0, lambda1, dir, chi_fc
    """
    if scaling_resolution is None:
        scaling_resolution = resolution
    res = {
        "lambda0": [],
        "lambda1": [],
        "dir": [],
        "fim": []}

    grid_shape = np.array([resolution, resolution])
    dir_iterable = FIDELITY_DIRECTIONS_2D.items()
    if verbose:
        dir_iterable = tqdm(dir_iterable)
    for direction, v in dir_iterable:
        cur_shape = grid_shape - np.abs(v)
        # lambda_int_idx.shape = (2, num_lambda0, num_lambda1):
        lambda_int_idx = np.mgrid[0:cur_shape[0], 0:cur_shape[1]]
        # lambda_int_a = lambda_int_idx + np.maximum(0, -v)[:, None, None]
        # lambda_int_b = lambda_int_idx + np.maximum(0, v)[:, None, None]
        # lambda_int_mid = (lambda_int_a + lambda_int_b) / 2
        #     = lambda_int_idx + np.abs(v)[:, None, None] / 2
        lambda_int_mid = lambda_int_idx + np.abs(v)[:, None, None] / 2
        lambdas_mid = (lambda_int_mid / scaling_resolution).reshape(2, -1)
        cur_lambda0, cur_lambda1 = lambdas_mid
        cur_fidelities = fidelities[tuple(v)]
        assert cur_fidelities.shape == tuple(cur_shape)
        cur_fidelities = cur_fidelities.ravel()
        cur_fim = (1 - cur_fidelities) * 8 * scaling_resolution**2
        cur_dir = np.full(cur_fim.shape, direction.encode("utf-8"))
        assert (cur_lambda0.shape == cur_lambda1.shape
                == cur_fim.shape == cur_dir.shape)
        res["lambda0"].append(cur_lambda0)
        res["lambda1"].append(cur_lambda1)
        res["dir"].append(cur_dir)
        res["fim"].append(cur_fim)

    res = {key: np.concatenate(value) for key, value in res.items()}

    return pd.DataFrame(res)

def encode_for_npz(d):
    """
    Encodes a dictionary for saving to npz file.
    """
    npz = {}
    for key, value in d.items():
        np_value = np.array(value)
        if np_value.dtype == object:
            np_value = np_value.astype(bytes)
        npz[key] = np_value
    return npz


def save_2d_chifc(chifc, filename, verbose=False, chifc_name="chifc"):
    """
    Save the output of compute_2d_chifc to a file.
    """
    chifc_npz = encode_for_npz(chifc)

    np.savez_compressed(
        filename,
        **chifc_npz)
    if verbose:
        print(f"Saved {chifc_name} to '{filename}'")

def meshgrid_transform_2D_fim(
        fim_df, scaling_resolution=None, verbose=False):
    """
    Transform 2D fim dataframe to a meshgrid.

    Args:
        fim_df: dataframe with columns lambda0, lambda1, dir, fim
            as produced by compute_2d_fim.
        scaling_resolution: resolution to be used for scaling lambda values.
            If None, use the number of unique lambda values in fim_df.

    Note:
        Returned 2D arrays fim_00, fim_01, fim_11 have 2 axes:
        - axis 0 corresponds to lambda1 (vertical axis in the plot),
        - axis 1 corresponds to lambda0 (horizontal axis in the plot).
        This is the convention used by `matplotlib.pyplot.pcolormesh`.

    Returns:
        A dictionary with the following keys:
        - lambda0, lambda1: 1D arrays of length resolution - 1
        np.ndarrays of shape (resolution - 1, resolution - 1):
        - fim_00, fim_01, fim_11 - fidelity susceptibility components
        - fim_err0 - mismatch between the Tr(fim)/2 computed from
            (a) the horizontal and vertical components and
            (b) the diagonal components.
        - fim_err1 - correction needed to ensure that estimated fim is
            positive semidefinite.
        Note that the points in the new grid are mid-points of the original
        grid, hence the size is smaller: resolution - 1 instead of resolution.
    """
    fim_df = as_data_frame(fim_df, decode=True)
    resolution = len(np.unique(fim_df["lambda1"][fim_df["dir"] == "0"]))
    if verbose:
        print(f"{resolution=}")
    if scaling_resolution is None:
        scaling_resolution = resolution
    assert fim_df.shape[0] == (resolution - 1) * (4 * resolution - 2), (
        f"{fim_df.shape[0]} != {resolution - 1} * {4 * resolution - 2}.")
    dir_dict_df = dict(iter(fim_df.groupby("dir")))
    dir_keys = ["+", "-", "0", "1"]
    assert list(dir_dict_df.keys()) == dir_keys
    # shifts[i][j] is 0 if lambda[i] corresponds to the nodes of the original
    # grid in dir_dict_df[j], and 1 if lambda[i] corresponds to the mid-points.
    shifts = np.array([[1, 1, 1, 0], [1, 1, 0, 1]])
    # resolutions[i] is the resolution of dir_dict_df along lambda[i] direction:
    resolutions = resolution - shifts
    lambda_colnames = ["lambda0", "lambda1"]
    dir_dict_np = {}
    for dir_idx, (key, df) in enumerate(dir_dict_df.items()):
        assert key == dir_keys[dir_idx]
        assert df.shape[0] == resolutions[0][dir_idx] * resolutions[1][dir_idx]
        lambdas_ints = []
        for i in range(2):
            cur_half_ints = (
                df[lambda_colnames[i]].to_numpy() * scaling_resolution
                + 0.5 * (1 - shifts[i][dir_idx]))
            cur_ints = cur_half_ints.astype(int) # n + 0.5 -> n
            max_err = np.max(np.abs(cur_half_ints - cur_ints - 0.5))
            assert max_err < 1e-6, (
                f"max_err = {max_err} >= 1e-6. "
                f"cur_dir = {key}, {i=}, "
                f"{cur_half_ints=}")
            lambdas_ints.append(cur_ints)

        df_index = lambdas_ints[1] * resolutions[0][dir_idx] + lambdas_ints[0]
        assert np.all(np.sort(df_index) == np.arange(df.shape[0]))
        df["index"] = df_index
        df.set_index("index", inplace=True, verify_integrity=True)
        df.sort_index(inplace=True)
        dir_dict_np[key] = df["fim"].to_numpy().reshape(
            tuple(reversed(resolutions[:, dir_idx])))
    dir_dict_np["0"] = (dir_dict_np["0"][:-1, :] + dir_dict_np["0"][1:, :]) / 2
    dir_dict_np["1"] = (dir_dict_np["1"][:, :-1] + dir_dict_np["1"][:, 1:]) / 2
    res = {}
    res["fim_01"] = (dir_dict_np["+"] - dir_dict_np["-"]) / 4
    fim_mean = (dir_dict_np["+"] + dir_dict_np["-"]) / 4
    fim_mean2 = (dir_dict_np["0"] + dir_dict_np["1"]) / 2
    res["fim_err0"] = fim_mean2 - fim_mean
    fim_diff = (dir_dict_np["1"] - dir_dict_np["0"]) / 2
    res["fim_00"] = fim_mean - fim_diff
    res["fim_11"] = fim_mean + fim_diff
    min_eigenvalue = fim_mean - (fim_diff**2 + res["fim_01"]**2)**0.5
    res["fim_err1"] = np.maximum(-min_eigenvalue, 0)
    res["fim_00"] += res["fim_err1"]
    res["fim_11"] += res["fim_err1"]
    res["lambda1"] = (np.arange(resolution - 1) + 0.5) / scaling_resolution
    res["lambda0"] = res["lambda1"]
    return res

