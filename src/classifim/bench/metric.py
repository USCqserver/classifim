# In this file, we define the metrics used to evaluate the quality of
# fidelity susceptibility estimates.

from numpy.ctypeslib import ndpointer
import ctypes
import functools
import itertools
import numpy as np
import os
import scipy.stats
import sys
from classifim_gen.linalg import average_consecutive

def _get_lib():
    extension = {
        "win32": ".dll",
        "darwin": ".dylib"
    }.get(sys.platform, ".so")
    lib_path = os.path.join(
        os.path.dirname(__file__),
        'lib',
        'libclassifim_gen' + extension)
    lib = ctypes.CDLL(lib_path)
    return lib

@functools.lru_cache(maxsize=1)
def _get_metric_lib():
    """
    Same as _get_lib() but with stand-alone functions.
    """
    lib = _get_lib()
    lib.count_inversions.argtypes = [
        ctypes.c_int,
        ndpointer(dtype=ctypes.c_double, ndim=1, flags="C_CONTIGUOUS")]
    lib.count_inversions.restype = ctypes.c_uint64
    return lib

def fim_mgrid_to_grid(fim_mgrid):
    """
    Extracts grid from fim_mgrid
    (as returned by fidelity.meshgrid_transform_2D_fim).
    """
    return [fim_mgrid["lambda0"], fim_mgrid["lambda1"]]

def fim_mgrid_to_metric(fim_mgrid):
    """
    Extracts metric from fim_mgrid
    """
    res = np.array(
        [[fim_mgrid["fim_00"], fim_mgrid["fim_01"]],
         [fim_mgrid["fim_01"], fim_mgrid["fim_11"]]])
    return np.moveaxis(res, (0, 1), (-2, -1))

class StraightLineDistance:
    """
    This class can be initialized by providing a grid of parameters
    and the values of metric tensor for each of the grid nodes.

    The metric is assumed to be a piecewise constant function
    equal to its value at the closest grid node.

    Method 'distance' returns the straight line distance between two
    given points in the parameter space.
    """

    def __init__(self, grid: list, metric: np.ndarray):
        """
        Args:
            grid: a list of len(grid_size) 1D numpy arrays,
                each of shape (grid_size[i], )
                containing the coordinates of the grid nodes.
            metric: a numpy array of shape
                (*grid_size, len(grid_size), len(grid_size)) containing
                the values of the metric tensor at the grid nodes.
        """
        self._grid_size = np.array([len(grid[i]) for i in range(len(grid))])
        self._grid = grid
        self._grid_mid = [
                (grid[i][1:] + grid[i][:-1]) / 2 for i in range(len(grid))]
        self._metric = metric
        assert metric.shape == (
                *self._grid_size, len(self._grid_size), len(self._grid_size))
        self._tie_mask_to_metric = {0: self._metric}

    @classmethod
    def from_fim_mgrid(cls, fim_mgrid):
        """
        Args:
            fim_mgrid: a dictionary containing the keys
                "lambda0", "lambda1", "fim_00", "fim_01", "fim_11"
                as returned by fidelity.meshgrid_transform_2D_fim.

        Returns:
            An instance of StraightLineDistance.
        """
        grid = fim_mgrid_to_grid(fim_mgrid)
        metric = fim_mgrid_to_metric(fim_mgrid)
        return cls(grid, metric)

    @classmethod
    def from_fim_1d(cls, fim_1d):
        """
        Args:
            fim_1d: a dictionary containing the keys
                "lambda0" and "fim_00" (or "fim").

        Returns:
            An instance of StraightLineDistance.
        """
        grid = [np.array(fim_1d["lambda0"])]
        fim_00 = fim_1d.get("fim_00", None)
        if fim_00 is None:
            fim_00 = fim_1d.get("fim")
        fim_00 = np.array(fim_00)
        assert fim_00.ndim == 1
        fim_00 = fim_00.reshape((len(grid[0]), 1, 1))
        return cls(grid, fim_00)

    def get_metric(self, tie_mask):
        """
        Args:
            tie_mask: bool tuple of the same shape as grid_size
                containing True for dim_idx values for which
                the mean between two adjacent grid positions
                should be computed. np.ndarray is OK too.

        Returns:
            A numpy array of shape (*(grid_size - tie_mask), m1, m1),
                where m1 = len(grid_size) - sum(tie_mask)
                containing the values of the metric tensor
                in the directions orthogonal to the tie_mask dimensions.
        """
        tie_mask_int = np.sum(2 ** np.where(tie_mask)[0])
        res = self._tie_mask_to_metric.get(tie_mask_int)
        if res is not None:
            return res
        tie_mask = np.array(tie_mask)
        assert tie_mask.shape == self._grid_size.shape
        assert np.sum(tie_mask) > 0
        # Replace the last True with False in tie_mask:
        prev_tie_mask = tie_mask.copy()
        flip_dim_idx = np.where(tie_mask)[0][-1]
        prev_tie_mask[flip_dim_idx] = False
        prev_metric = self.get_metric(prev_tie_mask)
        # Remove flip_dim_idx from last two dimensions of prev_metric:
        flip_dim_idx_adj = np.sum(1 - tie_mask[:flip_dim_idx])
        prev_metric = np.delete(prev_metric, flip_dim_idx_adj, axis=-1)
        prev_metric = np.delete(prev_metric, flip_dim_idx_adj, axis=-2)
        res = average_consecutive(prev_metric, axis=flip_dim_idx)
        self._tie_mask_to_metric[tie_mask_int] = res
        return res

    def distance(self, p0, p1):
        """
        Args:
            p0: a numpy array of shape len(grid_size) containing the coordinates
                of the first point in the parameter space.
            p1: a numpy array of shape len(grid_size) containing the coordinates
                of the second point in the parameter space.

        Returns:
            A float number representing the straight line distance between p1 and p2.
        """
        p0 = np.array(p0)
        p1 = np.array(p1)
        if np.all(p0 == p1):
            return 0.0
        assert p0.shape == self._grid_size.shape
        assert p1.shape == self._grid_size.shape
        # Let p(t) = p0 + t * (p1 - p0) for t in [0, 1].
        # We want to find all intersections of p(t) with the grid_mid hyperplanes.
        # I.e. the goal is to initialize the following:
        # t_int = [0, *intersection_times, 1]
        # dim_idx_int: a list of len len(t_int) - 2.
        #     dim_idx_int[j] is the dim_idx of the grid_mid hyperplane
        #     with which p(t) intersects at t = t_int[j + 1].
        # grid_idx: a list of len(grid_size) numpy arrays
        #     of shape (len(t_int) - 1, ) containing the indices of the grid nodes
        #     of the cell in which p(t) is for t in (t_int[i], t_int[i + 1]).
        t_ints = [[0]]
        dim_idx_ints = []
        grid_idx_0 = [] # starting grid cell
        grid_idx_1 = [] # ending grid cell
        dp = p1 - p0
        dp_sign = np.sign(dp)
        for dim_idx in range(len(self._grid_mid)):
            cur_dp_sign = dp_sign[dim_idx]
            cur_grid_idx_0 = np.searchsorted(
                self._grid_mid[dim_idx],
                p0[dim_idx],
                side=('right' if cur_dp_sign > 0 else 'left'))
            grid_idx_0.append(cur_grid_idx_0)
            cur_grid_idx_1 = np.searchsorted(
                self._grid_mid[dim_idx],
                p1[dim_idx],
                side=('left' if cur_dp_sign > 0 else 'right'))
            grid_idx_1.append(cur_grid_idx_1)
            if dp[dim_idx] == 0:
                continue
            if cur_grid_idx_0 == cur_grid_idx_1:
                continue
            int_coords = self._grid_mid[dim_idx][
                min(cur_grid_idx_0, cur_grid_idx_1):max(cur_grid_idx_0, cur_grid_idx_1)]
            if cur_dp_sign < 0:
                int_coords = int_coords[::-1]
            cur_t_int = (int_coords - p0[dim_idx]) / dp[dim_idx]
            assert len(cur_t_int) > 0
            t_ints.append(cur_t_int)
            dim_idx_ints.append([dim_idx] * len(cur_t_int))

        t_ints.append([1])
        t_int = np.concatenate(t_ints)
        if len(dim_idx_ints) == 0:
            dim_idx_int = np.zeros(shape=(0, ), dtype=int)
        else:
            dim_idx_int = np.concatenate(dim_idx_ints)
            ii = np.argsort(t_int)
            t_int = t_int[ii]
            dim_idx_int = dim_idx_int[ii[1:-1]-1]
        # dim_idx_int should be integer-like:
        if dim_idx_int.dtype.kind not in 'iu':
            print(f"{dim_idx_int.dtype=} {len(dim_idx_ints)=}")
            for i in range(len(dim_idx_ints)):
                print(f"dim_idx_ints[{i}] = {dim_idx_ints[i]}")
            raise ValueError("dim_idx_int should be integer-like")
        grid_idx_0 = np.array(grid_idx_0)
        grid_idx_1 = np.array(grid_idx_1)

        tie_mask = (dp_sign == 0) & (grid_idx_0 != grid_idx_1)
        dim_idx_map = np.concatenate([[0], np.cumsum(1 - tie_mask[:-1])])
        metric = self.get_metric(tie_mask)
        dp_adj = dp[~tie_mask]

        cur_grid_idx = np.array(grid_idx_0)
        res = 0.0
        for j in range(len(t_int) - 1):
            if j > 0:
                try:
                    cur_grid_idx[dim_idx_int[j - 1]] += dp_sign[dim_idx_int[j - 1]]
                except IndexError:
                    print(f"{p0=}, {p1=}, {j=}, {cur_grid_idx.shape=}, "
                        + f"{dim_idx_int.shape=}, {dim_idx_int.dtype=}, "
                        + f"{dp_sign.shape=}")
                    raise
            dt = t_int[j + 1] - t_int[j]
            cur_dp_adj = dt * dp_adj
            cur_metric = metric[tuple(cur_grid_idx)]
            assert cur_metric.shape == (len(cur_dp_adj), len(cur_dp_adj))
            cur_dist2 = cur_dp_adj @ cur_metric @ cur_dp_adj
            scale = np.mean(np.abs(np.diag(cur_metric)))
            assert cur_dist2 >= -1e-12 * scale, (
                f"{cur_dist2=}, {scale=}, {cur_metric=}, {cur_grid_idx=}, {tie_mask=}.")
            res += np.sqrt(max(cur_dist2, 0))

        return res

    def __call__(self, p0, p1):
        return self.distance(p0, p1)

class StraightLineDistanceCpp:
    """
    Drop-in replacement for StraightLineDistance implemented in C++.
    """

    def __init__(self, grid: list, metric: np.ndarray):
        space_dim = len(grid)
        self._space_dim = space_dim
        assert all(isinstance(g, np.ndarray) for g in grid)
        assert all(g.ndim == 1 for g in grid)
        grid_sizes = np.ascontiguousarray([len(g) for g in grid], dtype=ctypes.c_int)
        cpp_grid = np.ascontiguousarray(np.concatenate(grid), dtype=np.float64)
        assert cpp_grid.shape == (np.sum(grid_sizes), )
        assert metric.shape == tuple(grid_sizes) + (space_dim, space_dim)
        metric = np.ascontiguousarray(metric, dtype=np.float64)
        # Assign None first in case of exception:
        self._lib = None
        self._lib = self._get_lib()
        _straight_line_distance = self._lib.create_straight_line_distance(
            space_dim,
            grid_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            cpp_grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            metric.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self._straight_line_distance = ctypes.c_void_p(_straight_line_distance)

    @staticmethod
    def _get_lib():
        lib = _get_lib()

        # create_straight_line_distance
        lib.create_straight_line_distance.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.create_straight_line_distance.restype = ctypes.c_void_p

        # delete_straight_line_distance
        lib.delete_straight_line_distance.argtypes = [ctypes.c_void_p]
        lib.delete_straight_line_distance.restype = None

        # distances
        lib.straight_line_distance_distances.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.straight_line_distance_distances.restype = None
        # TODO:2: add other wrappers here

        return lib

    @classmethod
    def from_fim_mgrid(cls, fim_mgrid):
        return StraightLineDistance.from_fim_mgrid.__func__(cls, fim_mgrid)

    @classmethod
    def from_fim_1d(cls, fim_1d):
        return StraightLineDistance.from_fim_1d.__func__(cls, fim_1d)

    def distances(self, points):
        """
        Compute the distance between pairs of points.

        Args:
            points: an np.ndarray of shape (num_points, 2, space_dim) with
                pairs of points to measure the distance between.

        Returns:
            An np.ndarray of shape (num_points,) with distances.
        """
        # Ensure points are contiguous in memory:
        points = np.ascontiguousarray(points, dtype=np.float64)
        num_points = points.shape[0]
        assert points.shape == (num_points, 2, self._space_dim), (
            f"{points.shape = } != ({num_points}, 2, {self._space_dim}).")
        # Create empty result array:
        results = np.empty(points.shape[0], dtype=np.float64)
        # Call the C++ method:
        self._lib.straight_line_distance_distances(
            self._straight_line_distance,
            num_points,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return results

    def distance(self, p0, p1):
        return self.distances(np.array([[p0, p1]]))[0]

    def __call__(self, p0, p1):
        return self.distance(p0, p1)

    def __del__(self):
        if self._lib is not None:
            self._lib.delete_straight_line_distance(self._straight_line_distance)

class StraightLineDistance1DCpp:
    def __init__(self, grid, metric):
        if isinstance(grid, list):
            assert len(grid) == 1
            grid = grid[0]
        assert grid.ndim == 1
        grid = np.ascontiguousarray(grid, dtype=np.float64)
        if metric.ndim == 3:
            assert metric.shape[1:] == (1, 1)
            metric = metric[:, 0, 0]
        assert metric.ndim == 1
        metric = np.ascontiguousarray(metric, dtype=np.float64)
        self._lib = None
        self._lib = self._get_lib()
        _impl = self._lib.create_straight_line_distance1d(
            len(grid),
            grid.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            metric.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self._impl = ctypes.c_void_p(_impl)

    @staticmethod
    def _get_lib():
        lib = _get_lib()

        # create_straight_line_distance1d
        lib.create_straight_line_distance1d.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]
        lib.create_straight_line_distance1d.restype = ctypes.c_void_p

        # delete_straight_line_distance1d
        lib.delete_straight_line_distance1d.argtypes = [ctypes.c_void_p]
        lib.delete_straight_line_distance1d.restype = None

        # distances
        lib.straight_line_distance1d_distances.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        return lib

    @classmethod
    def from_fim_1d(cls, fim_1d):
        return StraightLineDistance.from_fim_1d.__func__(cls, fim_1d)

    def distances(self, points):
        """
        Compute the distance between pairs of points.

        Args:
            points: an np.ndarray of shape (num_points, 2) with
                pairs of points to measure the distance between.

        Returns:
            An np.ndarray of shape (num_points,) with distances.
        """
        # Ensure points are contiguous in memory:
        points = np.ascontiguousarray(points, dtype=np.float64)
        num_points = points.shape[0]
        assert points.shape == (num_points, 2), (
            f"{points.shape = } != ({num_points}, 2).")
        # Create empty result array:
        results = np.empty(points.shape[0], dtype=np.float64)
        # Call the C++ method:
        self._lib.straight_line_distance1d_distances(
            self._impl,
            num_points,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return results

    def __del__(self):
        if self._lib is not None:
            self._lib.delete_straight_line_distance1d(self._impl)

def count_inversions(arr) -> int:
    """
    Count the number of inversions of an array.

    That is, the number of pairs (i, j) such that i < j and arr[i] > arr[j].

    This implementation is based on the merge sort algorithm.

    Args:
        arr: a list of numbers.
    Returns:
        The number of inversions of arr.
    """
    arr = list(arr)
    def merge_and_count(a, b):
        count = 0
        output = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                output.append(a[i])
                i += 1
            else:
                output.append(b[j])
                count += len(a) - i
                j += 1
        output += a[i:]
        output += b[j:]
        return output, count

    def sort_and_count(arr):
        if len(arr) <= 1:
            return arr, 0
        half = len(arr) // 2
        left, inversions_left = sort_and_count(arr[:half])
        right, inversions_right = sort_and_count(arr[half:])
        merged, inversions_merge = merge_and_count(left, right)
        return merged, inversions_left + inversions_right + inversions_merge

    _, count = sort_and_count(arr)
    return count

def count_inversions_cpp(arr) -> int:
    arr_copy = np.ascontiguousarray(arr, dtype=np.float64)
    if arr_copy is arr:
        arr_copy = arr.copy()
    lib = _get_metric_lib()
    try:
        return lib.count_inversions(len(arr), arr_copy)
    except ctypes.ArgumentError:
        print(f"Note: {type(arr_copy)=}.", file=sys.stderr)
        raise

def compute_ranking_error(
        a0: np.ndarray, a1: np.ndarray, use_cpp=False) -> float:
    """
    Computes ranking error between 2 np.arrays.

    Ranking error is approximately equal to the probability that
    (a0[i] > a0[j]) != (a1[i] > a1[j]) for a random pair of distinct
    indices i, j.

    More precisely, it is equal to the average over all pairs (i, j)
    with i != j of the output of the following procedure:
    1. Let c0 = (a0[i] ? a0[j]), i.e. -1 if the sign is "<", 0 if "=",
        and 1 if ">". Same for c1 and a1.
    2. If c0 == 0 or c1 == 0, return 0.5.
    3. Return 1 if c0 != c1 and 0 otherwise.

    This function implements a more optimal algorithm equivalent to the above.

    Args:
        a0: a numpy array of shape (n, ).
        a1: a numpy array of shape (n, ).
        use_cpp: whether to use the C++ implementation.
    """
    assert a0.shape == a1.shape
    num_items = a0.shape[0]
    max_inversions = num_items * (num_items - 1) // 2
    count_inversions_f = count_inversions_cpp if use_cpp else count_inversions
    def inversions_lower(a0: np.ndarray, a1: np.ndarray) -> int:
        ii = np.lexsort((a1, a0))
        a1 = a1[ii]
        return count_inversions_f(a1)

    # Add lower bound and upper bound to resolve the ties in "average" way.
    total_inversions = (
        inversions_lower(a0, a1) + (max_inversions - inversions_lower(-a0, a1)))
    return total_inversions / (2 * max_inversions)

def normal_summary(a, digits=4):
    a_mean = np.mean(a)
    a_err = np.std(a)
    return f"{a_mean:.{digits}f} \pm {a_err:.{digits}f}"

def lambda_pairs_nd(
        resolution=64, num_pairs=83866, space_dim=2, seed=0, lambda_max=None):
    """
    Generates np.ndarray with pairs of lambda values for N-dimensional space.

    Args:
        resolution: resolution of the grid of lambdas.
        num_pairs: maximal number of pairs to generate.
        space_dim: number of dimensions of the space.
        seed: random seed for sampling.
        lambda_max: maximal value of lambda to sample.
            Defaults to (resolution - 1) / resolution.

    Returns:
        Pair of np.ndarrays of shape (num_pairs, space_dim) with lambda values.
        I.e. the overall shape is (2, num_pairs, space_dim).
    """
    # Generate an N-dimensional grid
    if lambda_max is None:
        lambda_max = (resolution - 1) / resolution
    ranges = [
        np.linspace(0, lambda_max, resolution)
        for _ in range(space_dim)]
    grid = np.meshgrid(*ranges, indexing='ij')
    lambdas = np.stack(grid, axis=-1).reshape(-1, space_dim)

    # Compute all possible pairs
    num_lambdas = lambdas.shape[0]
    assert num_lambdas == resolution ** space_dim
    num_all_pairs = num_lambdas * (num_lambdas - 1) // 2
    if num_pairs < num_all_pairs:
        rng = np.random.default_rng(seed)
        lambda_pair_ii = rng.choice(
            num_all_pairs, size=num_pairs, replace=False)
    else:
        lambda_pair_ii = np.arange(num_all_pairs)
    lambda_a_ii = (0.5 + np.sqrt(2 * lambda_pair_ii + 0.25)).astype(int)
    lambda_b_ii = lambda_pair_ii - lambda_a_ii * (lambda_a_ii - 1) // 2
    assert np.all(lambda_b_ii < lambda_a_ii)
    assert np.all(lambda_b_ii >= 0)
    return lambdas[lambda_a_ii], lambdas[lambda_b_ii]

def mse_perfect_scale(predicted, target):
    """
    Computes MSE between predicted and target, scaling predicted to minimize it.
    """
    predicted = np.array(predicted)
    target = np.array(target)
    assert predicted.shape == target.shape
    scale = (predicted @ target) / np.sum(predicted**2)
    return scale, np.mean((target - scale * predicted) ** 2)

def _finalize_distance_errors(distances, lambda_pairs, use_cpp):
    """
    Finalizes the results of compute_distance_errors.

    Args:
        distances: np.ndarray of shape (2, num_pairs) with SL distances.
        lambda_pairs: A pair of np.ndarrays of shape (num_pairs, space_dim).

    Returns:
        Dictionary with error metrics.
    """
    num_pairs, space_dim = lambda_pairs[0].shape
    assert distances.shape == (2, num_pairs)
    distSL_a = distances[0]
    distSL_b = distances[1]

    distSL_const = np.linalg.norm(lambda_pairs[0] - lambda_pairs[1], axis=1)
    _, distMSE_const = mse_perfect_scale(distSL_const, distSL_b)
    scale, mse = mse_perfect_scale(distSL_a, distSL_b)
    res = {
        "num_pairs": num_pairs,
        "lambda_pairs": lambda_pairs,
        "distances": distances,
        "distMSE": np.mean((distances[0] - distances[1]) ** 2),
        "distRE": compute_ranking_error(
            distances[0], distances[1], use_cpp=use_cpp),
        "distMSE_const": distMSE_const,
        "distRE_const": compute_ranking_error(
            distSL_const, distSL_b, use_cpp=use_cpp),
        "distRE_perfect": compute_ranking_error(
            distSL_b, distSL_b, use_cpp=use_cpp),
        "distMSE_perfect_scale": mse,
        "space_dim": space_dim,
        "scale": scale}
    return res

def compute_distance_errors(
        fim_mgrid_a, fim_mgrid_b, grid_resolution=64, num_pairs=83866,
        seed=0, use_cpp=False):
    """
    Computes distMSE and distRE between two metrics.

    Args:
        fim_mgrid_a, fim_mgrid_b: two fim_mgrid's two compare.
            Both should be compatible with StraightLineDistance.from_fim_mgrid.
        grid_resolution: resolution of 2D lambda grid.
        num_pairs: maximal number of pairs to sample.

    Returns:
        Dictionary with error metrics.
    """
    lambda_pairs = lambda_pairs_nd(
        resolution=grid_resolution, num_pairs=num_pairs, seed=seed, space_dim=2)
    if use_cpp:
        # lambda_pairs_nd returns the shape (2, num_pairs, space_dim)
        # but StraightLineDistanceCpp expects the shape
        # (num_pairs, 2, space_dim).
        lambda_pairs_cpp = np.array(lambda_pairs)
        lambda_pairs_cpp = np.moveaxis(lambda_pairs_cpp, 0, 1)
        assert lambda_pairs_cpp.shape == (num_pairs, 2, 2)
        distance_fs = (
            StraightLineDistanceCpp.from_fim_mgrid(g)
            for g in (fim_mgrid_a, fim_mgrid_b))
        distances = np.array([
            f.distances(lambda_pairs_cpp)
            for f in distance_fs])
    else:
        distance_fs = (
            StraightLineDistance.from_fim_mgrid(g)
            for g in (fim_mgrid_a, fim_mgrid_b))
        distances = np.array([
            [f(*lambdas) for lambdas in zip(*lambda_pairs)]
            for f in distance_fs])
    return _finalize_distance_errors(distances, lambda_pairs, use_cpp=use_cpp)

def compute_distance_errors_1d(
        fim_a, fim_b, num_pairs=83866, seed=0, cpp_version=None,
        lambda_min=None):
    """
    Computes distMSE and distRE between two metrics.

    Args:
        fim_a, fim_b: two fim's two compare.
            Both should be compatible with StraightLineDistance.from_fim_mgrid.
        grid_resolution: resolution of 2D lambda grid.
        num_pairs: maximal number of pairs to sample.
        seed: random seed.
        cpp_version: None, "cpp", or "cpp1d". End with "-" to avoid C++ in
            the computation of the ranking error.

    Returns:
        Dictionary with error metrics.
    """
    assert "lambda1" not in fim_a and "lambda1" not in fim_b
    lambda0s = np.array(fim_b["lambda0"])
    assert np.all(lambda0s[1:] > lambda0s[:-1])
    lambda_max = lambda0s[-1] * 1.5 - lambda0s[-2] * 0.5
    if lambda_min is None:
        lambda_min = lambda0s[0] * 1.5 - lambda0s[1] * 0.5
        assert np.abs(lambda_min) <= 2.0 * 2**(-23)
    else:
        assert lambda_min <= lambda0s[0]
        assert lambda_min >= 2 * lambda0s[0] - lambda0s[1]
    grid_resolution = len(lambda0s) + 1
    lambda_pairs = lambda_pairs_nd(
        resolution=grid_resolution, num_pairs=num_pairs, seed=seed, space_dim=1,
        lambda_max=lambda_max)
    if cpp_version is None:
        distance_fs = (
            StraightLineDistance.from_fim_1d(g)
            for g in (fim_a, fim_b))
        distances = np.array([
            [f(*lambdas) for lambdas in zip(*lambda_pairs)]
            for f in distance_fs])
        use_cpp = False
    else:
        if cpp_version.endswith("-"):
            cpp_version = cpp_version[:-1]
            use_cpp = False
        else:
            use_cpp = True
        assert cpp_version in ("cpp", "cpp1d")
        lambda_pairs_cpp = np.moveaxis(np.array(lambda_pairs), 0, 1)
        assert lambda_pairs_cpp.shape == (num_pairs, 2, 1), f"{lambda_pairs_cpp.shape=}"
        if cpp_version == "cpp1d":
            lambda_pairs_cpp = lambda_pairs_cpp.reshape((num_pairs, 2))
            cls = StraightLineDistance1DCpp
        else:
            cls = StraightLineDistanceCpp
        distance_fs = (cls.from_fim_1d(g) for g in (fim_a, fim_b))
        distances = np.array([
            f.distances(lambda_pairs_cpp)
            for f in distance_fs])
    return _finalize_distance_errors(distances, lambda_pairs, use_cpp=use_cpp)
