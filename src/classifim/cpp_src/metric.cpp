#include "metric.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xview.hpp>

#ifdef NDEBUG
// https://stackoverflow.com/a/985807
// http://web.archive.org/web/20201129200055/http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
#define CLASSIFIM_ASSERT(x) do { (void)sizeof(x); } while (0)
#else
#include <cassert>
#define CLASSIFIM_ASSERT(x) assert(x)
#endif

// We could use xtensor-blas for dot product,
// but for now its not worth the dependency.
// #include <xtensor-blas/xlinalg.hpp>

namespace classifim {
namespace {
// Returns the index of the least significant 1 bit in v.
// If v is 0, returns -1.
int find_index_of_one(std::uint64_t v) {
  // http://supertech.csail.mit.edu/papers/debruijn.pdf
  // https://stackoverflow.com/questions/757059/position-of-least-significant-bit-that-is-set
  // Alternatively, we could use __builtin_ctzll or ffsll,
  // but I'm not sure if those are portable.
  constexpr std::uint64_t debruijn64_m = 0x03f79d71b4cb0a89ULL;
  constexpr std::uint64_t debruijn64[64] = {
      0,  1,  3,  7,  15, 31, 63, 62, 61, 59, 55, 47, 30, 60, 57, 51,
      39, 14, 29, 58, 53, 43, 23, 46, 28, 56, 49, 35, 6,  13, 27, 54,
      45, 26, 52, 41, 19, 38, 12, 25, 50, 37, 11, 22, 44, 24, 48, 33,
      2,  5,  10, 21, 42, 20, 40, 17, 34, 4,  9,  18, 36, 8,  16, 32};
  return debruijn64[((v & -v) * debruijn64_m) >> 58] - (v == 0);
}

// Utility function for _init_metric. Equivalent to the following numpy code:
// def _average_consecutive(a, axis):
//     a = np.moveaxis(a, axis, 0)
//     a = (a[1:] + a[:-1]) / 2
//     return np.moveaxis(a, 0, axis)
//
// def _remove_axis_from_metric(prev_metric, drop_dim_idx, flip_dim_idx):
//      res = np.delete(prev_metric, drop_dim_idx, axis=-1)
//      res = np.delete(res, drop_dim_idx, axis=-2)
//      return _average_consecutive(res, axis=flip_dim_idx)
void _remove_axis_from_metric(const xt::xarray<double> &prev_metric,
                              std::size_t drop_dim_idx,
                              std::size_t flip_dim_idx,
                              xt::xarray<double> &new_metric) {
  std::size_t space_dim = prev_metric.dimension() - 2;
  xt::xdynamic_slice_vector slices_l, slices_r;
  slices_l.reserve(space_dim + 2);
  slices_r.reserve(space_dim + 2);
  for (std::size_t i = 0; i < flip_dim_idx; ++i) {
    slices_l.push_back(xt::all());
    slices_r.push_back(xt::all());
  }
  slices_l.push_back(xt::range(0, prev_metric.shape()[drop_dim_idx] - 1));
  slices_r.push_back(xt::range(1, prev_metric.shape()[drop_dim_idx]));
  for (std::size_t i = flip_dim_idx + 1; i < space_dim; ++i) {
    slices_l.push_back(xt::all());
    slices_r.push_back(xt::all());
  }
  slices_l.push_back(xt::drop(drop_dim_idx));
  slices_l.push_back(xt::drop(drop_dim_idx));
  slices_r.push_back(xt::drop(drop_dim_idx));
  slices_r.push_back(xt::drop(drop_dim_idx));
  CLASSIFIM_ASSERT(slices_l.size() == space_dim + 2);
  CLASSIFIM_ASSERT(slices_r.size() == space_dim + 2);
  auto metric_l = xt::dynamic_view(prev_metric, slices_l);
  auto metric_r = xt::dynamic_view(prev_metric, slices_r);
  new_metric = (metric_l + metric_r) / 2.0;
}

// Utility function to print shape (for debugging):
template <typename ShapeType> std::string shape_to_str(const ShapeType &shape) {
  std::ostringstream oss;
  oss << "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << ")";
  return oss.str();
}

//  Count the number of inversions of an array.
//  That is, the number of pairs (i, j) such that i < j and arr[i] > arr[j].
//  This implementation is based on the merge sort algorithm.
class InversionCounter {
public:
  static std::uint64_t count_inversions(std::span<double> arr) {
    std::vector<double> temp(arr.size());
    return sort_and_count(arr, temp, 0, arr.size());
  }
private:
  static std::uint64_t merge_and_count(std::span<double> arr, std::vector<double>& temp,
      int left, int mid, int right) {
    // merge arr[left:mid] and arr[mid:right] into temp[left:right]
    // then copy back to arr[left:right].
    int i = left, j = mid, k = left;
    std::uint64_t inversions = 0;
    while ((i < mid) && (j < right)) {
      if (arr[i] <= arr[j]) {
        temp[k++] = arr[i++];
      } else {
        temp[k++] = arr[j++];
        inversions += (mid - i);
      }
    }
    while (i < mid) {
      temp[k++] = arr[i++];
    }
    // [j, right) is already in place.
    for (i = left; i < k; ++i) {
      arr[i] = temp[i];
    }
    return inversions;
  }
  static std::uint64_t sort_and_count(std::span<double> arr, std::vector<double>& temp,
      int left, int right) {
    // Sort arr[left:right] and return the number of inversions.
    // If right - left <= 1, then arr[left:right] is already sorted.
    // Otherwise, we split arr[left:right] into two halves,
    // sort each half, and merge them.
    if (right - left <= 1) {
      return 0;
    }
    int mid = (left + right) / 2;
    std::uint64_t inversions = 0;
    inversions += sort_and_count(arr, temp, left, mid);
    inversions += sort_and_count(arr, temp, mid, right);
    inversions += merge_and_count(arr, temp, left, mid, right);
    return inversions;
  }
};
} // namespace

StraightLineDistance::StraightLineDistance(
    std::span<std::span<double>> grid_spans, std::span<double> metric_span)
    : space_dim(grid_spans.size()) {

  if (space_dim < 1 || space_dim > 64) {
    throw std::runtime_error("Invalid number of grid dimensions");
  }

  for (const auto &span : grid_spans) {
    _grid.emplace_back(span.begin(), span.end());
  }

  std::vector<std::size_t> metric_shape;
  for (const auto &g : _grid) {
    metric_shape.push_back(g.size());
  }
  metric_shape.push_back(space_dim);
  metric_shape.push_back(space_dim);

  auto xt_metric_span =
      xt::adapt(metric_span.data(), metric_span.size(), xt::no_ownership(),
                metric_shape, xt::layout_type::row_major);
  _tie_mask_to_metric.emplace(std::piecewise_construct,
                              std::forward_as_tuple(0ULL),
                              std::forward_as_tuple(xt_metric_span));

  _init();
}

// Utility function for _get_metric.
// Helps in recursive computation of _tie_mask_to_metric[tie_mask].
// We assume _tie_mask_to_metric[0] exists.
void StraightLineDistance::_init_metric(std::uint64_t tie_mask,
                                        xt::xarray<double> &new_metric) {
  CLASSIFIM_ASSERT(tie_mask != 0);
  int flip_dim_idx = find_index_of_one(tie_mask);
  CLASSIFIM_ASSERT(flip_dim_idx >= 0);
  CLASSIFIM_ASSERT(flip_dim_idx < space_dim);
  // Same as prev_tie_mask = tie_mask ^ (1ULL << flip_dim_idx);
  std::uint64_t prev_tie_mask = tie_mask & (tie_mask - 1);
  // This is a recursion. It will terminate because on each iteration
  // the number of set bits in tie_mask is decreased by 1.
  const auto &prev_metric = _get_metric(prev_tie_mask);
  // Note: by construction, the index we need to remove in
  // the last two dimensions of the metric is flip_dim_idx
  // --- the same as the index of the dimension we need to average over.
  _remove_axis_from_metric(prev_metric, flip_dim_idx, flip_dim_idx, new_metric);
  assert(new_metric.dimension() == static_cast<unsigned>(space_dim) + 2);
  assert(new_metric.shape()[space_dim] ==
      static_cast<unsigned>(space_dim) - std::popcount(tie_mask));
  assert(new_metric.shape()[space_dim + 1] ==
      static_cast<unsigned>(space_dim) - std::popcount(tie_mask));
}

void StraightLineDistance::_init() {
  // Init _grid_mid:
  std::size_t grid_mid_size = 0;
  for (auto &grid_axis : _grid) {
    if (grid_axis.size() <= 1) {
      throw std::runtime_error("Grid axis must have at least 1 point");
    }
    grid_mid_size += grid_axis.size() + 1;
  }
  _grid_mid_data = std::make_unique<double[]>(grid_mid_size);
  _grid_mid.reserve(space_dim);
  double *grid_mid_ptr = _grid_mid_data.get();
  for (auto &grid_axis : _grid) {
    std::span<double> &cur_grid_mid_span = _grid_mid.emplace_back(
        grid_mid_ptr, grid_mid_ptr + grid_axis.size() + 1);
    *grid_mid_ptr = -std::numeric_limits<double>::infinity();
    ++grid_mid_ptr;
    for (auto grid_it = grid_axis.begin() + 1; grid_it < grid_axis.end();
         ++grid_it, ++grid_mid_ptr) {
      *grid_mid_ptr = (*grid_it + *(grid_it - 1)) / 2.0;
    }
    *grid_mid_ptr = std::numeric_limits<double>::infinity();
    ++grid_mid_ptr;
    CLASSIFIM_ASSERT(std::span<double>::iterator(grid_mid_ptr)
        == cur_grid_mid_span.end());
  }
  assert(grid_mid_ptr == _grid_mid_data.get() + grid_mid_size);

  // Allocate scratch space for _compute_distance:
  _scratch_d.resize(space_dim * 3);
  _scratch_i.resize(space_dim * 1);
}

std::vector<double>
StraightLineDistance::distances(int num_point_pairs,
                                std::span<double> point_span) {
  std::vector<double> results(num_point_pairs);
  size_t step = 2 * space_dim;
  size_t i = 0;
  for (auto point0 = point_span.begin(), point1 = point0 + space_dim;
       point0 < point_span.end(); point0 += step, point1 += step, ++i) {
    results[i] = _compute_distance(std::span<double>(point0, space_dim),
                                   std::span<double>(point1, space_dim));
  }
  return results;
}

// This is C++ implementation of distance method in metric.py.
// This is not thread safe due to the use of _scratch_d and _scratch_i.
double StraightLineDistance::_compute_distance(std::span<double> point0,
                                               std::span<double> point1) {
  assert(_tie_mask_to_metric[0ULL].shape(
        _tie_mask_to_metric[0ULL].dimension() - 1)
      == static_cast<unsigned>(space_dim));
  assert(_tie_mask_to_metric[0ULL].shape(
        _tie_mask_to_metric[0ULL].dimension() - 2)
      == static_cast<unsigned>(space_dim));
  assert(point0.size() == space_dim);
  assert(point1.size() == space_dim);
  if (std::memcmp(point0.data(), point1.data(), space_dim * sizeof(double)) ==
      0) {
    return 0.0;
  }
  std::uint64_t tie_mask = 0;
  std::span<double> dpoint(_scratch_d.data(), space_dim);
  // Let p(t) = p0 + t * (p1 - p0) for t in [0, 1].
  // dimj_to_tint[i] is the next t for which p(t) intersects
  // with _grid_mid[i] lines.
  std::span<double> dimj_to_tint(_scratch_d.data() + space_dim, space_dim);
  std::span<double> dpoint_nontie(_scratch_d.data() + 2 * space_dim, space_dim);
  std::span<int> cur_cell_idx(_scratch_i.data(), space_dim);
  int num_nontie_dims = 0;
  for (int i = 0; i < space_dim; ++i) {
    dpoint[i] = point1[i] - point0[i];
    // compute j s.t. _grid_mid[i][j] < point0[i] <= _grid_mid[i][j + 1]:
    auto it = std::lower_bound(_grid_mid[i].begin() + 1, _grid_mid[i].end() - 1,
                               point0[i]);
    int cur_cell_idx_i = std::distance(_grid_mid[i].begin(), it) - 1;
    bool is_tie = (point0[i] == _grid_mid[i][cur_cell_idx_i + 1]);
    if (is_tie && dpoint[i] == 0) {
      tie_mask |= (1ULL << i);
    } else {
      dpoint_nontie[num_nontie_dims] = dpoint[i];
      ++num_nontie_dims;
    }
    if (is_tie) {
      // If are on the grid line and moving right, then we'll immediately
      // enter the next cell.
      cur_cell_idx_i += int(dpoint[i] > 0.0);
    }
    if (dpoint[i] == 0.0) {
      // We'll never cross the grid line again:
      dimj_to_tint[i] = std::numeric_limits<double>::infinity();
    } else {
      // We are between cur_cell_idx_i and cur_cell_idx_i + 1
      // indices in _grid_mid[i] array.
      int next_grid_mid_idx = cur_cell_idx_i + int(dpoint[i] > 0.0);
      double next_grid_mid = _grid_mid[i][next_grid_mid_idx];
      dimj_to_tint[i] = (next_grid_mid - point0[i]) / dpoint[i];
    }
    cur_cell_idx[i] = cur_cell_idx_i;
  }
  dpoint_nontie = dpoint_nontie.subspan(0, num_nontie_dims);
  const auto &metric = _get_metric(tie_mask);
  double distance = 0.0;
  double t = 0.0;
  // This is `while (t < 1.0)` but we break below instead.
  while (true) {
    auto it = std::min_element(dimj_to_tint.begin(), dimj_to_tint.end());
    int tint_argmin = std::distance(dimj_to_tint.begin(), it);
    double tnext = dimj_to_tint[tint_argmin];
    double dt = std::min(1.0, tnext) - t;
    distance += dt * _metric_dist(metric, cur_cell_idx, dpoint_nontie);
    if (tnext >= 1.0) {
      break;
    }
    t = tnext;
    // +1 if we are moving right, -1 if left:
    int index_step = int(dpoint[tint_argmin] > 0.0) * 2 - 1;
    cur_cell_idx[tint_argmin] += index_step;
    // Cell #j is bounded by _grid_mid[i][j] and _grid_mid[i][j + 1]:
    int next_grid_mid_idx =
        cur_cell_idx[tint_argmin] + int(dpoint[tint_argmin] > 0.0);

    // The following asserts should pass because otherwise the previous
    // next_grid_mid_idx would be on the boundary of _grid_mid[tint_argmin]
    // and tnext would be +infinity contradicting tnext < 1.0.
    assert(next_grid_mid_idx >= 0);
    assert(next_grid_mid_idx < _grid_mid[tint_argmin].size());

    double next_grid_mid = _grid_mid[tint_argmin][next_grid_mid_idx];
    dimj_to_tint[tint_argmin] =
        (next_grid_mid - point0[tint_argmin]) / dpoint[tint_argmin];
  }
  return distance;
}

double StraightLineDistance::_metric_dist(const xt::xarray<double> &metric,
                                          std::span<int> cur_cell_idx,
                                          std::span<double> vec) {
  assert(cur_cell_idx.size() == space_dim);
  assert(metric.dimension() == space_dim + 2);
  xt::xstrided_slice_vector sv;
  for (int idx : cur_cell_idx) {
    sv.push_back(idx);
  }
  sv.push_back(xt::all());
  sv.push_back(xt::all());
  auto g_view = xt::strided_view(metric, sv);
  std::size_t cur_space_size = vec.size();

  // Assert that g_view.shape is (cur_space_size, cur_space_size):
  assert(g_view.dimension() == 2);
  assert(g_view.shape()[0] == cur_space_size);
  assert(g_view.shape()[1] == cur_space_size);

  double dist2 = 0;
  for (std::size_t i = 0; i < cur_space_size; ++i) {
    for (std::size_t j = 0; j < cur_space_size; ++j) {
      dist2 += vec[i] * g_view(i, j) * vec[j];
    }
  }
  return std::sqrt(dist2);
}

StraightLineDistance1D::StraightLineDistance1D(std::span<double> grid_span,
                                               std::span<double> metric_span)
    : _grid(grid_span.begin(), grid_span.end()) {
  if (_grid.size() < 1) {
    throw std::runtime_error("Grid must have at least 1 point");
  }
  if (metric_span.size() != _grid.size()) {
    throw std::runtime_error("Metric must have the same size as grid");
  }
  _metric_sqrt.resize(_grid.size());
  for (std::size_t i = 0; i < _grid.size(); ++i) {
    double metric_val = metric_span[i];
    if (metric_val < 0.0) {
      throw std::runtime_error("Metric must be non-negative");
    }
    _metric_sqrt[i] = std::sqrt(metric_val);
  }
  _precompute();
}

void StraightLineDistance1D::_precompute() {
  std::size_t imax = _grid.size() - 1;
  _grid_mid.resize(imax + 2);
  _grid_mid[0] = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i <= imax; ++i) {
    if (_grid[i - 1] >= _grid[i]) {
      throw std::runtime_error("Grid points must be strictly increasing");
    }
    _grid_mid[i] = (_grid[i - 1] + _grid[i]) / 2.0;
  }
  _grid_mid[imax + 1] = std::numeric_limits<double>::infinity();

  _dp_lengths_scale = 0.0;
  std::vector<double> small_lengths(imax);
  // small_lengths[0] is unused.
  for (std::size_t i = 1; i < imax; ++i) {
    double cur_dist = (_grid_mid[i + 1] - _grid_mid[i]) * _metric_sqrt[i];
    small_lengths[i] = cur_dist;
    _dp_lengths_scale += cur_dist;
  }
  // Scaling to convert double to std::uint64_t:
  _dp_lengths_scale *=
      (1.0 + std::numeric_limits<double>::epsilon() * imax)
      / static_cast<double>(
          std::numeric_limits<std::uint64_t>::max() - imax);
  _dp_lengths.resize(imax);
  std::uint64_t cur_dp_length = 0;
  _dp_lengths[0] = 0;
  for (std::size_t i = 1; i < imax; ++i) {
    cur_dp_length += std::uint64_t(small_lengths[i] / _dp_lengths_scale + 0.5);
    _dp_lengths[i] = cur_dp_length;
  }
  assert(cur_dp_length > std::numeric_limits<std::uint64_t>::max() / 2);
}

double StraightLineDistance1D::_compute_distance(double point0, double point1) const {
  if (point1 < point0) {
    std::swap(point0, point1);
  }
  // find i0 and i1 s.t. point0 < _grid_mid[i0] and _grid_mid[i1] < point1:
  std::size_t i0 = std::upper_bound(_grid_mid.begin(), _grid_mid.end(), point0) -
                   _grid_mid.begin();
  std::size_t i1 = std::lower_bound(_grid_mid.begin(), _grid_mid.end(), point1) -
                   _grid_mid.begin() - 1;
  assert(0 < i0);
  assert(_grid_mid[i0 - 1] <= point0);
  assert(point0 < _grid_mid[i0]);
  assert(i1 + 1 < _grid_mid.size());
  assert(_grid_mid[i1] < point1);
  assert(point1 <= _grid_mid[i1 + 1]);
  if (i0 > i1) {
    assert(i0 == i1 + 1);
    // The interval [point0, point1] is inside the cell
    // between _grid_mid[i0 - 1] and _grid_mid[i0].
    return (point1 - point0) * _metric_sqrt[i0 - 1];
  }
  assert(0 < i0);
  assert(i1 <= _dp_lengths.size());
  return (_grid_mid[i0] - point0) * _metric_sqrt[i0 - 1]
    + (point1 - _grid_mid[i1]) * _metric_sqrt[i1]
    + (_dp_lengths[i1 - 1] - _dp_lengths[i0 - 1]) * _dp_lengths_scale;
}

void StraightLineDistance1D::distances(
    std::span<double> point_span, std::span<double> results) {
  size_t num_point_pairs = results.size();
  CLASSIFIM_ASSERT(point_span.size() == num_point_pairs * 2);
  size_t i = 0;
  for (auto point0 = point_span.begin(), point1 = point0 + 1;
       point0 < point_span.end(); point0 += 2, point1 += 2, ++i) {
    results[i] = _compute_distance(*point0, *point1);
  }
}

std::uint64_t count_inversions(std::span<double> arr) {
  return InversionCounter::count_inversions(arr);
}

} // namespace classifim
