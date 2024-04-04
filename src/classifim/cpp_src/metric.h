#ifndef INCLUDED_METRIC
#define INCLUDED_METRIC

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <xtensor/xarray.hpp>

namespace classifim {
class StraightLineDistance {
public:
  const int space_dim;

public:
  StraightLineDistance(std::span<std::span<double>> grid_spans,
                       std::span<double> metric_span);

  std::vector<double> distances(int num_point_pairs,
                                std::span<double> point_span);

private:
  // Main data
  std::vector<std::vector<double>> _grid;

  // Precomputed data for _compute_distance function:
  std::unique_ptr<double[]> _grid_mid_data;
  std::vector<std::span<double>> _grid_mid;

  // tie_mask is a bitmask that indicates which dimensions are tied.
  // In below explanation we write tie_mask[j] to mean (tie_mask >> j) & 1.
  // metric has shape (d[0] - tie_mask[0], d[1] - tie_mask[1], ...,
  // d[space_dim - 1] - tie_mask[space_dim - 1],
  // space_dim - num_ties, space_dim - num_ties)
  // The first components of the shape represent the space, where
  // for tied dimensions the points on _grid_mid are provided,
  // and for others the points on _grid are provided.
  // The last 2 components of the shape represent the metric
  // (a symmetric matrix) in the remaining (untied) dimensions.
  // Thus, in metric[..., j, k] the index j corresponds to the dimension
  // of the space matching the location of j-th 0 in tie_mask.
  std::unordered_map<std::uint64_t, xt::xarray<double>> _tie_mask_to_metric;

  std::vector<double> _scratch_d;
  std::vector<int> _scratch_i;

private:
  double _compute_distance(std::span<double> point0, std::span<double> point1);
  void _init_metric(std::uint64_t tie_mask, xt::xarray<double> &new_metric);
  const xt::xarray<double> &_get_metric(std::uint64_t tie_mask) {
    auto metric_it = _tie_mask_to_metric.find(tie_mask);
    if (metric_it == _tie_mask_to_metric.end()) {
      // Insert empty vector using emplace_hint.
      // Then compute the metric using _init_metric
      metric_it = _tie_mask_to_metric.emplace_hint(metric_it, tie_mask,
                                                   xt::xarray<double>());
      _init_metric(tie_mask, metric_it->second);
    }
    return metric_it->second;
  }
  void _init();
  double _metric_dist(const xt::xarray<double> &metric,
                      std::span<int> cur_cell_idx, std::span<double> vec);
};

class StraightLineDistance1D {
  // space_dim is 1
public:
  StraightLineDistance1D(std::span<double> grid_span,
                         std::span<double> metric_span);

  void distances(std::span<double> point_span, std::span<double> results);

private:
  std::vector<double> _grid;
  std::vector<double> _metric_sqrt;

  // Precomputed data for _compute_distance function:
  // _grid_mid contains midpoints of _grid padded with +/- infinity.
  // _metric[i] is the metric between _grid_mid[i] and _grid_mid[i + 1].
  std::vector<double> _grid_mid;
  // _dp_lengths_scale * _dp_lengths[i] is the distance
  // between _grid_mid[1] and _grid_mid[i + 1]:
  std::vector<std::uint64_t> _dp_lengths;
  double _dp_lengths_scale;

private:
  void _precompute();
  double _compute_distance(double point0, double point1) const;
};

std::uint64_t count_inversions(std::span<double> arr);

} // namespace classifim
#endif // INCLUDED_FIL24_HAMILTONIAN
