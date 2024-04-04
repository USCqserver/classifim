#include "metric_wrapper.h"

#include <span>
#include <vector>

#include "metric.h"

extern "C" {
void *create_straight_line_distance(int space_dim, int *grid_sizes,
                                    double *grid, double *metric) {
  std::span<int> grid_shape_span(grid_sizes, space_dim);
  std::vector<std::span<double>> grid_spans;
  double *grid_ptr = grid;
  std::size_t metric_size = space_dim * space_dim;
  for (int grid_size : grid_shape_span) {
    grid_spans.push_back(std::span<double>(grid_ptr, grid_size));
    grid_ptr += grid_size;
    metric_size *= grid_size;
  }

  classifim::StraightLineDistance *distanceObj =
      new classifim::StraightLineDistance(
          grid_spans, std::span<double>(metric, metric_size));
  return distanceObj;
}

void delete_straight_line_distance(void *obj) {
  classifim::StraightLineDistance *distanceObj =
      reinterpret_cast<classifim::StraightLineDistance *>(obj);
  delete distanceObj;
}

void straight_line_distance_distances(void *obj, int num_point_pairs,
                                      double *points, double *results) {
  classifim::StraightLineDistance *distanceObj =
      reinterpret_cast<classifim::StraightLineDistance *>(obj);
  std::span<double> point_span(points,
                               num_point_pairs * 2 * distanceObj->space_dim);
  std::vector<double> distances =
      distanceObj->distances(num_point_pairs, point_span);
  // TODO:5: Should we restructure the code to avoid a copy here?
  std::copy(distances.begin(), distances.end(), results);
}

void *create_straight_line_distance1d(int grid_size, double *grid,
                                      double *metric) {
  std::span<double> grid_span(grid, grid_size);
  std::span<double> metric_span(metric, grid_size);
  classifim::StraightLineDistance1D *distanceObj =
      new classifim::StraightLineDistance1D(grid_span, metric_span);
  return distanceObj;
}

void delete_straight_line_distance1d(void *obj) {
  classifim::StraightLineDistance1D *distanceObj =
      reinterpret_cast<classifim::StraightLineDistance1D *>(obj);
  delete distanceObj;
}

void straight_line_distance1d_distances(void *obj, int num_point_pairs,
                                        double *points, double *results) {
  classifim::StraightLineDistance1D *distanceObj =
      reinterpret_cast<classifim::StraightLineDistance1D *>(obj);
  std::span<double> point_span(points, num_point_pairs * 2);
  std::span<double> results_span(results, num_point_pairs);
  distanceObj->distances(point_span, results_span);
}

std::uint64_t count_inversions(int num_values, double *values) {
  std::span<double> values_span(values, num_values);
  return classifim::count_inversions(values_span);
}

} // extern "C"
