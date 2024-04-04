#ifndef INCLUDED_METRIC_WRAPPER
#define INCLUDED_METRIC_WRAPPER

#ifndef __cplusplus
#error "This header requires C++"
#endif

#include <cstdint>

extern "C" {
void *create_straight_line_distance(int space_dim, int *grid_sizes,
                                    double *grid, double *metric);
void delete_straight_line_distance(void *obj);
void straight_line_distance_distances(void *obj, int num_point_pairs,
                                      double *points, double *results);
void *create_straight_line_distance1d(
    int grid_size, double *grid, double *metric);
void delete_straight_line_distance1d(void *obj);
void straight_line_distance1d_distances(
    void *obj, int num_point_pairs, double *points, double *results);
std::uint64_t count_inversions(int num_values, double *values);

} // extern "C"
#endif // INCLUDED_FIL24_HAMILTONIAN_WRAPPER
