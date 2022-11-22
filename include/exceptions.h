#pragma once

#include <deal.II/lac/la_parallel_vector.h>

#define AssertThrowVectorZeroGhost(vec)                         \
  {                                                             \
    bool is_zero_ghost = !vec.has_ghost_elements();             \
    for (const auto i : vec.get_partitioner()->ghost_indices()) \
      is_zero_ghost &= (vec[i] == 0.0);                         \
    AssertThrow(is_zero_ghost, ExcInternalError());             \
  }


#define DEBUG_OUTPUT(a)                                        \
  {                                                            \
    MPI_Barrier(MPI_COMM_WORLD);                               \
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) \
      std::cout << a << std::endl;                             \
  }
