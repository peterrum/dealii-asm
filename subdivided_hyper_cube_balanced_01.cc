#include "include/grid_generator.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 3;

  for (unsigned int s = 1; s < 50; ++s)
    {
      const auto [n_refinements, subdivisions] =
        GridGenerator::internal::decompose_for_subdivided_hyper_cube_balanced(
          dim, s);

      printf("%5d %5d", s, n_refinements);

      for (const auto subdivision : subdivisions)
        printf("%5d", subdivision);

      auto n_subdivisions = 1;

      for (const auto subdivision : subdivisions)
        n_subdivisions *= subdivision;


      printf("%10.2e",
             static_cast<double>(n_subdivisions *
                                 Utilities::pow(2, dim * n_refinements)));

      printf("\n");
    }
}