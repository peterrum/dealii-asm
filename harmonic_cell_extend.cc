#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include "include/grid_tools.h"

using namespace dealii;

template <int dim>
void
test(unsigned int fe_degree, unsigned int n_global_refinements)
{
  Triangulation<dim> tria;
  if (false)
    GridGenerator::hyper_cube(tria);
  else
    GridGenerator::hyper_ball_balanced(tria);
  tria.refine_global(n_global_refinements);

  MappingQ<dim>          mapping(3);
  QGaussLobatto<dim - 1> quad(fe_degree + 1);

  {
    const auto harmonic_cell_extends =
      GridTools::compute_harmonic_cell_extend(mapping, tria, quad);

    std::cout << "GridTools::compute_harmonic_cell_extend:" << std::endl;
    for (const auto &harmonic_cell_extend : harmonic_cell_extends)
      {
        for (const auto &i : harmonic_cell_extend)
          printf("%f ", i);
        std::cout << std::endl;
      }
    std::cout << std::endl;
  }

  {
    const auto harmonic_patch_extends =
      GridTools::compute_harmonic_patch_extend(mapping, tria, quad);

    std::cout << "GridTools::compute_harmonic_patch_extend:" << std::endl;
    for (const auto &harmonic_patch_extend : harmonic_patch_extends)
      {
        for (const auto &d : harmonic_patch_extend)
          {
            for (const auto &i : d)
              printf("%f ", i);
            std::cout << "    ";
          }

        std::cout << std::endl;
      }
    std::cout << std::endl;
  }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 0;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements);
  else
    AssertThrow(false, ExcNotImplemented());
}