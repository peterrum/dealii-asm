#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"

using namespace dealii;

template <int dim>
void
test(unsigned int fe_degree, unsigned int n_global_refinements)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto cells =
          GridTools::extract_all_surrounding_cells_cartesian<dim>(cell);

        for (unsigned int n_overlap = 0; n_overlap <= 3; ++n_overlap)
          {
            const auto dof_indices =
              DoFTools::get_dof_indices_cell_with_overlap(dof_handler,
                                                          cells,
                                                          n_overlap);

            for (const auto index : dof_indices)
              std::cout << index << " ";
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
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 2;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 3;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements);
  else
    AssertThrow(false, ExcNotImplemented());
}
