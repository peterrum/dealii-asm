#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "include/grid_generator.h"
#include "include/grid_tools.h"

using namespace dealii;

template <int dim>
void
test(unsigned int, unsigned int n_global_refinements)
{
  const unsigned int mapping_degree = 3;

  Triangulation<dim> tria;
  if (true)
    GridGenerator::hyper_cube(tria);
  else
    GridGenerator::hyper_ball_balanced(tria);
  tria.refine_global(n_global_refinements);

  MappingQ<dim> mapping(mapping_degree);

  const auto runner = [&](const auto &       mapping,
                          const auto &       fu,
                          const auto &       fu_update_mapping,
                          const std::string &label) {
    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.attach_triangulation(tria);
    fu_update_mapping(tria);
    data_out.build_patches(mapping,
                           mapping_degree,
                           DataOut<dim>::curved_inner_cells);
    std::ofstream ostream(label + ".vtu");
    data_out.write_vtu(ostream);

    unsigned int counter = 0;

    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto all_surrounding_cells = fu(cell);

          Triangulation<dim> sub_tria;
          GridGenerator::create_mesh_from_cells(all_surrounding_cells,
                                                sub_tria);

          DataOut<dim>          data_out;
          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);
          data_out.attach_triangulation(sub_tria);
          fu_update_mapping(sub_tria);
          data_out.build_patches(mapping,
                                 mapping_degree,
                                 DataOut<dim>::curved_inner_cells);
          std::ofstream ostream(label + std::to_string(counter++) + ".vtu");
          data_out.write_vtu(ostream);
        }
  };

  runner(
    mapping,
    [&](const auto &cell) {
      return GridTools::extract_all_surrounding_cells<dim>(cell);
    },
    [](const auto &) {
      // nothing to do
    },
    "all_surrounding_cells");

  if (GridTools::is_mesh_structured(tria))
    {
      MappingQCache<dim> mapping_q_cache(mapping_degree);

      for (unsigned int level = 0; level <= dim; ++level)
        runner(
          mapping_q_cache,
          [&](const auto &cell) {
            return GridTools::extract_all_surrounding_cells_cartesian<dim>(
              cell, level);
          },
          [&](const auto &tria) {
            mapping_q_cache.initialize(
              mapping,
              tria,
              [](const auto &, const auto &point) {
                Point<dim> result;

                for (unsigned int d = 0; d < dim; ++d)
                  result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                              std::sin(numbers::PI * point[d]) * 0.1;

                return result;
              },
              true);
          },
          "cartesian_surrounding_cells_" + std::to_string(level) + "_");
    }
  else
    {
      std::cout << "Not running GridTools::extract_all_surrounding_cells()!"
                << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 3;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements);
  else
    AssertThrow(false, ExcNotImplemented());
}
