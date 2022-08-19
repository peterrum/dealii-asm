#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "include/kershaw.h"

using namespace dealii;

int
main()
{
  const unsigned int dim = 3;

  const auto generate_file_name = []() {
    static unsigned int counter = 0;

    return "mesh_types_types_" + std::to_string(counter++) + ".vtu";
  };

  if (true) // hyper-cube
    {
      Triangulation<dim> tria;
      GridGenerator::hyper_cube(tria, -1.0, +1.0);
      tria.refine_global(3);

      DataOut<dim> data_out;

      data_out.attach_triangulation(tria);
      data_out.build_patches();

      std::ofstream file(generate_file_name());
      data_out.write_vtu(file);
    }

  if (true) // hyper-rectangle
    {
      Triangulation<dim> tria;
      GridGenerator::hyper_rectangle(tria,
                                     Point<dim>(-3.0, -1.0, -1.0),
                                     Point<dim>(+3.0, +1.0, +1.0));
      tria.refine_global(3);

      DataOut<dim> data_out;

      data_out.attach_triangulation(tria);
      data_out.build_patches();

      std::ofstream file(generate_file_name());
      data_out.write_vtu(file);
    }


  for (unsigned int i = 0; i < 2; ++i) // hyper-ball
    {
      Triangulation<dim> tria;
      GridGenerator::hyper_ball_balanced(tria, Point<dim>(), 1.0);
      tria.refine_global(0);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      MappingQ<dim> mapping(3);

      const auto next_cell = [&](const auto &tria, const auto &cell_in) {
        auto cell = cell_in;

        while (true)
          {
            cell++;

            if (cell == tria.end())
              break;

            if (cell->is_active())
              {
                for (unsigned int d = 0; d < dim; ++d)
                  if (cell->center()[d] < 0.0)
                    return cell;
              }
          }

        return tria.end();
      };

      // output mesh
      const auto first_cell = [&](const auto &tria) {
        return next_cell(tria, tria.begin());
      };

      if (i == 0)
        data_out.set_cell_selection(first_cell, next_cell);

      data_out.attach_triangulation(tria);
      data_out.build_patches(
        mapping, 3, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      std::ofstream file(generate_file_name());
      data_out.write_vtu(file);
    }

  for (const double eps : {1.0, 0.9, 0.7, 0.5, 0.3})
    {
      Triangulation<dim> tria;
      GridGenerator::subdivided_hyper_cube(tria, 6);
      tria.refine_global(0);

      const double epsy = eps;
      const double epsz = eps;

      const auto transformation_function = [epsy, epsz](const auto &,
                                                        const auto &in_point) {
        Point<dim> out_point;
        // clang-format off
        kershaw(epsy, epsz, in_point[0], in_point[1], in_point[2], out_point[0], out_point[1], out_point[2]);
        // clang-format on

        return out_point;
      };

      MappingQCache<dim> mapping(3);

      const MappingQ1<dim> mapping_q1;
      mapping.initialize(mapping_q1, tria, transformation_function, false);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_triangulation(tria);
      data_out.build_patches(
        mapping, 3, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      std::ofstream file(generate_file_name());
      data_out.write_vtu(file);
    }
}
