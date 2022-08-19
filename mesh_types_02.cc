#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/kershaw.h"

using namespace dealii;

int
main()
{
  const unsigned int dim = 2;

  const auto generate_file_name = []() {
    static unsigned int counter = 0;

    std::pair<std::string, std::string> names = {
      "mesh_types_types_mesh_02." + std::to_string(counter) + ".vtu",
      "mesh_types_types_points_02." + std::to_string(counter) + ".vtu"};

    counter++;

    return names;
  };

  for (unsigned int i = 0; i < 3; ++i)
    {
      Triangulation<dim> tria;
      GridGenerator::subdivided_hyper_cube(tria, 3);

      DoFHandler<dim> dof_handler(tria);

      FE_Q<dim> fe(4);

      dof_handler.distribute_dofs(fe);

      MappingQ1<dim> mapping_q1;

      MappingQCache<dim> mapping(3);

      mapping.initialize(
        mapping_q1,
        tria,
        [](const auto &, const auto &point) {
          Point<dim> result;

          for (unsigned int d = 0; d < dim; ++d)
            result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                        std::sin(numbers::PI * point[d]) * 0.1;

          return result;
        },
        true);

      const auto file_names = generate_file_name();

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_triangulation(tria);
      data_out.build_patches(
        mapping, 3, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      std::ofstream file(file_names.first);
      data_out.write_vtu(file);

      std::vector<Point<dim>> support_points;

      auto cell = tria.begin();

      for (unsigned int i = 0; i < 4; ++i)
        cell++;

      const auto cells =
        GridTools::extract_all_surrounding_cells_cartesian<dim>(cell, i);

      const auto dofs = DoFTools::get_dof_indices_cell_with_overlap(dof_handler,
                                                                    cells,
                                                                    3,
                                                                    true);

      const std::set<types::global_dof_index> dof_set(dofs.begin(), dofs.end());

      std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

      for (const auto cell : cells)
        {
          if (cell.state() != IteratorState::valid)
            continue;

          const auto cell_dof = cell->as_dof_handler_iterator(dof_handler);

          auto points = fe.get_unit_support_points();

          for (auto &point : points)
            for (unsigned int d = 0; d < dim; ++d)
              point[d] = (point[d] - 0.5) * 0.999 + 0.5;

          Quadrature<dim> quadrature(points);

          FEValues<dim> fe_values(mapping,
                                  fe,
                                  quadrature,
                                  update_quadrature_points);

          fe_values.reinit(cell_dof);

          cell_dof->get_dof_indices(dof_indices);

          for (const auto q : fe_values.quadrature_point_indices())
            if (dof_set.find(dof_indices[q]) != dof_set.end())
              support_points.emplace_back(fe_values.quadrature_point(q));
        }

      Particles::ParticleHandler<dim> particle_handler(tria, mapping);
      particle_handler.insert_particles(support_points);

      Particles::DataOut<dim> data_out_particles;
      data_out_particles.build_patches(particle_handler);
      std::ofstream file_particles(file_names.second);
      data_out_particles.write_vtu(file_particles);
    }
}
