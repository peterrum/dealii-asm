#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

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

  const auto generate_file_name = []() -> std::array<std::string, 3> {
    static unsigned int counter = 0;

    const std::array<std::string, 3> names = {
      {"mesh_types_types_mesh_03." + std::to_string(counter) + ".vtu",
       "mesh_types_types_submesh_03." + std::to_string(counter) + ".vtu",
       "mesh_types_types_points_03." + std::to_string(counter) + ".vtu"}};

    counter++;

    return names;
  };

  const auto run = [&](const auto &       tria,
                       const auto &       mapping,
                       const unsigned int shift) {
    DoFHandler<dim> dof_handler(tria);

    FE_Q<dim> fe(4);

    dof_handler.distribute_dofs(fe);

    const auto file_names = generate_file_name();

    {
      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_triangulation(tria);
      data_out.build_patches(
        mapping, 3, DataOut<dim>::CurvedCellRegion::curved_inner_cells);

      std::ofstream file(file_names[0]);
      data_out.write_vtu(file);
    }

    std::vector<Point<dim>> support_points;


    auto cell = tria.begin_active();

    if (shift != 0)
      for (unsigned int i = 0; i < shift; ++i)
        cell++;

    const auto harmonic_cell_extend = GridTools::compute_harmonic_patch_extend(
      mapping,
      tria,
      QGaussLobatto<dim - 1>(fe.degree + 1))[cell->active_cell_index()];

    for (const auto &d : harmonic_cell_extend)
      {
        for (const auto &i : d)
          printf("%f ", i);
        std::cout << "    ";
      }

    std::cout << std::endl;

    auto center = [&]() {
      Quadrature<dim> quadrature(Point<dim>(0.5, 0.5));

      FEValues<dim> fe_values(mapping,
                              fe,
                              quadrature,
                              update_quadrature_points);

      fe_values.reinit(cell);

      return fe_values.quadrature_point(0);
    }();

    Point<dim>                       point_0, point_1;
    std::vector<std::vector<double>> step_sizes(dim);


    center[0] += 0.9;

    for (unsigned int d = 0; d < dim; ++d)
      {
        for (unsigned int i = 0; i < 3; ++i)
          step_sizes[d].push_back(harmonic_cell_extend[d][i]);

        point_0[d] = center[d] - harmonic_cell_extend[d][0] -
                     harmonic_cell_extend[d][1] / 2.0;
        point_1[d] = center[d] + harmonic_cell_extend[d][2] +
                     harmonic_cell_extend[d][1] / 2.0;
      }

    Triangulation<dim> sub_tria;

    GridGenerator::subdivided_hyper_rectangle(
      sub_tria, step_sizes, point_0, point_1, false);

    {
      DataOut<dim> data_out;

      // DataOutBase::VtkFlags flags;
      // flags.write_higher_order_cells = true;
      // data_out.set_flags(flags);

      data_out.attach_triangulation(sub_tria);
      data_out.build_patches();

      std::ofstream file(file_names[1]);
      data_out.write_vtu(file);
    }



    const auto cells =
      GridTools::extract_all_surrounding_cells_cartesian<dim>(cell, 0);

    const auto dofs =
      DoFTools::get_dof_indices_cell_with_overlap(dof_handler, cells, 3, true);

    const std::set<types::global_dof_index> dof_set(dofs.begin(), dofs.end());

    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    for (const auto cell : tria.active_cell_iterators())
      {
        auto points = fe.get_unit_support_points();

        for (auto &point : points)
          for (unsigned int d = 0; d < dim; ++d)
            point[d] = (point[d] - 0.5) * 0.999 + 0.5;

        Quadrature<dim> quadrature(points);

        FEValues<dim> fe_values(mapping,
                                fe,
                                quadrature,
                                update_quadrature_points);

        fe_values.reinit(cell);

        for (const auto q : fe_values.quadrature_point_indices())
          support_points.emplace_back(fe_values.quadrature_point(q));
      }

    Particles::ParticleHandler<dim> particle_handler(tria, mapping);
    particle_handler.insert_particles(support_points);

    Particles::DataOut<dim> data_out_particles;
    data_out_particles.build_patches(particle_handler);
    std::ofstream file_particles(file_names[2]);
    data_out_particles.write_vtu(file_particles);
  };

  if (true)
    {
      Triangulation<dim> tria;
      GridGenerator::subdivided_hyper_cube(tria, 6);

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

      run(tria, mapping, 15);
    }
}
