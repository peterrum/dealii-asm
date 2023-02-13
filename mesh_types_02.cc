#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <fstream>

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/kershaw.h"
#include "include/restrictors.h"

using namespace dealii;

namespace dealii
{
  namespace GridGenerator
  {
    void
    my_quarter_hyper_ball(Triangulation<2> &tria,
                          const Point<2> &  p      = {},
                          const double      radius = 1.0)
    {
      const unsigned int dim = 2;

      // the numbers 0.55647 and 0.42883 have been found by a search for the
      // best aspect ratio (defined as the maximal between the minimal singular
      // value of the Jacobian)
      const Point<dim> vertices[7] = {
        p + Point<dim>(0, 0) * radius,
        p + Point<dim>(+1, 0) * radius,
        p + Point<dim>(+1, 0) * (radius * 0.55647),
        p + Point<dim>(0, +1) * (radius * 0.55647),
        p + Point<dim>(+1, +1) * (radius * 0.42883),
        p + Point<dim>(0, +1) * radius,
        p + Point<dim>(+1, +1) * (radius / std::sqrt(2.0))};

      const int cell_vertices[3][4] = {{0, 2, 3, 4},
                                       {2, 1, 4, 6},
                                       {3, 4, 5, 6}};

      std::vector<CellData<dim>> cells(3, CellData<dim>());

      for (unsigned int i = 0; i < 3; ++i)
        {
          for (unsigned int j = 0; j < 4; ++j)
            cells[i].vertices[j] = cell_vertices[i][j];
          cells[i].material_id = 0;
        }

      tria.create_triangulation(std::vector<Point<dim>>(std::begin(vertices),
                                                        std::end(vertices)),
                                cells,
                                SubCellData()); // no boundary information

      Triangulation<dim>::cell_iterator cell = tria.begin();
      Triangulation<dim>::cell_iterator end  = tria.end();

      tria.set_all_manifold_ids_on_boundary(0);

      while (cell != end)
        {
          for (unsigned int i : GeometryInfo<dim>::face_indices())
            {
              if (cell->face(i)->boundary_id() ==
                  numbers::internal_face_boundary_id)
                continue;

              // If one the components is the same as the respective
              // component of the center, then this is part of the plane
              if (cell->face(i)->center()(0) < p(0) + 1.e-5 * radius ||
                  cell->face(i)->center()(1) < p(1) + 1.e-5 * radius)
                {
                  cell->face(i)->set_boundary_id(1);
                  cell->face(i)->set_manifold_id(numbers::flat_manifold_id);
                }
            }
          ++cell;
        }
      tria.set_manifold(0, SphericalManifold<2>(p));
    }
  } // namespace GridGenerator
} // namespace dealii


int
main()
{
  const unsigned int dim = 2;

  const auto generate_file_name = []() {
    static unsigned int counter = 0;

    std::tuple<std::string, std::string, std::string, std::string> names = {
      "mesh_types_types_mesh_02." + std::to_string(counter) + ".vtu",
      "mesh_types_types_points_02." + std::to_string(counter) + ".vtu",
      "mesh_types_types_02." + std::to_string(counter) + ".0.tex",
      "mesh_types_types_02." + std::to_string(counter) + ".1.tex"};

    counter++;

    return names;
  };

  const auto run = [&](const auto &       tria,
                       const auto &       mapping,
                       const unsigned int i,
                       const unsigned int shift) {
    DoFHandler<dim> dof_handler(tria);

    const unsigned int n_points = 5;
    FE_Q<dim>          fe(n_points - 1);

    dof_handler.distribute_dofs(fe);

    const auto file_names = generate_file_name();

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_triangulation(tria);
    data_out.build_patches(mapping,
                           3,
                           DataOut<dim>::CurvedCellRegion::curved_inner_cells);

    std::ofstream file(std::get<0>(file_names));
    data_out.write_vtu(file);

    std::vector<Point<dim>>           support_points;
    std::set<types::global_dof_index> dof_set;

    if (i <= 2)
      {
        auto cell = tria.begin_active();

        if (shift != 0)
          for (unsigned int i = 0; i < shift; ++i)
            cell++;

        const auto cells =
          GridTools::extract_all_surrounding_cells_cartesian<dim>(cell, i);

        const auto dofs = DoFTools::get_dof_indices_cell_with_overlap(
          dof_handler, cells, 3, true);

        dof_set.insert(dofs.begin(), dofs.end());
      }
    else if (i == 3)
      {
        auto cell = tria.begin_active();

        if (shift != 0)
          for (unsigned int i = 0; i < shift; ++i)
            cell++;

        const auto cells =
          GridTools::extract_all_surrounding_cells_cartesian<dim>(cell, 0);

        const auto dofs = DoFTools::get_dof_indices_cell_with_overlap(
          dof_handler, cells, 1, true);

        dof_set.insert(dofs.begin(), dofs.end());

        // for(int o = 1; o < 3; ++o)
        {
          std::set<types::global_dof_index> dof_set_new = dof_set;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              const auto cells =
                GridTools::extract_all_surrounding_cells_cartesian<dim>(cell,
                                                                        0);

              const auto dofs = DoFTools::get_dof_indices_cell_with_overlap(
                dof_handler, cells, 1, true);

              for (unsigned int i = 0; i < n_points; ++i)
                for (unsigned int j = 0; j < n_points; ++j)
                  for (int ii = -2; ii <= 2; ++ii)
                    for (int jj = -2; jj <= 2; ++jj)
                      if (dof_set.find(
                            dofs[std::min<int>(std::max<int>(0, i + ii),
                                               n_points - 1) +
                                 std::min<int>(std::max<int>(0, j + jj),
                                               n_points - 1) *
                                   n_points]) != dof_set.end())
                        dof_set_new.insert(dofs[i + j * n_points]);


              std::cout << dof_set_new.size() << std::endl;
            }

          dof_set = dof_set_new;
          std::cout << std::endl;
        }
      }
    else if (i == 4)
      {
        typename Restrictors::ElementCenteredRestrictor<
          Vector<double>>::AdditionalData ad;
        ad.type = "vertex_all";

        Restrictors::ElementCenteredRestrictor<Vector<double>> restrictor(
          dof_handler, ad);

        const auto dofs = restrictor.get_indices()[shift];
        dof_set.insert(dofs.begin(), dofs.end());
      }

    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
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

        cell->get_dof_indices(dof_indices);

        for (const auto q : fe_values.quadrature_point_indices())
          if (dof_set.find(dof_indices[q]) != dof_set.end())
            support_points.emplace_back(fe_values.quadrature_point(q));
      }

    Particles::ParticleHandler<dim> particle_handler(tria, mapping);
    particle_handler.insert_particles(support_points);

    Particles::DataOut<dim> data_out_particles;
    data_out_particles.build_patches(particle_handler);
    std::ofstream file_particles(std::get<1>(file_names));
    data_out_particles.write_vtu(file_particles);

    {
      std::ofstream file(std::get<3>(file_names));

      for (unsigned int p = 0; p < support_points.size(); ++p)
        {
          const auto point = support_points[p];

          file << "(";
          file << point[0];
          file << ",";
          file << point[1];
          file << ")";

          if (p + 1 != support_points.size())
            file << ",";
        }
    }

    {
      std::ofstream file(std::get<2>(file_names));

      for (const auto &cell : tria.active_cell_iterators())
        {
          const unsigned int n_subdivisions = 3;

          std::vector<Point<dim>> points;

          for (unsigned int i = 0; i <= n_subdivisions; ++i)
            points.emplace_back(1.0 / n_subdivisions * i, 0.0);

          for (unsigned int i = 0; i <= n_subdivisions; ++i)
            points.emplace_back(1.0 / n_subdivisions * i, 1.0);

          for (unsigned int i = 0; i <= n_subdivisions; ++i)
            points.emplace_back(0.0, 1.0 / n_subdivisions * i);

          for (unsigned int i = 0; i <= n_subdivisions; ++i)
            points.emplace_back(1.0, 1.0 / n_subdivisions * i);

          Quadrature<dim> quadrature(points);

          FEValues<dim> fe_values(mapping,
                                  fe,
                                  quadrature,
                                  update_quadrature_points);

          fe_values.reinit(cell);

          for (unsigned int q = 0; q < points.size();)
            {
              file << "\\draw [black] plot [smooth] coordinates {";
              for (unsigned int i = 0; i <= n_subdivisions; ++i, ++q)
                {
                  const auto point = fe_values.quadrature_point(q);

                  file << "(";
                  file << point[0];
                  file << ",";
                  file << point[1];
                  file << ")";
                }
              file << "};" << std::endl;
            }

          file << std::endl;

          // for (const auto q : fe_values.quadrature_point_indices())
          //  std::cout << fe_values.quadrature_point(q) << std::endl;
        }
    }
  };

  for (unsigned int i = 0; i <= 4; ++i)
    {
      Triangulation<dim> tria;
      GridGenerator::subdivided_hyper_cube(tria, 3);

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

      if (i != 3)
        run(tria, mapping, i, i == 4 ? 6 : 4);
    }

  for (const auto i : {0, 1, 3, 4})
    {
      Triangulation<dim> tria;
      GridGenerator::my_quarter_hyper_ball(tria);

      tria.refine_global(1);

      MappingQ<dim> mapping(3);

      run(tria, mapping, i, i == 4 ? 4 : 3);
    }
}
