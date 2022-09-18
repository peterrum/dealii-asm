#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <fstream>

using namespace dealii;

int
main()
{
  const unsigned int dim = 2;

  std::vector<Point<dim>>    vertices;
  std::vector<CellData<dim>> cells;

  for (unsigned int i = 0; i < 8; ++i)
    {
      double offset = i * 1.2;

      if (i >= 3)
        offset += 0.2;

      vertices.emplace_back(offset + 0.0, 0.0);
      vertices.emplace_back(offset + 0.0, 1.0);
      vertices.emplace_back(offset + 1.0, 0.0);
      vertices.emplace_back(offset + 1.0, 1.0);

      CellData<dim> cell;

      for (unsigned int v = 0; v < 4; ++v)
        cell.vertices[v] = i * 4 + v;

      cells.push_back(cell);
    }


  Triangulation<dim> tria;
  tria.create_triangulation(vertices, cells, {});

  tria.begin(0)->set_refine_flag();
  (++tria.begin(0))->set_refine_flag();

  tria.execute_coarsening_and_refinement();

  for (const auto &cell : tria.begin(0)->child_iterators())
    cell->set_refine_flag();

  tria.execute_coarsening_and_refinement();

  DataOut<dim> data_out;

  data_out.attach_triangulation(tria);
  data_out.build_patches();

  std::ofstream file("coarsening_types_mesh.vtu");
  data_out.write_vtu(file);

  std::vector<Point<dim>> support_points;

  MappingQ1<dim> mapping;

  unsigned int counter   = 0;
  unsigned int fe_degree = 5;
  for (const auto cell : tria.cell_iterators_on_level(0))
    {
      if (counter >= 3)
        {
          FE_Q<dim> fe(fe_degree);

          auto points = fe.get_unit_support_points();

          for (auto &point : points)
            for (unsigned int d = 0; d < dim; ++d)
              point[d] = (point[d] - 0.5) * 0.99 + 0.5;

          Quadrature<dim> quadrature(points);

          FEValues<dim> fe_values(mapping,
                                  fe,
                                  quadrature,
                                  update_quadrature_points);

          fe_values.reinit(cell);

          for (const auto &point : fe_values.get_quadrature_points())
            support_points.emplace_back(point);

          fe_degree--;
        }

      counter++;
    }

  Particles::ParticleHandler<dim> particle_handler(tria, mapping);
  particle_handler.insert_particles(support_points);

  Particles::DataOut<dim> data_out_particles;
  data_out_particles.build_patches(particle_handler);
  std::ofstream file_particles("coarsening_types_points.vtu");
  data_out_particles.write_vtu(file_particles);
}