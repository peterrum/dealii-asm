#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

#include "include/grid_tools.h"

using namespace dealii;

namespace dealii
{
  namespace GridTools
  {
    template <int dim>
    std::vector<typename Triangulation<dim>::cell_iterator>
    extract_all_surrounding_cells(
      const typename Triangulation<dim>::cell_iterator &cell)
    {
      const auto &tria = cell->get_triangulation();

      std::vector<std::vector<typename Triangulation<dim>::cell_iterator>>
        vertex_to_cells(tria.n_vertices());

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int v : cell->vertex_indices())
            vertex_to_cells[cell->vertex_index(v)].emplace_back(cell);

      std::vector<typename Triangulation<dim>::cell_iterator> cells;

      for (const unsigned int v : cell->vertex_indices())
        for (const auto other_cell : vertex_to_cells[cell->vertex_index(v)])
          cells.emplace_back(other_cell);

      std::sort(cells.begin(), cells.end());
      cells.erase(std::unique(cells.begin(), cells.end()), cells.end());

      return cells;
    }
  } // namespace GridTools
} // namespace dealii

namespace dealii
{
  namespace GridGenerator
  {
    template <int dim>
    void
    create_mesh_from_cells(
      const std::vector<typename Triangulation<dim>::cell_iterator> &cells,
      Triangulation<dim> &                                           sub_tria)
    {
      sub_tria.clear();

      if (cells.size() == 0)
        return; // nothing to do

      const auto &tria = cells.front()->get_triangulation();

      // copy manifolds
      for (const auto i : tria.get_manifold_ids())
        if (i != numbers::flat_manifold_id)
          sub_tria.set_manifold(i, tria.get_manifold(i));

      // renumerate vertices
      std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), 0);

      for (const auto &cell : cells)
        for (const unsigned int v : cell->vertex_indices())
          new_vertex_indices[cell->vertex_index(v)] = 1;

      for (unsigned int i = 0, c = 0; i < new_vertex_indices.size(); ++i)
        if (new_vertex_indices[i] == 0)
          new_vertex_indices[i] = numbers::invalid_unsigned_int;
        else
          new_vertex_indices[i] = c++;

      // collect points
      std::vector<Point<dim>> sub_points;
      for (unsigned int i = 0; i < new_vertex_indices.size(); ++i)
        if (new_vertex_indices[i] != numbers::invalid_unsigned_int)
          sub_points.emplace_back(tria.get_vertices()[i]);

      // create new cell and subcell data
      std::vector<CellData<dim>> sub_cells;

      for (const auto &cell : cells)
        {
          // cell
          CellData<dim> new_cell(cell->n_vertices());

          for (const auto v : cell->vertex_indices())
            new_cell.vertices[v] = new_vertex_indices[cell->vertex_index(v)];

          new_cell.material_id = cell->material_id();
          new_cell.manifold_id = cell->manifold_id();

          sub_cells.emplace_back(new_cell);
        }

      // create mesh
      sub_tria.create_triangulation(sub_points, sub_cells, {});

      auto sub_cell = sub_tria.begin();

      for (const auto &cell : cells)
        {
          // faces
          for (const auto f : cell->face_indices())
            {
              const auto face = cell->face(f);

              if (face->manifold_id() != numbers::flat_manifold_id)
                sub_cell->face(f)->set_manifold_id(face->manifold_id());

              if (face->boundary_id() != numbers::internal_face_boundary_id)
                sub_cell->face(f)->set_boundary_id(face->boundary_id());
            }

          // lines
          if (dim == 3)
            for (const auto l : cell->line_indices())
              {
                const auto line = cell->line(l);

                if (line->manifold_id() != numbers::flat_manifold_id)
                  sub_cell->line(l)->set_manifold_id(line->manifold_id());
              }

          sub_cell++;
        }
    }
  } // namespace GridGenerator
} // namespace dealii

template <int dim>
void
test(unsigned int, unsigned int n_global_refinements)
{
  Triangulation<dim> tria;
  if (false)
    GridGenerator::hyper_cube(tria);
  else
    GridGenerator::hyper_ball_balanced(tria);
  tria.refine_global(n_global_refinements);

  MappingQ<dim> mapping(3);

  DataOut<dim>          data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);
  data_out.attach_triangulation(tria);
  data_out.build_patches(mapping, 3, DataOut<dim>::curved_inner_cells);
  std::ofstream ostream("all_surrounding_cells.vtu");
  data_out.write_vtu(ostream);

  unsigned int counter = 0;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto all_surrounding_cells =
          GridTools::extract_all_surrounding_cells<dim>(cell);

        Triangulation<dim> sub_tria;
        GridGenerator::create_mesh_from_cells(all_surrounding_cells, sub_tria);

        DataOut<dim>          data_out;
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);
        data_out.attach_triangulation(sub_tria);
        data_out.build_patches(mapping, 3, DataOut<dim>::curved_inner_cells);
        std::ofstream ostream("all_surrounding_cells" +
                              std::to_string(counter++) + ".vtu");
        data_out.write_vtu(ostream);
      }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 1;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements);
  else
    AssertThrow(false, ExcNotImplemented());
}
