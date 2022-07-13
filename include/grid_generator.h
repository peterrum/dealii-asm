#pragma once

namespace dealii
{
  namespace GridGenerator
  {
    template <int dim, typename Container>
    void
    create_mesh_from_cells(const Container &cells, Triangulation<dim> &sub_tria)
    {
      sub_tria.clear();

      if (cells.size() == 0)
        return; // no cell

      const auto first_cell_ptr =
        std::find_if(cells.begin(), cells.end(), [](const auto &cell) {
          return cell.state() == IteratorState::valid;
        });

      if (first_cell_ptr == cells.end())
        return; // no cell valid

      const auto &tria = (*first_cell_ptr)->get_triangulation();

      // copy manifolds
      for (const auto i : tria.get_manifold_ids())
        if (i != numbers::flat_manifold_id)
          sub_tria.set_manifold(i, tria.get_manifold(i));

      // renumerate vertices
      std::vector<unsigned int> new_vertex_indices(tria.n_vertices(), 0);

      for (const auto &cell : cells)
        if (cell != tria.end())
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
        if (cell != tria.end())
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
        if (cell != tria.end())
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