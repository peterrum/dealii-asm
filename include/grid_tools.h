#pragma once

namespace dealii
{
  namespace GridTools
  {
    // Compute harminic extend of all non-artificial cells.
    template <int dim>
    std::vector<std::array<double, dim>>
    compute_harmonic_cell_extend(const Mapping<dim> &       mapping,
                                 const Triangulation<dim> & triangulation,
                                 const Quadrature<dim - 1> &quadrature)
    {
      std::vector<std::array<double, dim>> result(
        triangulation.n_active_cells());

      FE_Nothing<dim>   fe_nothing;
      FEFaceValues<dim> fe_face_values_0(mapping,
                                         fe_nothing,
                                         quadrature,
                                         update_quadrature_points);
      FEFaceValues<dim> fe_face_values_1(mapping,
                                         fe_nothing,
                                         quadrature,
                                         update_quadrature_points);

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_artificial() == false)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                fe_face_values_0.reinit(cell, 2 * d + 0);
                fe_face_values_1.reinit(cell, 2 * d + 1);

                double extend = 0.0;

                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  extend += fe_face_values_0.quadrature_point(q).distance(
                              fe_face_values_1.quadrature_point(q)) *
                            quadrature.weight(q);

                result[cell->active_cell_index()][d] = extend;
              }
          }

      return result;
    }

    // Compute harmonic extend of each locally owned cell including of each
    // of its neighbors. If there is no neigbor, its extend is zero.
    template <int dim>
    std::vector<dealii::ndarray<double, dim, 3>>
    compute_harmonic_patch_extend(const Mapping<dim> &       mapping,
                                  const Triangulation<dim> & triangulation,
                                  const Quadrature<dim - 1> &quadrature)
    {
      // 1) compute extend of each non-artificial cell
      const auto harmonic_cell_extends =
        GridTools::compute_harmonic_cell_extend(mapping,
                                                triangulation,
                                                quadrature);

      // 2) accumulate for each face the normal extend for the
      // neigboring cell(s)
      std::vector<double> face_extend(triangulation.n_faces(), 0.0);

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_artificial() == false)
          for (unsigned int d = 0; d < dim; ++d)
            {
              const auto extend =
                harmonic_cell_extends[cell->active_cell_index()][d];

              face_extend[cell->face(2 * d + 0)->index()] += extend;
              face_extend[cell->face(2 * d + 1)->index()] += extend;
            }

      // 3) cellect cell extend including those of the neighboring
      // cells, which corrsponds to the difference of extend of the
      // current cell and the face extend
      std::vector<dealii::ndarray<double, dim, 3>> result(
        triangulation.n_active_cells());

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          for (unsigned int d = 0; d < dim; ++d)
            {
              const auto cell_extend =
                harmonic_cell_extends[cell->active_cell_index()][d];

              const auto index = cell->active_cell_index();

              result[index][d][0] =
                face_extend[cell->face(2 * d + 0)->index()] - cell_extend;
              result[index][d][1] = cell_extend;
              result[index][d][2] =
                face_extend[cell->face(2 * d + 1)->index()] - cell_extend;
            }

      return result;
    }


    DeclExceptionMsg(ExcMeshIsNotStructured,
                     "You are using an ustructured mesh. "
                     "This function is not working for such kind of meshes!");

    template <int dim>
    bool
    is_mesh_structured(const Triangulation<dim> &tria)
    {
      bool is_structured = true;

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          for (const auto f : cell->face_indices())
            if (cell->at_boundary(f) == false)
              {
                const auto cell_neighbor = cell->neighbor(f);

                const unsigned int fo = f ^ 1;

                is_structured &= ((cell_neighbor->at_boundary(fo) == false) &&
                                  (cell_neighbor->neighbor(fo) == cell));
              }

      return Utilities::MPI::min(static_cast<double>(is_structured),
                                 tria.get_communicator()) == 1.0;
    }

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

    template <int dim>
    std::array<typename Triangulation<dim>::cell_iterator,
               Utilities::pow(3, dim)>
    extract_all_surrounding_cells_cartesian(
      const typename Triangulation<dim>::cell_iterator &cell,
      const unsigned int                                level,
      const bool                                        strict = false)
    {
      (void)strict; // TODO

      const auto &tria = cell->get_triangulation();

      std::array<typename Triangulation<dim>::cell_iterator,
                 Utilities::pow(3, dim)>
        cells;
      cells.fill(tria.end());

      const auto translate = [](const std::vector<unsigned int> &faces) {
        std::array<unsigned int, 3> index;
        index.fill(1);

        for (const auto face : faces)
          {
            const unsigned int direction = face / 2;
            const unsigned int side      = face % 2;

            index[direction] = (side == 1) ? 2 : 0;
          }

        if (dim == 1)
          return index[0];
        else if (dim == 2)
          return index[0] + 3 * index[1];
        else if (dim == 3)
          return index[0] + 3 * index[1] + 9 * index[2];
      };

      const auto set_entry = [&](const std::vector<unsigned int> &faces,
                                 const auto &                     value) {
        const unsigned int index = translate(faces);

        Assert(cells[index] == tria.end() || cells[index] == value,
               ExcInternalError());

        cells[index] = value;
      };

      set_entry({}, cell);

      if (level >= 1)
        {
          for (const auto f0 : cell->face_indices())
            if (cell->at_boundary(f0) == false)
              {
                const auto cell_neighbor = cell->neighbor(f0);

                set_entry({f0}, cell_neighbor);

                if (level >= 2)
                  for (const auto f1 : cell_neighbor->face_indices())
                    if ((f0 / 2 != f1 / 2) &&
                        (cell_neighbor->at_boundary(f1) == false))
                      {
                        const auto cell_neighbor_neigbor =
                          cell_neighbor->neighbor(f1);
                        set_entry({f0, f1}, cell_neighbor_neigbor);

                        if (level >= 3)
                          {
                            AssertThrow(false, ExcNotImplemented());
                          }
                      }
              }
        }

      return cells;
    }

  } // namespace GridTools
} // namespace dealii
