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

  } // namespace GridTools
} // namespace dealii
