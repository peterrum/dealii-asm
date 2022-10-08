#pragma once

#include <deal.II/fe/fe_tools.h>


namespace dealii
{
  namespace DoFTools
  {
    template <int dim>
    std::vector<types::global_dof_index>
    get_dof_indices_cell_with_overlap(
      const DoFHandler<dim> &                   dof_handler,
      const std::array<typename Triangulation<dim>::cell_iterator,
                       Utilities::pow(3, dim)> &cells,
      const unsigned int                        n_overlap  = 0,
      const bool                                return_all = false)
    {
      AssertDimension(dof_handler.get_fe_collection().size(), 1);

      const auto &fe = dof_handler.get_fe();

      AssertDimension(fe.n_components(), 1);

      const unsigned int fe_degree              = fe.tensor_degree();
      const unsigned int n_dofs_1D              = fe_degree + 1;
      const unsigned int n_dofs_with_overlap_1D = fe_degree - 1 + 2 * n_overlap;

      const auto lexicographic_to_hierarchic_numbering =
        Utilities::invert_permutation(
          FETools::hierarchic_to_lexicographic_numbering<dim>(fe_degree));

      const auto get_lexicographic_dof_indices = [&](const auto &cell) {
        const auto n_dofs_per_cell = fe.n_dofs_per_cell();

        std::vector<types::global_dof_index> indices_hierarchic(
          n_dofs_per_cell);
        std::vector<types::global_dof_index> indices_lexicographic(
          n_dofs_per_cell);

        (typename DoFHandler<dim>::cell_iterator(&cell->get_triangulation(),
                                                 cell->level(),
                                                 cell->index(),
                                                 &dof_handler))
          ->get_dof_indices(indices_hierarchic);

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          indices_lexicographic[i] =
            indices_hierarchic[lexicographic_to_hierarchic_numbering[i]];

        return indices_lexicographic;
      };

      if (n_overlap <= 1)
        {
          const auto dof_indices =
            get_lexicographic_dof_indices(cells[cells.size() / 2]);

          if (n_overlap == 1)
            return dof_indices;

          AssertDimension(dim, 2);

          std::vector<types::global_dof_index> inner_dof_indices;

          for (unsigned int j = 0, c = 0; j < n_dofs_1D; ++j)
            for (unsigned int i = 0; i < n_dofs_1D; ++i, ++c)
              if (i != 0 && i != fe_degree && j != 0 && j != fe_degree)
                inner_dof_indices.emplace_back(dof_indices[c]);
              else if (return_all)
                inner_dof_indices.emplace_back(numbers::invalid_unsigned_int);

          AssertDimension(inner_dof_indices.size(),
                          Utilities::pow(n_dofs_with_overlap_1D, dim));

          return inner_dof_indices;
        }
      else
        {
          std::array<std::vector<types::global_dof_index>,
                     Utilities::pow(3, dim)>
            all_dofs;

          for (unsigned int i = 0; i < cells.size(); ++i)
            if (cells[i].state() == IteratorState::valid)
              all_dofs[i] = get_lexicographic_dof_indices(cells[i]);

          std::vector<types::global_dof_index> dof_indices(
            Utilities::pow(n_dofs_with_overlap_1D, dim),
            numbers::invalid_unsigned_int);

          const auto translate =
            [&](const auto i,
                const bool flag) -> std::pair<unsigned int, unsigned int> {
            if (flag == false)
              return {0, 0};

            if (i < n_overlap - 1)
              return {0, fe_degree + 1 - n_overlap + i};
            else if (i < fe_degree + n_overlap)
              return {1, i - (n_overlap - 1)};
            else
              return {2, i - (n_overlap + fe_degree - 1)};
          };

          for (unsigned int k = 0, c = 0;
               k < (dim == 3 ? n_dofs_with_overlap_1D : 1);
               ++k)
            for (unsigned int j = 0;
                 j < (dim >= 2 ? n_dofs_with_overlap_1D : 1);
                 ++j)
              for (unsigned int i = 0; i < n_dofs_with_overlap_1D; ++i, ++c)
                {
                  const auto [i0, i1] = translate(i, dim >= 1);
                  const auto [j0, j1] = translate(j, dim >= 2);
                  const auto [k0, k1] = translate(k, dim >= 3);

                  const auto ii = i0 + 3 * j0 + 9 * k0;
                  const auto jj =
                    i1 + n_dofs_1D * j1 + n_dofs_1D * n_dofs_1D * k1;

                  if (cells[ii].state() == IteratorState::valid)
                    dof_indices[c] = all_dofs[ii][jj];
                }

          if (return_all)
            return dof_indices;

          std::vector<types::global_dof_index> dof_indices_cleaned;

          for (const auto i : dof_indices)
            if (i != numbers::invalid_unsigned_int)
              dof_indices_cleaned.push_back(i);

          return dof_indices_cleaned;
        }
    }

    template <int dim>
    unsigned int
    compute_shift_within_children(const unsigned int child,
                                  const unsigned int fe_shift_1d,
                                  const unsigned int fe_degree)
    {
      // we put the degrees of freedom of all child cells in lexicographic
      // ordering
      unsigned int c_tensor_index[dim];
      unsigned int tmp = child;
      for (unsigned int d = 0; d < dim; ++d)
        {
          c_tensor_index[d] = tmp % 2;
          tmp /= 2;
        }
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      unsigned int       factor          = 1;
      unsigned int       shift           = fe_shift_1d * c_tensor_index[0];
      for (unsigned int d = 1; d < dim; ++d)
        {
          factor *= n_child_dofs_1d;
          shift = shift + factor * fe_shift_1d * c_tensor_index[d];
        }
      return shift;
    }

    template <int dim>
    void
    get_child_offset(const unsigned int         child,
                     const unsigned int         fe_shift_1d,
                     const unsigned int         fe_degree,
                     std::vector<unsigned int> &local_dof_indices)
    {
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      const unsigned int shift =
        compute_shift_within_children<dim>(child, fe_shift_1d, fe_degree);
      const unsigned int n_components =
        local_dof_indices.size() / Utilities::fixed_power<dim>(fe_degree + 1);
      const unsigned int n_scalar_cell_dofs =
        Utilities::fixed_power<dim>(n_child_dofs_1d);
      for (unsigned int c = 0, m = 0; c < n_components; ++c)
        for (unsigned int k = 0; k < (dim > 2 ? (fe_degree + 1) : 1); ++k)
          for (unsigned int j = 0; j < (dim > 1 ? (fe_degree + 1) : 1); ++j)
            for (unsigned int i = 0; i < (fe_degree + 1); ++i, ++m)
              local_dof_indices[m] = c * n_scalar_cell_dofs +
                                     k * n_child_dofs_1d * n_child_dofs_1d +
                                     j * n_child_dofs_1d + i + shift;
    }

    template <int dim>
    std::vector<std::vector<unsigned int>>
    get_child_offsets(const unsigned int n_dofs_per_cell_coarse,
                      const unsigned int fe_shift_1d,
                      const unsigned int fe_degree)
    {
      std::vector<std::vector<unsigned int>> cell_local_chilren_indices(
        GeometryInfo<dim>::max_children_per_cell,
        std::vector<unsigned int>(n_dofs_per_cell_coarse));
      for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
           c++)
        get_child_offset<dim>(c,
                              fe_shift_1d,
                              fe_degree,
                              cell_local_chilren_indices[c]);
      return cell_local_chilren_indices;
    }

    template <int dim>
    std::vector<types::global_dof_index>
    get_dof_indices_vertex_patch(
      const DoFHandler<dim> &                   dof_handler,
      const std::array<typename Triangulation<dim>::cell_iterator,
                       Utilities::pow(2, dim)> &cells)
    {
      const unsigned int fe_degree = dof_handler.get_fe().degree;
      const unsigned int n_dofs_per_cell =
        dof_handler.get_fe().n_dofs_per_cell();
      const unsigned int n_dofs_1D_patch = fe_degree * 2 - 1;
      const unsigned int n_dofs_per_patch =
        Utilities::pow(n_dofs_1D_patch, dim);
      const unsigned int n_dofs_1D_patch_extended = fe_degree * 2 + 1;
      const unsigned int n_dofs_per_patch_extended =
        Utilities::pow(n_dofs_1D_patch_extended, dim);

      if (std::any_of(cells.begin(), cells.end(), [&](const auto &cell) {
            return cell == dof_handler.get_triangulation().end();
          }))
        return std::vector<types::global_dof_index>(n_dofs_per_patch,
                                                    numbers::invalid_dof_index);

      const auto map =
        FETools::hierarchic_to_lexicographic_numbering<dim>(fe_degree);

      std::vector<types::global_dof_index> result_all(
        n_dofs_per_patch_extended);

      std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices_lex(
        n_dofs_per_cell);

      const auto translate =
        get_child_offsets<dim>(n_dofs_per_cell, fe_degree, fe_degree);

      for (unsigned int c = 0; c < cells.size(); ++c)
        {
          typename DoFHandler<dim>::cell_iterator cell(
            &cells[c]->get_triangulation(),
            cells[c]->level(),
            cells[c]->index(),
            &dof_handler);

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            local_dof_indices_lex[map[i]] = local_dof_indices[i];

          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            result_all[translate[c][i]] = local_dof_indices_lex[i];
        }

      std::vector<types::global_dof_index> result;

      if (dim == 1)
        {
          for (unsigned int i = 0, c = 0; i < n_dofs_1D_patch_extended;
               ++i, ++c)
            {
              if (i == 0 || i == (n_dofs_1D_patch_extended - 1))
                continue;

              result.push_back(result_all[c]);
            }
        }
      else if (dim == 2)
        {
          for (unsigned int j = 0, c = 0; j < n_dofs_1D_patch_extended; ++j)
            for (unsigned int i = 0; i < n_dofs_1D_patch_extended; ++i, ++c)
              {
                if (i == 0 || i == (n_dofs_1D_patch_extended - 1) || j == 0 ||
                    j == (n_dofs_1D_patch_extended - 1))
                  continue;

                result.push_back(result_all[c]);
              }
        }
      else if (dim == 3)
        {
          for (unsigned int k = 0, c = 0; k < n_dofs_1D_patch_extended; ++k)
            for (unsigned int j = 0; j < n_dofs_1D_patch_extended; ++j)
              for (unsigned int i = 0; i < n_dofs_1D_patch_extended; ++i, ++c)
                {
                  if (i == 0 || i == (n_dofs_1D_patch_extended - 1) || j == 0 ||
                      j == (n_dofs_1D_patch_extended - 1) || k == 0 ||
                      k == (n_dofs_1D_patch_extended - 1))
                    continue;

                  result.push_back(result_all[c]);
                }
        }


      return result;
    }
  } // namespace DoFTools
} // namespace dealii
