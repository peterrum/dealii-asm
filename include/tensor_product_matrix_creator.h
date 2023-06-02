#pragma once

#include <deal.II/numerics/tensor_product_matrix_creator.h>

namespace dealii::TensorProductMatrixCreator
{
  template <int dim, typename Number>
  std::pair<std::array<FullMatrix<Number>, dim>,
            std::array<FullMatrix<Number>, dim>>
  create_laplace_tensor_product_matrix(
    const FiniteElement<1> &               fe,
    const Quadrature<1> &                  quadrature,
    const dealii::ndarray<double, dim, 2> &cell_extent)
  {
    // 1) create element mass and siffness matrix (without overlap)
    const auto create_reference_mass_and_stiffness_matrices =
      internal::create_reference_mass_and_stiffness_matrices<Number>(
        fe, quadrature);

    const auto &M_ref =
      std::get<0>(create_reference_mass_and_stiffness_matrices);
    const auto &K_ref =
      std::get<1>(create_reference_mass_and_stiffness_matrices);
    const auto &is_dg =
      std::get<2>(create_reference_mass_and_stiffness_matrices);

    AssertThrow(is_dg == false, ExcNotImplemented());

    // 2) loop over all dimensions and create 1D mass and stiffness
    // matrices so that boundary conditions and overlap are considered

    const unsigned int n_dofs_1D              = M_ref.n();
    const unsigned int n_dofs_1D_with_overlap = 2 * (n_dofs_1D - 1) - 1;

    std::array<FullMatrix<Number>, dim> Ms;
    std::array<FullMatrix<Number>, dim> Ks;

    for (unsigned int d = 0; d < dim; ++d)
      {
        Ms[d].reinit(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);
        Ks[d].reinit(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);

        for (unsigned int i = 0; i < (n_dofs_1D - 1); ++i)
          for (unsigned int j = 0; j < (n_dofs_1D - 1); ++j)
            {
              Ms[d][i][j] += M_ref[i + 1][j + 1] * cell_extent[d][0];
              Ks[d][i][j] += K_ref[i + 1][j + 1] / cell_extent[d][0];
            }

        for (unsigned int i = 0; i < (n_dofs_1D - 1); ++i)
          for (unsigned int j = 0; j < (n_dofs_1D - 1); ++j)
            {
              Ms[d][i + (n_dofs_1D - 2)][j + (n_dofs_1D - 2)] +=
                M_ref[i][j] * cell_extent[d][1];
              Ks[d][i + (n_dofs_1D - 2)][j + (n_dofs_1D - 2)] +=
                K_ref[i][j] / cell_extent[d][1];
            }
      }

    return {Ms, Ks};
  }


} // namespace dealii::TensorProductMatrixCreator
