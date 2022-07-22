#pragma once

#include <deal.II/lac/tensor_product_matrix.h>

namespace dealii
{
  template <int dim, typename Number>
  class MyTensorProductMatrixSymmetricSum
    : public TensorProductMatrixSymmetricSum<dim, Number, -1>
  {
  public:
    void
    internal_reinit(const std::array<Table<2, Number>, dim> mass_matrix,
                    const std::array<Table<2, Number>, dim> derivative_matrix,
                    const std::array<Table<2, Number>, dim> eigenvectors,
                    const std::array<AlignedVector<Number>, dim> eigenvalues,
                    const std::array<AlignedVector<Number>, dim> masks)
    {
      this->mass_matrix       = mass_matrix;
      this->derivative_matrix = derivative_matrix;
      this->eigenvectors      = eigenvectors;
      this->eigenvalues       = eigenvalues;
      this->masks             = masks;
    }

    void
    set_mask(const std::array<AlignedVector<Number>, dim> masks)
    {
      this->masks = masks;

      const unsigned int n_dofs_1D = this->eigenvalues[0].size();

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < n_dofs_1D; ++i)
          for (unsigned int j = 0; j < n_dofs_1D; ++j)
            this->eigenvectors[d][i][j] *= masks[d][i];
    }

    void
    apply_inverse(const ArrayView<Number> &      dst,
                  const ArrayView<const Number> &src) const
    {
      TensorProductMatrixSymmetricSum<dim, Number, -1>::apply_inverse(dst, src);

      const unsigned int n = this->eigenvalues[0].size();

      if (dim == 2)
        {
          for (unsigned int i1 = 0, c = 0; i1 < n; ++i1)
            for (unsigned int i0 = 0; i0 < n; ++i0, ++c)
              {
                const auto mask = masks[1][i1] * masks[0][i0];
                dst[c]          = mask * dst[c] + (Number(1) - mask) * src[c];
              }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    template <std::size_t N>
    static MyTensorProductMatrixSymmetricSum<dim, VectorizedArray<Number, N>>
    transpose(
      const std::array<MyTensorProductMatrixSymmetricSum<dim, Number>, N> &in)
    {
      std::array<Table<2, VectorizedArray<Number, N>>, dim> mass_matrix;
      std::array<Table<2, VectorizedArray<Number, N>>, dim> derivative_matrix;
      std::array<Table<2, VectorizedArray<Number, N>>, dim> eigenvectors;
      std::array<AlignedVector<VectorizedArray<Number, N>>, dim> eigenvalues;
      std::array<AlignedVector<VectorizedArray<Number, N>>, dim> masks;

      for (unsigned int d = 0; d < dim; ++d)
        {
          // allocate memory
          mass_matrix[d].reinit(in[0].mass_matrix[d].size(0),
                                in[0].mass_matrix[d].size(1));
          derivative_matrix[d].reinit(in[0].derivative_matrix[d].size(0),
                                      in[0].derivative_matrix[d].size(1));
          eigenvectors[d].reinit(in[0].eigenvectors[d].size(0),
                                 in[0].eigenvectors[d].size(1));
          eigenvalues[d].resize(in[0].eigenvalues.size());
          masks[d].resize(in[0].masks.size());

          // do actual transpose
          for (unsigned int v = 0; v < N; ++v)
            for (unsigned int i = 0; i < in[0].mass_matrix[d].size(0); ++i)
              for (unsigned int j = 0; j < in[0].mass_matrix[d].size(1); ++j)
                mass_matrix[d][i][j][v] = in[v].mass_matrix[d][i][j];

          for (unsigned int v = 0; v < N; ++v)
            for (unsigned int i = 0; i < in[0].derivative_matrix[d].size(0);
                 ++i)
              for (unsigned int j = 0; j < in[0].derivative_matrix[d].size(1);
                   ++j)
                derivative_matrix[d][i][j][v] =
                  in[v].derivative_matrix[d][i][j];

          for (unsigned int v = 0; v < N; ++v)
            for (unsigned int i = 0; i < in[0].eigenvectors[d].size(0); ++i)
              for (unsigned int j = 0; j < in[0].eigenvectors[d].size(1); ++j)
                eigenvectors[d][i][j][v] = in[v].eigenvectors[d][i][j];

          for (unsigned int v = 0; v < N; ++v)
            for (unsigned int i = 0; i < in[0].eigenvalues.size(); ++i)
              eigenvalues[d][i][v] = in[v].eigenvalues[d][i];

          for (unsigned int v = 0; v < N; ++v)
            for (unsigned int i = 0; i < in[0].masks.size(); ++i)
              masks[d][i][v] = in[v].masks[d][i];
        }

      MyTensorProductMatrixSymmetricSum<dim, VectorizedArray<Number, N>> out;
      out.internal_reinit(
        mass_matrix, derivative_matrix, eigenvectors, eigenvalues, masks);

      return out;
    }

  private:
    std::array<AlignedVector<Number>, dim> masks;
  };

} // namespace dealii
