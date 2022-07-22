#pragma once

namespace dealii
{
  template <int dim, typename Number>
  class MyTensorProductMatrixSymmetricSum
    : public TensorProductMatrixSymmetricSum<dim, Number, -1>
  {
  public:
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

  private:
    std::array<AlignedVector<Number>, dim> masks;
  };

} // namespace dealii
