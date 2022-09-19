#pragma once

#include <deal.II/lac/tensor_product_matrix.h>

namespace dealii
{
  template <typename Number>
  std::tuple<FullMatrix<Number>, FullMatrix<Number>, bool>
  create_referece_cell_matrices(const FiniteElement<1> &fe,
                                const Quadrature<1> &   quadrature)
  {
    Triangulation<1> tria;
    GridGenerator::hyper_cube(tria);

    DoFHandler<1> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ1<1> mapping;

    const unsigned int n_dofs_1D = fe.n_dofs_per_cell();

    FullMatrix<Number> mass_matrix_reference(n_dofs_1D, n_dofs_1D);
    FullMatrix<Number> derivative_matrix_reference(n_dofs_1D, n_dofs_1D);

    FEValues<1> fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

    fe_values.reinit(tria.begin());

    const auto lexicographic_to_hierarchic_numbering =
      Utilities::invert_permutation(
        FETools::hierarchic_to_lexicographic_numbering<1>(fe.tensor_degree()));

    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          {
            mass_matrix_reference(i, j) +=
              (fe_values.shape_value(lexicographic_to_hierarchic_numbering[i],
                                     q_index) *
               fe_values.shape_value(lexicographic_to_hierarchic_numbering[j],
                                     q_index) *
               fe_values.JxW(q_index));

            derivative_matrix_reference(i, j) +=
              (fe_values.shape_grad(lexicographic_to_hierarchic_numbering[i],
                                    q_index) *
               fe_values.shape_grad(lexicographic_to_hierarchic_numbering[j],
                                    q_index) *
               fe_values.JxW(q_index));
          }

    return {mass_matrix_reference, derivative_matrix_reference, false};
  }


  template <int dim, typename Number>
  std::pair<std::array<FullMatrix<Number>, dim>,
            std::array<FullMatrix<Number>, dim>>
  create_laplace_tensor_product_matrix(
    const typename Triangulation<dim>::cell_iterator &cell,
    const FiniteElement<1> &                          fe,
    const Quadrature<1> &                             quadrature,
    const dealii::ndarray<double, dim, 3> &           cell_extend,
    const unsigned int                                n_overlap)
  {
    // 1) create element mass and siffness matrix (without overlap)
    const auto [M_ref, K_ref, is_dg] =
      create_referece_cell_matrices<Number>(fe, quadrature);

    AssertIndexRange(n_overlap, M_ref.n());
    AssertIndexRange(0, n_overlap);
    AssertThrow(is_dg == false, ExcNotImplemented());

    // 2) loop over all dimensions and create 1D mass and stiffness
    // matrices so that boundary conditions and overlap are considered

    const unsigned int n_dofs_1D              = M_ref.n();
    const unsigned int n_dofs_1D_with_overlap = M_ref.n() - 2 + 2 * n_overlap;

    std::array<FullMatrix<Number>, dim> Ms;
    std::array<FullMatrix<Number>, dim> Ks;

    const auto clear_row_and_column = [&](const unsigned int n, auto &matrix) {
      for (unsigned int i = 0; i < n_dofs_1D_with_overlap; ++i)
        {
          matrix[i][n] = 0.0;
          matrix[n][i] = 0.0;
        }
    };

    for (unsigned int d = 0; d < dim; ++d)
      {
        FullMatrix<Number> M(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);
        FullMatrix<Number> K(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);

        // inner cell
        for (unsigned int i = 0; i < n_dofs_1D; ++i)
          for (unsigned int j = 0; j < n_dofs_1D; ++j)
            {
              const unsigned int i0 = i + n_overlap - 1;
              const unsigned int j0 = j + n_overlap - 1;
              M[i0][j0]             = M_ref[i][j] * cell_extend[d][1];
              K[i0][j0]             = K_ref[i][j] / cell_extend[d][1];
            }

        // left neighbor or left boundary
        if ((cell->at_boundary(2 * d) == false) ||
            cell->has_periodic_neighbor(2 * d))
          {
            // left neighbor
            Assert(cell_extend[d][0] > 0.0, ExcInternalError());

            for (unsigned int i = 0; i < n_overlap; ++i)
              for (unsigned int j = 0; j < n_overlap; ++j)
                {
                  const unsigned int i0 = n_dofs_1D - n_overlap + i;
                  const unsigned int j0 = n_dofs_1D - n_overlap + j;
                  M[i][j] += M_ref[i0][j0] * cell_extend[d][0];
                  K[i][j] += K_ref[i0][j0] / cell_extend[d][0];
                }
          }
        else
          {
            const auto bid = cell->face(2 * d)->boundary_id();
            if (bid == 1 /*DBC*/)
              {
                // left DBC
                const unsigned i0 = n_overlap - 1;
                clear_row_and_column(i0, M);
                clear_row_and_column(i0, K);
              }
            else if (bid == 2 /*NBC*/)
              {
                // left NBC -> nothing to do
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }

        // reight neighbor or right boundary
        if ((cell->at_boundary(2 * d + 1) == false) ||
            cell->has_periodic_neighbor(2 * d + 1))
          {
            Assert(cell_extend[d][2] > 0.0, ExcInternalError());

            for (unsigned int i = 0; i < n_overlap; ++i)
              for (unsigned int j = 0; j < n_overlap; ++j)
                {
                  const unsigned int i0 = n_overlap + n_dofs_1D + i - 2;
                  const unsigned int j0 = n_overlap + n_dofs_1D + j - 2;
                  M[i0][j0] += M_ref[i][j] * cell_extend[d][2];
                  K[i0][j0] += K_ref[i][j] / cell_extend[d][2];
                }
          }
        else
          {
            const auto bid = cell->face(2 * d + 1)->boundary_id();
            if (bid == 1 /*DBC*/)
              {
                // right DBC
                const unsigned i0 = n_overlap + n_dofs_1D - 2;
                clear_row_and_column(i0, M);
                clear_row_and_column(i0, K);
              }
            else if (bid == 2 /*NBC*/)
              {
                // right NBC -> nothing to do
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }

        // set zero diagonal entries to one so that the matrices are
        // invertible; we will ignore those entries with masking; zero
        // diagonal entries might be due to 1) DBC and 2) overlap into
        // a non-existent cell
        for (unsigned int i = 0; i < n_dofs_1D_with_overlap; ++i)
          if (K[i][i] == 0.0)
            {
              Assert(M[i][i] == 0.0, ExcInternalError());

              M[i][i] = 1.0;
              K[i][i] = 1.0;
            }

        Ms[d] = M;
        Ks[d] = K;
      }

    return {Ms, Ks};
  }



  namespace internal
  {
    namespace TensorProductMatrixSymmetricSum
    {
      template <typename Number>
      struct MatrixPairComparator
      {
        using MatrixPairType = std::pair<Table<2, Number>, Table<2, Number>>;

        bool
        operator()(const MatrixPairType &left,
                   const MatrixPairType &right) const
        {
          const auto &M_0 = left.first;
          const auto &K_0 = left.second;
          const auto &M_1 = right.first;
          const auto &K_1 = right.second;

          const unsigned int n_lanes = Number::size();

          std::array<bool, n_lanes> mask;

          using NumberScalar = typename Number::value_type;

          for (unsigned int v = 0; v < n_lanes; ++v)
            {
              NumberScalar a = 0.0;
              NumberScalar b = 0.0;

              for (unsigned int i = 0; i < M_0.size(0); ++i)
                for (unsigned int j = 0; j < M_0.size(1); ++j)
                  {
                    a += std::abs(M_0[i][j][v]);
                    b += std::abs(M_1[i][j][v]);
                  }

              mask[v] = (a != 0.0) && (b != 0.0);
            }

          const auto eps = std::pow<NumberScalar>(
            static_cast<NumberScalar>(10.0),
            static_cast<NumberScalar>(
              std::log10(std::numeric_limits<NumberScalar>::epsilon()) / 2.0));

          const auto less = [eps](const auto a, const auto b) -> bool {
            return (b - a) > std::max(eps, std::abs(a + b) * eps);
          };

          const auto greater = [eps](const auto a, const auto b) -> bool {
            return (a - b) > std::max(eps, std::abs(a + b) * eps);
          };

          for (unsigned int v = 0; v < n_lanes; ++v)
            if (mask[v])
              for (unsigned int i = 0; i < M_0.size(0); ++i)
                for (unsigned int j = 0; j < M_0.size(1); ++j)
                  if (less(M_0[i][j][v], M_1[i][j][v]))
                    return true;
                  else if (greater(M_0[i][j][v], M_1[i][j][v]))
                    return false;

          for (unsigned int v = 0; v < n_lanes; ++v)
            if (mask[v])
              for (unsigned int i = 0; i < K_0.size(0); ++i)
                for (unsigned int j = 0; j < K_0.size(1); ++j)
                  if (less(K_0[i][j][v], K_1[i][j][v]))
                    return true;
                  else if (greater(K_0[i][j][v], K_1[i][j][v]))
                    return false;

          return false;
        }
      };
    } // namespace TensorProductMatrixSymmetricSum
  }   // namespace internal



  template <int dim, typename Number, int n_rows_1d = -1>
  class TensorProductMatrixSymmetricSumCache
  {
    using MatrixPairType = std::pair<Table<2, Number>, Table<2, Number>>;

  public:
    void
    reserve(const unsigned int size)
    {
      indices.assign(size * dim, numbers::invalid_unsigned_int);
    }

    void
    insert(const unsigned int                       index,
           const std::array<Table<2, Number>, dim> &Ms,
           const std::array<Table<2, Number>, dim> &Ks)
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          const MatrixPairType matrix(Ms[d], Ks[d]);

          const auto ptr = cache.find(matrix);

          if (ptr != cache.end())
            indices[index * dim + d] = ptr->second;
          else
            {
              indices[index * dim + d] = cache.size();
              cache[matrix]            = cache.size();
            }
        }
    }

    void
    apply_inverse(const unsigned int             index,
                  const ArrayView<Number> &      dst_in,
                  const ArrayView<const Number> &src_in,
                  AlignedVector<Number> &        tmp_array) const
    {
      Number *      dst = dst_in.begin();
      const Number *src = src_in.begin();

      std::array<const Number *, dim> eigenvectors, eigenvalues;
      unsigned int                    n_rows_1d_non_templated = 0;

      for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int translated_index =
            (indices.size() > 0) ? indices[dim * index + d] : (dim * index + d);

          eigenvectors[d] = &vector_eigenvectors[translated_index](0, 0);
          eigenvalues[d]  = vector_eigenvalues[translated_index].data();
          n_rows_1d_non_templated = vector_eigenvalues[translated_index].size();
        }

      if (n_rows_1d != -1)
        internal::TensorProductMatrixSymmetricSum::apply_inverse<
          n_rows_1d == -1 ? 0 : n_rows_1d>(dst,
                                           src,
                                           tmp_array,
                                           n_rows_1d_non_templated,
                                           eigenvectors,
                                           eigenvalues);
      else
        internal::TensorProductMatrixSymmetricSum::select_apply_inverse<1>(
          dst,
          src,
          tmp_array,
          n_rows_1d_non_templated,
          eigenvectors,
          eigenvalues);
    }

    std::size_t
    memory_consumption() const
    {
      return MemoryConsumption::memory_consumption(indices) +
             MemoryConsumption::memory_consumption(vector_mass_matrix) +
             MemoryConsumption::memory_consumption(vector_derivative_matrix) +
             MemoryConsumption::memory_consumption(vector_eigenvectors) +
             MemoryConsumption::memory_consumption(vector_eigenvalues);
    }

    std::size_t
    size() const
    {
      return vector_mass_matrix.size();
    }

    void
    finalize()
    {
      vector_mass_matrix.resize(cache.size());
      vector_derivative_matrix.resize(cache.size());
      vector_eigenvectors.resize(cache.size());
      vector_eigenvalues.resize(cache.size());

      const auto store = [&](const unsigned int    index,
                             const MatrixPairType &M_and_K) {
        std::array<Table<2, Number>, 1> mass_matrices;
        mass_matrices[0] = M_and_K.first;

        std::array<Table<2, Number>, 1> derivative_matrices;
        derivative_matrices[0] = M_and_K.second;

        std::array<Table<2, Number>, 1>      eigenvectors;
        std::array<AlignedVector<Number>, 1> eigenvalues;

        internal::TensorProductMatrixSymmetricSum::setup(mass_matrices,
                                                         derivative_matrices,
                                                         eigenvectors,
                                                         eigenvalues);

        vector_mass_matrix[index]       = M_and_K.first;
        vector_derivative_matrix[index] = M_and_K.second;
        vector_eigenvectors[index]      = eigenvectors[0];
        vector_eigenvalues[index]       = eigenvalues[0];
      };

      if (cache.size() == indices.size())
        {
          std::map<unsigned int, MatrixPairType> inverted_cache;

          for (const auto &i : cache)
            inverted_cache[i.second] = i.first;

          for (unsigned int i = 0; i < indices.size(); ++i)
            store(i, inverted_cache[indices[i]]);

          indices.clear();
        }
      else
        {
          for (const auto &i : cache)
            store(i.second, i.first);
        }

      cache.clear();
    }

  private:
    std::map<
      MatrixPairType,
      unsigned int,
      internal::TensorProductMatrixSymmetricSum::MatrixPairComparator<Number>>
      cache;

    std::vector<unsigned int> indices;

    std::vector<Table<2, Number>>      vector_mass_matrix;
    std::vector<Table<2, Number>>      vector_derivative_matrix;
    std::vector<Table<2, Number>>      vector_eigenvectors;
    std::vector<AlignedVector<Number>> vector_eigenvalues;
  };

} // namespace dealii
