#pragma once

#include <deal.II/lac/tensor_product_matrix.h>

namespace dealii
{
  template <int dim, typename Number, int n_rows_1d = -1>
  class MyTensorProductMatrixSymmetricSum
    : public TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>
  {
  public:
    const std::array<AlignedVector<Number>, dim> &
    get_eigenvalues() const
    {
      return this->eigenvalues;
    }

    const std::array<Table<2, Number>, dim> &
    get_eigenvectors() const
    {
      return this->eigenvectors;
    }

  private:
  };



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

  template <int dim, typename VectorizedArrayType, int n_rows_1d = -1>
  class MyTensorProductMatrixSymmetricSumCache
  {
  public:
    struct compartor
    {
      bool
      operator()(
        const MyTensorProductMatrixSymmetricSum<dim,
                                                VectorizedArrayType,
                                                n_rows_1d> &left,
        const MyTensorProductMatrixSymmetricSum<dim,
                                                VectorizedArrayType,
                                                n_rows_1d> &right) const
      {
        const auto &eigenvalues_0  = left.get_eigenvalues();
        const auto &eigenvectors_0 = left.get_eigenvectors();
        const auto &eigenvalues_1  = right.get_eigenvalues();
        const auto &eigenvectors_1 = right.get_eigenvectors();

        const unsigned int NN = VectorizedArrayType::size();

        std::array<bool, NN> mask;

        using Number = typename VectorizedArrayType::value_type;

        for (unsigned int v = 0; v < NN; ++v)
          {
            Number a = 0.0;
            Number b = 0.0;

            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int i = 0; i < eigenvalues_0.size(); ++i)
                {
                  a += eigenvalues_0[d][i][v];
                  b += eigenvalues_1[d][i][v];
                }

            mask[v] = (a != 0.0) && (b != 0.0);
          }


        const auto eps = std::pow<Number>(
          static_cast<Number>(10.0),
          static_cast<Number>(
            std::log10(std::numeric_limits<Number>::epsilon()) / 2.0));

        const auto less = [eps](const auto a, const auto b) -> bool {
          return (b - a) > std::max(eps, std::abs(a + b) * eps);
        };

        const auto greater = [eps](const auto a, const auto b) -> bool {
          return (a - b) > std::max(eps, std::abs(a + b) * eps);
        };

        for (unsigned int v = 0; v < NN; ++v)
          if (mask[v])
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int i = 0; i < eigenvalues_0[d].size(); ++i)
                if (less(eigenvalues_0[d][i][v], eigenvalues_1[d][i][v]))
                  return true;
                else if (greater(eigenvalues_0[d][i][v],
                                 eigenvalues_1[d][i][v]))
                  return false;

        for (unsigned int v = 0; v < NN; ++v)
          if (mask[v])
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int i = 0; i < eigenvectors_0[d].size(0); ++i)
                for (unsigned int j = 0; j < eigenvectors_0[d].size(1); ++j)
                  if (less(eigenvectors_0[d][i][j][v],
                           eigenvectors_1[d][i][j][v]))
                    return true;
                  else if (greater(eigenvectors_0[d][i][j][v],
                                   eigenvectors_1[d][i][j][v]))
                    return false;

        return false;
      }
    };

    void
    reserve(const unsigned int size)
    {
      vector.resize(size);
      indices.assign(size, numbers::invalid_unsigned_int);
    }

    void
    insert(const unsigned int                                    index,
           const std::array<Table<2, VectorizedArrayType>, dim> &Ms,
           const std::array<Table<2, VectorizedArrayType>, dim> &Ks)
    {
      MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>
        matrix;
      matrix.reinit(Ms, Ks);

      vector[index] = matrix;

      const auto ptr = cache.find(matrix);

      if (ptr != cache.end())
        indices[index] = ptr->second;
      else
        {
          indices[index] = compressed_vector.size();
          cache[matrix]  = compressed_vector.size();
          compressed_vector.emplace_back(matrix);
        }
    }

    const MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>
      &
      get(const unsigned int index) const
    {
      if (compressed_vector.size() > 0)
        return compressed_vector[indices[index]];

      return vector[index];
    }

    std::size_t
    memory_consumption() const
    {
      return MemoryConsumption::memory_consumption(vector);
    }

    std::size_t
    size() const
    {
      if (compressed_vector.size() != 0)
        return compressed_vector.size();

      return vector.size();
    }

    void
    finalize()
    {
      cache.clear();

      if (compressed_vector.size() == vector.size())
        {
          indices.clear();
          compressed_vector.clear();
        }
    }

  private:
    std::map<
      MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>,
      unsigned int,
      compartor>
      cache;

    std::vector<unsigned int> indices;

    std::vector<
      MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>>
      compressed_vector;

    std::vector<
      MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>>
      vector;
  };

} // namespace dealii
