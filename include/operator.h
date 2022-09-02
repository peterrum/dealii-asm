#pragma once

#include "vector_access_reduced.h"

template <int dim, typename Number>
class LaplaceOperatorMatrixBased : public Subscriptor
{
public:
  static const int dimension = dim;
  using value_type           = Number;
  using vector_type          = LinearAlgebra::distributed::Vector<Number>;

  using VectorType = vector_type;

  LaplaceOperatorMatrixBased(const Mapping<dim> &      mapping,
                             const Triangulation<dim> &tria,
                             const FiniteElement<dim> &fe,
                             const Quadrature<dim> &   quadrature)
    : mapping(mapping)
    , dof_handler(tria)
    , quadrature(quadrature)
  {
    dof_handler.distribute_dofs(fe);

    DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
    constraints.close();

    // create system matrix
    sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                            dof_handler.get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    false);
    sparsity_pattern.compress();

    sparse_matrix.reinit(sparsity_pattern);

    MatrixCreator::
      create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
        mapping, dof_handler, quadrature, sparse_matrix, nullptr, constraints);

    partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());
  }

  static constexpr bool
  is_matrix_free()
  {
    return false;
  }

  types::global_dof_index
  m() const
  {
    return dof_handler.n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    sparse_matrix.vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    sparse_matrix.Tvmult(dst, src);
  }

  const Mapping<dim> &
  get_mapping() const
  {
    return mapping;
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return dof_handler.get_fe();
  }

  const Triangulation<dim> &
  get_triangulation() const
  {
    return dof_handler.get_triangulation();
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    return sparse_matrix;
  }

  const TrilinosWrappers::SparsityPattern &
  get_sparsity_pattern() const
  {
    return sparsity_pattern;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(partitioner);
  }

  const AffineConstraints<Number> &
  get_constraints() const
  {
    return constraints;
  }

  const Quadrature<dim> &
  get_quadrature() const
  {
    return quadrature;
  }

  void
  compute_inverse_diagonal(VectorType &vec) const
  {
    this->initialize_dof_vector(vec);

    for (const auto entry : sparse_matrix)
      if (entry.row() == entry.column())
        vec[entry.row()] = 1.0 / entry.value();
  }

private:
  const Mapping<dim> &              mapping;
  DoFHandler<dim>                   dof_handler;
  TrilinosWrappers::SparseMatrix    sparse_matrix;
  TrilinosWrappers::SparsityPattern sparsity_pattern;
  AffineConstraints<Number>         constraints;
  Quadrature<dim>                   quadrature;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
};


template <int dim,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
class LaplaceOperatorMatrixFree : public Subscriptor
{
public:
  static const int dimension  = dim;
  using value_type            = Number;
  using vectorized_array_type = VectorizedArrayType;
  using vector_type           = LinearAlgebra::distributed::Vector<Number>;

  using VectorType = vector_type;

  static const unsigned int n_components = 1;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

  LaplaceOperatorMatrixFree(const Mapping<dim> &      mapping,
                            const Triangulation<dim> &tria,
                            const FiniteElement<dim> &fe,
                            const Quadrature<dim> &   quadrature)
    : dof_handler_internal(tria)
    , matrix_free(matrix_free_internal)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(
              dof_handler_internal.get_communicator()) == 0)
  {
    dof_handler_internal.distribute_dofs(fe);

    DoFTools::make_zero_boundary_constraints(dof_handler_internal,
                                             1,
                                             constraints_internal);
    constraints_internal.close();

    matrix_free_internal.reinit(mapping,
                                dof_handler_internal,
                                constraints_internal,
                                quadrature);

    pcout << "- Create operator:" << std::endl;
    pcout << "  - n cells: "
          << dof_handler_internal.get_triangulation().n_global_active_cells()
          << std::endl;
    pcout << "  - n dofs:  " << dof_handler_internal.n_dofs() << std::endl;
    pcout << std::endl;

    setup_mapping_and_indices();
  }

  LaplaceOperatorMatrixFree(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(
              matrix_free.get_dof_handler().get_communicator()) == 0)
  {
    setup_mapping_and_indices();
  }

  void
  setup_mapping_and_indices()
  {
    const bool compress_indices = true; // TODO

    if (compress_indices)
      {
        auto compressed_rw = std::make_shared<ConstraintInfoReduced>();
        compressed_rw->initialize(matrix_free);
        this->compressed_rw = compressed_rw;
      }

    const std::string mapping_type = ""; // TODO


    // lineare geometry
    if (mapping_type == "linear geometry")
      {
        // adopted from
        // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L194-L266

        cell_vertex_coefficients.resize(matrix_free.n_cell_batches());

        FE_Nothing<dim> dummy_fe;

        const auto &mapping   = *matrix_free.get_mapping_info().mapping;
        const auto  mapping_q = dynamic_cast<const MappingQ<dim> *>(&mapping);

        AssertThrow(mapping_q,
                    ExcMessage(
                      "This function expects a mapping of type MappingQ!"));
        AssertThrow(mapping_q->get_degree() == 1,
                    ExcMessage("This function expects a linear mapping!"));

        FEValues<dim> fe_values(mapping,
                                dummy_fe,
                                QGaussLobatto<dim>(2),
                                update_quadrature_points);

        for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
          {
            unsigned int l = 0;

            for (; l < matrix_free.n_active_entries_per_cell_batch(c); ++l)
              {
                fe_values.reinit(typename Triangulation<dim>::cell_iterator(
                  matrix_free.get_cell_iterator(c, l)));

                const auto &v = fe_values.get_quadrature_points();

                if (dim == 2)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        cell_vertex_coefficients[c][0][d][l] = v[0][d];
                        cell_vertex_coefficients[c][1][d][l] =
                          v[1][d] - v[0][d];
                        cell_vertex_coefficients[c][2][d][l] =
                          v[2][d] - v[0][d];
                        cell_vertex_coefficients[c][3][d][l] =
                          v[3][d] - v[2][d] - (v[1][d] - v[0][d]);
                      }
                  }
                else if (dim == 3)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        cell_vertex_coefficients[c][0][d][l] = v[0][d];
                        cell_vertex_coefficients[c][1][d][l] =
                          v[1][d] - v[0][d];
                        cell_vertex_coefficients[c][2][d][l] =
                          v[2][d] - v[0][d];
                        cell_vertex_coefficients[c][3][d][l] =
                          v[4][d] - v[0][d];
                        cell_vertex_coefficients[c][4][d][l] =
                          v[3][d] - v[2][d] - (v[1][d] - v[0][d]);
                        cell_vertex_coefficients[c][5][d][l] =
                          v[5][d] - v[4][d] - (v[1][d] - v[0][d]);
                        cell_vertex_coefficients[c][6][d][l] =
                          v[6][d] - v[4][d] - (v[2][d] - v[0][d]);
                        cell_vertex_coefficients[c][7][d][l] =
                          (v[7][d] - v[6][d] - (v[5][d] - v[4][d]) -
                           (v[3][d] - v[2][d] - (v[1][d] - v[0][d])));
                      }
                  }
                else
                  AssertThrow(false, ExcNotImplemented());
              }

            for (; l < VectorizedArrayType::size(); ++l)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  cell_vertex_coefficients[c][d + 1][d][l] = 1.;
              }
          }
      }

    if (mapping_type == "quadratic geometry")
      {
        // adopted from
        // https://github.com/kronbichler/mf_data_locality/blob/de47ea43e7e705a71742885493a2f5c441824a73/common_code/poisson_operator.h#L136-L169

        cell_quadratic_coefficients.resize(matrix_free.n_cell_batches());

        FE_Nothing<dim> dummy_fe;

        const auto &mapping   = *matrix_free.get_mapping_info().mapping;
        const auto  mapping_q = dynamic_cast<const MappingQ<dim> *>(&mapping);

        AssertThrow(mapping_q,
                    ExcMessage(
                      "This function expects a mapping of type MappingQ!"));
        AssertThrow(mapping_q->get_degree() <= 2,
                    ExcMessage(
                      "This function expects at most a quadratic mapping!"));

        FEValues<dim> fe_values(mapping,
                                dummy_fe,
                                QGaussLobatto<dim>(3),
                                update_quadrature_points);

        for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
          {
            unsigned int l = 0;

            for (; l < matrix_free.n_active_entries_per_cell_batch(c); ++l)
              {
                fe_values.reinit(typename Triangulation<dim>::cell_iterator(
                  matrix_free.get_cell_iterator(c, l)));

                const double coeff[9] = {
                  1.0, -3.0, 2.0, 0.0, 4.0, -4.0, 0.0, -1.0, 2.0};
                constexpr unsigned int size_dim = Utilities::pow(3, dim);
                std::array<Tensor<1, dim>, size_dim> points;
                for (unsigned int i2 = 0; i2 < (dim > 2 ? 3 : 1); ++i2)
                  {
                    for (unsigned int i1 = 0; i1 < 3; ++i1)
                      for (unsigned int i0 = 0, i = 9 * i2 + 3 * i1; i0 < 3;
                           ++i0)
                        points[i + i0] =
                          coeff[i0] * fe_values.quadrature_point(i) +
                          coeff[i0 + 3] * fe_values.quadrature_point(i + 1) +
                          coeff[i0 + 6] * fe_values.quadrature_point(i + 2);
                    for (unsigned int i1 = 0; i1 < 3; ++i1)
                      {
                        const unsigned int            i   = 9 * i2 + i1;
                        std::array<Tensor<1, dim>, 3> tmp = {
                          {points[i], points[i + 3], points[i + 6]}};
                        for (unsigned int i0 = 0; i0 < 3; ++i0)
                          points[i + 3 * i0] = coeff[i0] * tmp[0] +
                                               coeff[i0 + 3] * tmp[1] +
                                               coeff[i0 + 6] * tmp[2];
                      }
                  }
                if (dim == 3)
                  for (unsigned int i = 0; i < 9; ++i)
                    {
                      std::array<Tensor<1, dim>, 3> tmp = {
                        {points[i], points[i + 9], points[i + 18]}};
                      for (unsigned int i0 = 0; i0 < 3; ++i0)
                        points[i + 9 * i0] = coeff[i0] * tmp[0] +
                                             coeff[i0 + 3] * tmp[1] +
                                             coeff[i0 + 6] * tmp[2];
                    }
                for (unsigned int i = 0; i < points.size(); ++i)
                  for (unsigned int d = 0; d < dim; ++d)
                    cell_quadratic_coefficients[c][i][d][l] = points[i][d];
              }

            for (; l < VectorizedArrayType::size(); ++l)
              {
                cell_quadratic_coefficients[c][1][0][l] = 1.;
                if (dim > 1)
                  cell_quadratic_coefficients[c][3][1][l] = 1.;
                if (dim > 2)
                  cell_quadratic_coefficients[c][9][2][l] = 1.;
              }
          }
      }

    if (mapping_type == "merged")
      {
        // adpoted from
        // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L272-L277

        const auto &quadrature = matrix_free.get_quadrature();
        const auto  n_q_points = quadrature.size();
        merged_coefficients.resize(n_q_points * matrix_free.n_cell_batches());

        FE_Nothing<dim> dummy_fe;
        FEValues<dim>   fe_values(*matrix_free.get_mapping_info().mapping,
                                dummy_fe,
                                quadrature,
                                update_jacobians | update_JxW_values);

        for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
          {
            for (unsigned int l = 0;
                 l < matrix_free.n_active_entries_per_cell_batch(c);
                 ++l)
              {
                fe_values.reinit(typename Triangulation<dim>::cell_iterator(
                  matrix_free.get_cell_iterator(c, l)));
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    const Tensor<2, dim> merged_coefficient =
                      fe_values.JxW(q) *
                      (invert(Tensor<2, dim>(fe_values.jacobian(q))) *
                       transpose(
                         invert(Tensor<2, dim>(fe_values.jacobian(q)))));
                    for (unsigned int d = 0, cc = 0; d < dim; ++d)
                      for (unsigned int e = d; e < dim; ++e, ++cc)
                        merged_coefficients[c * n_q_points + q][cc][l] =
                          merged_coefficient[d][e];
                  }
              }
          }
      }

    if (mapping_type == "generate q")
      {
        // adopted from
        // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L278-L280

        const auto &quadrature = matrix_free.get_quadrature();
        const auto  n_q_points = quadrature.size();
        quadrature_points.resize(n_q_points * dim *
                                 matrix_free.n_cell_batches());

        FE_Nothing<dim> dummy_fe;

        FEValues<dim> fe_values(*matrix_free.get_mapping_info().mapping,
                                dummy_fe,
                                quadrature,
                                update_quadrature_points);

        for (unsigned int c = 0; c < matrix_free.n_cell_batches(); ++c)
          {
            for (unsigned int l = 0;
                 l < matrix_free.n_active_entries_per_cell_batch(c);
                 ++l)
              {
                fe_values.reinit(typename Triangulation<dim>::cell_iterator(
                  matrix_free.get_cell_iterator(c, l)));
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      quadrature_points[c * dim * n_q_points + d * n_q_points +
                                        q][l] =
                        fe_values.quadrature_point(q)[d];
                  }
              }
          }
      }
  }

  static constexpr bool
  is_matrix_free()
  {
    return true;
  }

  const MatrixFree<dim, Number, VectorizedArrayType> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  types::global_dof_index
  m() const
  {
    return get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  set_partitioner(
    std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner) const
  {
    if ((vector_partitioner.get() == nullptr) ||
        (vector_partitioner.get() ==
         matrix_free.get_vector_partitioner().get()))
      return; // nothing to do

    constraint_info.reinit(matrix_free.n_physical_cells());

    const auto locally_owned_indices =
      vector_partitioner->locally_owned_range();
    std::vector<types::global_dof_index> relevant_dofs;

    cell_ptr = {0};
    for (unsigned int cell = 0, cell_counter = 0;
         cell < matrix_free.n_cell_batches();
         ++cell)
      {
        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v, ++cell_counter)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

            const auto cells =
              dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                cell_iterator, 0);

            auto dofs = dealii::DoFTools::get_dof_indices_cell_with_overlap(
              get_dof_handler(), cells, 1, false);

            for (auto &dof : dofs)
              if (dof != numbers::invalid_unsigned_int)
                {
                  if (get_constraints().is_constrained(dof))
                    dof = numbers::invalid_unsigned_int;
                  else if (locally_owned_indices.is_element(dof) == false)
                    relevant_dofs.push_back(dof);
                }

            constraint_info.read_dof_indices(cell_counter,
                                             dofs,
                                             vector_partitioner);
          }

        cell_ptr.push_back(cell_ptr.back() +
                           matrix_free.n_active_entries_per_cell_batch(cell));
      }

    constraint_info.finalize();

    std::sort(relevant_dofs.begin(), relevant_dofs.end());
    relevant_dofs.erase(std::unique(relevant_dofs.begin(), relevant_dofs.end()),
                        relevant_dofs.end());

    IndexSet relevant_ghost_indices(locally_owned_indices.size());
    relevant_ghost_indices.add_indices(relevant_dofs.begin(),
                                       relevant_dofs.end());

    auto embedded_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      locally_owned_indices, vector_partitioner->get_mpi_communicator());

    embedded_partitioner->set_ghost_indices(
      relevant_ghost_indices, vector_partitioner->ghost_indices());

    this->vector_partitioner   = vector_partitioner;
    this->embedded_partitioner = embedded_partitioner;
  }

  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    if (!cell_vertex_coefficients.empty())
      do_cell_integral_local_linear_geometry(integrator);
    else if (!cell_quadratic_coefficients.empty())
      do_cell_integral_local_quadratic_geometry(integrator);
    else if (!merged_coefficients.empty())
      do_cell_integral_local_merged(integrator);
    else if (!quadrature_points.empty())
      do_cell_integral_local_construct_q(integrator);
    else
      do_cell_integral_local_base(integrator);
  }

  void
  do_cell_integral_local_base(FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }

  template <typename T>
  static inline DEAL_II_ALWAYS_INLINE T
  do_invert(Tensor<2, 2, T> &t)
  {
    const T det     = t[0][0] * t[1][1] - t[1][0] * t[0][1];
    const T inv_det = 1.0 / det;
    const T tmp     = inv_det * t[0][0];
    t[0][0]         = inv_det * t[1][1];
    t[0][1]         = -inv_det * t[0][1];
    t[1][0]         = -inv_det * t[1][0];
    t[1][1]         = tmp;
    return det;
  }

  template <typename T>
  static inline DEAL_II_ALWAYS_INLINE T
  do_invert(Tensor<2, 3, T> &t)
  {
    const T tr00    = t[1][1] * t[2][2] - t[1][2] * t[2][1];
    const T tr10    = t[1][2] * t[2][0] - t[1][0] * t[2][2];
    const T tr20    = t[1][0] * t[2][1] - t[1][1] * t[2][0];
    const T det     = t[0][0] * tr00 + t[0][1] * tr10 + t[0][2] * tr20;
    const T inv_det = 1.0 / det;
    const T tr01    = t[0][2] * t[2][1] - t[0][1] * t[2][2];
    const T tr02    = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    const T tr11    = t[0][0] * t[2][2] - t[0][2] * t[2][0];
    const T tr12    = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    t[2][1]         = inv_det * (t[0][1] * t[2][0] - t[0][0] * t[2][1]);
    t[2][2]         = inv_det * (t[0][0] * t[1][1] - t[0][1] * t[1][0]);
    t[0][0]         = inv_det * tr00;
    t[0][1]         = inv_det * tr01;
    t[0][2]         = inv_det * tr02;
    t[1][0]         = inv_det * tr10;
    t[1][1]         = inv_det * tr11;
    t[1][2]         = inv_det * tr12;
    t[2][0]         = inv_det * tr20;
    return det;
  }

  void
  do_cell_integral_local_linear_geometry(FECellIntegrator &phi) const
  {
    // adopted from:
    // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L712

    phi.evaluate(EvaluationFlags::gradients);

    const std::array<Tensor<1, dim, VectorizedArrayType>,
                     GeometryInfo<dim>::vertices_per_cell> &v =
      cell_vertex_coefficients[phi.get_current_cell_index()];

    const auto &       quad       = matrix_free.get_quadrature();
    const unsigned int n_q_points = quad.size();

    const auto &quad_1d = matrix_free.get_quadrature().get_tensor_basis()[0];
    const unsigned int n_q_points_1d = quad_1d.size();

    VectorizedArrayType *phi_grads = phi.begin_gradients();
    if (dim == 2)
      {
        for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
          {
            // x-derivative, already complete
            Tensor<1, dim, VectorizedArrayType> x_con =
              v[1] + quad_1d.point(qy)[0] * v[3];
            for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
              {
                const double q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                Tensor<2, dim, VectorizedArrayType> jac;
                jac[1] = v[2] + quad_1d.point(qx)[0] * v[3];
                for (unsigned int d = 0; d < 2; ++d)
                  jac[0][d] = x_con[d];
                const VectorizedArrayType det = do_invert(jac);

                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    const unsigned int  offset = c * dim * n_q_points;
                    VectorizedArrayType tmp[dim];
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        tmp[d] = jac[d][0] * phi_grads[q + offset];
                        for (unsigned int e = 1; e < dim; ++e)
                          tmp[d] +=
                            jac[d][e] * phi_grads[q + e * n_q_points + offset];
                        tmp[d] *= det * q_weight;
                      }
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        phi_grads[q + d * n_q_points + offset] =
                          jac[0][d] * tmp[0];
                        for (unsigned int e = 1; e < dim; ++e)
                          phi_grads[q + d * n_q_points + offset] +=
                            jac[e][d] * tmp[e];
                      }
                  }
              }
          }
      }
    else if (dim == 3)
      {
        for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
          {
            for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
              {
                const auto z = quad_1d.point(qz)[0];
                const auto y = quad_1d.point(qy)[0];
                // x-derivative, already complete
                Tensor<1, dim, VectorizedArrayType> x_con = v[1] + z * v[5];
                x_con += y * (v[4] + z * v[7]);
                // y-derivative, constant part
                Tensor<1, dim, VectorizedArrayType> y_con = v[2] + z * v[6];
                // y-derivative, xi-dependent part
                Tensor<1, dim, VectorizedArrayType> y_var = v[4] + z * v[7];
                // z-derivative, constant part
                Tensor<1, dim, VectorizedArrayType> z_con = v[3] + y * v[6];
                // z-derivative, variable part
                Tensor<1, dim, VectorizedArrayType> z_var = v[5] + y * v[7];
                double q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                  {
                    const Number x = quad_1d.point(qx)[0];
                    Tensor<2, dim, VectorizedArrayType> jac;
                    jac[1] = y_con + x * y_var;
                    jac[2] = z_con + x * z_var;
                    for (unsigned int d = 0; d < dim; ++d)
                      jac[0][d] = x_con[d];
                    VectorizedArrayType det = do_invert(jac);
                    det = det * (q_weight_tmp * quad_1d.weight(qx));

                    for (unsigned int c = 0; c < n_components; ++c)
                      {
                        const unsigned int  offset = c * dim * n_q_points;
                        VectorizedArrayType tmp[dim];
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            tmp[d] = jac[d][0] * phi_grads[q + offset];
                            for (unsigned int e = 1; e < dim; ++e)
                              tmp[d] += jac[d][e] *
                                        phi_grads[q + e * n_q_points + offset];
                            tmp[d] *= det;
                          }
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            phi_grads[q + d * n_q_points + offset] =
                              jac[0][d] * tmp[0];
                            for (unsigned int e = 1; e < dim; ++e)
                              phi_grads[q + d * n_q_points + offset] +=
                                jac[e][d] * tmp[e];
                          }
                      }
                  }
              }
          }
      }

    phi.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_local_quadratic_geometry(FECellIntegrator &phi) const
  {
    // adopted from:
    // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L860

    phi.evaluate(EvaluationFlags::gradients);

    using TensorType = Tensor<1, dim, VectorizedArrayType>;
    std::array<TensorType, Utilities::pow(3, dim - 1)> xi;
    std::array<TensorType, Utilities::pow(3, dim - 1)> di;

    const auto &v = cell_quadratic_coefficients[phi.get_current_cell_index()];

    const auto &       quad       = matrix_free.get_quadrature();
    const unsigned int n_q_points = quad.size();

    const auto &quad_1d = matrix_free.get_quadrature().get_tensor_basis()[0];
    const unsigned int n_q_points_1d = quad_1d.size();

    VectorizedArrayType *phi_grads = phi.begin_gradients();
    if (dim == 2)
      {
        for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
          {
            const Number     y  = quad_1d.point(qy)[0];
            const TensorType x1 = v[1] + y * (v[4] + y * v[7]);
            const TensorType x2 = v[2] + y * (v[5] + y * v[8]);
            const TensorType d0 = v[3] + (y + y) * v[6];
            const TensorType d1 = v[4] + (y + y) * v[7];
            const TensorType d2 = v[5] + (y + y) * v[8];
            for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
              {
                const Number q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                const Number x        = quad_1d.point(qx)[0];
                Tensor<2, dim, VectorizedArrayType> jac;
                jac[0]                        = x1 + (x + x) * x2;
                jac[1]                        = d0 + x * d1 + (x * x) * d2;
                const VectorizedArrayType det = do_invert(jac);

                for (unsigned int c = 0; c < n_components; ++c)
                  {
                    const unsigned int  offset = c * dim * n_q_points;
                    VectorizedArrayType tmp[dim];
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        tmp[d] = jac[d][0] * phi_grads[q + offset];
                        for (unsigned int e = 1; e < dim; ++e)
                          tmp[d] +=
                            jac[d][e] * phi_grads[q + e * n_q_points + offset];
                        tmp[d] *= det * q_weight;
                      }
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        phi_grads[q + d * n_q_points + offset] =
                          jac[0][d] * tmp[0];
                        for (unsigned int e = 1; e < dim; ++e)
                          phi_grads[q + d * n_q_points + offset] +=
                            jac[e][d] * tmp[e];
                      }
                  }
              }
          }
      }
    else if (dim == 3)
      {
        for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
          {
            const Number z = quad_1d.point(qz)[0];
            di[0]          = v[9] + (z + z) * v[18];
            for (unsigned int i = 1; i < 9; ++i)
              {
                xi[i] = v[i] + z * (v[9 + i] + z * v[18 + i]);
                di[i] = v[9 + i] + (z + z) * v[18 + i];
              }
            for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
              {
                const auto       y   = quad_1d.point(qy)[0];
                const TensorType x1  = xi[1] + y * (xi[4] + y * xi[7]);
                const TensorType x2  = xi[2] + y * (xi[5] + y * xi[8]);
                const TensorType dy0 = xi[3] + (y + y) * xi[6];
                const TensorType dy1 = xi[4] + (y + y) * xi[7];
                const TensorType dy2 = xi[5] + (y + y) * xi[8];
                const TensorType dz0 = di[0] + y * (di[3] + y * di[6]);
                const TensorType dz1 = di[1] + y * (di[4] + y * di[7]);
                const TensorType dz2 = di[2] + y * (di[5] + y * di[8]);
                double q_weight_tmp  = quad_1d.weight(qz) * quad_1d.weight(qy);
                for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                  {
                    const Number x = quad_1d.point(qx)[0];
                    Tensor<2, dim, VectorizedArrayType> jac;
                    jac[0]                  = x1 + (x + x) * x2;
                    jac[1]                  = dy0 + x * (dy1 + x * dy2);
                    jac[2]                  = dz0 + x * (dz1 + x * dz2);
                    VectorizedArrayType det = do_invert(jac);
                    det = det * (q_weight_tmp * quad_1d.weight(qx));

                    for (unsigned int c = 0; c < n_components; ++c)
                      {
                        const unsigned int  offset = c * dim * n_q_points;
                        VectorizedArrayType tmp[dim];
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            tmp[d] = jac[d][0] * phi_grads[q + offset];
                            for (unsigned int e = 1; e < dim; ++e)
                              tmp[d] += jac[d][e] *
                                        phi_grads[q + e * n_q_points + offset];
                            tmp[d] *= det;
                          }
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            phi_grads[q + d * n_q_points + offset] =
                              jac[0][d] * tmp[0];
                            for (unsigned int e = 1; e < dim; ++e)
                              phi_grads[q + d * n_q_points + offset] +=
                                jac[e][d] * tmp[e];
                          }
                      }
                  }
              }
          }
      }

    phi.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_local_merged(FECellIntegrator &phi) const
  {
    // adopted from:
    // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L1003

    phi.evaluate(EvaluationFlags::gradients);

    const unsigned int cell = phi.get_current_cell_index();

    const auto &       quad       = matrix_free.get_quadrature();
    const unsigned int n_q_points = quad.size();

    VectorizedArrayType *phi_grads = phi.begin_gradients();
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        if (dim == 2)
          {
            for (unsigned int c = 0; c < n_components; ++c)
              {
                const unsigned int  offset = c * dim * n_q_points;
                VectorizedArrayType tmp    = phi_grads[q + offset];
                phi_grads[q + offset] =
                  merged_coefficients[cell * n_q_points + q][0] * tmp +
                  merged_coefficients[cell * n_q_points + q][1] *
                    phi_grads[q + n_q_points + offset];
                phi_grads[q + n_q_points + offset] =
                  merged_coefficients[cell * n_q_points + q][1] * tmp +
                  merged_coefficients[cell * n_q_points + q][2] *
                    phi_grads[q + n_q_points + offset];
              }
          }
        else if (dim == 3)
          {
            for (unsigned int c = 0; c < n_components; ++c)
              {
                const unsigned int  offset = c * dim * n_q_points;
                VectorizedArrayType tmp0   = phi_grads[q + offset];
                VectorizedArrayType tmp1   = phi_grads[q + n_q_points + offset];
                phi_grads[q + offset] =
                  (merged_coefficients[cell * n_q_points + q][0] * tmp0 +
                   merged_coefficients[cell * n_q_points + q][1] * tmp1 +
                   merged_coefficients[cell * n_q_points + q][2] *
                     phi_grads[q + 2 * n_q_points + offset]);
                phi_grads[q + n_q_points + offset] =
                  (merged_coefficients[cell * n_q_points + q][1] * tmp0 +
                   merged_coefficients[cell * n_q_points + q][3] * tmp1 +
                   merged_coefficients[cell * n_q_points + q][4] *
                     phi_grads[q + 2 * n_q_points + offset]);
                phi_grads[q + 2 * n_q_points + offset] =
                  (merged_coefficients[cell * n_q_points + q][2] * tmp0 +
                   merged_coefficients[cell * n_q_points + q][4] * tmp1 +
                   merged_coefficients[cell * n_q_points + q][5] *
                     phi_grads[q + 2 * n_q_points + offset]);
              }
          }
      }
    phi.integrate(EvaluationFlags::gradients);
  }

  void
  do_cell_integral_local_construct_q(FECellIntegrator &phi) const
  {
    // adopted from:
    // https://github.com/kronbichler/ceed_benchmarks_dealii/blob/e3da3c50d9d49666b324282255cdcb7ab25c128c/common_code/poisson_operator.h#L1065

    // not implemented since we need the number of quadrature points
    // as template arguments
    AssertThrow(false, ExcNotImplemented());

    (void)phi;
  }

  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType &      dst,
                          const VectorType &src) const
  {
    if (compressed_rw)
      compressed_rw->read_dof_values(src, integrator);
    else
      integrator.read_dof_values(src);

    do_cell_integral_local(integrator);

    if (compressed_rw)
      compressed_rw->distribute_local_to_global(dst, integrator);
    else
      integrator.distribute_local_to_global(dst);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if (vector_partitioner == nullptr)
      {
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, const auto cells) {
            FECellIntegrator phi(matrix_free);
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi.reinit(cell);
                do_cell_integral_global(phi, dst, src);
              }
          },
          dst,
          src,
          true);
      }
    else
      {
        update_ghost_values(src, embedded_partitioner); // TODO: overlap?

        FECellIntegrator phi(matrix_free);

        // data structures needed for zeroing dst
        const auto &task_info           = matrix_free.get_task_info();
        const auto &partition_row_index = task_info.partition_row_index;
        const auto &cell_partition_data = task_info.cell_partition_data;
        const auto &dof_info            = matrix_free.get_dof_info();
        const auto &vector_zero_range_list_index =
          dof_info.vector_zero_range_list_index;
        const auto &vector_zero_range_list = dof_info.vector_zero_range_list;

        for (unsigned int part = 0; part < partition_row_index.size() - 2;
             ++part)
          for (unsigned int i = partition_row_index[part];
               i < partition_row_index[part + 1];
               ++i)
            {
              // zero out range in dst
              for (unsigned int id = vector_zero_range_list_index[i];
                   id != vector_zero_range_list_index[i + 1];
                   ++id)
                std::memset(dst.begin() + vector_zero_range_list[id].first,
                            0,
                            (vector_zero_range_list[id].second -
                             vector_zero_range_list[id].first) *
                              sizeof(Number));

              // loop over cells
              for (unsigned int cell = cell_partition_data[i];
                   cell < cell_partition_data[i + 1];
                   ++cell)
                {
                  phi.reinit(cell);

                  internal::VectorReader<Number, VectorizedArrayType> reader;
                  constraint_info.read_write_operation(reader,
                                                       src,
                                                       phi.begin_dof_values(),
                                                       cell_ptr[cell],
                                                       cell_ptr[cell + 1] -
                                                         cell_ptr[cell],
                                                       phi.dofs_per_cell,
                                                       true);

                  do_cell_integral_local(phi);

                  internal::VectorDistributorLocalToGlobal<Number,
                                                           VectorizedArrayType>
                    writer;
                  constraint_info.read_write_operation(writer,
                                                       dst,
                                                       phi.begin_dof_values(),
                                                       cell_ptr[cell],
                                                       cell_ptr[cell + 1] -
                                                         cell_ptr[cell],
                                                       phi.dofs_per_cell,
                                                       true);
                }
            }

        compress(dst, embedded_partitioner); // TODO: overlap?
        src.zero_out_ghost_values();
      }
  }

  static void
  update_ghost_values(const VectorType &vec,
                      const std::shared_ptr<const Utilities::MPI::Partitioner>
                        &embedded_partitioner)
  {
    const auto &vector_partitioner = vec.get_partitioner();

    dealii::AlignedVector<Number> buffer;
    buffer.resize_fast(embedded_partitioner->n_import_indices()); // reuse?

    std::vector<MPI_Request> requests;

    embedded_partitioner
      ->template export_to_ghosted_array_start<Number, MemorySpace::Host>(
        0,
        dealii::ArrayView<const Number>(
          vec.begin(), embedded_partitioner->locally_owned_size()),
        dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);

    embedded_partitioner
      ->template export_to_ghosted_array_finish<Number, MemorySpace::Host>(
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);

    vec.set_ghost_state(true);
  }

  static void
  compress(VectorType &vec,
           const std::shared_ptr<const Utilities::MPI::Partitioner>
             &embedded_partitioner)
  {
    const auto &vector_partitioner = vec.get_partitioner();

    dealii::AlignedVector<Number> buffer;
    buffer.resize_fast(embedded_partitioner->n_import_indices()); // reuse?

    std::vector<MPI_Request> requests;

    embedded_partitioner
      ->template import_from_ghosted_array_start<Number, MemorySpace::Host>(
        dealii::VectorOperation::add,
        0,
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
        requests);

    embedded_partitioner
      ->template import_from_ghosted_array_finish<Number, MemorySpace::Host>(
        dealii::VectorOperation::add,
        dealii::ArrayView<const Number>(buffer.begin(), buffer.size()),
        dealii::ArrayView<Number>(vec.begin(),
                                  embedded_partitioner->locally_owned_size()),
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);
  }

  void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product) const
  {
    if (vector_partitioner == nullptr)
      {
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, const auto cells) {
            FECellIntegrator phi(matrix_free);
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi.reinit(cell);
                do_cell_integral_global(phi, dst, src);
              }
          },
          dst,
          src,
          operation_before_matrix_vector_product,
          operation_after_matrix_vector_product);
      }
    else
      {
        operation_before_matrix_vector_product(0, src.locally_owned_size());
        vmult(dst, src);
        operation_after_matrix_vector_product(0, src.locally_owned_size());
      }
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(false, ExcNotImplemented());
    (void)dst;
    (void)src;
  }

  const Mapping<dim> &
  get_mapping() const
  {
    return *matrix_free.get_mapping_info().mapping;
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return get_dof_handler().get_fe();
  }

  const Triangulation<dim> &
  get_triangulation() const
  {
    return get_dof_handler().get_triangulation();
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return matrix_free.get_dof_handler();
  }

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    if (sparse_matrix.m() == 0 && sparse_matrix.n() == 0)
      compute_system_matrix();

    return sparse_matrix;
  }

  const TrilinosWrappers::SparsityPattern &
  get_sparsity_pattern() const
  {
    if (sparse_matrix.m() == 0 && sparse_matrix.n() == 0)
      compute_system_matrix();

    return sparsity_pattern;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    if (vector_partitioner)
      vec.reinit(vector_partitioner);
    else
      matrix_free.initialize_dof_vector(vec);
  }

  const AffineConstraints<Number> &
  get_constraints() const
  {
    return get_matrix_free().get_affine_constraints();
  }

  const Quadrature<dim> &
  get_quadrature() const
  {
    return matrix_free.get_quadrature();
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &LaplaceOperatorMatrixFree::do_cell_integral_local,
      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

private:
  void
  compute_system_matrix() const
  {
    Assert((sparse_matrix.m() == 0 && sparse_matrix.n() == 0),
           ExcNotImplemented());

    const auto &dof_handler = get_dof_handler();

    sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                            dof_handler.get_triangulation().get_communicator());

    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    get_constraints());

    sparsity_pattern.compress();
    sparse_matrix.reinit(sparsity_pattern);

    MatrixFreeTools::compute_matrix(
      matrix_free,
      get_constraints(),
      sparse_matrix,
      &LaplaceOperatorMatrixFree::do_cell_integral_local,
      this);
  }

  // internal matrix-free
  DoFHandler<dim>                              dof_handler_internal;
  AffineConstraints<Number>                    constraints_internal;
  MatrixFree<dim, Number, VectorizedArrayType> matrix_free_internal;

  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

  // for own partitioner
  mutable std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner;
  mutable std::shared_ptr<const Utilities::MPI::Partitioner>
                                    embedded_partitioner;
  mutable std::vector<unsigned int> cell_ptr;
  mutable internal::MatrixFreeFunctions::ConstraintInfo<dim,
                                                        VectorizedArrayType>
    constraint_info;

  ConditionalOStream pcout;

  // only set up if required
  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;

  // compressed indices
  std::shared_ptr<ConstraintInfoReduced> compressed_rw;

  // mapping
  AlignedVector<std::array<Tensor<1, dim, VectorizedArrayType>,
                           GeometryInfo<dim>::vertices_per_cell>>
    cell_vertex_coefficients;

  AlignedVector<
    std::array<Tensor<1, dim, VectorizedArrayType>, Utilities::pow(3, dim)>>
    cell_quadratic_coefficients;

  AlignedVector<Tensor<1, (dim * (dim + 1) / 2), VectorizedArrayType>>
    merged_coefficients;

  AlignedVector<VectorizedArrayType> quadrature_points;
};
