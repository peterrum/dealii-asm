#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "dof_tools.h"
#include "exceptions.h"
#include "grid_tools.h"
#include "matrix_free_internal.h"
#include "vector_access_reduced.h"

/**
 * Base class for LaplaceOperatorMatrixBased and
 * LaplaceOperatorMatrixFree. It provedes an interface
 * for a simle vmult and not the version with pre/post operations.
 */
template <int dim, typename VectorType>
class LaplaceOperatorBase : public Subscriptor
{
public:
  using vector_type = VectorType;
  using Number      = typename vector_type::value_type;
  using value_type  = Number;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  virtual void
  initialize_dof_vector(VectorType &vec) const = 0;

  virtual const AffineConstraints<Number> &
  get_constraints() const = 0;

  virtual void
  rhs(VectorType &                                  vec,
      const std::shared_ptr<Function<dim, Number>> &rhs_func) = 0;

private:
};



/**
 * Matrix-based implementation of the Laplace operator.
 */
template <int dim, typename Number>
class LaplaceOperatorMatrixBased
  : public LaplaceOperatorBase<dim, LinearAlgebra::distributed::Vector<Number>>
{
public:
  static const int dimension = dim;
  using value_type           = Number;
  using vectorized_array_type =
    VectorizedArray<Number>; // dummy: for compilation
  using vector_type = LinearAlgebra::distributed::Vector<Number>;

  using VectorType = vector_type;

  // not used
  struct AdditionalData
  {
    AdditionalData(const bool        compress_indices = false,
                   const std::string mapping_type     = "")
      : compress_indices(compress_indices)
      , mapping_type(mapping_type)
    {}

    bool        compress_indices;
    std::string mapping_type;
  };

  LaplaceOperatorMatrixBased(
    const Mapping<dim> &                          mapping,
    const Triangulation<dim> &                    tria,
    const FiniteElement<dim> &                    fe,
    const Quadrature<dim> &                       quadrature,
    const AdditionalData &                        ad       = AdditionalData(),
    const std::shared_ptr<Function<dim, Number>> &dbc_func = {})
    : mapping(mapping)
    , dof_handler(tria)
    , quadrature(quadrature)
  {
    AssertThrow(ad.compress_indices == false, ExcNotImplemented());
    AssertThrow(ad.mapping_type == "", ExcNotImplemented());

    dof_handler.distribute_dofs(fe);

    if (dbc_func)
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 1, *dbc_func, constraints);
    else
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

  virtual void
  rhs(VectorType &vec, const std::shared_ptr<Function<dim, Number>> &rhs_func)
  {
    VectorTools::create_right_hand_side(
      mapping, dof_handler, quadrature, *rhs_func, vec, constraints);
  }

  virtual bool
  uses_compressed_indices() const
  {
    return false;
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



/**
 * Matrix-free implementation of the Laplace operator.
 */
template <int dim,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
class LaplaceOperatorMatrixFree
  : public LaplaceOperatorBase<dim, LinearAlgebra::distributed::Vector<Number>>
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

  struct AdditionalData
  {
    AdditionalData(const bool        compress_indices = false,
                   const std::string mapping_type     = "")
      : compress_indices(compress_indices)
      , mapping_type(mapping_type)
    {}

    bool        compress_indices;
    std::string mapping_type;
  };

  virtual void
  rhs(VectorType &vec, const std::shared_ptr<Function<dim, Number>> &rhs_func)
  {
    VectorTools::create_right_hand_side(get_mapping(),
                                        get_dof_handler(),
                                        get_quadrature(),
                                        *rhs_func,
                                        vec,
                                        get_constraints());
  }

  LaplaceOperatorMatrixFree(
    const Mapping<dim> &                          mapping,
    const Triangulation<dim> &                    tria,
    const FiniteElement<dim> &                    fe,
    const Quadrature<dim> &                       quadrature,
    const AdditionalData &                        ad       = AdditionalData(),
    const std::shared_ptr<Function<dim, Number>> &dbc_func = {})
    : dof_handler_internal(tria)
    , matrix_free(matrix_free_internal)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(
              dof_handler_internal.get_communicator()) == 0)
  {
    dof_handler_internal.distribute_dofs(fe);

    const auto setup_constraints = [&]() {
      constraints_internal.clear();
      if (dbc_func)
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler_internal, 1, *dbc_func, constraints_internal);
      else
        DoFTools::make_zero_boundary_constraints(dof_handler_internal,
                                                 1,
                                                 constraints_internal);
      constraints_internal.close();
    };

    setup_constraints();

    if (ad.compress_indices) // TODO: should decouple renumbering and
                             // compression?
      {
        // note: we need to renumber for double/float here the same way
        // since else the transfer between active and finest level will
        // not work (different numbering due to different number of lanes)
        typename MatrixFree<dim, float, VectorizedArray<float>>::AdditionalData
          additional_data;

        DoFRenumbering::matrix_free_data_locality(dof_handler_internal,
                                                  constraints_internal,
                                                  additional_data);
        setup_constraints();
      }

    matrix_free_internal.reinit(mapping,
                                dof_handler_internal,
                                constraints_internal,
                                quadrature);

    pcout << "- Create operator:" << std::endl;
    pcout << "  - n cells:          "
          << dof_handler_internal.get_triangulation().n_global_active_cells()
          << std::endl;
    pcout << "  - n dofs:           " << dof_handler_internal.n_dofs()
          << std::endl;
    pcout << "  - compress indices: "
          << (ad.compress_indices ? "true" : "false") << std::endl;
    pcout << "  - mapping type:     " << ad.mapping_type << std::endl;

    setup_mapping_and_indices(ad.compress_indices, ad.mapping_type);

    pcout << std::endl;
  }

  LaplaceOperatorMatrixFree(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AdditionalData &                              ad = AdditionalData())
    : matrix_free(matrix_free)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(
              matrix_free.get_dof_handler().get_communicator()) == 0)
  {
    pcout << "- Create operator:" << std::endl;
    pcout << "  - compress indices: "
          << (ad.compress_indices ? "true" : "false") << std::endl;
    pcout << "  - mapping type:     " << ad.mapping_type << std::endl;

    setup_mapping_and_indices(ad.compress_indices, ad.mapping_type);

    pcout << std::endl;
  }

  virtual bool
  uses_compressed_indices() const
  {
    return compressed_rw != nullptr;
  }

  void
  setup_mapping_and_indices(const bool        compress_indices,
                            const std::string mapping_type)
  {
    if (compress_indices)
      {
        auto       compressed_rw = std::make_shared<ConstraintInfoReduced>();
        const bool flag          = compressed_rw->initialize(matrix_free);

        if (flag)
          this->compressed_rw = compressed_rw;

        pcout << "  - compress indices: " << (flag ? "success" : "failure")
              << std::endl;
      }


    if (mapping_type == "")
      {
        // nothing to do
      }
    else if (mapping_type == "linear geometry")
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
    else if (mapping_type == "quadratic geometry")
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
    else if (mapping_type == "merged")
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
    else if (mapping_type == "construct q")
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
    else
      {
        AssertThrow(false,
                    ExcMessage("Mapping type <" + mapping_type +
                               "> is not known!"));
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

    constexpr unsigned int n_q_points_1d_static = 0; // TODO

    const auto cell = phi.get_current_cell_index();

    const auto &       quad       = matrix_free.get_quadrature();
    const unsigned int n_q_points = quad.size();

    const auto &quad_1d = matrix_free.get_quadrature().get_tensor_basis()[0];
    const unsigned int n_q_points_1d = quad_1d.size();


    AlignedVector<VectorizedArrayType> jacobians_x(dim * n_q_points_1d); // TODO
    AlignedVector<VectorizedArrayType> jacobians_y(dim * n_q_points_1d *
                                                   n_q_points_1d);    // TODO
    AlignedVector<VectorizedArrayType> jacobians_z(dim * n_q_points); // TODO

    phi.evaluate(EvaluationFlags::gradients);
    VectorizedArrayType *phi_grads = phi.begin_gradients();
    if (dim == 3)
      for (unsigned int d = 0; d < dim; ++d)
        dealii::internal::EvaluatorTensorProduct<
          dealii::internal::evaluate_evenodd,
          dim,
          n_q_points_1d_static,
          n_q_points_1d_static,
          VectorizedArrayType,
          VectorizedArrayType>({}, {}, {}, n_q_points_1d, n_q_points_1d)
          .template apply<2, true, false, 1>(
            matrix_free.get_shape_info()
              .data[0]
              .shape_gradients_collocation_eo.begin(),
            quadrature_points.begin() + (cell * dim + d) * n_q_points,
            jacobians_z.begin() + d * n_q_points);
    for (unsigned int q2 = 0, q = 0; q2 < (dim == 3 ? n_q_points_1d : 1); ++q2)
      {
        const unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

        for (unsigned int d = 0; d < dim; ++d)
          dealii::internal::EvaluatorTensorProduct<
            dealii::internal::evaluate_evenodd,
            2,
            n_q_points_1d_static,
            n_q_points_1d_static,
            VectorizedArrayType,
            VectorizedArrayType>({}, {}, {}, n_q_points_1d, n_q_points_1d)
            .template apply<1, true, false, 1>(
              matrix_free.get_shape_info()
                .data[0]
                .shape_gradients_collocation_eo.begin(),
              quadrature_points.begin() + (cell * dim + d) * n_q_points +
                q2 * n_q_points_2d,
              jacobians_y.begin() + d * n_q_points_2d);
        for (unsigned int q1 = 0; q1 < n_q_points_1d; ++q1)
          {
            for (unsigned int d = 0; d < dim; ++d)
              dealii::internal::EvaluatorTensorProduct<
                dealii::internal::evaluate_evenodd,
                1,
                n_q_points_1d_static,
                n_q_points_1d_static,
                VectorizedArrayType,
                VectorizedArrayType>({}, {}, {}, n_q_points_1d, n_q_points_1d)
                .template apply<0, true, false, 1>(
                  matrix_free.get_shape_info()
                    .data[0]
                    .shape_gradients_collocation_eo.begin(),
                  quadrature_points.begin() + (cell * dim + d) * n_q_points +
                    q2 * n_q_points_2d + q1 * n_q_points_1d,
                  jacobians_x.begin() + d * n_q_points_1d);
            for (unsigned int q0 = 0; q0 < n_q_points_1d; ++q0, ++q)
              {
                Tensor<2, dim, VectorizedArrayType> jac;
                for (unsigned int e = 0; e < dim; ++e)
                  jac[2][e] = jacobians_z[e * n_q_points + q];
                for (unsigned int e = 0; e < dim; ++e)
                  jac[1][e] =
                    jacobians_y[e * n_q_points_2d + q1 * n_q_points_1d + q0];
                for (unsigned int e = 0; e < dim; ++e)
                  jac[0][e] = jacobians_x[e * n_q_points_1d + q0];
                VectorizedArrayType det = do_invert(jac);
                det = det * (matrix_free.get_quadrature().weight(q));

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
    phi.integrate(EvaluationFlags::gradients);
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
    vmult(dst,
          src,
          [&](const auto start_range, const auto end_range) {
            if (end_range > start_range)
              std::memset(dst.begin() + start_range,
                          0,
                          sizeof(Number) * (end_range - start_range));
          },
          {});
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
        cell_loop(
          [&](const auto &, auto &dst, const auto &src, const auto cells) {
            FECellIntegrator phi(matrix_free);
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
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
          },
          dst,
          src,
          operation_before_matrix_vector_product,
          operation_after_matrix_vector_product);
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
  cell_loop(const std::function<
              void(const MatrixFree<dim, Number, VectorizedArrayType> &,
                   VectorType &,
                   const VectorType &,
                   const std::pair<unsigned int, unsigned int>)> &cell_function,
            VectorType &                                          dst,
            const VectorType &                                    src,
            const std::function<void(unsigned int, unsigned int)>
              &operation_before_loop,
            const std::function<void(unsigned int, unsigned int)>
              &operation_after_loop) const
  {
    VectorDataExchange<Number> exchanger_dst(embedded_partitioner, buffer_dst);
    VectorDataExchange<Number> exchanger_src(embedded_partitioner, buffer_src);

    MFWorker<dim, Number, VectorizedArrayType, VectorType> worker(
      matrix_free,
      matrix_free.get_dof_info().cell_loop_pre_list_index,
      matrix_free.get_dof_info().cell_loop_pre_list,
      matrix_free.get_dof_info().cell_loop_post_list_index,
      matrix_free.get_dof_info().cell_loop_post_list,
      exchanger_dst,
      exchanger_src,
      dst,
      src,
      cell_function,
      operation_before_loop,
      operation_after_loop);

    MFRunner runner;
    runner.loop(worker);
  }

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

  mutable dealii::AlignedVector<Number> buffer_dst;
  mutable dealii::AlignedVector<Number> buffer_src;
};
