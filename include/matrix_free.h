#pragma once

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include "dof_tools.h"
#include "grid_tools.h"
#include "preconditioners.h"
#include "restrictors.h"
#include "tensor_product_matrix.h"

template <int dim,
          typename Number,
          typename VectorizedArrayType,
          int n_rows_1d = -1>
class ASPoissonPreconditioner
  : public PreconditionerBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ASPoissonPreconditioner(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const unsigned int                                  n_overlap,
    const unsigned int                                  sub_mesh_approximation,
    const Mapping<dim> &                                mapping,
    const FiniteElement<1> &                            fe_1D,
    const Quadrature<dim - 1> &                         quadrature_face,
    const Quadrature<1> &                               quadrature_1D,
    const Restrictors::WeightingType                    weight_type =
      Restrictors::WeightingType::post)
    : matrix_free(matrix_free)
    , fe_degree(matrix_free.get_dof_handler().get_fe().tensor_degree())
    , n_overlap(n_overlap)
  {
    AssertThrow((n_rows_1d == -1) || (static_cast<unsigned int>(n_rows_1d) ==
                                      fe_1D.degree + 2 * n_overlap - 1),
                ExcNotImplemented());

    AssertThrow(weight_type == Restrictors::WeightingType::post ||
                  weight_type == Restrictors::WeightingType::none,
                ExcNotImplemented());

    const auto &dof_handler = matrix_free.get_dof_handler();

    // set up ConstraintInfo
    // ... allocate memory
    constraint_info.reinit(matrix_free.n_physical_cells());

    auto partitioner_for_fdm = matrix_free.get_vector_partitioner();

    if (n_overlap > 1)
      {
        const auto &locally_owned_dofs = dof_handler.locally_owned_dofs();

        std::vector<types::global_dof_index> ghost_indices;

        for (unsigned int cell = 0, cell_counter = 0;
             cell < matrix_free.n_cell_batches();
             ++cell)
          {
            for (unsigned int v = 0;
                 v < matrix_free.n_active_entries_per_cell_batch(cell);
                 ++v, ++cell_counter)
              {
                const auto cells =
                  dealii::GridTools::extract_all_surrounding_cells_cartesian<
                    dim>(matrix_free.get_cell_iterator(cell, v),
                         n_overlap <= 1 ? 0 : sub_mesh_approximation);

                const auto local_dofs =
                  dealii::DoFTools::get_dof_indices_cell_with_overlap(
                    dof_handler, cells, n_overlap, true);

                for (const auto i : local_dofs)
                  if ((locally_owned_dofs.is_element(i) == false) &&
                      (i != numbers::invalid_unsigned_int))
                    ghost_indices.push_back(i);
              }
          }

        std::sort(ghost_indices.begin(), ghost_indices.end());
        ghost_indices.erase(std::unique(ghost_indices.begin(),
                                        ghost_indices.end()),
                            ghost_indices.end());

        IndexSet is_ghost_indices(locally_owned_dofs.size());
        is_ghost_indices.add_indices(ghost_indices.begin(),
                                     ghost_indices.end());

        partitioner_for_fdm = std::make_shared<Utilities::MPI::Partitioner>(
          locally_owned_dofs, is_ghost_indices, dof_handler.get_communicator());
      }

    fdm.resize(matrix_free.n_cell_batches());

    const auto harmonic_patch_extend =
      GridTools::compute_harmonic_patch_extend(mapping,
                                               dof_handler.get_triangulation(),
                                               quadrature_face);

    // ... collect DoF indices
    cell_ptr = {0};
    for (unsigned int cell = 0, cell_counter = 0;
         cell < matrix_free.n_cell_batches();
         ++cell)
      {
        std::array<MyTensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>,
                   VectorizedArrayType::size()>
          scalar_fdm;

        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v, ++cell_counter)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

            const auto cells =
              dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                cell_iterator, n_overlap <= 1 ? 0 : sub_mesh_approximation);

            constraint_info.read_dof_indices(
              cell_counter,
              dealii::DoFTools::get_dof_indices_cell_with_overlap(
                dof_handler, cells, n_overlap, true),
              partitioner_for_fdm);

            scalar_fdm[v] = setup_fdm<dim, Number, n_rows_1d>(
              cell_iterator,
              fe_1D,
              quadrature_1D,
              harmonic_patch_extend[cell_iterator->active_cell_index()],
              n_overlap);
          }

        cell_ptr.push_back(cell_ptr.back() +
                           matrix_free.n_active_entries_per_cell_batch(cell));

        fdm[cell] = MyTensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::
          template transpose<VectorizedArrayType::size()>(
            scalar_fdm, matrix_free.n_active_entries_per_cell_batch(cell));
      }

    constraint_info.finalize();

    src_.reinit(partitioner_for_fdm);
    dst_.reinit(partitioner_for_fdm);

    {
      AlignedVector<VectorizedArrayType> dst__(
        Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

      dst_ = 0.0;

      for (auto &i : dst__)
        i = 1.0;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
            writer;
          constraint_info.read_write_operation(writer,
                                               dst_,
                                               dst__,
                                               cell_ptr[cell],
                                               cell_ptr[cell + 1] -
                                                 cell_ptr[cell],
                                               dst__.size(),
                                               true);
        }

      dst_.compress(VectorOperation::add);

      matrix_free.initialize_dof_vector(weights);

      weights.copy_locally_owned_data_from(dst_);

      for (auto &i : weights)
        i = (i == 0.0) ? 1.0 : (1.0 / i);
    }

    if (weight_type == Restrictors::WeightingType::none)
      weights.reinit(0);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if (src_.get_partitioner().get() ==
        matrix_free.get_vector_partitioner().get())
      {
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free,
              auto &      dst,
              const auto &src,
              const auto  cells) {
            FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_src(
              matrix_free);
            FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_dst(
              matrix_free);

            AlignedVector<VectorizedArrayType> tmp;

            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi_src.reinit(cell);
                phi_dst.reinit(cell);

                phi_src.read_dof_values(src);

                if (true)
                  {
                    fdm[cell].apply_inverse(
                      ArrayView<VectorizedArrayType>(phi_dst.begin_dof_values(),
                                                     phi_dst.dofs_per_cell),
                      ArrayView<const VectorizedArrayType>(
                        phi_src.begin_dof_values(), phi_src.dofs_per_cell),
                      tmp);
                  }
                else
                  {
                    for (unsigned int i = 0; i < phi_src.dofs_per_cell; ++i)
                      phi_dst.begin_dof_values()[i] =
                        phi_src.begin_dof_values()[i];
                  }

                phi_dst.distribute_local_to_global(dst);
              }
          },
          dst,
          src,
          true);
      }
    else
      {
        AlignedVector<VectorizedArrayType> src__(
          Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
        AlignedVector<VectorizedArrayType> dst__(
          Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

        // update ghost values
        src_.copy_locally_owned_data_from(src);
        src_.update_ghost_values();

        dst_ = 0.0;

        for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
          {
            // 1) gather
            internal::VectorReader<Number, VectorizedArrayType> reader;
            constraint_info.read_write_operation(reader,
                                                 src_,
                                                 src__,
                                                 cell_ptr[cell],
                                                 cell_ptr[cell + 1] -
                                                   cell_ptr[cell],
                                                 src__.size(),
                                                 true);

            // 2) cell operation: fast diagonalization method
            fdm[cell].apply_inverse(make_array_view(dst__.begin(), dst__.end()),
                                    make_array_view(src__.begin(),
                                                    src__.end()));

            // 3) scatter
            internal::VectorDistributorLocalToGlobal<Number,
                                                     VectorizedArrayType>
              writer;
            constraint_info.read_write_operation(writer,
                                                 dst_,
                                                 dst__,
                                                 cell_ptr[cell],
                                                 cell_ptr[cell + 1] -
                                                   cell_ptr[cell],
                                                 dst__.size(),
                                                 true);
          }

        // compress
        dst_.compress(VectorOperation::add);
        dst.copy_locally_owned_data_from(dst_);
      }

    if (weights.size() > 0)
      dst.scale(weights);
  }

  std::size_t
  memory_consumption() const
  {
    return MemoryConsumption::memory_consumption(fdm);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  fe_degree;
  const unsigned int                                  n_overlap;

  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
                            constraint_info;
  std::vector<unsigned int> cell_ptr;

  std::vector<
    MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType, n_rows_1d>>
    fdm;

  mutable VectorType src_;
  mutable VectorType dst_;

  VectorType weights;
};



template <int dim, typename Number, typename VectorizedArrayType>
class PoissonOperator : public Subscriptor
{
public:
  using value_type = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  PoissonOperator(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
  {}

  void
  initialize_dof_vector(VectorType &vec)
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  rhs(VectorType &vec) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&](const auto &, auto &dst, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(
          matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.dofs_per_cell; ++q)
              phi.submit_value(1.0, q);

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      vec,
      dummy,
      true);
  }


  void
  do_cell_integral_local(FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }


  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType &      dst,
                          const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }


  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(
          matrix_free);
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


  void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(
          matrix_free);
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


  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &PoissonOperator::do_cell_integral_local,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }


  types::global_dof_index
  m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }


  Number
  el(unsigned int, unsigned int) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }


private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
};