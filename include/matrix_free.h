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
#include "vector_access_reduced.h"

template <int dim,
          typename Number,
          typename VectorizedArrayType,
          int n_rows_1d = -1>
class ASPoissonPreconditioner
  : public PreconditionerBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  ASPoissonPreconditioner(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const unsigned int                                  n_overlap,
    const unsigned int                                  sub_mesh_approximation,
    const Mapping<dim> &                                mapping,
    const FiniteElement<1> &                            fe_1D,
    const Quadrature<dim - 1> &                         quadrature_face,
    const Quadrature<1> &                               quadrature_1D,
    const Restrictors::WeightingType                    weight_type =
      Restrictors::WeightingType::post,
    const bool compress_indices = true)
    : matrix_free(matrix_free)
    , fe_degree(matrix_free.get_dof_handler().get_fe().tensor_degree())
    , n_overlap(n_overlap)
    , weight_type(weight_type)
  {
    AssertThrow((n_rows_1d == -1) || (static_cast<unsigned int>(n_rows_1d) ==
                                      fe_1D.degree + 2 * n_overlap - 1),
                ExcNotImplemented());

    const auto &dof_handler = matrix_free.get_dof_handler();
    const auto &constraints = matrix_free.get_affine_constraints();

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(
                               dof_handler.get_communicator()) == 0);

    // set up ConstraintInfo
    // ... allocate memory
    constraint_info.reinit(matrix_free.n_physical_cells());

    partitioner_for_fdm = matrix_free.get_vector_partitioner();

    const auto resolve_constraint = [&](auto &i) {
      const auto *entries_ptr = constraints.get_constraint_entries(i);

      if (entries_ptr != nullptr)
        {
          const auto &                  entries   = *entries_ptr;
          const types::global_dof_index n_entries = entries.size();
          if (n_entries == 1 && std::abs(entries[0].second - 1.) <
                                  100 * std::numeric_limits<double>::epsilon())
            {
              i = entries[0].first; // identity constraint
            }
          else if (n_entries == 0)
            {
              i = numbers::invalid_dof_index; // homogeneous
                                              // Dirichlet
            }
          else
            {
              // other constraints, e.g., hanging-node
              // constraints; not implemented yet
              AssertThrow(false, ExcNotImplemented());
            }
        }
      else
        {
          // not constrained -> nothing to do
        }
    };

    if (n_overlap == 1)
      {
        if (compress_indices)
          {
            auto compressed_rw = std::make_shared<ConstraintInfoReduced>();
            compressed_rw->initialize(matrix_free);
            this->compressed_rw = compressed_rw;
          }
      }
    else
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

                auto local_dofs =
                  dealii::DoFTools::get_dof_indices_cell_with_overlap(
                    dof_handler, cells, n_overlap, true);

                for (auto &i : local_dofs)
                  resolve_constraint(i);

                for (const auto &i : local_dofs)
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


    pcout << "    - compress indices:       "
          << ((this->compressed_rw != nullptr) ? "true" : "false") << std::endl;

    fdm.reserve(matrix_free.n_cell_batches());

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
        std::array<Table<2, VectorizedArrayType>, dim> Ms;
        std::array<Table<2, VectorizedArrayType>, dim> Ks;

        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v, ++cell_counter)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

            const auto cells =
              dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                cell_iterator, n_overlap <= 1 ? 0 : sub_mesh_approximation);

            auto local_dofs =
              dealii::DoFTools::get_dof_indices_cell_with_overlap(dof_handler,
                                                                  cells,
                                                                  n_overlap,
                                                                  true);

            for (auto &i : local_dofs)
              resolve_constraint(i);

            constraint_info.read_dof_indices(cell_counter,
                                             local_dofs,
                                             partitioner_for_fdm);

            const auto [Ms_scalar, Ks_scalar] =
              create_laplace_tensor_product_matrix<dim, Number>(
                cell_iterator,
                fe_1D,
                quadrature_1D,
                harmonic_patch_extend[cell_iterator->active_cell_index()],
                n_overlap);

            for (unsigned int d = 0; d < dim; ++d)
              {
                if (Ms[d].size(0) == 0 || Ms[d].size(1) == 0)
                  {
                    Ms[d].reinit(Ms_scalar[d].size(0), Ms_scalar[d].size(1));
                    Ks[d].reinit(Ks_scalar[d].size(0), Ks_scalar[d].size(1));
                  }

                for (unsigned int i = 0; i < Ms_scalar[d].size(0); ++i)
                  for (unsigned int j = 0; j < Ms_scalar[d].size(0); ++j)
                    Ms[d][i][j][v] = Ms_scalar[d][i][j];

                for (unsigned int i = 0; i < Ks_scalar[d].size(0); ++i)
                  for (unsigned int j = 0; j < Ks_scalar[d].size(0); ++j)
                    Ks[d][i][j][v] = Ks_scalar[d][i][j];
              }
          }

        cell_ptr.push_back(cell_ptr.back() +
                           matrix_free.n_active_entries_per_cell_batch(cell));

        fdm.insert(cell, Ms, Ks);
      }

    fdm.finalize();

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
                                               dst__.data(),
                                               cell_ptr[cell],
                                               cell_ptr[cell + 1] -
                                                 cell_ptr[cell],
                                               dst__.size(),
                                               true);
        }

      dst_.compress(VectorOperation::add);

      weights.reinit(partitioner_for_fdm);

      weights.copy_locally_owned_data_from(dst_);

      for (auto &i : weights)
        i = (i == 0.0) ?
              1.0 :
              (1.0 / ((weight_type == Restrictors::WeightingType::symm) ?
                        std::sqrt(i) :
                        i));

      weights.update_ghost_values();
    }

    if (fe_1D.degree >= 2 && n_overlap == 1)
      {
        weights_compressed_q2.resize(matrix_free.n_cell_batches());

        FECellIntegrator phi(matrix_free);

        for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values_plain(weights);

            const bool success = dealii::internal::
              compute_weights_fe_q_dofs_by_entity<dim, -1, VectorizedArrayType>(
                phi.begin_dof_values(),
                1,
                fe_degree + 1,
                weights_compressed_q2[cell].begin());

            AssertThrow(success, ExcInternalError());
          }
      }

    if (weight_type == Restrictors::WeightingType::none)
      {
        weights.reinit(0);
        weights_compressed_q2.clear();
      }
  }

  /**
   * General matrix-vector product.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if ((partitioner_for_fdm.get() ==
         matrix_free.get_vector_partitioner().get()) &&
        (partitioner_for_fdm.get() == src.get_partitioner().get()))
      {
        // use matrix-free version
        vmult_internal(dst,
                       src,
                       [&](const auto start_range, const auto end_range) {
                         if (end_range > start_range)
                           std::memset(dst.begin() + start_range,
                                       0,
                                       sizeof(Number) *
                                         (end_range - start_range));
                       },
                       {});
      }
    else
      {
        // use general version
        vmult_internal(dst, src);
      }
  }

  /**
   * Matrix-vector product with pre- and post-operations to be used
   * by PreconditionRelaxation and PreconditionChebyshev.
   */
  virtual void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product = {}) const
  {
    if ((partitioner_for_fdm.get() ==
         matrix_free.get_vector_partitioner().get()) &&
        (partitioner_for_fdm.get() == src.get_partitioner().get()))
      {
        // use matrix-free version
        vmult_internal(dst,
                       src,
                       operation_before_matrix_vector_product,
                       operation_after_matrix_vector_product);
      }
    else
      {
        // use general version; note: the pre-operation cleares the content
        // of dst so that we can skip zeroing
        operation_before_matrix_vector_product(0, src.locally_owned_size());
        vmult_internal(dst, src, /*dst is zero*/ true);

        if (operation_after_matrix_vector_product)
          operation_after_matrix_vector_product(0, src.locally_owned_size());
      }
  }

  std::size_t
  memory_consumption() const
  {
    return fdm.memory_consumption();
  }

  std::shared_ptr<const Utilities::MPI::Partitioner>
  get_partitioner() const
  {
    return partitioner_for_fdm;
  }

  unsigned int
  n_fdm_instances() const
  {
    return fdm.storage_size();
  }


private:
  void
  vmult_internal(VectorType &      dst,
                 const VectorType &src,
                 const bool        dst_is_zero = false) const
  {
    const bool do_weights_global = true; // TODO

    const bool do_inplace_dst =
      partitioner_for_fdm.get() == src.get_partitioner().get();
    const bool do_inplace_src =
      do_inplace_dst &&
      ((do_weights_global == false) ||
       ((weight_type == Restrictors::WeightingType::pre ||
         weight_type == Restrictors::WeightingType::symm) == false));

    auto &      dst_ptr = do_inplace_dst ? dst : this->dst_;
    const auto &src_ptr = do_inplace_src ? src : this->src_;

    // apply weights and copy vector (both optional)
    if (do_weights_global && (weight_type == Restrictors::WeightingType::pre ||
                              weight_type == Restrictors::WeightingType::symm))
      {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (std::size_t i = 0; i < src.locally_owned_size(); ++i)
          this->src_.local_element(i) =
            weights.local_element(i) * src.local_element(i);
      }
    else if (do_inplace_src == false)
      this->src_.copy_locally_owned_data_from(src);

    // update ghost values
    src_ptr.update_ghost_values();

    if ((do_inplace_dst == false) || (dst_is_zero == false))
      dst_ptr = 0.0;

    // data structures needed for the cell loop
    AlignedVector<VectorizedArrayType> tmp;

    AlignedVector<VectorizedArrayType> src_local(
      Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
    AlignedVector<VectorizedArrayType> dst_local(
      Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
    AlignedVector<VectorizedArrayType> weights_local;

    if ((do_weights_global == false) &&
        (weight_type == Restrictors::WeightingType::none))
      weights_local.resize(Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

    // loop over cells
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        // 1) gather src (optional)
        internal::VectorReader<Number, VectorizedArrayType> reader;
        constraint_info.read_write_operation(reader,
                                             src_ptr,
                                             src_local.data(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             src_local.size(),
                                             true);

        // 2) apply weights (optional)
        if (weight_type != Restrictors::WeightingType::post)
          apply_weights_local(cell, weights_local, src_local, true);

        // 3) cell operation: fast diagonalization method
        fdm.apply_inverse(cell,
                          make_array_view(dst_local.begin(), dst_local.end()),
                          make_array_view(src_local.begin(), src_local.end()),
                          tmp);

        // 4) apply weights (optional)
        if (weight_type != Restrictors::WeightingType::post)
          apply_weights_local(cell, weights_local, dst_local, true);

        // 5) scatter
        internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
          writer;
        constraint_info.read_write_operation(writer,
                                             dst_ptr,
                                             dst_local.data(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             dst_local.size(),
                                             true);
      }

    // compress
    src_ptr.zero_out_ghost_values();
    dst_ptr.compress(VectorOperation::add);

    // apply weights and copy vector back (both optional)
    if (do_weights_global && (weight_type == Restrictors::WeightingType::post ||
                              weight_type == Restrictors::WeightingType::symm))
      {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (std::size_t i = 0; i < dst.locally_owned_size(); ++i)
          dst.local_element(i) =
            weights.local_element(i) * dst_ptr.local_element(i);
      }
    else if (do_inplace_dst == false)
      dst.copy_locally_owned_data_from(dst_ptr);
  }

  void
  vmult_internal(
    VectorType &      dst,
    const VectorType &src,
    const std::function<void(const unsigned int, const unsigned int)>
      &operation_before_matrix_vector_product,
    const std::function<void(const unsigned int, const unsigned int)>
      &operation_after_matrix_vector_product) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](
        const auto &matrix_free, auto &dst, const auto &src, const auto cells) {
        FECellIntegrator phi_src(matrix_free);
        FECellIntegrator phi_dst(matrix_free);
        FECellIntegrator phi_weights(matrix_free);

        AlignedVector<VectorizedArrayType> tmp;

        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi_src.reinit(cell);
            phi_dst.reinit(cell);

            if (compressed_rw)
              compressed_rw->read_dof_values(src, phi_src);
            else
              phi_src.read_dof_values(src);

            if (weight_type != Restrictors::WeightingType::post)
              apply_weights_local(phi_weights, phi_src, true);

            fdm.apply_inverse(
              cell,
              ArrayView<VectorizedArrayType>(phi_dst.begin_dof_values(),
                                             phi_dst.dofs_per_cell),
              ArrayView<const VectorizedArrayType>(phi_src.begin_dof_values(),
                                                   phi_src.dofs_per_cell),
              tmp);

            if (weight_type != Restrictors::WeightingType::post)
              apply_weights_local(phi_weights, phi_dst, false);

            if (compressed_rw)
              compressed_rw->distribute_local_to_global(dst, phi_dst);
            else
              phi_dst.distribute_local_to_global(dst);
          }
      },
      dst,
      src,
      operation_before_matrix_vector_product,
      [&](const auto begin, const auto end) {
        if (weight_type == Restrictors::WeightingType::post)
          {
            const auto dst_ptr     = dst.begin();
            const auto weights_ptr = weights.begin();

            DEAL_II_OPENMP_SIMD_PRAGMA
            for (std::size_t i = begin; i < end; ++i)
              dst_ptr[i] *= weights_ptr[i];
          }

        if (operation_after_matrix_vector_product)
          operation_after_matrix_vector_product(begin, end);
      });
  }

  void
  apply_weights_local(FECellIntegrator &phi_weights,
                      FECellIntegrator &phi,
                      const bool        first_call) const
  {
    const unsigned int cell = phi.get_current_cell_index();

    if (weights_compressed_q2.size() > 0)
      {
        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post)))
          internal::weight_fe_q_dofs_by_entity<dim, -1, VectorizedArrayType>(
            &weights_compressed_q2[cell][0],
            1 /* TODO*/,
            fe_degree + 1,
            phi.begin_dof_values());
      }
    else
      {
        if ((first_call == true) &&
            (weight_type != Restrictors::WeightingType::none))
          {
            phi_weights.reinit(cell);
            phi_weights.read_dof_values_plain(weights);
          }

        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post)))
          {
            for (unsigned int i = 0; i < phi_weights.dofs_per_cell; ++i)
              phi.begin_dof_values()[i] *= phi_weights.begin_dof_values()[i];
          }
      }
  }

  void
  apply_weights_local(const unsigned int                  cell,
                      AlignedVector<VectorizedArrayType> &weights_local,
                      AlignedVector<VectorizedArrayType> &data,
                      const bool                          first_call) const
  {
    if (weights_compressed_q4.size() > 0)
      {
        AssertThrow(false, ExcNotImplemented());

        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post)))
          internal::
            weight_fe_q_dofs_by_entity<dim, -1, -1, VectorizedArrayType>(
              &weights_compressed_q4[cell][0],
              1 /* TODO*/,
              fe_degree + 1,
              n_overlap - 1,
              data.data());
      }
    else
      {
        if ((first_call == true) &&
            (weight_type != Restrictors::WeightingType::none))
          {
            internal::VectorReader<Number, VectorizedArrayType> reader;
            constraint_info.read_write_operation(reader,
                                                 weights,
                                                 weights_local.data(),
                                                 cell_ptr[cell],
                                                 cell_ptr[cell + 1] -
                                                   cell_ptr[cell],
                                                 weights_local.size(),
                                                 true);
          }

        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post)))
          {
            for (unsigned int i = 0; i < weights_local.size(); ++i)
              data[i] *= weights_local[i];
          }
      }
  }

  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  fe_degree;
  const unsigned int                                  n_overlap;
  const Restrictors::WeightingType                    weight_type;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_for_fdm;

  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
                            constraint_info;
  std::vector<unsigned int> cell_ptr;

  TensorProductMatrixSymmetricSumCollection<dim, VectorizedArrayType, n_rows_1d>
    fdm;

  mutable VectorType src_;
  mutable VectorType dst_;

  VectorType weights;
  AlignedVector<std::array<VectorizedArrayType, Utilities::pow(3, dim)>>
    weights_compressed_q2;
  AlignedVector<std::array<VectorizedArrayType, Utilities::pow(5, dim)>>
    weights_compressed_q4;

  std::shared_ptr<ConstraintInfoReduced> compressed_rw;
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