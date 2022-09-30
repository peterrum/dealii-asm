#pragma once

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include "dof_tools.h"
#include "grid_tools.h"
#include "matrix_free_internal.h"
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
    const bool        compress_indices    = true,
    const std::string weight_local_global = "global")
    : matrix_free(matrix_free)
    , fe_degree(matrix_free.get_dof_handler().get_fe().tensor_degree())
    , n_overlap(n_overlap)
    , weight_type(weight_type)
    , do_weights_global(weight_local_global == "global")
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

    partitioner_fdm = matrix_free.get_vector_partitioner();

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

        partitioner_fdm = std::make_shared<Utilities::MPI::Partitioner>(
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

    const auto &task_info = matrix_free.get_task_info();

    const unsigned int n_dofs = partitioner_fdm->locally_owned_size() +
                                partitioner_fdm->n_ghost_indices();

    const unsigned int chunk_size_zero_vector =
      internal::MatrixFreeFunctions::DoFInfo::chunk_size_zero_vector;

    std::vector<unsigned int> touched_first_by(
      (n_dofs + chunk_size_zero_vector - 1) / chunk_size_zero_vector,
      numbers::invalid_unsigned_int);

    std::vector<unsigned int> touched_last_by(
      (n_dofs + chunk_size_zero_vector - 1) / chunk_size_zero_vector,
      numbers::invalid_unsigned_int);

    for (unsigned int part = 0, cell_counter = 0;
         part < task_info.partition_row_index.size() - 2;
         ++part)
      for (unsigned int chunk = task_info.partition_row_index[part];
           chunk < task_info.partition_row_index[part + 1];
           ++chunk)
        for (unsigned int cell = task_info.cell_partition_data[chunk];
             cell < task_info.cell_partition_data[chunk + 1];
             ++cell)
          {
            std::array<Table<2, VectorizedArrayType>, dim> Ms;
            std::array<Table<2, VectorizedArrayType>, dim> Ks;

            for (unsigned int v = 0;
                 v < matrix_free.n_active_entries_per_cell_batch(cell);
                 ++v, ++cell_counter)
              {
                const auto cell_iterator =
                  matrix_free.get_cell_iterator(cell, v);

                const auto cells =
                  dealii::GridTools::extract_all_surrounding_cells_cartesian<
                    dim>(cell_iterator,
                         n_overlap <= 1 ? 0 : sub_mesh_approximation);

                auto local_dofs =
                  dealii::DoFTools::get_dof_indices_cell_with_overlap(
                    dof_handler, cells, n_overlap, true);

                for (auto &i : local_dofs)
                  resolve_constraint(i);

                for (const auto &i : local_dofs)
                  if (i != numbers::invalid_unsigned_int)
                    {
                      const unsigned int myindex =
                        partitioner_fdm->global_to_local(i) /
                        chunk_size_zero_vector;
                      if (touched_first_by[myindex] ==
                          numbers::invalid_unsigned_int)
                        touched_first_by[myindex] = chunk;
                      touched_last_by[myindex] = chunk;
                    }

                constraint_info.read_dof_indices(cell_counter,
                                                 local_dofs,
                                                 partitioner_fdm);

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
                        Ms[d].reinit(Ms_scalar[d].size(0),
                                     Ms_scalar[d].size(1));
                        Ks[d].reinit(Ks_scalar[d].size(0),
                                     Ks_scalar[d].size(1));
                      }

                    for (unsigned int i = 0; i < Ms_scalar[d].size(0); ++i)
                      for (unsigned int j = 0; j < Ms_scalar[d].size(0); ++j)
                        Ms[d][i][j][v] = Ms_scalar[d][i][j];

                    for (unsigned int i = 0; i < Ks_scalar[d].size(0); ++i)
                      for (unsigned int j = 0; j < Ks_scalar[d].size(0); ++j)
                        Ks[d][i][j][v] = Ks_scalar[d][i][j];
                  }
              }

            cell_ptr.push_back(
              cell_ptr.back() +
              matrix_free.n_active_entries_per_cell_batch(cell));

            fdm.insert(cell, Ms, Ks);
          }

    fdm.finalize();

    constraint_info.finalize();

    src_.reinit(partitioner_fdm);
    dst_.reinit(partitioner_fdm);

    {
      const auto vector_partitioner = partitioner_fdm;

      // ensure that all indices are touched at least during the last round
      for (auto &index : touched_first_by)
        if (index == numbers::invalid_unsigned_int)
          index =
            task_info
              .partition_row_index[task_info.partition_row_index.size() - 2] -
            1;

      // lambda to convert from a map, with keys associated to the buckets by
      // which we sliced the index space, length chunk_size_zero_vector, and
      // values equal to the slice index which are touched by the respective
      // partition, to a "vectors-of-vectors" like data structure. Rather than
      // using the vectors, we set up a sparsity-pattern like structure where
      // one index specifies the start index (range_list_index), and the other
      // the actual ranges (range_list).
      auto convert_map_to_range_list =
        [=](const unsigned int n_partitions,
            const std::map<unsigned int, std::vector<unsigned int>> &ranges_in,
            std::vector<unsigned int> &range_list_index,
            std::vector<std::pair<unsigned int, unsigned int>> &range_list,
            const unsigned int                                  max_size) {
          range_list_index.resize(n_partitions + 1);
          range_list_index[0] = 0;
          range_list.clear();
          for (unsigned int partition = 0; partition < n_partitions;
               ++partition)
            {
              auto it = ranges_in.find(partition);
              if (it != ranges_in.end())
                {
                  for (unsigned int i = 0; i < it->second.size(); ++i)
                    {
                      const unsigned int first_i = i;
                      while (i + 1 < it->second.size() &&
                             it->second[i + 1] == it->second[i] + 1)
                        ++i;
                      range_list.emplace_back(
                        std::min(it->second[first_i] * chunk_size_zero_vector,
                                 max_size),
                        std::min((it->second[i] + 1) * chunk_size_zero_vector,
                                 max_size));
                    }
                  range_list_index[partition + 1] = range_list.size();
                }
              else
                range_list_index[partition + 1] = range_list_index[partition];
            }
        };

      // first we determine the ranges to zero the vector
      std::map<unsigned int, std::vector<unsigned int>> chunk_must_zero_vector;
      for (unsigned int i = 0; i < touched_first_by.size(); ++i)
        chunk_must_zero_vector[touched_first_by[i]].push_back(i);
      const unsigned int n_partitions =
        task_info.partition_row_index[task_info.partition_row_index.size() - 2];
      convert_map_to_range_list(n_partitions,
                                chunk_must_zero_vector,
                                vector_zero_range_list_index,
                                vector_zero_range_list,
                                vector_partitioner->locally_owned_size());

      // the other two operations only work on the local range (without
      // ghosts), so we skip the latter parts of the vector now
      touched_first_by.resize((vector_partitioner->locally_owned_size() +
                               chunk_size_zero_vector - 1) /
                              chunk_size_zero_vector);

      // set the import indices in the vector partitioner to one index higher
      // to indicate that we want to process it first. This additional index
      // is reflected in the argument 'n_partitions+1' in the
      // convert_map_to_range_list function below.
      for (auto it : vector_partitioner->import_indices())
        for (unsigned int i = it.first; i < it.second; ++i)
          touched_first_by[i / chunk_size_zero_vector] = n_partitions;
      std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_pre;
      for (unsigned int i = 0; i < touched_first_by.size(); ++i)
        chunk_must_do_pre[touched_first_by[i]].push_back(i);
      convert_map_to_range_list(n_partitions + 1,
                                chunk_must_do_pre,
                                cell_loop_pre_list_index,
                                cell_loop_pre_list,
                                vector_partitioner->locally_owned_size());

      touched_last_by.resize((vector_partitioner->locally_owned_size() +
                              chunk_size_zero_vector - 1) /
                             chunk_size_zero_vector);

      // set the indices which were not touched by the cell loop (i.e.,
      // constrained indices) to the last valid partition index. Since
      // partition_row_index contains one extra slot for ghosted faces (which
      // are not part of the cell/face loops), we use the second to last entry
      // in the partition list.
      for (auto &index : touched_last_by)
        if (index == numbers::invalid_unsigned_int)
          index =
            task_info
              .partition_row_index[task_info.partition_row_index.size() - 2] -
            1;
      for (auto it : vector_partitioner->import_indices())
        for (unsigned int i = it.first; i < it.second; ++i)
          touched_last_by[i / chunk_size_zero_vector] = n_partitions;
      std::map<unsigned int, std::vector<unsigned int>> chunk_must_do_post;
      for (unsigned int i = 0; i < touched_last_by.size(); ++i)
        chunk_must_do_post[touched_last_by[i]].push_back(i);
      convert_map_to_range_list(n_partitions + 1,
                                chunk_must_do_post,
                                cell_loop_post_list_index,
                                cell_loop_post_list,
                                vector_partitioner->locally_owned_size());
    }

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

      weights.reinit(partitioner_fdm);

      weights.copy_locally_owned_data_from(dst_);

      for (auto &i : weights)
        i = (i == 0.0) ?
              1.0 :
              (1.0 / ((weight_type == Restrictors::WeightingType::symm) ?
                        std::sqrt(i) :
                        i));

      weights.update_ghost_values();
    }

    if (weight_local_global == "compressed")
      if (fe_1D.degree >= 2 && n_overlap == 1)
        {
          weights_compressed_q2.resize(matrix_free.n_cell_batches());

          FECellIntegrator phi(matrix_free);

          for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
               ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values_plain(weights);

              const bool success =
                dealii::internal::compute_weights_fe_q_dofs_by_entity<
                  dim,
                  -1,
                  VectorizedArrayType>(phi.begin_dof_values(),
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

  VectorType &
  get_scratch_src_vector(const VectorType &src) const
  {
    if (do_weights_global && (weight_type == Restrictors::WeightingType::pre ||
                              weight_type == Restrictors::WeightingType::symm))
      return this->src_;
    else
      return const_cast<VectorType &>(src);
  }

  VectorType &
  get_scratch_src_vector(VectorType &src) const
  {
    return src;
  }

  /**
   * General matrix-vector product.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    vmult_internal(dst,
                   src,
                   get_scratch_src_vector(src),
                   [&](const auto start_range, const auto end_range) {
                     if (end_range > start_range)
                       std::memset(dst.begin() + start_range,
                                   0,
                                   sizeof(Number) * (end_range - start_range));
                   },
                   {});
  }

  /**
   * Matrix-vector product with pre- and post-operations to be used
   * by PreconditionRelaxation and PreconditionChebyshev.
   */
  virtual void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &pre_operation,
        const std::function<void(const unsigned int, const unsigned int)>
          &post_operation = {}) const
  {
    vmult_internal(
      dst, src, get_scratch_src_vector(src), pre_operation, post_operation);
  }

  /**
   * Matrix-vector product with pre- and post-operations to be used
   * by PreconditionRelaxation and PreconditionChebyshev.
   */
  virtual void
  vmult(VectorType &dst,
        VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &pre_operation,
        const std::function<void(const unsigned int, const unsigned int)>
          &post_operation = {}) const
  {
    vmult_internal(
      dst, src, get_scratch_src_vector(src), pre_operation, post_operation);
  }

  std::size_t
  memory_consumption() const
  {
    return fdm.memory_consumption();
  }

  std::shared_ptr<const Utilities::MPI::Partitioner>
  get_partitioner() const
  {
    return partitioner_fdm;
  }

  unsigned int
  n_fdm_instances() const
  {
    return fdm.storage_size();
  }


private:
  void
  vmult_internal(
    VectorType &      dst,
    const VectorType &src,
    VectorType &      src_scratch,
    const std::function<void(const unsigned int, const unsigned int)>
      &pre_operation,
    const std::function<void(const unsigned int, const unsigned int)>
      &post_operation) const
  {
    AlignedVector<VectorizedArrayType> tmp;
    AlignedVector<VectorizedArrayType> src_local;
    AlignedVector<VectorizedArrayType> dst_local;
    AlignedVector<VectorizedArrayType> weights_local;

    // version 1) of cell operation: overlap=1 -> use dealii::FEEvaluation
    const auto cell_operation_normal = [&](const auto &matrix_free,
                                           auto &      dst,
                                           const auto &src,
                                           const auto  cells) {
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

          if (do_weights_global == false)
            apply_weights_local(phi_weights, phi_src, true);

          fdm.apply_inverse(
            cell,
            ArrayView<VectorizedArrayType>(phi_dst.begin_dof_values(),
                                           phi_dst.dofs_per_cell),
            ArrayView<const VectorizedArrayType>(phi_src.begin_dof_values(),
                                                 phi_src.dofs_per_cell),
            tmp);

          if (do_weights_global == false)
            apply_weights_local(phi_weights, phi_dst, false);

          if (compressed_rw)
            compressed_rw->distribute_local_to_global(dst, phi_dst);
          else
            phi_dst.distribute_local_to_global(dst);
        }
    };

    // version 2) of cell operation: overlap>1 -> use ConstraintInfo and
    // work on own buffers
    const auto cell_operation_overlap =
      [&](const MatrixFree<dim, Number, VectorizedArrayType> &,
          VectorType &                                dst_ptr,
          const VectorType &                          src_ptr,
          const std::pair<unsigned int, unsigned int> cell_range) {
        src_local.resize_fast(
          Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
        dst_local.resize_fast(
          Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
        if (do_weights_global == false)
          weights_local.resize_fast(
            Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
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
            if (do_weights_global == false)
              apply_weights_local(cell, weights_local, src_local, true);

            // 3) cell operation: fast diagonalization method
            fdm.apply_inverse(
              cell,
              make_array_view(dst_local.begin(), dst_local.end()),
              make_array_view(src_local.begin(), src_local.end()),
              tmp);

            // 4) apply weights (optional)
            if (do_weights_global == false)
              apply_weights_local(cell, weights_local, dst_local, false);

            // 5) scatter
            internal::VectorDistributorLocalToGlobal<Number,
                                                     VectorizedArrayType>
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
      };

    // version 1) of pre operation: consistent partitioners
    const auto pre_operation_with_weighting = [&](const auto begin,
                                                  const auto end) {
      if (pre_operation)
        pre_operation(begin, end);

      if (do_weights_global &&
          (weight_type == Restrictors::WeightingType::pre ||
           weight_type == Restrictors::WeightingType::symm))
        {
          const auto src_scratch_ptr = src_scratch.begin();
          const auto src_ptr         = src.begin();
          const auto weights_ptr     = weights.begin();

          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            src_scratch_ptr[i] = src_ptr[i] * weights_ptr[i];
        }
    };

    // version 2) of pre operation: inconsistent partitioners -> copy src
    const auto pre_operation_with_copying_and_weighting = [&](const auto begin,
                                                              const auto end) {
      if (pre_operation)
        pre_operation(begin, end);

      const auto src_scratch_ptr = this->src_.begin();
      const auto src_ptr         = src.begin();
      const auto dst_scratch_ptr = this->dst_.begin();
      const auto weights_ptr     = weights.begin();

      if (do_weights_global &&
          (weight_type == Restrictors::WeightingType::pre ||
           weight_type == Restrictors::WeightingType::symm))
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            src_scratch_ptr[i] = src_ptr[i] * weights_ptr[i];
        }
      else
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            src_scratch_ptr[i] = src_ptr[i];
        }

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (std::size_t i = begin; i < end; ++i)
        dst_scratch_ptr[i] = 0.0; // note: zeroing
    };

    // version 1) of post operation: consistent partitioners
    const auto post_operation_with_weighting = [&](const auto begin,
                                                   const auto end) {
      if (do_weights_global &&
          (weight_type == Restrictors::WeightingType::post ||
           weight_type == Restrictors::WeightingType::symm))
        {
          const auto dst_ptr     = dst.begin();
          const auto weights_ptr = weights.begin();

          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            dst_ptr[i] *= weights_ptr[i];
        }

      if (post_operation)
        post_operation(begin, end);
    };

    // version 2) of post operation: inconsistent partitioners -> copy dst
    const auto post_operation_with_weighting_and_copying = [&](const auto begin,
                                                               const auto end) {
      const auto dst_scratch_ptr = this->dst_.begin();
      const auto dst_ptr         = dst.begin();
      const auto weights_ptr     = weights.begin();

      if (do_weights_global &&
          (weight_type == Restrictors::WeightingType::post ||
           weight_type == Restrictors::WeightingType::symm))
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            dst_ptr[i] += dst_scratch_ptr[i] * weights_ptr[i]; // note: add
        }
      else
        {
          DEAL_II_OPENMP_SIMD_PRAGMA
          for (std::size_t i = begin; i < end; ++i)
            dst_ptr[i] += dst_scratch_ptr[i]; // note: add
        }

      if (post_operation)
        post_operation(begin, end);
    };

    if ((partitioner_fdm.get() == matrix_free.get_vector_partitioner().get()) &&
        (partitioner_fdm.get() == src.get_partitioner().get()))
      {
        // version 1) with overlap=1 -> use dealii::MatrixFree
        matrix_free.template cell_loop<VectorType, VectorType>(
          cell_operation_normal,
          dst,
          src_scratch,
          pre_operation_with_weighting,
          post_operation_with_weighting);
      }
    else if (partitioner_fdm.get() == src.get_partitioner().get())
      {
        // version 2) with overlap>1 and consistent partitioner -> use
        // own matrix-free infrastructure
        VectorDataExchange<Number> exchanger_dst(partitioner_fdm, buffer_dst);
        VectorDataExchange<Number> exchanger_src(partitioner_fdm, buffer_src);

        MFWorker<dim, Number, VectorizedArrayType, VectorType> worker(
          matrix_free,
          cell_loop_pre_list_index,
          cell_loop_pre_list,
          cell_loop_post_list_index,
          cell_loop_post_list,
          exchanger_dst,
          exchanger_src,
          dst,
          src_scratch,
          cell_operation_overlap,
          pre_operation_with_weighting,
          post_operation_with_weighting);

        MFRunner runner;
        runner.loop(worker);
      }
    else
      {
        // version 3) with overlap>1 and inconsistent partitioner -> use
        // own matrix-free infrastructure and copy vectors on the fly
        VectorDataExchange<Number> exchanger_dst(partitioner_fdm, buffer_dst);
        VectorDataExchange<Number> exchanger_src(partitioner_fdm, buffer_src);

        MFWorker<dim, Number, VectorizedArrayType, VectorType> worker(
          matrix_free,
          cell_loop_pre_list_index,
          cell_loop_pre_list,
          cell_loop_post_list_index,
          cell_loop_post_list,
          exchanger_dst,
          exchanger_src,
          this->dst_,
          this->src_,
          cell_operation_overlap,
          pre_operation_with_copying_and_weighting,
          post_operation_with_weighting_and_copying);

        MFRunner runner;
        runner.loop(worker);
      }
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

  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  fe_degree;
  const unsigned int                                  n_overlap;
  const Restrictors::WeightingType                    weight_type;
  const bool                                          do_weights_global;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fdm;

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

  std::shared_ptr<ConstraintInfoReduced> compressed_rw;

  std::vector<unsigned int> vector_zero_range_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> vector_zero_range_list;
  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;
  std::vector<unsigned int>                          cell_loop_post_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_post_list;

  mutable dealii::AlignedVector<Number> buffer_dst;
  mutable dealii::AlignedVector<Number> buffer_src;
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
          &pre_operation,
        const std::function<void(const unsigned int, const unsigned int)>
          &post_operation) const
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
      pre_operation,
      post_operation);
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