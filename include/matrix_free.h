#pragma once

#include <deal.II/lac/tensor_product_matrix.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/numerics/tensor_product_matrix_creator.h>

#include "dof_tools.h"
#include "exceptions.h"
#include "grid_tools.h"
#include "matrix_free_internal.h"
#include "preconditioners.h"
#include "read_write_operation.h"
#include "restrictors.h"
#include "symmetry.h"
#include "tensor_product_matrix_creator.h"
#include "vector_access_reduced.h"

// clang-format off
#define EXPAND_OPERATIONS_RWV(OPERATION)                                \
  switch (patch_size_1d)                                                \
    {                                                                   \
      case  3: OPERATION((( 2 <= MAX_DEGREE_RW) ?  3 : -1), -1); break; \
      case  5: OPERATION((( 3 <= MAX_DEGREE_RW) ?  5 : -1), -1); break; \
      case  7: OPERATION((( 4 <= MAX_DEGREE_RW) ?  7 : -1), -1); break; \
      case  9: OPERATION((( 5 <= MAX_DEGREE_RW) ?  9 : -1), -1); break; \
      case 11: OPERATION((( 6 <= MAX_DEGREE_RW) ? 11 : -1), -1); break; \
      case 13: OPERATION((( 7 <= MAX_DEGREE_RW) ? 13 : -1), -1); break; \
      default:                                                          \
        OPERATION(-1, -1);                                              \
    }
// clang-format on

template <typename T>
std::tuple<T, T>
my_compute_prefix_sum(const T &value, const MPI_Comm &comm)
{
#ifndef DEAL_II_WITH_MPI
  (void)comm;
  return {0, value};
#else
  T prefix = {};

  int ierr = MPI_Exscan(&value,
                        &prefix,
                        1,
                        Utilities::MPI::mpi_type_id_for_type<decltype(value)>,
                        MPI_SUM,
                        comm);
  AssertThrowMPI(ierr);

  T sum = Utilities::MPI::sum(value, comm);

  return {prefix, sum};
#endif
}

template <int dim, typename Number, typename VectorizedArrayType>
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
    const std::string weight_local_global = "global",
    const bool        overlap_pre_post    = true,
    const bool        element_centric     = true)
    : matrix_free(matrix_free)
    , fe_degree(matrix_free.get_dof_handler().get_fe().tensor_degree())
    , n_overlap(n_overlap)
    , patch_size_1d(element_centric ? (fe_degree + 2 * n_overlap - 1) :
                                      (fe_degree * 2 - 1))
    , patch_size(Utilities::pow(patch_size_1d, dim))
    , weight_type(weight_type)
    , do_weights_global(weight_local_global == "global")
    , overlap_pre_post(overlap_pre_post)
    , element_centric(element_centric)
    , needs_compression(true)
  {
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
      if (i == numbers::invalid_dof_index)
        return;

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

    if (element_centric && (n_overlap == 1))
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

        for (const auto i :
             matrix_free.get_vector_partitioner()->locally_owned_range())
          ghost_indices.push_back(i);
        for (const auto i :
             matrix_free.get_vector_partitioner()->ghost_indices())
          ghost_indices.push_back(i);

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
                         element_centric ?
                           (n_overlap <= 1 ? 0 : sub_mesh_approximation) :
                           dim);

                const auto cells_vertex_patch =
                  collect_cells_for_vertex_patch(cells);

                auto local_dofs =
                  element_centric ?
                    dealii::DoFTools::get_dof_indices_cell_with_overlap(
                      dof_handler, cells, n_overlap, true) :
                    dealii::DoFTools::get_dof_indices_vertex_patch(
                      dof_handler, cells_vertex_patch);

                for (auto &i : local_dofs)
                  resolve_constraint(i);

                for (const auto &i : local_dofs)
                  if ((locally_owned_dofs.is_element(i) == false) &&
                      (i != numbers::invalid_dof_index))
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

    if ((element_centric == false) && compress_indices && (fe_degree >= 2))
      compressed_dof_indices_vertex_patch.resize(
        VectorizedArrayType::size() * matrix_free.n_cell_batches() *
          Utilities::pow(3, dim),
        dealii::numbers::invalid_unsigned_int);

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
                         element_centric ?
                           (n_overlap <= 1 ? 0 : sub_mesh_approximation) :
                           dim);

                const auto cells_vertex_patch =
                  collect_cells_for_vertex_patch(cells);

                auto local_dofs =
                  element_centric ?
                    dealii::DoFTools::get_dof_indices_cell_with_overlap(
                      dof_handler, cells, n_overlap, true) :
                    dealii::DoFTools::get_dof_indices_vertex_patch(
                      dof_handler, cells_vertex_patch);

                for (auto &i : local_dofs)
                  resolve_constraint(i);

#if false
                if(cell_counter == 0)
                {
                            for (unsigned int i = 0, c = 0; i < patch_size_1d;
                                 ++i)
                              {
                                for (unsigned int j = 0; j < patch_size_1d; ++j)
                                  {
                                      std::cout << local_dofs[c++] << " ";
                                  }
                                std::cout << std::endl;
                              }
                            std::cout << std::endl;
                }
#endif

                for (const auto &i : local_dofs)
                  if (i != numbers::invalid_dof_index)
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

                if (compressed_dof_indices_vertex_patch.size())
                  {
                    std::vector<unsigned int> compressed_dof_indices;

                    const auto success =
                      read_write_operation_setup(local_dofs,
                                                 dim,
                                                 patch_size_1d,
                                                 compressed_dof_indices,
                                                 partitioner_fdm);

                    AssertThrow(success, ExcInternalError());

                    for (unsigned int i = 0; i < compressed_dof_indices.size();
                         ++i)
                      compressed_dof_indices_vertex_patch
                        [cell * VectorizedArrayType::size() *
                           compressed_dof_indices.size() +
                         i * VectorizedArrayType::size() + v] =
                          compressed_dof_indices[i];
                  }

                const auto patch_extend =
                  harmonic_patch_extend[cell_iterator->active_cell_index()];

                const auto patch_extend_for_vertex_patch =
                  collect_patch_extend(patch_extend);

                const auto [Ms_scalar, Ks_scalar] =
                  element_centric ?
                    TensorProductMatrixCreator::
                      create_laplace_tensor_product_matrix<dim, Number>(
                        cell_iterator,
                        {1},
                        {2},
                        fe_1D,
                        quadrature_1D,
                        patch_extend,
                        n_overlap) :
                    TensorProductMatrixCreator::
                      create_laplace_tensor_product_matrix<dim, Number>(
                        fe_1D, quadrature_1D, patch_extend_for_vertex_patch);

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

    if (compressed_dof_indices_vertex_patch.size() > 0)
      {
        all_indices_uniform_vertex_patch.resize(matrix_free.n_cell_batches() *
                                                Utilities::pow(3, dim));

        for (unsigned int i = 0;
             i < matrix_free.n_cell_batches() * Utilities::pow(3, dim);
             ++i)
          {
            char not_constrained = 1;

            for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
              if (compressed_dof_indices_vertex_patch
                    [i * VectorizedArrayType::size() + v] ==
                  numbers::invalid_unsigned_int)
                not_constrained = 0;

            all_indices_uniform_vertex_patch[i] = not_constrained;
          }
      }

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

    unsigned int cell_batch_prefix = 0;

    if (weight_type == Restrictors::WeightingType::ras)
      {
        AlignedVector<VectorizedArrayType> dst__(patch_size);

        auto [prefix, sum] =
          my_compute_prefix_sum(matrix_free.n_cell_batches(), MPI_COMM_WORLD);
        cell_batch_prefix = prefix;

        const unsigned int invalid_cell_id =
          sum * VectorizedArrayType::size() + 1;

        dst_ = invalid_cell_id;

        auto norm_0 = dst_.l2_norm();

        unsigned int counter = 0;

        do
          {
            norm_0 = dst_.l2_norm();
            dst_.update_ghost_values();

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
                 ++cell)
              {
                for (unsigned int v = 0;
                     v < matrix_free.n_active_entries_per_cell_batch(cell);
                     ++v)
                  {
                    // gather
                    internal::VectorReader<Number, VectorizedArrayType> reader;
                    constraint_info.read_write_operation(reader,
                                                         dst_,
                                                         dst__.data(),
                                                         cell_ptr[cell] + v,
                                                         1,
                                                         dst__.size(),
                                                         true);

                    const auto predicate_1D = [&](const auto i) {
                      if (element_centric)
                        return (n_overlap - 1 <= i) &&
                               (i < (fe_degree + n_overlap));
                      else
                        return (((patch_size_1d / 2 - counter) <= i) &&
                                (i <= (patch_size_1d / 2 + counter)));
                    };

                    const auto predicate =
                      [&](const auto i, const auto j, const auto k) {
                        if (predicate_1D(i) == false)
                          return false;
                        if (dim >= 2 && (predicate_1D(j) == false))
                          return false;
                        if (dim == 3 && (predicate_1D(k) == false))
                          return false;

                        return true;
                      };

                    for (unsigned int k = 0, c = 0;
                         k < ((dim == 3) ? patch_size_1d : 1);
                         ++k)
                      for (unsigned int j = 0;
                           j < ((dim >= 2) ? patch_size_1d : 1);
                           ++j)
                        for (unsigned int i = 0; i < patch_size_1d; ++i, ++c)
                          if (predicate(i, j, k))
                            dst__[c][0] =
                              std::min<double>(dst__[c][0],
                                               (cell_batch_prefix + cell) *
                                                   VectorizedArrayType::size() +
                                                 v + 1);

                    // scatter
                    internal::VectorSetter<Number, VectorizedArrayType> writer;
                    constraint_info.read_write_operation(writer,
                                                         dst_,
                                                         dst__.data(),
                                                         cell_ptr[cell] + v,
                                                         1,
                                                         dst__.size(),
                                                         true);

#if false
                    if(cell == 0 && v == 0)
                    {
                    constraint_info.read_write_operation(reader,
                                                         dst_,
                                                         dst__.data(),
                                                         cell_ptr[cell] + v,
                                                         1,
                                                         dst__.size(),
                                                         true);

                    
                            for (unsigned int i = 0, c = 0; i < patch_size_1d;
                                 ++i)
                              {
                                for (unsigned int j = 0; j < patch_size_1d; ++j)
                                  {
                                      std::cout << dst__[c++][0] << " ";
                                  }
                                std::cout << std::endl;
                              }
                            std::cout << std::endl;
                    }
#endif
                  }
              }

            dst_.set_ghost_state(false);
            dst_.compress(VectorOperation::min);

            counter++;
        } while (norm_0 != dst_.l2_norm());


        if (element_centric)
          {
            bool succes = true;

            for (const auto i : dst_)
              succes &= (i == invalid_cell_id) || (i == 0) ||
                        ((prefix * VectorizedArrayType::size() + 1 <= i) &&
                         (i < (prefix + matrix_free.n_cell_batches()) *
                                  VectorizedArrayType::size() +
                                1));

            AssertThrow(succes, ExcNotImplemented());

            needs_compression = false;
          }

        weights.reinit(partitioner_fdm);
        weights.copy_locally_owned_data_from(dst_);
        weights.update_ghost_values();
      }
    else
      {
        AlignedVector<VectorizedArrayType> dst__(patch_size);

        dst_ = 0.0;

        for (auto &i : dst__)
          i = 1.0;

        for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
          {
            internal::VectorDistributorLocalToGlobal<Number,
                                                     VectorizedArrayType>
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
                0.0 :
                (1.0 / ((weight_type == Restrictors::WeightingType::symm) ?
                          std::sqrt(i) :
                          i));

        weights.update_ghost_values();
      }

    if (element_centric && (n_overlap == 1))
      {
        const bool actually_use_compression =
          (weight_local_global == "compressed" && fe_1D.degree >= 2);
        const bool actually_use_dg = (weight_local_global == "dg");

        if (actually_use_compression || actually_use_dg)
          {
            if (actually_use_compression)
              weights_compressed_q2.resize(matrix_free.n_cell_batches());

            if (actually_use_dg)
              weights_dg.reinit(matrix_free.n_cell_batches(), patch_size);

            FECellIntegrator phi(matrix_free);

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
                 ++cell)
              {
                phi.reinit(cell);
                phi.read_dof_values_plain(weights);

                auto weights_local = phi.begin_dof_values();

                if (weight_type == Restrictors::WeightingType::ras)
                  for (unsigned int i = 0; i < patch_size; ++i)
                    for (unsigned int v = 0;
                         v < matrix_free.n_active_entries_per_cell_batch(cell);
                         ++v)
                      {
                        if (weights_local[i][v] ==
                            ((cell_batch_prefix + cell) *
                               VectorizedArrayType::size() +
                             v + 1))
                          weights_local[i][v] = 1.0;
                        else
                          weights_local[i][v] = 0.0;
                      }

                if (actually_use_compression)
                  {
                    const bool success =
                      dealii::internal::compute_weights_fe_q_dofs_by_entity<
                        dim,
                        -1,
                        VectorizedArrayType>(
                        weights_local,
                        1,
                        patch_size_1d,
                        weights_compressed_q2[cell].begin());
                    AssertThrow(success, ExcInternalError());
                  }
                else if (actually_use_dg)
                  {
                    for (unsigned int i = 0; i < patch_size; ++i)
                      weights_dg[cell][i] = weights_local[i];
                  }
              }
          }
      }
    else
      {
        const bool actually_use_compression =
          (weight_local_global == "compressed" && fe_1D.degree >= 2 &&
           (element_centric == false));
        const bool actually_use_dg = (weight_local_global == "dg");

        if (actually_use_compression || actually_use_dg)
          {
            if (actually_use_compression)
              weights_compressed_q2.resize(matrix_free.n_cell_batches());

            if (actually_use_dg)
              weights_dg.reinit(matrix_free.n_cell_batches(), patch_size);

            AlignedVector<VectorizedArrayType> weights_local;
            weights_local.resize_fast(patch_size);

            for (unsigned int cell = 0; cell < matrix_free.n_cell_batches();
                 ++cell)
              {
                internal::VectorReader<Number, VectorizedArrayType> reader;

                for (auto &i : weights_local)
                  i = 0.0;

                constraint_info.read_write_operation(reader,
                                                     weights,
                                                     weights_local.data(),
                                                     cell_ptr[cell],
                                                     cell_ptr[cell + 1] -
                                                       cell_ptr[cell],
                                                     weights_local.size(),
                                                     true);

                if (weight_type == Restrictors::WeightingType::ras)
                  for (unsigned int i = 0; i < patch_size; ++i)
                    for (unsigned int v = 0;
                         v < matrix_free.n_active_entries_per_cell_batch(cell);
                         ++v)
                      {
                        if (weights_local[i][v] ==
                            ((cell_batch_prefix + cell) *
                               VectorizedArrayType::size() +
                             v + 1))
                          weights_local[i][v] = 1.0;
                        else
                          weights_local[i][v] = 0.0;
                      }

#if false
                if (true || weight_type == Restrictors::WeightingType::ras)
                  {
                    for (unsigned int v = 0;
                         v < matrix_free.n_active_entries_per_cell_batch(cell);
                         ++v)
                      {
                            for (unsigned int i = 0, c = 0; i < patch_size_1d;
                                 ++i)
                              {
                                for (unsigned int j = 0; j < patch_size_1d; ++j)
                                  {
                                      std::cout << weights_local[c++][v] << " ";
                                  }
                                std::cout << std::endl;
                              }
                            std::cout << std::endl;
                      }
                  }
#endif

                if (actually_use_compression)
                  {
                    const bool success = dealii::internal::
                      compute_weights_fe_q_dofs_by_entity_shifted<
                        dim,
                        -1,
                        VectorizedArrayType>(
                        weights_local.begin(),
                        1,
                        patch_size_1d,
                        weights_compressed_q2[cell].begin());
                    if (success == false)
                      {
                        for (unsigned int v = 0;
                             v < VectorizedArrayType::size();
                             ++v)
                          {
                            for (unsigned int i = 0, c = 0; i < patch_size_1d;
                                 ++i)
                              {
                                for (unsigned int j = 0; j < patch_size_1d; ++j)
                                  {
                                    for (unsigned int k = 0; k < patch_size_1d;
                                         ++k)
                                      std::cout << weights_local[c++][v] << " ";
                                    std::cout << std::endl;
                                  }
                                std::cout << std::endl;
                              }
                            std::cout << std::endl;
                          }
                        std::cout << std::endl;
                      }
                    AssertThrow(success, ExcInternalError());
                  }
                else if (actually_use_dg)
                  {
                    for (unsigned int i = 0; i < patch_size; ++i)
                      weights_dg[cell][i] = weights_local[i];
                  }
              }
          }
      }

    if (weight_type == Restrictors::WeightingType::none)
      {
        weights.reinit(0);
        weights_compressed_q2.clear();
      }
  }

  SymmetryType::SymmetryType
  is_symmetric() const
  {
    if (weight_type == Restrictors::WeightingType::none ||
        weight_type == Restrictors::WeightingType::symm)
      return SymmetryType::symmetric;
    else
      return SymmetryType::non_symmetric;
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
  vmult(VectorType &dst, const VectorType &src) const override
  {
    const auto pre_operation = [&](const auto start_range,
                                   const auto end_range) {
      if (end_range > start_range)
        std::memset(dst.begin() + start_range,
                    0,
                    sizeof(Number) * (end_range - start_range));
    };

    vmult_internal(dst, src, get_scratch_src_vector(src), pre_operation, {});
  }

  /**
   * General matrix-vector product.
   */
  void
  vmult(VectorType &dst, /*const*/ VectorType &src) const
  {
    const auto pre_operation = [&](const auto start_range,
                                   const auto end_range) {
      if (end_range > start_range)
        std::memset(dst.begin() + start_range,
                    0,
                    sizeof(Number) * (end_range - start_range));
    };

    vmult_internal(dst, src, get_scratch_src_vector(src), pre_operation, {});
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
  vmult(VectorType &          dst,
        /*const*/ VectorType &src,
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
        src_local.resize_fast(patch_size);
        dst_local.resize_fast(patch_size);
        if (do_weights_global == false)
          weights_local.resize_fast(patch_size);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            // 1) gather src (optional)
            internal::VectorReader<Number, VectorizedArrayType> reader;
            if (compressed_dof_indices_vertex_patch.size() > 0)
              {
                const auto indices =
                  compressed_dof_indices_vertex_patch.data() +
                  cell * VectorizedArrayType::size() *
                    dealii::Utilities::pow(3, dim);

                const auto mask = all_indices_uniform_vertex_patch.data() +
                                  cell * dealii::Utilities::pow(3, dim);

#define OPERATION(c, d)                      \
  AssertThrow(c != -1, ExcNotImplemented()); \
                                             \
  read_write_operation<dim, c>(              \
    reader, src_ptr, dim, patch_size_1d, indices, mask, src_local.data());

                EXPAND_OPERATIONS_RWV(OPERATION);
#undef OPERATION
              }
            else
              {
                constraint_info.read_write_operation(reader,
                                                     src_ptr,
                                                     src_local.data(),
                                                     cell_ptr[cell],
                                                     cell_ptr[cell + 1] -
                                                       cell_ptr[cell],
                                                     src_local.size(),
                                                     true);
              }

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
            if (compressed_dof_indices_vertex_patch.size() > 0)
              {
                const auto indices =
                  compressed_dof_indices_vertex_patch.data() +
                  cell * VectorizedArrayType::size() *
                    dealii::Utilities::pow(3, dim);

                const auto mask = all_indices_uniform_vertex_patch.data() +
                                  cell * dealii::Utilities::pow(3, dim);

#define OPERATION(c, d)                      \
  AssertThrow(c != -1, ExcNotImplemented()); \
                                             \
  read_write_operation<dim, c>(              \
    writer, dst_ptr, dim, patch_size_1d, indices, mask, dst_local.data());

                EXPAND_OPERATIONS_RWV(OPERATION);
#undef OPERATION
              }
            else
              {
                constraint_info.read_write_operation(writer,
                                                     dst_ptr,
                                                     dst_local.data(),
                                                     cell_ptr[cell],
                                                     cell_ptr[cell + 1] -
                                                       cell_ptr[cell],
                                                     dst_local.size(),
                                                     true);
              }
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
           weight_type == Restrictors::WeightingType::ras ||
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
           weight_type == Restrictors::WeightingType::ras ||
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
        if (overlap_pre_post)
          {
            VectorDataExchange<Number> exchanger_dst(partitioner_fdm,
                                                     buffer_dst);
            VectorDataExchange<Number> exchanger_src(partitioner_fdm,
                                                     buffer_src);

            MFWorker<dim, Number, VectorizedArrayType, VectorType> worker(
              matrix_free,
              matrix_free.get_dof_info().cell_loop_pre_list_index,
              matrix_free.get_dof_info().cell_loop_pre_list,
              matrix_free.get_dof_info().cell_loop_post_list_index,
              matrix_free.get_dof_info().cell_loop_post_list,
              exchanger_dst,
              exchanger_src,
              dst,
              src_scratch,
              cell_operation_normal,
              pre_operation_with_weighting,
              post_operation_with_weighting,
              needs_compression);

            MFRunner runner(overlap_pre_post);
            runner.loop(worker);
          }
        else
          {
            const unsigned int chunk_size_zero_vector =
              internal::MatrixFreeFunctions::DoFInfo::chunk_size_zero_vector;

            for (unsigned int i = 0; i < src.locally_owned_size();
                 i += chunk_size_zero_vector)
              pre_operation_with_weighting(
                i,
                std::min<unsigned int>(i + chunk_size_zero_vector,
                                       src.locally_owned_size()));

            matrix_free.template cell_loop<VectorType, VectorType>(
              cell_operation_normal, dst, src_scratch);

            for (unsigned int i = 0; i < src.locally_owned_size();
                 i += chunk_size_zero_vector)
              post_operation_with_weighting(
                i,
                std::min<unsigned int>(i + chunk_size_zero_vector,
                                       src.locally_owned_size()));
          }
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
          post_operation_with_weighting,
          needs_compression);

        MFRunner runner(overlap_pre_post);
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
          post_operation_with_weighting_and_copying,
          needs_compression);

        MFRunner runner(overlap_pre_post);
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
              weight_type == Restrictors::WeightingType::post ||
              weight_type == Restrictors::WeightingType::ras)))
          internal::weight_fe_q_dofs_by_entity<dim, -1, VectorizedArrayType>(
            &weights_compressed_q2[cell][0],
            1 /* TODO*/,
            patch_size_1d,
            phi.begin_dof_values());
      }
    else if (weights_dg.size(0) > 0)
      {
        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post ||
              weight_type == Restrictors::WeightingType::ras)))
          {
            for (unsigned int i = 0; i < weights_dg.size(1); ++i)
              phi.begin_dof_values()[i] *= weights_dg[cell][i];
          }
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
              weight_type == Restrictors::WeightingType::post ||
              weight_type == Restrictors::WeightingType::ras)))
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
    if (weights_compressed_q2.size() > 0)
      {
        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post ||
              weight_type == Restrictors::WeightingType::ras)))
          internal::
            weight_fe_q_dofs_by_entity_shifted<dim, -1, VectorizedArrayType>(
              &weights_compressed_q2[cell][0],
              1 /* TODO*/,
              patch_size_1d,
              data.begin());
      }
    else if (weights_dg.size(0) > 0)
      {
        if (((first_call == true) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::pre)) ||
            ((first_call == false) &&
             (weight_type == Restrictors::WeightingType::symm ||
              weight_type == Restrictors::WeightingType::post ||
              weight_type == Restrictors::WeightingType::ras)))
          {
            for (unsigned int i = 0; i < weights_dg.size(1); ++i)
              data[i] *= weights_dg[cell][i];
          }
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

  static std::array<typename Triangulation<dim>::cell_iterator,
                    Utilities::pow(2, dim)>
  collect_cells_for_vertex_patch(
    const std::array<typename Triangulation<dim>::cell_iterator,
                     Utilities::pow(3, dim)> &cells_all)
  {
    std::array<typename Triangulation<dim>::cell_iterator,
               Utilities::pow(2, dim)>
      cells;

    for (unsigned int k = 0; k < ((dim == 3) ? 2 : 1); ++k)
      for (unsigned int j = 0; j < ((dim >= 2) ? 2 : 1); ++j)
        for (unsigned int i = 0; i < 2; ++i)
          cells[4 * k + 2 * j + i] =
            cells_all[9 * ((dim == 3) ? (k + 1) : 0) +
                      3 * ((dim >= 2) ? (j + 1) : 0) + (i + 1)];

    return cells;
  }

  static dealii::ndarray<double, dim, 2>
  collect_patch_extend(const dealii::ndarray<double, dim, 3> &patch_extend_all)
  {
    dealii::ndarray<double, dim, 2> patch_extend;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_extend[d][0] =
          (patch_extend_all[d][1] != 0.0) ? patch_extend_all[d][1] : 1.0;
        patch_extend[d][1] =
          (patch_extend_all[d][2] != 0.0) ? patch_extend_all[d][2] : 1.0;
      }

    return patch_extend;
  }

  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  fe_degree;
  const unsigned int                                  n_overlap;
  const unsigned int                                  patch_size_1d;
  const unsigned int                                  patch_size;
  const Restrictors::WeightingType                    weight_type;
  const bool                                          do_weights_global;
  const bool                                          overlap_pre_post;
  const bool                                          element_centric;
  bool                                                needs_compression;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fdm;

  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
                            constraint_info;
  std::vector<unsigned int> cell_ptr;

  TensorProductMatrixSymmetricSumCollection<dim, VectorizedArrayType> fdm;

  mutable VectorType src_;
  mutable VectorType dst_;

  VectorType weights;
  AlignedVector<std::array<VectorizedArrayType, Utilities::pow(3, dim)>>
    weights_compressed_q2;

  Table<2, VectorizedArrayType> weights_dg;

  std::shared_ptr<ConstraintInfoReduced> compressed_rw;

  std::vector<unsigned int> vector_zero_range_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> vector_zero_range_list;
  std::vector<unsigned int>                          cell_loop_pre_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_pre_list;
  std::vector<unsigned int>                          cell_loop_post_list_index;
  std::vector<std::pair<unsigned int, unsigned int>> cell_loop_post_list;

  mutable dealii::AlignedVector<Number> buffer_dst;
  mutable dealii::AlignedVector<Number> buffer_src;

  std::vector<unsigned int>  compressed_dof_indices_vertex_patch;
  std::vector<unsigned char> all_indices_uniform_vertex_patch;
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
