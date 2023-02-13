#pragma once

template <typename Number>
struct VectorDataExchange
{
  VectorDataExchange(const std::shared_ptr<const Utilities::MPI::Partitioner>
                       &                    embedded_partitioner,
                     AlignedVector<Number> &buffer)
    : embedded_partitioner(embedded_partitioner)
    , buffer(buffer)
  {}

  template <typename VectorType>
  void
  update_ghost_values_start(const VectorType &vec) const
  {
    const auto &vector_partitioner = vec.get_partitioner();

    buffer.resize_fast(embedded_partitioner->n_import_indices());

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
  }

  template <typename VectorType>
  void
  update_ghost_values_finish(const VectorType &vec) const
  {
    const auto &vector_partitioner = vec.get_partitioner();

    embedded_partitioner
      ->template export_to_ghosted_array_finish<Number, MemorySpace::Host>(
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);

    vec.set_ghost_state(true);
  }

  template <typename VectorType>
  void
  compress_start(VectorType &vec) const
  {
    const auto &vector_partitioner = vec.get_partitioner();

    buffer.resize_fast(embedded_partitioner->n_import_indices());

    embedded_partitioner
      ->template import_from_ghosted_array_start<Number, MemorySpace::Host>(
        dealii::VectorOperation::add,
        0,
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
        requests);
  }

  template <typename VectorType>
  void
  compress_finish(VectorType &vec) const
  {
    const auto &vector_partitioner = vec.get_partitioner();

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

  template <typename VectorType>
  void
  zero_out_ghost_values(const VectorType &vec) const
  {
    const auto &vector_partitioner = vec.get_partitioner();

    ArrayView<Number> ghost_array(
      const_cast<LinearAlgebra::distributed::Vector<Number> &>(vec).begin() +
        vector_partitioner->locally_owned_size(),
      vector_partitioner->n_ghost_indices());

    for (const auto &my_ghosts :
         embedded_partitioner->ghost_indices_within_larger_ghost_set())
      for (unsigned int j = my_ghosts.first; j < my_ghosts.second; ++j)
        ghost_array[j] = 0.;

    vec.set_ghost_state(false);
  }

private:
  const std::shared_ptr<const Utilities::MPI::Partitioner> embedded_partitioner;
  dealii::AlignedVector<Number> &                          buffer;
  mutable std::vector<MPI_Request>                         requests;
};



template <int dim,
          typename Number,
          typename VectorizedArrayType,
          typename VectorType>
struct MFWorker
{
public:
  MFWorker(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const std::vector<unsigned int> &cell_loop_pre_list_index,
    const std::vector<std::pair<unsigned int, unsigned int>>
      &                              cell_loop_pre_list,
    const std::vector<unsigned int> &cell_loop_post_list_index,
    const std::vector<std::pair<unsigned int, unsigned int>>
      &                               cell_loop_post_list,
    const VectorDataExchange<Number> &exchanger_dst,
    const VectorDataExchange<Number> &exchanger_src,
    VectorType &                      dst,
    const VectorType &                src,
    const std::function<
      void(const MatrixFree<dim, Number, VectorizedArrayType> &,
           VectorType &,
           const VectorType &,
           const std::pair<unsigned int, unsigned int>)> &cell_function,
    const std::function<void(unsigned int, unsigned int)>
      &operation_before_loop,
    const std::function<void(unsigned int, unsigned int)> &operation_after_loop,
    const bool needs_compression = true)
    : matrix_free(matrix_free)
    , cell_loop_pre_list_index(cell_loop_pre_list_index)
    , cell_loop_pre_list(cell_loop_pre_list)
    , cell_loop_post_list_index(cell_loop_post_list_index)
    , cell_loop_post_list(cell_loop_post_list)
    , exchanger_dst(exchanger_dst)
    , exchanger_src(exchanger_src)
    , dst(dst)
    , src(src)
    , cell_function(cell_function)
    , operation_before_loop(operation_before_loop)
    , operation_after_loop(operation_after_loop)
    , zero_dst_vector_setting(false)
    , needs_compression(needs_compression)
  {}

  const std::vector<unsigned int> &
  get_partition_row_index()
  {
    return matrix_free.get_task_info().partition_row_index;
  }

  virtual void
  vector_update_ghosts_start()
  {
    exchanger_src.update_ghost_values_start(src);
  }

  virtual void
  vector_update_ghosts_finish()
  {
    exchanger_src.update_ghost_values_finish(src);
  }

  virtual void
  vector_compress_start()
  {
    if (needs_compression)
      exchanger_dst.compress_start(dst);
  }

  virtual void
  vector_compress_finish()
  {
    if (needs_compression)
      exchanger_dst.compress_finish(dst);
    exchanger_src.zero_out_ghost_values(src);
  }

  virtual void
  zero_dst_vector_range(const unsigned int)
  {
    AssertThrow(zero_dst_vector_setting == false, ExcNotImplemented());
  }

  virtual void
  cell_loop_pre_range(const unsigned int range_index)
  {
    if (operation_before_loop)
      {
        if (range_index == numbers::invalid_unsigned_int)
          {
            operation_before_loop(0, src.locally_owned_size());
          }
        else
          {
            AssertIndexRange(range_index, cell_loop_pre_list_index.size() - 1);
            for (unsigned int id = cell_loop_pre_list_index[range_index];
                 id != cell_loop_pre_list_index[range_index + 1];
                 ++id)
              operation_before_loop(cell_loop_pre_list[id].first,
                                    cell_loop_pre_list[id].second);
          }
      }
  }

  virtual void
  cell_loop_post_range(const unsigned int range_index)
  {
    if (operation_after_loop)
      {
        // Run unit matrix operation on constrained dofs if we are at the
        // last range
        const std::vector<unsigned int> &partition_row_index =
          matrix_free.get_task_info().partition_row_index;
        if (range_index ==
            partition_row_index[partition_row_index.size() - 2] - 1)
          apply_operation_to_constrained_dofs(
            matrix_free.get_constrained_dofs(), src, dst);

        if (range_index == numbers::invalid_unsigned_int)
          {
            operation_after_loop(0, src.locally_owned_size());
          }
        else
          {
            AssertIndexRange(range_index, cell_loop_post_list_index.size() - 1);
            for (unsigned int id = cell_loop_post_list_index[range_index];
                 id != cell_loop_post_list_index[range_index + 1];
                 ++id)
              operation_after_loop(cell_loop_post_list[id].first,
                                   cell_loop_post_list[id].second);
          }
      }
  }

  void
  apply_operation_to_constrained_dofs(
    const std::vector<unsigned int> &constrained_dofs,
    const VectorType &               src,
    VectorType &                     dst)
  {
    for (const unsigned int i : constrained_dofs)
      dst.local_element(i) = src.local_element(i);
  }

  virtual void
  cell(const unsigned int range_index)
  {
    if (cell_function)
      {
        const auto &task_info           = matrix_free.get_task_info();
        const auto &cell_partition_data = task_info.cell_partition_data;

        cell_function(matrix_free,
                      dst,
                      src,
                      std::pair<unsigned int, unsigned int>{
                        cell_partition_data[range_index],
                        cell_partition_data[range_index + 1]});
      }
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

  const std::vector<unsigned int> &cell_loop_pre_list_index;
  const std::vector<std::pair<unsigned int, unsigned int>> &cell_loop_pre_list;
  const std::vector<unsigned int> &cell_loop_post_list_index;
  const std::vector<std::pair<unsigned int, unsigned int>> &cell_loop_post_list;

  const VectorDataExchange<Number> &exchanger_dst;
  const VectorDataExchange<Number> &exchanger_src;
  VectorType &                      dst;
  const VectorType &                src;
  const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType> &,
                           VectorType &,
                           const VectorType &,
                           const std::pair<unsigned int, unsigned int>)>
                                                        cell_function;
  const std::function<void(unsigned int, unsigned int)> operation_before_loop;
  const std::function<void(unsigned int, unsigned int)> operation_after_loop;
  const bool                                            zero_dst_vector_setting;
  const bool                                            needs_compression;
};

struct MFRunner
{
  MFRunner(const bool overlap_pre_post                      = true,
           const bool overlap_communication_and_computation = false)
    : overlap_pre_post(overlap_pre_post)
    , overlap_communication_and_computation(
        overlap_communication_and_computation)
  {
    AssertThrow(overlap_communication_and_computation == false,
                ExcNotImplemented());
  }

  template <typename WorkerType>
  void
  loop(WorkerType &worker) const
  {
    const auto &partition_row_index = worker.get_partition_row_index();

    if (overlap_pre_post)
      worker.cell_loop_pre_range(
        partition_row_index[partition_row_index.size() - 2]);
    else
      worker.cell_loop_pre_range(numbers::invalid_unsigned_int);

    worker.vector_update_ghosts_start();

    if (overlap_communication_and_computation == false)
      worker.vector_update_ghosts_finish();

    for (unsigned int part = 0; part < partition_row_index.size() - 2; ++part)
      {
        if (overlap_communication_and_computation && (part == 1))
          worker.vector_update_ghosts_finish();

        for (unsigned int i = partition_row_index[part];
             i < partition_row_index[part + 1];
             ++i)
          {
            if (overlap_pre_post)
              worker.cell_loop_pre_range(i);

            worker.zero_dst_vector_range(i); // note: not used

            worker.cell(i);

            if (overlap_pre_post)
              worker.cell_loop_post_range(i);
          }

        if (overlap_communication_and_computation && (part == 1))
          worker.vector_compress_start();
      }

    if (overlap_communication_and_computation == false)
      worker.vector_compress_start();
    worker.vector_compress_finish();

    if (overlap_pre_post)
      worker.cell_loop_post_range(
        partition_row_index[partition_row_index.size() - 2]);
    else
      worker.cell_loop_post_range(numbers::invalid_unsigned_int);
  }

private:
  const bool overlap_pre_post;
  const bool overlap_communication_and_computation;
};
