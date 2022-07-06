#pragma once

template <typename PreconditionerType,
          typename SparseMatrixType,
          typename SparsityPattern>
class DomainPreconditioner
{
public:
  DomainPreconditioner() = default;

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern & global_sparsity_pattern,
             const IndexSet &              local_index_set,
             const IndexSet &              active_index_set = {})
  {
    SparseMatrixTools::restrict_to_serial_sparse_matrix(global_sparse_matrix,
                                                        global_sparsity_pattern,
                                                        local_index_set,
                                                        active_index_set,
                                                        sparse_matrix,
                                                        sparsity_pattern);

    preconditioner.initialize(sparse_matrix);

    IndexSet union_index_set = local_index_set;
    union_index_set.add_indices(active_index_set);

    local_src.reinit(union_index_set.n_elements());
    local_dst.reinit(union_index_set.n_elements());
  }

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    src.update_ghost_values();

    for (unsigned int i = 0; i < local_src.size(); ++i)
      local_src[i] = src.local_element(i);

    preconditioner.vmult(local_dst, local_src);

    for (unsigned int i = 0; i < local_dst.size(); ++i)
      dst.local_element(i) = local_dst[i];

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  SparsityPattern    sparsity_pattern;
  SparseMatrixType   sparse_matrix;
  PreconditionerType preconditioner;

  mutable Vector<typename SparseMatrixType::value_type> local_src, local_dst;
};

template <typename Number, int dim, int spacedim = dim>
class InverseCellBlockPreconditioner
{
public:
  InverseCellBlockPreconditioner(const DoFHandler<dim, spacedim> &dof_handler)
    : dof_handler(dof_handler)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern & global_sparsity_pattern)
  {
    SparseMatrixTools::restrict_to_cells(global_sparse_matrix,
                                         global_sparsity_pattern,
                                         dof_handler,
                                         blocks);

    for (auto &block : blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();
  }

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.0;
    src.update_ghost_values();

    Vector<double> vector_src;
    Vector<double> vector_dst;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);

        cell->get_dof_values(src, vector_src);

        blocks[cell->active_cell_index()].vmult(vector_dst, vector_src);

        cell->distribute_local_to_global(vector_dst, dst);
      }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;
  std::vector<FullMatrix<Number>>  blocks;
};