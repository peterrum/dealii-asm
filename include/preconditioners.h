#pragma once

#include "dof_tools.h"
#include "grid_generator.h"
#include "grid_tools.h"

enum class WeightingType
{
  none,
  left,
  right,
  symm
};

template <typename PreconditionerType,
          typename SparseMatrixType,
          typename SparsityPattern>
class DomainPreconditioner
{
public:
  DomainPreconditioner(const WeightingType weighting_type = WeightingType::none)
    : weighting_type(weighting_type)
  {}

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

    const auto comm = global_sparse_matrix.get_mpi_communicator();

    if (active_index_set.size() == 0)
      this->partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(local_index_set, comm);
    else
      this->partitioner =
        std::make_shared<Utilities::MPI::Partitioner>(local_index_set,
                                                      active_index_set,
                                                      comm);
  }

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    VectorType dst_, src_, multiplicity;
    dst_.reinit(partitioner);
    src_.reinit(partitioner);

    if (weighting_type != WeightingType::none)
      {
        multiplicity.reinit(partitioner);
        for (unsigned int i = 0; i < local_src.size(); ++i)
          multiplicity.local_element(i) = 1.0;
        multiplicity.compress(VectorOperation::add);

        for (auto &i : multiplicity)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);

        multiplicity.update_ghost_values();
      }

    src_.copy_locally_owned_data_from(src); // TODO: inplace
    src_.update_ghost_values();

    if (weighting_type == WeightingType::symm ||
        weighting_type == WeightingType::right)
      for (unsigned int i = 0; i < local_src.size(); ++i)
        src_.local_element(i) *= multiplicity.local_element(i);

    for (unsigned int i = 0; i < local_src.size(); ++i)
      local_src[i] = src_.local_element(i);

    preconditioner.vmult(local_dst, local_src);

    for (unsigned int i = 0; i < local_dst.size(); ++i)
      dst_.local_element(i) = local_dst[i];

    src_.zero_out_ghost_values();

    if (weighting_type == WeightingType::symm ||
        weighting_type == WeightingType::left)
      for (unsigned int i = 0; i < local_dst.size(); ++i)
        dst_.local_element(i) *= multiplicity.local_element(i);

    dst_.compress(VectorOperation::add);
    dst.copy_locally_owned_data_from(dst_); // TODO: inplace
  }

private:
  SparsityPattern    sparsity_pattern;
  SparseMatrixType   sparse_matrix;
  PreconditionerType preconditioner;

  std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

  mutable Vector<typename SparseMatrixType::value_type> local_src, local_dst;

  const WeightingType weighting_type;
};

template <typename Number, int dim, int spacedim = dim>
class InverseCellBlockPreconditioner
{
public:
  InverseCellBlockPreconditioner(
    const DoFHandler<dim, spacedim> &dof_handler,
    const WeightingType              weighting_type = WeightingType::none)
    : dof_handler(dof_handler)
    , weighting_type(weighting_type)
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

    Vector<double> vector_src, vector_dst, vector_weights;

    VectorType weights;

    if (weighting_type != WeightingType::none)
      {
        weights.reinit(src);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            vector_weights.reinit(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] = 1.0;

            cell->distribute_local_to_global(vector_weights, weights);
          }

        weights.compress(VectorOperation::add);
        for (auto &i : weights)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);
        weights.update_ghost_values();
      }


    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);
        if (weighting_type != WeightingType::none)
          vector_weights.reinit(dofs_per_cell);

        cell->get_dof_values(src, vector_src);
        if (weighting_type != WeightingType::none)
          cell->get_dof_values(weights, vector_weights);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::right)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_src[i] *= vector_weights[i];

        blocks[cell->active_cell_index()].vmult(vector_dst, vector_src);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::left)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_dst[i] *= vector_weights[i];

        cell->distribute_local_to_global(vector_dst, dst);
      }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;
  std::vector<FullMatrix<Number>>  blocks;

  const WeightingType weighting_type;
};



template <typename Number>
class MatrixView
{
public:
  using value_type = Number;

  virtual void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const = 0;

  virtual void
  invert()
  {
    AssertThrow(false, ExcNotImplemented());
  }

  virtual unsigned int
  size() const = 0;

  virtual unsigned int
  size(const unsigned int c) const = 0;

private:
};



template <typename MatrixType0, typename MatrixType1>
class CGMatrixView : public MatrixView<typename MatrixType0::value_type>
{
private:
  using Number = typename MatrixType0::value_type;

  template <typename MatrixType>
  class MatrixWrapper
  {
  public:
    MatrixWrapper(const MatrixType &matrix, const unsigned int c)
      : matrix(matrix)
      , c(c)
    {}

    void
    vmult(Vector<Number> &dst, const Vector<Number> &src) const
    {
      matrix.vmult(c, dst, src);
    }

  private:
    const MatrixType & matrix;
    const unsigned int c;
  };

public:
  CGMatrixView() = default;

  CGMatrixView(const std::shared_ptr<const MatrixType0> &matrix_0,
               const std::shared_ptr<const MatrixType1> &matrix_1)
  {
    this->initialize(matrix_0, matrix_1);
  }

  void
  initialize(const std::shared_ptr<const MatrixType0> &matrix_0,
             const std::shared_ptr<const MatrixType1> &matrix_1)
  {
    this->matrix_0 = matrix_0;
    this->matrix_1 = matrix_1;
  }

  void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const final
  {
    MatrixWrapper<MatrixType0> matrix(*matrix_0, c);
    MatrixWrapper<MatrixType1> precon(*matrix_1, c);

    IterationNumberControl   solver_control(10 /*TODO*/);
    SolverCG<Vector<Number>> solver_cg(solver_control);
    solver_cg.solve(matrix, dst, src, precon);
  }

  unsigned int
  size() const final
  {
    AssertDimension(matrix_0->size(), matrix_1->size());
    return matrix_0->size();
  }

  unsigned int
  size(const unsigned int c) const final
  {
    AssertDimension(matrix_0->size(c), matrix_1->size(c));
    return matrix_0->size(c);
  }

private:
  std::shared_ptr<const MatrixType0> matrix_0;
  std::shared_ptr<const MatrixType1> matrix_1;
};



template <typename Number>
class DiagonalMatrixView : public MatrixView<Number>
{
public:
  DiagonalMatrixView() = default;

  template <typename MatrixType>
  DiagonalMatrixView(const std::shared_ptr<MatrixType> &matrix)
  {
    this->initialize(matrix);
  }

  template <typename MatrixType>
  void
  initialize(const std::shared_ptr<MatrixType> &matrix)
  {
    diagonals.resize(matrix->size());

    Vector<Number> dst, src;

    for (unsigned int d = 0; d < diagonals.size(); ++d)
      {
        const unsigned int n = matrix->size(d);

        dst.reinit(n);
        src.reinit(n);

        diagonals[d].resize(n);

        for (unsigned int i = 0; i < n; ++i)
          {
            for (unsigned int j = 0; j < n; ++j)
              src[j] = (i == j);

            matrix->vmult(d, dst, src);

            diagonals[d][i] = dst[i];
          }
      }
  }
  virtual void
  invert()
  {
    for (auto &diagonal : diagonals)
      for (auto &entry : diagonal)
        entry = Number(1.0) / entry;
  }

  void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const final
  {
    for (unsigned int i = 0; i < src.size(); ++i)
      dst[i] = diagonals[c][i] * src[i];
  }

  unsigned int
  size() const final
  {
    return diagonals.size();
  }

  unsigned int
  size(const unsigned int c) const final
  {
    return diagonals[c].size();
  }

private:
  std::vector<std::vector<Number>> diagonals;
};



template <typename Number>
class RestrictedMatrixView : public MatrixView<Number>
{
public:
  RestrictedMatrixView() = default;

  template <typename RestrictorType,
            typename GlobalSparseMatrixType,
            typename GlobalSparsityPattern>
  RestrictedMatrixView(const std::shared_ptr<const RestrictorType> &restrictor,
                       const GlobalSparseMatrixType &global_sparse_matrix,
                       const GlobalSparsityPattern & global_sparsity_pattern)
  {
    this->initialize(restrictor, global_sparse_matrix, global_sparsity_pattern);
  }

  /**
   * Initialize class with a sparse matrix and a restrictor.
   */
  template <typename RestrictorType,
            typename GlobalSparseMatrixType,
            typename GlobalSparsityPattern>
  void
  initialize(const std::shared_ptr<const RestrictorType> &restrictor,
             const GlobalSparseMatrixType &               global_sparse_matrix,
             const GlobalSparsityPattern &global_sparsity_pattern)
  {
    dealii::SparseMatrixTools::restrict_to_full_matrices(
      global_sparse_matrix,
      global_sparsity_pattern,
      restrictor->get_indices(),
      this->blocks);
  }

  void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const final
  {
    blocks[c].vmult(dst, src);
  }

  virtual void
  invert()
  {
    for (auto &block : this->blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();
  }

  unsigned int
  size() const final
  {
    return blocks.size();
  }

  unsigned int
  size(const unsigned int c) const final
  {
    AssertDimension(blocks[c].m(), blocks[c].n());
    return blocks[c].m();
  }

private:
  std::vector<FullMatrix<Number>> blocks;
};



template <typename Number>
class SubMeshMatrixView : public MatrixView<Number>
{
public:
  struct AdditionalData
  {
    AdditionalData(const unsigned int sub_mesh_approximation = 1)
      : sub_mesh_approximation(sub_mesh_approximation)
    {}

    unsigned int sub_mesh_approximation;
  };

  SubMeshMatrixView() = default;

  template <typename OperatorType, typename RestrictorType>
  SubMeshMatrixView(const std::shared_ptr<const OperatorType> &  op,
                    const std::shared_ptr<const RestrictorType> &restrictor,
                    const AdditionalData &additional_data = AdditionalData())
  {
    this->initialize(op, restrictor, additional_data);
  }

  /**
   * Initialize class with a sparse matrix and a restrictor.
   */
  template <typename OperatorType, typename RestrictorType>
  void
  initialize(const std::shared_ptr<const OperatorType> &  op,
             const std::shared_ptr<const RestrictorType> &restrictor,
             const AdditionalData &additional_data = AdditionalData())
  {
    this->blocks.resize(op->get_triangulation().n_active_cells());

    for (const auto &cell : op->get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          // create subgrid
          auto sub_cells =
            dealii::GridTools::extract_all_surrounding_cells_cartesian<
              OperatorType::dimension>(cell,
                                       additional_data.sub_mesh_approximation);

          Triangulation<OperatorType::dimension> sub_tria;
          dealii::GridGenerator::create_mesh_from_cells(sub_cells, sub_tria);

          // define operator on subgrid
          OperatorType sub_op(op->get_mapping(),
                              sub_tria,
                              op->get_fe(),
                              op->get_quadrature());

          // make cells local
          auto sub_tria_iterator = sub_tria.begin();
          for (auto &sub_cell : sub_cells)
            if (sub_cell.state() == IteratorState::valid)
              sub_cell = sub_tria_iterator++;
            else
              sub_cell = sub_tria.end();
          Assert(sub_tria_iterator == sub_tria.end(), ExcInternalError());

          // extrac local dof indices
          const auto local_dof_indices =
            dealii::DoFTools::get_dof_indices_cell_with_overlap(
              sub_op.get_dof_handler(), sub_cells, restrictor->get_n_overlap());

          const unsigned int dofs_per_cell = local_dof_indices.size();

          // extract submatrix
          auto &cell_matrix = this->blocks[cell->active_cell_index()];
          cell_matrix       = FullMatrix<Number>(dofs_per_cell, dofs_per_cell);

          const auto &system_matrix    = sub_op.get_sparse_matrix();
          const auto &sparsity_pattern = sub_op.get_sparsity_pattern();

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) =
                sparsity_pattern.exists(local_dof_indices[i],
                                        local_dof_indices[j]) ?
                  system_matrix(local_dof_indices[i], local_dof_indices[j]) :
                  0;
        }
  }

  void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const final
  {
    blocks[c].vmult(dst, src);
  }

  virtual void
  invert()
  {
    for (auto &block : this->blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();
  }

  unsigned int
  size() const final
  {
    return blocks.size();
  }

  unsigned int
  size(const unsigned int c) const final
  {
    AssertDimension(blocks[c].m(), blocks[c].n());
    return blocks[c].m();
  }

private:
  std::vector<FullMatrix<Number>> blocks;
};



template <typename VectorType>
class PreconditionerBase
{
public:
  using vector_type = VectorType;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;
};



template <typename VectorType,
          typename InverseMatrixType,
          typename RestrictorType>
class RestrictedPreconditioner : public PreconditionerBase<VectorType>
{
public:
  using Number = typename VectorType::value_type;

  RestrictedPreconditioner() = default;


  RestrictedPreconditioner(
    const std::shared_ptr<const InverseMatrixType> &inverse_matrix,
    const std::shared_ptr<const RestrictorType> &   restrictor)
  {
    initialize(inverse_matrix, restrictor);
  }

  void
  initialize(const std::shared_ptr<const InverseMatrixType> &inverse_matrix,
             const std::shared_ptr<const RestrictorType> &   restrictor)
  {
    this->inverse_matrix = inverse_matrix;
    this->restrictor     = restrictor;
  }

  /**
   * Perform matrix-vector product by looping over all blocks,
   * restricting the source vector, apply the inverted
   * block matrix, and distributing the result back into the
   * global vector.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    dealii::Vector<Number> src_, dst_;

    dst = 0.0;
    src.update_ghost_values();

    for (unsigned int c = 0; c < restrictor->n_blocks(); ++c)
      {
        const unsigned int n_entries = restrictor->n_entries(c);

        if (n_entries == 0)
          continue;

        src_.reinit(n_entries);
        dst_.reinit(n_entries);

        restrictor->read_dof_values(c, src, src_);

        inverse_matrix->vmult(c, dst_, src_);

        restrictor->distribute_dof_values(c, dst_, dst);
      }

    dst.compress(dealii::VectorOperation::add);
    src.zero_out_ghost_values();
  }

private:
  std::shared_ptr<const InverseMatrixType> inverse_matrix;
  std::shared_ptr<const RestrictorType>    restrictor;
};
