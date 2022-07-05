#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() = default;

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)p;
    (void)component;

    return 1.0;
  }

private:
};

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
             const IndexSet &              active_index_set)
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

template <int dim>
void
test(const unsigned int fe_degree            = 1,
     const unsigned int n_global_refinements = 6)
{
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  pcout << "System statistics:" << std::endl;
  pcout << " -n cells: " << tria.n_global_active_cells() << std::endl;
  pcout << " -n dofs:  " << dof_handler.n_dofs() << std::endl;
  pcout << std::endl;

  QGauss<dim> quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // create system matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern;
  sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                          dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix laplace_matrix;
  laplace_matrix.reinit(sparsity_pattern);

  MatrixCreator::
    create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
      dof_handler, quadrature, laplace_matrix, nullptr, constraints);


  // create vectors
  VectorType solution, rhs;

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_active_dofs(dof_handler),
    dof_handler.get_communicator());

  solution.reinit(partitioner);
  rhs.reinit(partitioner);

  VectorTools::create_right_hand_side(
    dof_handler, quadrature, RightHandSide<dim>(), rhs, constraints);

  {
    TrilinosWrappers::PreconditionAMG precondition;
    precondition.initialize(laplace_matrix);

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << reduction_control.last_step() << std::endl;
  }

  {
    DomainPreconditioner<TrilinosWrappers::PreconditionAMG,
                         TrilinosWrappers::SparseMatrix,
                         TrilinosWrappers::SparsityPattern>
      precondition;

    precondition.initialize(laplace_matrix,
                            sparsity_pattern,
                            partitioner->locally_owned_range(),
                            partitioner->ghost_indices());

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << reduction_control.last_step() << std::endl;
  }

  {
    InverseCellBlockPreconditioner<double, dim> precondition(dof_handler);

    precondition.initialize(laplace_matrix, sparsity_pattern);

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << reduction_control.last_step() << std::endl;
  }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2>();
}