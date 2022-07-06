#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

#include "include/preconditioners.h"

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

template <int dim>
void
test(const unsigned int fe_degree, const unsigned int n_global_refinements)
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
  pcout << " - n cells: " << tria.n_global_active_cells() << std::endl;
  pcout << " - n dofs:  " << dof_handler.n_dofs() << std::endl;
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

  pcout << "Running with different preconditioners:" << std::endl;

  // inverse diagonal matrix
  {
    DiagonalMatrix<VectorType> precondition;

    auto &vec = precondition.get_vector();
    vec.reinit(partitioner);

    for (const auto entry : laplace_matrix)
      if (entry.row() == entry.column())
        vec[entry.row()] = 1.0 / entry.value();

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << " - inverse diagonal matrix:         "
          << reduction_control.last_step() << std::endl;
  }

  // AMG
  {
    TrilinosWrappers::PreconditionAMG precondition;
    precondition.initialize(laplace_matrix);

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << " - AMG:                             "
          << reduction_control.last_step() << std::endl;
  }

  // ASM on partition level with AMG
  for (unsigned int i = 0; i < 2; ++i)
    {
      DomainPreconditioner<TrilinosWrappers::PreconditionAMG,
                           TrilinosWrappers::SparseMatrix,
                           TrilinosWrappers::SparsityPattern>
        precondition;

      if (i == 0)
        precondition.initialize(laplace_matrix,
                                sparsity_pattern,
                                partitioner->locally_owned_range(),
                                partitioner->ghost_indices());
      else
        precondition.initialize(laplace_matrix,
                                sparsity_pattern,
                                partitioner->locally_owned_range());

      ReductionControl reduction_control;

      SolverCG<VectorType> solver_cg(reduction_control);

      solution = 0;
      solver_cg.solve(laplace_matrix, solution, rhs, precondition);

      pcout << " - ASM on partition level with AMG: "
            << reduction_control.last_step() << std::endl;
    }

  // ASM on cell level
  {
    InverseCellBlockPreconditioner<double, dim> precondition(dof_handler);

    precondition.initialize(laplace_matrix, sparsity_pattern);

    ReductionControl reduction_control;

    SolverCG<VectorType> solver_cg(reduction_control);

    solution = 0;
    solver_cg.solve(laplace_matrix, solution, rhs, precondition);

    pcout << " - ASM on cell level:               "
          << reduction_control.last_step() << std::endl;
  }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 6;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements);
  else
    AssertThrow(false, ExcNotImplemented());
}
