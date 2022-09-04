#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/restrictors.h"

using namespace dealii;

#include "include/matrix_free.h"
#include "include/operator.h"
#include "include/vector_access_reduced.h"

template <typename VectorType>
class MyDiagonalMatrix
{
public:
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    diagonal_matrix.vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    diagonal_matrix.vmult(dst, src);
  }

  VectorType &
  get_vector()
  {
    return diagonal_matrix.get_vector();
  }

private:
  DiagonalMatrix<VectorType> diagonal_matrix;
};


template <int dim, typename Number = double>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const std::string  type,
     const unsigned int n_iterations,
     ConvergenceTable & table)
{
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  // grid
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, 0, 1, false);
  tria.refine_global(n_global_refinements);

  // dofs
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  AffineConstraints<Number> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  // matrixfree
  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(fe_degree + 1);

  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  // operator
  LaplaceOperatorMatrixFree<dim, Number, VectorizedArrayType> op(matrix_free);

  // preconditioner
  MyDiagonalMatrix<VectorType> precon;
  op.compute_inverse_diagonal(precon.get_vector());

  VectorType src, dst;

  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);

  src = 1.0;
  constraints.set_zero(src);

  const unsigned int n_repetitions = 10;

  IterationNumberControl reduction_control(n_iterations, 1e-20);

  static unsigned int counter = 0;

  double time = 999.0;

  for (unsigned int i = 0; i <= n_repetitions; ++i)
    {
      dst = 0.0;

      std::string label = "solver_";

      if (counter < 10)
        label = label + "000" + std::to_string(counter);
      else if (counter < 100)
        label = label + "00" + std::to_string(counter);
      else if (counter < 1000)
        label = label + "0" + std::to_string(counter);

      if (i != 0)
        LIKWID_MARKER_START(label.c_str());

      const auto timer = std::chrono::system_clock::now();

      if (type == "CG")
        {
          SolverCG<VectorType> solver(reduction_control);
          solver.solve(op, dst, src, precon);
        }
      else if (type == "FCG")
        {
          SolverFlexibleCG<VectorType> solver(reduction_control);
          solver.solve(op, dst, src, precon);
        }
      else if (type == "GMRES")
        {
          typename SolverGMRES<VectorType>::AdditionalData additional_data;
          additional_data.right_preconditioning = true;

          SolverGMRES<VectorType> solver(reduction_control, additional_data);
          solver.solve(op, dst, src, precon);
        }
      else if (type == "FGMRES")
        {
          SolverFGMRES<VectorType> solver(reduction_control);
          solver.solve(op, dst, src, precon);
        }
      else
        {
          AssertThrow(false, ExcMessage("Solver <" + type + "> is not known!"))
        }

      time = std::min(time,
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - timer)
                          .count() /
                        1e9);

      if (i != 0)
        LIKWID_MARKER_STOP(label.c_str());
    }

  counter++;

  table.add_value("type", type);
  table.add_value("n_dofs", dof_handler.n_dofs());
  table.add_value("n_rep", n_repetitions);
  table.add_value("n_it", reduction_control.last_step());
  table.add_value("time", time);

  pcout << dof_handler.n_dofs() << " x " << n_repetitions << " x "
        << reduction_control.last_step() << std::endl;
}

/**
 * mpirun -np 40 ./outer_solver_01 3 4 6 GMRES  1 40
 * mpirun -np 40 ./outer_solver_01 3 4 6 CG     1 40
 * mpirun -np 40 ./outer_solver_01 3 4 6 FGMRES 1 40
 * mpirun -np 40 ./outer_solver_01 3 4 6 FCG    1 40
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const unsigned int dim           = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree     = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_refinements = (argc >= 4) ? std::atoi(argv[3]) : 6;
  const std::string  type =
    (argc >= 5) ? std::string(argv[4]) : std::string("cg");
  const unsigned int n_iterations_min = (argc >= 6) ? std::atoi(argv[5]) : 100;
  const unsigned int n_iterations_max = (argc >= 7) ? std::atoi(argv[6]) : 100;
  const bool         verbose          = true;

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

  ConvergenceTable table;

  for (unsigned int n_iterations = n_iterations_min;
       n_iterations <= n_iterations_max;
       ++n_iterations)
    {
      if (dim == 2)
        test<2, double>(fe_degree, n_refinements, type, n_iterations, table);
      else if (dim == 3)
        test<3, double>(fe_degree, n_refinements, type, n_iterations, table);
      else
        AssertThrow(false, ExcNotImplemented());


      if (is_root && verbose)
        {
          table.write_text(std::cout, ConvergenceTable::org_mode_table);
          std::cout << std::endl;
        }
    }

  if (is_root)
    {
      table.write_text(std::cout, ConvergenceTable::org_mode_table);
      std::cout << std::endl;
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
