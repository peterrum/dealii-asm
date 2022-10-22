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
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_idr.h>
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
     const std::string  test_type,
     const std::string  orthogonalization_type,
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

  const unsigned int n_repetitions = 100;

  IterationNumberControl reduction_control(n_iterations, 1e-20);

  typename SolverGMRES<VectorType>::AdditionalData::OrthogonalizationStrategy
    orthogonalization_strategy;

  if (orthogonalization_type == "modified")
    orthogonalization_strategy = SolverGMRES<VectorType>::AdditionalData::
      OrthogonalizationStrategy::modified_gram_schmidt;
  else if (orthogonalization_type == "classical")
    orthogonalization_strategy = SolverGMRES<VectorType>::AdditionalData::
      OrthogonalizationStrategy::classical_gram_schmidt;
  else
    AssertThrow(false, ExcNotImplemented());

  unsigned int n_reorthogonalize = 0;

  const std::function<void(int)> reorthogonalize_signal = [&](int) -> void {
    n_reorthogonalize++;
  };

  if (test_type == "solver") // test GMRES solver
    {
      dst = 0.0;

      std::string label = "solver";

      for (unsigned int i = 0; i <= n_repetitions; ++i)
        {
          if (i != 0)
            LIKWID_MARKER_START(label.c_str());

          typename SolverGMRES<VectorType>::AdditionalData additional_data;
          additional_data.right_preconditioning = true;
          additional_data.orthogonalization_strategy =
            orthogonalization_strategy;

          SolverGMRES<VectorType> solver(reduction_control, additional_data);
          solver.connect_re_orthogonalization_slot(reorthogonalize_signal);
          solver.solve(op, dst, src, precon);

          if (i != 0)
            LIKWID_MARKER_STOP(label.c_str());
        }
    }
  else if (test_type == "iteration") // test iterated Gram-Schmidt algorithm
    {
      LinearAlgebra::distributed::Vector<Number> vv;
      op.initialize_dof_vector(vv);

      GrowingVectorMemory<VectorType> vmem;

      internal::SolverGMRESImplementation::TmpVectors<VectorType>
        orthogonal_vectors(n_iterations, vmem);

      for (unsigned int i = 0; i < n_iterations; ++i)
        orthogonal_vectors(i, vv) = 1.0;

      const unsigned int accumulated_iterations = 0;
      Vector<double>     h(n_iterations);

      bool reorthogonalize = false;

      std::string label = "iteration";

      boost::signals2::signal<void(int)> reorthogonalize_signal_boost;
      reorthogonalize_signal_boost.connect(reorthogonalize_signal);

      for (unsigned int i = 0; i <= n_repetitions; ++i)
        {
          if (i != 0)
            LIKWID_MARKER_START(label.c_str());

          internal::SolverGMRESImplementation::iterated_gram_schmidt(
            orthogonalization_strategy,
            orthogonal_vectors,
            n_iterations,
            accumulated_iterations,
            vv,
            h,
            reorthogonalize,
            reorthogonalize_signal_boost);

          if (i != 0)
            LIKWID_MARKER_STOP(label.c_str());
        }
    }
  else
    {
      AssertThrow(false, ExcInternalError());
    }

  table.add_value("test_type", test_type);
  table.add_value("orthogonalization_type", orthogonalization_type);
  table.add_value("n_dofs", dof_handler.n_dofs());
  table.add_value("n_rep", n_repetitions);
  table.add_value("n_it", reduction_control.last_step());
  table.add_value("n_reorthogonalize", n_reorthogonalize);

  pcout << dof_handler.n_dofs() << " x " << n_repetitions << " x "
        << reduction_control.last_step() << std::endl;
}


/**
 * mpirun -np 40 ./outer_solver_01 3 4 6 solver classical 3
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
  const std::string  test_type =
    (argc >= 5) ? std::string(argv[4]) : std::string("solver");
  const std::string orthogonalization_type =
    (argc >= 6) ? std::string(argv[5]) : std::string("classical");
  const unsigned int n_iterations_min = (argc >= 7) ? std::atoi(argv[6]) : 100;
  const unsigned int n_iterations_max =
    (argc >= 8) ? std::atoi(argv[7]) : n_iterations_min;
  const bool verbose = true;

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

  ConvergenceTable table;

  for (unsigned int n_iterations = n_iterations_min;
       n_iterations <= n_iterations_max;
       ++n_iterations)
    {
      if (dim == 2)
        test<2, double>(fe_degree,
                        n_refinements,
                        test_type,
                        orthogonalization_type,
                        n_iterations,
                        table);
      else if (dim == 3)
        test<3, double>(fe_degree,
                        n_refinements,
                        test_type,
                        orthogonalization_type,
                        n_iterations,
                        table);
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
