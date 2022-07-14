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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace dealii;

#include "include/preconditioners.h"
#include "include/restrictors.h"

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

boost::property_tree::ptree
try_get_child(const boost::property_tree::ptree params, std::string label)
{
  try
    {
      return params.get_child(label);
    }
  catch (const boost::wrapexcept<boost::property_tree::ptree_bad_path> &)
    {
      return {};
    }
}

template <typename MatrixType, typename PreconditionerType, typename VectorType>
std::shared_ptr<ReductionControl>
solve(const MatrixType &                               A,
      VectorType &                                     x,
      const VectorType &                               b,
      const std::shared_ptr<const PreconditionerType> &preconditioner,
      const boost::property_tree::ptree                params)
{
  const auto max_iterations = params.get<unsigned int>("max iterations", 100);
  const auto abs_tolerance  = params.get<double>("abs tolerance", 1e-10);
  const auto rel_tolerance  = params.get<double>("rel tolerance", 1e-2);
  const auto type           = params.get<std::string>("type", "");

  auto reduction_control = std::make_shared<ReductionControl>(max_iterations,
                                                              abs_tolerance,
                                                              rel_tolerance);

  x = 0;

  if (type == "CG")
    {
      SolverCG<VectorType> solver(*reduction_control);
      solver.solve(A, x, b, *preconditioner);
    }
  else if (type == "GMRES")
    {
      SolverGMRES<VectorType> solver(*reduction_control);
      solver.solve(A, x, b, *preconditioner);
    }
  else
    {
      AssertThrow(false, ExcMessage("Solver <" + type + "> is not known!"))
    }

  return reduction_control;
}



template <int dim, typename VectorType>
std::shared_ptr<const PreconditionerBase<VectorType>>
create_system_preconditioner(
  DoFHandler<dim> &                        dof_handler,
  const TrilinosWrappers::SparseMatrix &   laplace_matrix,
  const TrilinosWrappers::SparsityPattern &sparsity_pattern,
  const boost::property_tree::ptree        params)
{
  const auto type = params.get<std::string>("type", "");

  if (type == "AdditiveSchwarzPreconditioner")
    {
      using RestictorType = Restrictors::ElementCenteredRestrictor<VectorType>;

      typename RestictorType::AdditionalData restrictor_ad;

      restrictor_ad.n_overlap      = 2;
      restrictor_ad.weighting_type = Restrictors::WeightingType::symm;

      const auto restrictor =
        std::make_shared<const RestictorType>(dof_handler, restrictor_ad);

      return std::make_shared<
        const AdditiveSchwarzPreconditioner<VectorType, RestictorType>>(
        restrictor, laplace_matrix, sparsity_pattern);
    }

  AssertThrow(false, ExcNotImplemented());

  return {};
}



template <int dim>
void
test(const boost::property_tree::ptree params)
{
  const unsigned int fe_degree = params.get<unsigned int>("degree", 1);
  const unsigned int n_global_refinements =
    params.get<unsigned int>("n refinements", 6);

  const auto solver_parameters = try_get_child(params, "solver");
  const auto preconditioner_parameters =
    try_get_child(params, "preconditioner");

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

  std::shared_ptr<ReductionControl> reduction_control;

  // ASM on cell level
  if (true)
    {
      const auto preconditioner = create_system_preconditioner<dim, VectorType>(
        dof_handler,
        laplace_matrix,
        sparsity_pattern,
        preconditioner_parameters);

      reduction_control =
        solve(laplace_matrix, solution, rhs, preconditioner, solver_parameters);
    }

  pcout << " - ASM on cell level:               "
        << reduction_control->last_step() << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef DEBUG
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "DEBUG!" << std::endl;
#endif

  AssertThrow(argc == 2, ExcMessage("You need to provide a JSON file!"));

  // get parameters
  boost::property_tree::ptree params;
  boost::property_tree::read_json(argv[1], params);

  const unsigned int dim = params.get<unsigned int>("dim", 2);

  if (dim == 2)
    test<2>(params);
  else
    AssertThrow(false, ExcNotImplemented());
}
