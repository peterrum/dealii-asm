#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

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
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/tensor_product_matrix.h"

using namespace dealii;

#include "include/matrix_free.h"

template <typename OperatorType>
class MyOperator : public Subscriptor
{
public:
  using value_type = typename OperatorType::value_type;
  using VectorType = typename OperatorType::VectorType;

  MyOperator(const OperatorType &op)
    : op(op)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

  types::global_dof_index
  m() const
  {
    return op.m();
  }


  value_type
  el(unsigned int i, unsigned int j) const
  {
    return op.el(i, j);
  }

private:
  const OperatorType &op;
};

template <typename VectorType>
class MyDiagonalMatrix : public Subscriptor
{
public:
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

  VectorType &
  get_vector()
  {
    return op.get_vector();
  }

private:
  DiagonalMatrix<VectorType> op;
};

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap,
     const unsigned int chebyshev_degree)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<double>;

  const unsigned int mapping_degree = fe_degree;

  FE_Q<dim> fe(fe_degree);
  FE_Q<1>   fe_1D(fe_degree);

  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> quadrature_face(fe_degree + 1);
  QGauss<1>       quadrature_1D(fe_degree + 1);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);

  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  MappingQ<dim>      mapping(mapping_degree);
  MappingQCache<dim> mapping_q_cache(mapping_degree);

  mapping_q_cache.initialize(
    mapping,
    tria,
    [](const auto &, const auto &point) {
      Point<dim> result;

      return result;

      for (unsigned int d = 0; d < dim; ++d)
        result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                    std::sin(numbers::PI * point[d]) * 0.1;

      return result;
    },
    true);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
  constraints.close();

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(mapping_q_cache, dof_handler, constraints, quadrature);

  using OperatorType = PoissonOperator<dim, Number, VectorizedArrayType>;
  using MyOperatorType =
    MyOperator<PoissonOperator<dim, Number, VectorizedArrayType>>;
  using PreconditionerType =
    ASPoissonPreconditioner<dim, Number, VectorizedArrayType>;

  OperatorType   op(matrix_free);
  MyOperatorType my_op(op);

  VectorType src, dst;

  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);

  op.rhs(src);

  const auto precon_fdm = std::make_shared<PreconditionerType>(matrix_free,
                                                               n_overlap,
                                                               mapping_q_cache,
                                                               fe_1D,
                                                               quadrature_face,
                                                               quadrature_1D);

  const auto precon_diag = std::make_shared<DiagonalMatrix<VectorType>>();
  op.compute_inverse_diagonal(precon_diag->get_vector());

  const auto precon_my_diag = std::make_shared<MyDiagonalMatrix<VectorType>>();
  op.compute_inverse_diagonal(precon_my_diag->get_vector());

  PreconditionChebyshev<OperatorType, VectorType, PreconditionerType>
    precon_chebyshev_fdm;

  {
    typename PreconditionChebyshev<OperatorType,
                                   VectorType,
                                   PreconditionerType>::AdditionalData
      chebyshev_ad;

    chebyshev_ad.preconditioner = precon_fdm;
    chebyshev_ad.constraints.copy_from(constraints);
    chebyshev_ad.degree = chebyshev_degree;

    precon_chebyshev_fdm.initialize(op, chebyshev_ad);
  }

  PreconditionChebyshev<OperatorType, VectorType, DiagonalMatrix<VectorType>>
    precon_chebyshev_diag;

  {
    typename PreconditionChebyshev<OperatorType,
                                   VectorType,
                                   DiagonalMatrix<VectorType>>::AdditionalData
      chebyshev_ad;

    chebyshev_ad.preconditioner = precon_diag;
    chebyshev_ad.constraints.copy_from(constraints);
    chebyshev_ad.degree = chebyshev_degree;

    precon_chebyshev_diag.initialize(op, chebyshev_ad);
  }

  PreconditionChebyshev<MyOperatorType, VectorType, DiagonalMatrix<VectorType>>
    precon_chebyshev_diag_my_op;

  {
    typename PreconditionChebyshev<MyOperatorType,
                                   VectorType,
                                   DiagonalMatrix<VectorType>>::AdditionalData
      chebyshev_ad;

    chebyshev_ad.preconditioner = precon_diag;
    chebyshev_ad.constraints.copy_from(constraints);
    chebyshev_ad.degree = chebyshev_degree;

    precon_chebyshev_diag_my_op.initialize(my_op, chebyshev_ad);
  }

  PreconditionChebyshev<OperatorType, VectorType, MyDiagonalMatrix<VectorType>>
    precon_chebyshev_my_diag;

  {
    typename PreconditionChebyshev<OperatorType,
                                   VectorType,
                                   MyDiagonalMatrix<VectorType>>::AdditionalData
      chebyshev_ad;

    chebyshev_ad.preconditioner = precon_my_diag;
    chebyshev_ad.constraints.copy_from(constraints);
    chebyshev_ad.degree = chebyshev_degree;

    precon_chebyshev_my_diag.initialize(op, chebyshev_ad);
  }


  if (false)
    {
      const auto evs = precon_chebyshev_fdm.estimate_eigenvalues(src);

      pcout << evs.min_eigenvalue_estimate << " " << evs.max_eigenvalue_estimate
            << std::endl;

      pcout << 2.0 / (evs.min_eigenvalue_estimate + evs.max_eigenvalue_estimate)
            << std::endl;
    }

  const auto run = [](const auto &runnable) {
    double     time_total = 0.0;
    const auto timer      = std::chrono::system_clock::now();

    for (unsigned int i = 0; i < 100; ++i)
      runnable();

    time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() - timer)
                    .count() /
                  1e9;

    return time_total;
  };

  // vmult
  const auto time_total_0 = run([&]() { op.vmult(dst, src); });

  // fdm
  const auto time_total_1 = run([&]() { precon_fdm->vmult(dst, src); });
  // fdm + chebyshev
  const auto time_total_2 =
    run([&]() { precon_chebyshev_fdm.vmult(dst, src); });

  // diagonal
  const auto time_total_3 = run([&]() { precon_diag->vmult(dst, src); });
  // diagonal + chebyshev
  const auto time_total_4 =
    run([&]() { precon_chebyshev_diag.vmult(dst, src); });
  // diagonal + chebyshev (no pre/post)
  const auto time_total_5 =
    run([&]() { precon_chebyshev_diag_my_op.vmult(dst, src); });
  // diagonal + chebyshev (without exploiting the fact that we have a diagonal)
  const auto time_total_6 =
    run([&]() { precon_chebyshev_my_diag.vmult(dst, src); });

  pcout << dof_handler.n_dofs() << " " << time_total_0 << " " << time_total_1
        << " " << time_total_2 << " " << time_total_3 << " " << time_total_4
        << " " << time_total_5 << " " << time_total_6 << std::endl;

  if (false)
    {
      pcout << Utilities::MPI::sum(precon_fdm->memory_consumption(),
                                   MPI_COMM_WORLD)
            << std::endl;
      pcout << Utilities::MPI::sum(src.memory_consumption(), MPI_COMM_WORLD)
            << std::endl;
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
  const unsigned int n_overlap = (argc >= 5) ? std::atoi(argv[4]) : 1;

  for (unsigned int chebyshev_degree = 1; chebyshev_degree <= 5;
       ++chebyshev_degree)
    {
      if (dim == 2)
        test<2>(fe_degree, n_global_refinements, n_overlap, chebyshev_degree);
      else if (dim == 3)
        test<3>(fe_degree, n_global_refinements, n_overlap, chebyshev_degree);
      else
        AssertThrow(false, ExcNotImplemented());
    }
}