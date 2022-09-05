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

template <int dim, typename Number, typename VectorizedArrayType>
class MyOperator : public Subscriptor
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  MyOperator(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
  {}

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

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FECellIntegrator integrator(matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            integrator.reinit(cell);
            do_cell_integral_global(integrator, dst, src);
          }
      },
      dst,
      src,
      true);
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &MyOperator::do_cell_integral_local,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

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
    integrator.read_dof_values(src);
    do_cell_integral_local(integrator);
    integrator.distribute_local_to_global(dst);
  }
};

template <int dim, typename Number>
void
test(const unsigned int fe_degree, const unsigned int n_subdivision)
{
  const bool chebyshev_degree = 3;    // TODO
  const bool do_vmult         = true; // TODO

  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  FE_Q<dim>      fe(fe_degree);
  QGauss<dim>    quadrature(fe_degree + 1);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  tria.refine_global(
    GridGenerator::subdivided_hyper_cube_balanced(tria, n_subdivision));

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  if (true)
    DoFRenumbering::matrix_free_data_locality(dof_handler,
                                              constraints,
                                              additional_data);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  using OperatorType = MyOperator<dim, Number, VectorizedArrayType>;

  const OperatorType op(matrix_free);

  const auto precon_diag = std::make_shared<DiagonalMatrix<VectorType>>();
  op.compute_inverse_diagonal(precon_diag->get_vector());

  typename PreconditionChebyshev<OperatorType,
                                 VectorType,
                                 DiagonalMatrix<VectorType>>::AdditionalData
    chebyshev_ad;

  chebyshev_ad.preconditioner = precon_diag;
  chebyshev_ad.constraints.copy_from(constraints);
  chebyshev_ad.degree = chebyshev_degree;

  PreconditionChebyshev<OperatorType, VectorType, DiagonalMatrix<VectorType>>
    precon_chebyshev;

  precon_chebyshev.initialize(op, chebyshev_ad);

  VectorType src, dst;

  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);
  src = 1;

  if (do_vmult)
    precon_chebyshev.vmult(dst, src);
  else
    precon_chebyshev.step(dst, src);
}



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
  const bool         use_float     = (argc >= 5) ? std::atoi(argv[4]) : 1;


  if (dim == 2 && use_float)
    test<2, float>(fe_degree, n_refinements);
  else if (dim == 3 && use_float)
    test<3, float>(fe_degree, n_refinements);
  else if (dim == 2)
    test<2, double>(fe_degree, n_refinements);
  else if (dim == 3)
    test<3, double>(fe_degree, n_refinements);
  else
    AssertThrow(false, ExcInternalError());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
