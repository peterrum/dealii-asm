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
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FECellIntegrator phi(matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            do_cell_integral_global(phi, dst, src);
          }
      },
      dst,
      src,
      operation_before_matrix_vector_product,
      operation_after_matrix_vector_product);
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

  const AffineConstraints<Number> &
  get_constraints() const
  {
    return matrix_free.get_affine_constraints();
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


template <typename VectorType>
class DiagonalMatrixPrePost : public Subscriptor
{
public:
  DiagonalMatrixPrePost(const DiagonalMatrix<VectorType> &op)
    : op(op)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

  void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product) const
  {
    const auto dst_ptr  = dst.begin();
    const auto src_ptr  = src.begin();
    const auto diag_ptr = op.get_vector().begin();

    const auto locally_owned_size = dst.locally_owned_size();

    for (unsigned int i = 0; i < locally_owned_size; i += 100)
      {
        const unsigned int begin = i;
        const unsigned int end   = std::min(begin + 100, locally_owned_size);

        operation_before_matrix_vector_product(begin, end);

        for (unsigned int j = begin; j < end; ++j)
          dst_ptr[j] = src_ptr[j] * diag_ptr[j];

        operation_after_matrix_vector_product(begin, end);
      }
  }

private:
  const DiagonalMatrix<VectorType> &op;
};


template <typename VectorType>
class DiagonalMatrixAdapter : public Subscriptor
{
public:
  DiagonalMatrixAdapter(const DiagonalMatrix<VectorType> &op)
    : op(op)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

private:
  const DiagonalMatrix<VectorType> &op;
};



template <int dim, typename Number>
void
test(const unsigned int fe_degree,
     const unsigned int n_subdivision,
     const std::string  preconditioner_type,
     const unsigned int optimization_level,
     const bool         do_vmult)
{
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  const unsigned int n_repetitions = 10;
  const unsigned int max_degree    = 5;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  FE_Q<dim>      fe(fe_degree);
  QGauss<dim>    quadrature(fe_degree + 1);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  tria.refine_global(
    GridGenerator::subdivided_hyper_cube_balanced(tria, n_subdivision));

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  pcout << "Info" << std::endl;
  pcout << " - n dofs: " << dof_handler.n_dofs() << std::endl;
  pcout << std::endl << std::endl;

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

  const auto precon_pre_post =
    std::make_shared<DiagonalMatrixPrePost<VectorType>>(*precon_diag);

  const auto precon_adapter =
    std::make_shared<DiagonalMatrixAdapter<VectorType>>(*precon_diag);

  const auto run = [&](const auto &op, const auto &precon) {
    VectorType src, dst;

    op.initialize_dof_vector(src);
    op.initialize_dof_vector(dst);
    src = 1;

    for (unsigned int c = 0; c < n_repetitions; ++c)
      {
        if (do_vmult)
          precon.vmult(dst, src);
        else
          precon.step(dst, src);
      }

    static unsigned int likwid_counter = 1;

    const std::string label =
      preconditioner_type + "_" + std::to_string(likwid_counter);

    MPI_Barrier(MPI_COMM_WORLD);
    LIKWID_MARKER_START(label.c_str());

    for (unsigned int c = 0; c < n_repetitions; ++c)
      {
        if (do_vmult)
          precon.vmult(dst, src);
        else
          precon.step(dst, src);
      }

    MPI_Barrier(MPI_COMM_WORLD);
    LIKWID_MARKER_STOP(label.c_str());

    likwid_counter++;
  };

  const auto run_chebyshev = [&](const auto &       op,
                                 const auto &       precon,
                                 const unsigned int chebyshev_degree) {
    using PreconditionerType = typename std::remove_cv<
      typename std::remove_reference<decltype(*precon)>::type>::type;

    using VectorType = typename OperatorType::VectorType;

    typename PreconditionChebyshev<OperatorType,
                                   VectorType,
                                   PreconditionerType>::AdditionalData
      chebyshev_ad;

    chebyshev_ad.preconditioner = precon;
    chebyshev_ad.constraints.copy_from(op.get_constraints());
    chebyshev_ad.degree = chebyshev_degree;

    PreconditionChebyshev<OperatorType, VectorType, PreconditionerType>
      precon_chebyshev;

    precon_chebyshev.initialize(op, chebyshev_ad);

    run(op, precon_chebyshev);
  };

  const auto run_relaxation = [&](const auto &       op,
                                  const auto &       precon,
                                  const unsigned int chebyshev_degree) {
    using PreconditionerType = typename std::remove_cv<
      typename std::remove_reference<decltype(*precon)>::type>::type;

    using VectorType = typename OperatorType::VectorType;

    typename PreconditionRelaxation<OperatorType,
                                    PreconditionerType>::AdditionalData
      relaxation_ad;

    relaxation_ad.preconditioner = precon;
    relaxation_ad.n_iterations   = chebyshev_degree;
    relaxation_ad.relaxation     = 1.1;

    PreconditionRelaxation<OperatorType, PreconditionerType> precon_relaxation;

    precon_relaxation.initialize(op, relaxation_ad);

    run(op, precon_relaxation);
  };

  for (unsigned int d = 1; d <= max_degree; ++d)
    {
      if (preconditioner_type == "chebyshev")
        {
          if (optimization_level == 0)
            {
              run_chebyshev(op, precon_adapter, d);
            }
          else if (optimization_level == 1)
            {
              run_chebyshev(op, precon_pre_post, d);
            }
          else if (optimization_level == 2)
            {
              run_chebyshev(op, precon_diag, d);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }
      else if (preconditioner_type == "relaxation")
        {
          if (optimization_level == 0)
            {
              run_relaxation(op, precon_adapter, d);
            }
          else if (optimization_level == 1)
            {
              run_relaxation(op, precon_pre_post, d);
            }
          else if (optimization_level == 2)
            {
              run_relaxation(op, precon_diag, d);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
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
  const unsigned int n_subdivision = (argc >= 4) ? std::atoi(argv[3]) : 6;
  const bool         use_float     = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const std::string  preconditioner_type =
    (argc >= 6) ? std::string(argv[5]) : std::string("chebyshev");
  const unsigned int optimization_level = (argc >= 7) ? std::atoi(argv[6]) : 2;
  const bool         do_vmult           = (argc >= 8) ? std::atoi(argv[7]) : 1;


  if (dim == 2 && use_float)
    test<2, float>(fe_degree,
                   n_subdivision,
                   preconditioner_type,
                   optimization_level,
                   do_vmult);
  else if (dim == 3 && use_float)
    test<3, float>(fe_degree,
                   n_subdivision,
                   preconditioner_type,
                   optimization_level,
                   do_vmult);
  else if (dim == 2)
    test<2, double>(fe_degree,
                    n_subdivision,
                    preconditioner_type,
                    optimization_level,
                    do_vmult);
  else if (dim == 3)
    test<3, double>(fe_degree,
                    n_subdivision,
                    preconditioner_type,
                    optimization_level,
                    do_vmult);
  else
    AssertThrow(false, ExcInternalError());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
