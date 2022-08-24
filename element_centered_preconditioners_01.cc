#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
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

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace dealii;

#include "include/kershaw.h"
#include "include/matrix_free.h"
#include "include/multigrid.h"
#include "include/preconditioners.h"
#include "include/restrictors.h"

#define COMPILE_MB 1
#define COMPILE_MF 1
#define COMPILE_2D 1
#define COMPILE_3D 1
#define MAX_N_ROWS_FDM 8

static bool print_timings;

template <typename T>
using print_timings_t = decltype(std::declval<T const>().print_timings());

template <typename T>
constexpr bool has_timing_functionality =
  dealii::internal::is_supported_operation<print_timings_t, T>;

// clang-format off
#define EXPAND_OPERATIONS(OPERATION)                                     \
  switch (n_rows)                                                        \
    {                                                                    \
      case  2: OPERATION((( 2 <= MAX_N_ROWS_FDM) ?  2 : -1), -1); break; \
      case  3: OPERATION((( 3 <= MAX_N_ROWS_FDM) ?  3 : -1), -1); break; \
      case  4: OPERATION((( 4 <= MAX_N_ROWS_FDM) ?  4 : -1), -1); break; \
      case  5: OPERATION((( 5 <= MAX_N_ROWS_FDM) ?  5 : -1), -1); break; \
      case  6: OPERATION((( 6 <= MAX_N_ROWS_FDM) ?  6 : -1), -1); break; \
      case  7: OPERATION((( 7 <= MAX_N_ROWS_FDM) ?  7 : -1), -1); break; \
      case  8: OPERATION((( 8 <= MAX_N_ROWS_FDM) ?  8 : -1), -1); break; \
      case  9: OPERATION((( 9 <= MAX_N_ROWS_FDM) ?  9 : -1), -1); break; \
      case 10: OPERATION(((10 <= MAX_N_ROWS_FDM) ? 10 : -1), -1); break; \
      default:                                                           \
        OPERATION(-1, -1);                                               \
    }
// clang-format on

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


Restrictors::WeightingType
get_weighting_type(const boost::property_tree::ptree params)
{
  const auto type = params.get<std::string>("weighting type", "symm");

  if (type == "symm")
    return Restrictors::WeightingType::symm;
  else if (type == "pre")
    return Restrictors::WeightingType::pre;
  else if (type == "post")
    return Restrictors::WeightingType::post;
  else if (type == "none")
    return Restrictors::WeightingType::none;

  AssertThrow(false, ExcMessage("Weighting type <" + type + "> is not known!"))

    return Restrictors::WeightingType::none;
}

template <typename MatrixType, typename PreconditionerType, typename VectorType>
std::shared_ptr<ReductionControl>
solve(const MatrixType &                              A,
      VectorType &                                    x,
      const VectorType &                              b,
      const std::shared_ptr<const PreconditionerType> preconditioner,
      const boost::property_tree::ptree               params,
      ConvergenceTable &                              table)
{
  const auto max_iterations = params.get<unsigned int>("max iterations", 1000);
  const auto abs_tolerance  = params.get<double>("abs tolerance", 1e-10);
  const auto rel_tolerance  = params.get<double>("rel tolerance", 1e-2);
  const auto type           = params.get<std::string>("type", "");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << " - Solving with " << type << std::endl;
  pcout << "   - max iterations: " << max_iterations << std::endl;
  pcout << "   - abs tolerance:  " << abs_tolerance << std::endl;
  pcout << "   - rel tolrance:   " << rel_tolerance << std::endl;

  auto reduction_control = std::make_shared<ReductionControl>(max_iterations,
                                                              abs_tolerance,
                                                              rel_tolerance);

  const auto dispatch = [&]() {
    x = 0;

    if (type == "CG")
      {
        SolverCG<VectorType> solver(*reduction_control);
        solver.solve(A, x, b, *preconditioner);
      }
    else if (type == "FCG")
      {
        SolverFlexibleCG<VectorType> solver(*reduction_control);
        solver.solve(A, x, b, *preconditioner);
      }
    else if (type == "GMRES")
      {
        typename SolverGMRES<VectorType>::AdditionalData additional_data;
        additional_data.right_preconditioning = true;

        SolverGMRES<VectorType> solver(*reduction_control, additional_data);
        solver.solve(A, x, b, *preconditioner);
      }
    else if (type == "FGMRES")
      {
        SolverFGMRES<VectorType> solver(*reduction_control);
        solver.solve(A, x, b, *preconditioner);
      }
    else
      {
        AssertThrow(false, ExcMessage("Solver <" + type + "> is not known!"))
      }
  };

  dispatch(); // warm up

  if constexpr (has_timing_functionality<PreconditionerType>)
    preconditioner->clear_timings();

  const auto timer = std::chrono::system_clock::now();
  dispatch();
  const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - timer)
                        .count() /
                      1e9;

  pcout << "   - n iterations:   " << reduction_control->last_step()
        << std::endl;
  pcout << "   - time:           " << time << " #" << std::endl;
  pcout << std::endl;


  table.add_value("it", reduction_control->last_step());

  if (print_timings)
    {
      table.add_value("time", time);

      if constexpr (has_timing_functionality<PreconditionerType>)
        preconditioner->print_timings();
    }

  return reduction_control;
}



template <typename OperatorType>
std::shared_ptr<const OperatorType>
get_approximation(const OperatorType &              op,
                  const boost::property_tree::ptree params)
{
  std::shared_ptr<const OperatorType> op_approx;

  const std::string matrix_approximation =
    params.get<std::string>("matrix approximation", "none");

  if (matrix_approximation == "none")
    {
      op_approx.reset(&op, [](auto *) {
        // nothing to do
      });
    }
  else if (matrix_approximation == "lobatto")
    {
      const unsigned int fe_degree = op.get_fe().tensor_degree();
      const unsigned int dim       = OperatorType::dimension;

      const auto subdivision_point =
        QGaussLobatto<1>(fe_degree + 1).get_points();
      const FE_Q_iso_Q1<dim> fe_q1_n(subdivision_point);
      const QIterated<dim>   quad_q1_n(QGauss<1>(2), subdivision_point);

      op_approx = std::make_shared<OperatorType>(op.get_mapping(),
                                                 op.get_triangulation(),
                                                 fe_q1_n,
                                                 quad_q1_n);
    }
  else if (matrix_approximation == "equidistant")
    {
      const unsigned int fe_degree = op.get_fe().tensor_degree();
      const unsigned int dim       = OperatorType::dimension;

      const FE_Q_iso_Q1<dim> fe_q1_h(fe_degree);
      const QIterated<dim>   quad_q1_h(QGauss<1>(2), fe_degree + 1);

      op_approx = std::make_shared<OperatorType>(op.get_mapping(),
                                                 op.get_triangulation(),
                                                 fe_q1_h,
                                                 quad_q1_h);
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Matrix approximation <" + matrix_approximation +
                             "> is not known!"))
    }

  return op_approx;
}



template <typename OperatorType, typename PreconditionerType>
std::shared_ptr<const PreconditionChebyshev<OperatorType,
                                            typename OperatorType::vector_type,
                                            PreconditionerType>>
create_chebyshev_preconditioner(
  const OperatorType &                       op,
  const std::shared_ptr<PreconditionerType> &precon,
  const boost::property_tree::ptree          params)
{
  using ChebyshevPreconditionerType =
    PreconditionChebyshev<OperatorType,
                          typename OperatorType::vector_type,
                          PreconditionerType>;

  typename ChebyshevPreconditionerType::AdditionalData
    chebyshev_additional_data;

  chebyshev_additional_data.preconditioner = precon;
  chebyshev_additional_data.constraints.copy_from(op.get_constraints());
  chebyshev_additional_data.degree = params.get<unsigned int>("degree", 3);
  chebyshev_additional_data.smoothing_range     = 20;
  chebyshev_additional_data.eig_cg_n_iterations = 20;

  const auto ev_algorithm =
    params.get<std::string>("ev algorithm", "power iteration");

  if (ev_algorithm == "lanczos")
    {
      chebyshev_additional_data.eigenvalue_algorithm =
        ChebyshevPreconditionerType::AdditionalData::EigenvalueAlgorithm::
          lanczos;
    }
  else if (ev_algorithm == "power iteration")
    {
      chebyshev_additional_data.eigenvalue_algorithm =
        ChebyshevPreconditionerType::AdditionalData::EigenvalueAlgorithm::
          power_iteration;
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Eigen-value algorithm <" + ev_algorithm +
                             "> is not known!"))
    }

  auto chebyshev = std::make_shared<ChebyshevPreconditionerType>();
  chebyshev->initialize(op, chebyshev_additional_data);

  return chebyshev;
}



template <typename OperatorType>
std::shared_ptr<
  const ASPoissonPreconditionerBase<typename OperatorType::vector_type>>
create_fdm_preconditioner(const OperatorType &              op,
                          const boost::property_tree::ptree params)
{
  using VectorType = typename OperatorType::vector_type;
  using Number     = typename VectorType::value_type;

  if constexpr (OperatorType::is_matrix_free())
    {
      ConditionalOStream pcout(
        std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      pcout << "- Create system preconditioner: FDM" << std::endl;

      const int dim             = OperatorType::dimension;
      using VectorizedArrayType = typename OperatorType::vectorized_array_type;

      const auto &matrix_free = op.get_matrix_free();
      const auto &mapping     = op.get_mapping();
      const auto &quadrature  = op.get_quadrature();

      const unsigned int fe_degree =
        matrix_free.get_dof_handler().get_fe().degree;

      FE_Q<1> fe_1D(fe_degree);

      const auto quadrature_1D = quadrature.get_tensor_basis()[0];
      const Quadrature<dim - 1> quadrature_face(quadrature_1D);

      const unsigned int n_overlap =
        std::min(params.get<unsigned int>("n overlap", 1), fe_degree);
      const auto weight_type = get_weighting_type(params);

      const unsigned int sub_mesh_approximation =
        params.get<unsigned int>("sub mesh approximation",
                                 OperatorType::dimension);

      const auto reuse_partitioner =
        params.get<bool>("reuse partitioner", true);

      pcout << "    - n overlap:              " << n_overlap << std::endl;
      pcout << "    - sub mesh approximation: " << sub_mesh_approximation
            << std::endl;
      pcout << "    - reuse partitioner:      "
            << (reuse_partitioner ? "true" : "false") << std::endl;
      pcout << std::endl;

      const unsigned int n_rows = fe_degree + 2 * n_overlap - 1;

      std::shared_ptr<const ASPoissonPreconditionerBase<VectorType>> precon;

#define OPERATION(c, d)                                                  \
  if (c == -1)                                                           \
    pcout << "Warning: FDM with <" + std::to_string(n_rows) +            \
               "> is not precompiled!"                                   \
          << std::endl;                                                  \
                                                                         \
  precon = std::make_shared<                                             \
    const ASPoissonPreconditioner<dim, Number, VectorizedArrayType, c>>( \
    matrix_free,                                                         \
    n_overlap,                                                           \
    sub_mesh_approximation,                                              \
    mapping,                                                             \
    fe_1D,                                                               \
    quadrature_face,                                                     \
    quadrature_1D,                                                       \
    weight_type);

      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      if (reuse_partitioner)
        op.set_partitioner(precon->get_partitioner());

      return precon;
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("FDM can only used with matrix-free operator!"));
      return {};
    }
}



template <typename OperatorType>
std::shared_ptr<const PreconditionerBase<typename OperatorType::vector_type>>
create_system_preconditioner(const OperatorType &              op,
                             const boost::property_tree::ptree params)
{
  using VectorType = typename OperatorType::vector_type;
  using Number     = typename VectorType::value_type;

  const auto type = params.get<std::string>("type", "");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  if (type == "Relaxation")
    {
      const auto preconditioner_parameters =
        try_get_child(params, "preconditioner");

      const auto preconditioner_type =
        preconditioner_parameters.get<std::string>("type", "");

      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto setup_chebshev = [&](const auto precon) {
        using RelaxationPreconditionerType = PreconditionRelaxation<
          OperatorType,
          typename std::remove_cv<
            typename std::remove_reference<decltype(*precon)>::type>::type>;

        const auto degree = params.get<unsigned int>("degree", 3);
        auto       omega  = params.get<double>("omega", 0.0);


        pcout << "- Create system preconditioner: Relaxation" << std::endl;
        pcout << "    - degree: " << degree << std::endl;

        if (omega == 0.0)
          {
            const auto chebyshev =
              create_chebyshev_preconditioner(op, precon, params);

            VectorType vec;
            op.initialize_dof_vector(vec);
            const auto evs = chebyshev->estimate_eigenvalues(vec);

            const unsigned int smoothing_range = 20;

            const double alpha =
              (smoothing_range > 1. ?
                 evs.max_eigenvalue_estimate / smoothing_range :
                 std::min(0.9 * evs.max_eigenvalue_estimate,
                          evs.min_eigenvalue_estimate));

            omega = 2.0 / (alpha + evs.max_eigenvalue_estimate);

            pcout << "    - min ev: " << evs.min_eigenvalue_estimate
                  << std::endl;
            pcout << "    - max ev: " << evs.max_eigenvalue_estimate
                  << std::endl;
            pcout << std::endl;
          }


        pcout << "    - omega:  " << omega << std::endl;

        typename RelaxationPreconditionerType::AdditionalData additional_data;

        additional_data.preconditioner = precon;
        additional_data.n_iterations   = degree;
        additional_data.relaxation     = omega;

        auto relexation = std::make_shared<RelaxationPreconditionerType>();
        relexation->initialize(op, additional_data);

        return std::make_shared<
          PreconditionerAdapter<VectorType, RelaxationPreconditionerType>>(
          relexation);
      };

      if (preconditioner_type == "Diagonal")
        {
          pcout << "- Create system preconditioner: Diagonal" << std::endl
                << std::endl;

          const auto precon = std::make_shared<DiagonalMatrix<VectorType>>();
          op.compute_inverse_diagonal(precon->get_vector());

          return setup_chebshev(precon);
        }
      else
        {
          const auto precon =
            create_system_preconditioner(op, preconditioner_parameters);

          return setup_chebshev(
            std::const_pointer_cast<
              PreconditionerBase<typename OperatorType::vector_type>>(precon));
        }
    }
  else if (type == "Chebyshev")
    {
      const auto preconditioner_parameters =
        try_get_child(params, "preconditioner");

      const auto preconditioner_optimize = params.get<bool>("optimize", true);

      const auto preconditioner_type =
        preconditioner_parameters.get<std::string>("type", "");

      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto setup_chebshev = [&](const auto precon) {
        using PreconditionerType = PreconditionChebyshev<
          OperatorType,
          VectorType,
          typename std::remove_cv<
            typename std::remove_reference<decltype(*precon)>::type>::type>;

        const auto chebyshev =
          create_chebyshev_preconditioner(op, precon, params);

        VectorType vec;
        op.initialize_dof_vector(vec);
        const auto evs = chebyshev->estimate_eigenvalues(vec);

        pcout << "- Create system preconditioner: Chebyshev" << std::endl;

        pcout << "    - degree: " << params.get<unsigned int>("degree", 3)
              << std::endl;
        pcout << "    - min ev: " << evs.min_eigenvalue_estimate << std::endl;
        pcout << "    - max ev: " << evs.max_eigenvalue_estimate << std::endl;
        pcout << "    - omega:  "
              << 2.0 /
                   (evs.min_eigenvalue_estimate + evs.max_eigenvalue_estimate)
              << std::endl;
        pcout << std::endl;

        return std::make_shared<
          PreconditionerAdapter<VectorType, PreconditionerType>>(chebyshev);
      };

      if (preconditioner_optimize && (preconditioner_type == "Diagonal"))
        {
          pcout << "- Create system preconditioner: Diagonal" << std::endl
                << std::endl;

          const auto precon = std::make_shared<DiagonalMatrix<VectorType>>();
          op.compute_inverse_diagonal(precon->get_vector());

          return setup_chebshev(precon);
        }
      else if (preconditioner_optimize && (preconditioner_type == "FDM"))
        {
          return setup_chebshev(
            std::const_pointer_cast<ASPoissonPreconditionerBase<VectorType>>(
              create_fdm_preconditioner(op, preconditioner_parameters)));
        }
      else
        {
          const auto precon =
            create_system_preconditioner(op, preconditioner_parameters);

          return setup_chebshev(
            std::const_pointer_cast<
              PreconditionerBase<typename OperatorType::vector_type>>(precon));
        }
    }
  else if (type == "FDM")
    {
      return std::make_shared<
        PreconditionerAdapter<VectorType,
                              ASPoissonPreconditionerBase<VectorType>>>(
        create_fdm_preconditioner(op, params));
    }
  else if (type == "AMG")
    {
      pcout << "- Create system preconditioner: AMG" << std::endl << std::endl;

      using PreconditionerType = TrilinosWrappers::PreconditionAMG;

      typename PreconditionerType::AdditionalData additional_data;

      const auto preconitioner = std::make_shared<PreconditionerType>();

      preconitioner->initialize(op.get_sparse_matrix(), additional_data);

      return std::make_shared<
        PreconditionerAdapter<VectorType, PreconditionerType, double>>(
        preconitioner);
    }
  else if (type == "AdditiveSchwarzPreconditioner")
    {
      pcout << "- Create system preconditioner: AdditiveSchwarzPreconditioner"
            << std::endl
            << std::endl;

      using RestictorType = Restrictors::ElementCenteredRestrictor<VectorType>;
      using InverseMatrixType = RestrictedMatrixView<Number>;
      using PreconditionerType =
        RestrictedPreconditioner<VectorType, InverseMatrixType, RestictorType>;

      // approximate matrix
      const auto op_approx = get_approximation(op, params);

      // restrictor
      typename RestictorType::AdditionalData restrictor_ad;

      const unsigned int fe_degree = op.get_dof_handler().get_fe().degree;

      restrictor_ad.n_overlap =
        std::min(params.get<unsigned int>("n overlap", 1), fe_degree + 1);
      restrictor_ad.weighting_type = get_weighting_type(params);

      const auto restrictor =
        std::make_shared<const RestictorType>(op_approx->get_dof_handler(),
                                              restrictor_ad);

      const auto &sparse_matrix    = op_approx->get_sparse_matrix();
      const auto &sparsity_pattern = op_approx->get_sparsity_pattern();

      // inverse matrix
      const auto inverse_matrix =
        std::make_shared<InverseMatrixType>(restrictor,
                                            sparse_matrix,
                                            sparsity_pattern);

      inverse_matrix->invert();

      // preconditioner
      return std::make_shared<const PreconditionerType>(inverse_matrix,
                                                        restrictor);
    }
  else if (type == "SubMeshPreconditioner")
    {
      pcout << "- Create system preconditioner: SubMeshPreconditioner"
            << std::endl
            << std::endl;

      using RestictorType = Restrictors::ElementCenteredRestrictor<VectorType>;
      using InverseMatrixType = SubMeshMatrixView<Number>;
      using PreconditionerType =
        RestrictedPreconditioner<VectorType, InverseMatrixType, RestictorType>;

      typename InverseMatrixType::AdditionalData preconditioner_ad;

      preconditioner_ad.sub_mesh_approximation =
        params.get<unsigned int>("sub mesh approximation",
                                 OperatorType::dimension);

      // approximate matrix
      const auto op_approx = get_approximation(op, params);

      // restrictor
      typename RestictorType::AdditionalData restrictor_ad;

      restrictor_ad.n_overlap      = params.get<unsigned int>("n overlap", 1);
      restrictor_ad.weighting_type = get_weighting_type(params);

      AssertThrow((preconditioner_ad.sub_mesh_approximation ==
                   OperatorType::dimension) ||
                    (restrictor_ad.n_overlap == 1),
                  ExcNotImplemented());

      const auto restrictor =
        std::make_shared<const RestictorType>(op_approx->get_dof_handler(),
                                              restrictor_ad);

      // inverse matrix
      const auto inverse_matrix =
        std::make_shared<InverseMatrixType>(op_approx,
                                            restrictor,
                                            preconditioner_ad);
      inverse_matrix->invert();

      // preconditioner
      return std::make_shared<const PreconditionerType>(inverse_matrix,
                                                        restrictor);
    }
  else if (type == "CGPreconditioner")
    {
      pcout << "- Create system preconditioner: CGPreconditioner" << std::endl
            << std::endl;

      using RestictorType = Restrictors::ElementCenteredRestrictor<VectorType>;
      using MatrixType0   = SubMeshMatrixView<Number>;
      using MatrixType1   = DiagonalMatrixView<Number>;
      using InverseMatrixType = CGMatrixView<MatrixType0, MatrixType1>;
      using PreconditionerType =
        RestrictedPreconditioner<VectorType, InverseMatrixType, RestictorType>;

      typename MatrixType0::AdditionalData preconditioner_ad; // TODO

      preconditioner_ad.sub_mesh_approximation =
        params.get<unsigned int>("sub mesh approximation",
                                 OperatorType::dimension);

      // approximate matrix
      const auto op_approx = get_approximation(op, params);

      // restrictor
      typename RestictorType::AdditionalData restrictor_ad;

      restrictor_ad.n_overlap      = params.get<unsigned int>("n overlap", 1);
      restrictor_ad.weighting_type = get_weighting_type(params);

      AssertThrow((preconditioner_ad.sub_mesh_approximation ==
                   OperatorType::dimension) ||
                    (restrictor_ad.n_overlap == 1),
                  ExcNotImplemented());

      const auto restrictor =
        std::make_shared<const RestictorType>(op_approx->get_dof_handler(),
                                              restrictor_ad);

      // inverse matrix
      const auto matrix =
        std::make_shared<MatrixType0>(op_approx, restrictor, preconditioner_ad);

      const auto precon = std::make_shared<MatrixType1>(matrix);
      precon->invert();

      const auto cg = std::make_shared<InverseMatrixType>(matrix, precon);

      // preconditioner
      return std::make_shared<const PreconditionerType>(cg, restrictor);
    }

  AssertThrow(false, ExcMessage("Preconditioner <" + type + "> is not known!"));

  return {};
}

template <typename VectorType>
class WrapperForGMG : public Subscriptor
{
public:
  struct AdditionalData
  {};

  WrapperForGMG() = default;

  WrapperForGMG(
    const std::shared_ptr<const PreconditionerBase<VectorType>> &base)
    : base(base)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    base->vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    Assert(false, ExcNotImplemented());
    (void)dst;
    (void)src;
  }

  void
  clear()
  {
    Assert(false, ExcNotImplemented());
  }

  std::shared_ptr<const PreconditionerBase<VectorType>> base;
};

template <int dim,
          typename LevelMatrixType_,
          typename VectorType,
          typename VectorTypeOuter>
class MyMultigrid : public PreconditionerGMG<dim,
                                             LevelMatrixType_,
                                             WrapperForGMG<VectorType>,
                                             VectorType,
                                             VectorTypeOuter>
{
public:
  using Base = PreconditionerGMG<dim,
                                 LevelMatrixType_,
                                 WrapperForGMG<VectorType>,
                                 VectorType,
                                 VectorTypeOuter>;

  using LevelMatrixType = typename Base::LevelMatrixType;
  using SmootherType    = typename Base::SmootherType;

  MyMultigrid(
    const boost::property_tree::ptree params,
    const DoFHandler<dim> &           dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<
      std::shared_ptr<const AffineConstraints<typename VectorType::value_type>>>
      &                                                    mg_constraints,
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
    : PreconditionerGMG<dim,
                        LevelMatrixType_,
                        WrapperForGMG<VectorType>,
                        VectorType,
                        VectorTypeOuter>(dof_handler,
                                         mg_dof_handlers,
                                         mg_constraints,
                                         mg_operators)
    , params(params)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    this->do_update();
  }

  SmootherType
  create_mg_level_smoother(unsigned int           level,
                           const LevelMatrixType &level_matrix) final
  {
    pcout << "- Setting up smoother on level " << level << "" << std::endl
          << std::endl;

    (void)level;

    return WrapperForGMG<VectorType>(
      create_system_preconditioner<LevelMatrixType>(
        level_matrix, try_get_child(params, "mg smoother")));
  }

  SmootherType
  create_mg_coarse_grid_solver(unsigned int           level,
                               const LevelMatrixType &level_matrix) final
  {
    pcout << "- Setting up coarse-grid solver on level " << level << ""
          << std::endl
          << std::endl;

    (void)level;

    return WrapperForGMG<VectorType>(
      create_system_preconditioner<LevelMatrixType>(
        level_matrix, try_get_child(params, "mg coarse grid solver")));
  }

private:
  const boost::property_tree::ptree params;
  ConditionalOStream                pcout;
};


template <int dim, typename Number>
class LaplaceOperatorMatrixBased : public Subscriptor
{
public:
  static const int dimension = dim;
  using value_type           = Number;
  using vector_type          = LinearAlgebra::distributed::Vector<Number>;

  using VectorType = vector_type;

  LaplaceOperatorMatrixBased(const Mapping<dim> &      mapping,
                             const Triangulation<dim> &tria,
                             const FiniteElement<dim> &fe,
                             const Quadrature<dim> &   quadrature)
    : mapping(mapping)
    , dof_handler(tria)
    , quadrature(quadrature)
  {
    dof_handler.distribute_dofs(fe);

    DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
    constraints.close();

    // create system matrix
    sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                            dof_handler.get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    false);
    sparsity_pattern.compress();

    sparse_matrix.reinit(sparsity_pattern);

    MatrixCreator::
      create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
        mapping, dof_handler, quadrature, sparse_matrix, nullptr, constraints);

    partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());
  }

  static constexpr bool
  is_matrix_free()
  {
    return false;
  }

  types::global_dof_index
  m() const
  {
    return dof_handler.n_dofs();
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
    sparse_matrix.vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    sparse_matrix.Tvmult(dst, src);
  }

  const Mapping<dim> &
  get_mapping() const
  {
    return mapping;
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return dof_handler.get_fe();
  }

  const Triangulation<dim> &
  get_triangulation() const
  {
    return dof_handler.get_triangulation();
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    return sparse_matrix;
  }

  const TrilinosWrappers::SparsityPattern &
  get_sparsity_pattern() const
  {
    return sparsity_pattern;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(partitioner);
  }

  const AffineConstraints<Number> &
  get_constraints() const
  {
    return constraints;
  }

  const Quadrature<dim> &
  get_quadrature() const
  {
    return quadrature;
  }

  void
  compute_inverse_diagonal(VectorType &vec) const
  {
    this->initialize_dof_vector(vec);

    for (const auto entry : sparse_matrix)
      if (entry.row() == entry.column())
        vec[entry.row()] = 1.0 / entry.value();
  }

private:
  const Mapping<dim> &              mapping;
  DoFHandler<dim>                   dof_handler;
  TrilinosWrappers::SparseMatrix    sparse_matrix;
  TrilinosWrappers::SparsityPattern sparsity_pattern;
  AffineConstraints<Number>         constraints;
  Quadrature<dim>                   quadrature;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
};


template <int dim,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
class LaplaceOperatorMatrixFree : public Subscriptor
{
public:
  static const int dimension  = dim;
  using value_type            = Number;
  using vectorized_array_type = VectorizedArrayType;
  using vector_type           = LinearAlgebra::distributed::Vector<Number>;

  using VectorType = vector_type;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  LaplaceOperatorMatrixFree(const Mapping<dim> &      mapping,
                            const Triangulation<dim> &tria,
                            const FiniteElement<dim> &fe,
                            const Quadrature<dim> &   quadrature)
    : mapping(mapping)
    , dof_handler(tria)
    , quadrature(quadrature)
    , pcout(std::cout,
            Utilities::MPI::this_mpi_process(dof_handler.get_communicator()) ==
              0)
  {
    dof_handler.distribute_dofs(fe);

    DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
    constraints.close();

    matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

    pcout << "- Create operator:" << std::endl;
    pcout << "  - n cells: "
          << dof_handler.get_triangulation().n_global_active_cells()
          << std::endl;
    pcout << "  - n dofs:  " << dof_handler.n_dofs() << std::endl;
    pcout << std::endl;
  }

  static constexpr bool
  is_matrix_free()
  {
    return true;
  }

  const MatrixFree<dim, Number, VectorizedArrayType> &
  get_matrix_free() const
  {
    return matrix_free;
  }

  types::global_dof_index
  m() const
  {
    return dof_handler.n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    AssertThrow(false, ExcNotImplemented());
    return 0;
  }

  void
  set_partitioner(
    std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner) const
  {
    if ((vector_partitioner.get() == nullptr) ||
        (vector_partitioner.get() ==
         matrix_free.get_vector_partitioner().get()))
      return; // nothing to do

    constraint_info.reinit(matrix_free.n_physical_cells());

    const auto locally_owned_indices =
      vector_partitioner->locally_owned_range();
    std::vector<types::global_dof_index> relevant_dofs;

    cell_ptr = {0};
    for (unsigned int cell = 0, cell_counter = 0;
         cell < matrix_free.n_cell_batches();
         ++cell)
      {
        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v, ++cell_counter)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

            const auto cells =
              dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                cell_iterator, 0);

            auto dofs = dealii::DoFTools::get_dof_indices_cell_with_overlap(
              dof_handler, cells, 1, false);

            for (auto &dof : dofs)
              if (dof != numbers::invalid_unsigned_int)
                {
                  if (constraints.is_constrained(dof))
                    dof = numbers::invalid_unsigned_int;
                  else if (locally_owned_indices.is_element(dof) == false)
                    relevant_dofs.push_back(dof);
                }

            constraint_info.read_dof_indices(cell_counter,
                                             dofs,
                                             vector_partitioner);
          }

        cell_ptr.push_back(cell_ptr.back() +
                           matrix_free.n_active_entries_per_cell_batch(cell));
      }

    constraint_info.finalize();

    std::sort(relevant_dofs.begin(), relevant_dofs.end());
    relevant_dofs.erase(std::unique(relevant_dofs.begin(), relevant_dofs.end()),
                        relevant_dofs.end());

    IndexSet relevant_ghost_indices(locally_owned_indices.size());
    relevant_ghost_indices.add_indices(relevant_dofs.begin(),
                                       relevant_dofs.end());

    auto embedded_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      locally_owned_indices, vector_partitioner->get_mpi_communicator());

    embedded_partitioner->set_ghost_indices(
      relevant_ghost_indices, vector_partitioner->ghost_indices());

    this->vector_partitioner   = vector_partitioner;
    this->embedded_partitioner = embedded_partitioner;
  }

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
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    if (vector_partitioner == nullptr)
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
          true);
      }
    else
      {
        update_ghost_values(src, embedded_partitioner); // TODO: overlap?

        FECellIntegrator phi(matrix_free);

        // data structures needed for zeroing dst
        const auto &task_info           = matrix_free.get_task_info();
        const auto &partition_row_index = task_info.partition_row_index;
        const auto &cell_partition_data = task_info.cell_partition_data;
        const auto &dof_info            = matrix_free.get_dof_info();
        const auto &vector_zero_range_list_index =
          dof_info.vector_zero_range_list_index;
        const auto &vector_zero_range_list = dof_info.vector_zero_range_list;

        for (unsigned int part = 0; part < partition_row_index.size() - 2;
             ++part)
          for (unsigned int i = partition_row_index[part];
               i < partition_row_index[part + 1];
               ++i)
            {
              // zero out range in dst
              for (unsigned int id = vector_zero_range_list_index[i];
                   id != vector_zero_range_list_index[i + 1];
                   ++id)
                std::memset(dst.begin() + vector_zero_range_list[id].first,
                            0,
                            (vector_zero_range_list[id].second -
                             vector_zero_range_list[id].first) *
                              sizeof(Number));

              // loop over cells
              for (unsigned int cell = cell_partition_data[i];
                   cell < cell_partition_data[i + 1];
                   ++cell)
                {
                  phi.reinit(cell);

                  internal::VectorReader<Number, VectorizedArrayType> reader;
                  constraint_info.read_write_operation(reader,
                                                       src,
                                                       phi.begin_dof_values(),
                                                       cell_ptr[cell],
                                                       cell_ptr[cell + 1] -
                                                         cell_ptr[cell],
                                                       phi.dofs_per_cell,
                                                       true);

                  do_cell_integral_local(phi);

                  internal::VectorDistributorLocalToGlobal<Number,
                                                           VectorizedArrayType>
                    writer;
                  constraint_info.read_write_operation(writer,
                                                       dst,
                                                       phi.begin_dof_values(),
                                                       cell_ptr[cell],
                                                       cell_ptr[cell + 1] -
                                                         cell_ptr[cell],
                                                       phi.dofs_per_cell,
                                                       true);
                }
            }

        compress(dst, embedded_partitioner); // TODO: overlap?
        src.zero_out_ghost_values();
      }
  }

  static void
  update_ghost_values(const VectorType &vec,
                      const std::shared_ptr<const Utilities::MPI::Partitioner>
                        &embedded_partitioner)
  {
    const auto &vector_partitioner = vec.get_partitioner();

    dealii::AlignedVector<Number> buffer;
    buffer.resize_fast(embedded_partitioner->n_import_indices()); // reuse?

    std::vector<MPI_Request> requests;

    embedded_partitioner
      ->template export_to_ghosted_array_start<Number, MemorySpace::Host>(
        0,
        dealii::ArrayView<const Number>(
          vec.begin(), embedded_partitioner->locally_owned_size()),
        dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);

    embedded_partitioner
      ->template export_to_ghosted_array_finish<Number, MemorySpace::Host>(
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);

    vec.set_ghost_state(true);
  }

  static void
  compress(VectorType &vec,
           const std::shared_ptr<const Utilities::MPI::Partitioner>
             &embedded_partitioner)
  {
    const auto &vector_partitioner = vec.get_partitioner();

    dealii::AlignedVector<Number> buffer;
    buffer.resize_fast(embedded_partitioner->n_import_indices()); // reuse?

    std::vector<MPI_Request> requests;

    embedded_partitioner
      ->template import_from_ghosted_array_start<Number, MemorySpace::Host>(
        dealii::VectorOperation::add,
        0,
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        dealii::ArrayView<Number>(buffer.begin(), buffer.size()),
        requests);

    embedded_partitioner
      ->template import_from_ghosted_array_finish<Number, MemorySpace::Host>(
        dealii::VectorOperation::add,
        dealii::ArrayView<const Number>(buffer.begin(), buffer.size()),
        dealii::ArrayView<Number>(vec.begin(),
                                  embedded_partitioner->locally_owned_size()),
        dealii::ArrayView<Number>(const_cast<Number *>(vec.begin()) +
                                    embedded_partitioner->locally_owned_size(),
                                  vector_partitioner->n_ghost_indices()),
        requests);
  }

  void
  vmult(VectorType &      dst,
        const VectorType &src,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_before_matrix_vector_product,
        const std::function<void(const unsigned int, const unsigned int)>
          &operation_after_matrix_vector_product) const
  {
    AssertThrow(vector_partitioner == nullptr, ExcNotImplemented());

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
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(false, ExcNotImplemented());
    (void)dst;
    (void)src;
  }

  const Mapping<dim> &
  get_mapping() const
  {
    return mapping;
  }

  const FiniteElement<dim> &
  get_fe() const
  {
    return dof_handler.get_fe();
  }

  const Triangulation<dim> &
  get_triangulation() const
  {
    return dof_handler.get_triangulation();
  }

  const DoFHandler<dim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  const TrilinosWrappers::SparseMatrix &
  get_sparse_matrix() const
  {
    if (sparse_matrix.m() == 0 && sparse_matrix.n() == 0)
      compute_system_matrix();

    return sparse_matrix;
  }

  const TrilinosWrappers::SparsityPattern &
  get_sparsity_pattern() const
  {
    if (sparse_matrix.m() == 0 && sparse_matrix.n() == 0)
      compute_system_matrix();

    return sparsity_pattern;
  }

  void
  initialize_dof_vector(VectorType &vec) const
  {
    if (vector_partitioner)
      vec.reinit(vector_partitioner);
    else
      matrix_free.initialize_dof_vector(vec);
  }

  const AffineConstraints<Number> &
  get_constraints() const
  {
    return constraints;
  }

  const Quadrature<dim> &
  get_quadrature() const
  {
    return quadrature;
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &LaplaceOperatorMatrixFree::do_cell_integral_local,
      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

private:
  void
  compute_system_matrix() const
  {
    Assert((sparse_matrix.m() == 0 && sparse_matrix.n() == 0),
           ExcNotImplemented());

    sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                            dof_handler.get_triangulation().get_communicator());

    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    this->constraints);

    sparsity_pattern.compress();
    sparse_matrix.reinit(sparsity_pattern);

    MatrixFreeTools::compute_matrix(
      matrix_free,
      constraints,
      sparse_matrix,
      &LaplaceOperatorMatrixFree::do_cell_integral_local,
      this);
  }

  const Mapping<dim> &      mapping;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<Number> constraints;
  Quadrature<dim>           quadrature;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  // for own partitioner
  mutable std::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner;
  mutable std::shared_ptr<const Utilities::MPI::Partitioner>
                                    embedded_partitioner;
  mutable std::vector<unsigned int> cell_ptr;
  mutable internal::MatrixFreeFunctions::ConstraintInfo<dim,
                                                        VectorizedArrayType>
    constraint_info;

  ConditionalOStream pcout;

  // only set up if required
  mutable TrilinosWrappers::SparsityPattern sparsity_pattern;
  mutable TrilinosWrappers::SparseMatrix    sparse_matrix;
};


template <typename OperatorTrait>
void
test(const boost::property_tree::ptree params, ConvergenceTable &table)
{
  const unsigned int fe_degree = params.get<unsigned int>("degree", 1);
  const unsigned int n_global_refinements =
    params.get<unsigned int>("n refinements", 6);

  const auto solver_parameters = try_get_child(params, "solver");
  const auto preconditioner_parameters =
    try_get_child(params, "preconditioner");

  const auto preconditioner_type =
    preconditioner_parameters.get<std::string>("type", "");

  using OperatorType      = typename OperatorTrait::OperatorType;
  using LevelOperatorType = typename OperatorTrait::LevelOperatorType;
  const int dim           = OperatorType::dimension;
  using VectorType        = typename OperatorType::vector_type;
  using LevelNumber       = typename LevelOperatorType::value_type;
  using LevelVectorType   = typename LevelOperatorType::vector_type;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  const auto mesh_parameters = try_get_child(params, "mesh");

  const std::string geometry_name =
    mesh_parameters.get<std::string>("name", "hypercube");
  unsigned int mapping_degree = 1;

  std::function<Point<dim>(const typename Triangulation<dim>::cell_iterator &,
                           const Point<dim> &)>
    transformation_function;

  if (geometry_name == "hypercube")
    {
      pcout << "- Create mesh: hypercube" << std::endl;
      pcout << std::endl;

      GridGenerator::hyper_cube(tria);
      mapping_degree = 1;
    }
  else if (geometry_name == "anisotropy")
    {
      const auto stratch = mesh_parameters.get<double>("stratch", 1.0);

      pcout << "- Create mesh: anisotropy" << std::endl;
      pcout << "  - stratch: " << stratch << std::endl;
      pcout << std::endl;

      GridGenerator::hyper_cube(tria);
      mapping_degree          = 1;
      transformation_function = [stratch](const auto &, const auto &old_point) {
        auto new_point = old_point;

        new_point[dim - 1] *= stratch;

        return new_point;
      };
    }
  else if (geometry_name == "kershaw")
    {
      auto epsy = mesh_parameters.get<double>("epsy", 0.0);
      auto epsz = mesh_parameters.get<double>("epsz", 0.0);

      if (epsy == 0.0 || epsz == 0.0)
        {
          auto eps = mesh_parameters.get<double>("eps", 1.0);

          epsy = eps;
          epsz = eps;
        }

      pcout << "- Create mesh: kershaw" << std::endl;
      pcout << "  - epsx: " << 1.0 << std::endl;
      pcout << "  - epsy: " << epsy << std::endl;
      pcout << "  - epsz: " << epsz << std::endl;
      pcout << std::endl;

      GridGenerator::subdivided_hyper_cube(tria, 6);
      mapping_degree = 3; // TODO

      transformation_function = [epsy, epsz](const auto &,
                                             const auto &in_point) {
        Point<dim> out_point;
        // clang-format off
        kershaw(epsy, epsz, in_point[0], in_point[1], in_point[2], out_point[0], out_point[1], out_point[2]);
        // clang-format on

        return out_point;
      };
    }
  else if (geometry_name == "hyperball")
    {
      GridGenerator::hyper_ball_balanced(tria);
      mapping_degree = 2;
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Geometry with the name <" + geometry_name +
                             "> is not known!"));
    }


  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

  tria.refine_global(n_global_refinements);

  const MappingQ1<dim> mapping_q1;

  MappingQCache<dim> mapping(mapping_degree);

  if (transformation_function)
    mapping.initialize(mapping_q1, tria, transformation_function, false);
  else
    mapping.initialize(mapping_q1, tria);

  const FE_Q<dim>   fe(fe_degree);
  const QGauss<dim> quadrature(fe_degree + 1);

  OperatorType op(mapping, tria, fe, quadrature);

  table.add_value("n_cells", tria.n_global_active_cells());
  table.add_value("L", tria.n_global_levels());
  table.add_value("n_dofs", op.get_dof_handler().n_dofs());

  // create vectors
  VectorType solution, rhs;

  op.initialize_dof_vector(solution);
  op.initialize_dof_vector(rhs);

  VectorTools::create_right_hand_side(op.get_dof_handler(),
                                      op.get_quadrature(),
                                      RightHandSide<dim>(),
                                      rhs,
                                      op.get_constraints());

  std::shared_ptr<ReductionControl> reduction_control;

  // ASM on cell level
  if (preconditioner_type == "Identity")
    {
      // note: handle it seperatly to exploit template specialization available
      // for SoverCG + PreconditionIdentity

      pcout << "- Create system preconditioner: Identity" << std::endl
            << std::endl;

      const auto preconditioner =
        std::make_shared<const PreconditionIdentity>();

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters, table);
    }
  else if (preconditioner_type == "Diagonal")
    {
      // note: handle it seperatly to exploit template specialization available
      // for SoverCG + DiagonalMatrix

      pcout << "- Create system preconditioner: Diagonal" << std::endl
            << std::endl;

      auto preconditioner = std::make_shared<DiagonalMatrix<VectorType>>();
      op.compute_inverse_diagonal(preconditioner->get_vector());

      reduction_control =
        solve(op,
              solution,
              rhs,
              std::const_pointer_cast<const DiagonalMatrix<VectorType>>(
                preconditioner),
              solver_parameters,
              table);
    }
  else if (preconditioner_type == "Multigrid")
    {
      // note: handle it seperatly, since we need to set up the levels

      pcout << "- Create system preconditioner: Multigrid" << std::endl;

      MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
      MGLevelObject<std::shared_ptr<const AffineConstraints<LevelNumber>>>
                                                         mg_constraints;
      MGLevelObject<std::shared_ptr<LevelOperatorType>>  mg_operators;
      MGLevelObject<std::shared_ptr<MappingQCache<dim>>> mg_mapping;

      MGLevelObject<MGTwoLevelTransfer<dim, LevelVectorType>> transfers;
      std::unique_ptr<MGTransferGlobalCoarsening<dim, LevelVectorType>>
        transfer;

      bool use_pmg = false;

      const auto mg_type =
        preconditioner_parameters.get<std::string>("mg type", "h");

      if (mg_type == "h")
        {
          use_pmg = false;
        }
      else if (mg_type == "p")
        {
          use_pmg = true;
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("Multigrid variant <" + mg_type +
                                 "> is not known!"));
        }

      const auto mg_p_sequence_string =
        preconditioner_parameters.get<std::string>("mg p sequence", "bisect");

      MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
        mg_p_sequence = MGTransferGlobalCoarseningTools::
          PolynomialCoarseningSequenceType::bisect;

      if (mg_p_sequence_string == "bisect")
        {
          mg_p_sequence = MGTransferGlobalCoarseningTools::
            PolynomialCoarseningSequenceType::bisect;
        }
      else if (mg_p_sequence_string == "decrease by one")
        {
          mg_p_sequence = MGTransferGlobalCoarseningTools::
            PolynomialCoarseningSequenceType::decrease_by_one;
        }
      else if (mg_p_sequence_string == "go to one")
        {
          mg_p_sequence = MGTransferGlobalCoarseningTools::
            PolynomialCoarseningSequenceType::go_to_one;
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("Multigrid p sequence <" +
                                 mg_p_sequence_string + "> is not known!"));
        }

      pcout << " - type:       " << mg_type << std::endl;
      pcout << " - p sequence: " << mg_p_sequence_string << std::endl;
      pcout << std::endl;

      const auto mg_degress =
        create_polynomial_coarsening_sequence(fe_degree, mg_p_sequence);
      const auto mg_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          tria);

      const unsigned int min_level = 0;
      const unsigned int max_level =
        (use_pmg ? mg_degress.size() : mg_triangulations.size()) - 1;

      mg_dof_handlers.resize(min_level, max_level);
      mg_constraints.resize(min_level, max_level);
      mg_operators.resize(min_level, max_level);
      mg_mapping.resize(min_level, max_level);

      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          const unsigned int mg_fe_degree = use_pmg ? mg_degress[l] : fe_degree;
          const FE_Q<dim>    mg_fe(mg_fe_degree);
          const auto &       mg_tria = use_pmg ?
                                         static_cast<Triangulation<dim> &>(tria) :
                                         *mg_triangulations[l];

          const QGauss<dim> mg_quadrature(mg_fe_degree + 1);


          mg_mapping[l] = std::make_shared<MappingQCache<dim>>(mapping_degree);

          if (transformation_function)
            mg_mapping[l]->initialize(mapping_q1,
                                      tria,
                                      transformation_function,
                                      false);
          else
            mg_mapping[l]->initialize(mapping_q1, tria);

          mg_operators[l] = std::make_shared<LevelOperatorType>(*mg_mapping[l],
                                                                mg_tria,
                                                                mg_fe,
                                                                mg_quadrature);
        }

      for (auto l = min_level; l <= max_level; ++l)
        {
          mg_dof_handlers[l] = std::shared_ptr<const DoFHandler<dim>>(
            &mg_operators[l]->get_dof_handler(),
            [](auto *) { /*nothing to do*/ });
          mg_constraints[l] =
            std::shared_ptr<const AffineConstraints<LevelNumber>>(
              &mg_operators[l]->get_constraints(),
              [](auto *) { /*nothing to do*/ });
        }

      const auto preconditioner = std::make_shared<
        const MyMultigrid<dim, LevelOperatorType, LevelVectorType, VectorType>>(
        preconditioner_parameters,
        op.get_dof_handler(),
        mg_dof_handlers,
        mg_constraints,
        mg_operators);

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters, table);
    }
  else
    {
      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto preconditioner =
        create_system_preconditioner(op, preconditioner_parameters);

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters, table);
    }
}

template <int dim>
struct LaplaceOperatorMatrixBasedTrait
{
  using OperatorType      = LaplaceOperatorMatrixBased<dim, double>;
  using LevelOperatorType = LaplaceOperatorMatrixBased<dim, double>;
};

template <int dim>
struct LaplaceOperatorMatrixFreeTrait
{
  using OperatorType      = LaplaceOperatorMatrixFree<dim, double>;
  using LevelOperatorType = LaplaceOperatorMatrixFree<dim, float>;
};

void
run(const std::string file_name, ConvergenceTable &table)
{
  // get parameters
  boost::property_tree::ptree params;
  boost::property_tree::read_json(file_name, params);

  const auto dim  = params.get<unsigned int>("dim", 2);
  const auto type = params.get<std::string>("type", "matrixbased");

  if (type == "matrixbased")
    {
#if COMPILE_MB > 0
#  if COMPILE_2D > 0
      if (dim == 2)
        test<LaplaceOperatorMatrixBasedTrait<2>>(params, table);
      else
#  endif
#  if COMPILE_3D > 0
        if (dim == 3)
        test<LaplaceOperatorMatrixBasedTrait<3>>(params, table);
      else
#  endif
#endif
        AssertThrow(false, ExcNotImplemented());
    }
  else if (type == "matrixfree")
    {
#if COMPILE_MF > 0
#  if COMPILE_2D > 0
      if (dim == 2)
        test<LaplaceOperatorMatrixFreeTrait<2>>(params, table);
      else
#  endif
#  if COMPILE_3D > 0
        if (dim == 3)
        test<LaplaceOperatorMatrixFreeTrait<3>>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());
#  endif
#endif
    }
  else
    AssertThrow(false, ExcNotImplemented());
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef DEBUG
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "DEBUG!" << std::endl;
#endif

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  const bool verbose = true;

  ConvergenceTable table;

  for (int i = 1; i < argc; ++i)
    {
      const std::string file_name = std::string(argv[i]);

      boost::property_tree::ptree params;
      boost::property_tree::read_json(file_name, params);

      const auto print_timings_ = params.get<bool>("print timing", false);

      if (i == 1)
        print_timings = print_timings_;

      AssertThrow(print_timings == print_timings_, ExcNotImplemented());

      if (is_root && print_timings)
        std::cout << "Processing " << file_name << std::endl << std::endl;

      table.add_value("name", params.get<std::string>("name", "---"));

      run(file_name, table);

      if (is_root && (verbose || ((i + 1) == argc)))
        {
          table.write_text(std::cout, ConvergenceTable::org_mode_table);
          std::cout << std::endl;
        }
    }
}
