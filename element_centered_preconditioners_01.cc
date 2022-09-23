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
#include "include/operator.h"
#include "include/preconditioners.h"
#include "include/restrictors.h"

#define COMPILE_MB 1
#define COMPILE_MF 1
#define COMPILE_2D 1
#define COMPILE_3D 1
#define MAX_N_ROWS_FDM 10

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

    try
      {
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
            AssertThrow(false,
                        ExcMessage("Solver <" + type + "> is not known!"))
          }
      }
    catch (...)
      {
        return false;
      }

    return true;
  };

  const bool converged = dispatch(); // warm up

  double time = 999.0;

  if (converged)
    {
      if constexpr (has_timing_functionality<PreconditionerType>)
        preconditioner->clear_timings();

      const auto timer = std::chrono::system_clock::now();
      dispatch();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now() - timer)
               .count() /
             1e9;

      pcout << "   - n iterations:   " << reduction_control->last_step()
            << std::endl;
      pcout << "   - time:           " << time << " #" << std::endl;
      pcout << std::endl;
    }
  else
    {
      pcout << "   - DID NOT CONVERGE!" << std::endl;
      pcout << std::endl;
    }


  if (converged)
    table.add_value("it", reduction_control->last_step());
  else
    table.add_value("it", 999);

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
  const ASPoissonPreconditioner<OperatorType::dimension,
                                typename OperatorType::value_type,
                                typename OperatorType::vectorized_array_type,
                                -1>>
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

      auto precon = std::make_shared<
        const ASPoissonPreconditioner<dim, Number, VectorizedArrayType, -1>>(
        matrix_free,
        n_overlap,
        sub_mesh_approximation,
        mapping,
        fe_1D,
        quadrature_face,
        quadrature_1D,
        weight_type);

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
  const int dim             = OperatorType::dimension;
  using VectorType          = typename OperatorType::vector_type;
  using Number              = typename VectorType::value_type;
  using VectorizedArrayType = typename OperatorType::vectorized_array_type;

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

      const auto setup_relaxation = [&](const auto precon) {
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

          return setup_relaxation(precon);
        }
      else
        {
          const auto precon =
            create_system_preconditioner(op, preconditioner_parameters);

          return setup_relaxation(
            std::const_pointer_cast<
              PreconditionerBase<typename OperatorType::vector_type>>(precon));
        }
    }
  else if (type == "Chebyshev")
    {
      const auto preconditioner_parameters =
        try_get_child(params, "preconditioner");

      const auto preconditioner_type =
        preconditioner_parameters.get<std::string>("type", "");

      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto setup_chebshev = [&](const auto &op, const auto precon) {
        using MyOperatorType = typename std::remove_cv<
          typename std::remove_reference<decltype(op)>::type>::type;
        using PreconditionerType = PreconditionChebyshev<
          MyOperatorType,
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

      if (preconditioner_type == "Diagonal")
        {
          pcout << "- Create system preconditioner: Diagonal" << std::endl
                << std::endl;

          const auto precon = std::make_shared<DiagonalMatrix<VectorType>>();
          op.compute_inverse_diagonal(precon->get_vector());

          return setup_chebshev(op, precon);
        }
      else if (preconditioner_type == "FDM")
        {
          const unsigned int n_overlap =
            preconditioner_parameters.get<unsigned int>("n overlap", 1);

          const auto preconditioner_optimize =
            params.get<unsigned int>("optimize", (n_overlap == 1) ? 2 : 1);

          const auto fdm =
            create_fdm_preconditioner(op, preconditioner_parameters);

          if (preconditioner_optimize == 0)
            {
              // optimization 0: A (-) and P (-)
              return setup_chebshev(
                static_cast<const LaplaceOperatorBase<VectorType> &>(op),
                std::make_shared<PreconditionerAdapter<
                  VectorType,
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>>(
                  fdm));
            }
          else if (preconditioner_optimize == 1)
            {
              // optimization 1: A (-) and P (pp)
              return setup_chebshev(
                static_cast<const LaplaceOperatorBase<VectorType> &>(op),
                std::const_pointer_cast<
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>(
                  fdm));
            }
          else if (preconditioner_optimize == 2)
            {
              // optimization 2: A (pp) and P (pp)
              return setup_chebshev(
                op,
                std::const_pointer_cast<
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>(
                  fdm));
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }
      else
        {
          const auto precon =
            create_system_preconditioner(op, preconditioner_parameters);

          return setup_chebshev(
            op,
            std::const_pointer_cast<
              PreconditionerBase<typename OperatorType::vector_type>>(precon));
        }
    }
  else if (type == "FDM")
    {
      return std::make_shared<PreconditionerAdapter<
        VectorType,
        ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>>(
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
      using MatrixType0   = RestrictedMatrixView<Number>;
      using MatrixType1   = SubMeshMatrixView<Number>;
      using InverseMatrixType = CGMatrixView<MatrixType0, MatrixType1>;
      using PreconditionerType =
        RestrictedPreconditioner<VectorType, InverseMatrixType, RestictorType>;

      typename MatrixType1::AdditionalData preconditioner_ad; // TODO

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
        std::make_shared<MatrixType0>(restrictor,
                                      op.get_sparse_matrix(),
                                      op.get_sparsity_pattern());

      const auto precon =
        std::make_shared<MatrixType1>(op_approx, restrictor, preconditioner_ad);
      precon->invert();

      typename InverseMatrixType::AdditionalData cg_ad;

      cg_ad.n_iterations = params.get<unsigned int>("n iterations", 1);

      const auto cg =
        std::make_shared<InverseMatrixType>(matrix, precon, cg_ad);

      // preconditioner
      return std::make_shared<const PreconditionerType>(cg, restrictor);
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

      typename InverseMatrixType::AdditionalData cg_ad;

      cg_ad.n_iterations = params.get<unsigned int>("n iterations", 1);

      const auto cg =
        std::make_shared<InverseMatrixType>(matrix, precon, cg_ad);

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

  const auto op_mapping_type =
    params.get<std::string>("operator mapping type", "");

  const auto op_compress_indices =
    params.get<bool>("operator compress indices", false);

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
  unsigned int mapping_degree = params.get<unsigned int>("mapping degree", 10);

  std::function<Point<dim>(const typename Triangulation<dim>::cell_iterator &,
                           const Point<dim> &)>
    transformation_function;

  if (geometry_name == "hypercube")
    {
      pcout << "- Create mesh: hypercube" << std::endl;
      pcout << std::endl;

      GridGenerator::hyper_cube(tria);
      mapping_degree = std::min(mapping_degree, 1u);
    }
  else if (geometry_name == "anisotropy")
    {
      const auto stratch = mesh_parameters.get<double>("stratch", 1.0);

      pcout << "- Create mesh: anisotropy" << std::endl;
      pcout << "  - stratch: " << stratch << std::endl;
      pcout << std::endl;

      GridGenerator::hyper_cube(tria);
      mapping_degree          = std::min(mapping_degree, 1u);
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

      // replace 6 coarse cells by 3 coarse cells and
      // an additional refinement to favor GMG (TODO: is this
      // according to specification)
      GridGenerator::subdivided_hyper_cube(tria, 3);
      tria.refine_global(1);

      mapping_degree = std::min(mapping_degree, 3u /*TODO*/);

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
      mapping_degree = std::min(mapping_degree, 2u);
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

  OperatorType op(mapping,
                  tria,
                  fe,
                  quadrature,
                  typename OperatorType::AdditionalData(op_compress_indices,
                                                        op_mapping_type));

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

      const auto mg_type =
        preconditioner_parameters.get<std::string>("mg type", "h");

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

      std::vector<std::pair<unsigned int, unsigned int>> levels;

      if (mg_type == "h")
        {
          for (unsigned int r = 0; r < mg_triangulations.size(); ++r)
            levels.emplace_back(r, mg_degress.back());
        }
      else if (mg_type == "p")
        {
          for (const auto mg_degree : mg_degress)
            levels.emplace_back(mg_triangulations.size() - 1, mg_degree);
        }
      else if (mg_type == "hp")
        {
          for (const auto mg_degree : mg_degress)
            levels.emplace_back(0, mg_degree);

          for (unsigned int r = 0; r < mg_triangulations.size(); ++r)
            levels.emplace_back(r, mg_degress.back());
        }
      else if (mg_type == "ph")
        {
          for (unsigned int r = 0; r < mg_triangulations.size(); ++r)
            levels.emplace_back(r, mg_degress.front());

          for (const auto mg_degree : mg_degress)
            levels.emplace_back(mg_triangulations.size() - 1, mg_degree);
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("Multigrid variant <" + mg_type +
                                 "> is not known!"));
        }

      const unsigned int min_level = 0;
      const unsigned int max_level = levels.size() - 1;

      mg_dof_handlers.resize(min_level, max_level);
      mg_constraints.resize(min_level, max_level);
      mg_operators.resize(min_level, max_level);
      mg_mapping.resize(min_level, max_level);

      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          const unsigned int mg_fe_degree = levels[l].second;
          const FE_Q<dim>    mg_fe(mg_fe_degree);
          const auto &       mg_tria = *mg_triangulations[levels[l].first];

          const QGauss<dim> mg_quadrature(mg_fe_degree + 1);

          mg_mapping[l] = std::make_shared<MappingQCache<dim>>(mapping_degree);

          if (transformation_function)
            mg_mapping[l]->initialize(mapping_q1,
                                      tria,
                                      transformation_function,
                                      false);
          else
            mg_mapping[l]->initialize(mapping_q1, tria);

          mg_operators[l] = std::make_shared<LevelOperatorType>(
            *mg_mapping[l],
            mg_tria,
            mg_fe,
            mg_quadrature,
            typename LevelOperatorType::AdditionalData(op_compress_indices,
                                                       op_mapping_type));
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
