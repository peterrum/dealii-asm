#pragma once

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include "matrix_free.h"
#include "precondition.h"
#include "restrictors.h"

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
  else if (type == "ras")
    return Restrictors::WeightingType::ras;
  else if (type == "none")
    return Restrictors::WeightingType::none;

  AssertThrow(false, ExcMessage("Weighting type <" + type + "> is not known!"))

    return Restrictors::WeightingType::none;
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
  chebyshev_additional_data.smoothing_range =
    params.get<double>("smoothing range", 20.);
  chebyshev_additional_data.eig_cg_n_iterations = 40;

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

  const auto poly_type = params.get<std::string>("polynomial type", "1st kind");

  if (poly_type == "1st kind")
    {
      chebyshev_additional_data.polynomial_type =
        ChebyshevPreconditionerType::AdditionalData::PolynomialType::first_kind;
    }
  else if (poly_type == "4th kind")
    {
      chebyshev_additional_data.polynomial_type = ChebyshevPreconditionerType::
        AdditionalData::PolynomialType::fourth_kind;
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("Polynomial type <" + ev_algorithm +
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
                                typename OperatorType::vectorized_array_type>>
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

      const auto do_weights_global =
        params.get<std::string>("weight sequence",
                                n_overlap > 1 ? "global" : "compressed");

      const auto overlap_pre_post = params.get<bool>("overlap pre post", true);
      const auto element_centric  = params.get<bool>("element centric", true);

      pcout << "    - n overlap:              " << n_overlap << std::endl;
      pcout << "    - sub mesh approximation: " << sub_mesh_approximation
            << std::endl;
      pcout << "    - reuse partitioner:      "
            << (reuse_partitioner ? "true" : "false") << std::endl;

      auto precon = std::make_shared<
        const ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>(
        matrix_free,
        n_overlap,
        sub_mesh_approximation,
        mapping,
        fe_1D,
        quadrature_face,
        quadrature_1D,
        weight_type,
        op.uses_compressed_indices(),
        do_weights_global,
        overlap_pre_post,
        element_centric);

      if (reuse_partitioner)
        op.set_partitioner(precon->get_partitioner());

      pcout << std::endl;

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

      const auto setup_relaxation = [&](const auto &op, const auto precon) {
        using MyOperatorType = typename std::remove_cv<
          typename std::remove_reference<decltype(op)>::type>::type;
        using RelaxationPreconditionerType = PreconditionRelaxation<
          MyOperatorType,
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

          const auto preconditioner_optimize =
            params.get<unsigned int>("optimize", 3);

          const auto diag = std::make_shared<DiagonalMatrix<VectorType>>();
          op.compute_inverse_diagonal(diag->get_vector());

          const auto my_diag =
            std::make_shared<DiagonalMatrixPrePost<VectorType>>(diag);

          if (preconditioner_optimize == 0)
            {
              // optimization 0: A (-) and P (-)
              return setup_relaxation(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                std::make_shared<
                  PreconditionerAdapter<VectorType,
                                        DiagonalMatrix<VectorType>>>(diag));
            }
          else if (preconditioner_optimize == 1)
            {
              // optimization 1: A (-) and P (pp)
              return setup_relaxation(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                my_diag);
            }
          else if (preconditioner_optimize == 2)
            {
              // optimization 2: A (pp) and P (pp)
              return setup_relaxation(op, my_diag);
            }
          else if (preconditioner_optimize == 3)
            {
              // optimization 2: A (pp) and P (diag)
              return setup_relaxation(op, diag);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
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
              return setup_relaxation(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                std::make_shared<PreconditionerAdapterWithoutStep<
                  VectorType,
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>>(
                  fdm));
            }
          else if (preconditioner_optimize == 1)
            {
              // optimization 1: A (-) and P (pp)
              return setup_relaxation(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                std::const_pointer_cast<
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>(
                  fdm));
            }
          else if (preconditioner_optimize == 2)
            {
              // optimization 2: A (pp) and P (pp)
              return setup_relaxation(
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

          return setup_relaxation(
            op,
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

          const auto preconditioner_optimize =
            params.get<unsigned int>("optimize", 3);

          const auto diag = std::make_shared<DiagonalMatrix<VectorType>>();
          op.compute_inverse_diagonal(diag->get_vector());

          const auto my_diag =
            std::make_shared<DiagonalMatrixPrePost<VectorType>>(diag);

          if (preconditioner_optimize == 0)
            {
              // optimization 0: A (-) and P (-)
              return setup_chebshev(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                std::make_shared<
                  PreconditionerAdapter<VectorType,
                                        DiagonalMatrix<VectorType>>>(diag));
            }
          else if (preconditioner_optimize == 1)
            {
              // optimization 1: A (-) and P (pp)
              return setup_chebshev(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                my_diag);
            }
          else if (preconditioner_optimize == 2)
            {
              // optimization 2: A (pp) and P (pp)
              return setup_chebshev(op, my_diag);
            }
          else if (preconditioner_optimize == 3)
            {
              // optimization 2: A (pp) and P (diag)
              return setup_chebshev(op, diag);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
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
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
                std::make_shared<PreconditionerAdapter<
                  VectorType,
                  ASPoissonPreconditioner<dim, Number, VectorizedArrayType>>>(
                  fdm));
            }
          else if (preconditioner_optimize == 1)
            {
              // optimization 1: A (-) and P (pp)
              return setup_chebshev(
                static_cast<const LaplaceOperatorBase<dim, VectorType> &>(op),
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
      additional_data.n_cycles = 1;

      std::shared_ptr<PreconditionerType> preconitioner;

      if (sub_comm != MPI_COMM_NULL)
        {
          preconitioner = std::make_shared<PreconditionerType>();
          preconitioner->initialize(op.get_sparse_matrix(sub_comm),
                                    additional_data);
        }

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
      restrictor_ad.type =
        params.get<std::string>("restriction type", "element");


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
