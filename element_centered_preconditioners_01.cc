#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

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

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace dealii;

#include "include/multigrid.h"
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
      const boost::property_tree::ptree               params)
{
  const auto max_iterations = params.get<unsigned int>("max iterations", 1000);
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
      typename SolverGMRES<VectorType>::AdditionalData additional_data;
      additional_data.right_preconditioning = true;

      SolverGMRES<VectorType> solver(*reduction_control, additional_data);
      solver.solve(A, x, b, *preconditioner);
    }
  else
    {
      AssertThrow(false, ExcMessage("Solver <" + type + "> is not known!"))
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



template <typename OperatorType>
std::shared_ptr<const PreconditionerBase<typename OperatorType::vector_type>>
create_system_preconditioner(const OperatorType &              op,
                             const boost::property_tree::ptree params)
{
  using VectorType = typename OperatorType::vector_type;
  using Number     = typename VectorType::value_type;

  const auto type = params.get<std::string>("type", "");

  if (type == "Chebyshev")
    {
      const auto preconditioner_parameters =
        try_get_child(params, "preconditioner");

      const auto preconditioner_type =
        preconditioner_parameters.get<std::string>("type", "");

      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto setup_chebshev = [&](const auto precon) {
        using PreconditionerType = PreconditionChebyshev<
          OperatorType,
          VectorType,
          typename std::remove_cv<
            typename std::remove_reference<decltype(*precon)>::type>::type>;

        typename PreconditionerType::AdditionalData additional_data;

        additional_data.preconditioner = precon;
        additional_data.constraints.copy_from(op.get_constraints());
        additional_data.degree = params.get<unsigned int>("degree", 3);

        auto chebyshev = std::make_shared<PreconditionerType>();
        chebyshev->initialize(op, additional_data);

        // TODO: print eigenvalues

        return std::make_shared<
          PreconditionerAdapter<VectorType, PreconditionerType>>(chebyshev);
      };

      if (preconditioner_type == "Diagonal")
        {
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
  else if (type == "AdditiveSchwarzPreconditioner")
    {
      using RestictorType = Restrictors::ElementCenteredRestrictor<VectorType>;
      using InverseMatrixType = RestrictedMatrixView<Number>;
      using PreconditionerType =
        RestrictedPreconditioner<VectorType, InverseMatrixType, RestictorType>;

      // approximate matrix
      const auto op_approx = get_approximation(op, params);

      // restrictor
      typename RestictorType::AdditionalData restrictor_ad;

      restrictor_ad.n_overlap      = params.get<unsigned int>("n overlap", 1);
      restrictor_ad.weighting_type = get_weighting_type(params);

      const auto restrictor =
        std::make_shared<const RestictorType>(op_approx->get_dof_handler(),
                                              restrictor_ad);

      // inverse matrix
      const auto inverse_matrix =
        std::make_shared<InverseMatrixType>(restrictor,
                                            op_approx->get_sparse_matrix(),
                                            op_approx->get_sparsity_pattern());
      inverse_matrix->invert();

      // preconditioner
      return std::make_shared<const PreconditionerType>(inverse_matrix,
                                                        restrictor);
    }
  else if (type == "SubMeshPreconditioner")
    {
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

  virtual void
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

template <int dim, typename LevelMatrixType_, typename VectorType>
class MyMultigrid : public PreconditionerGMG<dim,
                                             LevelMatrixType_,
                                             WrapperForGMG<VectorType>,
                                             VectorType>
{
public:
  using Base = PreconditionerGMG<dim,
                                 LevelMatrixType_,
                                 WrapperForGMG<VectorType>,
                                 VectorType>;

  using LevelMatrixType = typename Base::LevelMatrixType;
  using SmootherType    = typename Base::SmootherType;

  MyMultigrid(
    const boost::property_tree::ptree params,
    const DoFHandler<dim> &           dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
      &                                                    mg_constraints,
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
    : PreconditionerGMG<dim,
                        LevelMatrixType_,
                        WrapperForGMG<VectorType>,
                        VectorType>(dof_handler,
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
    pcout << "MyMultigrid::create_mg_level_smoother(" << level << ")"
          << std::endl;

    (void)level;

    return WrapperForGMG<VectorType>(
      create_system_preconditioner<LevelMatrixType>(
        level_matrix, try_get_child(params, "smoother")));
  }

private:
  const boost::property_tree::ptree params;
  ConditionalOStream                pcout;
};


template <int dim, typename Number>
class LaplaceOperator : public Subscriptor
{
public:
  static const int dimension = dim;
  using value_type           = Number;
  using vector_type          = LinearAlgebra::distributed::Vector<double>;

  using VectorType = vector_type;

  LaplaceOperator(const Mapping<dim> &      mapping,
                  const Triangulation<dim> &tria,
                  const FiniteElement<dim> &fe,
                  const Quadrature<dim> &   quadrature)
    : mapping(mapping)
    , dof_handler(tria)
    , quadrature(quadrature)
  {
    dof_handler.distribute_dofs(fe);

    // pcout << "System statistics:" << std::endl;
    // pcout << " - n cells: " << tria.n_global_active_cells() << std::endl;
    // pcout << " - n dofs:  " << dof_handler.n_dofs() << std::endl;
    // pcout << std::endl;

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

  const auto preconditioner_type =
    preconditioner_parameters.get<std::string>("type", "");

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);

  for (const auto &face : tria.active_face_iterators())
    face->set_boundary_id(1);

  tria.refine_global(n_global_refinements);

  FE_Q<dim>      fe(fe_degree);
  QGauss<dim>    quadrature(fe_degree + 1);
  MappingQ1<dim> mapping;

  LaplaceOperator<dim, Number> op(mapping, tria, fe, quadrature);

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

      const auto preconditioner =
        std::make_shared<const PreconditionIdentity>();

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters);
    }
  else if (preconditioner_type == "Diagonal")
    {
      // note: handle it seperatly to exploit template specialization available
      // for SoverCG + DiagonalMatrix

      auto preconditioner = std::make_shared<DiagonalMatrix<VectorType>>();
      op.compute_inverse_diagonal(preconditioner->get_vector());

      reduction_control =
        solve(op,
              solution,
              rhs,
              std::const_pointer_cast<const DiagonalMatrix<VectorType>>(
                preconditioner),
              solver_parameters);
    }
  else if (preconditioner_type == "Multigrid")
    {
      // note: handle it seperatly, since we need to set up the levels

      using OperatorType = LaplaceOperator<dim, Number>;

      MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
      MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
                                                   mg_constraints;
      MGLevelObject<std::shared_ptr<OperatorType>> mg_operators;

      MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>           transfers;
      std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorType>> transfer;

      const auto mg_degress = create_polynomial_coarsening_sequence(
        fe_degree,
        MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
          bisect);
      const auto mg_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          tria);

      const bool use_pmg = false;

      const unsigned int min_level = 0;
      const unsigned int max_level =
        (use_pmg ? mg_degress.size() : mg_triangulations.size()) - 1;

      mg_dof_handlers.resize(min_level, max_level);
      mg_constraints.resize(min_level, max_level);
      mg_operators.resize(min_level, max_level);

      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          const FE_Q<dim> mg_fe(use_pmg ? mg_degress[l] : fe_degree);
          const auto &    mg_tria = use_pmg ?
                                      static_cast<Triangulation<dim> &>(tria) :
                                      *mg_triangulations[l];

          mg_operators[l] =
            std::make_shared<OperatorType>(mapping, mg_tria, mg_fe, quadrature);
        }

      for (auto l = min_level; l <= max_level; ++l)
        {
          mg_dof_handlers[l] = std::shared_ptr<const DoFHandler<dim>>(
            &mg_operators[l]->get_dof_handler(),
            [](auto *) { /*nothing to do*/ });
          mg_constraints[l] = std::shared_ptr<const AffineConstraints<double>>(
            &mg_operators[l]->get_constraints(),
            [](auto *) { /*nothing to do*/ });
        }

      const auto preconditioner =
        std::make_shared<const MyMultigrid<dim, OperatorType, VectorType>>(
          preconditioner_parameters,
          op.get_dof_handler(),
          mg_dof_handlers,
          mg_constraints,
          mg_operators);

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters);
    }
  else
    {
      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto preconditioner =
        create_system_preconditioner(op, preconditioner_parameters);

      reduction_control =
        solve(op, solution, rhs, preconditioner, solver_parameters);
    }

  pcout << "Running with different preconditioners:" << std::endl;
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
