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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

#include "include/functions.h"
#include "include/json.h"
#include "include/kershaw.h"
#include "include/matrix_free.h"
#include "include/multigrid.h"
#include "include/operator.h"
#include "include/precondition.h"
#include "include/preconditioners.h"
#include "include/restrictors.h"

#define COMPILE_MB 1
#define COMPILE_MF 1
#define COMPILE_2D 1
#define COMPILE_3D 1

static bool print_timings;

template <typename T>
using print_timings_t = decltype(std::declval<T const>().print_timings());

template <typename T>
constexpr bool has_timing_functionality =
  dealii::internal::is_supported_operation<print_timings_t, T>;



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



template <typename MatrixType, typename PreconditionerType, typename VectorType>
void
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
  const auto control_type =
    params.get<std::string>("control type", "ReductionControl");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << " - Solving with " << type << std::endl;
  pcout << "   - max iterations: " << max_iterations << std::endl;
  pcout << "   - abs tolerance:  " << abs_tolerance << std::endl;
  pcout << "   - rel tolrance:   " << rel_tolerance << std::endl;

  std::shared_ptr<SolverControl> solver_control;

  if (control_type == "ReductionControl")
    {
      solver_control = std::make_shared<ReductionControl>(max_iterations,
                                                          abs_tolerance,
                                                          rel_tolerance);
    }
  else if (control_type == "IterationNumberControl")
    {
      solver_control =
        std::make_shared<IterationNumberControl>(max_iterations, abs_tolerance);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  const auto dispatch = [&]() {
    x = 0;

    try
      {
        if (type == "CG")
          {
            SolverCG<VectorType> solver(*solver_control);
            solver.solve(A, x, b, *preconditioner);
          }
        else if (type == "FCG")
          {
            SolverFlexibleCG<VectorType> solver(*solver_control);
            solver.solve(A, x, b, *preconditioner);
          }
        else if (type == "GMRES")
          {
            typename SolverGMRES<VectorType>::AdditionalData additional_data;
            additional_data.right_preconditioning = true;
            additional_data.orthogonalization_strategy =
              SolverGMRES<VectorType>::AdditionalData::
                OrthogonalizationStrategy::classical_gram_schmidt;

            const auto max_n_tmp_vectors =
              params.get<int>("max n tmp vectors", 0);
            if (max_n_tmp_vectors > 0)
              additional_data.max_n_tmp_vectors = max_n_tmp_vectors;

            SolverGMRES<VectorType> solver(*solver_control, additional_data);
            solver.solve(A, x, b, *preconditioner);
          }
        else if (type == "FGMRES")
          {
            SolverFGMRES<VectorType> solver(*solver_control);
            solver.solve(A, x, b, *preconditioner);
          }
        else
          {
            AssertThrow(false,
                        ExcMessage("Solver <" + type + "> is not known!"))
          }
      }
    catch (const SolverControl::NoConvergence &)
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

      pcout << "   - n iterations:   " << solver_control->last_step()
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
    table.add_value("it", solver_control->last_step());
  else
    table.add_value("it", 999);

  if (print_timings)
    {
      table.add_value("time", time);

      if constexpr (has_timing_functionality<PreconditionerType>)
        preconditioner->print_timings();
    }
}



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
      auto n_subdivisions = mesh_parameters.get<int>("n subdivisions", 1);

      pcout << "- Create mesh: hypercube" << std::endl;
      pcout << std::endl;

      GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
      mapping_degree = std::min(mapping_degree, 1u);
    }
  else if (geometry_name == "symmetric hypercube")
    {
      auto n_subdivisions = mesh_parameters.get<int>("n subdivisions", 1);

      pcout << "- Create mesh: symmetric hypercube" << std::endl;
      pcout << std::endl;

      GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, -1.0, +1.0);
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

      auto n_intial_refinements =
        mesh_parameters.get<int>("n initial refinements", 1);
      auto n_subdivisions = mesh_parameters.get<int>("n subdivisions", 3);

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
      GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
      tria.refine_global(n_intial_refinements);

      mapping_degree = std::min(mapping_degree, 3u /*TODO*/);

      transformation_function = [epsy, epsz](const auto &,
                                             const auto &in_point) {
        Point<dim> out_point;
        double     dummy = 0.0;
        // clang-format off
        kershaw(epsy, epsz, in_point[0], in_point[1], dim == 3 ? in_point[2] : dummy, out_point[0], out_point[1], dim == 3 ? out_point[2] : dummy);
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

  for (const auto &cell : tria.cell_iterators())
    for (const auto &face : cell->face_iterators())
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

  std::shared_ptr<Function<dim>> rhs_func =
    std::make_shared<RightHandSide<dim>>();
  std::shared_ptr<Function<dim>> dbc_func =
    std::make_shared<Functions::ZeroFunction<dim>>();

  const std::string rhs_name = params.get<std::string>("rhs", "constant");

  if (rhs_name == "constant")
    {
      rhs_func = std::make_shared<RightHandSide<dim>>();
      dbc_func = std::make_shared<Functions::ZeroFunction<dim>>();
    }
  else if (rhs_name == "gaussian")
    {
      const std::vector<Point<dim>> points = {Point<dim>(-0.5, -0.5, -0.5)};
      const double                  width  = 0.1;

      rhs_func = std::make_shared<GaussianRightHandSide<dim>>(points, width);
      dbc_func = std::make_shared<GaussianSolution<dim>>(points, width);
    }
  else if (rhs_name == "gaussian-jw")
    {
      const std::vector<Point<dim>> points = {Point<dim>(0.0, 0.0, 0.0),
                                              Point<dim>(0.25, 0.85, 0.85),
                                              Point<dim>(0.6, 0.4, 0.4)};
      const double                  width  = 0.3;

      rhs_func = std::make_shared<GaussianRightHandSide<dim>>(points, width);
      dbc_func = std::make_shared<GaussianSolution<dim>>(points, width);
    }
  else
    {
      AssertThrow(false,
                  ExcMessage("RHS with the name <" + rhs_name +
                             "> is not known!"));
    }

  OperatorType op(mapping,
                  tria,
                  fe,
                  quadrature,
                  typename OperatorType::AdditionalData(op_compress_indices,
                                                        op_mapping_type),
                  dbc_func);

  table.add_value("n_cells", tria.n_global_active_cells());
  table.add_value("L", tria.n_global_levels());
  table.add_value("n_dofs", op.get_dof_handler().n_dofs());

  // create vectors
  VectorType solution, rhs;

  op.initialize_dof_vector(solution);
  op.initialize_dof_vector(rhs);

  op.rhs(rhs, rhs_func);
  rhs.zero_out_ghost_values();

  // ASM on cell level
  if (preconditioner_type == "Identity")
    {
      // note: handle it seperatly to exploit template specialization available
      // for SoverCG + PreconditionIdentity

      pcout << "- Create system preconditioner: Identity" << std::endl
            << std::endl;

      const auto preconditioner =
        std::make_shared<const PreconditionIdentity>();

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

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<std::shared_ptr<MappingQCache<dim>>>     mg_mapping;
      MGLevelObject<std::shared_ptr<LevelOperatorType>>      mg_operators;
      MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>  mg_dof_handlers;
      MGLevelObject<std::shared_ptr<const AffineConstraints<LevelNumber>>>
        mg_constraints;

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
      mg_triangulations =
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

      auto result = std::find_if(levels.rbegin(),
                                 levels.rend(),
                                 [](const auto &i) { return i.second == 1; });

      const unsigned int intermediate_level =
        ((result != levels.rend()) ?
           (std::distance(result, levels.rend()) - 1) :
           0) +
        min_level;

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
                                      mg_tria,
                                      transformation_function,
                                      false);
          else
            mg_mapping[l]->initialize(mapping_q1, mg_tria);

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
        mg_operators,
        intermediate_level);

      solve(op, solution, rhs, preconditioner, solver_parameters, table);
    }
  else
    {
      AssertThrow(preconditioner_type != "", ExcNotImplemented());

      const auto preconditioner =
        create_system_preconditioner(op, preconditioner_parameters);

      solve(op, solution, rhs, preconditioner, solver_parameters, table);
    }


  if (params.get<bool>("do output", false))
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(op.get_dof_handler());

      solution.update_ghost_values();
      op.get_constraints().distribute(solution);
      data_out.add_data_vector(solution, "solution");

      data_out.build_patches(mapping, 3);

      data_out.write_vtu_in_parallel("multigrid.vtu", MPI_COMM_WORLD);
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
