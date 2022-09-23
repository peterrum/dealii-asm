#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>

using namespace dealii;

#include "include/vector_access_reduced.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif


template <int dim,
          int n_components,
          typename Number,
          std::size_t width = VectorizedArray<Number>::size()>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const bool         apply_dbcs,
     ConvergenceTable & table)
{
  using VectorizedArrayType        = VectorizedArray<Number, width>;
  using VectorType                 = LinearAlgebra::distributed::Vector<Number>;
  const unsigned int n_repetitions = 10;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  FESystem<dim>  fe(FE_Q<dim>(fe_degree), n_components);
  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(fe_degree + 1);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  if (false)
    {
      GridGenerator::hyper_cube(tria);
    }
  else
    {
      GridGenerator::hyper_ball_balanced(tria);
    }

  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;

  constraints.clear();
  if (apply_dbcs)
    DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  DoFRenumbering::matrix_free_data_locality(dof_handler,
                                            constraints,
                                            additional_data);

  constraints.clear();
  if (apply_dbcs)
    DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  VectorType src, dst;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);

  for (const auto i : src.locally_owned_elements())
    src[i] = i;

  ConstraintInfoReduced cir;

  cir.initialize(matrix_free);

  pcout << "- n dofs:           " << dof_handler.n_dofs() << std::endl;
  pcout << "- compression type: "
        << Utilities::MPI::max(cir.compression_level(), MPI_COMM_WORLD)
        << std::endl;

  static unsigned int likwid_counter = 1;

  table.add_value("n_dofs", dof_handler.n_dofs());

  for (unsigned int i = 0; i < 3; ++i)
    {
      const std::string label = "test_" + std::to_string(likwid_counter);

      if (i == 2)
        cir.set_do_adjust_for_orientation(false);
      MPI_Barrier(MPI_COMM_WORLD);
      LIKWID_MARKER_START(label.c_str());

      const auto timer = std::chrono::system_clock::now();

      for (unsigned int c = 0; c < n_repetitions; ++c)
        {
          matrix_free.template cell_loop<VectorType, VectorType>(
            [&](const auto &matrix_free,
                auto &      dst,
                const auto &src,
                const auto  range) {
              FEEvaluation<dim,
                           -1,
                           0,
                           n_components,
                           Number,
                           VectorizedArrayType>
                phi(matrix_free);

              for (unsigned int cell = range.first; cell < range.second; ++cell)
                {
                  if (i == 0)
                    {
                      phi.reinit(cell);
                      phi.read_dof_values(src);
                      phi.distribute_local_to_global(dst);
                    }
                  else
                    {
                      phi.reinit(cell);
                      cir.read_dof_values(src, phi);
                      cir.distribute_local_to_global(dst, phi);
                    }
                }
            },
            dst,
            src);
        }

      const double time_total =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - timer)
          .count() /
        1e9;

      MPI_Barrier(MPI_COMM_WORLD);
      LIKWID_MARKER_STOP(label.c_str());

      table.add_value("time_" + std::to_string(i), time_total);
      table.set_scientific("time_" + std::to_string(i), true);

      likwid_counter++;
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
  const unsigned int n_components  = (argc >= 4) ? std::atoi(argv[3]) : 1;
  const unsigned int n_refinements = (argc >= 5) ? std::atoi(argv[4]) : 6;
  const bool         apply_dbcs    = (argc >= 6) ? std::atoi(argv[5]) : 0;
  const bool         use_float     = (argc >= 7) ? std::atoi(argv[6]) : 1;

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

  ConvergenceTable table;

  if (use_float)
    {
      if (dim == 2 && n_components == 1)
        test<2, 1, float>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 3 && n_components == 1)
        test<3, 1, float>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 2 && n_components == 2)
        test<2, 2, float>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 3 && n_components == 3)
        test<3, 3, float>(fe_degree, n_refinements, apply_dbcs, table);
      else
        AssertThrow(false, ExcInternalError());
    }
  else
    {
      if (dim == 2 && n_components == 1)
        test<2, 1, double>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 3 && n_components == 1)
        test<3, 1, double>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 2 && n_components == 2)
        test<2, 2, double>(fe_degree, n_refinements, apply_dbcs, table);
      else if (dim == 3 && n_components == 3)
        test<3, 3, double>(fe_degree, n_refinements, apply_dbcs, table);
      else
        AssertThrow(false, ExcInternalError());
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
