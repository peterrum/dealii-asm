#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include "include/grid_generator.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

static unsigned int likwid_counter = 0;

template <int dim, typename Number>
void
test(const unsigned int fe_degree_fine,
     const unsigned int n_subdivision,
     ConvergenceTable & table)
{
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  AssertThrow(fe_degree_fine > 1, ExcNotImplemented());

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  tria.refine_global(
    GridGenerator::subdivided_hyper_cube_balanced(tria, n_subdivision));

  DoFHandler<dim> dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(FE_Q<dim>(fe_degree_fine));

  const auto partitioner_fine =
    std::make_shared<const Utilities::MPI::Partitioner>(
      dof_handler_fine.locally_owned_dofs(),
      DoFTools::extract_locally_active_dofs(dof_handler_fine),
      MPI_COMM_WORLD);

  std::set<unsigned int> fe_degrees_coarse = {1,
                                              fe_degree_fine / 2,
                                              fe_degree_fine - 1};

  for (const auto fe_degree_coarse : fe_degrees_coarse)
    {
      DoFHandler<dim> dof_handler_coarse(tria);
      dof_handler_coarse.distribute_dofs(FE_Q<dim>(fe_degree_coarse));

      const auto partitioner_coarse =
        std::make_shared<const Utilities::MPI::Partitioner>(
          dof_handler_coarse.locally_owned_dofs(),
          DoFTools::extract_locally_active_dofs(dof_handler_coarse),
          MPI_COMM_WORLD);

      MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(0, 1);
      transfers[1].reinit(dof_handler_fine, dof_handler_coarse);

      MGTransferGlobalCoarsening<dim, VectorType> transfer(
        transfers, [&](const auto l, auto &vec) {
          if (l == 0)
            vec.reinit(partitioner_coarse);
          else if (l == 1)
            vec.reinit(partitioner_fine);
          else
            AssertThrow(false, ExcNotImplemented());
        });

      VectorType vector_fine, vector_coarse;
      vector_fine.reinit(partitioner_fine);
      vector_coarse.reinit(partitioner_coarse);

      vector_fine = 1.0;

      const unsigned int n_repetitions = 10;

      const auto run = [&](const auto &fu, const std::string prefix) {
        std::string label = prefix + "_";

        if (likwid_counter < 10)
          label = label + "000" + std::to_string(likwid_counter);
        else if (likwid_counter < 100)
          label = label + "00" + std::to_string(likwid_counter);
        else if (likwid_counter < 1000)
          label = label + "0" + std::to_string(likwid_counter);
        else
          AssertThrow(false, ExcNotImplemented());

        for (unsigned int c = 0; c < n_repetitions; ++c)
          fu();

        MPI_Barrier(MPI_COMM_WORLD);

        if (label != "")
          LIKWID_MARKER_START(label.c_str());

        auto temp_time = std::chrono::system_clock::now();

        for (unsigned int c = 0; c < n_repetitions; ++c)
          fu();

        MPI_Barrier(MPI_COMM_WORLD);

        const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now() - temp_time)
                            .count() /
                          1e9;

        if (label != "")
          LIKWID_MARKER_STOP(label.c_str());

        return time;
      };

      const auto time_r =
        run([&]() { transfer.restrict_and_add(1, vector_coarse, vector_fine); },
            "restrict_and_add");

      const auto time_p = run(
        [&]() { transfer.prolongate_and_add(1, vector_fine, vector_coarse); },
        "prolongate_and_add");

      table.add_value("n_subdivision", n_subdivision);
      table.add_value("n_cells", tria.n_global_active_cells());

      table.add_value("degree_fine", fe_degree_fine);
      table.add_value("degree_coarse", fe_degree_coarse);

      table.add_value("n_dofs_fine", dof_handler_fine.n_dofs());
      table.add_value("n_dofs_coarse", dof_handler_coarse.n_dofs());

      table.add_value("time_restriction",
                      dof_handler_fine.n_dofs() * n_repetitions / time_r);
      table.set_scientific("time_restriction", true);
      table.add_value("time_prolongation",
                      dof_handler_fine.n_dofs() * n_repetitions / time_p);
      table.set_scientific("time_prolongation", true);
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

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  const bool verbose = true;

  const unsigned int dim               = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree         = (argc >= 3) ? std::atoi(argv[2]) : 4;
  const unsigned int min_n_subdivision = (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int max_n_subdivision =
    (argc >= 5) ? std::atoi(argv[4]) : min_n_subdivision;

  std::vector<unsigned int> n_subdivisions;

  for (unsigned int i = min_n_subdivision; i <= max_n_subdivision; ++i)
    n_subdivisions.emplace_back(i);

  ConvergenceTable table;

  for (unsigned int i = 0; i < n_subdivisions.size(); ++i)
    {
      if (dim == 2)
        test<2, double>(fe_degree, n_subdivisions[i], table);
      else if (dim == 3)
        test<3, double>(fe_degree, n_subdivisions[i], table);
      else
        AssertThrow(false, ExcNotImplemented());

      if (is_root && (verbose || ((i + 1) == n_subdivisions.size())))
        {
          table.write_text(std::cout, ConvergenceTable::org_mode_table);
          std::cout << std::endl;
        }
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}