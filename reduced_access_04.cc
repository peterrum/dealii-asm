#include <deal.II/base/conditional_ostream.h>
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


template <int dim,
          int n_components,
          typename Number,
          std::size_t width = VectorizedArray<Number>::size()>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const bool         apply_dbcs)
{
  using VectorizedArrayType = VectorizedArray<Number, width>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

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

  VectorType src;
  matrix_free.initialize_dof_vector(src);

  for (const auto i : src.locally_owned_elements())
    src[i] = i;

  ConstraintInfoReduced cir;

  cir.initialize(matrix_free);

  pcout << "- n dofs:           " << dof_handler.n_dofs() << std::endl;
  pcout << "- compression type: " << cir.compression_level() << std::endl;

  FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType> phi0(
    matrix_free);
  FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType> phi1(
    matrix_free);

  const auto print = [&](const auto &phi) {
    if (pcout.is_active())
      {
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          printf("%5.0f ", phi.begin_dof_values()[i][0]);
        printf("\n");
      }
  };


  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      phi0.reinit(cell);
      phi0.read_dof_values(src);

      phi1.reinit(cell);
      cir.read_dof_values(src, phi1);

      for (unsigned int v = 0;
           v < matrix_free.n_active_entries_per_cell_batch(cell);
           ++v)
        for (unsigned int i = 0; i < phi0.dofs_per_cell; ++i)
          {
            if (phi0.begin_dof_values()[i][v] != phi1.begin_dof_values()[i][v])
              {
                print(phi0);
                print(phi1);

                AssertThrow(
                  false,
                  ExcMessage(
                    "Values do not match <" +
                    std::to_string(phi0.begin_dof_values()[i][v]) + "> vs. <" +
                    std::to_string(phi1.begin_dof_values()[i][v]) + "> !"));
              }
          }

      phi0.distribute_local_to_global(src);
      cir.distribute_local_to_global(src, phi1);
    }

  pcout << "OK!" << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim           = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree     = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_components  = (argc >= 4) ? std::atoi(argv[3]) : 1;
  const unsigned int n_refinements = (argc >= 5) ? std::atoi(argv[4]) : 6;
  const bool         apply_dbcs    = (argc >= 6) ? std::atoi(argv[5]) : 0;

  if (dim == 2 && n_components == 1)
    test<2, 1, double>(fe_degree, n_refinements, apply_dbcs);
  else if (dim == 3 && n_components == 1)
    test<3, 1, double>(fe_degree, n_refinements, apply_dbcs);
  else if (dim == 2 && n_components == 2)
    test<2, 2, double>(fe_degree, n_refinements, apply_dbcs);
  else if (dim == 3 && n_components == 3)
    test<3, 3, double>(fe_degree, n_refinements, apply_dbcs);
  else
    AssertThrow(false, ExcInternalError());
}
