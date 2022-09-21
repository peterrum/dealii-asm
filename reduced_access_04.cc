#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>

using namespace dealii;

#include "include/vector_access_reduced.h"


template <int dim, typename Number, std::size_t width = 1>
void
test(const unsigned int fe_degree, const unsigned int n_global_refinements)
{
  using VectorizedArrayType = VectorizedArray<Number, width>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  FE_Q<dim>      fe(fe_degree);
  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(fe_degree + 1);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  if(false)
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
  constraints.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  DoFRenumbering::matrix_free_data_locality(dof_handler,
                                            constraints,
                                            additional_data);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  VectorType src;
  matrix_free.initialize_dof_vector(src);

  for (const auto i : src.locally_owned_elements())
    src[i] = i;

  ConstraintInfoReduced cir;

  cir.initialize(matrix_free);

  pcout << cir.compression_level() << std::endl;

  FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(matrix_free);

  const auto print = [&]() {
    if (pcout.is_active())
      {
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          printf("%5.0f ", phi.begin_dof_values()[i][0]);
        printf("\n");
      }
  };

  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      phi.reinit(cell);

      phi.read_dof_values(src);
      print();

      cir.read_dof_values(src, phi);
      print();
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim           = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree     = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_refinements = (argc >= 4) ? std::atoi(argv[3]) : 6;

  if (dim == 2)
    test<2, double>(fe_degree, n_refinements);
  else if (dim == 3)
    test<3, double>(fe_degree, n_refinements);
  else
    AssertThrow(false, ExcInternalError());
}
