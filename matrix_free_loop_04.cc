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

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/restrictors.h"

using namespace dealii;

#include "include/matrix_free.h"
#include "include/operator.h"

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap,
     const bool         use_cartesian_mesh)
{
  using Number              = float;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  const unsigned int mapping_degree = fe_degree;

  FE_Q<dim> fe(fe_degree);
  FE_Q<1>   fe_1D(fe_degree);

  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> quadrature_face(fe_degree + 1);
  QGauss<1>       quadrature_1D(fe_degree + 1);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, 0, 1, true);

  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;
  for (unsigned int d = 0; d < dim; ++d)
    GridTools::collect_periodic_faces(
      tria, 2 * d, 2 * d + 1, d, periodic_faces);
  tria.add_periodicity(periodic_faces);

  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  MappingQ<dim>      mapping(mapping_degree);
  MappingQCache<dim> mapping_q_cache(mapping_degree);

  mapping_q_cache.initialize(
    mapping,
    tria,
    [use_cartesian_mesh](const auto &, const auto &point) {
      Point<dim> result;

      if (use_cartesian_mesh)
        return result;

      for (unsigned int d = 0; d < dim; ++d)
        result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                    std::sin(numbers::PI * point[d]) * 0.1;

      return result;
    },
    true);

  AffineConstraints<Number> constraints;

  for (unsigned int d = 0; d < dim; ++d)
    DoFTools::make_periodicity_constraints(
      dof_handler, 2 * d, 2 * d + 1, d, constraints);

  constraints.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping_q_cache, dof_handler, constraints, quadrature, additional_data);

  LaplaceOperatorMatrixFree<dim, Number, VectorizedArrayType> op(matrix_free);

  ASPoissonPreconditioner<dim, Number, VectorizedArrayType, -1> fdm(
    matrix_free,
    n_overlap,
    dim,
    mapping_q_cache,
    fe_1D,
    quadrature_face,
    quadrature_1D);

  std::cout << fdm.n_fdm_instances() << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int n_overlap          = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const bool         use_cartesian_mesh = (argc >= 6) ? std::atoi(argv[5]) : 1;


  if (dim == 2)
    test<2>(fe_degree, n_global_refinements, n_overlap, use_cartesian_mesh);
  else
    AssertThrow(false, ExcInternalError());
}
