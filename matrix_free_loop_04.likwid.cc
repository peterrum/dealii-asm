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

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/restrictors.h"

using namespace dealii;

#include "include/matrix_free.h"
#include "include/operator.h"
#include "include/vector_access_reduced.h"

#define MAX_N_ROWS_FDM 10

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

template <int dim, typename Number>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap,
     const bool         use_cartesian_mesh,
     const bool         compress_indices,
     const unsigned int mapping_type)
{
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  const unsigned int mapping_degree = 1;

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

  pcout << "- n dofs: " << dof_handler.n_dofs() << std::endl;

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

  DoFRenumbering::matrix_free_data_locality(dof_handler,
                                            constraints,
                                            additional_data);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping_q_cache, dof_handler, constraints, quadrature, additional_data);

  typename LaplaceOperatorMatrixFree<dim, Number, VectorizedArrayType>::
    AdditionalData op_as;

  op_as.compress_indices = compress_indices;

  if (mapping_type == 0)
    op_as.mapping_type = "";
  else if (mapping_type == 1)
    op_as.mapping_type = "linear geometry";
  else if (mapping_type == 2)
    op_as.mapping_type = "quadratic geometry";
  else if (mapping_type == 3)
    op_as.mapping_type = "merged";
  else if (mapping_type == 4)
    op_as.mapping_type = "construct q";
  else
    AssertThrow(false, ExcNotImplemented());

  LaplaceOperatorMatrixFree<dim, Number, VectorizedArrayType> op(matrix_free,
                                                                 op_as);

  std::shared_ptr<ASPoissonPreconditionerBase<VectorType>> precon_fdm;

  const unsigned int n_rows = fe_degree + 2 * n_overlap - 1;

#define OPERATION(c, d)                                            \
  if (c == -1)                                                     \
    pcout << "Warning: FDM with <" + std::to_string(n_rows) +      \
               "> is not precompiled!"                             \
          << std::endl;                                            \
                                                                   \
  precon_fdm = std::make_shared<                                   \
    ASPoissonPreconditioner<dim, Number, VectorizedArrayType, c>>( \
    matrix_free,                                                   \
    n_overlap,                                                     \
    dim,                                                           \
    mapping_q_cache,                                               \
    fe_1D,                                                         \
    quadrature_face,                                               \
    quadrature_1D,                                                 \
    Restrictors::WeightingType::none);

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  op.set_partitioner(precon_fdm->get_partitioner());

  VectorType src, dst;

  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);

  src = 1;

  if (false)
    {
      LIKWID_MARKER_START("fdm");
      for (unsigned int i = 0; i < 10; ++i)
        precon_fdm->vmult(dst, src);
      LIKWID_MARKER_STOP("fdm");
    }
  else
    {
      LIKWID_MARKER_START("vmult");
      for (unsigned int i = 0; i < 10; ++i)
        op.vmult(dst, src);
      LIKWID_MARKER_STOP("vmult");
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

  const unsigned int dim              = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree        = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_refinements    = (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int n_overlap        = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const bool         cartesian_mesh   = (argc >= 6) ? std::atoi(argv[5]) : 1;
  const bool         use_float        = (argc >= 7) ? std::atoi(argv[6]) : 1;
  const bool         compress_indices = (argc >= 8) ? std::atoi(argv[7]) : 1;
  const unsigned int mapping_type     = (argc >= 9) ? std::atoi(argv[8]) : 0;


  if (dim == 2 && use_float)
    test<2, float>(fe_degree,
                   n_refinements,
                   n_overlap,
                   cartesian_mesh,
                   compress_indices,
                   mapping_type);
  else if (dim == 3 && use_float)
    test<3, float>(fe_degree,
                   n_refinements,
                   n_overlap,
                   cartesian_mesh,
                   compress_indices,
                   mapping_type);
  else if (dim == 2)
    test<2, double>(fe_degree,
                    n_refinements,
                    n_overlap,
                    cartesian_mesh,
                    compress_indices,
                    mapping_type);
  else if (dim == 3)
    test<3, double>(fe_degree,
                    n_refinements,
                    n_overlap,
                    cartesian_mesh,
                    compress_indices,
                    mapping_type);
  else
    AssertThrow(false, ExcInternalError());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
