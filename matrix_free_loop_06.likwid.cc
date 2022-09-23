#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_tools.h>

#include <memory>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

#include "include/json.h"
#include "include/operator.h"
#include "include/precondition.h"

static unsigned int likwid_counter = 1;

struct Parameters
{
  unsigned int fe_degree           = 3;
  unsigned int n_subdivision       = 1;
  std::string  preconditioner_type = "relaxation";
  unsigned int optimization_level  = 2;
  bool         do_vmult            = true;

  unsigned int n_repetitions = 10;
  unsigned int max_degree    = 5;
};

template <int dim, typename Number>
void
setup_constraints(const DoFHandler<dim> &    dof_handler,
                  AffineConstraints<Number> &constraints)
{
  constraints.clear();
  for (unsigned int d = 0; d < dim; ++d)
    DoFTools::make_periodicity_constraints(
      dof_handler, 2 * d, 2 * d + 1, d, constraints);
  constraints.close();
}

template <int dim, typename Number>
void
test(const Parameters params_in)
{
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  FE_Q<dim>      fe(params_in.fe_degree);
  QGauss<dim>    quadrature(params_in.fe_degree + 1);
  MappingQ1<dim> mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  tria.refine_global(
    GridGenerator::subdivided_hyper_cube_balanced(tria,
                                                  params_in.n_subdivision));

  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;
  for (unsigned int d = 0; d < dim; ++d)
    GridTools::collect_periodic_faces(
      tria, 2 * d, 2 * d + 1, d, periodic_faces);
  tria.add_periodicity(periodic_faces);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  pcout << "Info" << std::endl;
  pcout << " - n dofs: " << dof_handler.n_dofs() << std::endl;
  pcout << std::endl << std::endl;

  AffineConstraints<Number> constraints;
  setup_constraints(dof_handler, constraints);

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  if (true) // TODO
    {
      DoFRenumbering::matrix_free_data_locality(dof_handler,
                                                constraints,
                                                additional_data);
      setup_constraints(dof_handler, constraints);
    }

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  // create linear operator
  typename LaplaceOperatorMatrixFree<dim, Number>::AdditionalData ad_operator;
  ad_operator.compress_indices = false;
  ad_operator.mapping_type     = "";

  LaplaceOperatorMatrixFree<dim, Number> op(matrix_free, ad_operator);

  // create preconditioner

  boost::property_tree::ptree params; // TODO: fill
  (void)params_in.preconditioner_type;
  (void)params_in.optimization_level;
  (void)params_in.max_degree;

  const auto precondition = create_system_preconditioner(op, params);

  VectorType src, dst;

  op.initialize_dof_vector(src);
  op.initialize_dof_vector(dst);

  for (unsigned int i = 0; i < params_in.n_repetitions; ++i)
    {
      (void)params_in.do_vmult;
      precondition->vmult(dst, src);
    }

  const std::string label =
    params_in.preconditioner_type + "_" + std::to_string(likwid_counter);

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_START(label.c_str());

  for (unsigned int i = 0; i < params_in.n_repetitions; ++i)
    {
      (void)params_in.do_vmult;
      precondition->vmult(dst, src);
    }

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_STOP(label.c_str());

  likwid_counter++;
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
  const unsigned int n_subdivision = (argc >= 4) ? std::atoi(argv[3]) : 6;
  const bool         use_float     = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const std::string  preconditioner_type =
    (argc >= 6) ? std::string(argv[5]) : std::string("chebyshev");
  const unsigned int optimization_level = (argc >= 7) ? std::atoi(argv[6]) : 2;
  const bool         do_vmult           = (argc >= 8) ? std::atoi(argv[7]) : 1;

  Parameters params;
  params.fe_degree           = fe_degree;
  params.n_subdivision       = n_subdivision;
  params.preconditioner_type = preconditioner_type;
  params.optimization_level  = optimization_level;
  params.do_vmult            = do_vmult;

  if (dim == 2 && use_float)
    test<2, float>(params);
  else if (dim == 3 && use_float)
    test<3, float>(params);
  else if (dim == 2)
    test<2, double>(params);
  else if (dim == 3)
    test<3, double>(params);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}