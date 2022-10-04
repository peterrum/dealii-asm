#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

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
  unsigned int dim         = 3;
  std::string  number_type = "double";

  unsigned int fe_degree     = 3;
  unsigned int n_subdivision = 1;

  std::string preconditioner_types = "post-1-c";

  bool dof_renumbering  = true;
  bool compress_indices = true;

  unsigned int n_repetitions = 10;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("dim", dim);
    prm.add_parameter("number type",
                      number_type,
                      "",
                      Patterns::Selection("double|float"));

    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("n subdivisions", n_subdivision);

    prm.add_parameter("preconditioner types", preconditioner_types);

    prm.add_parameter("n repetitions", n_repetitions);
    prm.add_parameter("dof renumbering", dof_renumbering);
  }
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

std::vector<std::string>
split_string(const std::string text, const char deliminator)
{
  std::stringstream stream;
  stream << text;

  std::string              substring;
  std::vector<std::string> substring_list;

  while (std::getline(stream, substring, deliminator))
    substring_list.push_back(substring);

  return substring_list;
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
  const unsigned int                        n_global_refinements =
    GridGenerator::subdivided_hyper_cube_balanced(tria,
                                                  params_in.n_subdivision,
                                                  true);

  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;
  for (unsigned int d = 0; d < dim; ++d)
    GridTools::collect_periodic_faces(
      tria, 2 * d, 2 * d + 1, d, periodic_faces);
  tria.add_periodicity(periodic_faces);

  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  pcout << "Info" << std::endl;
  pcout << " - n dofs: " << dof_handler.n_dofs() << std::endl;
  pcout << std::endl << std::endl;

  AffineConstraints<Number> constraints;
  setup_constraints(dof_handler, constraints);

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  if (params_in.dof_renumbering)
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
  ad_operator.compress_indices = params_in.compress_indices;
  ad_operator.mapping_type     = "";

  const auto labels = split_string(params_in.preconditioner_types, ' ');

  for (const auto label : labels)
    {
      // extract properties
      const auto props              = split_string(label, '-');
      const auto type               = props[0];
      const auto n_overlap          = props[1];
      const auto weighting_sequence = props[2];

      const bool overlap_pre_post =
        (weighting_sequence == "g") ? (props[3] == "p") : true;
      const std::string constness =
        (weighting_sequence == "g") ? (props[4]) : std::string("c");

      // configure preconditioner
      boost::property_tree::ptree params;
      params.put("weighting type", (type == "add") ? "none" : type);
      params.put("n overlap", n_overlap);

      params.put("weight sequence",
                 weighting_sequence == "g" ?
                   "global" :
                   (weighting_sequence == "l" ?
                      "local" :
                      (weighting_sequence == "dg" ? "DG" : "compressed")));

      params.put("overlap pre post", overlap_pre_post);

      // create preconditioner
      LaplaceOperatorMatrixFree<dim, Number> op(matrix_free, ad_operator);
      const auto precondition = create_fdm_preconditioner(op, params);

      // create vectors
      VectorType src, dst;
      op.initialize_dof_vector(src);
      op.initialize_dof_vector(dst);
      src = 1.0;

      // function to be excuated
      const auto fu = [&]() {
        if (type == "add")
          {
            precondition->vmult(dst, src, {}, {});
          }
        else if (constness == "c")
          {
            precondition->vmult(dst, static_cast<const VectorType &>(src));
          }
        else if (constness == "n")
          {
            precondition->vmult(dst, src);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
      };

      // warm up
      for (unsigned int i = 0; i < params_in.n_repetitions; ++i)
        fu();

      // time
      const auto add_padding = [](const int value) -> std::string {
        if (value < 10)
          return "000" + std::to_string(value);
        if (value < 100)
          return "00" + std::to_string(value);
        if (value < 1000)
          return "0" + std::to_string(value);
        if (value < 10000)
          return "" + std::to_string(value);

        AssertThrow(false, ExcInternalError());

        return "";
      };

      const std::string likwid_label =
        "likwid_" + add_padding(likwid_counter) + "_" + label; // TODO
      likwid_counter++;

      MPI_Barrier(MPI_COMM_WORLD);
      LIKWID_MARKER_START(likwid_label.c_str());
      const auto timer = std::chrono::system_clock::now();

      for (unsigned int i = 0; i < params_in.n_repetitions; ++i)
        fu();

      MPI_Barrier(MPI_COMM_WORLD);
      LIKWID_MARKER_STOP(likwid_label.c_str());

      const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now() - timer)
                            .count() /
                          1e9;

      pcout << ">> " << label << " " << std::to_string(dof_handler.n_dofs())
            << " " << std::to_string(params_in.n_repetitions) << " " << time
            << " " << std::endl;
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

  AssertThrow(argc == 2, ExcNotImplemented());

  Parameters params;
  params.parse(argv[1]);

  if ((params.dim == 2) && (params.number_type == "float"))
    test<2, float>(params);
  else if ((params.dim == 3) && (params.number_type == "float"))
    test<3, float>(params);
  else if ((params.dim == 2) && (params.number_type == "double"))
    test<2, double>(params);
  else if ((params.dim == 3) && (params.number_type == "double"))
    test<3, double>(params);
  else
    AssertThrow(false, ExcNotImplemented());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}