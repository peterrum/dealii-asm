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
split_string(const std::string  text,
             const char         deliminator,
             const unsigned int size = numbers::invalid_unsigned_int)
{
  std::stringstream stream;
  stream << text;

  std::string              substring;
  std::vector<std::string> substring_list;

  while (std::getline(stream, substring, deliminator))
    substring_list.push_back(substring);

  if (size != numbers::invalid_unsigned_int)
    for (unsigned int i = substring_list.size(); i < size; ++i)
      substring_list.push_back("-");

  return substring_list;
}

void
process_fdm_parameters(const unsigned int              offset,
                       const std::vector<std::string> &props,
                       boost::property_tree::ptree &   params,
                       std::string &                   constness)
{
  const auto type               = props[offset + 0];
  const auto n_overlap          = props[offset + 1];
  const auto weighting_sequence = props[offset + 2];

  const bool overlap_pre_post =
    (weighting_sequence == "g") ? (props[offset + 3] == "p") : true;
  constness =
    (weighting_sequence == "g") ? (props[offset + 4]) : std::string("c");

  // configure preconditioner
  params.put("weighting type", (type == "add") ? "none" : type);

  if (n_overlap == "v")
    {
      params.put("element centric", false);
    }
  else
    {
      params.put("n overlap", n_overlap);
      params.put("element centric", true);
    }

  params.put("weight sequence",
             weighting_sequence == "g" ?
               "global" :
               (weighting_sequence == "l" ?
                  "local" :
                  (weighting_sequence == "dg" ? "DG" : "compressed")));

  params.put("overlap pre post", overlap_pre_post);
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
  pcout << " - degree: " << params_in.fe_degree << std::endl;
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
      const auto   props     = split_string(label, '-', 10);
      const auto   type      = props[0];
      std::string  constness = "c";
      unsigned int factor    = 1;

      // create preconditioner
      LaplaceOperatorMatrixFree<dim, Number> op(matrix_free, ad_operator);

      std::shared_ptr<
        const ASPoissonPreconditioner<dim, Number, VectorizedArray<Number>>>
        precondition_fdm;

      std::shared_ptr<const PreconditionerBase<VectorType>> precondition;

      if (type != "vmult")
        {
          if (type == "cheby")
            {
              boost::property_tree::ptree params;

              boost::property_tree::ptree params_fdm;

              if (props[3] == "diag")
                {
                  params_fdm.put("type", "Diagonal");
                }
              else
                {
                  std::string constness;
                  process_fdm_parameters(3, props, params_fdm, constness);
                  params_fdm.put("type", "FDM");
                }

              params.add_child("preconditioner", params_fdm);

              params.put("type", "Chebyshev");
              params.put("degree", std::atoi(props[1].c_str()));
              params.put("optimize", std::atoi(props[2].c_str()));

              factor = std::atoi(props[1].c_str());

              precondition = create_system_preconditioner(op, params);
            }
          else
            {
              boost::property_tree::ptree params;
              process_fdm_parameters(0, props, params, constness);
              precondition_fdm = create_fdm_preconditioner(op, params);
            }
        }

      // create vectors
      VectorType src, dst;
      op.initialize_dof_vector(src);
      op.initialize_dof_vector(dst);
      src = 1.0;

      // function to be excuated
      const auto fu = [&]() {
        if (type == "vmult")
          {
            op.vmult(dst, src);
          }
        else if (type == "add")
          {
            AssertThrow(precondition_fdm, ExcNotImplemented());

            precondition_fdm->vmult(dst, src, {}, {});
          }
        else if (constness == "c")
          {
            if (precondition_fdm)
              {
                precondition_fdm->vmult(dst,
                                        static_cast<const VectorType &>(src));
              }
            else if (precondition)
              {
                precondition->step(dst, src);
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }
        else if (constness == "n")
          {
            AssertThrow(precondition_fdm, ExcNotImplemented());
            precondition_fdm->vmult(dst, src);
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
            << " " << std::to_string(params_in.n_repetitions * factor) << " "
            << time << " " << std::to_string(sizeof(Number)) << " "
            << std::to_string(params_in.fe_degree) << " " << std::endl;
    }
}



/**
 * see: ../experiments/matrix_free_loop_08.sh
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  AssertThrow(argc >= 2, ExcNotImplemented());
  for (int i = 1; i < argc; ++i)
    {
      Parameters params;
      params.parse(argv[i]);

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
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}