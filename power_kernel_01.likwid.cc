#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#include "include/grid_generator.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

static unsigned int likwid_counter = 0;

struct Parameters
{
  unsigned int dim              = 3;
  unsigned int fe_degree        = 1;
  unsigned int n_components     = 1;
  unsigned int subdivisions     = 34;
  unsigned int n_lanes          = 0;
  unsigned int cell_granularity = 0;
  unsigned int n_repetitions    = 10;
  bool         dof_renumbering  = true;
  bool         use_dg           = false;
  bool         do_computation   = false;

  std::string number_type = "double";

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

  void
  print()
  {
    print(ParameterHandler::OutputStyle::ShortJSON);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("dim", dim);
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("n components", n_components);
    prm.add_parameter("n subdivisions", subdivisions);
    prm.add_parameter("n lanes", n_lanes);
    prm.add_parameter("cell granularity", cell_granularity);
    prm.add_parameter("n repetitions", n_repetitions);
    prm.add_parameter("dof renumbering", dof_renumbering);
    prm.add_parameter("use dg", use_dg);
    prm.add_parameter("do computation", do_computation);
    prm.add_parameter("number type",
                      number_type,
                      "",
                      Patterns::Selection("double|float"));
  }

  void
  print(const ParameterHandler::OutputStyle style)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    if (pcout.is_active())
      prm.print_parameters(pcout.get_stream(), style);
  }
};

template <int dim, typename Number>
class Fu : public Function<dim, Number>
{
public:
  virtual Number
  value(const Point<dim> &p, const unsigned int component) const override
  {
    (void)component;

    return std::sin(p[0]);
  }
};

struct PrePost
{
  std::vector<unsigned int> pre_indices;
  std::vector<unsigned int> pre_indices_ptr;
  std::vector<unsigned int> post_indices;
  std::vector<unsigned int> post_indices_ptr;
};


template <typename Fu>
PrePost
determine_pre_post(const Fu &         matrix_free_cell_loop,
                   const unsigned int batch_size,
                   const bool         track_individual_cell,
                   const unsigned int cell_granularity)
{
  unsigned int n_vertices     = 0;
  unsigned int n_cell_batches = 0;

  matrix_free_cell_loop(
    [&](const auto &data, auto &, const auto &, const auto) {
      n_vertices     = data.get_dof_handler().get_triangulation().n_vertices();
      n_cell_batches = data.n_cell_batches();
    });

  std::vector<std::pair<unsigned int, unsigned int>> vertex_tracker(
    n_vertices,
    std::pair<unsigned int, unsigned int>(numbers::invalid_unsigned_int, 0));

  unsigned int counter = 0;

  matrix_free_cell_loop([&](const auto &data,
                            auto &,
                            const auto &,
                            const auto cells) {
    for (unsigned int cell = cells.first; cell < cells.second; ++cell)
      {
        const auto n_active_entries_per_cell_batch =
          data.n_active_entries_per_cell_batch(cell);

        for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
          {
            const auto cell_iterator = data.get_cell_iterator(cell, v);

            for (const auto i : cell_iterator->vertex_indices())
              {
                vertex_tracker[cell_iterator->vertex_index(i)].first =
                  std::min(vertex_tracker[cell_iterator->vertex_index(i)].first,
                           counter);
                vertex_tracker[cell_iterator->vertex_index(i)].second =
                  std::max(
                    vertex_tracker[cell_iterator->vertex_index(i)].second,
                    counter);
              }
          }
      }
    counter++;
  });

  const unsigned n_entities_to_track =
    track_individual_cell ? (n_cell_batches * batch_size) : n_cell_batches;

  std::vector<unsigned int> min_vector(n_entities_to_track,
                                       numbers::invalid_unsigned_int);
  std::vector<unsigned int> max_vector(n_entities_to_track, 0);

  counter = 0;
  matrix_free_cell_loop(
    [&](const auto &data, auto &, const auto &, const auto cells) {
      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          const auto n_active_entries_per_cell_batch =
            data.n_active_entries_per_cell_batch(cell);

          for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
            {
              const auto cell_iterator = data.get_cell_iterator(cell, v);

              for (const auto i : cell_iterator->vertex_indices())
                {
                  const unsigned int index =
                    track_individual_cell ? (cell * batch_size + v) : cell;

                  min_vector[index] = std::min(
                    min_vector[index],
                    vertex_tracker[cell_iterator->vertex_index(i)].first);

                  max_vector[index] = std::max(
                    max_vector[index],
                    vertex_tracker[cell_iterator->vertex_index(i)].second);
                }
            }
        }
      counter++;
    });

  const auto process = [&](const auto &ids) {
    std::vector<std::vector<unsigned int>> temp(counter);

    for (unsigned int i = 0; i < ids.size(); ++i)
      if (ids[i] != numbers::invalid_unsigned_int)
        {
          if (false)
            {
              if ((static_cast<int>(ids[i]) - 0) <=
                  static_cast<int>(i / (track_individual_cell ?
                                          (cell_granularity) :
                                          (cell_granularity / batch_size))))
                temp[ids[i]].push_back(i);
              else
                temp.back().push_back(i);
            }
          else
            {
              temp[ids[i]].push_back(i);
            }
        }

    if (false)
      for (auto &vector : temp)
        {
          std::sort(
            vector.begin(), vector.end(), [&](const auto &a, const auto &b) {
              const auto a_batch =
                a / (track_individual_cell ? (cell_granularity) :
                                             (cell_granularity / batch_size));
              const auto b_batch =
                b / (track_individual_cell ? (cell_granularity) :
                                             (cell_granularity / batch_size));

              if (a_batch != b_batch)
                return b_batch < a_batch;

              return a < b;
            });
        }

    std::vector<unsigned int> indices;
    std::vector<unsigned int> ptr = {0};

    for (const auto &vector : temp)
      {
        for (const auto &v : vector)
          indices.push_back(v);

        ptr.push_back(indices.size());
      }

    return std::make_pair(indices, ptr);
  };

  PrePost result;
  std::tie(result.pre_indices, result.pre_indices_ptr)   = process(min_vector);
  std::tie(result.post_indices, result.post_indices_ptr) = process(max_vector);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << result.post_indices.size() << " "
                << result.post_indices_ptr.size() << std::endl;

      for (unsigned int i = 0; i < result.post_indices_ptr.size() - 1; ++i)
        {
          printf("%4d %4d ",
                 i,
                 result.post_indices_ptr[i + 1] - result.post_indices_ptr[i]);

          unsigned int counter_total = 0;

          for (unsigned int k = 0;
               counter_total !=
               result.post_indices_ptr[i + 1] - result.post_indices_ptr[i];
               ++k)
            {
              unsigned int counter = 0;

              for (unsigned int j = result.post_indices_ptr[i];
                   j < result.post_indices_ptr[i + 1];
                   ++j)
                if ((i - k) == result.post_indices[j] /
                                 (track_individual_cell ?
                                    (cell_granularity) :
                                    (cell_granularity / batch_size)))
                  counter++;

              printf("%d ", counter);

              counter_total += counter;
            }

          printf("\n");
        }
      std::cout << std::endl;
    }

  return result;
}

template <int dim, typename Number, std::size_t n_lanes>
void
run(const Parameters &params, ConvergenceTable &table)
{
  const unsigned int fe_degree    = params.fe_degree;
  const unsigned int n_components = params.n_components;

  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number, n_lanes>;
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  Triangulation<dim> tria;
  tria.refine_global(
    GridGenerator::subdivided_hyper_cube_balanced(tria, params.subdivisions));

  MappingQ1<dim>   mapping;
  QGaussLobatto<1> quad(fe_degree + 1);

  DoFHandler<dim> dof_handler(tria);
  if (params.use_dg)
    dof_handler.distribute_dofs(
      FESystem<dim>{FE_DGQ<dim>(fe_degree), n_components});
  else
    dof_handler.distribute_dofs(
      FESystem<dim>{FE_Q<dim>(fe_degree), n_components});

  AffineConstraints<Number> constraints;

  // setup matrixfree
  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData mf_data;
  mf_data.tasks_parallel_scheme = MatrixFree<dim, Number, VectorizedArrayType>::
    AdditionalData::TasksParallelScheme::none;

  if (params.dof_renumbering)
    DoFRenumbering::matrix_free_data_locality(dof_handler,
                                              constraints,
                                              mf_data);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  matrix_free.reinit(mapping, dof_handler, constraints, quad, mf_data);

  // wrap matrix-free loop to be able to control granularity
  const auto matrix_free_cell_loop = [&](const auto fu) {
    Number dummy = 0;

    if (params.cell_granularity == 0)
      {
        matrix_free.template cell_loop<Number, Number>(fu, dummy, dummy);
      }
    else
      {
        AssertThrow(params.cell_granularity >= VectorizedArrayType::size(),
                    ExcInternalError());

        const unsigned int stride =
          params.cell_granularity / VectorizedArrayType::size();
        const unsigned int n_cell_batches = matrix_free.n_cell_batches();

        for (unsigned int i = 0; i < n_cell_batches; i += stride)
          fu(matrix_free,
             dummy,
             dummy,
             std::make_pair(i, std::min(i + stride, n_cell_batches)));
      }
  };

  const auto pre_post_own = determine_pre_post(matrix_free_cell_loop,
                                               VectorizedArrayType::size(),
                                               true,
                                               params.cell_granularity);

  const auto pre_post_batch = determine_pre_post(matrix_free_cell_loop,
                                                 VectorizedArrayType::size(),
                                                 false,
                                                 params.cell_granularity);

  // intialize vectors
  VectorType src, dst_0, dst_1;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_0);
  matrix_free.initialize_dof_vector(dst_1);
  VectorTools::interpolate(mapping, dof_handler, Fu<dim, Number>(), src);

  // define vmult operations and ...
  const auto process_batch_vmult =
    [&](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);

      if (params.do_computation)
        {
          phi.evaluate(EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);

          phi.integrate(EvaluationFlags::gradients);
        }

      phi.distribute_local_to_global(dst);
    };

  // ... post operation
  const auto process_batch_post =
    [&](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);

      if (params.do_computation)
        {
          phi.evaluate(EvaluationFlags::values);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_value(phi.get_value(q), q);

          phi.integrate(EvaluationFlags::values);
        }

      phi.distribute_local_to_global(dst);
    };

  // helper function to run performance study
  const auto run = [&](const auto &fu, const std::string prefix) {
    dst_0 = 0.0;
    dst_1 = 0.0;

    std::string label = prefix + "_";

    if (likwid_counter < 10)
      label = label + "000" + std::to_string(likwid_counter);
    else if (likwid_counter < 100)
      label = label + "00" + std::to_string(likwid_counter);
    else if (likwid_counter < 1000)
      label = label + "0" + std::to_string(likwid_counter);
    else
      AssertThrow(false, ExcNotImplemented());

    for (unsigned int c = 0; c < params.n_repetitions; ++c)
      fu();

    MPI_Barrier(MPI_COMM_WORLD);

    if (label != "")
      LIKWID_MARKER_START(label.c_str());

    auto temp_time = std::chrono::system_clock::now();

    for (unsigned int c = 0; c < params.n_repetitions; ++c)
      fu();

    MPI_Barrier(MPI_COMM_WORLD);

    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - temp_time)
                        .count() /
                      1e9;

    if (label != "")
      LIKWID_MARKER_STOP(label.c_str());

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << src.l2_norm() << " " << dst_0.l2_norm() << " "
                << dst_1.l2_norm() << std::endl;

    return time;
  };

  // version 1: power kernel (use own batches)
  const auto time_power_own = run(
    [&]() {
      unsigned int counter = 0;

      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi_0(data);
          FECellIntegrator phi_1(data);

          // vmult
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            process_batch_vmult(cell, phi_0, dst_0, src);

          // post vmult
          for (unsigned int i = pre_post_own.post_indices_ptr[counter];
               i < pre_post_own.post_indices_ptr[counter + 1];)
            {
              std::array<unsigned int, VectorizedArrayType::size()> ids = {};
              ids.fill(numbers::invalid_unsigned_int);

              for (unsigned int v = 0;
                   (v < VectorizedArrayType::size()) &&
                   (i < pre_post_own.post_indices_ptr[counter + 1]);
                   ++v, ++i)
                ids[v] = pre_post_own.post_indices[i];

              process_batch_post(ids, phi_1, dst_1, dst_0);
            }

          counter++;
        });
    },
    "powero");

  // version 2: power kernel (use matrix-free batches)
  const auto time_power_batch = run(
    [&]() {
      unsigned int counter = 0;

      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi_0(data);
          // FECellIntegrator phi_1(data);

          // vmult
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            process_batch_vmult(cell, phi_0, dst_0, src);

          // post vmult
          if (pre_post_batch.post_indices_ptr.size() == 2)
            {
              for (unsigned int cell = cells.first; cell < cells.second; ++cell)
                process_batch_post(cell, phi_0, dst_1, dst_0);
            }
          else
            {
              for (unsigned int i = pre_post_batch.post_indices_ptr[counter];
                   i < pre_post_batch.post_indices_ptr[counter + 1];
                   ++i)
                process_batch_post(pre_post_batch.post_indices[i],
                                   phi_0,
                                   dst_1,
                                   dst_0);
            }

          counter++;
        });
    },
    "powerb");

  // version 3: run sequentially
  const auto time_sequential = run(
    [&]() {
      unsigned int counter = 0;
      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);

          if (true)
            {
              for (unsigned int cell = cells.first; cell < cells.second; ++cell)
                process_batch_vmult(cell, phi, dst_0, src);
            }
          else
            {
              for (unsigned int i = pre_post_batch.post_indices_ptr[counter];
                   i < pre_post_batch.post_indices_ptr[counter + 1];
                   ++i)
                process_batch_post(pre_post_batch.post_indices[i],
                                   phi,
                                   dst_0,
                                   src);
              counter++;
            }
        });

      counter = 0;
      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);

          if (true)
            {
              for (unsigned int cell = cells.first; cell < cells.second; ++cell)
                process_batch_post(cell, phi, dst_1, dst_0);
            }
          else
            {
              for (unsigned int i = pre_post_batch.post_indices_ptr[counter];
                   i < pre_post_batch.post_indices_ptr[counter + 1];
                   ++i)
                process_batch_post(pre_post_batch.post_indices[i],
                                   phi,
                                   dst_1,
                                   dst_0);
              counter++;
            }
        });
    },
    "sequential");

  const unsigned int n_mpi_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);


  table.add_value("degree", fe_degree);
  table.add_value("n_lanes", VectorizedArrayType::size());
  table.add_value("granularity", params.cell_granularity);
  table.add_value("n_repetitions", params.n_repetitions);
  table.add_value("n_procs", n_mpi_processes);
  table.add_value("n_cells", tria.n_active_cells());
  table.add_value("n_dofs", dof_handler.n_dofs());

  table.add_value("s_own", time_sequential / time_power_own);
  table.set_scientific("s_own", true);
  table.add_value("s_batch", time_sequential / time_power_batch);
  table.set_scientific("s_batch", true);

  table.add_value("t_own", time_power_own);
  table.set_scientific("t_own", true);
  table.add_value("t_batch", time_power_batch);
  table.set_scientific("t_batch", true);
  table.add_value("t_sequential", time_sequential);
  table.set_scientific("t_sequential", true);

  const unsigned int dofs_ =
    2 * dof_handler.n_dofs() * params.n_repetitions * n_mpi_processes;

  table.add_value("tp_own", dofs_ / time_power_own);
  table.set_scientific("tp_own", true);
  table.add_value("tp_batch", dofs_ / time_power_batch);
  table.set_scientific("tp_batch", true);
  table.add_value("tp_sequential", dofs_ / time_sequential);
  table.set_scientific("tp_sequential", true);

  likwid_counter++;
}

template <int dim, typename T>
void
run_number(const Parameters &params, ConvergenceTable &table)
{
  unsigned int n_lanes = params.n_lanes;

  constexpr std::size_t n_lanes_max = VectorizedArray<T>::size();

  if (n_lanes == 0)
    n_lanes = n_lanes_max;

  AssertThrow(n_lanes <= n_lanes_max, ExcNotImplemented());

  if (n_lanes == 1)
    run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params, table);
  else if (n_lanes == 2)
    run<dim, T, std::min<std::size_t>(2, n_lanes_max)>(params, table);
  else if (n_lanes == 4)
    run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params, table);
  else if (n_lanes == 8)
    run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params, table);
  else if (n_lanes == 16)
    run<dim, T, std::min<std::size_t>(16, n_lanes_max)>(params, table);
  else
    AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void
run_dim(const Parameters &params, ConvergenceTable &table)
{
  unsigned int n_lanes = params.n_lanes;

  if (params.number_type == "double")
    {
      using T = double;

      constexpr std::size_t n_lanes_max = VectorizedArray<T>::size();

      if (n_lanes == 0)
        n_lanes = n_lanes_max;

      AssertThrow(n_lanes <= n_lanes_max, ExcNotImplemented());

      if (n_lanes == 1)
        run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params, table);
      else if (n_lanes == 2)
        run<dim, T, std::min<std::size_t>(2, n_lanes_max)>(params, table);
      else if (n_lanes == 4)
        run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params, table);
      else if (n_lanes == 8)
        run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());
    }
  else if (params.number_type == "float")
    {
      using T = float;

      constexpr std::size_t n_lanes_max = VectorizedArray<T>::size();

      if (n_lanes == 0)
        n_lanes = n_lanes_max;

      AssertThrow(n_lanes <= n_lanes_max, ExcNotImplemented());

      if (n_lanes == 1)
        run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params, table);
      else if (n_lanes == 4)
        run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params, table);
      else if (n_lanes == 8)
        run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params, table);
      else if (n_lanes == 16)
        run<dim, T, std::min<std::size_t>(16, n_lanes_max)>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());
    }
  else
    AssertThrow(false, ExcNotImplemented());
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  const bool verbose = true;

  std::vector<std::string> input_files;

  for (int i = 1; i < argc; ++i)
    input_files.emplace_back(std::string(argv[i]));

  if (input_files.empty())
    input_files.push_back("");

  ConvergenceTable table;

  for (unsigned int i = 0; i < input_files.size(); ++i)
    {
      const auto file_name = input_files[i];

      Parameters params;

      if (file_name != "")
        params.parse(file_name);

      params.print();

      if (params.dim == 2)
        run_dim<2>(params, table);
      else if (params.dim == 3)
        run_dim<3>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());

      if (is_root && (verbose || ((i + 1) == input_files.size())))
        {
          table.write_text(std::cout, ConvergenceTable::org_mode_table);
          std::cout << std::endl;
        }
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
