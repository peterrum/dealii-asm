
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

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

//#include "../common_code/curved_manifold.h"
//#include "../common_code/diagonal_matrix_blocked.h"
//#include "../common_code/poisson_operator.h"

using namespace dealii;

struct Parameters
{
  unsigned int dim              = 3;
  unsigned int fe_degree        = 1;
  unsigned int n_components     = 1;
  unsigned int subdivisions     = 34;
  unsigned int n_lanes          = 0;
  unsigned int cell_granularity = 0;
  unsigned int n_repetitions    = 10;

  std::string number_type = "double";
};

template <int dim, typename Number>
class Fu : public Function<dim, Number>
{
public:
  virtual Number
  value(const Point<dim> &p, const unsigned int component) const
  {
    (void)component;

    return std::sin(p[0]);
  }
};

template <int dim>
void
create_grid(Triangulation<dim> &tria, const unsigned int s)
{
  unsigned int       n_refine  = s / 6;
  const unsigned int remainder = s % 6;

  std::vector<unsigned int> subdivisions(dim, 1);
  if (remainder == 1 && s > 1)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
      subdivisions[2] = 2;
      n_refine -= 1;
    }
  if (remainder == 2)
    subdivisions[0] = 2;
  else if (remainder == 3)
    subdivisions[0] = 3;
  else if (remainder == 4)
    subdivisions[0] = subdivisions[1] = 2;
  else if (remainder == 5)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
    }

  Point<dim> p2;
  for (unsigned int d = 0; d < dim; ++d)
    p2[d] = subdivisions[d];
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            subdivisions,
                                            Point<dim>(),
                                            p2);

  tria.refine_global(n_refine);
}

template <int dim, typename Number, std::size_t n_lanes>
void
run(const Parameters &params)
{
  const unsigned int fe_degree    = params.fe_degree;
  const unsigned int n_components = params.n_components;

  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number, n_lanes>;
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  Triangulation<dim> tria;
  create_grid(tria, params.subdivisions);

  MappingQ1<dim>   mapping;
  FE_Q<dim>        fe_scalar(fe_degree);
  FESystem<dim>    fe_q(fe_scalar, n_components);
  QGaussLobatto<1> quad(fe_degree + 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<Number> constraints;

  // setup matrixfree
  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData mf_data;
  mf_data.tasks_parallel_scheme = MatrixFree<dim, Number, VectorizedArrayType>::
    AdditionalData::TasksParallelScheme::none;

  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  matrix_free.reinit(mapping, dof_handler, constraints, quad, mf_data);

  // wrap matrix-free loop to be able to control granularity
  const auto matrix_free_cell_loop = [&](const auto fu) {
    if (params.cell_granularity == 0)
      {
        Number dummy = 0;
        matrix_free.template cell_loop<Number, Number>(fu, dummy, dummy);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  };

  std::vector<unsigned int> min_vector(matrix_free.n_cell_batches() *
                                         VectorizedArrayType::size(),
                                       numbers::invalid_unsigned_int);
  std::vector<unsigned int> max_vector(matrix_free.n_cell_batches() *
                                         VectorizedArrayType::size(),
                                       0);


  std::vector<std::pair<unsigned int, unsigned int>> vertex_tracker(
    tria.n_vertices(),
    std::pair<unsigned int, unsigned int>(numbers::invalid_unsigned_int, 0));

  unsigned int counter = 0;

  matrix_free_cell_loop([&](const auto &data,
                            auto &,
                            const auto &,
                            const auto cells) {
    (void)data;

    for (unsigned int cell = cells.first; cell < cells.second; ++cell)
      {
        const auto n_active_entries_per_cell_batch =
          data.n_active_entries_per_cell_batch(cell);

        for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

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

  std::vector<std::vector<unsigned int>> my_indices(counter);

  counter = 0;
  matrix_free_cell_loop(
    [&](const auto &data, auto &, const auto &, const auto cells) {
      (void)data;

      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          const auto n_active_entries_per_cell_batch =
            data.n_active_entries_per_cell_batch(cell);

          for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
            {
              my_indices[counter].push_back(cell * VectorizedArrayType::size() +
                                            v);

              const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

              for (const auto i : cell_iterator->vertex_indices())
                {
                  min_vector[cell * VectorizedArrayType::size() + v] = std::min(
                    min_vector[cell * VectorizedArrayType::size() + v],
                    vertex_tracker[cell_iterator->vertex_index(i)].first);

                  max_vector[cell * VectorizedArrayType::size() + v] = std::max(
                    max_vector[cell * VectorizedArrayType::size() + v],
                    vertex_tracker[cell_iterator->vertex_index(i)].second);
                }
            }
        }
      counter++;
    });

  const auto process = [counter](const auto &ids) {
    std::vector<std::vector<unsigned int>> temp(counter);

    for (unsigned int i = 0; i < ids.size(); ++i)
      if (ids[i] != numbers::invalid_unsigned_int)
        temp[ids[i]].push_back(i);

    return temp;
  };

  const auto pre_indices  = process(min_vector);
  const auto post_indices = process(max_vector);

  // intialize vectors
  VectorType src, dst_0, dst_1;
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_0);
  matrix_free.initialize_dof_vector(dst_1);
  VectorTools::interpolate(mapping, dof_handler, Fu<dim, Number>(), src);

  // define vmult operations and ...
  const auto process_batch_vmult =
    [](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);
      phi.distribute_local_to_global(dst);
    };

  // ... post operation
  const auto process_batch_post =
    [](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);
      phi.distribute_local_to_global(dst);
    };

  // helper function to run performance study
  const auto run = [&](const auto &fu, const std::string label) {
    dst_0 = 0.0;
    dst_1 = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);

    if (label != "")
      LIKWID_MARKER_START("power");

    auto temp_time = std::chrono::system_clock::now();

    for (unsigned int c = 0; c < params.n_repetitions; ++c)
      fu();

    MPI_Barrier(MPI_COMM_WORLD);

    if (label != "")
      LIKWID_MARKER_STOP("power");

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << src.l2_norm() << " " << dst_0.l2_norm() << " "
                << dst_1.l2_norm() << std::endl;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now() - temp_time)
             .count() /
           1e9;
  };

  // version 1: power kernel
  const auto time_power = run(
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
          for (unsigned int i = 0; i < post_indices[counter].size();
               i += VectorizedArrayType::size())
            {
              std::array<unsigned int, VectorizedArrayType::size()> ids = {};
              ids.fill(numbers::invalid_unsigned_int);

              for (unsigned int v = 0;
                   v < std::min(VectorizedArrayType::size(),
                                post_indices[counter].size() - i);
                   ++v)
                ids[v] = post_indices[counter][i + v];

              process_batch_post(ids, phi_1, dst_1, dst_0);
            }

          counter++;
        });
    },
    "power");

  // version 2: run sequentially
  const auto time_sequential = run(
    [&]() {
      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);

          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            process_batch_vmult(cell, phi, dst_0, src);
        });

      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);

          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            process_batch_post(cell, phi, dst_1, dst_0);
        });
    },
    "sequential");


  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << tria.n_active_cells() << " " << dof_handler.n_dofs() << " "
              << (time_power / time_sequential) << " " << time_power << " "
              << time_sequential << std::endl;
}

template <int dim, typename T>
void
run_number(const Parameters &params)
{
  unsigned int n_lanes = params.n_lanes;

  constexpr std::size_t n_lanes_max = VectorizedArray<T>::size();

  if (n_lanes == 0)
    n_lanes = n_lanes_max;

  AssertThrow(n_lanes <= n_lanes_max, ExcNotImplemented());

  if (n_lanes == 1)
    run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params);
  else if (n_lanes == 2)
    run<dim, T, std::min<std::size_t>(2, n_lanes_max)>(params);
  else if (n_lanes == 4)
    run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params);
  else if (n_lanes == 8)
    run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params);
  else if (n_lanes == 16)
    run<dim, T, std::min<std::size_t>(16, n_lanes_max)>(params);
  else
    AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void
run_dim(const Parameters &params)
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
        run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params);
      else if (n_lanes == 2)
        run<dim, T, std::min<std::size_t>(2, n_lanes_max)>(params);
      else if (n_lanes == 4)
        run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params);
      else if (n_lanes == 8)
        run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params);
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
        run<dim, T, std::min<std::size_t>(1, n_lanes_max)>(params);
      else if (n_lanes == 4)
        run<dim, T, std::min<std::size_t>(4, n_lanes_max)>(params);
      else if (n_lanes == 8)
        run<dim, T, std::min<std::size_t>(8, n_lanes_max)>(params);
      else if (n_lanes == 16)
        run<dim, T, std::min<std::size_t>(16, n_lanes_max)>(params);
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

  Parameters params;

  if (params.dim == 2)
    run_dim<2>(params);
  else if (params.dim == 3)
    run_dim<3>(params);
  else
    AssertThrow(false, ExcNotImplemented());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
