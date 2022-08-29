
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

template <int dim,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
run(const unsigned int s,
    const unsigned int fe_degree,
    const bool         deformed_mesh,
    const unsigned int n_components = 1)
{
  AssertThrow(deformed_mesh == false, ExcNotImplemented());

  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

  unsigned int       n_refine  = s / 6;
  const unsigned int remainder = s % 6;

  // MyManifold<dim>           manifold;
  Triangulation<dim>        tria;
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

  // if (deformed_mesh)
  //  {
  //    GridTools::transform(
  //      std::bind(&MyManifold<dim>::push_forward, manifold,
  //      std::placeholders::_1), tria);
  //    tria.set_all_manifold_ids(1);
  //    tria.set_manifold(1, manifold);
  //  }

  tria.refine_global(n_refine);

  MappingQGeneric<dim> mapping(2);

  FE_Q<dim>       fe_scalar(fe_degree);
  FESystem<dim>   fe_q(fe_scalar, n_components);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<Number> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim, Number>(
                                             n_components),
                                           constraints);
  constraints.close();
  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData mf_data;

  mf_data.mapping_update_flags  = update_gradients;
  mf_data.tasks_parallel_scheme = MatrixFree<dim, Number, VectorizedArrayType>::
    AdditionalData::TasksParallelScheme::none;

  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim, Number>(
                                             n_components),
                                           constraints);
  constraints.close();

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  matrix_free.reinit(mapping,
                     dof_handler,
                     constraints,
                     QGaussLobatto<1>(fe_degree + 1),
                     mf_data);


  std::vector<unsigned int> min_vector(matrix_free.n_cell_batches() *
                                         VectorizedArrayType::size(),
                                       numbers::invalid_unsigned_int);
  std::vector<unsigned int> max_vector(matrix_free.n_cell_batches() *
                                         VectorizedArrayType::size(),
                                       0);


  std::vector<std::pair<unsigned int, unsigned int>> vertex_tracker(
    tria.n_vertices(),
    std::pair<unsigned int, unsigned int>(numbers::invalid_unsigned_int, 0));

  Number       dummy   = 0;
  unsigned int counter = 0;

  matrix_free.template loop_cell_centric<Number, Number>(
    [&](const auto &data, auto &, const auto &, const auto cells) {
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
                    std::min(
                      vertex_tracker[cell_iterator->vertex_index(i)].first,
                      counter);
                  vertex_tracker[cell_iterator->vertex_index(i)].second =
                    std::max(
                      vertex_tracker[cell_iterator->vertex_index(i)].second,
                      counter);
                }
            }
        }
      counter++;
    },
    dummy,
    dummy);

  std::vector<std::vector<unsigned int>> my_indices(counter);

  counter = 0;
  matrix_free.template loop_cell_centric<Number, Number>(
    [&](const auto &data, auto &, const auto &, const auto cells) {
      (void)data;

      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          const auto n_active_entries_per_cell_batch =
            data.n_active_entries_per_cell_batch(cell);

          // std::cout << n_active_entries_per_cell_batch << std::endl;

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

      // std::cout << cells.first << " " << cells.second << std::endl;
    },
    dummy,
    dummy);

  const auto process = [counter](const auto &ids) {
    std::vector<std::vector<unsigned int>> temp(counter);

    for (unsigned int i = 0; i < ids.size(); ++i)
      if (ids[i] != numbers::invalid_unsigned_int)
        temp[ids[i]].push_back(i);

    return temp;
  };

  const auto pre_indices  = process(min_vector);
  const auto post_indices = process(max_vector);

  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  VectorType src, dst_0, dst_1;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_0);
  matrix_free.initialize_dof_vector(dst_1);

  VectorTools::interpolate(mapping, dof_handler, Fu<dim, Number>(), src);

  const auto process_batch_vmult =
    [](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);

      if (false)
        {
          phi.evaluate(EvaluationFlags::gradients);

          for (const auto q : phi.quadrature_point_indices())
            phi.submit_gradient(phi.get_gradient(q), q);

          phi.integrate(EvaluationFlags::gradients);
        }

      phi.distribute_local_to_global(dst);
    };

  const auto process_batch_post =
    [](const auto &id, auto &phi, auto &dst, const auto &src) {
      phi.reinit(id);
      phi.read_dof_values(src);
      if (false)
        {
          phi.evaluate(EvaluationFlags::values);

          for (const auto q : phi.quadrature_point_indices())
            phi.submit_value(phi.get_value(q), q);

          phi.integrate(EvaluationFlags::values);
        }
      phi.distribute_local_to_global(dst);
    };


  const auto matrix_free_cell_loop = [&](const auto fu) {
    Number dummy = 0;
    matrix_free.template cell_loop<Number, Number>(fu, dummy, dummy);
  };

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_START("power");

  auto temp_time = std::chrono::system_clock::now();

  const unsigned int n_repetitions = 10;

  for (unsigned int c = 0; c < n_repetitions; ++c)
    {
      counter = 0;
      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);
          FECellIntegrator phi_(data);

          // vmult
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            {
              process_batch_vmult(cell, phi_, dst_0, src);
            }

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

              process_batch_post(ids, phi, dst_1, dst_0);
            }

          counter++;
        });
    }

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_STOP("power");

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << src.l2_norm() << " " << dst_0.l2_norm() << " "
              << dst_1.l2_norm() << std::endl;


  if (false && (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    {
      for (unsigned int i = 0; i < counter; ++i)
        {
          std::set<unsigned int> indices_my;
          std::set<unsigned int> indices_pre;
          std::set<unsigned int> indices_post;
          std::set<unsigned int> indices_all;
          std::set<unsigned int> indices_2;
          std::set<unsigned int> indices_3;

          for (auto j : my_indices[i])
            {
              indices_my.insert(j);
              indices_all.insert(j);
            }


          for (auto j : pre_indices[i])
            {
              indices_pre.insert(j);
              indices_all.insert(j);
            }

          for (auto j : post_indices[i])
            {
              indices_post.insert(j);
              indices_all.insert(j);
            }

          for (auto j : indices_all)
            {
              if ((indices_pre.count(j) + indices_post.count(j) +
                   indices_my.count(j)) > 1)
                indices_2.insert(j);
              if ((indices_pre.count(j) + indices_post.count(j) +
                   indices_my.count(j)) == 3)
                indices_3.insert(j);
            }


          std::cout
            << pre_indices[i].size()                             // pre
            << " "                                               //
            << my_indices[i].size()                              // current
            << " "                                               //
            << post_indices[i].size()                            // post
            << " "                                               //
            << indices_all.size()                                // all
            << " "                                               //
            << (indices_all.size() * 100 / my_indices[i].size()) // increase [%]
            << " "                                               //
            << (indices_2.size() * 100 / indices_all.size()) // reusage: 2x [%]
            << " "                                           //
            << (indices_3.size() * 100 / indices_all.size()) // reusage: 3x [%]
            << " " << std::endl;
        }


      std::cout << std::endl;
    }

  const double time_power =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now() - temp_time)
      .count() /
    1e9;


  dst_0 = 0.0;
  dst_1 = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_START("normal");

  temp_time = std::chrono::system_clock::now();
  for (unsigned int c = 0; c < n_repetitions; ++c)
    for (unsigned int c = 0; c < 2; ++c)
      matrix_free_cell_loop(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FECellIntegrator phi(data);

          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            if (c == 0)
              process_batch_vmult(cell, phi, dst_0, src);
            else if (c == 1)
              process_batch_post(cell, phi, dst_1, dst_0);
        });

  MPI_Barrier(MPI_COMM_WORLD);
  LIKWID_MARKER_STOP("normal");

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << src.l2_norm() << " " << dst_0.l2_norm() << " "
              << dst_1.l2_norm() << std::endl;

  const double time_normal =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now() - temp_time)
      .count() /
    1e9;


  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << tria.n_active_cells() << " " << dof_handler.n_dofs() << " "
              << (time_power / time_normal) << " " << time_power << " "
              << time_normal << std::endl;
}

template <int dim>
void
run_dim(const unsigned int s,
        const unsigned int fe_degree,
        const bool         deformed_mesh,
        const unsigned int n_components = 1)
{
  using T = double;
  run<dim, T, VectorizedArray<T>>(s, fe_degree, deformed_mesh, n_components);
}


int
main(int argc, char **argv)
{
  // mpirun -np 40 ./benchmark_matrix_power_kernel/bench 3 5 34 0
  // mpirun -np 40 ./benchmark_matrix_power_kernel/bench 3 5 31 1
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 3, ExcNotImplemented());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const unsigned int dim           = std::atoi(argv[1]);
  const unsigned int degree        = std::atoi(argv[2]);
  const unsigned int n_steps       = std::atoi(argv[3]);
  const unsigned int deformed_mesh = std::atoi(argv[4]);

  if (dim == 2)
    run_dim<2>(n_steps, degree, deformed_mesh);
  else if (dim == 3)
    run_dim<3>(n_steps, degree, deformed_mesh);
  else
    AssertThrow(false, ExcNotImplemented());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
