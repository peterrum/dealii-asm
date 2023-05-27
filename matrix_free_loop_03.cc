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

template <typename OperatorType>
class MyOperator : public Subscriptor
{
public:
  using value_type = typename OperatorType::value_type;
  using VectorType = typename OperatorType::VectorType;

  MyOperator(const OperatorType &op)
    : op(op)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

  types::global_dof_index
  m() const
  {
    return op.m();
  }


  value_type
  el(unsigned int i, unsigned int j) const
  {
    return op.el(i, j);
  }

private:
  const OperatorType &op;
};

template <typename VectorType>
class MyDiagonalMatrix : public Subscriptor
{
public:
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op.vmult(dst, src);
  }

  VectorType &
  get_vector()
  {
    return op.get_vector();
  }

private:
  DiagonalMatrix<VectorType> op;
};

template <typename OperatorType>
class Adapter : public Subscriptor
{
public:
  Adapter(std::shared_ptr<OperatorType> op)
    : op(op)
  {}

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    op->vmult(dst, src);
  }

private:
  std::shared_ptr<OperatorType> op;
};

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap,
     const unsigned int chebyshev_degree,
     const bool         do_vmult,
     const bool         use_cartesian_mesh,
     const bool         use_renumbering,
     ConvergenceTable & table)
{
  (void)chebyshev_degree;
  (void)do_vmult;
  (void)use_renumbering;

  const unsigned int sub_mesh_approximation = dim; // TODO

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
  GridGenerator::hyper_cube(tria);

  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

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
  DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
  constraints.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping_q_cache, dof_handler, constraints, quadrature, additional_data);

  VectorType src, dst;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);

  src = 1.0;
  dst = 0.0;

  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
    constraint_info;
  constraint_info.reinit(matrix_free.n_physical_cells());

  auto partitioner_for_fdm = matrix_free.get_vector_partitioner();

  if (n_overlap > 1)
    {
      const auto &locally_owned_dofs = dof_handler.locally_owned_dofs();

      std::vector<types::global_dof_index> ghost_indices;

      for (unsigned int cell = 0, cell_counter = 0;
           cell < matrix_free.n_cell_batches();
           ++cell)
        {
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(cell);
               ++v, ++cell_counter)
            {
              const auto cells =
                dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                  matrix_free.get_cell_iterator(cell, v),
                  n_overlap <= 1 ? 0 : sub_mesh_approximation);

              const auto local_dofs =
                dealii::DoFTools::get_dof_indices_cell_with_overlap(dof_handler,
                                                                    cells,
                                                                    n_overlap,
                                                                    true);

              for (const auto i : local_dofs)
                if ((locally_owned_dofs.is_element(i) == false) &&
                    (i != numbers::invalid_unsigned_int))
                  ghost_indices.push_back(i);
            }
        }

      std::sort(ghost_indices.begin(), ghost_indices.end());
      ghost_indices.erase(std::unique(ghost_indices.begin(),
                                      ghost_indices.end()),
                          ghost_indices.end());

      IndexSet is_ghost_indices(locally_owned_dofs.size());
      is_ghost_indices.add_indices(ghost_indices.begin(), ghost_indices.end());

      partitioner_for_fdm = std::make_shared<Utilities::MPI::Partitioner>(
        locally_owned_dofs, is_ghost_indices, dof_handler.get_communicator());
    }

  const auto harmonic_patch_extend =
    GridTools::compute_harmonic_patch_extend(mapping,
                                             dof_handler.get_triangulation(),
                                             quadrature_face);

  // ... collect DoF indices
  std::vector<unsigned int> cell_ptr = {0};
  for (unsigned int cell = 0, cell_counter = 0;
       cell < matrix_free.n_cell_batches();
       ++cell)
    {
      for (unsigned int v = 0;
           v < matrix_free.n_active_entries_per_cell_batch(cell);
           ++v, ++cell_counter)
        {
          const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

          const auto cells =
            dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
              cell_iterator, 0);

          constraint_info.read_dof_indices(
            cell_counter,
            dealii::DoFTools::get_dof_indices_cell_with_overlap(
              dof_handler, cells, n_overlap, false),
            partitioner_for_fdm);
        }

      cell_ptr.push_back(cell_ptr.back() +
                         matrix_free.n_active_entries_per_cell_batch(cell));
    }

  constraint_info.finalize();

  VectorType src_, dst_;
  src_.reinit(partitioner_for_fdm);
  dst_.reinit(partitioner_for_fdm);

  const auto run = [](const auto &runnable) {
    runnable();

    double     time_total = 0.0;
    const auto timer      = std::chrono::system_clock::now();

    for (unsigned int i = 0; i < 100; ++i)
      runnable();

    time_total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now() - timer)
                    .count() /
                  1e9;

    return time_total;
  };

  const auto mf_normal = run([&]() {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [](
        const auto &matrix_free, auto &dst, const auto &src, const auto range) {
        FECellIntegrator integrator(matrix_free);

        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            integrator.reinit(cell);
            integrator.read_dof_values(src);

            integrator.evaluate(EvaluationFlags::gradients);

            for (unsigned int q = 0; q < integrator.n_q_points; ++q)
              integrator.submit_gradient(integrator.get_gradient(q), q);

            integrator.integrate(EvaluationFlags::gradients);

            integrator.distribute_local_to_global(dst);
          }
      },
      dst,
      src);
  });

  const auto mf_own_communication = run([&]() {
    FECellIntegrator integrator(matrix_free);

    src.update_ghost_values();

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values(src);

        integrator.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate(EvaluationFlags::gradients);

        integrator.distribute_local_to_global(dst);
      }

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
  });

  const auto mf_own_gather = run([&]() {
    FECellIntegrator integrator(matrix_free);

    src_.update_ghost_values();

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        integrator.reinit(cell);

        internal::VectorReader<Number, VectorizedArrayType> reader;
        constraint_info.read_write_operation(reader,
                                             src_,
                                             integrator.begin_dof_values(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             integrator.dofs_per_cell,
                                             true);

        integrator.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate(EvaluationFlags::gradients);

        internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
          writer;
        constraint_info.read_write_operation(writer,
                                             dst_,
                                             integrator.begin_dof_values(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             integrator.dofs_per_cell,
                                             true);
      }

    dst_.compress(VectorOperation::add);
    src_.zero_out_ghost_values();
  });

  const auto mf_own_gather_and_copy = run([&]() {
    FECellIntegrator integrator(matrix_free);

    src_.copy_locally_owned_data_from(src);
    src_.update_ghost_values();

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        integrator.reinit(cell);

        internal::VectorReader<Number, VectorizedArrayType> reader;
        constraint_info.read_write_operation(reader,
                                             src_,
                                             integrator.begin_dof_values(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             integrator.dofs_per_cell,
                                             true);

        integrator.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < integrator.n_q_points; ++q)
          integrator.submit_gradient(integrator.get_gradient(q), q);

        integrator.integrate(EvaluationFlags::gradients);

        internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
          writer;
        constraint_info.read_write_operation(writer,
                                             dst_,
                                             integrator.begin_dof_values(),
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             integrator.dofs_per_cell,
                                             true);
      }

    dst_.compress(VectorOperation::add);
    dst.copy_locally_owned_data_from(dst_);
  });



  table.add_value("mf_normal", mf_normal);
  table.add_value("mf_own_communication", mf_own_communication);
  table.add_value("mf_own_gather", mf_own_gather);
  table.add_value("mf_own_gather_and_copy", mf_own_gather_and_copy);
}


/**
 * mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 1 1
 * mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 1 0
 * mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 0 1
 * mpirun -np 40 ./matrix_free_loop_02 3 4 6 1 0 0
 */
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int n_overlap          = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const bool         do_vmult           = (argc >= 6) ? std::atoi(argv[5]) : 1;
  const bool         use_cartesian_mesh = (argc >= 7) ? std::atoi(argv[6]) : 1;
  const bool         use_renumbering    = (argc >= 8) ? std::atoi(argv[7]) : 1;
  const bool         verbose            = true;

  (void)n_overlap;

  const bool is_root = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;

  ConvergenceTable table;

  for (unsigned int n_overlap = 1; n_overlap <= 4; ++n_overlap)
    {
      if (dim == 2)
        test<2>(fe_degree,
                n_global_refinements,
                n_overlap,
                1,
                do_vmult,
                use_cartesian_mesh,
                use_renumbering,
                table);
      else if (dim == 3)
        test<3>(fe_degree,
                n_global_refinements,
                n_overlap,
                1,
                do_vmult,
                use_cartesian_mesh,
                use_renumbering,
                table);
      else
        AssertThrow(false, ExcNotImplemented());

      if (is_root && verbose)
        {
          table.write_text(std::cout, ConvergenceTable::org_mode_table);
          std::cout << std::endl;
        }
    }

  if (is_root)
    {
      table.write_text(std::cout, ConvergenceTable::org_mode_table);
      std::cout << std::endl;
    }
}
