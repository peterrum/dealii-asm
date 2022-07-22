#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"

using namespace dealii;


template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<double>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  QGauss<dim> quadrature(fe_degree + 1);

  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

  // set up ConstraintInfo
  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
    constraint_info;

  // ... allocate memory
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
                  n_overlap <= 1 ? 0 : dim);

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
          const auto cells =
            dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
              matrix_free.get_cell_iterator(cell, v), n_overlap <= 1 ? 0 : dim);

          constraint_info.read_dof_indices(
            cell_counter,
            dealii::DoFTools::get_dof_indices_cell_with_overlap(
              dof_handler, cells, n_overlap, true),
            partitioner_for_fdm);
        }

      cell_ptr.push_back(cell_ptr.back() +
                         matrix_free.n_active_entries_per_cell_batch(cell));
    }

  constraint_info.finalize();

  VectorType src, dst, src_, dst_;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);
  src_.reinit(partitioner_for_fdm);
  dst_.reinit(partitioner_for_fdm);

  const auto vmult = [&](auto &dst, const auto &src) {
    AlignedVector<VectorizedArrayType> scratch_data(
      Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

    // update ghost values
    src_.copy_locally_owned_data_from(src);
    src_.update_ghost_values();

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        // 1) gather
        internal::VectorReader<Number, VectorizedArrayType> reader;
        constraint_info.read_write_operation(reader,
                                             src_,
                                             scratch_data,
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             scratch_data.size(),
                                             true);

        // 2) cell operation
        // TODO: fast diagonalization method

        // 3) scatter
        internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
          writer;
        constraint_info.read_write_operation(writer,
                                             dst_,
                                             scratch_data,
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             scratch_data.size(),
                                             true);
      }

    // compress
    dst_.compress(VectorOperation::add);
    dst.copy_locally_owned_data_from(dst_);
  };

  vmult(dst, src);
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int n_overlap = (argc >= 5) ? std::atoi(argv[4]) : 1;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements, n_overlap);
  else
    AssertThrow(false, ExcNotImplemented());
}