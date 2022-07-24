#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

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
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/tensor_product_matrix.h"

using namespace dealii;

template <int dim, typename Number, typename VectorizedArrayType>
class ASPoissonPreconditioner
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  ASPoissonPreconditioner(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const unsigned int                                  n_overlap,
    const Mapping<dim> &                                mapping,
    const FiniteElement<1> &                            fe_1D,
    const QGauss<dim - 1> &                             quadrature_face,
    const Quadrature<1> &                               quadrature_1D)
    : matrix_free(matrix_free)
    , fe_degree(matrix_free.get_dof_handler().get_fe().tensor_degree())
    , n_overlap(n_overlap)
  {
    const auto &dof_handler = matrix_free.get_dof_handler();

    // set up ConstraintInfo
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
                  dealii::GridTools::extract_all_surrounding_cells_cartesian<
                    dim>(matrix_free.get_cell_iterator(cell, v),
                         n_overlap <= 1 ? 0 : dim);

                const auto local_dofs =
                  dealii::DoFTools::get_dof_indices_cell_with_overlap(
                    dof_handler, cells, n_overlap, true);

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
        is_ghost_indices.add_indices(ghost_indices.begin(),
                                     ghost_indices.end());

        partitioner_for_fdm = std::make_shared<Utilities::MPI::Partitioner>(
          locally_owned_dofs, is_ghost_indices, dof_handler.get_communicator());
      }

    fdm.resize(matrix_free.n_cell_batches());

    const auto harmonic_patch_extend =
      GridTools::compute_harmonic_patch_extend(mapping,
                                               dof_handler.get_triangulation(),
                                               quadrature_face);

    // ... collect DoF indices
    cell_ptr = {0};
    for (unsigned int cell = 0, cell_counter = 0;
         cell < matrix_free.n_cell_batches();
         ++cell)
      {
        std::array<MyTensorProductMatrixSymmetricSum<dim, Number>,
                   VectorizedArrayType::size()>
          scalar_fdm;

        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v, ++cell_counter)
          {
            const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

            const auto cells =
              dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                cell_iterator, n_overlap <= 1 ? 0 : dim);

            constraint_info.read_dof_indices(
              cell_counter,
              dealii::DoFTools::get_dof_indices_cell_with_overlap(
                dof_handler, cells, n_overlap, true),
              partitioner_for_fdm);

            scalar_fdm[v] = setup_fdm<dim, Number>(
              cell_iterator,
              fe_1D,
              quadrature_1D,
              harmonic_patch_extend[cell_iterator->active_cell_index()],
              n_overlap);
          }

        cell_ptr.push_back(cell_ptr.back() +
                           matrix_free.n_active_entries_per_cell_batch(cell));

        fdm[cell] =
          MyTensorProductMatrixSymmetricSum<dim, Number>::template transpose<
            VectorizedArrayType::size()>(
            scalar_fdm, matrix_free.n_active_entries_per_cell_batch(cell));
      }

    constraint_info.finalize();

    src_.reinit(partitioner_for_fdm);
    dst_.reinit(partitioner_for_fdm);

    {
      AlignedVector<VectorizedArrayType> dst__(
        Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

      dst_ = 0.0;

      for (auto &i : dst__)
        i = 1.0;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
            writer;
          constraint_info.read_write_operation(writer,
                                               dst_,
                                               dst__,
                                               cell_ptr[cell],
                                               cell_ptr[cell + 1] -
                                                 cell_ptr[cell],
                                               dst__.size(),
                                               true);
        }

      dst_.compress(VectorOperation::add);

      matrix_free.initialize_dof_vector(weights);

      weights.copy_locally_owned_data_from(dst_);

      for (auto &i : weights)
        i = (i == 0.0) ? 1.0 : (1.0 / i);
    }
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    AlignedVector<VectorizedArrayType> src__(
      Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));
    AlignedVector<VectorizedArrayType> dst__(
      Utilities::pow(fe_degree + 2 * n_overlap - 1, dim));

    // update ghost values
    src_.copy_locally_owned_data_from(src);
    src_.update_ghost_values();

    dst_ = 0.0;

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        // 1) gather
        internal::VectorReader<Number, VectorizedArrayType> reader;
        constraint_info.read_write_operation(reader,
                                             src_,
                                             src__,
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             src__.size(),
                                             true);

        // 2) cell operation: fast diagonalization method
        fdm[cell].apply_inverse(make_array_view(dst__.begin(), dst__.end()),
                                make_array_view(src__.begin(), src__.end()));

        // 3) scatter
        internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
          writer;
        constraint_info.read_write_operation(writer,
                                             dst_,
                                             dst__,
                                             cell_ptr[cell],
                                             cell_ptr[cell + 1] -
                                               cell_ptr[cell],
                                             dst__.size(),
                                             true);
      }

    // compress
    dst_.compress(VectorOperation::add);
    dst.copy_locally_owned_data_from(dst_);
    dst.scale(weights);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  fe_degree;
  const unsigned int                                  n_overlap;

  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
                            constraint_info;
  std::vector<unsigned int> cell_ptr;

  std::vector<MyTensorProductMatrixSymmetricSum<dim, VectorizedArrayType>> fdm;

  mutable VectorType src_;
  mutable VectorType dst_;

  VectorType weights;
};



template <int dim, typename Number, typename VectorizedArrayType>
class PoissonOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  PoissonOperator(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
  {}

  void
  initialize_dof_vector(VectorType &vec)
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  rhs(VectorType &vec) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&](const auto &, auto &dst, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(
          matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.dofs_per_cell; ++q)
              phi.submit_value(1.0, q);

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      vec,
      dummy,
      true);
  }


  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(
          matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.dofs_per_cell; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      },
      dst,
      src,
      true);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
};



template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_overlap)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<double>;

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

  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
  constraints.close();

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quadrature);

  PoissonOperator<dim, Number, VectorizedArrayType> op(matrix_free);

  VectorType b, x;

  op.initialize_dof_vector(b);
  op.initialize_dof_vector(x);

  op.rhs(b);

  ASPoissonPreconditioner<dim, Number, VectorizedArrayType> precon(
    matrix_free, n_overlap, mapping, fe_1D, quadrature_face, quadrature_1D);

  ReductionControl reduction_control(100);

  SolverGMRES<VectorType>::AdditionalData additional_data;
  additional_data.right_preconditioning = true;

  SolverGMRES<VectorType> solver(reduction_control, additional_data);

  if (true)
    solver.solve(op, x, b, precon);
  else
    solver.solve(op, x, b, PreconditionIdentity());

  pcout << reduction_control.last_step() << std::endl;
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