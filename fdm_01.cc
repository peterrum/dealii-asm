#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/grid_tools.h"
#include "include/restrictors.h"
#include "include/tensor_product_matrix.h"

#define QUADRATURE_TYP QGauss

using namespace dealii;

template <int dim>
void
test(const unsigned int fe_degree, const unsigned int n_overlap)
{
  FE_Q<dim> fe(fe_degree);
  FE_Q<1>   fe_1D(fe_degree);

  QUADRATURE_TYP<dim>     quadrature(fe_degree + 1);
  QUADRATURE_TYP<dim - 1> quadrature_face(fe_degree + 1);
  QUADRATURE_TYP<1>       quadrature_1D(fe_degree + 1);

  MappingQ1<dim> mapping;

  Triangulation<dim> tria;

  std::vector<std::vector<double>> step_sizes(dim);

#if false
  const double right = 1.0;

  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < 3; ++i)
      step_sizes[d].push_back(right / 3.0);
#else
  for (unsigned int d = 0, c = 1; d < dim; ++d)
    for (unsigned int i = 0; i < 3; ++i, ++c)
      step_sizes[d].push_back(c * 1.1);
#endif

  Point<dim> point;

  for (unsigned int d = 0; d < dim; ++d)
    for (const auto &value : step_sizes[d])
      point[d] += value;

  GridGenerator::subdivided_hyper_rectangle(
    tria, step_sizes, Point<dim>(), point, false);

  DataOut<dim> data_out;
  data_out.attach_triangulation(tria);
  data_out.build_patches();
  data_out.write_vtu_in_parallel("mesh.vtu", tria.get_communicator());


  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints);
  constraints.close();

  // create system matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern;
  sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                          dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix laplace_matrix;
  laplace_matrix.reinit(sparsity_pattern);

  MatrixCreator::
    create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
      mapping, dof_handler, quadrature, laplace_matrix, nullptr, constraints);

  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
    if (constraints.is_constrained(i))
      laplace_matrix.set(i, i, 1.0);

  const auto harmonic_patch_extend =
    GridTools::compute_harmonic_patch_extend(mapping, tria, quadrature_face);


  Restrictors::ElementCenteredRestrictor<Vector<double>>::AdditionalData ad;
  ad.n_overlap = n_overlap;

  Restrictors::ElementCenteredRestrictor<Vector<double>> restrictor;
  restrictor.reinit(dof_handler, ad);

  std::vector<FullMatrix<double>> blocks;
  dealii::SparseMatrixTools::restrict_to_full_matrices(laplace_matrix,
                                                       sparsity_pattern,
                                                       restrictor.get_indices(),
                                                       blocks);

  for (const auto &cell : tria.active_cell_iterators())
    {
      if (cell->active_cell_index() != 2)
        continue;

      const auto &patch_extend =
        harmonic_patch_extend[cell->active_cell_index()];

#if false
      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int i = 0; i < 3; ++i)
            std::cout << patch_extend[d][i] << " ";
          std::cout << std::endl;
        }
      std::cout << std::endl;
#endif

      const auto fdm = setup_fdm<dim, double>(
        cell, fe_1D, quadrature_1D, patch_extend, n_overlap);

      Vector<double> src, dst;

      FullMatrix<double> matrix(fdm.m(), fdm.n());

      for (unsigned int i = 0; i < fdm.m(); ++i)
        {
          src.reinit(fdm.m());
          dst.reinit(fdm.m());

          for (unsigned int j = 0; j < fdm.m(); ++j)
            src[j] = i == j;

          fdm.apply_inverse(make_array_view(dst), make_array_view(src));

          for (unsigned int j = 0; j < fdm.m(); ++j)
            matrix[j][i] = dst[j];
        }

      matrix.print_formatted(std::cout, 3, true, 10);
      std::cout << std::endl;

      blocks[cell->active_cell_index()].gauss_jordan();
      blocks[cell->active_cell_index()].print_formatted(std::cout, 3, true, 10);
      std::cout << std::endl << std::endl << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = argc > 1 ? atoi(argv[1]) : 2;
  const unsigned int fe_degree = argc > 2 ? atoi(argv[2]) : 2;
  const unsigned int n_overlap = argc > 3 ? atoi(argv[3]) : 1;

  if (dim == 2)
    test<2>(fe_degree, n_overlap);
  else
    AssertThrow(false, ExcNotImplemented());
}