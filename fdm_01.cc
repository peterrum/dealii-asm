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
#include <deal.II/lac/tensor_product_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/grid_tools.h"

using namespace dealii;

template <int dim, typename Number>
TensorProductMatrixSymmetricSum<dim, Number, -1>
setup_fdm(const typename Triangulation<dim>::cell_iterator &cell,
          const FiniteElement<1> &                          fe,
          const Quadrature<1> &                             quadrature,
          const dealii::ndarray<double, dim, 3> &           cell_extend)
{
  // 1) create element mass and siffness matrix (without overlap)
  Triangulation<1> tria;
  GridGenerator::hyper_cube(tria);

  DoFHandler<1> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingQ1<1> mapping;

  const unsigned int n_dofs_1D_without_overlap = fe.n_dofs_per_cell();

  FullMatrix<Number> mass_matrix_reference(n_dofs_1D_without_overlap,
                                           n_dofs_1D_without_overlap);
  FullMatrix<Number> derivative_matrix_reference(n_dofs_1D_without_overlap,
                                                 n_dofs_1D_without_overlap);

  FEValues<1> fe_values(mapping,
                        fe,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);

  fe_values.reinit(tria.begin());

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    for (const unsigned int i : fe_values.dof_indices())
      for (const unsigned int j : fe_values.dof_indices())
        {
          mass_matrix_reference(i, j) +=
            (fe_values.shape_value(i, q_index) *
             fe_values.shape_value(j, q_index) * fe_values.JxW(q_index));

          derivative_matrix_reference(i, j) +=
            (fe_values.shape_grad(i, q_index) *
             fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));
        }

  const unsigned int n_overlap = 1;

  const unsigned int n_dofs_1D = fe.n_dofs_per_cell() - 2 + 2 * n_overlap;

  std::array<FullMatrix<Number>, dim> mass_matrices;
  std::array<FullMatrix<Number>, dim> derivative_matrices;
  std::array<std::vector<bool>, dim>  masks;

  const auto clear_row_and_column =
    [](const unsigned int n, FullMatrix<Number> &matrix, const bool set_one) {
      AssertDimension(matrix.m(), matrix.n());

      const unsigned int size = matrix.m();

      for (unsigned int i = 0; i < size; ++i)
        {
          matrix[i][n] = 0.0;
          matrix[n][i] = 0.0;
        }

      if (set_one)
        matrix[n][n] = 1.0;
    };

  // 2) loop over all dimensions and create mass and stiffness
  // matrix so that boundary conditions and overlap are considered
  for (unsigned int d = 0; d < dim; ++d)
    {
      FullMatrix<Number> mass_matrix(n_dofs_1D, n_dofs_1D);
      FullMatrix<Number> derivative_matrix(n_dofs_1D, n_dofs_1D);

      for (unsigned int i = 0; i < n_dofs_1D_without_overlap; ++i)
        for (unsigned int j = 0; j < n_dofs_1D_without_overlap; ++j)
          {
            mass_matrix[i][j] = mass_matrix_reference[i][j] * cell_extend[d][1];
            derivative_matrix[i][j] =
              derivative_matrix_reference[i][j] / cell_extend[d][1];
          }

      if (cell->at_boundary(2 * d) == false)
        {
          if (n_overlap > 1)
            {
              Assert(cell_extend[d][0] > 0.0, ExcInternalError());

              // TODO
              Assert(false, ExcNotImplemented());
            }
        }
      else if (cell->face(2 * d)->boundary_id() == 1 /*DBC*/)
        {
          clear_row_and_column(0 /*TODO*/, derivative_matrix, true);
        }


      if (cell->at_boundary(2 * d + 1) == false)
        {
          if (n_overlap > 1)
            {
              Assert(cell_extend[d][2] > 0.0, ExcInternalError());

              // TODO
              Assert(false, ExcNotImplemented());
            }
        }
      else if (cell->face(2 * d + 1)->boundary_id() == 1 /*DBC*/)
        {
          clear_row_and_column(n_dofs_1D - 1 /*TODO*/, derivative_matrix, true);
        }

      masks[d].assign(n_dofs_1D, true);
      for (unsigned int i = 0; i < n_dofs_1D; ++i)
        if (derivative_matrix[i][i] == 0.0)
          {
            masks[d][i] = false;
          }

      mass_matrix.print(std::cout);
      std::cout << std::endl;
      derivative_matrix.print(std::cout);
      std::cout << std::endl;
      std::cout << std::endl;

      mass_matrices[d]       = mass_matrix;
      derivative_matrices[d] = derivative_matrix;
    }

  TensorProductMatrixSymmetricSum<dim, Number, -1> fdm;
  fdm.reinit(mass_matrices, derivative_matrices);

  return fdm;
}

template <int dim>
void
test(const unsigned int fe_degree = 1)
{
  FE_Q<dim> fe(fe_degree);
  FE_Q<1>   fe_1D(fe_degree);

  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> quadrature_face(fe_degree + 1);
  QGauss<1>       quadrature_1D(fe_degree + 1);

  MappingQ1<dim> mapping;

  Triangulation<dim> tria;

  const double left  = 0.0;
  const double right = 1.0;

  std::vector<std::vector<double>> step_sizes(dim);

  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < 3; ++i)
      step_sizes[d].push_back((right - left) / 3);

  GridGenerator::subdivided_hyper_rectangle(
    tria, step_sizes, Point<dim>(left, left), Point<dim>(right, right), false);

  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
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

  const auto harmonic_patch_extend =
    GridTools::compute_harmonic_patch_extend(mapping, tria, quadrature_face);

  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto &patch_extend =
        harmonic_patch_extend[cell->active_cell_index()];

      for (unsigned int d = 0; d < dim; ++d)
        {
          for (unsigned int i = 0; i < 3; ++i)
            std::cout << patch_extend[d][i] << " ";
          std::cout << std::endl;
        }

      std::cout << std::endl;

      const auto fdm =
        setup_fdm<dim, double>(cell, fe_1D, quadrature_1D, patch_extend);
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = argc > 1 ? atoi(argv[1]) : 2;

  if (dim == 2)
    test<2>();
  else
    AssertThrow(false, ExcNotImplemented());
}