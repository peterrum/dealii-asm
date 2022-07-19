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
#include <deal.II/lac/tensor_product_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/grid_tools.h"
#include "include/restrictors.h"

using namespace dealii;

template <int dim, typename Number>
class MyTensorProductMatrixSymmetricSum
  : public TensorProductMatrixSymmetricSum<dim, Number, -1>
{
public:
  std::array<Table<2, Number>, dim> &
  get_eigenvectors()
  {
    return this->eigenvectors;
  }

private:
};

template <int dim, typename Number>
MyTensorProductMatrixSymmetricSum<dim, Number>
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

  const auto lexicographic_to_hierarchic_numbering =
    Utilities::invert_permutation(
      FETools::hierarchic_to_lexicographic_numbering<1>(fe.tensor_degree()));

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    for (const unsigned int i : fe_values.dof_indices())
      for (const unsigned int j : fe_values.dof_indices())
        {
          mass_matrix_reference(i, j) +=
            (fe_values.shape_value(lexicographic_to_hierarchic_numbering[i],
                                   q_index) *
             fe_values.shape_value(lexicographic_to_hierarchic_numbering[j],
                                   q_index) *
             fe_values.JxW(q_index));

          derivative_matrix_reference(i, j) +=
            (fe_values.shape_grad(lexicographic_to_hierarchic_numbering[i],
                                  q_index) *
             fe_values.shape_grad(lexicographic_to_hierarchic_numbering[j],
                                  q_index) *
             fe_values.JxW(q_index));
        }

  const unsigned int n_overlap = 1;

  const unsigned int n_dofs_1D = fe.n_dofs_per_cell() - 2 + 2 * n_overlap;

  std::array<FullMatrix<Number>, dim> mass_matrices;
  std::array<FullMatrix<Number>, dim> derivative_matrices;
  std::array<std::vector<bool>, dim>  masks;

  const auto clear_row_and_column = [&](const unsigned int n, auto &matrix) {
    for (unsigned int i = 0; i < n_dofs_1D; ++i)
      {
        matrix[i][n] = 0.0;
        matrix[n][i] = 0.0;
      }
  };

  // 2) loop over all dimensions and create mass and stiffness
  // matrix so that boundary conditions and overlap are considered
  for (unsigned int d = 0; d < dim; ++d)
    {
      FullMatrix<Number> mass_matrix(n_dofs_1D, n_dofs_1D);
      FullMatrix<Number> derivative_matrix(n_dofs_1D, n_dofs_1D);

      masks[d].assign(n_dofs_1D, true);

      // inner DoFs
      for (unsigned int i = 0; i < n_dofs_1D_without_overlap; ++i)
        for (unsigned int j = 0; j < n_dofs_1D_without_overlap; ++j)
          {
            mass_matrix[i][j] = mass_matrix_reference[i][j] * cell_extend[d][1];
            derivative_matrix[i][j] =
              derivative_matrix_reference[i][j] / cell_extend[d][1];
          }

      // left neighbor or left boundary
      if (cell->at_boundary(2 * d) == false)
        {
          // left neighbor
          Assert(cell_extend[d][0] > 0.0, ExcInternalError());

          mass_matrix[0][0] +=
            mass_matrix_reference[n_dofs_1D_without_overlap - 1]
                                 [n_dofs_1D_without_overlap - 1] *
            cell_extend[d][0];
          derivative_matrix[0][0] +=
            derivative_matrix_reference[n_dofs_1D_without_overlap - 1]
                                       [n_dofs_1D_without_overlap - 1] /
            cell_extend[d][0];
        }
      else if (cell->face(2 * d)->boundary_id() == 1 /*DBC*/)
        {
          // left DBC
          clear_row_and_column(0 /*TODO*/, mass_matrix);
          clear_row_and_column(0 /*TODO*/, derivative_matrix);
        }
      else
        {
          // left NBC -> nothing to do
        }

      // reight neighbor or right boundary
      if (cell->at_boundary(2 * d + 1) == false)
        {
          Assert(cell_extend[d][2] > 0.0, ExcInternalError());

          mass_matrix[n_dofs_1D - 1][n_dofs_1D - 1] +=
            mass_matrix_reference[0][0] * cell_extend[d][2];
          derivative_matrix[n_dofs_1D - 1][n_dofs_1D - 1] +=
            derivative_matrix_reference[0][0] / cell_extend[d][2];
        }
      else if (cell->face(2 * d + 1)->boundary_id() == 1 /*DBC*/)
        {
          // right DBC
          clear_row_and_column(n_dofs_1D - 1 /*TODO*/, mass_matrix);
          clear_row_and_column(n_dofs_1D - 1 /*TODO*/, derivative_matrix);
        }
      else
        {
          // right NBC -> nothing to do
        }

      for (unsigned int i = 0; i < n_dofs_1D; ++i)
        if (derivative_matrix[i][i] == 0.0)
          {
            mass_matrix[i][i]       = 1.0;
            derivative_matrix[i][i] = 1.0;
            masks[d][i]             = false;
          }

#if true
      mass_matrix.print_formatted(std::cout, 3, true, 10);
      std::cout << std::endl;
      derivative_matrix.print_formatted(std::cout, 3, true, 10);
      std::cout << std::endl;
      std::cout << std::endl;
#endif

      mass_matrices[d]       = mass_matrix;
      derivative_matrices[d] = derivative_matrix;
    }

  MyTensorProductMatrixSymmetricSum<dim, Number> fdm;
  fdm.reinit(mass_matrices, derivative_matrices);

  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int i = 0; i < n_dofs_1D; ++i)
      if (masks[d][i] == false)
        for (unsigned int j = 0; j < n_dofs_1D; ++j)
          fdm.get_eigenvectors()[d][i][j] = 0.0;

  return fdm;
}

template <int dim>
void
test(const unsigned int fe_degree)
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

  Restrictors::ElementCenteredRestrictor<Vector<double>> restrictor;
  restrictor.reinit(dof_handler);

  std::vector<FullMatrix<double>> blocks;
  dealii::SparseMatrixTools::restrict_to_full_matrices(laplace_matrix,
                                                       sparsity_pattern,
                                                       restrictor.get_indices(),
                                                       blocks);

  for (const auto &cell : tria.active_cell_iterators())
    {
      if (cell->active_cell_index() != 5)
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

      const auto fdm =
        setup_fdm<dim, double>(cell, fe_1D, quadrature_1D, patch_extend);

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

  if (dim == 2)
    test<2>(fe_degree);
  else
    AssertThrow(false, ExcNotImplemented());
}