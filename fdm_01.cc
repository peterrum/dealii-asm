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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include "include/grid_tools.h"
#include "include/restrictors.h"

#define QUADRATURE_TYP QGauss

using namespace dealii;

template <int dim, typename Number>
class MyTensorProductMatrixSymmetricSum
  : public TensorProductMatrixSymmetricSum<dim, Number, -1>
{
public:
  void
  set_mask(const std::array<std::vector<bool>, dim> masks)
  {
    this->masks = masks;

    const unsigned int n_dofs_1D = this->eigenvalues[0].size();

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_dofs_1D; ++i)
        if (masks[d][i] == false)
          for (unsigned int j = 0; j < n_dofs_1D; ++j)
            this->eigenvectors[d][i][j] = 0.0;
  }

  void
  apply_inverse(const ArrayView<Number> &      dst,
                const ArrayView<const Number> &src) const
  {
    TensorProductMatrixSymmetricSum<dim, Number, -1>::apply_inverse(dst, src);

    const unsigned int n = this->eigenvalues[0].size();

    if (dim == 2)
      {
        for (unsigned int i1 = 0, c = 0; i1 < n; ++i1)
          for (unsigned int i0 = 0; i0 < n; ++i0, ++c)
            if ((masks[1][i1] == false) || (masks[0][i0] == false))
              dst[c] = src[c];
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }

private:
  std::array<std::vector<bool>, dim> masks;
};


template <typename Number>
std::tuple<FullMatrix<Number>, FullMatrix<Number>, bool>
create_referece_cell_matrices(const FiniteElement<1> &fe,
                              const Quadrature<1> &   quadrature)
{
  Triangulation<1> tria;
  GridGenerator::hyper_cube(tria);

  DoFHandler<1> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingQ1<1> mapping;

  const unsigned int n_dofs_1D = fe.n_dofs_per_cell();

  FullMatrix<Number> mass_matrix_reference(n_dofs_1D, n_dofs_1D);
  FullMatrix<Number> derivative_matrix_reference(n_dofs_1D, n_dofs_1D);

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

  return {mass_matrix_reference, derivative_matrix_reference, false};
}


template <int dim, typename Number>
MyTensorProductMatrixSymmetricSum<dim, Number>
setup_fdm(const typename Triangulation<dim>::cell_iterator &cell,
          const FiniteElement<1> &                          fe,
          const Quadrature<1> &                             quadrature,
          const dealii::ndarray<double, dim, 3> &           cell_extend,
          const unsigned int                                n_overlap)
{
  // 1) create element mass and siffness matrix (without overlap)
  const auto [M_ref, K_ref, is_dg] =
    create_referece_cell_matrices<Number>(fe, quadrature);

  AssertIndexRange(n_overlap, M_ref.n());
  AssertIndexRange(0, n_overlap);
  AssertThrow(is_dg == false, ExcNotImplemented());

  const unsigned int n_dofs_1D              = M_ref.n();
  const unsigned int n_dofs_1D_with_overlap = M_ref.n() - 2 + 2 * n_overlap;

  std::array<FullMatrix<Number>, dim> Ms;
  std::array<FullMatrix<Number>, dim> Ks;
  std::array<std::vector<bool>, dim>  masks;

  const auto clear_row_and_column = [&](const unsigned int n, auto &matrix) {
    for (unsigned int i = 0; i < n_dofs_1D_with_overlap; ++i)
      {
        matrix[i][n] = 0.0;
        matrix[n][i] = 0.0;
      }
  };

  // 2) loop over all dimensions and create mass and stiffness
  // matrix so that boundary conditions and overlap are considered
  for (unsigned int d = 0; d < dim; ++d)
    {
      FullMatrix<Number> M(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);
      FullMatrix<Number> K(n_dofs_1D_with_overlap, n_dofs_1D_with_overlap);

      masks[d].assign(n_dofs_1D_with_overlap, true);

      // inner cell
      for (unsigned int i = 0; i < n_dofs_1D; ++i)
        for (unsigned int j = 0; j < n_dofs_1D; ++j)
          {
            const unsigned int i0 = i + n_overlap - 1;
            const unsigned int j0 = j + n_overlap - 1;
            M[i0][j0]             = M_ref[i][j] * cell_extend[d][1];
            K[i0][j0]             = K_ref[i][j] / cell_extend[d][1];
          }

      // left neighbor or left boundary
      if (cell->at_boundary(2 * d) == false)
        {
          // left neighbor
          Assert(cell_extend[d][0] > 0.0, ExcInternalError());

          for (unsigned int i = 0; i < n_overlap; ++i)
            for (unsigned int j = 0; j < n_overlap; ++j)
              {
                const unsigned int i0 = n_dofs_1D - n_overlap + i;
                const unsigned int j0 = n_dofs_1D - n_overlap + j;
                M[i][j] += M_ref[i0][j0] * cell_extend[d][0];
                K[i][j] += K_ref[i0][j0] / cell_extend[d][0];
              }
        }
      else if (cell->face(2 * d)->boundary_id() == 1 /*DBC*/)
        {
          // left DBC
          const unsigned i0 = n_overlap - 1;
          clear_row_and_column(i0, M);
          clear_row_and_column(i0, K);
        }
      else
        {
          // left NBC -> nothing to do
        }

      // reight neighbor or right boundary
      if (cell->at_boundary(2 * d + 1) == false)
        {
          Assert(cell_extend[d][2] > 0.0, ExcInternalError());

          for (unsigned int i = 0; i < n_overlap; ++i)
            for (unsigned int j = 0; j < n_overlap; ++j)
              {
                const unsigned int i0 = n_overlap + n_dofs_1D + i - 2;
                const unsigned int j0 = n_overlap + n_dofs_1D + j - 2;
                M[i0][j0] += M_ref[i][j] * cell_extend[d][2];
                K[i0][j0] += K_ref[i][j] / cell_extend[d][2];
              }
        }
      else if (cell->face(2 * d + 1)->boundary_id() == 1 /*DBC*/)
        {
          // right DBC
          const unsigned i0 = n_overlap + n_dofs_1D - 2;
          clear_row_and_column(i0, M);
          clear_row_and_column(i0, K);
        }
      else
        {
          // right NBC -> nothing to do
        }

      for (unsigned int i = 0; i < n_dofs_1D_with_overlap; ++i)
        if (K[i][i] == 0.0)
          {
            Assert(M[i][i] == 0.0, ExcInternalError());

            M[i][i]     = 1.0;
            K[i][i]     = 1.0;
            masks[d][i] = false;
          }

      Ms[d] = M;
      Ks[d] = K;
    }

  MyTensorProductMatrixSymmetricSum<dim, Number> fdm;
  fdm.reinit(Ms, Ks);
  fdm.set_mask(masks);

  return fdm;
}

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