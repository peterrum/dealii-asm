#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

using namespace dealii;

template <int dim, int spacedim, typename Number = double>
FullMatrix<Number>
compute_restricted_element_matrix(const FiniteElement<dim, spacedim> &fe,
                                  const Quadrature<dim> &             quad)
{
  const MappingQ1<dim> mapping;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            std::vector<unsigned int>{3},
                                            Point<dim>(0.0),
                                            Point<dim>(3.0),
                                            false);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix<dim, dim>(
    mapping, dof_handler, quad, system_matrix, nullptr, constraints);

  auto cell = dof_handler.begin();
  cell++;

  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<Number> matrix(n_dofs_per_cell, n_dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);

  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
      matrix(i, j) = system_matrix(local_dof_indices[i], local_dof_indices[j]);


  return matrix;
}

template <int dim, int spacedim, typename Number = double>
FullMatrix<Number>
compute_element_matrix(const FiniteElement<dim, spacedim> &fe,
                       const Quadrature<dim> &             quad)
{
  const MappingQ1<dim> mapping;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            std::vector<unsigned int>{1},
                                            Point<dim>(0.0),
                                            Point<dim>(1.0),
                                            false);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  // DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  MatrixCreator::create_laplace_matrix<dim, dim>(
    mapping, dof_handler, quad, system_matrix, nullptr, constraints);

  auto cell = dof_handler.begin();
  // cell++;

  const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<Number> matrix(n_dofs_per_cell, n_dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
  cell->get_dof_indices(local_dof_indices);

  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
      matrix(i, j) = system_matrix(local_dof_indices[i], local_dof_indices[j]);

  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    matrix(i, i) += 1;

  return matrix;
}

template <typename Number>
void
compute_eigen_values(const FullMatrix<Number> &A,
                     const FullMatrix<Number> &P_in)
{
  FullMatrix<Number> P = P_in;

  FullMatrix<Number> PA(A.m(), A.n());
  P.gauss_jordan();
  P.mmult(PA, A);

  LAPACKFullMatrix<double> PA_lapack(A.m(), A.n());
  PA_lapack = PA;
  PA_lapack.compute_eigenvalues();

  std::vector<double> eigenvalues(A.m());

  for (unsigned int i = 0; i < eigenvalues.size(); ++i)
    eigenvalues[i] = std::abs(PA_lapack.eigenvalue(i));

  std::sort(eigenvalues.begin(), eigenvalues.end());

  for (const auto i : eigenvalues)
    std::cout << i << " ";

  std::cout << "-> " << (eigenvalues.back() / eigenvalues.front()) << std::endl;
}

int
main(int argc, char *argv[])
{
  const unsigned int dim           = 1;
  const unsigned int fe_degree_min = argc > 1 ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree_max =
    argc > 2 ? std::atoi(argv[2]) : fe_degree_min;

  for (unsigned int fe_degree = fe_degree_min; fe_degree <= fe_degree_max;
       ++fe_degree)
    {
      // 0) normal FE_Q(degree)
      const FE_Q<dim>   fe(fe_degree);
      const QGauss<dim> quad(fe_degree + 1);

      // 1) normal FE_Q(1) with subdivisions at degree+1 support points
      const auto subdivision_point =
        QGaussLobatto<1>(fe_degree + 1).get_points();
      const FE_Q_iso_Q1<dim> fe_q1_n(subdivision_point);
      const QIterated<dim>   quad_q1_n(QGauss<1>(2), subdivision_point);

      // 2) normal FE_Q(1) with subdivisions at degree+1 equidistant support
      // points
      const FE_Q_iso_Q1<dim> fe_q1_h(fe_degree);
      const QIterated<dim>   quad_q1_h(QGauss<1>(2), fe_degree + 1);

      // compute element stiffness matrices
      const auto matrix_fe_0 = compute_restricted_element_matrix(fe, quad);
      auto matrix_fe_1 = compute_restricted_element_matrix(fe_q1_n, quad_q1_n);
      auto matrix_fe_2 = compute_restricted_element_matrix(fe_q1_h, quad_q1_h);
      auto matrix_fe_3 = compute_element_matrix(fe, quad);

      // print eigenvalues
      std::cout << "degree " << std::to_string(fe_degree) << ":" << std::endl;
      compute_eigen_values(matrix_fe_0, matrix_fe_0); // should be 1
      compute_eigen_values(matrix_fe_0, matrix_fe_1);
      compute_eigen_values(matrix_fe_0, matrix_fe_2);
      compute_eigen_values(matrix_fe_0, matrix_fe_3);
      std::cout << std::endl;
    }
}