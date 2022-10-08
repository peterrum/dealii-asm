#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/tensor_product_matrix_creator.h>

using namespace dealii;

int
main(int argc, char *argv[])
{
  const unsigned int dim       = 1;
  const unsigned int fe_degree = argc >= 2 ? std::atoi(argv[1]) : 4;

  FE_Q<1>   fe_1D(fe_degree);
  QGauss<1> quadrature_1D(fe_degree + 1);

  {
    const auto [M, K] =
      TensorProductMatrixCreator::create_laplace_tensor_product_matrix<dim,
                                                                       double>(
        fe_1D, quadrature_1D, {{{{1.0, 1.0}}}});

    M[0].print_formatted(std::cout, 2, true, 10);
    std::cout << std::endl;

    K[0].print_formatted(std::cout, 2, true, 10);
    std::cout << std::endl;
  }

  {
    const auto [M, K] =
      TensorProductMatrixCreator::create_laplace_tensor_product_matrix<dim,
                                                                       double>(
        fe_1D,
        quadrature_1D,
        {{{{TensorProductMatrixCreator::LaplaceBoundaryType::internal_boundary,
            TensorProductMatrixCreator::LaplaceBoundaryType::
              internal_boundary}}}},
        {{{{1.0, 1.0, 1.0}}}},
        2);

    M[0].print_formatted(std::cout, 2, true, 10);
    std::cout << std::endl;

    K[0].print_formatted(std::cout, 2, true, 10);
    std::cout << std::endl;
  }
}