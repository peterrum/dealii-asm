#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <iomanip>
#include <iostream>

using namespace dealii;

namespace dealii
{
  namespace internal
  {} // namespace internal
} // namespace dealii

int
main()
{
  {
    const unsigned int dim       = 2;
    const unsigned int fe_degree = 5;
    using Number                 = double;

    std::vector<Number> weights(Utilities::pow(3, dim));
    for (unsigned int i = 0; i < weights.size(); ++i)
      weights[i] = i;

    std::vector<Number> values(Utilities::pow(fe_degree + 1, dim), 1.0);

    internal::weight_fe_q_dofs_by_entity<dim, -1, Number>(weights.data(),
                                                          1,
                                                          fe_degree + 1,
                                                          values.data());

    for (unsigned int i_1 = 0, c = 0; i_1 < fe_degree + 1; ++i_1)
      {
        for (unsigned int i_0 = 0; i_0 < fe_degree + 1; ++i_0, ++c)
          std::cout << values[c] << " ";
        std::cout << std::endl;
      }
    std::cout << std::endl;

    for (auto &i : weights)
      i = 0.0;

    internal::compute_weights_fe_q_dofs_by_entity<dim, -1, Number>(
      values.data(), 1, fe_degree + 1, weights.data());

    for (const auto i : weights)
      std::cout << i << " ";

    std::cout << std::endl;
    std::cout << std::endl;
  }

  {
    const unsigned int dim       = 2;
    const unsigned int fe_degree = 4;
    using Number                 = double;

    std::vector<Number> weights(Utilities::pow(3, dim));
    for (unsigned int i = 0; i < weights.size(); ++i)
      weights[i] = i;

    std::vector<Number> values(Utilities::pow((2 * fe_degree - 1), dim), 1.0);

    internal::weight_fe_q_dofs_by_entity_shifted<dim, -1, Number>(
      weights.data(), 1, 2 * fe_degree - 1, values.data());

    for (unsigned int i_1 = 0, c = 0; i_1 < (2 * fe_degree - 1); ++i_1)
      {
        for (unsigned int i_0 = 0; i_0 < (2 * fe_degree - 1); ++i_0, ++c)
          std::cout << values[c] << " ";
        std::cout << std::endl;
      }
    std::cout << std::endl;

    for (auto &i : weights)
      i = 0.0;

    internal::compute_weights_fe_q_dofs_by_entity_shifted<dim, -1, Number>(
      values.data(), 1, 2 * fe_degree - 1, weights.data());

    for (const auto i : weights)
      std::cout << i << " ";

    std::cout << std::endl;
    std::cout << std::endl;
  }
}