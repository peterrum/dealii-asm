
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/vector_access_internal.h>

using namespace dealii;

#include "include/read_write_operation.h"

int
main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number, 1>;

  const int          dim         = 2;
  const unsigned int fe_degree   = 3;
  const unsigned int n_points_1d = fe_degree * 2 - 1;
  const unsigned int n_points    = Utilities::pow(n_points_1d, dim);

  std::cout << n_points << std::endl;

  VectorType vec(n_points);

  for (unsigned int i = 0; i < n_points; ++i)
    vec[i] = i;

  AlignedVector<VectorizedArrayType> data(n_points);
  std::vector<unsigned int>          compressed_dof_indices;
  compressed_dof_indices.push_back(0);

  for (unsigned int j = 0; j < 3; ++j)
    for (unsigned int i = 0; i < 3; ++i)
      {
        unsigned int n_dofs;

        if ((i == 0 || i == 2) && (j == 0 || j == 2))
          n_dofs = (fe_degree - 1) * (fe_degree - 1);
        else if ((i == 0 || i == 2) || (j == 0 || j == 2))
          n_dofs = (fe_degree - 1);
        else
          n_dofs = 1;

        compressed_dof_indices.push_back(compressed_dof_indices.back() +
                                         n_dofs);
      }

  for (const auto i : compressed_dof_indices)
    std::cout << i << " ";
  std::cout << std::endl;

  internal::VectorReader<Number, VectorizedArrayType> reader;
  read_write_operation<dim, -1>(
    reader, vec, dim, n_points_1d, compressed_dof_indices.data(), data.data());

  for (unsigned int i_1 = 0, c = 0; i_1 < n_points_1d; ++i_1)
    {
      for (unsigned int i_0 = 0; i_0 < n_points_1d; ++i_0, ++c)
        printf("%4.0f", data[c][0]);
      std::cout << std::endl;
    }
  std::cout << std::endl;

  for (const auto v : vec)
    printf("%4.0f", v);
  std::cout << std::endl;

  internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType>
    distributor;
  read_write_operation<dim, -1>(distributor,
                                vec,
                                dim,
                                n_points_1d,
                                compressed_dof_indices.data(),
                                data.data());

  for (const auto v : vec)
    printf("%4.0f", v);
  std::cout << std::endl;
}