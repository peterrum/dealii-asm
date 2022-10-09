
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

template <typename Number, typename VectorizedArrayType>
void
gather(const dealii::LinearAlgebra::distributed::Vector<Number> &vec,
       const unsigned int                                        dim,
       const unsigned int                                        n_points_1d,
       const std::vector<unsigned int> &compressed_dof_indices,
       VectorizedArrayType *            dof_values)
{
  internal::VectorReader<Number, VectorizedArrayType> processor;

  const unsigned int *cell_indices = compressed_dof_indices.data();

  const unsigned int n_inside_1d = n_points_1d / 2;
  const unsigned int n_lanes     = 1;

  unsigned int compressed_index[100];

  unsigned int c = 0;
  for (unsigned int i = 0; i < n_inside_1d; ++i)
    compressed_index[c++] = 0;
  compressed_index[c++] = 1;
  for (unsigned int i = 0; i < n_inside_1d; ++i)
    compressed_index[c++] = 2;

  for (unsigned int k = 0, c = 0, k_offset = 0; k < (dim > 2 ? n_points_1d : 1);
       ++k)
    {
      for (unsigned int j = 0, j_offset = 0; j < (dim > 1 ? n_points_1d : 1);
           ++j)
        {
          const unsigned int *indices =
            cell_indices +
            3 * n_lanes * (3 * compressed_index[k] + compressed_index[j]);

          for (unsigned int i = 0; i < n_inside_1d; ++i, ++c)
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                processor.process_dof(indices[v] +
                                        k_offset * n_inside_1d * n_inside_1d +
                                        j_offset * n_inside_1d + i,
                                      vec,
                                      dof_values[c][v]);

          indices += n_lanes;

          for (unsigned int v = 0; v < n_lanes; ++v)
            if (indices[v] != dealii::numbers::invalid_unsigned_int)
              processor.process_dof(indices[v] + n_inside_1d * n_inside_1d +
                                      j_offset,
                                    vec,
                                    dof_values[c][v]);

          c += 1;
          indices += n_lanes;

          for (unsigned int i = 0; i < n_inside_1d; ++i, ++c)
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                processor.process_dof(indices[v] +
                                        k_offset * n_inside_1d * n_inside_1d +
                                        j_offset * n_inside_1d + i,
                                      vec,
                                      dof_values[c][v]);

          if (((j + 1) == n_inside_1d) || (j == n_inside_1d))
            j_offset = 0;
          else
            j_offset++;
        }

      if (((k + 1) == n_inside_1d) || (k == n_inside_1d))
        k_offset = 0;
      else
        k_offset++;
    }
}

int
main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using VectorizedArrayType = VectorizedArray<Number, 1>;

  const unsigned int dim         = 2;
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

  gather(vec, dim, n_points_1d, compressed_dof_indices, data.data());



  for (unsigned int i_1 = 0, c = 0; i_1 < n_points_1d; ++i_1)
    {
      for (unsigned int i_0 = 0; i_0 < n_points_1d; ++i_0, ++c)
        printf("%4.0f", data[c][0]);
      std::cout << std::endl;
    }
  std::cout << std::endl;
}