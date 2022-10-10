#pragma once

template <int dim_templated,
          int n_points_1d_templated,
          typename ProcessorType,
          typename VectorType,
          typename VectorizedArrayType>
void
read_write_operation(const ProcessorType &processor,
                     VectorType &         vec,
                     const unsigned int   dim_non_templated,
                     const unsigned int   n_points_1d_non_templated,
                     const unsigned int * cell_indices,
                     VectorizedArrayType *dof_values)
{
  const unsigned int dim =
    (dim_templated != -1) ? dim_templated : dim_non_templated;
  const unsigned int n_points_1d = (n_points_1d_templated != -1) ?
                                     n_points_1d_templated :
                                     n_points_1d_non_templated;

  const unsigned int n_inside_1d = n_points_1d / 2;
  const unsigned int n_lanes     = VectorizedArrayType::size();

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
          const unsigned int  offset = (j == n_inside_1d) ?
                                         (k_offset) :
                                         (k_offset * n_inside_1d + j_offset);
          const unsigned int *indices =
            cell_indices +
            3 * n_lanes * (3 * compressed_index[k] + compressed_index[j]);

          // left end
          for (unsigned int i = 0; i < n_inside_1d; ++i, ++c)
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                processor.process_dof(indices[v] + offset * n_inside_1d + i,
                                      vec,
                                      dof_values[c][v]);

          indices += n_lanes;

          // interior
          for (unsigned int v = 0; v < n_lanes; ++v)
            if (indices[v] != dealii::numbers::invalid_unsigned_int)
              processor.process_dof(indices[v] + offset, vec, dof_values[c][v]);

          c += 1;
          indices += n_lanes;

          // right end
          for (unsigned int i = 0; i < n_inside_1d; ++i, ++c)
            for (unsigned int v = 0; v < n_lanes; ++v)
              if (indices[v] != dealii::numbers::invalid_unsigned_int)
                processor.process_dof(indices[v] + offset * n_inside_1d + i,
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

bool
read_write_operation_setup(
  const std::vector<types::global_dof_index> &              dof_indices,
  const unsigned int                                        dim,
  const unsigned int                                        n_points_1d,
  std::vector<unsigned int> &                               compressed,
  const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner = {})
{
  const unsigned int n_inside_1d = n_points_1d / 2;

  unsigned int compressed_index[100];

  unsigned int c = 0;
  for (unsigned int i = 0; i < n_inside_1d; ++i)
    compressed_index[c++] = 0;
  compressed_index[c++] = 1;
  for (unsigned int i = 0; i < n_inside_1d; ++i)
    compressed_index[c++] = 2;

  compressed.assign(Utilities::pow(3, dim),
                    dealii::numbers::invalid_unsigned_int);

  const auto global_to_local = [&](const auto index) -> unsigned int {
    if (index == dealii::numbers::invalid_unsigned_int)
      return dealii::numbers::invalid_unsigned_int;
    else if (partitioner)
      return partitioner->global_to_local(index);
    else
      return index;
  };

  const auto try_to_set =
    [&](auto &compressed_index, const auto i, const auto index) -> bool {
    if (i == 0)
      {
        AssertThrow(compressed_index == dealii::numbers::invalid_unsigned_int,
                    ExcInternalError());
        compressed_index = global_to_local(index);

        return true;
      }
    else if ((compressed_index == dealii::numbers::invalid_unsigned_int) &&
             (index == dealii::numbers::invalid_unsigned_int))
      {
        return true;
      }
    else
      {
        return (compressed_index + i) == global_to_local(index);
      }
  };

  for (unsigned int k = 0, c = 0, k_offset = 0; k < (dim > 2 ? n_points_1d : 1);
       ++k)
    {
      for (unsigned int j = 0, j_offset = 0; j < (dim > 1 ? n_points_1d : 1);
           ++j)
        {
          unsigned int *indices =
            compressed.data() +
            3 * (3 * compressed_index[k] + compressed_index[j]);

          const unsigned int offset = (j == n_inside_1d) ?
                                        (k_offset) :
                                        (k_offset * n_inside_1d + j_offset);

          for (unsigned int i = 0; i < n_inside_1d; ++i)
            if (!try_to_set(indices[0],
                            offset * n_inside_1d + i,
                            dof_indices[c++]))
              return false;

          if (!try_to_set(indices[1], offset, dof_indices[c++]))
            return false;

          for (unsigned int i = 0; i < n_inside_1d; ++i)
            if (!try_to_set(indices[2],
                            offset * n_inside_1d + i,
                            dof_indices[c++]))
              return false;

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

  return true;
}
