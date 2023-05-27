#pragma once

#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <bit>

unsigned int
get_orientation_line(const std::vector<types::global_dof_index> &dofs,
                     const unsigned int                          degree,
                     const unsigned int                          n_components)
{
  bool flag;

  flag = true;
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int i = 0; i < degree - 1; ++i)
      flag &= ((dofs[i + c * (degree - 1)]) ==
               (dofs[0 + c * (degree - 1)] + i * n_components));

  if (flag)
    return 0; // normal ordering

  flag = true;
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int i = 0; i < degree - 1; ++i)
      flag &= ((dofs[i + c * (degree - 1)]) ==
               (dofs[0 + c * (degree - 1)] - i * n_components));

  if (flag)
    return 1; // flipped

  AssertThrow(false, ExcNotImplemented());

  return numbers::invalid_unsigned_int;
}

unsigned int
get_orientation_quad(const std::vector<types::global_dof_index> &dofs,
                     const unsigned int                          n_components,
                     const Table<2, unsigned int> &orientation_table)
{
  for (unsigned int i = 0; i < orientation_table.n_rows(); ++i)
    {
      bool flag = true;

      for (unsigned int c = 0; c < n_components; ++c)
        {
          const auto min =
            *std::min_element(dofs.begin() + c * orientation_table.n_cols(),
                              dofs.begin() +
                                (c + 1) * orientation_table.n_cols());

          for (unsigned int j = 0; j < orientation_table.n_cols(); ++j)
            flag &= ((dofs[j + c * orientation_table.n_cols()]) ==
                     (min + orientation_table[i][j] * n_components));
        }

      if (flag)
        return i;
    }

  AssertThrow(false, ExcNotImplemented()); // TODO

  return numbers::invalid_unsigned_int;
}

unsigned int
compress_orientation(const std::vector<unsigned int> &orientations,
                     const bool                       do_post = false)
{
  if (std::find(orientations.begin(),
                orientations.end(),
                numbers::invalid_unsigned_int) != orientations.end())
    return numbers::invalid_unsigned_int;

  unsigned int orientation = 0;

  if (orientations.size() == 4) // 2D
    {
      if (do_post)
        {
          orientation += orientations[2] << 0;
          orientation += orientations[3] << 1;
          orientation += orientations[0] << 2;
          orientation += orientations[1] << 3;
        }
      else
        {
          // lines in lexicographic order
          orientation += orientations[2] << 0;
          orientation += orientations[0] << 1;
          orientation += orientations[1] << 2;
          orientation += orientations[3] << 3;
        }
    }
  else if (orientations.size() == 18) // 3D
    {
      if (do_post)
        {
          unsigned int index_shift = 0;
          unsigned int dof_shift   = 0;

          for (const auto i : {2, 3, 6, 7, 0, 1, 4, 5, 8, 9, 10, 11}) // lines
            orientation |= orientations[i] << (dof_shift++);

          index_shift += 12;

          for (unsigned int i = 0; i < 6;
               ++i, ++index_shift, dof_shift += 3) // quads
            orientation |= orientations[index_shift] << dof_shift;
        }
      else
        {
          // lines and quads in lexicographic order
          std::vector<std::pair<unsigned int, unsigned int>> orientation_table{
            // bottom layer
            {1, orientations[2]},
            {1, orientations[0]},
            {3, orientations[16]},
            {1, orientations[1]},
            {1, orientations[3]},
            // middle layer
            {1, orientations[8]},
            {3, orientations[14]},
            {1, orientations[9]},
            {3, orientations[12]},
            {3, orientations[13]},
            {1, orientations[10]},
            {3, orientations[15]},
            {1, orientations[11]},
            // bottom layer
            {1, orientations[6]},
            {1, orientations[4]},
            {3, orientations[17]},
            {1, orientations[5]},
            {1, orientations[7]}};

          for (unsigned int i = 0, s = 0; i < orientation_table.size(); ++i)
            {
              orientation += orientation_table[i].second << s;
              s += orientation_table[i].first;
            }
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  return orientation;
}

std::pair<unsigned int, std::vector<types::global_dof_index>>
compress_indices(const std::vector<types::global_dof_index> &dofs,
                 const unsigned int                          dim,
                 const unsigned int                          degree,
                 const unsigned int                          n_components,
                 const bool                                  do_post = false)
{
  const auto orientation_table =
    internal::MatrixFreeFunctions::ShapeInfo<double>::compute_orientation_table(
      degree - 1); // TODO

  std::vector<std::pair<unsigned int, unsigned int>> dpo; // TODO

  if (dim == 1)
    {
      dpo.emplace_back(2, 1);
      dpo.emplace_back(1, degree - 1);
    }
  else if (dim == 2)
    {
      dpo.emplace_back(4, 1);
      dpo.emplace_back(4, degree - 1);
      dpo.emplace_back(1, (degree - 1) * (degree - 1));
    }
  else if (dim == 3)
    {
      dpo.emplace_back(8, 1);
      dpo.emplace_back(12, degree - 1);
      dpo.emplace_back(6, (degree - 1) * (degree - 1));
      dpo.emplace_back(1, (degree - 1) * (degree - 1) * (degree - 1));
    }

  std::vector<unsigned int>            obj_orientations;
  std::vector<types::global_dof_index> obj_start_indices;

  // loop over all dimension
  for (unsigned int d = 0, dof_counter = 0; d <= dim; ++d)
    {
      const auto entry = dpo[d];

      // loop over all objects of the given dimension
      for (unsigned int i = 0; i < entry.first; ++i)
        {
          // extract indices of object
          std::vector<types::global_dof_index> dofs_of_object(entry.second *
                                                              n_components);
          for (unsigned int j = 0; j < entry.second * n_components; ++j)
            dofs_of_object[j] = dofs[dof_counter + j];

          // deal with constraints
          const unsigned int n_constrained_dofs =
            std::count(dofs_of_object.begin(),
                       dofs_of_object.end(),
                       numbers::invalid_dof_index);

          if (0 < n_constrained_dofs &&
              n_constrained_dofs < dofs_of_object.size())
            {
              AssertThrow(false, ExcNotImplemented());
              return {numbers::invalid_unsigned_int, {}}; // TODO
            }

          if (n_constrained_dofs == dofs_of_object.size())
            {
              // all dofs of object are constrained
              obj_start_indices.emplace_back(numbers::invalid_dof_index);

              if ((dim >= 2 && d == 1) || (dim == 3 && d == 2))
                obj_orientations.emplace_back(0);
            }
          else
            {
              // no dof of object is constrained

              // store minimal index of object
              const auto min_ptr =
                std::min_element(dofs_of_object.begin(), dofs_of_object.end());

              AssertThrow(min_ptr != dofs_of_object.end(), ExcInternalError());

              obj_start_indices.emplace_back(*min_ptr);

              // if(false)
              if (dim == 3 &&
                  (d == 2 &&
                   (i == 2 || i == 3))) // reorientate quad 2 + 3 (lex)
                {
                  auto dofs_of_object_copy = dofs_of_object;

                  for (unsigned int c = 0; c < n_components; ++c)
                    for (unsigned int j = 0; j < entry.second; ++j)
                      dofs_of_object[j + c * entry.second] =
                        dofs_of_object_copy[orientation_table[1][j] +
                                            c * entry.second];
                }

              // sanity check for multiple components

              if (dim >= 2 && d == 1) // line orientations
                {
                  const auto orientation =
                    get_orientation_line(dofs_of_object, degree, n_components);

                  obj_orientations.emplace_back(orientation);
                }
              else if (dim == 3 && d == 2) // quad orientations
                {
                  const auto orientation =
                    get_orientation_quad(dofs_of_object,
                                         n_components,
                                         orientation_table);

                  obj_orientations.emplace_back(orientation);
                }
            }

          dof_counter += entry.second * n_components;

          // no compression is possible
          if ((obj_orientations.empty() == false) &&
              obj_orientations.back() == numbers::invalid_dof_index)
            return {numbers::invalid_unsigned_int, {}};
        }
    }

  // compress indices to a single
  const auto orientation_compressed =
    compress_orientation(obj_orientations, do_post);

  // return orientation and start indices
  return {orientation_compressed, obj_start_indices};
}

template <typename Number>
void
gather(const std::vector<Number> &      global_vector,
       const unsigned int               dim,
       const unsigned int               degree,
       const std::vector<unsigned int> &dofs_of_cell,
       const unsigned int               orientation_in,
       const Table<2, unsigned int> &   orientation_table,
       std::vector<Number> &            local_vector)
{
  unsigned int orientation = orientation_in;

  if (dim == 2)
    {
      unsigned int counter    = 0;
      unsigned int offset     = 0;
      unsigned int compressed = 0;

      for (unsigned int j = 0; j <= degree; ++j)
        {
          const auto indices = dofs_of_cell.begin() + compressed * 3;

          if (orientation && (orientation & 0b1) && ((j == 0) || (j == degree)))
            {
              // non-standard bottom or top layer (vertex-line-vertex)

              // vertex 0 or vertex 2
              local_vector[counter++] = global_vector[indices[0]];

              // line 2 or line 3
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] + (degree - 2 - i)];

              // vertex 1 or vertex 3
              local_vector[counter++] = global_vector[indices[2]];
            }
          else if (orientation && (orientation & 0b11) && (0 < j) &&
                   (j < degree))
            {
              // non-standard middle layers (line-quad-line)

              // line 0
              if (orientation & 0b01)
                local_vector[counter++] =
                  global_vector[indices[0] + (degree - 2 - offset)];
              else
                local_vector[counter++] = global_vector[indices[0] + offset];

              // quad 0
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] + offset * (degree - 1) + i];

              // line 1
              if (orientation & 0b10)
                local_vector[counter++] =
                  global_vector[indices[2] + (degree - 2 - offset)];
              else
                local_vector[counter++] = global_vector[indices[2] + offset];
            }
          else
            {
              // standard layer -> nothing to do

              local_vector[counter++] = global_vector[indices[0] + offset];

              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] + offset * (degree - 1) + i];

              local_vector[counter++] = global_vector[indices[2] + offset];
            }

          if ((j == 0) || (j == (degree - 1)))
            {
              compressed++;
              offset = 0;

              if (j == 0)
                orientation = orientation >> 1;
              else
                orientation = orientation >> 2;
            }
          else
            offset++;
        }
    }
  else
    {
      for (unsigned int k            = 0,
                        compressed_k = 0,
                        offset_k     = 0,
                        c            = 0,
                        o_ptr        = orientation;
           k <= degree;
           ++k)
        {
          for (unsigned int j = 0, compressed_j = 0, offset_j = 0; j <= degree;
               ++j)
            {
              const unsigned int offset =
                (compressed_j == 1 ? degree - 1 : 1) * offset_k + offset_j;

              const auto indices =
                dofs_of_cell.begin() + 3 * (compressed_k * 3 + compressed_j);

              if ((orientation != 0) && (o_ptr & 0b1) &&
                  (k == 0 || k == degree) && (j == 0 || j == degree))
                {
                  // case 1: vertex-line-vertex

                  // vertex
                  local_vector[c++] = global_vector[indices[0]];

                  // line
                  for (unsigned int i = 0; i < degree - 1; ++i)
                    local_vector[c++] =
                      global_vector[indices[1] + (degree - 2 - i)];

                  // vertex
                  local_vector[c++] = global_vector[indices[2]];
                }
              else if ((orientation != 0) && (o_ptr & 0b11111) &&
                       (((k == 0 || k == degree) &&
                         ((0 < j) && (j < degree))) ||
                        (((0 < k) && (k < degree)) && (j == 0 || j == degree))))
                {
                  // case 2: line-quad-line

                  const unsigned int jk = (k == 0 || k == degree) ? j : k;

                  // line
                  if (o_ptr & 0b00001)
                    local_vector[c++] =
                      global_vector[indices[0] + (degree - 1 - jk)];
                  else
                    local_vector[c++] = global_vector[indices[0] + (jk - 1)];

                  // quad (ij or ik)
                  const unsigned int quad_flag = (o_ptr >> 1) & 0b111;
                  for (unsigned int i = 0; i < degree - 1; ++i)
                    if (quad_flag != 0)
                      local_vector[c++] = global_vector
                        [indices[1] +
                         orientation_table[quad_flag]
                                          [(degree - 1) * (jk - 1) + i]];
                    else
                      local_vector[c++] =
                        global_vector[indices[1] + (degree - 1) * (jk - 1) + i];

                  // line
                  if (o_ptr & 0b10000)
                    local_vector[c++] =
                      global_vector[indices[2] + (degree - 1 - jk)];
                  else
                    local_vector[c++] = global_vector[indices[2] + (jk - 1)];
                }
              else if ((orientation != 0) && (o_ptr & 0b111111) && (0 < k) &&
                       (k < degree) && (0 < j) && (j < degree))
                {
                  // case 3: quad-hex-quad

                  // quad (jk)
                  const unsigned int quad_flag_0 = (o_ptr >> 0) & 0b111;
                  if (quad_flag_0 != 0)
                    local_vector[c++] =
                      global_vector[indices[0] +
                                    orientation_table[quad_flag_0][offset]];
                  else
                    local_vector[c++] = global_vector[indices[0] + offset];

                  // hex
                  for (unsigned int i = 0; i < degree - 1; ++i)
                    local_vector[c++] =
                      global_vector[indices[1] + offset * (degree - 1) + i];

                  // quad (jk)
                  const unsigned int quad_flag_1 = (o_ptr >> 3) & 0b111;
                  if (quad_flag_1 != 0)
                    local_vector[c++] =
                      global_vector[indices[2] +
                                    orientation_table[quad_flag_1][offset]];
                  else
                    local_vector[c++] = global_vector[indices[2] + offset];
                }
              else
                {
                  // case 4: standard -> nothing to do

                  local_vector[c++] = global_vector[indices[0] + offset];

                  for (unsigned int i = 0; i < degree - 1; ++i)
                    local_vector[c++] =
                      global_vector[indices[1] + offset * (degree - 1) + i];

                  local_vector[c++] = global_vector[indices[2] + offset];
                }

              if (j == 0 || j == degree - 1)
                {
                  ++compressed_j;
                  offset_j = 0;
                }
              else
                {
                  ++offset_j;
                }

              if (k == 0 || k == degree)
                {
                  if (j == 0 || j == degree)
                    o_ptr = std::rotr(o_ptr, 1);
                  else if (j == (degree - 1))
                    o_ptr = std::rotr(o_ptr, 5);
                }
              else
                {
                  if (j == 0 || j == degree)
                    o_ptr = std::rotr(o_ptr, 5);
                  else if (j == (degree - 1))
                    o_ptr = std::rotr(o_ptr, 6);
                }

              if (0 < k && k < (degree - 1) && j == degree)
                o_ptr = std::rotl(o_ptr, 16);
            }

          if (k == 0 || k == degree - 1)
            {
              ++compressed_k;
              offset_k = 0;
            }
          else
            {
              ++offset_k;
            }
        }
    }
}

template <typename Number, int dim_template = -1, int degree_template = -1>
void
adjust_for_orientation(const unsigned int               dim_non_template,
                       const unsigned int               degree_non_template,
                       const unsigned int               n_components,
                       const bool                       integrate,
                       const unsigned int               cell,
                       const std::vector<unsigned int> &orientation_in,
                       const Table<2, unsigned int> &   orientation_table,
                       Number *                         local_vector)
{
  if (dim_non_template == 1 || orientation_in.empty())
    return; // nothing to do for 1D

  const unsigned int dim =
    (dim_template != -1) ? dim_template : dim_non_template;
  const unsigned int degree =
    (degree_template != -1) ? degree_template : degree_non_template;

  const unsigned int n_lines_per_cell    = dim == 2 ? 4 : 12;
  const unsigned int n_vertices_per_face = dim == 2 ? 2 : 4;
  const unsigned int n_quads_per_cell    = dim == 2 ? 0 : 6;

  const unsigned int np  = degree + 1;
  const unsigned int np2 = np * np;

  const unsigned int dofs_per_comp = Utilities::pow(np, dim);

  using VectorizedArrayTrait = internal::VectorizedArrayTrait<Number>;

  for (unsigned int v = 0; v < VectorizedArrayTrait::width(); ++v)
    {
      unsigned int orientation =
        orientation_in[cell * VectorizedArrayTrait::width() + v];

      if ((dim >= 2) && (orientation != 0)) // process lines
        {
          if (((dim == 2) && (orientation != 0)) ||
              ((dim == 3) &&
               (orientation &
                0b111111111111))) // at least one line is irregular
            {
              for (unsigned int l = 0; l < n_lines_per_cell;
                   ++l) // loop over all lines
                {
                  if (orientation & 1) // check bit
                    {
                      // determine stride and begin
                      const unsigned int stride =
                        Utilities::pow(degree + 1, l / n_vertices_per_face);

                      unsigned int begin = 0;
                      if (dim == 2)
                        {
                          if (l == 0)
                            begin = 1;
                          else if (l == 1)
                            begin = degree * degree + degree + 1;
                          else if (l == 2)
                            begin = degree + 1;
                          else
                            begin = 2 * degree + 1;
                        }
                      else
                        {
                          if (l == 0)
                            begin = 1;
                          else if (l == 1)
                            begin = np * degree + 1;
                          else if (l == 2)
                            begin = np2 * degree + 1;
                          else if (l == 3)
                            begin = np2 * degree + np * degree + 1;
                          else if (l == 4)
                            begin = np;
                          else if (l == 5)
                            begin = np + degree;
                          else if (l == 6)
                            begin = np2 * degree + np;
                          else if (l == 7)
                            begin = np2 * degree + np + degree;
                          else if (l == 8)
                            begin = np2;
                          else if (l == 9)
                            begin = np2 + degree;
                          else if (l == 10)
                            begin = np2 + np * degree;
                          else if (l == 11)
                            begin = np2 + np * degree + degree;
                        }

                      // perform reorientation
                      for (unsigned int c = 0; c < n_components; ++c)
                        for (unsigned int i0 = 0; i0 < (degree - 1) / 2; ++i0)
                          std::swap(VectorizedArrayTrait::get(
                                      local_vector[(c * dofs_per_comp + begin) +
                                                   i0 * stride],
                                      v),
                                    VectorizedArrayTrait::get(
                                      local_vector[(c * dofs_per_comp + begin) +
                                                   (degree - 2 - i0) * stride],
                                      v));
                    }

                  orientation = orientation >> 1; //  go to next bit
                }
            }
          else if (dim == 3) // all lines are regular
            {
              orientation = orientation >> 12;
            }
        }

      if ((dim == 3) && (orientation != 0)) // process quads
        {
          for (unsigned int q = 0; q < n_quads_per_cell;
               ++q) // loop over all quads
            {
              const unsigned int flag = orientation & 0b111;

              if (flag != 0) // check bits
                {
                  typename VectorizedArrayTrait::value_type temp[100];

                  const unsigned int d       = q / 2;
                  const unsigned int stride0 = (d == 0) ? np : 1;
                  const unsigned int stride1 = (d == 2) ? np : np2;
                  const unsigned int begin =
                    ((q % 2) == 0) ? 0 : (Utilities::pow(np, q / 2) * degree);

                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      // copy values into buffer
                      for (unsigned int i1 = 1, i = 0; i1 < degree; ++i1)
                        for (unsigned int i0 = 1; i0 < degree; ++i0, ++i)
                          if (integrate)
                            {
                              // TODO: can this be done easier?
                              const unsigned int j =
                                orientation_table[flag]
                                                 [(i0 - 1) +
                                                  (i1 - 1) * (degree - 1)];

                              const unsigned int i0_ = (j % (degree - 1)) + 1;
                              const unsigned int i1_ = (j / (degree - 1)) + 1;

                              temp[i] = VectorizedArrayTrait::get(
                                local_vector[(c * dofs_per_comp + begin) +
                                             i0_ * stride0 + i1_ * stride1],
                                v);
                            }
                          else
                            {
                              temp[orientation_table[flag][i]] =
                                VectorizedArrayTrait::get(
                                  local_vector[(c * dofs_per_comp + begin) +
                                               i0 * stride0 + i1 * stride1],
                                  v);
                            }

                      // perform permuation
                      for (unsigned int i1 = 1, i = 0; i1 < degree; ++i1)
                        for (unsigned int i0 = 1; i0 < degree; ++i0, ++i)
                          VectorizedArrayTrait::get(
                            local_vector[(c * dofs_per_comp + begin) +
                                         i0 * stride0 + i1 * stride1],
                            v) = temp[i];
                    }
                }

              orientation = orientation >> 3; //  go to next bits
            }
        }
    }
}

template <typename Number>
void
gather_post(const std::vector<Number> &      global_vector,
            const unsigned int               dim,
            const unsigned int               degree,
            const unsigned int               n_components,
            const bool                       integrate,
            const unsigned int               cell,
            const std::vector<unsigned int> &dofs_of_cell,
            const std::vector<unsigned int>  orientation,
            const Table<2, unsigned int> &   orientation_table,
            Number *                         local_vector)
{
  for (unsigned int i2 = 0, compressed_i2 = 0, offset_k = 0, i = 0;
       i2 <= (dim == 2 ? 0 : degree);
       ++i2)
    {
      for (unsigned int i1 = 0, compressed_i1 = 0, offset_j = 0; i1 <= degree;
           ++i1)
        {
          const unsigned int offset =
            (compressed_i1 == 1 ? degree - 1 : 1) * offset_k + offset_j;

          const auto indices =
            dofs_of_cell.begin() + 3 * (compressed_i2 * 3 + compressed_i1);

          local_vector[i] = global_vector[indices[0] + offset];
          ++i;

          for (unsigned int i0 = 0; i0 < degree - 1; ++i0, ++i)
            local_vector[i] =
              global_vector[indices[1] + offset * (degree - 1) + i0];

          local_vector[i] = global_vector[indices[2] + offset];
          ++i;

          if (i1 == 0 || i1 == degree - 1)
            {
              ++compressed_i1;
              offset_j = 0;
            }
          else
            {
              ++offset_j;
            }
        }

      if (i2 == 0 || i2 == degree - 1)
        {
          ++compressed_i2;
          offset_k = 0;
        }
      else
        {
          ++offset_k;
        }
    }

  adjust_for_orientation(dim,
                         degree,
                         n_components,
                         integrate,
                         cell,
                         orientation,
                         orientation_table,
                         local_vector);
}
