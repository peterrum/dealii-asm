#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/shape_info.h>

#include <array>
#include <bit>
#include <iostream>
#include <vector>

using namespace dealii;

#include "reduced_access.h"

template <typename Number>
void
gather(const std::vector<Number> &      global_vector,
       const unsigned int               dim,
       const unsigned int               degree,
       const std::vector<unsigned int> &dofs_of_cell,
       const std::vector<unsigned int> &orientations, // TODO: compress
       const Table<2, unsigned int> &   orientation_table,
       std::vector<Number> &            local_vector)
{
  (void)dim;

  unsigned int orientation = compress_orientation(orientations, false);

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

template <typename Number>
void
gather_post(const std::vector<Number> &      global_vector,
            const unsigned int               dim,
            const unsigned int               degree,
            const std::vector<unsigned int> &dofs_of_cell,
            const std::vector<unsigned int> &orientations, // TODO: compress
            const Table<2, unsigned int> &   orientation_table,
            std::vector<Number> &            local_vector)
{
  unsigned int orientation = compress_orientation(orientations, true);

  for (unsigned int k = 0, compressed_k = 0, offset_k = 0, c = 0;
       k <= (dim == 2 ? 0 : degree);
       ++k)
    {
      for (unsigned int j = 0, compressed_j = 0, offset_j = 0; j <= degree; ++j)
        {
          const unsigned int offset =
            (compressed_j == 1 ? degree - 1 : 1) * offset_k + offset_j;

          const auto indices =
            dofs_of_cell.begin() + 3 * (compressed_k * 3 + compressed_j);

          local_vector[c++] = global_vector[indices[0] + offset];

          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[c++] =
              global_vector[indices[1] + offset * (degree - 1) + i];

          local_vector[c++] = global_vector[indices[2] + offset];

          if (j == 0 || j == degree - 1)
            {
              ++compressed_j;
              offset_j = 0;
            }
          else
            {
              ++offset_j;
            }
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

  if (dim == 2)
    {
      if (orientation != 0)                  // process lines
        for (unsigned int l = 0; l < 4; ++l) // loop over all lines
          {
            if (orientation & 1) // check bit
              {
                // determine stride and begin
                const unsigned int stride = (l < 2) ? (degree + 1) : 1;

                unsigned int begin = 0;
                if (l == 0)
                  begin = degree + 1;
                else if (l == 1)
                  begin = 2 * degree + 1;
                else if (l == 2)
                  begin = 1;
                else
                  begin = degree * degree + degree + 1;

                // perform reorientation
                for (unsigned int i = 0; i < (degree - 1) / 2; ++i)
                  std::swap(local_vector[begin + i * stride],
                            local_vector[begin + (degree - 2 - i) * stride]);
              }

            orientation = orientation >> 1; //  go to next bit
          }
    }
  else
    {
      const unsigned int np  = degree + 1;
      const unsigned int np2 = np * np;

      if (orientation != 0) // process lines
        {
          if (orientation & 0b111111111111) // at least one line is irregular
            {
              for (unsigned int l = 0; l < 12; ++l) // loop over all lines
                {
                  if (orientation & 1) // check bit
                    {
                      // determine stride and begin
                      const unsigned int stride =
                        Utilities::pow(degree + 1, l / 4);

                      unsigned int begin = 0;
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

                      // perform reorientation
                      for (unsigned int i = 0; i < (degree - 1) / 2; ++i)
                        std::swap(
                          local_vector[begin + i * stride],
                          local_vector[begin + (degree - 2 - i) * stride]);
                    }

                  orientation = orientation >> 1; //  go to next bit
                }
            }
          else // all lines are regular
            {
              orientation = orientation >> 12;
            }
        }

      if (orientation != 0) // process quads
        {
          for (unsigned int q = 0; q < 6; ++q) // loop over all quads
            {
              const unsigned int flag = orientation & 0b111;

              if (flag != 0) // check bits
                {
                  Number temp[100];

                  const unsigned int d       = q / 2;
                  const unsigned int stride  = (d == 0) ? np : 1;
                  const unsigned int stride2 = (d == 2) ? np : np2;
                  const unsigned int begin =
                    ((q % 2) == 0) ? 0 : (Utilities::pow(np, q / 2) * degree);

                  // copy values into buffer
                  for (unsigned int g = 1, c = 0; g < degree; ++g)
                    for (unsigned int k = 1; k < degree; ++k, ++c)
                      temp[c] = local_vector[begin + k * stride + stride2 * g];

                  // perform permuation
                  for (unsigned int g = 1, c = 0; g < degree; ++g)
                    for (unsigned int k = 1; k < degree; ++k, ++c)
                      local_vector[begin + k * stride + stride2 * g] =
                        temp[orientation_table[flag][c]];
                }

              orientation = orientation >> 3; //  go to next bits
            }
        }
    }
}

/**
 * ./reduced_access_02 3
 * ./reduced_access_02 3   0 0 0 0 0 0 0 0 0 0 0 0   0 0 0 0 0 1
 */
int
main(int argc, char *argv[])
{
  AssertThrow(argc == 2 || argc == 20, ExcNotImplemented());

  for (int i = 0; i < argc; ++i)
    std::cout << std::string(argv[i]) << " ";
  std::cout << std::endl << std::endl;

  const unsigned int dim    = 3;
  const unsigned int degree = atoi(argv[1]);

  std::vector<unsigned int> orientations(18, 0);

  if (argc == 20)
    for (unsigned int i = 0; i < 18; ++i)
      orientations[i] = atoi(argv[2 + i]);

  // setup dpo object
  std::vector<std::pair<unsigned int, unsigned int>> dpo;
  dpo.emplace_back(8, 1);
  dpo.emplace_back(12, degree - 1);
  dpo.emplace_back(6, (degree - 1) * (degree - 1));
  dpo.emplace_back(1, (degree - 1) * (degree - 1) * (degree - 1));

  // determine staring dof each entity
  std::vector<unsigned int> dofs;

  unsigned int dof_counter = 0;
  for (const auto &entry : dpo)
    for (unsigned int i = 0; i < entry.first; ++i)
      {
        dofs.emplace_back(dof_counter);
        dof_counter += entry.second;
      }

  // determine starting dof of each entity of cell

  // clang-format off
  std::vector<unsigned int> entities_of_cell = {
     0, 10,  1,  8, 24,  9,  2, 11,  3, // bottom layer
    16, 22, 17, 20, 26, 21, 18, 23, 19, // middle layer
     4, 14,  5, 12, 25, 13,  6, 15,  7  // top layer
    };
  // clang-format on

  std::vector<unsigned int> dofs_of_cell;

  for (const auto i : entities_of_cell)
    dofs_of_cell.emplace_back(dofs[i]);

  // create dummy global vector
  std::vector<double> global_vector;
  for (unsigned int i = 0; i < dof_counter; ++i)
    global_vector.emplace_back(i);

  const auto orientation_table =
    internal::MatrixFreeFunctions::ShapeInfo<double>::compute_orientation_table(
      degree - 1);

  // gather values and print to terminal
  std::vector<double> local_vector(dof_counter);

  if (true)
    gather(global_vector,
           dim,
           degree,
           dofs_of_cell,
           orientations,
           orientation_table,
           local_vector);
  else
    gather_post(global_vector,
                dim,
                degree,
                dofs_of_cell,
                orientations,
                orientation_table,
                local_vector);

  for (unsigned int k = 0, c = 0; k <= degree; ++k)
    {
      for (unsigned int j = 0; j <= degree; ++j)
        {
          for (unsigned int i = 0; i <= degree; ++i, ++c)
            printf("%4.0f", local_vector[c]);

          printf("\n");
        }
      printf("\n");
    }
  printf("\n\n");
}