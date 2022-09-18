#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/shape_info.h>

#include <array>
#include <bit>
#include <iostream>
#include <vector>

using namespace dealii;

template <typename Number>
void
gather(const std::vector<Number> &      global_vector,
       const unsigned int               degree,
       const std::vector<unsigned int> &dofs_of_cell,
       const std::vector<unsigned int> &orientations, // TODO: compress
       const Table<2, unsigned int> &   orientation_table,
       std::vector<Number> &            local_vector)
{
  std::vector<std::pair<unsigned int, unsigned int>> orientation{
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

  unsigned int o = 0;

  for (unsigned int i = 0, s = 0; i < orientation.size(); ++i)
    {
      o += orientation[i].second << s;
      s += orientation[i].first;
    }

  for (unsigned int k = 0, compressed_k = 0, offset_k = 0, c = 0, o_ptr = o;
       k <= degree;
       ++k)
    {
      for (unsigned int j = 0, compressed_j = 0, offset_j = 0; j <= degree; ++j)
        {
          const unsigned int offset =
            (compressed_j == 1 ? degree - 1 : 1) * offset_k + offset_j;

          const auto indices =
            dofs_of_cell.begin() + 3 * (compressed_k * 3 + compressed_j);

          if ((o != 0) && (o_ptr & 0b1) && (k == 0 || k == degree) &&
              (j == 0 || j == degree))
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
          else if ((o != 0) && (o_ptr & 0b11111) &&
                   (((k == 0 || k == degree) && ((0 < j) && (j < degree))) ||
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
                     orientation_table[quad_flag][(degree - 1) * (jk - 1) + i]];
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
          else if ((o != 0) && (o_ptr & 0b111111) && (0 < k) && (k < degree) &&
                   (0 < j) && (j < degree))
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

template <typename Number>
void
gather_post(const std::vector<Number> &      global_vector,
            const unsigned int               degree,
            const std::vector<unsigned int> &dofs_of_cell,
            const std::vector<unsigned int> &orientations, // TODO: compress
            const Table<2, unsigned int> &   orientation_table,
            std::vector<Number> &            local_vector)
{
  std::vector<std::pair<unsigned int, unsigned int>> orientations_{
    // lines - x
    {1, orientations[2]},
    {1, orientations[3]},
    {1, orientations[6]},
    {1, orientations[7]},
    // lines - y
    {1, orientations[0]},
    {1, orientations[1]},
    {1, orientations[4]},
    {1, orientations[5]},
    // lines - z
    {1, orientations[8]},
    {1, orientations[9]},
    {1, orientations[10]},
    {1, orientations[11]},
    // quads
    {3, orientations[12]},
    {3, orientations[13]},
    {3, orientations[14]},
    {3, orientations[15]},
    {3, orientations[16]},
    {3, orientations[17]}};

  unsigned int orientation = 0;

  for (unsigned int i = 0, s = 0; i < orientations_.size(); ++i)
    {
      orientation += orientations_[i].second << s;
      s += orientations_[i].first;
    }

  for (unsigned int k = 0, compressed_k = 0, offset_k = 0, c = 0; k <= degree;
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

  if (orientation != 0)
    {
      for (unsigned int l = 0; l < 12; ++l) // loop over all lines
        {
          if (orientation & 1) // check bit
            {
              // determine stride and begin
              const unsigned int stride = Utilities::pow(degree + 1, l / 4);

              unsigned int begin = 0;

              if (l == 0)
                begin = 1;
              else if (l == 2)
                begin = (degree + 1) * degree + 1;
              else if (l == 3)
                begin = (degree + 1) * (degree + 1) * degree + 1;
              else if (l == 4)
                begin = (degree + 1) * (degree + 1) * degree +
                        (degree + 1) * degree + 1;
              else if (l == 5)
                begin = (degree + 1);
              else if (l == 6)
                begin = (degree + 1) + degree;
              else if (l == 7)
                begin = (degree + 1) * (degree + 1) * degree + (degree + 1);
              else if (l == 8)
                begin =
                  (degree + 1) * (degree + 1) * degree + (degree + 1) + degree;
              else if (l == 9)
                begin = (degree + 1) * (degree + 1);
              else if (l == 10)
                begin = (degree + 1) * (degree + 1) + degree;
              else if (l == 11)
                begin =
                  (degree + 1) * (degree + 1) * degree + (degree + 1) * degree;
              else if (l == 12)
                begin = (degree + 1) * (degree + 1) * degree +
                        (degree + 1) * degree + degree;

              // perform reorientation
              for (unsigned int i = 0; i < (degree - 1) / 2; ++i)
                std::swap(local_vector[begin + i * stride],
                          local_vector[begin + (degree - 2 - i) * stride]);
            }

          orientation = orientation >> 1; //  go to next bit
        }

      for (unsigned int q = 0; q < 6; ++q) // loop over all quads
        {
          const unsigned int flag = orientation & 0b111;

          if (flag != 0) // check bits
            {
              Number temp[100];

              const unsigned int d      = q / 2;
              const unsigned int stride = (d == 0) ? (degree + 1) : 1;
              const unsigned int stride2 =
                (d == 2) ? (degree + 1) : ((degree + 1) * (degree + 1));

              unsigned int begin = 0;
              if ((q % 2) == 0)
                begin = 0;
              else if (q == 1)
                begin = degree;
              else if (q == 3)
                begin = (degree + 1) * degree;
              else if (q == 5)
                begin = (degree + 1) * (degree + 1) * degree;

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
           degree,
           dofs_of_cell,
           orientations,
           orientation_table,
           local_vector);
  else
    gather_post(global_vector,
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