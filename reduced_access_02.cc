#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/shape_info.h>

#include <array>
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
  // helper function to reorientate indices on line and ...
  const auto reorientate_line = [degree](const unsigned int i,
                                         const bool         flag) {
    if (flag)
      return degree - i - 2;
    else
      return i;
  };

  // .... on quads
  const auto reorientate_quad =
    [&orientation_table](const unsigned int i, const unsigned int orientation) {
      if (orientation == 0)
        return i;
      else
        return orientation_table[orientation][i];
    };

  std::vector<unsigned int> orientation{orientations[2],
                                        orientations[0],
                                        orientations[16],
                                        orientations[1],
                                        orientations[3],
                                        orientations[8],
                                        orientations[14],
                                        orientations[9],
                                        orientations[12],
                                        orientations[13],
                                        orientations[10],
                                        orientations[15],
                                        orientations[11],
                                        orientations[6],
                                        orientations[4],
                                        orientations[17],
                                        orientations[5],
                                        orientations[7]};

  auto o_ptr = orientation.data();

  unsigned int counter = 0;

  for (unsigned int k = 0, compressed_k = 0; k <= degree; ++k)
    {
      for (unsigned int j = 0, compressed_j = 0; j <= degree; ++j)
        {
          const auto indices =
            dofs_of_cell.begin() + 3 * (compressed_k * 3 + compressed_j);

          if ((k == 0 || k == degree) && (j == 0 || j == degree))
            {
              const bool line_flag = o_ptr[0];

              // vertex
              local_vector[counter++] = global_vector[indices[0]];

              // line
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] + reorientate_line(i, line_flag)];

              // vertex
              local_vector[counter++] = global_vector[indices[2]];
            }
          else if (((k == 0 || k == degree) && ((0 < j) && (j < degree))) ||
                   (((0 < k) && (k < degree)) && (j == 0 || j == degree)))
            {
              const bool         line_flag_0 = o_ptr[0];
              const unsigned int quad_flag   = o_ptr[1];
              const bool         line_flag_1 = o_ptr[2];

              const unsigned int jk = (k == 0 || k == degree) ? j : k;

              // line
              local_vector[counter++] =
                global_vector[indices[0] +
                              reorientate_line(jk - 1, line_flag_0)];

              // quad (ij or ik)
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] +
                                reorientate_quad((degree - 1) * (jk - 1) + i,
                                                 quad_flag)];

              // line
              local_vector[counter++] =
                global_vector[indices[2] +
                              reorientate_line(jk - 1, line_flag_1)];
            }
          else if (((0 < k) && (k < degree)) && ((0 < j) && (j < degree)))
            {
              const unsigned int quad_flag_0 = o_ptr[0];
              const unsigned int quad_flag_1 = o_ptr[1];

              // quad (jk)
              local_vector[counter++] =
                global_vector[indices[0] +
                              reorientate_quad((k - 1) * (degree - 1) + (j - 1),
                                               quad_flag_0)];

              // hex
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[indices[1] +
                                (k - 1) * (degree - 1) * (degree - 1) +
                                (j - 1) * (degree - 1) + i];

              // quad (jk)
              local_vector[counter++] =
                global_vector[indices[2] +
                              reorientate_quad((k - 1) * (degree - 1) + (j - 1),
                                               quad_flag_1)];
            }

          if (j == 0 || j == degree - 1)
            {
              ++compressed_j;
            }
          else
            {}

          if (k == 0 || k == degree)
            {
              if (j == 0 || j == degree)
                o_ptr += 1;
              else if (j == (degree - 1))
                o_ptr += 3;
            }
          else
            {
              if (j == 0 || j == degree)
                o_ptr += 3;
              else if (j == (degree - 1))
                o_ptr += 2;
            }

          if (0 < k && k < (degree - 1) && j == degree)
            o_ptr -= 8;
        }

      if (k == 0 || k == degree - 1)
        {
          ++compressed_k;
        }
      else
        {}
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

  gather(global_vector,
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