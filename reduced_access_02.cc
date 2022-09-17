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

  unsigned int counter = 0;

  for (unsigned int k = 0; k <= degree; ++k)
    {
      if (k == 0)
        {
          // bottom layer (k=0)

          const bool line_flag_0 = orientations[0] == 1;
          const bool line_flag_1 = orientations[1] == 1;
          const bool line_flag_2 = orientations[2] == 1;
          const bool line_flag_3 = orientations[3] == 1;

          const unsigned int quad_flag_4 = orientations[16];

          // vertex 0
          local_vector[counter++] = global_vector[dofs_of_cell[0]];

          // line 2
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[1] + reorientate_line(i, line_flag_2)];

          // vertex 1
          local_vector[counter++] = global_vector[dofs_of_cell[2]];

          for (unsigned int j = 0; j < degree - 1; ++j)
            {
              // line 0
              local_vector[counter++] =
                global_vector[dofs_of_cell[3] +
                              reorientate_line(j, line_flag_0)];

              // quad 4 (ij)
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[dofs_of_cell[4] +
                                reorientate_quad((degree - 1) * j + i,
                                                 quad_flag_4)];

              // line 1
              local_vector[counter++] =
                global_vector[dofs_of_cell[5] +
                              reorientate_line(j, line_flag_1)];
            }

          // vertex 2
          local_vector[counter++] = global_vector[dofs_of_cell[6]];

          // line 3
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[7] + reorientate_line(i, line_flag_3)];

          // vertex 3
          local_vector[counter++] = global_vector[dofs_of_cell[8]];
        }
      else if ((0 < k) && (k < degree))
        {
          // middle layers (0<k<p)

          const bool line_flag_8  = orientations[8] == 1;
          const bool line_flag_9  = orientations[9] == 1;
          const bool line_flag_10 = orientations[10] == 1;
          const bool line_flag_11 = orientations[11] == 1;

          const unsigned int quad_flag_0 = orientations[12];
          const unsigned int quad_flag_1 = orientations[13];
          const unsigned int quad_flag_2 = orientations[14];
          const unsigned int quad_flag_3 = orientations[15];

          // line 8
          local_vector[counter++] =
            global_vector[dofs_of_cell[9] +
                          reorientate_line(k - 1, line_flag_8)];

          // quad 2 (ik)
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[10] +
                            reorientate_quad((k - 1) * (degree - 1) + i,
                                             quad_flag_2)];

          // line 9
          local_vector[counter++] =
            global_vector[dofs_of_cell[11] +
                          reorientate_line(k - 1, line_flag_9)];

          for (unsigned int j = 0; j < degree - 1; ++j)
            {
              // quad 0 (jk)
              local_vector[counter++] =
                global_vector[dofs_of_cell[12] +
                              reorientate_quad((k - 1) * (degree - 1) + j,
                                               quad_flag_0)];

              // hex 0
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[dofs_of_cell[13] +
                                (k - 1) * (degree - 1) * (degree - 1) +
                                j * (degree - 1) + i];

              // quad 1 (jk)
              local_vector[counter++] =
                global_vector[dofs_of_cell[14] +
                              reorientate_quad((k - 1) * (degree - 1) + j,
                                               quad_flag_1)];
            }

          // line 10
          local_vector[counter++] =
            global_vector[dofs_of_cell[15] +
                          reorientate_line(k - 1, line_flag_10)];

          // quad 3 (ik)
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[16] +
                            reorientate_quad((k - 1) * (degree - 1) + i,
                                             quad_flag_3)];

          // line 11
          local_vector[counter++] =
            global_vector[dofs_of_cell[17] +
                          reorientate_line(k - 1, line_flag_11)];
        }
      else
        {
          // top layer (k=p)

          const bool line_flag_4 = orientations[4] == 1;
          const bool line_flag_5 = orientations[5] == 1;
          const bool line_flag_6 = orientations[6] == 1;
          const bool line_flag_7 = orientations[7] == 1;

          const unsigned int quad_flag_5 = orientations[17];

          // vertex 4
          local_vector[counter++] = global_vector[dofs_of_cell[18]];

          // line 6
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[19] +
                            reorientate_line(i, line_flag_6)];

          // vertex 5
          local_vector[counter++] = global_vector[dofs_of_cell[20]];

          for (unsigned int j = 0; j < degree - 1; ++j)
            {
              // line 4
              local_vector[counter++] =
                global_vector[dofs_of_cell[21] +
                              reorientate_line(j, line_flag_4)];

              // quad 5 (ij)
              for (unsigned int i = 0; i < degree - 1; ++i)
                local_vector[counter++] =
                  global_vector[dofs_of_cell[22] +
                                reorientate_quad((degree - 1) * j + i,
                                                 quad_flag_5)];

              // line 5
              local_vector[counter++] =
                global_vector[dofs_of_cell[23] +
                              reorientate_line(j, line_flag_5)];
            }

          // vertex 6
          local_vector[counter++] = global_vector[dofs_of_cell[24]];

          // line 7
          for (unsigned int i = 0; i < degree - 1; ++i)
            local_vector[counter++] =
              global_vector[dofs_of_cell[25] +
                            reorientate_line(i, line_flag_7)];

          // vertex 7
          local_vector[counter++] = global_vector[dofs_of_cell[26]];
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