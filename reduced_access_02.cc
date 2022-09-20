#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/shape_info.h>

#include <array>
#include <bit>
#include <iostream>
#include <vector>

using namespace dealii;

#include "reduced_access.h"

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

  if (false)
    gather(global_vector,
           dim,
           degree,
           dofs_of_cell,
           compress_orientation(orientations, false),
           orientation_table,
           local_vector);
  else
    gather_post(global_vector,
                dim,
                degree,
                dofs_of_cell,
                compress_orientation(orientations, true),
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