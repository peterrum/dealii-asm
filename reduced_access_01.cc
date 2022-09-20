#include <deal.II/base/exceptions.h>

#include <deal.II/matrix_free/shape_info.h>

#include <array>
#include <iostream>
#include <vector>

using namespace dealii;

#include "reduced_access.h"

/**
 * ./reduced_access_01 3   0 0 0 1
 */
int
main(int argc, char *argv[])
{
  AssertThrow(argc == 7, ExcNotImplemented());

  for (int i = 0; i < argc; ++i)
    std::cout << std::string(argv[i]) << " ";
  std::cout << std::endl << std::endl;

  const unsigned int dim     = 2;
  const unsigned int degree  = atoi(argv[1]);
  const bool         do_post = atoi(argv[2]);

  std::vector<unsigned int> orientations(4);

  for (unsigned int i = 0; i < 4; ++i)
    orientations[i] = atoi(argv[2 + i]);

  // setup dpo object
  std::vector<std::pair<unsigned int, unsigned int>> dpo;
  dpo.emplace_back(4, 1);
  dpo.emplace_back(4, degree - 1);
  dpo.emplace_back(1, (degree - 1) * (degree - 1));

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
  std::vector<unsigned int> entities_of_cell = {0, 6, 1, 4, 8, 5, 2, 7, 3};

  std::vector<unsigned int> dofs_of_cell;

  for (const auto i : entities_of_cell)
    dofs_of_cell.emplace_back(dofs[i]);

  // create dummy global vector
  std::vector<double> global_vector;
  for (unsigned int i = 0; i < dof_counter; ++i)
    global_vector.emplace_back(i);

  // gather values and print to terminal
  std::vector<double> local_vector(dof_counter);

  if (do_post == false)
    gather(global_vector,
           dim,
           degree,
           dofs_of_cell,
           compress_orientation(orientations, false),
           {},
           local_vector);
  else
    gather_post(global_vector,
                dim,
                degree,
                dofs_of_cell,
                compress_orientation(orientations, true),
                {},
                local_vector);

  for (unsigned int j = 0, c = 0; j <= degree; ++j)
    {
      for (unsigned int j = 0; j <= degree; ++j, ++c)
        printf("%4.0f", local_vector[c]);

      printf("\n");
    }
  printf("\n");
}