#include <deal.II/base/exceptions.h>

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
       std::vector<Number> &            local_vector)
{
  // helper function to reorientate indices on line
  const auto reorientate_line = [degree](const unsigned int i,
                                         const bool         flag) {
    if (flag)
      return degree - i - 2;
    else
      return i;
  };

  unsigned int counter = 0;

  // bottom layer (j=0; vertex-line-vertex)
  {
    const bool flag = orientations[2] == 1;

    // vertex 0
    local_vector[counter++] = global_vector[dofs_of_cell[0]];

    // line 2
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] =
        global_vector[dofs_of_cell[1] + reorientate_line(i, flag)];

    // vertex 1
    local_vector[counter++] = global_vector[dofs_of_cell[2]];
  }

  // middle layers (0<j<p; line-quad-line)
  {
    const bool flag0 = orientations[0] == 1;
    const bool flag1 = orientations[1] == 1;

    for (unsigned int j = 0, quad_counter = 0; j < degree - 1; ++j)
      {
        // line 0
        local_vector[counter++] =
          global_vector[dofs_of_cell[3] + reorientate_line(j, flag0)];

        // quad 0
        for (unsigned int i = 0; i < degree - 1; ++i, ++quad_counter)
          local_vector[counter++] =
            global_vector[dofs_of_cell[4] + quad_counter];

        // line 1
        local_vector[counter++] =
          global_vector[dofs_of_cell[5] + reorientate_line(j, flag1)];
      }
  }

  // top layer (j=p; vertex-line-vertex)
  {
    const bool flag = orientations[3] == 1;

    // vertex 2
    local_vector[counter++] = global_vector[dofs_of_cell[6]];

    // line 3
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] =
        global_vector[dofs_of_cell[7] + reorientate_line(i, flag)];

    // vertex 3
    local_vector[counter++] = global_vector[dofs_of_cell[8]];
  }
}

int
main(int argc, char *argv[])
{
  AssertThrow(argc == 6, ExcNotImplemented());

  const unsigned int degree = atoi(argv[1]);

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

  gather(global_vector, degree, dofs_of_cell, orientations, local_vector);

  for (unsigned int j = 0, c = 0; j <= degree; ++j)
    {
      for (unsigned int j = 0; j <= degree; ++j, ++c)
        printf("%4.0f", local_vector[c]);

      printf("\n");
    }
}