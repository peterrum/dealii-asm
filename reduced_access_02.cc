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
  (void)orientations; // TODO

  unsigned int counter = 0;

  // bottom layer (k=0)
  {
    // vertex 0
    local_vector[counter++] = global_vector[dofs_of_cell[0]];

    // line 2 (TODO)
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] = global_vector[dofs_of_cell[1] + i];

    // vertex 1
    local_vector[counter++] = global_vector[dofs_of_cell[2]];

    for (unsigned int j = 0, quad_counter = 0; j < degree - 1; ++j)
      {
        // line 0 (TODO)
        local_vector[counter++] = global_vector[dofs_of_cell[3] + j];

        // quad 4 (TODO)
        for (unsigned int i = 0; i < degree - 1; ++i, ++quad_counter)
          local_vector[counter++] =
            global_vector[dofs_of_cell[4] + quad_counter];

        // line 1 (TODO)
        local_vector[counter++] = global_vector[dofs_of_cell[5] + j];
      }

    // vertex 2
    local_vector[counter++] = global_vector[dofs_of_cell[6]];

    // line 3 (TODO)
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] = global_vector[dofs_of_cell[7] + i];

    // vertex 3
    local_vector[counter++] = global_vector[dofs_of_cell[8]];
  }

  // middle layers (0<k<p)
  {}

  // top layer (k=p)
  {
    // vertex 4
    local_vector[counter++] = global_vector[dofs_of_cell[18]];

    // line 6 (TODO)
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] = global_vector[dofs_of_cell[19] + i];

    // vertex 5
    local_vector[counter++] = global_vector[dofs_of_cell[20]];

    for (unsigned int j = 0, quad_counter = 0; j < degree - 1; ++j)
      {
        // line 4 (TODO)
        local_vector[counter++] = global_vector[dofs_of_cell[21] + j];

        // quad 5 (TODO)
        for (unsigned int i = 0; i < degree - 1; ++i, ++quad_counter)
          local_vector[counter++] =
            global_vector[dofs_of_cell[22] + quad_counter];

        // line 5 (TODO)
        local_vector[counter++] = global_vector[dofs_of_cell[23] + j];
      }

    // vertex 6
    local_vector[counter++] = global_vector[dofs_of_cell[24]];

    // line 7 (TODO)
    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] = global_vector[dofs_of_cell[25] + i];

    // vertex 7
    local_vector[counter++] = global_vector[dofs_of_cell[26]];
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