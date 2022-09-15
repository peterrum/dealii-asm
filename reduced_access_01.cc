#include <array>
#include <iostream>
#include <vector>

int
main()
{
  const unsigned int degree = 4;

  std::vector<std::pair<unsigned int, unsigned int>> dpo;
  dpo.emplace_back(4, 1);
  dpo.emplace_back(4, degree - 1);
  dpo.emplace_back(1, (degree - 1) * (degree - 1));

  std::vector<unsigned int> dofs;

  unsigned int dof_counter = 0;
  for (const auto &entry : dpo)
    for (unsigned int i = 0; i < entry.first; ++i)
      {
        dofs.emplace_back(dof_counter);
        dof_counter += entry.second;
      }

  std::vector<unsigned int> entities_of_cell = {0, 6, 1, 4, 8, 5, 2, 7, 3};

  std::vector<unsigned int> orientations = {1, 0, 0, 0};

  std::vector<unsigned int> dofs_of_cell;

  for (const auto i : entities_of_cell)
    dofs_of_cell.emplace_back(dofs[i]);

  std::vector<double> global_vector;

  for (unsigned int i = 0; i < dof_counter; ++i)
    global_vector.emplace_back(i);

  std::vector<double> local_vector(dof_counter);

  unsigned int counter = 0;

  const auto reorientate_line = [degree](const unsigned int i,
                                         const bool         flag) {
    if (flag)
      return degree - i - 2;
    else
      return i;
  };

  // bottom layer (vertex-line-vertex)
  {
    const bool flag = orientations[2] == 1;

    local_vector[counter++] = global_vector[dofs_of_cell[0]];

    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] =
        global_vector[dofs_of_cell[1] + reorientate_line(i, flag)];

    local_vector[counter++] = global_vector[dofs_of_cell[2]];
  }

  // middle layers (line-quad-line)
  {
    const bool flag0 = orientations[0] == 1;
    const bool flag1 = orientations[1] == 1;

    std::array<unsigned int, 3> counters;
    counters.fill(0);

    for (unsigned int j = 0; j < degree - 1; ++j)
      {
        local_vector[counter++] =
          global_vector[dofs_of_cell[3] +
                        reorientate_line(counters[0]++, flag0)];

        for (unsigned int i = 0; i < degree - 1; ++i)
          local_vector[counter++] =
            global_vector[dofs_of_cell[4] + (counters[1]++)];

        local_vector[counter++] =
          global_vector[dofs_of_cell[5] +
                        reorientate_line(counters[2]++, flag1)];
      }
  }

  // top layer (vertex-line-vertex)
  {
    const bool flag = orientations[3] == 1;

    local_vector[counter++] = global_vector[dofs_of_cell[6]];

    for (unsigned int i = 0; i < degree - 1; ++i)
      local_vector[counter++] =
        global_vector[dofs_of_cell[7] + reorientate_line(i, flag)];

    local_vector[counter++] = global_vector[dofs_of_cell[8]];
  }

  for (unsigned int j = 0, c = 0; j <= degree; ++j)
    {
      for (unsigned int j = 0; j <= degree; ++j, ++c)
        printf("%4.0f", local_vector[c]);

      printf("\n");
    }
}