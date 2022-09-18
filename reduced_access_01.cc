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
  unsigned int counter    = 0;
  unsigned int offset     = 0;
  unsigned int compressed = 0;

  unsigned int orientation = 0;        // TODO
  orientation += orientations[2] << 0; //
  orientation += orientations[0] << 1; //
  orientation += orientations[1] << 2; //
  orientation += orientations[3] << 3; //

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
      else if (orientation && (orientation & 0b11) && (0 < j) && (j < degree))
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

template <typename Number>
void
gather_post(const std::vector<Number> &      global_vector,
            const unsigned int               degree,
            const std::vector<unsigned int> &dofs_of_cell,
            const std::vector<unsigned int> &orientations, // TODO: compress
            std::vector<Number> &            local_vector)
{
  unsigned int orientation = 0;        // TODO
  orientation += orientations[0] << 0; //
  orientation += orientations[1] << 1; //
  orientation += orientations[2] << 2; //
  orientation += orientations[3] << 3; //

  for (unsigned int j = 0, c = 0, offset = 0, compressed = 0; j <= degree; ++j)
    {
      const auto indices = dofs_of_cell.begin() + compressed * 3;

      local_vector[c++] = global_vector[indices[0] + offset];

      for (unsigned int i = 0; i < degree - 1; ++i)
        local_vector[c++] =
          global_vector[indices[1] + offset * (degree - 1) + i];

      local_vector[c++] = global_vector[indices[2] + offset];

      if ((j == 0) || (j == (degree - 1)))
        {
          compressed++;
          offset = 0;
        }
      else
        offset++;
    }

  if (orientation != 0)
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

  gather_post(global_vector, degree, dofs_of_cell, orientations, local_vector);

  for (unsigned int j = 0, c = 0; j <= degree; ++j)
    {
      for (unsigned int j = 0; j <= degree; ++j, ++c)
        printf("%4.0f", local_vector[c]);

      printf("\n");
    }
  printf("\n");
}