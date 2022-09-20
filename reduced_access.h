

unsigned int
get_orientation_line(const std::vector<types::global_dof_index> &dofs,
                     const unsigned int                          degree)
{
  bool flag;

  flag = true;
  for (unsigned int i = 0; i < degree - 1; ++i)
    flag &= (dofs[i] == (dofs[0] + i));

  if (flag)
    return 0;


  return numbers::invalid_unsigned_int;
}

unsigned int
get_orientation_quad(const std::vector<types::global_dof_index> &dofs,
                     const Table<2, unsigned int> &orientation_table)
{
  const auto min = *std::min_element(dofs.begin(), dofs.end());

  for (unsigned int i = 0; i < orientation_table.n_rows(); ++i)
    {
      bool flag = true;

      for (unsigned int j = 0; j < orientation_table.n_cols(); ++j)
        flag &= (dofs[j] == (min + orientation_table[i][j]));

      if (flag)
        return i;
    }

  return numbers::invalid_unsigned_int;
}

unsigned int
compress_orientation(const std::vector<unsigned int> &orientations,
                     const bool                       do_post = false)
{
  if (std::find(orientations.begin(),
                orientations.end(),
                numbers::invalid_unsigned_int) != orientations.end())
    return numbers::invalid_unsigned_int;

  unsigned int orientation = 0;

  if (orientations.size() == 4) // 2D
    {
      for (unsigned int i = 0; i < 4; ++i) // lines
        orientation |= orientations[i] << i;
    }
  else if (orientations.size() == 18) // 3D
    {
      // lines
      if (do_post)
        {
          unsigned int index_shift = 0;
          unsigned int dof_shift   = 0;

          for (const auto i : {2, 3, 6, 7, 0, 1, 4, 5, 8, 9, 10, 11})
            orientation |= orientations[i] << (dof_shift++);

          index_shift += 12;

          for (unsigned int i = 0; i < 6;
               ++i, ++index_shift, dof_shift += 3) // quads
            orientation |= orientations[index_shift] << dof_shift;
        }
      else
        {
          std::vector<std::pair<unsigned int, unsigned int>> orientation_table{
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

          for (unsigned int i = 0, s = 0; i < orientation_table.size(); ++i)
            {
              orientation += orientation_table[i].second << s;
              s += orientation_table[i].first;
            }
        }
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  return orientation;
}

std::pair<unsigned int, std::vector<types::global_dof_index>>
compress_indices(const std::vector<types::global_dof_index> &dofs,
                 const unsigned int                          dim,
                 const unsigned int                          degree)
{
  const auto orientation_table =
    internal::MatrixFreeFunctions::ShapeInfo<double>::compute_orientation_table(
      degree - 1); // TODO

  std::vector<std::pair<unsigned int, unsigned int>> dpo; // TODO

  if (dim == 1)
    {
      dpo.emplace_back(2, 1);
      dpo.emplace_back(1, degree - 1);
    }
  else if (dim == 2)
    {
      dpo.emplace_back(4, 1);
      dpo.emplace_back(4, degree - 1);
      dpo.emplace_back(1, (degree - 1) * (degree - 1));
    }
  else if (dim == 3)
    {
      dpo.emplace_back(8, 1);
      dpo.emplace_back(12, degree - 1);
      dpo.emplace_back(6, (degree - 1) * (degree - 1));
      dpo.emplace_back(1, (degree - 1) * (degree - 1) * (degree - 1));
    }

  std::vector<unsigned int> obj_orientations;
  std::vector<unsigned int> obj_start_indices;

  // loop over all dimension
  for (unsigned int d = 0, dof_counter = 0; d <= dim; ++d)
    {
      const auto entry = dpo[d];

      // loop over all objects of the given dimension
      for (unsigned int i = 0; i < entry.first; ++i)
        {
          // extract indices of object
          std::vector<types::global_dof_index> dofs_of_object(entry.second);
          for (unsigned int j = 0; j < entry.second; ++j)
            dofs_of_object[j] = dofs[dof_counter + j];

          // store minimal index of object
          obj_start_indices.emplace_back(
            *std::min_element(dofs_of_object.begin(), dofs_of_object.end()));

          if (d == 2 && (i == 2 || i == 3)) // reorientate quad 2 + 3 (lex)
            {
              auto dofs_of_object_copy = dofs_of_object;

              for (unsigned int j = 0; j < entry.second; ++j)
                dofs_of_object[j] =
                  dofs_of_object_copy[orientation_table[1][j]];
            }

          if (dim >= 2 && d == 1) // line orientations
            {
              const auto orientation =
                get_orientation_line(dofs_of_object, degree);

              obj_orientations.emplace_back(orientation);
            }
          else if (dim == 3 && d == 2) // quad orientations
            {
              const auto orientation =
                get_orientation_quad(dofs_of_object, orientation_table);

              obj_orientations.emplace_back(orientation);
            }

          dof_counter += entry.second;
        }

      // no compression is possible
      if (dim >= 2 && obj_orientations.back() == numbers::invalid_unsigned_int)
        return {numbers::invalid_unsigned_int, {}};
    }

  // compress indices to a single
  const auto orientation_compressed = compress_orientation(obj_orientations);

  // return orientation and start indices
  return {orientation_compressed, obj_start_indices};
}
