#pragma once

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/mapping_q1.h>

namespace MyDoFRenumbering
{
  using namespace dealii;

  // Compute a vector of lists with number of unknowns of the same category
  // in terms of influence from other MPI processes, starting with unknowns
  // touched only by the local process and finally a new set of indices
  // going to different MPI neighbors. Later passes of the algorithm will
  // re-order unknowns within each of these sets.
  std::vector<std::vector<unsigned int>>
  group_dofs_by_rank_access(
    const dealii::Utilities::MPI::Partitioner &partitioner)
  {
    // count the number of times a locally owned DoF is accessed by the
    // remote ghost data
    std::vector<unsigned int> touch_count(partitioner.locally_owned_size());
    for (const auto &p : partitioner.import_indices())
      for (unsigned int i = p.first; i < p.second; ++i)
        touch_count[i]++;

    // category 0: DoFs never touched by ghosts
    std::vector<std::vector<unsigned int>> result(1);
    for (unsigned int i = 0; i < touch_count.size(); ++i)
      if (touch_count[i] == 0)
        result.back().push_back(i);

    // DoFs with 1 appearance can be simply categorized by their (single)
    // MPI rank, whereas we need to go an extra round for the remaining DoFs
    // by collecting the owning processes by hand
    std::map<unsigned int, std::vector<unsigned int>> multiple_ranks_access_dof;
    const std::vector<std::pair<unsigned int, unsigned int>> &import_targets =
      partitioner.import_targets();
    auto it = partitioner.import_indices().begin();
    for (const std::pair<unsigned int, unsigned int> &proc : import_targets)
      {
        result.emplace_back();
        unsigned int count_dofs = 0;
        while (count_dofs < proc.second)
          {
            for (unsigned int i = it->first; i < it->second; ++i, ++count_dofs)
              {
                if (touch_count[i] == 1)
                  result.back().push_back(i);
                else
                  multiple_ranks_access_dof[i].push_back(proc.first);
              }
            ++it;
          }
      }
    Assert(it == partitioner.import_indices().end(), ExcInternalError());

    // Now go from the computed map on DoFs to a map on the processes for
    // DoFs with multiple owners, and append the DoFs found this way to our
    // global list
    std::map<std::vector<unsigned int>,
             std::vector<unsigned int>,
             std::function<bool(const std::vector<unsigned int> &,
                                const std::vector<unsigned int> &)>>
      dofs_by_rank{[](const std::vector<unsigned int> &a,
                      const std::vector<unsigned int> &b) {
        if (a.size() < b.size())
          return true;
        if (a.size() == b.size())
          {
            for (unsigned int i = 0; i < a.size(); ++i)
              if (a[i] < b[i])
                return true;
              else if (a[i] > b[i])
                return false;
          }
        return false;
      }};
    for (const auto &entry : multiple_ranks_access_dof)
      dofs_by_rank[entry.second].push_back(entry.first);

    for (const auto &procs : dofs_by_rank)
      result.push_back(procs.second);

    return result;
  }



  // Compute two vectors, the first indicating the best numbers for a
  // MatrixFree::cell_loop and the second the count of how often a DoF is
  // touched by different cell groups, in order to later figure out DoFs
  // with far reach and those with only local influence.
  template <int dim, typename Number, typename VectorizedArrayType>
  std::pair<std::vector<unsigned int>, std::vector<unsigned char>>
  compute_mf_numbering(
    const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
    const unsigned int                                          component,
    const std::function<std::vector<types::global_dof_index>(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &)> &collect_indices)
  {
    const IndexSet &owned_dofs = matrix_free.get_dof_info(component)
                                   .vector_partitioner->locally_owned_range();
    const unsigned int n_comp =
      matrix_free.get_dof_handler(component).get_fe().n_components();
    Assert(matrix_free.get_dof_handler(component).get_fe().n_base_elements() ==
             1,
           ExcNotImplemented());
    Assert(dynamic_cast<const FE_Q_Base<dim> *>(
             &matrix_free.get_dof_handler(component).get_fe().base_element(0)),
           ExcNotImplemented("Matrix-free renumbering only works for "
                             "FE_Q elements"));

    const unsigned int fe_degree =
      matrix_free.get_dof_handler(component).get_fe().degree;
    const unsigned int nn = fe_degree - 1;

    // Data structure used further down for the identification of various
    // entities in the hierarchical numbering of unknowns. The first number
    // indicates the offset from which a given object starts its range of
    // numbers in the hierarchical DoF numbering of FE_Q, and the second the
    // number of unknowns per component on that particular component. The
    // numbers are group by the 3^dim possible objects, listed in
    // lexicographic order.
    std::array<std::pair<unsigned int, unsigned int>,
               dealii::Utilities::pow(3, dim)>
      dofs_on_objects;
    if (dim == 1)
      {
        dofs_on_objects[0] = std::make_pair(0U, 1U);
        dofs_on_objects[1] = std::make_pair(2 * n_comp, nn);
        dofs_on_objects[2] = std::make_pair(n_comp, 1U);
      }
    else if (dim == 2)
      {
        dofs_on_objects[0] = std::make_pair(0U, 1U);
        dofs_on_objects[1] = std::make_pair(n_comp * (4 + 2 * nn), nn);
        dofs_on_objects[2] = std::make_pair(n_comp, 1U);
        dofs_on_objects[3] = std::make_pair(n_comp * 4, nn);
        dofs_on_objects[4] = std::make_pair(n_comp * (4 + 4 * nn), nn * nn);
        dofs_on_objects[5] = std::make_pair(n_comp * (4 + 1 * nn), nn);
        dofs_on_objects[6] = std::make_pair(2 * n_comp, 1U);
        dofs_on_objects[7] = std::make_pair(n_comp * (4 + 3 * nn), nn);
        dofs_on_objects[8] = std::make_pair(3 * n_comp, 1U);
      }
    else if (dim == 3)
      {
        dofs_on_objects[0] = std::make_pair(0U, 1U);
        dofs_on_objects[1] = std::make_pair(n_comp * (8 + 2 * nn), nn);
        dofs_on_objects[2] = std::make_pair(n_comp, 1U);
        dofs_on_objects[3] = std::make_pair(n_comp * 8, nn);
        dofs_on_objects[4] =
          std::make_pair(n_comp * (8 + 12 * nn + 4 * nn * nn), nn * nn);
        dofs_on_objects[5] = std::make_pair(n_comp * (8 + 1 * nn), nn);
        dofs_on_objects[6] = std::make_pair(n_comp * 2, 1U);
        dofs_on_objects[7] = std::make_pair(n_comp * (8 + 3 * nn), nn);
        dofs_on_objects[8] = std::make_pair(n_comp * 3, 1U);
        dofs_on_objects[9] = std::make_pair(n_comp * (8 + 8 * nn), nn);
        dofs_on_objects[10] =
          std::make_pair(n_comp * (8 + 12 * nn + 2 * nn * nn), nn * nn);
        dofs_on_objects[11] = std::make_pair(n_comp * (8 + 9 * nn), nn);
        dofs_on_objects[12] = std::make_pair(n_comp * (8 + 12 * nn), nn * nn);
        dofs_on_objects[13] =
          std::make_pair(n_comp * (8 + 12 * nn + 6 * nn * nn), nn * nn * nn);
        dofs_on_objects[14] =
          std::make_pair(n_comp * (8 + 12 * nn + 1 * nn * nn), nn * nn);
        dofs_on_objects[15] = std::make_pair(n_comp * (8 + 10 * nn), nn);
        dofs_on_objects[16] =
          std::make_pair(n_comp * (8 + 12 * nn + 3 * nn * nn), nn * nn);
        dofs_on_objects[17] = std::make_pair(n_comp * (8 + 11 * nn), nn);
        dofs_on_objects[18] = std::make_pair(n_comp * 4, 1U);
        dofs_on_objects[19] = std::make_pair(n_comp * (8 + 6 * nn), nn);
        dofs_on_objects[20] = std::make_pair(n_comp * 5, 1U);
        dofs_on_objects[21] = std::make_pair(n_comp * (8 + 4 * nn), nn);
        dofs_on_objects[22] =
          std::make_pair(n_comp * (8 + 12 * nn + 5 * nn * nn), nn * nn);
        dofs_on_objects[23] = std::make_pair(n_comp * (8 + 5 * nn), nn);
        dofs_on_objects[24] = std::make_pair(n_comp * 6, 1U);
        dofs_on_objects[25] = std::make_pair(n_comp * (8 + 7 * nn), nn);
        dofs_on_objects[26] = std::make_pair(n_comp * 7, 1U);
      }

    const auto renumber_func = [](const types::global_dof_index dof_index,
                                  const IndexSet &              owned_dofs,
                                  std::vector<unsigned int> &   result,
                                  unsigned int &counter_dof_numbers) {
      const types::global_dof_index local_dof_index =
        owned_dofs.index_within_set(dof_index);
      if (local_dof_index != numbers::invalid_dof_index)
        {
          AssertIndexRange(local_dof_index, result.size());
          if (result[local_dof_index] == numbers::invalid_unsigned_int)
            result[local_dof_index] = counter_dof_numbers++;
        }
    };

    unsigned int                         counter_dof_numbers = 0;
    std::vector<unsigned int>            dofs_extracted;
    std::vector<types::global_dof_index> dof_indices(
      matrix_free.get_dof_handler(component).get_fe().dofs_per_cell);

    // We now define a lambda function that does two things: (a) determine
    // DoF numbers in a way that fit with the order a MatrixFree loop
    // travels through the cells (variable 'dof_numbers_mf_order'), and (b)
    // determine which unknowns are only touched from within a single range
    // of cells and which ones span multiple ranges (variable
    // 'touch_count'). Note that this process is done by calling into
    // MatrixFree::cell_loop, which gives the right level of granularity
    // (when executed in serial) for the scheduled vector operations. Note
    // that we pick the unconstrained indices in the hierarchical order for
    // part (a) as this makes it easy to identify the DoFs on the individual
    // entities, whereas we pick the numbers with constraints eliminated for
    // part (b). For the latter, we keep track of each DoF's interaction
    // with different ranges of cell batches, i.e., call-backs into the
    // operation_on_cell_range() function.
    const unsigned int        n_owned_dofs = owned_dofs.n_elements();
    std::vector<unsigned int> dof_numbers_mf_order(
      n_owned_dofs, dealii::numbers::invalid_unsigned_int);
    std::vector<unsigned int> last_touch_by_cell_batch_range(
      n_owned_dofs, dealii::numbers::invalid_unsigned_int);
    std::vector<unsigned char> touch_count(n_owned_dofs);


    const auto resolve_constraint = [&](auto &i) {
      if (i == numbers::invalid_dof_index)
        return;

      const auto *entries_ptr =
        matrix_free.get_affine_constraints().get_constraint_entries(i);

      if (entries_ptr != nullptr)
        {
          const auto &                  entries   = *entries_ptr;
          const types::global_dof_index n_entries = entries.size();
          if (n_entries == 1 && std::abs(entries[0].second - 1.) <
                                  100 * std::numeric_limits<double>::epsilon())
            {
              i = entries[0].first; // identity constraint
            }
          else if (n_entries == 0)
            {
              i = numbers::invalid_dof_index; // homogeneous
                                              // Dirichlet
            }
          else
            {
              // other constraints, e.g., hanging-node
              // constraints; not implemented yet
              AssertThrow(false, ExcNotImplemented());
            }
        }
      else
        {
          // not constrained -> nothing to do
        }
    };

    const auto operation_on_cell_range =
      [&](const MatrixFree<dim, Number, VectorizedArrayType> &data,
          unsigned int &,
          const unsigned int &,
          const std::pair<unsigned int, unsigned int> &cell_range) {
        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            // part (a): assign beneficial numbers
            for (unsigned int v = 0;
                 v < data.n_active_entries_per_cell_batch(cell);
                 ++v)
              {
                // get the indices for the dofs in cell_batch
                if (data.get_mg_level() == numbers::invalid_unsigned_int)
                  data.get_cell_iterator(cell, v, component)
                    ->get_dof_indices(dof_indices);
                else
                  data.get_cell_iterator(cell, v, component)
                    ->get_mg_dof_indices(dof_indices);

                for (unsigned int a = 0; a < dofs_on_objects.size(); ++a)
                  {
                    const auto &r = dofs_on_objects[a];
                    if (a == 10 || a == 16)
                      // switch order x-z for y faces in 3d to lexicographic
                      // layout
                      for (unsigned int i1 = 0; i1 < nn; ++i1)
                        for (unsigned int i0 = 0; i0 < nn; ++i0)
                          for (unsigned int c = 0; c < n_comp; ++c)
                            renumber_func(dof_indices[r.first + r.second * c +
                                                      i1 + i0 * nn],
                                          owned_dofs,
                                          dof_numbers_mf_order,
                                          counter_dof_numbers);
                    else
                      for (unsigned int i = 0; i < r.second; ++i)
                        for (unsigned int c = 0; c < n_comp; ++c)
                          renumber_func(dof_indices[r.first + r.second * c + i],
                                        owned_dofs,
                                        dof_numbers_mf_order,
                                        counter_dof_numbers);
                  }
              }
          }

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            // part (b): increment the touch count of a dof appearing in the
            // current cell batch if it was last touched by another than the
            // present cell batch range (we track them via storing the last
            // cell batch range that touched a particular dof)
            dofs_extracted.clear();

            for (unsigned int v = 0;
                 v < data.n_active_entries_per_cell_batch(cell);
                 ++v)
              {
                // create iterator
                const auto cell_iterator =
                  matrix_free.get_cell_iterator(cell, v);

                // collect indices
                std::vector<types::global_dof_index> dof_indices =
                  collect_indices(cell_iterator);

                // resolve constraints
                for (auto &i : dof_indices)
                  resolve_constraint(i);

                // global to local
                for (const auto i : dof_indices)
                  if (owned_dofs.is_element(i) &&
                      (i != numbers::invalid_dof_index))
                    dofs_extracted.push_back(owned_dofs.index_within_set(i));
              }

            for (unsigned int dof_index : dofs_extracted)
              if (dof_index < n_owned_dofs &&
                  last_touch_by_cell_batch_range[dof_index] != cell_range.first)
                {
                  ++touch_count[dof_index];
                  last_touch_by_cell_batch_range[dof_index] = cell_range.first;
                }
          }
      };

    // Finally run the matrix-free loop.
    Assert(matrix_free.get_task_info().scheme ==
             dealii::internal::MatrixFreeFunctions::TaskInfo::none,
           ExcNotImplemented("Renumbering only available for non-threaded "
                             "version of MatrixFree::cell_loop"));

    matrix_free.template cell_loop<unsigned int, unsigned int>(
      operation_on_cell_range, counter_dof_numbers, counter_dof_numbers);

    AssertDimension(counter_dof_numbers, n_owned_dofs);

    return std::make_pair(dof_numbers_mf_order, touch_count);
  }



  template <int dim,
            int spacedim,
            typename Number,
            typename VectorizedArrayType>
  std::vector<types::global_dof_index>
  compute_matrix_free_data_locality(
    const DoFHandler<dim, spacedim> &                           dof_handler,
    const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
    const std::function<std::vector<types::global_dof_index>(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &)> &collect_indices)
  {
    Assert(matrix_free.indices_initialized(),
           ExcMessage("You need to set up indices in MatrixFree "
                      "to be able to compute a renumbering!"));

    // try to locate the `DoFHandler` within the given MatrixFree object.
    unsigned int component = 0;
    for (; component < matrix_free.n_components(); ++component)
      if (&matrix_free.get_dof_handler(component) == &dof_handler)
        break;

    Assert(component < matrix_free.n_components(),
           ExcMessage("Could not locate the given DoFHandler in MatrixFree"));

    // Summary of the algorithm below:
    // (a) renumber each DoF in order the corresponding object appears in the
    //     mf loop
    // (b) determine by how many cell groups (call-back places in the loop) a
    //     dof is touched -> first type of category
    // (c) determine by how many MPI processes a dof is touched -> second type
    //     of category
    // (d) combine both category types (second, first) and sort the indices
    //     according to this new category type but also keeping the order
    //     within the other category.

    const std::vector<std::vector<unsigned int>> dofs_by_rank_access =
      group_dofs_by_rank_access(
        *matrix_free.get_dof_info(component).vector_partitioner);

    const std::pair<std::vector<unsigned int>, std::vector<unsigned char>>
      local_numbering =
        compute_mf_numbering<dim>(matrix_free, component, collect_indices);

    // Now construct the new numbering
    const IndexSet &owned_dofs = matrix_free.get_dof_info(component)
                                   .vector_partitioner->locally_owned_range();
    std::vector<unsigned int> new_numbers;
    new_numbers.reserve(owned_dofs.n_elements());

    // step 1: Take all DoFs with reference only to the current MPI process
    // and touched once ("perfect overlap" case). We define a custom
    // comparator for std::sort to then order the unknowns by the specified
    // matrix-free loop order
    const std::vector<unsigned int> &purely_local_dofs = dofs_by_rank_access[0];
    for (unsigned int i : purely_local_dofs)
      if (local_numbering.second[i] == 1)
        new_numbers.push_back(i);

    const auto comparator = [&](const unsigned int a, const unsigned int b) {
      return (local_numbering.first[a] < local_numbering.first[b]);
    };

    std::sort(new_numbers.begin(), new_numbers.end(), comparator);
    unsigned int sorted_size = new_numbers.size();

    // step 2: Take all DoFs with reference to only the current MPI process
    // and touched multiple times (more strain on caches).
    for (auto i : purely_local_dofs)
      if (local_numbering.second[i] > 1)
        new_numbers.push_back(i);
    std::sort(new_numbers.begin() + sorted_size, new_numbers.end(), comparator);
    sorted_size = new_numbers.size();

    // step 3: Get all DoFs with reference from other MPI ranks
    for (unsigned int chunk = 1; chunk < dofs_by_rank_access.size(); ++chunk)
      for (auto i : dofs_by_rank_access[chunk])
        new_numbers.push_back(i);
    std::sort(new_numbers.begin() + sorted_size, new_numbers.end(), comparator);
    sorted_size = new_numbers.size();

    // step 4: Put all DoFs without any reference (constrained DoFs)
    for (auto i : purely_local_dofs)
      if (local_numbering.second[i] == 0)
        new_numbers.push_back(i);
    std::sort(new_numbers.begin() + sorted_size, new_numbers.end(), comparator);

    AssertDimension(new_numbers.size(), owned_dofs.n_elements());

    std::vector<dealii::types::global_dof_index> new_global_numbers(
      owned_dofs.n_elements());
    for (unsigned int i = 0; i < new_numbers.size(); ++i)
      new_global_numbers[new_numbers[i]] = owned_dofs.nth_index_in_set(i);

    return new_global_numbers;
  }



  template <int dim, int spacedim, typename Number, typename AdditionalDataType>
  std::vector<types::global_dof_index>
  compute_matrix_free_data_locality(
    const DoFHandler<dim, spacedim> &dof_handler,
    const AffineConstraints<Number> &constraints,
    const AdditionalDataType &       matrix_free_data,
    const std::function<std::vector<types::global_dof_index>(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &)> &collect_indices)
  {
    AdditionalDataType my_mf_data    = matrix_free_data;
    my_mf_data.initialize_mapping    = false;
    my_mf_data.tasks_parallel_scheme = AdditionalDataType::none;

    typename AdditionalDataType::MatrixFreeType separate_matrix_free;
    separate_matrix_free.reinit(dealii::MappingQ1<dim>(),
                                dof_handler,
                                constraints,
                                dealii::QGauss<1>(2),
                                my_mf_data);
    return compute_matrix_free_data_locality<dim>(dof_handler,
                                                  separate_matrix_free,
                                                  collect_indices);
  }



  template <int dim, int spacedim, typename Number, typename AdditionalDataType>
  void
  matrix_free_data_locality(
    DoFHandler<dim, spacedim> &      dof_handler,
    const AffineConstraints<Number> &constraints,
    const AdditionalDataType &       matrix_free_data,
    const std::function<std::vector<types::global_dof_index>(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &)> &collect_indices)
  {
    const std::vector<types::global_dof_index> new_global_numbers =
      compute_matrix_free_data_locality<dim>(dof_handler,
                                             constraints,
                                             matrix_free_data,
                                             collect_indices);
    if (matrix_free_data.mg_level == dealii::numbers::invalid_unsigned_int)
      dof_handler.renumber_dofs(new_global_numbers);
    else
      dof_handler.renumber_dofs(matrix_free_data.mg_level, new_global_numbers);
  }

} // namespace MyDoFRenumbering
