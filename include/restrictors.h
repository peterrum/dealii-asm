#pragma once

#include "dof_tools.h"
#include "grid_tools.h"

namespace Restrictors
{
  enum class WeightingType
  {
    none,
    pre,
    post,
    symm
  };

  template <typename VectorType>
  class ElementCenteredRestrictor
  {
  public:
    using Number = typename VectorType::value_type;

    struct AdditionalData
    {
      AdditionalData(const unsigned int n_overlap      = 1,
                     WeightingType      weighting_type = WeightingType::none)
        : n_overlap(n_overlap)
        , weighting_type(weighting_type)
      {}

      unsigned int  n_overlap;
      WeightingType weighting_type;
    };

    ElementCenteredRestrictor() = default;

    template <int dim>
    ElementCenteredRestrictor(
      const dealii::DoFHandler<dim> &dof_handler,
      const AdditionalData &         additional_data = AdditionalData())
    {
      this->reinit(dof_handler, additional_data);
    }

    template <int dim>
    void
    reinit(const dealii::DoFHandler<dim> &dof_handler,
           const AdditionalData &         additional_data = AdditionalData())
    {
      this->weighting_type = additional_data.weighting_type;

      // 1) compute indices
      {
        this->indices.resize(dof_handler.get_triangulation().n_active_cells());

        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const auto cells =
                dealii::GridTools::extract_all_surrounding_cells_cartesian<dim>(
                  cell, additional_data.n_overlap <= 1 ? 0 : dim);

              this->indices[cell->active_cell_index()] =
                dealii::DoFTools::get_dof_indices_cell_with_overlap(
                  dof_handler, cells, additional_data.n_overlap);
            }
      }

      // 2) create a partitioner compatible with the indices
      {
        std::vector<dealii::types::global_dof_index> ghost_indices_vector;

        for (const auto &i : indices)
          ghost_indices_vector.insert(ghost_indices_vector.end(),
                                      i.begin(),
                                      i.end());

        std::sort(ghost_indices_vector.begin(), ghost_indices_vector.end());
        ghost_indices_vector.erase(std::unique(ghost_indices_vector.begin(),
                                               ghost_indices_vector.end()),
                                   ghost_indices_vector.end());

        dealii::IndexSet ghost_indices(dof_handler.n_dofs());
        ghost_indices.add_indices(ghost_indices_vector.begin(),
                                  ghost_indices_vector.end());

        this->partitioner =
          std::make_shared<dealii::Utilities::MPI::Partitioner>(
            dof_handler.locally_owned_dofs(),
            ghost_indices,
            dof_handler.get_communicator());
      }

      // 3) compute weights
      if (weighting_type != WeightingType::none)
        {
          dealii::LinearAlgebra::distributed::Vector<Number> weight_vector(
            partitioner);

          for (const auto &cell_indices : indices)
            for (const auto &i : cell_indices)
              weight_vector[i] += 1.0;

          weight_vector.compress(dealii::VectorOperation::add);

          for (auto &i : weight_vector)
            i = 1.0 /
                ((weighting_type == WeightingType::symm) ? std::sqrt(i) : i);

          weight_vector.update_ghost_values();

          weights.resize(indices.size());

          for (unsigned int i = 0; i < indices.size(); ++i)
            {
              weights[i].resize(indices[i].size());

              for (unsigned int j = 0; j < indices[i].size(); ++j)
                weights[i][j] = weight_vector[indices[i][j]];
            }
        }
    }

    void
    read_dof_values(const unsigned int      index,
                    const VectorType &      global_vector,
                    dealii::Vector<Number> &local_vector) const
    {
      local_vector.reinit(indices[index].size());

      for (unsigned int i = 0; i < local_vector.size(); ++i)
        {
          const Number weight = (weighting_type == WeightingType::pre ||
                                 weighting_type == WeightingType::symm) ?
                                  weights[index][i] :
                                  1.0;

          local_vector[i] = global_vector[indices[index][i]] * weight;
        }
    }

    void
    distribute_dof_values(const unsigned int            index,
                          const dealii::Vector<Number> &local_vector,
                          VectorType &                  global_vector) const
    {
      AssertDimension(local_vector.size(), indices[index].size());

      for (unsigned int i = 0; i < local_vector.size(); ++i)
        {
          const Number weight = (weighting_type == WeightingType::post ||
                                 weighting_type == WeightingType::symm) ?
                                  weights[index][i] :
                                  1.0;

          global_vector[indices[index][i]] += local_vector[i] * weight;
        }
    }

    const std::shared_ptr<dealii::Utilities::MPI::Partitioner> &
    get_partitioner()
    {
      return this->partitioner;
    }

    const std::vector<std::vector<dealii::types::global_dof_index>> &
    get_indices() const
    {
      return indices;
    }

  private:
    std::shared_ptr<dealii::Utilities::MPI::Partitioner> partitioner;

    std::vector<std::vector<dealii::types::global_dof_index>> indices;
    std::vector<std::vector<Number>>                          weights;

    WeightingType weighting_type;
  };
} // namespace Restrictors
