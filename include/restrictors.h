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
      AdditionalData(const unsigned int  n_overlap      = 1,
                     const WeightingType weighting_type = WeightingType::none,
                     const std::string   type           = "element")
        : n_overlap(n_overlap)
        , weighting_type(weighting_type)
        , type(type)
      {}

      unsigned int  n_overlap;
      WeightingType weighting_type;
      std::string   type;
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
      this->n_overlap      = additional_data.n_overlap;
      this->weighting_type = additional_data.weighting_type;

      AssertDimension(dof_handler.get_fe_collection().size(), 1);
      AssertIndexRange(additional_data.n_overlap,
                       dof_handler.get_fe().tensor_degree() + 2);

      // 1) compute indices
      if (additional_data.type == "element")
        {
          this->indices.resize(
            dof_handler.get_triangulation().n_active_cells());

          for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto cells =
                  dealii::GridTools::extract_all_surrounding_cells_cartesian<
                    dim>(cell, additional_data.n_overlap <= 1 ? 0 : dim);

                this->indices[cell->active_cell_index()] =
                  dealii::DoFTools::get_dof_indices_cell_with_overlap(
                    dof_handler, cells, additional_data.n_overlap);
              }
        }
      else if (additional_data.type == "vertex" ||
               additional_data.type == "vertex_all")
        {
          this->indices.clear();

          const auto &tria = dof_handler.get_triangulation();
          const auto &fe   = dof_handler.get_fe();

          // create map: vertices -> entities
          std::vector<std::array<std::set<unsigned int>, dim>>
            vertices_to_entities(tria.n_vertices());
          std::vector<std::pair<unsigned int, unsigned int>>
            acitve_cell_to_cell_level_index(tria.n_active_cells());

          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_artificial() == false)
              {
                acitve_cell_to_cell_level_index[cell->active_cell_index()] = {
                  cell->level(), cell->index()};

                if (dim >= 2)
                  {
                    for (const auto l : cell->line_indices())
                      {
                        const auto line = cell->line(l);

                        for (const auto v : line->vertex_indices())
                          vertices_to_entities[line->vertex_index(v)][0].insert(
                            line->index());
                      }
                  }

                if (dim == 3)
                  {
                    for (const auto l : cell->face_indices())
                      {
                        const auto face = cell->face(l);

                        for (const auto v : face->vertex_indices())
                          vertices_to_entities[face->vertex_index(v)][1].insert(
                            face->index());
                      }
                  }

                for (const auto v : cell->vertex_indices())
                  vertices_to_entities[cell->vertex_index(v)][dim - 1].insert(
                    cell->active_cell_index());
              }

          for (unsigned int i = 0; i < vertices_to_entities.size(); ++i)
            {
              const auto &vertex_to_entities = vertices_to_entities[i];

              if (vertex_to_entities[dim - 1].size() == 0)
                continue; // vertex is not associated to a patch

              if (additional_data.type == "vertex")
                if (vertex_to_entities[dim - 1].size() !=
                    dealii::Utilities::pow(2, dim))
                  continue; // patch is not complete

              std::set<unsigned int> ranks;

              for (const auto &c : vertex_to_entities[dim - 1])
                {
                  const auto [level, index] =
                    acitve_cell_to_cell_level_index[c];

                  dealii::DoFAccessor<dim, dim, dim, false> cell(&tria,
                                                                 level,
                                                                 index,
                                                                 &dof_handler);

                  ranks.insert(cell.subdomain_id());
                }

              if (*ranks.begin() != dealii::Utilities::MPI::this_mpi_process(
                                      dof_handler.get_communicator()))
                continue;

              std::vector<dealii::types::global_dof_index> indices_all;
              std::vector<dealii::types::global_dof_index> indices;

              {
                dealii::DoFAccessor<0, dim, dim, false> vertex(&tria,
                                                               0,
                                                               i,
                                                               &dof_handler);

                for (unsigned int i = 0; i < fe.n_dofs_per_vertex(); ++i)
                  indices_all.push_back(vertex.vertex_dof_index(0, i));
              }

              if (dim >= 2) // process lines
                for (const auto &l : vertex_to_entities[0])
                  {
                    dealii::DoFAccessor<1, dim, dim, false> line(&tria,
                                                                 0,
                                                                 l,
                                                                 &dof_handler);

                    const unsigned int offset =
                      fe.n_dofs_per_vertex() * line.n_vertices();

                    indices.resize(fe.n_dofs_per_line() + offset);
                    line.get_dof_indices(indices);

                    for (unsigned int i = offset; i < indices.size(); ++i)
                      indices_all.push_back(indices[i]);
                  }

              if (dim == 3) // process quads
                for (const auto &q : vertex_to_entities[1])
                  {
                    dealii::DoFAccessor<2, dim, dim, false> quad(&tria,
                                                                 0,
                                                                 q,
                                                                 &dof_handler);

                    const unsigned int offset =
                      fe.n_dofs_per_vertex() * quad.n_vertices() +
                      fe.n_dofs_per_line() + quad.n_lines();

                    indices.resize(fe.n_dofs_per_line() + offset);
                    quad.get_dof_indices(indices);

                    for (unsigned int i = offset; i < indices.size(); ++i)
                      indices_all.push_back(indices[i]);
                  }


              // process cell
              for (const auto &c : vertex_to_entities[dim - 1])
                {
                  const auto [level, index] =
                    acitve_cell_to_cell_level_index[c];

                  dealii::DoFAccessor<dim, dim, dim, false> cell(&tria,
                                                                 level,
                                                                 index,
                                                                 &dof_handler);

                  unsigned int offset =
                    fe.n_dofs_per_vertex() * cell.n_vertices();

                  if (dim >= 2)
                    offset += fe.n_dofs_per_line() * cell.n_lines();
                  if (dim == 3)
                    offset += fe.n_dofs_per_quad() * cell.n_faces();

                  indices.resize(fe.n_dofs_per_cell());
                  cell.get_dof_indices(indices);

                  for (unsigned int i = offset; i < indices.size(); ++i)
                    indices_all.push_back(indices[i]);
                }

              std::sort(indices_all.begin(), indices_all.end());
              indices_all.erase(std::unique(indices_all.begin(),
                                            indices_all.end()),
                                indices_all.end());

              if (indices_all.size() > 0)
                this->indices.push_back(indices_all);
            }
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
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
    get_partitioner() const
    {
      return this->partitioner;
    }

    const std::vector<std::vector<dealii::types::global_dof_index>> &
    get_indices() const
    {
      return indices;
    }

    unsigned int
    get_n_overlap() const
    {
      return n_overlap;
    }

    unsigned int
    n_blocks() const
    {
      return indices.size();
    }

    unsigned int
    n_entries(const unsigned int c) const
    {
      return indices[c].size();
    }

  private:
    std::shared_ptr<dealii::Utilities::MPI::Partitioner> partitioner;

    std::vector<std::vector<dealii::types::global_dof_index>> indices;
    std::vector<std::vector<Number>>                          weights;

    WeightingType weighting_type;
    unsigned int  n_overlap;
  };
} // namespace Restrictors
