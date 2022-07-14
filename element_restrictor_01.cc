
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>


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

    void
    reinit(const AdditionalData &additional_data = AdditionalData())
    {
      (void)additional_data;

      this->weighting_type = additional_data.weighting_type;
    }

    template <int dim>
    void
    read_dof_values(const typename dealii::DoFHandler<dim>::cell_iterator &cell,
                    const VectorType &      global_vector,
                    dealii::Vector<Number> &local_vector) const
    {
      const auto index = cell->active_cell_index();

      for (unsigned int i = 0; i < local_vector.size(); ++i)
        {
          const Number weight = (weighting_type == WeightingType::pre ||
                                 weighting_type == WeightingType::symm) ?
                                  weights[index][i] :
                                  1.0;

          local_vector[i] = global_vector[indices[index][i]] * weight;
        }
    }

    template <int dim>
    void
    distribute_dof_values(
      const typename dealii::DoFHandler<dim>::cell_iterator &cell,
      const dealii::Vector<Number> &                         local_vector,
      VectorType &global_vector) const
    {
      const auto index = cell->active_cell_index();

      for (unsigned int i = 0; i < local_vector.size(); ++i)
        {
          const Number weight = (weighting_type == WeightingType::post ||
                                 weighting_type == WeightingType::symm) ?
                                  weights[index][i] :
                                  1.0;

          global_vector[indices[index][i]] += local_vector[i] * weight;
        }
    }

  private:
    std::vector<std::vector<dealii::types::global_dof_index>> indices;
    std::vector<std::vector<Number>>                          weights;

    WeightingType weighting_type;
  };
} // namespace Restrictors


using namespace dealii;


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim = 2;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  typename Restrictors::ElementCenteredRestrictor<VectorType>::AdditionalData
    restrictor_additional_data;

  Restrictors::ElementCenteredRestrictor<VectorType> restrictor;
  restrictor.reinit(restrictor_additional_data);

  VectorType src, dst;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  DoFHandler<dim> dof_handler(tria);

  Vector<Number> local_dofs;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      restrictor.template read_dof_values<dim>(cell, src, local_dofs);
      restrictor.template distribute_dof_values<dim>(cell, local_dofs, dst);
    }
}
