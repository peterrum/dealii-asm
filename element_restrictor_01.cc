#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include "include/dof_tools.h"
#include "include/grid_tools.h"
#include "include/restrictors.h"


using namespace dealii;


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim                  = 2;
  const unsigned int n_global_refinements = 3;
  const unsigned int fe_degree            = 3;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>       fe_q(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  typename Restrictors::ElementCenteredRestrictor<VectorType>::AdditionalData
    restrictor_additional_data;
  restrictor_additional_data.n_overlap      = 2;
  restrictor_additional_data.weighting_type = Restrictors::WeightingType::symm;

  Restrictors::ElementCenteredRestrictor<VectorType> restrictor;
  restrictor.reinit(dof_handler, restrictor_additional_data);

  VectorType src(restrictor.get_partitioner());
  VectorType dst(restrictor.get_partitioner());

  src = 1.0;

  const auto vmult = [&](VectorType &dst, const VectorType &src) {
    dst = 0.0;
    src.update_ghost_values();

    Vector<Number> local_dofs;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int index = cell->active_cell_index();
          restrictor.read_dof_values(index, src, local_dofs);
          restrictor.distribute_dof_values(index, local_dofs, dst);
        }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  };

  vmult(dst, src);

  pcout << dst.size() << " " << dst.l1_norm() << std::endl;
}
