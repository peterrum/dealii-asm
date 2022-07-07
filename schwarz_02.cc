#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

#include "include/preconditioners.h"

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() = default;

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)p;
    (void)component;

    return 1.0;
  }

private:
};

template <int dim, int spacedim>
void
compute_active_cell_halo_layer(const Triangulation<dim, spacedim> &tria,
                               std::vector<bool> &                 flags,
                               const unsigned int                  n_layers = 1)
{
  std::map<unsigned int, std::vector<unsigned int>> coinciding_vertex_groups;
  std::map<unsigned int, unsigned int> vertex_to_coinciding_vertex_group;

  GridTools::collect_coinciding_vertices(tria,
                                         coinciding_vertex_groups,
                                         vertex_to_coinciding_vertex_group);


  for (unsigned int i = 0; i < n_layers; ++i)
    {
      std::vector<bool> marked_vertices(tria.n_vertices());

      for (const auto &cell : tria.active_cell_iterators())
        if (flags[cell->active_cell_index()] == true)
          for (const auto v : cell->vertex_indices())
            {
              marked_vertices[cell->vertex_index(v)] = true;
              const auto coinciding_vertex_group =
                vertex_to_coinciding_vertex_group.find(cell->vertex_index(v));
              if (coinciding_vertex_group !=
                  vertex_to_coinciding_vertex_group.end())
                for (const auto &co_vertex : coinciding_vertex_groups.at(
                       coinciding_vertex_group->second))
                  marked_vertices[co_vertex] = true;
            }

      for (const auto &cell : tria.active_cell_iterators())
        if (flags[cell->active_cell_index()] == false)
          for (const auto v : cell->vertex_indices())
            if (marked_vertices[cell->vertex_index(v)])
              {
                flags[cell->active_cell_index()] = true;
                break;
              }
    }
}

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_global_refinements,
     const unsigned int n_halo_layers)
{
  (void)fe_degree;

  const MPI_Comm comm = MPI_COMM_WORLD;

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  // create grid
  parallel::shared::Triangulation<dim> tria(comm);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  // determine halo layers
  std::vector<bool> locally_owned_or_halo_cells(tria.n_active_cells(), false);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      locally_owned_or_halo_cells[cell->active_cell_index()] = true;

  compute_active_cell_halo_layer(tria,
                                 locally_owned_or_halo_cells,
                                 n_halo_layers);

  const auto next_cell = [&](const auto &tria, const auto &cell_in) {
    auto cell = cell_in;

    while (true)
      {
        cell++;

        if (cell == tria.end())
          break;

        if (cell->is_active() &&
            locally_owned_or_halo_cells[cell->active_cell_index()])
          return cell;
      }

    return tria.end();
  };

  // output mesh
  const auto first_cell = [&](const auto &tria) {
    return next_cell(tria, tria.begin());
  };

  Vector<double> ranks(tria.n_active_cells());
  for (const auto &cell : tria.active_cell_iterators())
    ranks[cell->active_cell_index()] = cell->subdomain_id();

  Vector<double> subdomain(tria.n_active_cells());
  subdomain = Utilities::MPI::this_mpi_process(comm);

  DataOut<dim> data_out;
  data_out.attach_triangulation(tria);
  data_out.set_cell_selection(first_cell, next_cell);
  data_out.add_data_vector(ranks, "ranks");
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();
  data_out.write_vtu_in_parallel("mesh.vtu", comm);

  // setup DoFHandler
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  pcout << "System statistics:" << std::endl;
  pcout << " - n cells: " << tria.n_global_active_cells() << std::endl;
  pcout << " - n dofs:  " << dof_handler.n_dofs() << std::endl;
  pcout << std::endl;

  QGauss<dim> quadrature(fe_degree + 1);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // create system matrix
  TrilinosWrappers::SparsityPattern sparsity_pattern;
  sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                          dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  sparsity_pattern,
                                  constraints,
                                  false);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix laplace_matrix;
  laplace_matrix.reinit(sparsity_pattern);

  MatrixCreator::
    create_laplace_matrix<dim, dim, TrilinosWrappers::SparseMatrix>(
      dof_handler, quadrature, laplace_matrix, nullptr, constraints);


  // create vectors
  VectorType solution, rhs;

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_active_dofs(dof_handler),
    dof_handler.get_communicator());

  solution.reinit(partitioner);
  rhs.reinit(partitioner);

  VectorTools::create_right_hand_side(
    dof_handler, quadrature, RightHandSide<dim>(), rhs, constraints);


  if (true)
    {
      DomainPreconditioner<TrilinosWrappers::PreconditionAMG,
                           TrilinosWrappers::SparseMatrix,
                           TrilinosWrappers::SparsityPattern>
        precondition;

      std::vector<types::global_dof_index> ghost_indices;
      std::vector<types::global_dof_index> indices_temp;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() == false &&
            locally_owned_or_halo_cells[cell->active_cell_index()])
          {
            indices_temp.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices_temp);

            ghost_indices.insert(ghost_indices.end(),
                                 indices_temp.begin(),
                                 indices_temp.end());
          }

      std::sort(ghost_indices.begin(), ghost_indices.end());
      ghost_indices.erase(std::unique(ghost_indices.begin(),
                                      ghost_indices.end()),
                          ghost_indices.end());

      IndexSet ghost_indices_is(dof_handler.n_dofs());
      ghost_indices_is.add_indices(ghost_indices.begin(), ghost_indices.end());

      precondition.initialize(laplace_matrix,
                              sparsity_pattern,
                              partitioner->locally_owned_range(),
                              ghost_indices_is);

      ReductionControl reduction_control;

      SolverCG<VectorType> solver_cg(reduction_control);

      solution = 0;
      solver_cg.solve(laplace_matrix, solution, rhs, precondition);

      pcout << " - ASM on partition level with AMG: "
            << reduction_control.last_step() << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int fe_degree = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const unsigned int n_global_refinements =
    (argc >= 4) ? std::atoi(argv[3]) : 6;
  const unsigned int n_halo_layers = (argc >= 5) ? std::atoi(argv[4]) : 1;

  if (dim == 2)
    test<2>(fe_degree, n_global_refinements, n_halo_layers);
  else
    AssertThrow(false, ExcNotImplemented());
}
