#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/tensor_product_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/tensor_product_matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "include/grid_tools.h"
#include "include/restrictors.h"

using namespace dealii;

template <int dim>
void
test(const unsigned int fe_degree)
{
  FE_Q<dim> fe(fe_degree);
  FE_Q<1>   fe_1D(fe_degree);

  QGauss<dim>     quadrature(fe_degree + 1);
  QGauss<dim - 1> quadrature_face(fe_degree + 1);
  QGauss<1>       quadrature_1D(fe_degree + 1);

  MappingQ1<dim> mapping;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  for (const auto &face : tria.active_face_iterators())
    if (face->at_boundary())
      face->set_boundary_id(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const auto harmonic_patch_extend =
    GridTools::compute_harmonic_patch_extend(mapping, tria, quadrature_face);

  Restrictors::ElementCenteredRestrictor<Vector<double>>::AdditionalData ad;
  ad.type = "vertex";

  Restrictors::ElementCenteredRestrictor<Vector<double>> restrictor;
  restrictor.reinit(dof_handler, ad);

  std::cout << std::endl;

  for (const auto &cell : tria.active_cell_iterators())
    {
      const auto cell_index = cell->active_cell_index();
      const auto cells_all =
        GridTools::extract_all_surrounding_cells_cartesian<dim>(cell);
      const auto patch_extend_all = harmonic_patch_extend[cell_index];

      std::array<typename Triangulation<dim>::cell_iterator,
                 Utilities::pow(2, dim)>
        cells;

      dealii::ndarray<double, dim, 2> patch_extend;

      unsigned int c = 0;

      for (unsigned int k = 0; k < ((dim == 3) ? 2 : 1); ++k)
        for (unsigned int j = 0; j < ((dim >= 2) ? 2 : 1); ++j)
          for (unsigned int i = 0; i < 2; ++i, ++c)
            cells[4 * k + 2 * j + i] =
              cells_all[9 * ((dim == 3) ? (k + 1) : 0) +
                        3 * ((dim >= 2) ? (j + 1) : 0) + (i + 1)];

      for (unsigned int d = 0; d < dim; ++d)
        {
          patch_extend[d][0] = patch_extend_all[d][1];
          patch_extend[d][1] = patch_extend_all[d][2];
        }

      // FDM
      const auto [M, K] = TensorProductMatrixCreator::
        create_laplace_tensor_product_matrix<dim, double>(fe_1D,
                                                          quadrature_1D,
                                                          patch_extend);

      // indices
      auto indices =
        DoFTools::get_dof_indices_vertex_patch<dim>(dof_handler, cells);

      std::sort(indices.begin(), indices.end());

      for (const auto i : indices)
        std::cout << i << " ";
      std::cout << std::endl;
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const unsigned int dim       = argc > 1 ? atoi(argv[1]) : 2;
  const unsigned int fe_degree = argc > 2 ? atoi(argv[2]) : 2;

  if (dim == 2)
    test<2>(fe_degree);
  else
    AssertThrow(false, ExcNotImplemented());
}
