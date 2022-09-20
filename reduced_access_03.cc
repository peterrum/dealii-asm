
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>


using namespace dealii;

#include "reduced_access.h"

int
main(int argc, char *argv[])
{
  AssertThrow(argc == 2, ExcNotImplemented());

  {
    const auto quad = ReferenceCells::Quadrilateral;

    std::array<unsigned int, 4> indices_0 = {{0, 1, 2, 3}};
    std::array<unsigned int, 4> indices_1 = {{0, 2, 1, 3}};

    const unsigned int orientation =
      quad.compute_orientation(indices_0, indices_1);

    const auto indices_2 =
      quad.permute_according_orientation(indices_1, orientation);

    std::cout << orientation << std::endl;

    for (const auto i : indices_2)
      std::cout << i << " ";
    std::cout << std::endl;
  }

  {
    const unsigned int degree = std::atoi(argv[1]);
    const unsigned int dim    = 3;

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria);

    FE_Q<dim> fe(degree);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);


    // setup dpo object
    std::vector<std::pair<unsigned int, unsigned int>> dpo;
    dpo.emplace_back(8, 1);
    dpo.emplace_back(12, degree - 1);
    dpo.emplace_back(6, (degree - 1) * (degree - 1));
    dpo.emplace_back(1, (degree - 1) * (degree - 1) * (degree - 1));

    const auto print_indices = [&]() {
      const auto                           cell = dof_handler.begin();
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      cell->get_dof_indices(dofs);

      const auto orientation_table = internal::MatrixFreeFunctions::ShapeInfo<
        double>::compute_orientation_table(degree - 1);

      unsigned int dof_counter = 0;

      std::vector<unsigned int> obj_orientations;
      std::vector<unsigned int> obj_start_indices;

      for (unsigned int d = 0; d <= dim; ++d)
        {
          const auto entry = dpo[d];

          for (unsigned int i = 0; i < entry.first; ++i)
            {
              std::vector<types::global_dof_index> dofs_of_object(entry.second);

              for (unsigned int j = 0; j < entry.second; ++j)
                dofs_of_object[j] = dofs[dof_counter + j];

              obj_start_indices.emplace_back(
                *std::min_element(dofs_of_object.begin(),
                                  dofs_of_object.end()));

              if (d == 2 && (i == 2 || i == 3))
                {
                  auto dofs_of_object_copy = dofs_of_object;

                  for (unsigned int j = 0; j < entry.second; ++j)
                    dofs_of_object[j] =
                      dofs_of_object_copy[orientation_table[1][j]];
                }

              for (const auto dof : dofs_of_object)
                printf("%4u", dof);

              if (d == 1)
                {
                  const auto orientation =
                    get_orientation_line(dofs_of_object, degree);
                  printf(" -> %1u", orientation);

                  obj_orientations.emplace_back(orientation);
                }
              else if (d == 2)
                {
                  const auto orientation =
                    get_orientation_quad(dofs_of_object, orientation_table);
                  printf(" -> %1u", orientation);

                  obj_orientations.emplace_back(orientation);
                }

              printf("\n");

              dof_counter += entry.second;
            }
        }

      std::cout << std::endl;

      std::cout << "indices:     ";

      for (const auto i : obj_start_indices)
        std::cout << i << " ";
      std::cout << std::endl;

      const auto orientation = compress_orientation(obj_orientations);

      std::cout << "orientation: " << orientation << std::endl;
      std::cout << std::endl;

      compress_indices(dofs, dim, degree);
    };

    print_indices();

    typename MatrixFree<dim, double>::AdditionalData ad;

    DoFRenumbering::matrix_free_data_locality(dof_handler,
                                              AffineConstraints<double>(),
                                              ad);

    print_indices();
  }
}