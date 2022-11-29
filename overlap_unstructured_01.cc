#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/shape_info.h>

using namespace dealii;

namespace dealii
{
  namespace GridGenerator
  {
    void
    my_non_standard_orientation_mesh(Triangulation<2> &tria,
                                     const bool        rotate_left_square,
                                     const bool        rotate_right_square)
    {
      constexpr unsigned int dim = 2;

      const unsigned int         n_cells = 2;
      std::vector<CellData<dim>> cells(n_cells);

      // Corner points of the cube [0,1]^2
      const std::vector<Point<dim>> vertices = {Point<dim>(0, 0),  // 0
                                                Point<dim>(1, 0),  // 1
                                                Point<dim>(0, 1),  // 2
                                                Point<dim>(1, 1),  // 3
                                                Point<dim>(2, 0),  // 4
                                                Point<dim>(2, 1)}; // 5


      // consistent orientation
      unsigned int cell_vertices[n_cells][4] = {{0, 1, 2, 3},  // unit cube
                                                {1, 4, 3, 5}}; // shifted cube

      // all 4 true-false combinations of (rotate_left_square | rotate_right_square) to a number 0..3
      unsigned int this_case = 2 * rotate_left_square + rotate_right_square;

      switch (this_case)
        {
          case /* rotate only right square */ 1:
            {
              cell_vertices[1][0] = 4;
              cell_vertices[1][1] = 5;
              cell_vertices[1][2] = 1;
              cell_vertices[1][3] = 3;
              break;
            }

          case /* rotate only left square */ 2:
            {
              cell_vertices[0][0] = 1;
              cell_vertices[0][1] = 3;
              cell_vertices[0][2] = 0;
              cell_vertices[0][3] = 2;
              break;
            }

          case /* rotate both squares (again consistent orientation) */ 3:
            {
              cell_vertices[0][0] = 1;
              cell_vertices[0][1] = 3;
              cell_vertices[0][2] = 0;
              cell_vertices[0][3] = 2;

              cell_vertices[1][0] = 4;
              cell_vertices[1][1] = 5;
              cell_vertices[1][2] = 1;
              cell_vertices[1][3] = 3;
              break;
            }

          default /* 0 */:
            break;
        } // switch

      cells.resize(n_cells, CellData<dim>());

      for (unsigned int cell_index = 0; cell_index < n_cells; ++cell_index)
        {
          for (const unsigned int vertex_index :
               GeometryInfo<dim>::vertex_indices())
            {
              cells[cell_index].vertices[vertex_index] =
                cell_vertices[cell_index][vertex_index];
              cells[cell_index].material_id = 0;
            }
        }

      tria.create_triangulation(vertices, cells, SubCellData());
    }
  } // namespace GridGenerator
} // namespace dealii

int
main()
{
  const int dim       = 2;
  const int fe_degree = 2;
  const int n_overlap = 1;

  Triangulation<dim> tria;
  GridGenerator::my_non_standard_orientation_mesh(tria, 0, 1);

  FE_DGQ<dim> fe(fe_degree);
  QGauss<dim> quad(fe_degree + 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs_per_cell = Utilities::pow(fe_degree + 1, dim);
  const unsigned int n_dofs_per_face =
    n_overlap * Utilities::pow(fe_degree + 1, dim - 1);

  std::vector<types::global_dof_index> dof_indices_face(n_dofs_per_face);
  std::vector<types::global_dof_index> dof_indices_temp(n_dofs_per_face);
  std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);

  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info(quad, fe);

  const auto get_face_indices_of_neighbor = [&](const auto &cell,
                                                const auto  face_no,
                                                auto &      dof_indices) {
    const auto exterior_face_no = cell->neighbor_face_no(face_no);
    const auto neighbor         = cell->neighbor(face_no);

    cell->neighbor(face_no)->get_dof_indices(dof_indices);

    // TODO: lex ordering

    for (unsigned int i = 0; i < n_dofs_per_face; ++i)
      dof_indices_face[i] =
        dof_indices[shape_info.face_to_cell_index_nodal[exterior_face_no][i]];

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      dof_indices[i] = 0;

    if (dim == 3) // adjust orientation
      {
        const unsigned int exterior_face_orientation =
          !neighbor->face_orientation(exterior_face_no) +
          2 * neighbor->face_flip(exterior_face_no) +
          4 * neighbor->face_rotation(exterior_face_no);

        for (unsigned int i = 0; i < n_dofs_per_face; ++i)
          dof_indices_temp
            [shape_info.face_orientations_quad[exterior_face_orientation][i]] =
              dof_indices_face[i];

        for (unsigned int i = 0; i < n_dofs_per_face; ++i)
          dof_indices_face[i] = dof_indices_temp[i];
      }

    for (unsigned int i = 0; i < n_dofs_per_face; ++i)
      dof_indices[shape_info.face_to_cell_index_nodal[face_no][i]] =
        dof_indices_face[i];
  };

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face_index : cell->face_indices())
        if (cell->at_boundary(face_index) == false)
          {
            std::cout << cell->face_orientation(face_index) << std::endl;

            get_face_indices_of_neighbor(cell, face_index, dof_indices);

            for (const auto i : dof_indices)
              std::cout << i << " ";
            std::cout << std::endl;
          }
      std::cout << std::endl;
    }
}
