#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

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

template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int n_overlap,
     const unsigned int orientation,
     const bool         use_dg)
{
  Triangulation<dim> tria;

  if constexpr (dim == 2)
    {
      GridGenerator::my_non_standard_orientation_mesh(tria, 0, orientation);
    }
  else if constexpr (dim == 3)
    {
      std::cout << "o " << ((orientation & 1)) << " " << ((orientation & 2))
                << " " << ((orientation & 4)) << std::endl;

      GridGenerator::non_standard_orientation_mesh(
        tria, (orientation & 4), (orientation & 2), (orientation & 1), false);
    }
  else
    AssertThrow(false, ExcNotImplemented());

  std::unique_ptr<FiniteElement<dim>> fe;

  if (use_dg)
    fe = std::make_unique<FE_DGQ<dim>>(fe_degree);
  else
    fe = std::make_unique<FE_Q<dim>>(fe_degree);

  QGauss<dim> quad(fe_degree + 1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  const unsigned int n_dofs_per_cell  = Utilities::pow(fe_degree + 1, dim);
  const unsigned int n_dofs_per_layer = Utilities::pow(fe_degree + 1, dim - 1);
  const unsigned int n_dofs_per_face  = n_overlap * n_dofs_per_layer;

  std::vector<types::global_dof_index> dof_indices_face(n_dofs_per_face);
  std::vector<types::global_dof_index> dof_indices_temp(n_dofs_per_layer);
  std::vector<types::global_dof_index> dof_indices(n_dofs_per_cell);

  internal::MatrixFreeFunctions::ShapeInfo<double> shape_info(quad, *fe);

  dealii::Table<2, unsigned int> face_to_cell_index_nodal(2 * dim,
                                                          n_dofs_per_cell);

  if (dim == 2)
    {
      const auto to_index = [&](const unsigned int i, const unsigned int j) {
        return i + j * (fe_degree + 1);
      };

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int i = 0; i <= fe_degree; ++i)
          face_to_cell_index_nodal[0][c++] = to_index(o, i);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int i = 0; i <= fe_degree; ++i)
          face_to_cell_index_nodal[1][c++] = to_index(fe_degree - o, i);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int i = 0; i <= fe_degree; ++i)
          face_to_cell_index_nodal[2][c++] = to_index(i, o);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int i = 0; i <= fe_degree; ++i)
          face_to_cell_index_nodal[3][c++] = to_index(i, fe_degree - o);
    }
  else if (dim == 3)
    {
      const auto to_index = [&](const unsigned int i,
                                const unsigned int j,
                                const unsigned int k) {
        return i + j * (fe_degree + 1) + k * (fe_degree + 1) * (fe_degree + 1);
      };

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[0][c++] = to_index(o, i, j);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[1][c++] = to_index(fe_degree - o, i, j);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[2][c++] = to_index(i, o, j);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[3][c++] = to_index(i, fe_degree - o, j);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[4][c++] = to_index(i, j, o);

      for (unsigned int o = 0, c = 0; o < n_overlap; ++o)
        for (unsigned int j = 0; j <= fe_degree; ++j)
          for (unsigned int i = 0; i <= fe_degree; ++i)
            face_to_cell_index_nodal[5][c++] = to_index(i, j, fe_degree - o);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented())
    }

  const auto lexicographic_to_hierarchic_numbering =
    Utilities::invert_permutation(
      FETools::hierarchic_to_lexicographic_numbering<dim>(fe_degree));

  const auto get_face_indices_of_neighbor = [&](const auto &cell,
                                                const auto  face_no,
                                                auto &      dof_indices) {
    const auto exterior_face_no = cell->neighbor_face_no(face_no);
    const auto neighbor         = cell->neighbor(face_no);

    std::cout << exterior_face_no << std::endl;

    cell->neighbor(face_no)->get_dof_indices(dof_indices);

    // lex ordering
    if (fe->n_dofs_per_vertex() > 0)
      {
        auto temp = dof_indices;

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          dof_indices[i] = temp[lexicographic_to_hierarchic_numbering[i]];
      }

    for (unsigned int i = 0; i < n_dofs_per_face; ++i)
      dof_indices_face[i] =
        dof_indices[face_to_cell_index_nodal[exterior_face_no][i]];

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      dof_indices[i] = 0;

    if (dim == 3) // adjust orientation
      {
        const unsigned int interior_face_orientation =
          !cell->face_orientation(face_no) + 2 * cell->face_flip(face_no) +
          4 * cell->face_rotation(face_no);

        const unsigned int exterior_face_orientation =
          !neighbor->face_orientation(exterior_face_no) +
          2 * neighbor->face_flip(exterior_face_no) +
          4 * neighbor->face_rotation(exterior_face_no);

        auto face_orientation = interior_face_orientation;

        if (face_orientation == 0)
          {
            constexpr std::array<std::uint8_t, 8> table{
              {0, 1, 2, 3, 6, 5, 4, 7}};
            face_orientation = table[exterior_face_orientation];
          }

        for (unsigned int l = 0; l < n_overlap; ++l)
          {
            for (unsigned int i = 0; i < n_dofs_per_layer; ++i)
              dof_indices_temp[shape_info
                                 .face_orientations_quad[face_orientation][i]] =
                dof_indices_face[i + l * n_dofs_per_layer];

            for (unsigned int i = 0; i < n_dofs_per_layer; ++i)
              dof_indices_face[i + l * n_dofs_per_layer] = dof_indices_temp[i];
          }
      }

    std::cout << (face_no ^ (1 << (face_no / 2))) << std::endl;

    for (unsigned int i = 0; i < n_dofs_per_face; ++i)
      dof_indices[face_to_cell_index_nodal[face_no ^ (1 << (face_no / 2))][i]] =
        dof_indices_face[i];
  };

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face_index : cell->face_indices())
        if (cell->at_boundary(face_index) == false)
          {
            std::cout << "f " << face_index << std::endl;

            get_face_indices_of_neighbor(cell, face_index, dof_indices);

            for (const auto i : dof_indices)
              std::cout << i << " ";
            std::cout << std::endl;
          }
      std::cout << std::endl;
    }
}

/**
 * ./overlap_unstructured_01 3 1 2 2 0
 */
int
main(int argc, char *argv[])
{
  AssertThrow(argc == 6, ExcNotImplemented());

  const int dim         = std::atoi(argv[1]);
  const int fe_degree   = std::atoi(argv[2]);
  const int n_overlap   = std::atoi(argv[3]);
  const int orientation = std::atoi(argv[4]);
  const int use_dg      = std::atoi(argv[5]);

  if (dim == 2)
    test<2>(fe_degree, n_overlap, orientation, use_dg);
  else if (dim == 3)
    test<3>(fe_degree, n_overlap, orientation, use_dg);
  else
    AssertThrow(false, ExcNotImplemented());
}
