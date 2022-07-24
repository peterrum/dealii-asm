#include <deal.II/base/config.h>

using namespace dealii;

#include "include/preconditioners.h"

template <typename Number>
class Matrix : public MatrixView<Number>
{
public:
  Matrix(const unsigned int n_dofs)
  {
    matrix.reinit(n_dofs, n_dofs);

    for (unsigned int i = 0; i < n_dofs; ++i)
      {
        if (i != 0)
          matrix[i][i - 1] = -1;
        matrix[i][i] = 2;
        if (i + 1 != n_dofs)
          matrix[i][i + 1] = -1;
      }
  }

  void
  vmult(const unsigned int    c,
        Vector<Number> &      dst,
        const Vector<Number> &src) const final
  {
    AssertDimension(c, 0);
    matrix.vmult(dst, src);
  }

  unsigned int
  size() const final
  {
    return 1;
  }

  unsigned int
  size(const unsigned int c) const final
  {
    AssertDimension(c, 0);

    return matrix.m();
  }

private:
  FullMatrix<Number> matrix;
};

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  using Number = double;

  const unsigned int n_dofs = 10;

  const auto matrix = std::make_shared<Matrix<double>>(n_dofs);

  TriDiagonalMatrixView<Number> tridiagonal_matrix;
  tridiagonal_matrix.initialize(matrix);
  tridiagonal_matrix.invert();

  Vector<Number> src(n_dofs);
  Vector<Number> dst(n_dofs);

  for (unsigned int i = 0; i < n_dofs; ++i)
    src[i] = i;

  tridiagonal_matrix.vmult(0, dst, src);
  src.print(std::cout);
  dst.print(std::cout);


  matrix->vmult(0, src, dst);
  src.print(std::cout);
}