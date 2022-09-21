#pragma once

#include "reduced_access.h"

#define MAX_DEGREE_RW 6

// clang-format off
#define EXPAND_OPERATIONS_RW(OPERATION)                              \
  switch (degree)                                                        \
    {                                                                    \
      case  2: OPERATION((( 2 <= MAX_DEGREE_RW) ?  2 : -1), -1); break; \
      case  3: OPERATION((( 3 <= MAX_DEGREE_RW) ?  3 : -1), -1); break; \
      case  4: OPERATION((( 4 <= MAX_DEGREE_RW) ?  4 : -1), -1); break; \
      case  5: OPERATION((( 5 <= MAX_DEGREE_RW) ?  5 : -1), -1); break; \
      case  6: OPERATION((( 6 <= MAX_DEGREE_RW) ?  6 : -1), -1); break; \
      default:                                                           \
        OPERATION(-1, -1);                                               \
    }
// clang-format on

class ConstraintInfoReduced
{
public:
  DeclExceptionMsg(ExcExpectedContiguousNumbering,
                   "Expected contiguous numbering!");

  template <int dim, typename Number, typename VectorizedArrayType>
  void
  initialize(const MatrixFree<dim, Number, VectorizedArrayType> &data_)
  {
    const auto  data         = &data_;
    const auto &fe           = data->get_dof_handler().get_fe();
    const auto  fe_degree    = fe.degree;
    const auto  n_components = fe.n_components();

    if (fe_degree > 2)
      {
        try
          {
            const auto &constraints = data->get_affine_constraints();

            compressed_dof_indices.resize(Utilities::pow(3, dim) *
                                            VectorizedArrayType::size() *
                                            data->n_cell_batches(),
                                          numbers::invalid_unsigned_int);
            all_indices_uniform.resize(Utilities::pow(3, dim) *
                                         data->n_cell_batches(),
                                       1);
            orientations.resize(data->n_cell_batches() *
                                  VectorizedArrayType::size(),
                                0);

            std::vector<types::global_dof_index> dof_indices(
              data->get_dof_handler().get_fe().dofs_per_cell);

            constexpr unsigned int    n_lanes = VectorizedArrayType::size();
            std::vector<unsigned int> renumber_lex =
              FETools::hierarchic_to_lexicographic_numbering<dim>(2);
            for (auto &i : renumber_lex)
              i *= n_lanes;

            for (unsigned int c = 0; c < data->n_cell_batches(); ++c)
              {
                for (unsigned int l = 0;
                     l < data->n_active_entries_per_cell_batch(c);
                     ++l)
                  {
                    const unsigned int offset =
                      Utilities::pow(3, dim) * n_lanes * c + l;

                    const typename DoFHandler<dim>::cell_iterator cell =
                      data->get_cell_iterator(c, l);

                    cell->get_dof_indices(dof_indices);

                    // TODO: constraints, component
                    AssertThrow(n_components == 1, ExcNotImplemented());
                    AssertThrow(constraints.n_constraints() == 0,
                                ExcNotImplemented());

                    const auto [orientation, objec_indices] =
                      compress_indices(dof_indices, dim, fe_degree, true);

                    AssertThrow(orientation != numbers::invalid_unsigned_int,
                                ExcExpectedContiguousNumbering());

                    AssertThrow(objec_indices.size() == renumber_lex.size(),
                                ExcInternalError());

                    for (unsigned int i = 0; i < objec_indices.size(); ++i)
                      compressed_dof_indices[offset + renumber_lex[i]] =
                        objec_indices[i];

                    orientations[n_lanes * c + l] = orientation;
                  }

                for (unsigned int i = 0;
                     i < Utilities::pow<unsigned int>(3, dim);
                     ++i)
                  for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                    if (compressed_dof_indices
                          [Utilities::pow<unsigned int>(3, dim) *
                             (VectorizedArrayType::size() * c) +
                           i * VectorizedArrayType::size() + v] ==
                        numbers::invalid_unsigned_int)
                      all_indices_uniform[Utilities::pow(3, dim) * c + i] = 0;
              }

            orientation_table = internal::MatrixFreeFunctions::ShapeInfo<
              double>::compute_orientation_table(fe_degree - 1);

            if (std::all_of(orientations.begin(),
                            orientations.end(),
                            [&](const auto &e) { return e == 0; }))
              orientations.clear();
          }
        catch (const ExcExpectedContiguousNumbering &)
          {
            compressed_dof_indices.clear();
            all_indices_uniform.clear();

            orientations.clear();
          }
      }
  }

  unsigned int
  compression_level() const
  {
    if (compressed_dof_indices.empty())
      return 0;
    else if (orientations.empty())
      return 1;
    else
      return 2;
  }

  template <int dim,
            int fe_degree,
            int n_q_points,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  read_dof_values(const dealii::LinearAlgebra::distributed::Vector<Number> &vec,
                  FEEvaluation<dim,
                               fe_degree,
                               n_q_points,
                               n_components,
                               Number,
                               VectorizedArrayType> &                       phi)
  {
    if (compressed_dof_indices.empty() && all_indices_uniform.empty())
      {
        phi.read_dof_values(vec);
        return;
      }

    if constexpr (fe_degree != -1)
      {
        read_dof_values_compressed<dim, fe_degree, n_components>(
          vec, phi.get_current_cell_index(), phi.begin_dof_values());
      }
    else
      {
        const auto degree = phi.get_shape_info().data.front().fe_degree;

#define OPERATION(c, d)                             \
  AssertThrow(c != -1, ExcNotImplemented());        \
                                                    \
  read_dof_values_compressed<dim, c, n_components>( \
    vec, phi.get_current_cell_index(), phi.begin_dof_values());

        EXPAND_OPERATIONS_RW(OPERATION);
#undef OPERATION
      }
  }

  template <int dim,
            int fe_degree,
            int n_q_points,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  distribute_local_to_global(
    dealii::LinearAlgebra::distributed::Vector<Number> &vec,
    FEEvaluation<dim,
                 fe_degree,
                 n_q_points,
                 n_components,
                 Number,
                 VectorizedArrayType> &                 phi)
  {
    if (compressed_dof_indices.empty() && all_indices_uniform.empty())
      {
        phi.distribute_local_to_global(vec);
        return;
      }

    if constexpr (fe_degree != -1)
      {
        distribute_local_to_global_compressed<dim, fe_degree, n_components>(
          vec, phi.get_current_cell_index(), phi.begin_dof_values());
      }
    else
      {
        const auto degree = phi.get_shape_info().data.front().fe_degree;

#define OPERATION(c, d)                                        \
  AssertThrow(c != -1, ExcNotImplemented());                   \
                                                               \
  distribute_local_to_global_compressed<dim, c, n_components>( \
    vec, phi.get_current_cell_index(), phi.begin_dof_values());

        EXPAND_OPERATIONS_RW(OPERATION);
#undef OPERATION
      }
  }

private:
  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  read_dof_values_compressed(
    const dealii::LinearAlgebra::distributed::Vector<Number> &vec,
    const unsigned int                                        cell_no,
    VectorizedArrayType *                                     dof_values)
  {
    AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) *
                       VectorizedArrayType::size(),
                     compressed_dof_indices.size());
    constexpr unsigned int n_lanes = VectorizedArrayType::size();
    const unsigned int *   cell_indices =
      compressed_dof_indices.data() +
      cell_no * n_lanes * dealii::Utilities::pow(3, dim);
    const unsigned char *cell_unconstrained =
      all_indices_uniform.data() + cell_no * dealii::Utilities::pow(3, dim);
    constexpr unsigned int dofs_per_comp =
      dealii::Utilities::pow(fe_degree + 1, dim);
    dealii::internal::VectorReader<Number, VectorizedArrayType> reader;
    for (int i2 = 0, i = 0, compressed_i2 = 0, offset_i2 = 0;
         i2 < (dim == 3 ? (fe_degree + 1) : 1);
         ++i2)
      {
        for (int i1 = 0, compressed_i1 = 0, offset_i1 = 0;
             i1 < (dim > 1 ? (fe_degree + 1) : 1);
             ++i1)
          {
            const unsigned int offset =
              (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
            const unsigned int *indices =
              cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
            const unsigned char *unconstrained =
              cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

            // left end point
            if (unconstrained[0])
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(indices,
                                          vec,
                                          offset * n_components + c,
                                          dof_values[i + c * dofs_per_comp],
                                          std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  dof_values[i + c * dofs_per_comp][v] =
                    (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                      0. :
                      vec.local_element(indices[v] + offset * n_components + c);
            ++i;
            indices += n_lanes;

            // interior points of line
            if (unconstrained[1])
              {
                VectorizedArrayType
                  tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
                constexpr unsigned int n_regular =
                  (fe_degree - 1) * n_components / 4 * 4;
                dealii::vectorized_load_and_transpose(
                  n_regular,
                  vec.begin() + offset * (fe_degree - 1) * n_components,
                  indices,
                  tmp);
                for (int i0 = n_regular; i0 < (fe_degree - 1) * n_components;
                     ++i0)
                  reader.process_dof_gather(
                    indices,
                    vec,
                    offset * (fe_degree - 1) * n_components + i0,
                    tmp[i0],
                    std::integral_constant<bool, true>());
                for (int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                  for (unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp] =
                      tmp[i0 * n_components + c];
              }
            else
              for (int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    for (unsigned int c = 0; c < n_components; ++c)
                      dof_values[i + c * dofs_per_comp][v] = vec.local_element(
                        indices[v] +
                        (offset * (fe_degree - 1) + i0) * n_components + c);
                  else
                    for (unsigned int c = 0; c < n_components; ++c)
                      dof_values[i + c * dofs_per_comp][v] = 0.;
            indices += n_lanes;

            // right end point
            if (unconstrained[2])
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(indices,
                                          vec,
                                          offset * n_components + c,
                                          dof_values[i + c * dofs_per_comp],
                                          std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  dof_values[i + c * dofs_per_comp][v] =
                    (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                      0. :
                      vec.local_element(indices[v] + offset * n_components + c);
            ++i;

            if (i1 == 0 || i1 == fe_degree - 1)
              {
                ++compressed_i1;
                offset_i1 = 0;
              }
            else
              ++offset_i1;
          }
        if (i2 == 0 || i2 == fe_degree - 1)
          {
            ++compressed_i2;
            offset_i2 = 0;
          }
        else
          ++offset_i2;
      }

    adjust_for_orientation<VectorizedArrayType, dim, fe_degree>(
      dim,
      fe_degree,
      n_components,
      false,
      cell_no,
      orientations,
      orientation_table,
      dof_values);
  }

  template <int dim,
            int fe_degree,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  void
  distribute_local_to_global_compressed(
    dealii::LinearAlgebra::distributed::Vector<Number> &vec,
    const unsigned int                                  cell_no,
    VectorizedArrayType *                               dof_values)
  {
    AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) *
                       VectorizedArrayType::size(),
                     compressed_dof_indices.size());
    constexpr unsigned int n_lanes = VectorizedArrayType::size();
    const unsigned int *   cell_indices =
      compressed_dof_indices.data() +
      cell_no * n_lanes * dealii::Utilities::pow(3, dim);
    const unsigned char *cell_unconstrained =
      all_indices_uniform.data() + cell_no * dealii::Utilities::pow(3, dim);
    constexpr unsigned int dofs_per_comp =
      dealii::Utilities::pow(fe_degree + 1, dim);
    dealii::internal::VectorDistributorLocalToGlobal<Number,
                                                     VectorizedArrayType>
      distributor;

    adjust_for_orientation<VectorizedArrayType, dim, fe_degree>(
      dim,
      fe_degree,
      n_components,
      true,
      cell_no,
      orientations,
      orientation_table,
      dof_values);

    for (int i2 = 0, i = 0, compressed_i2 = 0, offset_i2 = 0;
         i2 < (dim == 3 ? (fe_degree + 1) : 1);
         ++i2)
      {
        for (int i1 = 0, compressed_i1 = 0, offset_i1 = 0;
             i1 < (dim > 1 ? (fe_degree + 1) : 1);
             ++i1)
          {
            const unsigned int offset =
              (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
            const unsigned int *indices =
              cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
            const unsigned char *unconstrained =
              cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

            // left end point
            if (unconstrained[0])
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(
                  indices,
                  vec,
                  offset * n_components + c,
                  dof_values[i + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + offset * n_components + c) +=
                      dof_values[i + c * dofs_per_comp][v];
            ++i;
            indices += n_lanes;

            // interior points of line
            if (unconstrained[1])
              {
                VectorizedArrayType
                  tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
                for (int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                  for (unsigned int c = 0; c < n_components; ++c)
                    tmp[i0 * n_components + c] =
                      dof_values[i + c * dofs_per_comp];

                constexpr unsigned int n_regular =
                  (fe_degree - 1) * n_components / 4 * 4;
                dealii::vectorized_transpose_and_store(
                  true,
                  n_regular,
                  tmp,
                  indices,
                  vec.begin() + offset * (fe_degree - 1) * n_components);
                for (int i0 = n_regular; i0 < (fe_degree - 1) * n_components;
                     ++i0)
                  distributor.process_dof_gather(
                    indices,
                    vec,
                    offset * (fe_degree - 1) * n_components + i0,
                    tmp[i0],
                    std::integral_constant<bool, true>());
              }
            else
              for (int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    for (unsigned int c = 0; c < n_components; ++c)
                      vec.local_element(
                        indices[v] +
                        (offset * (fe_degree - 1) + i0) * n_components + c) +=
                        dof_values[i + c * dofs_per_comp][v];
            indices += n_lanes;

            // right end point
            if (unconstrained[2])
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(
                  indices,
                  vec,
                  offset * n_components + c,
                  dof_values[i + c * dofs_per_comp],
                  std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + offset * n_components + c) +=
                      dof_values[i + c * dofs_per_comp][v];
            ++i;

            if (i1 == 0 || i1 == fe_degree - 1)
              {
                ++compressed_i1;
                offset_i1 = 0;
              }
            else
              ++offset_i1;
          }
        if (i2 == 0 || i2 == fe_degree - 1)
          {
            ++compressed_i2;
            offset_i2 = 0;
          }
        else
          ++offset_i2;
      }
  }

  std::vector<unsigned int>  compressed_dof_indices;
  std::vector<unsigned char> all_indices_uniform;

  Table<2, unsigned int>    orientation_table; // for reorienation
  std::vector<unsigned int> orientations;
};
