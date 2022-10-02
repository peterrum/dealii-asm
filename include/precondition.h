#pragma once

#include "matrix_free.h"
#include "multigrid.h"
#include "preconditioners.h"

template <typename OperatorType>
std::shared_ptr<
  const ASPoissonPreconditioner<OperatorType::dimension,
                                typename OperatorType::value_type,
                                typename OperatorType::vectorized_array_type,
                                -1>>
create_fdm_preconditioner(const OperatorType &              op,
                          const boost::property_tree::ptree params);

template <typename OperatorType>
std::shared_ptr<const PreconditionerBase<typename OperatorType::vector_type>>
create_system_preconditioner(const OperatorType &              op,
                             const boost::property_tree::ptree params);



template <typename VectorType>
class WrapperForGMG : public Subscriptor
{
public:
  struct AdditionalData
  {};

  WrapperForGMG() = default;

  WrapperForGMG(
    const std::shared_ptr<const PreconditionerBase<VectorType>> &base)
    : base(base)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    base->vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    Assert(false, ExcNotImplemented());
    (void)dst;
    (void)src;
  }

  void
  step(VectorType &dst, const VectorType &src) const
  {
    base->step(dst, src);
  }

  void
  Tstep(VectorType &dst, const VectorType &src) const
  {
    Assert(false, ExcNotImplemented());
    (void)dst;
    (void)src;
  }

  void
  clear()
  {
    Assert(false, ExcNotImplemented());
  }

  std::shared_ptr<const PreconditionerBase<VectorType>> base;
};



template <int dim,
          typename LevelMatrixType_,
          typename VectorType,
          typename VectorTypeOuter>
class MyMultigrid : public PreconditionerGMG<dim,
                                             LevelMatrixType_,
                                             WrapperForGMG<VectorType>,
                                             VectorType,
                                             VectorTypeOuter>
{
public:
  using Base = PreconditionerGMG<dim,
                                 LevelMatrixType_,
                                 WrapperForGMG<VectorType>,
                                 VectorType,
                                 VectorTypeOuter>;

  using LevelMatrixType = typename Base::LevelMatrixType;
  using SmootherType    = typename Base::SmootherType;

  MyMultigrid(
    const boost::property_tree::ptree params,
    const DoFHandler<dim> &           dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<
      std::shared_ptr<const AffineConstraints<typename VectorType::value_type>>>
      &                                                    mg_constraints,
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
    : PreconditionerGMG<dim,
                        LevelMatrixType_,
                        WrapperForGMG<VectorType>,
                        VectorType,
                        VectorTypeOuter>(dof_handler,
                                         mg_dof_handlers,
                                         mg_constraints,
                                         mg_operators)
    , params(params)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    this->do_update();
  }

  SmootherType
  create_mg_level_smoother(unsigned int           level,
                           const LevelMatrixType &level_matrix) final
  {
    pcout << "- Setting up smoother on level " << level << "" << std::endl
          << std::endl;

    (void)level;

    return WrapperForGMG<VectorType>(
      create_system_preconditioner<LevelMatrixType>(
        level_matrix, try_get_child(params, "mg smoother")));
  }

  SmootherType
  create_mg_coarse_grid_solver(unsigned int           level,
                               const LevelMatrixType &level_matrix) final
  {
    pcout << "- Setting up coarse-grid solver on level " << level << ""
          << std::endl
          << std::endl;

    (void)level;

    return WrapperForGMG<VectorType>(
      create_system_preconditioner<LevelMatrixType>(
        level_matrix, try_get_child(params, "mg coarse grid solver")));
  }

private:
  const boost::property_tree::ptree params;
  ConditionalOStream                pcout;
};

#include "precondition.templates.h"
