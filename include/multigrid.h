#pragma once

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

DEAL_II_NAMESPACE_OPEN

/**
 * Coarse grid solver using a preconditioner only. This is a little wrapper,
 * transforming a preconditioner into a coarse grid solver.
 */
template <class VectorType, class PreconditionerType>
class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
{
public:
  /**
   * Default constructor.
   */
  MGCoarseGridApplyPreconditioner();

  /**
   * Constructor. Store a pointer to the preconditioner for later use.
   */
  MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

  /**
   * Clear the pointer.
   */
  void
  clear();

  /**
   * Initialize new data.
   */
  void
  initialize(const PreconditionerType &precondition);

  /**
   * Implementation of the abstract function.
   */
  virtual void
  operator()(const unsigned int level,
             VectorType &       dst,
             const VectorType & src) const override;

private:
  /**
   * Reference to the preconditioner.
   */
  SmartPointer<const PreconditionerType,
               MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
    preconditioner;
};



template <class VectorType, class PreconditionerType>
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
  MGCoarseGridApplyPreconditioner()
  : preconditioner(0, typeid(*this).name())
{}



template <class VectorType, class PreconditionerType>
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
  MGCoarseGridApplyPreconditioner(const PreconditionerType &preconditioner)
  : preconditioner(&preconditioner, typeid(*this).name())
{}



template <class VectorType, class PreconditionerType>
void
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
  const PreconditionerType &preconditioner_)
{
  preconditioner = &preconditioner_;
}



template <class VectorType, class PreconditionerType>
void
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
{
  preconditioner = 0;
}


namespace internal
{
  namespace MGCoarseGridApplyPreconditioner
  {
    template <class VectorType,
              class PreconditionerType,
              typename std::enable_if<
                std::is_same<typename VectorType::value_type, double>::value,
                VectorType>::type * = nullptr>
    void
    solve(const PreconditionerType preconditioner,
          VectorType &             dst,
          const VectorType &       src)
    {
      // to allow the case that the preconditioner was only set up on a
      // subset of processes
      if (preconditioner != nullptr)
        preconditioner->vmult(dst, src);
    }

    template <class VectorType,
              class PreconditionerType,
              typename std::enable_if<
                !std::is_same<typename VectorType::value_type, double>::value,
                VectorType>::type * = nullptr>
    void
    solve(const PreconditionerType preconditioner,
          VectorType &             dst,
          const VectorType &       src)
    {
      LinearAlgebra::distributed::Vector<double> src_;
      LinearAlgebra::distributed::Vector<double> dst_;

      src_ = src;
      dst_ = dst;

      // to allow the case that the preconditioner was only set up on a
      // subset of processes
      if (preconditioner != nullptr)
        preconditioner->vmult(dst_, src_);

      dst = dst_;
    }
  } // namespace MGCoarseGridApplyPreconditioner
} // namespace internal


template <class VectorType, class PreconditionerType>
void
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
  const unsigned int /*level*/,
  VectorType &      dst,
  const VectorType &src) const
{
  internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
}

template <int dim_,
          typename LevelMatrixType_,
          typename SmootherType_,
          typename VectorType_>
class PreconditionerGMG
{
public:
  static const int dim  = dim_;
  using LevelMatrixType = LevelMatrixType_;
  using SmootherType    = SmootherType_;
  using VectorType      = VectorType_;

private:
  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;

public:
  PreconditionerGMG(
    const DoFHandler<dim> &dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
      &                                                    mg_constraints,
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
    : dof_handler(dof_handler)
    , mg_dof_handlers(mg_dof_handlers)
    , mg_constraints(mg_constraints)
    , mg_operators(mg_operators)
    , min_level(mg_dof_handlers.min_level())
    , max_level(mg_dof_handlers.max_level())
    , transfers(min_level, max_level)
  {
    // setup transfer operators
    for (auto l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                              *mg_dof_handlers[l],
                              *mg_constraints[l + 1],
                              *mg_constraints[l]);

    transfer =
      std::make_shared<MGTransferType>(transfers, [&](const auto l, auto &vec) {
        this->mg_operators[l]->initialize_dof_vector(vec);
      });
  }

  virtual SmootherType
  create_mg_level_smoother(unsigned int           level,
                           const LevelMatrixType &level_matrix) = 0;

  virtual SmootherType
  create_mg_coarse_grid_solver(unsigned int           level,
                               const LevelMatrixType &level_matrix) = 0;

  void
  do_update()
  {
    // setup coarse-grid solver
    coarse_grid_solver =
      this->create_mg_coarse_grid_solver(min_level, *mg_operators[min_level]);

    mg_coarse = std::make_unique<
      MGCoarseGridApplyPreconditioner<VectorType, SmootherType>>(
      coarse_grid_solver);

    // setup smoothers on each level
    mg_smoother.initialize_matrices(mg_operators);

    for (unsigned int level = min_level + 1; level <= max_level; ++level)
      mg_smoother.smoothers[level] =
        this->create_mg_level_smoother(level, *mg_operators[level]);

    // wrap level operators
    mg_matrix = std::make_unique<mg::Matrix<VectorType>>(mg_operators);

    // create multigrid algorithm (put level operators, smoothers, transfer
    // operators and smoothers together)
    mg = std::make_unique<Multigrid<VectorType>>(*mg_matrix,
                                                 *mg_coarse,
                                                 *transfer,
                                                 mg_smoother,
                                                 mg_smoother,
                                                 min_level,
                                                 max_level);

    // convert multigrid algorithm to preconditioner
    preconditioner =
      std::make_unique<PreconditionMG<dim, VectorType, MGTransferType>>(
        dof_handler, *mg, *transfer);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    preconditioner->vmult(dst, src);
  }

protected:
  const DoFHandler<dim> &dof_handler;

  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
                                                        mg_constraints;
  const MGLevelObject<std::shared_ptr<LevelMatrixType>> mg_operators;

  const unsigned int min_level;
  const unsigned int max_level;

  // transfer
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  std::shared_ptr<MGTransferType>                    transfer;

  // coarse-grid solver
  mutable SmootherType                                  coarse_grid_solver;
  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  // smoothers
  mutable MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
    mg_smoother;

  // multigrid
  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;
  mutable std::unique_ptr<Multigrid<VectorType>>  mg;
  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};

DEAL_II_NAMESPACE_CLOSE
