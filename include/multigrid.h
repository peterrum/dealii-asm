#pragma once

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

DEAL_II_NAMESPACE_OPEN

struct PreconditionerGMGAdditionalData
{
  double       smoothing_range               = 20;
  unsigned int smoothing_degree              = 5;
  unsigned int smoothing_eig_cg_n_iterations = 20;

  unsigned int coarse_grid_smoother_sweeps = 1;
  unsigned int coarse_grid_n_cycles        = 1;
  std::string  coarse_grid_smoother_type   = "ILU";

  unsigned int coarse_grid_maxiter = 1000;
  double       coarse_grid_abstol  = 1e-20;
  double       coarse_grid_reltol  = 1e-4;
};

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

  void
  do_update()
  {
    PreconditionerGMGAdditionalData additional_data;

    // wrap level operators
    mg_matrix = std::make_unique<mg::Matrix<VectorType>>(mg_operators);

    // setup smoothers on each level
    mg_smoother.initialize_matrices(mg_operators);

    for (unsigned int level = min_level; level <= max_level; ++level)
      mg_smoother.smoothers[level] =
        this->create_mg_level_smoother(level, *mg_operators[level]);

    // setup coarse-grid solver
    coarse_grid_solver_control =
      std::make_unique<ReductionControl>(additional_data.coarse_grid_maxiter,
                                         additional_data.coarse_grid_abstol,
                                         additional_data.coarse_grid_reltol,
                                         false,
                                         false);
    coarse_grid_solver =
      std::make_unique<SolverCG<VectorType>>(*coarse_grid_solver_control);

    mg_coarse =
      std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                   SolverCG<VectorType>,
                                                   LevelMatrixType,
                                                   SmootherType>>(
        *coarse_grid_solver,
        *mg_operators[min_level],
        mg_smoother.smoothers[min_level]);

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

  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
  std::shared_ptr<MGTransferType>                    transfer;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
    mg_smoother;

  mutable std::unique_ptr<ReductionControl> coarse_grid_solver_control;

  mutable std::unique_ptr<SolverCG<VectorType>> coarse_grid_solver;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};

DEAL_II_NAMESPACE_CLOSE
