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


template <class VectorType, class PreconditionerType>
void
MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
  const unsigned int /*level*/,
  VectorType &      dst,
  const VectorType &src) const
{
  // internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  preconditioner->vmult(dst, src);
}

template <int dim_,
          typename LevelMatrixType_,
          typename SmootherType_,
          typename VectorType_,
          typename VectorTypeOuter>
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
    const MGLevelObject<std::shared_ptr<
      const AffineConstraints<typename VectorType_::value_type>>>
      &                                                    mg_constraints,
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
    : dof_handler(dof_handler)
    , mg_dof_handlers(mg_dof_handlers)
    , mg_constraints(mg_constraints)
    , mg_operators(mg_operators)
    , min_level(mg_dof_handlers.min_level())
    , max_level(mg_dof_handlers.max_level())
    , transfers(min_level, max_level)
  {}

  virtual SmootherType
  create_mg_level_smoother(unsigned int           level,
                           const LevelMatrixType &level_matrix) = 0;

  virtual SmootherType
  create_mg_coarse_grid_solver(unsigned int           level,
                               const LevelMatrixType &level_matrix) = 0;

  void
  print_timings() const
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << " - Times of multigrid (levels):" << std::endl;

        const auto print_line = [](const auto &vector) {
          for (const auto &i : vector)
            printf("%10.2e", i.first);

          double sum = 0;

          for (const auto &i : vector)
            sum += i.first;

          printf("   | %10.2e", sum);

          printf("\n");
        };

        for (unsigned int l = 0; l < all_mg_timers.size(); ++l)
          {
            printf("%4d: ", l);

            print_line(all_mg_timers[l]);
          }

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          sums(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size(); ++j)
            sums[i].first += all_mg_timers[j][i].first;

        printf(
          "   ----------------------------------------------------------------------------+-----------\n");
        printf("      ");
        print_line(sums);

        pcout << std::endl;

        std::cout << " - Times of multigrid (solver <-> mg): ";

        for (const auto i : all_mg_precon_timers)
          pcout << i.first << " ";
        pcout << std::endl;
        pcout << std::endl;
      }
  }

  void
  clear_timings() const
  {
    for (auto &is : all_mg_timers)
      for (auto &i : is)
        i.first = 0.0;

    for (auto &i : all_mg_precon_timers)
      i.first = 0.0;
  }

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
    mg_smoother.smoothers.resize(min_level, max_level);

    for (unsigned int level = min_level + 1; level <= max_level; ++level)
      mg_smoother.smoothers[level] =
        this->create_mg_level_smoother(level, *mg_operators[level]);

    // wrap level operators
    mg_matrix = std::make_unique<mg::Matrix<VectorType>>(mg_operators);

    // setup transfer operators (note: we do it here, since the smoothers
    // might change the ghosting of the level operators)
    for (auto l = min_level; l < max_level; ++l)
      transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                              *mg_dof_handlers[l],
                              *mg_constraints[l + 1],
                              *mg_constraints[l]);

    transfer =
      std::make_shared<MGTransferType>(transfers, [&](const auto l, auto &vec) {
        this->mg_operators[l]->initialize_dof_vector(vec);
      });

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

    // timers
    if (true)
      {
        all_mg_timers.resize((max_level - min_level + 1));
        for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
          all_mg_timers[i].resize(7);

        const auto create_mg_timer_function = [&](const unsigned int i,
                                                  const std::string &label) {
          return [i, label, this](const bool flag, const unsigned int level) {
            if (false && flag)
              std::cout << label << " " << level << std::endl;
            if (flag)
              all_mg_timers[level][i].second = std::chrono::system_clock::now();
            else
              all_mg_timers[level][i].first +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now() -
                  all_mg_timers[level][i].second)
                  .count() /
                1e9;
          };
        };

        {
          mg->connect_pre_smoother_step(
            create_mg_timer_function(0, "pre_smoother_step"));
          mg->connect_residual_step(
            create_mg_timer_function(1, "residual_step"));
          mg->connect_restriction(create_mg_timer_function(2, "restriction"));
          mg->connect_coarse_solve(create_mg_timer_function(3, "coarse_solve"));
          mg->connect_prolongation(create_mg_timer_function(4, "prolongation"));
          mg->connect_edge_prolongation(
            create_mg_timer_function(5, "edge_prolongation"));
          mg->connect_post_smoother_step(
            create_mg_timer_function(6, "post_smoother_step"));
        }


        all_mg_precon_timers.resize(2);

        const auto create_mg_precon_timer_function = [&](const unsigned int i) {
          return [i, this](const bool flag) {
            if (flag)
              all_mg_precon_timers[i].second = std::chrono::system_clock::now();
            else
              all_mg_precon_timers[i].first +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now() -
                  all_mg_precon_timers[i].second)
                  .count() /
                1e9;
          };
        };

        preconditioner->connect_transfer_to_mg(
          create_mg_precon_timer_function(0));
        preconditioner->connect_transfer_to_global(
          create_mg_precon_timer_function(1));
      }
  }

  virtual void
  vmult(VectorTypeOuter &dst, const VectorTypeOuter &src) const
  {
    preconditioner->vmult(dst, src);
  }

protected:
  const DoFHandler<dim> &dof_handler;

  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<
    std::shared_ptr<const AffineConstraints<typename VectorType::value_type>>>
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
  mutable MGSmootherRelaxation<LevelMatrixType, SmootherType, VectorType>
    mg_smoother;

  // multigrid
  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;
  mutable std::unique_ptr<Multigrid<VectorType>>  mg;
  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;

  mutable std::vector<std::vector<
    std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
    all_mg_timers;

  mutable std::vector<
    std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
    all_mg_precon_timers;
};

DEAL_II_NAMESPACE_CLOSE
