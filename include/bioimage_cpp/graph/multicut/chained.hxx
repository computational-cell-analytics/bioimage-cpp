#pragma once

#include "bioimage_cpp/graph/multicut/objective.hxx"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut {

class ChainedSolver final : public CloneableSolverBase {
public:
    explicit ChainedSolver(
        const std::vector<const CloneableSolverBase *> &solvers
    ) {
        if (solvers.empty()) {
            throw std::invalid_argument("solvers must not be empty");
        }
        solvers_.reserve(solvers.size());
        for (const auto *solver : solvers) {
            if (solver == nullptr) {
                throw std::invalid_argument("solvers must not contain null");
            }
            solvers_.push_back(clone_solver(*solver));
        }
    }

    ChainedSolver(const ChainedSolver &) = delete;
    ChainedSolver &operator=(const ChainedSolver &) = delete;
    ChainedSolver(ChainedSolver &&) noexcept = default;
    ChainedSolver &operator=(ChainedSolver &&) noexcept = default;

    std::vector<std::uint64_t> optimize(Objective &objective) const override {
        for (const auto &solver : solvers_) {
            solver->optimize(objective);
        }
        return objective.labels();
    }

    std::unique_ptr<CloneableSolverBase> clone() const override {
        std::vector<std::unique_ptr<CloneableSolverBase>> cloned;
        cloned.reserve(solvers_.size());
        for (const auto &solver : solvers_) {
            cloned.push_back(clone_solver(*solver));
        }
        return std::unique_ptr<CloneableSolverBase>(
            new ChainedSolver(std::move(cloned))
        );
    }

private:
    explicit ChainedSolver(
        std::vector<std::unique_ptr<CloneableSolverBase>> solvers
    )
        : solvers_(std::move(solvers)) {
    }

    std::vector<std::unique_ptr<CloneableSolverBase>> solvers_;
};

} // namespace bioimage_cpp::graph::multicut
