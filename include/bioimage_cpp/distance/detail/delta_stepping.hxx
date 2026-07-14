#pragma once

#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::distance::detail_delta_stepping {

inline constexpr std::size_t kSequentialProblemThreshold = 32768;
inline constexpr std::size_t kParallelFrontierThreshold = 4096;
inline constexpr std::size_t kMaximumBatchProposals = 1U << 20;

template <class Node>
inline constexpr Node kNoNode = std::numeric_limits<Node>::max();

template <class Node>
struct Proposal {
    Node target = 0;
    double candidate = 0.0;
    double predecessor_distance = 0.0;
    Node predecessor = 0;
};

template <class Node>
struct ProposalLess {
    bool operator()(const Proposal<Node> &a, const Proposal<Node> &b) const noexcept {
        if (a.target != b.target) {
            return a.target < b.target;
        }
        if (a.candidate != b.candidate) {
            return a.candidate < b.candidate;
        }
        if (a.predecessor_distance != b.predecessor_distance) {
            return a.predecessor_distance < b.predecessor_distance;
        }
        return a.predecessor < b.predecessor;
    }
};

struct DeltaSteppingStats {
    std::size_t buckets = 0;
    std::size_t light_rounds = 0;
    std::size_t light_relaxations = 0;
    std::size_t heavy_relaxations = 0;
    std::size_t proposals = 0;
    std::size_t accepted_updates = 0;
    std::size_t peak_batch_proposals = 0;

    void reset() noexcept {
        *this = {};
    }
};

template <class Node>
struct DeltaSteppingWorkspace {
    static_assert(std::is_unsigned_v<Node>);

    std::vector<double> distances;
    std::vector<Node> predecessors;
    std::vector<std::uint64_t> generations;
    std::uint64_t generation = 0;
    std::vector<Node> touched;

    std::map<std::uint64_t, std::vector<Node>> buckets;
    std::vector<Node> frontier;
    std::vector<Node> removed;
    std::vector<std::vector<Proposal<Node>>> proposal_buffers;

    void begin(const std::size_t n, const std::size_t n_threads) {
        distances.resize(n);
        predecessors.resize(n);
        if (generations.size() != n) {
            generations.assign(n, 0);
            generation = 0;
        }
        ++generation;
        if (generation == 0) {
            std::fill(generations.begin(), generations.end(), std::uint64_t{0});
            generation = 1;
        }
        touched.clear();
        buckets.clear();
        frontier.clear();
        removed.clear();
        proposal_buffers.resize(n_threads);
        for (auto &buffer : proposal_buffers) {
            buffer.clear();
        }
    }

    [[nodiscard]] bool known(const Node node) const noexcept {
        return generations[static_cast<std::size_t>(node)] == generation;
    }

};

template <class Node>
struct DeltaSteppingResult {
    bool completed = false;
    Node reached_target = kNoNode<Node>;
};

inline bool bucket_index(
    const double distance,
    const double delta,
    std::uint64_t &index
) noexcept {
    if (!(std::isfinite(distance) && std::isfinite(delta) && delta > 0.0)) {
        return false;
    }
    const long double quotient = std::floor(
        static_cast<long double>(distance) / static_cast<long double>(delta)
    );
    if (!(quotient >= 0.0L) ||
        quotient > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return false;
    }
    index = static_cast<std::uint64_t>(quotient);
    return true;
}

template <class Node>
inline void sort_unique(std::vector<Node> &nodes) {
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
}

template <class Node, class ForEachNeighbor>
bool relax_nodes(
    const std::span<const Node> nodes,
    const bool light,
    const double delta,
    const std::size_t requested_threads,
    const std::size_t maximum_degree,
    ForEachNeighbor &&for_each_neighbor,
    DeltaSteppingWorkspace<Node> &workspace,
    DeltaSteppingStats *stats
) {
    if (nodes.empty()) {
        return true;
    }
    if (nodes.size() < kParallelFrontierThreshold) {
        for (const auto source : nodes) {
            const auto source_index = static_cast<std::size_t>(source);
            const double source_distance = workspace.distances[source_index];
            bool valid = true;
            for_each_neighbor(
                source,
                [&](const Node target, const double weight) {
                    if (!valid || (weight <= delta) != light) {
                        return;
                    }
                    const double candidate = source_distance + weight;
                    if (!std::isfinite(candidate)) {
                        return;
                    }
                    const auto target_index = static_cast<std::size_t>(target);
                    const bool known = workspace.known(target);
                    if (known && !(candidate < workspace.distances[target_index])) {
                        return;
                    }
                    std::uint64_t target_bucket = 0;
                    if (!bucket_index(candidate, delta, target_bucket)) {
                        valid = false;
                        return;
                    }
                    if (!known) {
                        workspace.generations[target_index] = workspace.generation;
                        workspace.touched.push_back(target);
                    }
                    workspace.distances[target_index] = candidate;
                    workspace.predecessors[target_index] = source;
                    workspace.buckets[target_bucket].push_back(target);
                    if (stats != nullptr) {
                        ++stats->accepted_updates;
                    }
                }
            );
            if (!valid) {
                return false;
            }
        }
        return true;
    }
    const auto safe_degree = std::max<std::size_t>(1, maximum_degree);
    const auto batch_nodes = std::max<std::size_t>(
        1, kMaximumBatchProposals / safe_degree
    );
    const ProposalLess<Node> proposal_less;

    for (std::size_t batch_begin = 0; batch_begin < nodes.size();
         batch_begin += batch_nodes) {
        const auto batch_end = std::min(nodes.size(), batch_begin + batch_nodes);
        const auto batch_size = batch_end - batch_begin;
        const auto n_threads = batch_size < kParallelFrontierThreshold
            ? std::size_t{1}
            : bioimage_cpp::detail::normalize_thread_count(
                  requested_threads, batch_size
              );
        for (std::size_t thread = 0; thread < n_threads; ++thread) {
            workspace.proposal_buffers[thread].clear();
        }

        bioimage_cpp::detail::parallel_for_chunks(
            n_threads, batch_size,
            [&](const std::size_t thread, const std::size_t begin, const std::size_t end) {
                auto &proposals = workspace.proposal_buffers[thread];
                for (std::size_t i = begin; i < end; ++i) {
                    const auto source = nodes[batch_begin + i];
                    const auto source_index = static_cast<std::size_t>(source);
                    const double source_distance = workspace.distances[source_index];
                    for_each_neighbor(
                        source,
                        [&](const Node target, const double weight) {
                            if ((weight <= delta) != light) {
                                return;
                            }
                            const double candidate = source_distance + weight;
                            if (!std::isfinite(candidate)) {
                                return;
                            }
                            const auto target_index = static_cast<std::size_t>(target);
                            if (workspace.known(target) &&
                                !(candidate < workspace.distances[target_index])) {
                                return;
                            }
                            proposals.push_back({
                                target, candidate, source_distance, source
                            });
                        }
                    );
                }
            }
        );

        std::size_t proposal_count = 0;
        for (std::size_t thread = 0; thread < n_threads; ++thread) {
            auto &proposals = workspace.proposal_buffers[thread];
            std::sort(proposals.begin(), proposals.end(), proposal_less);
            proposal_count += proposals.size();
        }
        if (stats != nullptr) {
            stats->proposals += proposal_count;
            stats->peak_batch_proposals = std::max(
                stats->peak_batch_proposals, proposal_count
            );
        }

        struct Cursor {
            std::size_t thread = 0;
            std::size_t index = 0;
        };
        const auto cursor_greater = [&](const Cursor &a, const Cursor &b) {
            return proposal_less(
                workspace.proposal_buffers[b.thread][b.index],
                workspace.proposal_buffers[a.thread][a.index]
            );
        };
        std::priority_queue<
            Cursor, std::vector<Cursor>, decltype(cursor_greater)
        > cursors(cursor_greater);
        for (std::size_t thread = 0; thread < n_threads; ++thread) {
            if (!workspace.proposal_buffers[thread].empty()) {
                cursors.push({thread, 0});
            }
        }

        while (!cursors.empty()) {
            auto cursor = cursors.top();
            cursors.pop();
            auto best = workspace.proposal_buffers[cursor.thread][cursor.index];
            const auto target = best.target;

            auto advance = [&](Cursor current) {
                ++current.index;
                if (current.index < workspace.proposal_buffers[current.thread].size()) {
                    cursors.push(current);
                }
            };
            advance(cursor);
            while (!cursors.empty()) {
                const auto &next = workspace.proposal_buffers[
                    cursors.top().thread
                ][cursors.top().index];
                if (next.target != target) {
                    break;
                }
                cursor = cursors.top();
                cursors.pop();
                const auto &proposal = workspace.proposal_buffers[
                    cursor.thread
                ][cursor.index];
                if (proposal_less(proposal, best)) {
                    best = proposal;
                }
                advance(cursor);
            }

            const auto target_index = static_cast<std::size_t>(target);
            const bool known = workspace.known(target);
            if (known && !(best.candidate < workspace.distances[target_index])) {
                continue;
            }
            std::uint64_t target_bucket = 0;
            if (!bucket_index(best.candidate, delta, target_bucket)) {
                return false;
            }
            if (!known) {
                workspace.generations[target_index] = workspace.generation;
                workspace.touched.push_back(target);
            }
            workspace.distances[target_index] = best.candidate;
            workspace.predecessors[target_index] = best.predecessor;
            workspace.buckets[target_bucket].push_back(target);
            if (stats != nullptr) {
                ++stats->accepted_updates;
            }
        }
    }
    return true;
}

template <class Node, class ForEachNeighbor>
DeltaSteppingResult<Node> run(
    const std::size_t number_of_nodes,
    const std::span<const Node> sources,
    const std::span<const Node> targets,
    const double delta,
    const std::size_t requested_threads,
    const std::size_t maximum_degree,
    ForEachNeighbor &&for_each_neighbor,
    DeltaSteppingWorkspace<Node> &workspace,
    DeltaSteppingStats *stats = nullptr
) {
    static_assert(std::is_unsigned_v<Node>);
    if (number_of_nodes > static_cast<std::size_t>(kNoNode<Node>)) {
        return {};
    }
    const auto n_threads = bioimage_cpp::detail::normalize_thread_count(
        requested_threads, number_of_nodes
    );
    workspace.begin(number_of_nodes, n_threads);
    if (stats != nullptr) {
        stats->reset();
    }
    if (!(std::isfinite(delta) && delta > 0.0)) {
        return {};
    }

    std::vector<Node> ordered_sources(sources.begin(), sources.end());
    sort_unique(ordered_sources);
    for (const auto source : ordered_sources) {
        const auto index = static_cast<std::size_t>(source);
        workspace.generations[index] = workspace.generation;
        workspace.distances[index] = 0.0;
        workspace.predecessors[index] = source;
        workspace.touched.push_back(source);
        workspace.buckets[0].push_back(source);
    }

    while (!workspace.buckets.empty()) {
        const auto current_bucket = workspace.buckets.begin()->first;
        workspace.removed.clear();
        if (stats != nullptr) {
            ++stats->buckets;
        }

        while (true) {
            const auto current = workspace.buckets.find(current_bucket);
            if (current == workspace.buckets.end()) {
                break;
            }
            workspace.frontier = std::move(current->second);
            workspace.buckets.erase(current);
            workspace.frontier.erase(
                std::remove_if(
                    workspace.frontier.begin(), workspace.frontier.end(),
                    [&](const Node node) {
                        if (!workspace.known(node)) {
                            return true;
                        }
                        std::uint64_t index = 0;
                        return !bucket_index(
                                   workspace.distances[static_cast<std::size_t>(node)],
                                   delta,
                                   index
                               ) || index != current_bucket;
                    }
                ),
                workspace.frontier.end()
            );
            sort_unique(workspace.frontier);
            if (workspace.frontier.empty()) {
                continue;
            }
            workspace.removed.insert(
                workspace.removed.end(),
                workspace.frontier.begin(), workspace.frontier.end()
            );
            if (stats != nullptr) {
                ++stats->light_rounds;
            }
            if (!relax_nodes<Node>(
                    workspace.frontier, true, delta, n_threads, maximum_degree,
                    for_each_neighbor, workspace, stats)) {
                return {};
            }
        }

        sort_unique(workspace.removed);
        if (!relax_nodes<Node>(
                workspace.removed, false, delta, n_threads, maximum_degree,
                for_each_neighbor, workspace, stats)) {
            return {};
        }

        // A heavy FP64 addition can round back into the current bucket. Close
        // it before declaring a target final.
        if (workspace.buckets.contains(current_bucket)) {
            continue;
        }
        Node reached = kNoNode<Node>;
        double reached_distance = std::numeric_limits<double>::infinity();
        for (const auto node : targets) {
            if (!workspace.known(node)) {
                continue;
            }
            const double distance = workspace.distances[static_cast<std::size_t>(node)];
            std::uint64_t target_bucket = 0;
            if (!bucket_index(distance, delta, target_bucket) ||
                target_bucket != current_bucket) {
                continue;
            }
            if (distance < reached_distance ||
                (distance == reached_distance && node < reached)) {
                reached = node;
                reached_distance = distance;
            }
        }
        if (reached != kNoNode<Node>) {
            return {true, reached};
        }
    }
    return {true, kNoNode<Node>};
}

} // namespace bioimage_cpp::distance::detail_delta_stepping
