#pragma once

// Lightweight per-phase profiling utility. Active only when the translation
// unit is compiled with `-DBIOIMAGE_PROFILE`; otherwise every macro is a no-op
// and adds no overhead. Use it in development/profile-mode builds to find
// hotspots; do not enable it in production wheels.

#ifdef BIOIMAGE_PROFILE
#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::detail {

class Profiler {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;

    void accumulate(const char *name, const Duration delta) {
        auto it = totals_.find(name);
        if (it == totals_.end()) {
            order_.push_back(name);
            totals_.emplace(name, delta.count());
        } else {
            it->second += delta.count();
        }
    }

    void report() const {
        std::fprintf(stderr, "[bioimage profile]\n");
        double total = 0.0;
        for (const auto *name : order_) {
            total += totals_.at(name);
        }
        for (const auto *name : order_) {
            const auto seconds = totals_.at(name);
            const auto fraction = total > 0.0 ? (100.0 * seconds / total) : 0.0;
            std::fprintf(stderr, "  %-22s %8.4f s  (%5.1f%%)\n", name, seconds, fraction);
        }
        std::fprintf(stderr, "  %-22s %8.4f s\n", "total", total);
    }

private:
    std::vector<const char *> order_;
    std::unordered_map<const char *, double> totals_;
};

class ProfileTimer {
public:
    ProfileTimer(Profiler &profiler, const char *name)
        : profiler_(profiler), name_(name), start_(Profiler::Clock::now()) {}

    ~ProfileTimer() {
        profiler_.accumulate(name_, Profiler::Clock::now() - start_);
    }

private:
    Profiler &profiler_;
    const char *name_;
    Profiler::Clock::time_point start_;
};

} // namespace bioimage_cpp::detail

#define BIOIMAGE_PROFILE_INIT(var) ::bioimage_cpp::detail::Profiler var;
#define BIOIMAGE_PROFILE_SCOPE(var, name) ::bioimage_cpp::detail::ProfileTimer _bp_##__LINE__(var, name);
#define BIOIMAGE_PROFILE_REPORT(var) (var).report();

#else

namespace bioimage_cpp::detail {
struct NullProfiler {};
} // namespace bioimage_cpp::detail

#define BIOIMAGE_PROFILE_INIT(var) ::bioimage_cpp::detail::NullProfiler var;
#define BIOIMAGE_PROFILE_SCOPE(var, name) (void)var;
#define BIOIMAGE_PROFILE_REPORT(var) (void)var;

#endif
