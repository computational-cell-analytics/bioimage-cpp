#pragma once

// Lightweight per-phase profiling utility. Active only when the translation
// unit is compiled with `-DBIOIMAGE_PROFILE`; otherwise every macro is a no-op
// and adds no overhead. Use it in development/profile-mode builds to find
// hotspots; do not enable it in production wheels.

// `NullProfiler` is always available; helper templates can request "no
// profiling here" via the same type regardless of build mode.
namespace bioimage_cpp::detail {
struct NullProfiler {};
} // namespace bioimage_cpp::detail

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

// Overload that accepts a NullProfiler and does nothing. Lets the same
// macros expand cleanly when a profiled translation unit calls into a helper
// that explicitly does not want to participate in profiling (e.g. work inside
// parallel workers, where we measure wall-clock at the dispatch level).
class ProfileTimerNull {
public:
    ProfileTimerNull(NullProfiler &, const char *) {}
};

inline ProfileTimer make_profile_timer(Profiler &profiler, const char *name) {
    return ProfileTimer(profiler, name);
}

inline ProfileTimerNull make_profile_timer(NullProfiler &profiler, const char *name) {
    return ProfileTimerNull(profiler, name);
}

} // namespace bioimage_cpp::detail

#define BIOIMAGE_PROFILE_INIT(var) ::bioimage_cpp::detail::Profiler var;
// Two-level indirection so __LINE__ is expanded before the token paste; without
// it every BIOIMAGE_PROFILE_SCOPE in a translation unit would declare the same
// identifier `_bp___LINE__`, breaking two scopes in one block.
#define BIOIMAGE_PROFILE_CONCAT_(a, b) a##b
#define BIOIMAGE_PROFILE_CONCAT(a, b) BIOIMAGE_PROFILE_CONCAT_(a, b)
#define BIOIMAGE_PROFILE_SCOPE(var, name) auto BIOIMAGE_PROFILE_CONCAT(_bp_, __LINE__) = ::bioimage_cpp::detail::make_profile_timer(var, name);
#define BIOIMAGE_PROFILE_REPORT(var) (var).report();

#else

#define BIOIMAGE_PROFILE_INIT(var) ::bioimage_cpp::detail::NullProfiler var;
#define BIOIMAGE_PROFILE_SCOPE(var, name) (void)var;
#define BIOIMAGE_PROFILE_REPORT(var) (void)var;

#endif
