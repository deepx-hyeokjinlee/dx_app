/**
 * @file progress.hpp
 * @brief Machine-readable progress for front ends: "[PROGRESS] frames=N".
 *
 * A front end showing a running example (DX AI Studio's live view) cannot tell how
 * many frames have been processed: most runners print nothing per frame, and the
 * [DET] / [OBB] / [ISEG] tags print one line per OBJECT. With DXAPP_PROGRESS=1 every
 * runner calls progressTick() once per completed frame and this prints the running
 * total at most once per second, plus once when the process ends. Without the
 * variable nothing is printed and the cost is one branch.
 *
 * stdoutMutex() is shared by everything that writes pipeline-parsed lines, so lines
 * from concurrent async callbacks are written whole ("3.50273.5913" was two scores
 * from two callbacks interleaved).
 *
 * Contract: tests/progress/test_progress.cpp
 */
#ifndef DXAPP_PROGRESS_HPP
#define DXAPP_PROGRESS_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace dxapp {

/** Serialises writes of pipeline-parsed stdout lines across threads. */
inline std::mutex& stdoutMutex() {
    static std::mutex m;
    return m;
}

class Progress {
public:
    explicit Progress(double interval_s = 1.0)
        : interval_ns_(static_cast<int64_t>(interval_s * 1e9)), enabled_(envEnabled()) {}
    ~Progress() { finish(); }
    Progress(const Progress&) = delete;
    Progress& operator=(const Progress&) = delete;

    /** One frame finished. Thread-safe. */
    void tick() {
        if (!enabled_) return;
        const long n = ++frames_;
        const int64_t t = nowNs();
        int64_t last = last_ns_.load(std::memory_order_relaxed);
        if (last != 0 && t - last < interval_ns_) return;
        if (last_ns_.compare_exchange_strong(last, t)) print(n);
    }

    /** Final total; printed once. */
    void finish() {
        if (!enabled_ || finished_.exchange(true)) return;
        print(frames_.load());
    }

    long frames() const { return frames_.load(); }

private:
    static bool envEnabled() {
        const char* v = std::getenv("DXAPP_PROGRESS");
        return v && *v && std::strcmp(v, "0") != 0;
    }
    static int64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static void print(long n) {
        std::lock_guard<std::mutex> lock(stdoutMutex());
        std::cout << "[PROGRESS] frames=" << n << '\n' << std::flush;
    }

    const int64_t interval_ns_;
    const bool enabled_;
    std::atomic<long> frames_{0};
    std::atomic<int64_t> last_ns_{0};
    std::atomic<bool> finished_{false};
};

/** The process-wide counter the runners tick. */
inline Progress& progress() {
    static Progress p(1.0);
    return p;
}

inline void progressTick() { progress().tick(); }

}  // namespace dxapp

#endif  // DXAPP_PROGRESS_HPP
