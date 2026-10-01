// Contract tests for dxapp::DisplayPump (src/cpp_example/common/utility/display_pump.hpp).
// Compiled against the mock OpenCV in tests/display/mock_include.
#include <opencv2/opencv.hpp>
#include "display_pump.hpp"
#include "display_size.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

static int g_failures = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_failures; } \
        else         { std::cout << "  ok  : " << (msg) << "\n"; }            \
    } while (0)

using dxapp::DisplayPump;

// I2: producers never block, only the newest frame survives.
static void test_offer_is_lossy_and_latest_wins() {
    std::cout << "[test] offer() is lossy, depth 1, latest-wins\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);  // 0 => no rate limit, show every pump

    cv::Mat a(8, 8), b(8, 8), c(8, 8);
    pump.offer(a);
    pump.offer(b);
    pump.offer(c);
    CHECK(pump.dropped() == 2, "two stale frames dropped before any pump");

    CHECK(pump.pump(), "pump() returns true (no quit)");
    CHECK(pump.shown() == 1, "exactly one frame shown");
    CHECK(cvmock::counters().imshow.load() == 1, "imshow called once");
}

// I2: offer() must not deep-copy pixels (cv::Mat is refcounted).
static void test_offer_does_not_copy_pixels() {
    std::cout << "[test] offer() does not deep-copy pixels\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    int before = cv::Mat::clone_count();
    cv::Mat f(1080, 1920);
    for (int i = 0; i < 500; ++i) pump.offer(f);
    CHECK(cv::Mat::clone_count() == before, "500 offers caused zero clone()");
}

// I3: exactly one event pump per tick, window created once.
static void test_single_event_pump_per_tick() {
    std::cout << "[test] one waitKey per pump, namedWindow once\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    for (int i = 0; i < 10; ++i) { pump.offer(cv::Mat(8, 8)); pump.pump(); }
    CHECK(cvmock::counters().wait_key.load() == 10, "waitKey called exactly 10x for 10 pumps");
    CHECK(cvmock::counters().named_window.load() == 1, "namedWindow called exactly once");
    CHECK(cvmock::counters().imshow.load() == 10, "imshow called 10x");
}

// Quit handling must survive: 'q' / ESC / window closed.
static void test_quit_detection() {
    std::cout << "[test] quit detection\n";
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().next_key = 'q';
        CHECK(!pump.pump(), "'q' stops the pump");
        CHECK(pump.stopped(), "pump reports stopped after 'q'");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().next_key = 27;
        CHECK(!pump.pump(), "ESC stops the pump");
    }
    {
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(8, 8)); pump.pump();
        cvmock::counters().window_visible = 0.0;  // user closed the window
        CHECK(!pump.pump(), "window close stops the pump");
    }
}

// A backend that always returns -1 for WND_PROP_VISIBLE (e.g. GTK2) must not
// be mistaken for "user closed the window".
static void test_backend_without_window_property() {
    std::cout << "[test] backend returning -1 for WND_PROP_VISIBLE is tolerated\n";
    cvmock::counters().reset();
    cvmock::counters().window_visible = -1.0;
    DisplayPump pump("Output", 0.0);
    pump.offer(cv::Mat(8, 8));
    CHECK(pump.pump(), "first pump survives probe");
    pump.offer(cv::Mat(8, 8));
    CHECK(pump.pump(), "second pump still alive (probe disabled the check)");
}

// wouldShow() reserves one preview slot so producers do not all draw.
static void test_would_show_reserves_slot() {
    std::cout << "[test] wouldShow reserves a single preview slot\n";
    if (!dxapp::hasDisplay()) {
        std::cout << "  ok  : skipped (no display server)\n";
        return;
    }
    DisplayPump pump("Output", 50.0);  // 20 ms between admitted frames
    CHECK(pump.wouldShow(), "first slot is open");
    CHECK(!pump.wouldShow(), "second call inside the interval is refused");
}

// I3: rate limiting caps imshow frequency without capping the pump loop.
static void test_rate_limit() {
    std::cout << "[test] rate limit caps imshow, not the event pump\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 50.0);  // 50 Hz => >= 20 ms between imshow
    auto t0 = std::chrono::steady_clock::now();
    int pumps = 0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(100)) {
        pump.offer(cv::Mat(8, 8));
        pump.pump();
        ++pumps;
    }
    int shown = pump.shown();
    CHECK(pumps > shown, "pump loop ran more often than imshow");
    CHECK(shown <= 8, "imshow rate-limited to ~50 Hz over 100 ms (got <=8)");
    CHECK(shown >= 2, "imshow still happened (got >=2)");
}

// I1: the headline property — a slow GUI must not slow the producer down.
static void test_slow_display_does_not_throttle_producer() {
    std::cout << "[test] slow display does not throttle the producer\n";
    cvmock::counters().reset();
    DisplayPump pump("Output", 0.0);
    std::atomic<bool> done{false};
    std::atomic<long> produced{0};

    std::thread producer([&] {
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(200)) {
            pump.offer(cv::Mat(64, 64));
            produced.fetch_add(1, std::memory_order_relaxed);
        }
        done.store(true);
    });

    // Consumer deliberately far slower than the producer (10 ms per frame).
    while (!done.load()) {
        pump.pump();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    producer.join();

    long n = produced.load();
    std::cout << "    produced=" << n << " shown=" << pump.shown()
              << " dropped=" << pump.dropped() << "\n";
    CHECK(n > 5000, "producer ran freely (>5000 offers in 200 ms despite a 10 ms display)");
    CHECK(pump.shown() < 40, "display only kept up with its own rate");
    CHECK(pump.dropped() > 0, "stale frames were dropped, not queued");
}

// runPipelineWithDisplay(): the control-flow primitive the runners now rely on.
// The pipeline must run to completion on a worker thread while this thread pumps,
// and a user quit must tear the pipeline down rather than leak the worker.
static void test_run_pipeline_with_display() {
    std::cout << "[test] runPipelineWithDisplay drives pipeline off the GUI thread\n";
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        std::atomic<long> iters{0};
        std::atomic<bool> finished{false};
        std::thread::id gui_tid = std::this_thread::get_id();
        std::atomic<bool> ran_on_other_thread{false};

        dxapp::runPipelineWithDisplay(
            [&] {
                ran_on_other_thread.store(std::this_thread::get_id() != gui_tid);
                for (int i = 0; i < 2000; ++i) {
                    pump.offer(cv::Mat(32, 32));
                    iters.fetch_add(1, std::memory_order_relaxed);
                }
                finished.store(true);
            },
            pump, /*no_display=*/false, [] {});

        CHECK(ran_on_other_thread.load(), "pipeline ran on a worker thread, not the GUI thread");
        CHECK(finished.load(), "pipeline ran to completion");
        CHECK(iters.load() == 2000, "every iteration executed");
        CHECK(pump.shown() > 0, "GUI thread showed frames meanwhile");
    }
    {   // user quit must stop the pipeline and still join the worker
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> stop_requested{false};
        std::atomic<bool> worker_exited{false};
        cvmock::counters().next_key = 'q';

        dxapp::runPipelineWithDisplay(
            [&] {
                while (!stop_requested.load()) {
                    pump.offer(cv::Mat(32, 32));
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                worker_exited.store(true);
            },
            pump, /*no_display=*/false,
            [&] { stop_requested.store(true); });

        CHECK(stop_requested.load(), "user quit propagated to the pipeline");
        CHECK(worker_exited.load(), "worker was joined, not leaked");
    }
    {   // --no-display: no GUI calls at all
        cvmock::counters().reset();
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> done{false};
        dxapp::runPipelineWithDisplay([&] { done.store(true); }, pump,
                                      /*no_display=*/true, [] {});
        CHECK(done.load(), "pipeline ran with no_display");
        CHECK(cvmock::counters().imshow.load() == 0, "no imshow in --no-display");
        CHECK(cvmock::counters().wait_key.load() == 0, "no waitKey in --no-display");
    }
}

// Headless: the pump must go inert WITHOUT reporting a user quit, or
// runPipelineWithDisplay() would tear the inference pipeline down.
static void test_headless_is_inert_not_quit() {
    std::cout << "[test] headless pump is inert, not a quit signal\n";
    cvmock::counters().reset();
    unsetenv("DISPLAY");
    unsetenv("WAYLAND_DISPLAY");
    CHECK(!dxapp::hasDisplay(), "hasDisplay() false with no DISPLAY/WAYLAND_DISPLAY");
    {
        DisplayPump pump("Output", 0.0);
        CHECK(!pump.guiAvailable(), "pump reports no GUI");
        for (int i = 0; i < 20; ++i) { pump.offer(cv::Mat(8, 8)); }
        bool alive = true;
        for (int i = 0; i < 20; ++i) alive = alive && pump.pump();
        CHECK(alive, "pump() keeps returning true (NOT a user quit)");
        CHECK(!pump.stopped(), "pump is not 'stopped'");
        CHECK(cvmock::counters().named_window.load() == 0, "namedWindow never called");
        CHECK(cvmock::counters().imshow.load() == 0, "imshow never called");
        CHECK(cvmock::counters().wait_key.load() == 0, "waitKey never called");
        CHECK(pump.shown() == 0, "nothing shown");
    }
    // A headless pipeline must still run to completion.
    {
        DisplayPump pump("Output", 0.0);
        std::atomic<bool> finished{false};
        dxapp::runPipelineWithDisplay([&] { finished.store(true); }, pump,
                                      /*no_display=*/false, [] {});
        CHECK(finished.load(), "pipeline completed even with display requested but absent");
    }
    setenv("DISPLAY", ":99", 1);   // restore for the GUI tests
    CHECK(dxapp::hasDisplay(), "hasDisplay() true again once DISPLAY is set");
}

// The async runners never called setInitialWindowSize, so their window opened at the
// backend default (400x300 under Qt) while the sync runners sized theirs to half the
// screen. Without a hint the pump now applies the same rule as showOutput().
static void test_window_defaults_to_half_the_screen() {
    std::cout << "[test] no hint -> half the screen, frame aspect kept\n";
    auto a = dxapp::fitHalfScreen(1920, 1080, 1280, 720);
    CHECK(a.first == 640 && a.second == 360, "16:9 frame on 1280x720 -> 640x360");
    auto b = dxapp::fitHalfScreen(480, 640, 1280, 720);
    CHECK(b.first == 270 && b.second == 360, "portrait frame keeps its aspect -> 270x360");

    setenv("DXAPP_SCREEN_W", "1280", 1);
    setenv("DXAPP_SCREEN_H", "720", 1);
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().resize_window.load() == 1, "window sized once");
        CHECK(cvmock::counters().last_resize_w.load() == 640 && cvmock::counters().last_resize_h.load() == 360,
              "sized to 640x360");
    }
    cvmock::counters().reset();
    {
        DisplayPump pump("Output", 0.0);
        pump.setInitialWindowSize(300, 200);
        pump.offer(cv::Mat(1080, 1920)); pump.pump();
        CHECK(cvmock::counters().last_resize_w.load() == 300 && cvmock::counters().last_resize_h.load() == 200,
              "an explicit hint still wins");
    }
    unsetenv("DXAPP_SCREEN_W");
    unsetenv("DXAPP_SCREEN_H");
}

int main() {
    // The GUI contract tests below exercise the display path, so pin DISPLAY
    // on: otherwise results would depend on whether the machine running the
    // tests happens to have a display server.
    setenv("DISPLAY", ":99", 1);

    test_offer_is_lossy_and_latest_wins();
    test_offer_does_not_copy_pixels();
    test_single_event_pump_per_tick();
    test_quit_detection();
    test_backend_without_window_property();
    test_would_show_reserves_slot();
    test_rate_limit();
    test_slow_display_does_not_throttle_producer();
    test_run_pipeline_with_display();
    test_headless_is_inert_not_quit();
    test_window_defaults_to_half_the_screen();

    if (g_failures) { std::cerr << "\n" << g_failures << " CHECK(s) FAILED\n"; return 1; }
    std::cout << "\nAll DisplayPump contract tests PASSED\n";
    return 0;
}
