// Contract tests for dxapp::Progress (src/cpp_example/common/utility/progress.hpp).
// Build: g++ -std=c++17 -I src/cpp_example/common/utility tests/progress/test_progress.cpp -lpthread
//
// A front end that shows a running example (DX AI Studio's live view) could not tell
// how many frames had been processed: most runners print nothing per frame, and
// [DET]/[OBB]/[ISEG] print one line per OBJECT. With DXAPP_PROGRESS=1 every runner
// now prints "[PROGRESS] frames=N" at most once per interval and once at the end.
#include "progress.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_failures; } \
        else         { std::cout << "  ok  : " << (msg) << "\n"; }              \
    } while (0)

static std::string capture(void (*fn)()) {
    std::ostringstream buf;
    auto* old = std::cout.rdbuf(buf.rdbuf());
    fn();
    std::cout.rdbuf(old);
    return buf.str();
}

static int count(const std::string& s, const std::string& what) {
    int n = 0;
    for (size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) ++n;
    return n;
}

int main() {
    std::cout << "[test] off unless DXAPP_PROGRESS is set\n";
    unsetenv("DXAPP_PROGRESS");
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            for (int i = 0; i < 10; ++i) p.tick();
            p.finish();
        });
        CHECK(out.empty(), "nothing printed without the env var");
    }

    std::cout << "[test] every tick with interval 0, final line on finish\n";
    setenv("DXAPP_PROGRESS", "1", 1);
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            for (int i = 0; i < 3; ++i) p.tick();
            p.finish();
        });
        CHECK(count(out, "[PROGRESS] frames=") == 4, "three ticks + finish -> four lines");
        CHECK(out.find("[PROGRESS] frames=3\n") != std::string::npos, "the count is the number of frames");
    }

    std::cout << "[test] rate limited\n";
    {
        auto out = capture([] {
            dxapp::Progress p(3600.0);
            for (int i = 0; i < 1000; ++i) p.tick();
            p.finish();
        });
        CHECK(count(out, "[PROGRESS]") <= 2, "a long interval prints at most the first tick and finish");
        CHECK(out.find("frames=1000") != std::string::npos, "finish reports the total");
    }

    std::cout << "[test] concurrent ticks are all counted, lines never torn\n";
    {
        auto out = capture([] {
            dxapp::Progress p(0.0);
            std::vector<std::thread> ts;
            for (int t = 0; t < 8; ++t) ts.emplace_back([&p] { for (int i = 0; i < 500; ++i) p.tick(); });
            for (auto& t : ts) t.join();
            p.finish();
        });
        CHECK(out.find("frames=4000\n") != std::string::npos, "8 threads x 500 ticks = 4000");
        std::istringstream lines(out);
        std::string line;
        bool torn = false;
        while (std::getline(lines, line))
            if (line.rfind("[PROGRESS] frames=", 0) != 0) torn = true;
        CHECK(!torn, "every line is a whole [PROGRESS] line");
    }
    unsetenv("DXAPP_PROGRESS");

    if (g_failures) { std::cerr << "\n" << g_failures << " CHECK(s) FAILED\n"; return 1; }
    std::cout << "\nAll Progress contract tests PASSED\n";
    return 0;
}
