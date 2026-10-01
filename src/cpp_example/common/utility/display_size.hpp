/**
 * @file display_size.hpp
 * @brief Where a preview window should open: half the screen, frame aspect kept.
 *
 * Shared by the sync path (showOutput in common_util.hpp) and the async path
 * (DisplayPump). The async runners never passed DisplayPump a size, so their
 * window opened at the backend default (400x300 under Qt) while the sync runners
 * used half the screen. Kept free of OpenCV so the DisplayPump unit test can
 * still build against its mock.
 */
#ifndef DXAPP_DISPLAY_SIZE_HPP
#define DXAPP_DISPLAY_SIZE_HPP

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace dxapp {

/**
 * @brief Screen resolution in pixels.
 *
 * Tries (in order):
 *   1. Environment variables DXAPP_SCREEN_W / DXAPP_SCREEN_H
 *   2. xdpyinfo (X11) parsing "dimensions: WxH"
 * Falls back to 1920x1080 if detection fails.
 */
inline std::pair<int, int> screenResolution() {
    const char* env_w = std::getenv("DXAPP_SCREEN_W");
    const char* env_h = std::getenv("DXAPP_SCREEN_H");
    if (env_w && env_h) {
        int w = std::atoi(env_w);
        int h = std::atoi(env_h);
        if (w > 0 && h > 0) return {w, h};
    }
#ifndef _WIN32
    FILE* pipe = popen("xdpyinfo 2>/dev/null | grep dimensions", "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            int w = 0, h = 0;
            if (sscanf(buf, " dimensions: %dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                pclose(pipe);
                return {w, h};
            }
        }
        pclose(pipe);
    }
#endif
    return {1920, 1080};
}

/**
 * @brief Fit a frame_w x frame_h frame into half of a screen_w x screen_h screen,
 *        keeping the frame's aspect ratio. Returns {0, 0} for an empty frame.
 */
inline std::pair<int, int> fitHalfScreen(int frame_w, int frame_h, int screen_w, int screen_h) {
    if (frame_w <= 0 || frame_h <= 0 || screen_w <= 0 || screen_h <= 0) return {0, 0};
    const double scale = std::min(static_cast<double>(screen_w / 2) / frame_w,
                                  static_cast<double>(screen_h / 2) / frame_h);
    return {static_cast<int>(frame_w * scale), static_cast<int>(frame_h * scale)};
}

}  // namespace dxapp

#endif  // DXAPP_DISPLAY_SIZE_HPP
