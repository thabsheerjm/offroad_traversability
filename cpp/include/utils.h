#pragma once
#include <chrono>

inline float get_fps(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end) {
    return 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

