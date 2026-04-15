#pragma once
#include <chrono>

inline float get_fps(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end) {
    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    return (ms > 0.0f) ? 1000.0f / ms : 0.0f;
}

