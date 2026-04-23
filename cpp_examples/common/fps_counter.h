#pragma once

#include <chrono>
#include <deque>

// Rolling-window FPS counter. Call update() once per rendered frame;
// getFps() returns an average over the last `windowSize` intervals.
class FpsCounter {
public:
    explicit FpsCounter(int windowSize = 30);
    void update();
    double getFps() const;

private:
    using Clock = std::chrono::steady_clock;
    int m_windowSize;
    std::deque<double> m_intervals;
    Clock::time_point m_last;
    bool m_hasLast = false;
};
