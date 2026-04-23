#include "fps_counter.h"

FpsCounter::FpsCounter(int windowSize) : m_windowSize(windowSize) {}

void FpsCounter::update() {
    const auto now = Clock::now();
    if (m_hasLast) {
        const double dt = std::chrono::duration<double>(now - m_last).count();
        m_intervals.push_back(dt);
        if (static_cast<int>(m_intervals.size()) > m_windowSize) {
            m_intervals.pop_front();
        }
    }
    m_last = now;
    m_hasLast = true;
}

double FpsCounter::getFps() const {
    if (m_intervals.empty()) return 0.0;
    double total = 0.0;
    for (double dt : m_intervals) total += dt;
    return (total > 0.0)
        ? static_cast<double>(m_intervals.size()) / total
        : 0.0;
}
