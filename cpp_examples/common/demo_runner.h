#pragma once

#include <functional>
#include <string>
#include <opencv2/core.hpp>

#include "input_source.h"

struct DemoConfig {
    std::string windowTitle = "DeepX Demo";
    bool showFps            = true;
    bool loop               = true;    // auto-rewind VIDEO sources at EOF
    int displayWidth        = 0;       // 0 = show at native resolution
    int displayHeight       = 0;
};

// Mutates `frame` in place: runs inference and draws results on the frame.
using InferAndDrawFn = std::function<void(cv::Mat& frame)>;

// Frame-loop driver. Reads frames from `source`, calls `inferAndDraw`,
// overlays FPS, and displays with cv::imshow. Exits on 'q' or ESC.
void runDemo(InputSource& source,
             InferAndDrawFn inferAndDraw,
             const DemoConfig& cfg = {});
