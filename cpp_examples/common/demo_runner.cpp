#include "demo_runner.h"

#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "fps_counter.h"
#include "visualizer.h"

void runDemo(InputSource& source,
             InferAndDrawFn inferAndDraw,
             const DemoConfig& cfg)
{
    FpsCounter fps;
    cv::Mat frame;
    cv::Mat displayFrame;

    std::printf("[INFO] Streaming — press 'q' or ESC in the window to quit.\n");
    cv::namedWindow(cfg.windowTitle, cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!source.read(frame)) {
            if (cfg.loop) {
                source.rewind();
                if (!source.read(frame)) {
                    std::printf("[INFO] Input source exhausted.\n");
                    break;
                }
            } else {
                std::printf("[INFO] Input source exhausted.\n");
                break;
            }
        }
        if (frame.empty()) continue;

        try {
            inferAndDraw(frame);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[WARN] Inference failed on this frame: %s\n", e.what());
            continue;
        }

        fps.update();
        if (cfg.showFps) vis::drawFps(frame, fps.getFps());

        if (cfg.displayWidth > 0 && cfg.displayHeight > 0 &&
            (frame.cols != cfg.displayWidth || frame.rows != cfg.displayHeight))
        {
            cv::resize(frame, displayFrame,
                       cv::Size(cfg.displayWidth, cfg.displayHeight));
            cv::imshow(cfg.windowTitle, displayFrame);
        } else {
            cv::imshow(cfg.windowTitle, frame);
        }

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::printf("[INFO] Quit requested by user.\n");
            break;
        }
    }

    source.release();
    cv::destroyAllWindows();
}
