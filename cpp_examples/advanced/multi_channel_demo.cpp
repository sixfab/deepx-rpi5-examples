// multi_channel_demo.cpp  --  N video sources through one shared
// InferenceEngine with a mutex around every RunAsync/Wait pair,
// composited into a grid window. Ports MultiChannelAdapter.cpp
// (single-engine, per-channel worker threads; main thread owns
// HighGUI and the compositor).

#include "config_loader.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int kCellW = 480;
constexpr int kCellH = 270;

struct ChannelState {
    int         index;
    std::string path;
    std::mutex  mtx;         // guards frame
    cv::Mat     frame;       // last rendered tile (kCellH x kCellW BGR)
    std::string error;
    std::thread worker;
};

cv::Mat placeholderTile(int index, const std::string& msg) {
    cv::Mat tile(kCellH, kCellW, CV_8UC3, cv::Scalar(20, 20, 20));
    char buf[96];
    std::snprintf(buf, sizeof(buf), "CH %d: %s", index, msg.c_str());
    cv::putText(tile, buf, cv::Point(20, kCellH / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(200, 200, 200), 2, cv::LINE_AA);
    return tile;
}

void channelLoop(ChannelState&             state,
                 dxrt::InferenceEngine&    engine,
                 std::mutex&               engineMtx,
                 std::atomic<bool>&        stop,
                 int inputH, int inputW,
                 float conf, float iou,
                 const std::vector<std::string>& labs)
{
    InputSource src(SourceType::VIDEO, state.path);
    if (!src.isOpened()) {
        std::lock_guard<std::mutex> lk(state.mtx);
        state.error = "open failed";
        return;
    }
    const auto anchors = ppu::defaultYolov5Anchors();
    while (!stop.load()) {
        cv::Mat bgr;
        if (!src.read(bgr) || bgr.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }
        auto lb = sdk::letterbox(bgr, inputH, inputW);

        dxrt::TensorPtrs outs;
        try {
            std::lock_guard<std::mutex> engineLk(engineMtx);
            auto reqId = engine.RunAsync(lb.image.data, nullptr, nullptr);
            outs = engine.Wait(reqId);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ch%d] inference: %s\n", state.index, e.what());
            continue;
        }

        auto dets = ppu::decodeYolov5Ppu(outs, inputW, inputH, conf, iou, anchors);
        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        for (const auto& d : dets) boxes.push_back(d.box);
        auto px = sdk::unletterboxBoxes(boxes, lb.gain, lb.pad, bgr.size());
        for (size_t i = 0; i < dets.size(); ++i) {
            scores.push_back(dets[i].score);
            classIds.push_back(dets[i].classId);
        }
        vis::drawDetections(bgr, px, scores, classIds, labs);

        cv::Mat tile;
        cv::resize(bgr, tile, cv::Size(kCellW, kCellH));
        char tag[16];
        std::snprintf(tag, sizeof(tag), "CH %d", state.index);
        cv::putText(tile, tag, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        {
            std::lock_guard<std::mutex> lk(state.mtx);
            state.frame = std::move(tile);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    if (p.channelSources.empty()) {
        std::fprintf(stderr, "[ERROR] No channels in config.\n");
        return 1;
    }

    int inputH = p.inputHeight, inputW = p.inputWidth;
    auto engine = sdk::loadEngine(p.modelPath, inputH, inputW);
    if (!engine) { std::fprintf(stderr, "[ERROR] model load failed\n"); return 1; }

    std::mutex         engineMtx;
    std::atomic<bool>  stop{false};

    std::vector<std::unique_ptr<ChannelState>> channels;
    channels.reserve(p.channelSources.size());
    for (size_t i = 0; i < p.channelSources.size(); ++i) {
        auto s = std::make_unique<ChannelState>();
        s->index = static_cast<int>(i);
        s->path  = p.channelSources[i];
        channels.push_back(std::move(s));
    }
    for (auto& ch : channels) {
        ch->worker = std::thread(channelLoop,
            std::ref(*ch), std::ref(*engine), std::ref(engineMtx), std::ref(stop),
            inputH, inputW, p.confThreshold, p.iouThreshold, std::cref(labs));
    }

    const int n       = static_cast<int>(channels.size());
    const int cols    = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    const int rows    = (n + cols - 1) / cols;
    const int gridW   = cols * kCellW;
    const int gridH   = rows * kCellH;

    cv::namedWindow(p.windowTitle, cv::WINDOW_NORMAL);
    cv::Mat composite(gridH, gridW, CV_8UC3, cv::Scalar(0, 0, 0));

    std::fprintf(stderr, "[INFO] %d channels -> %dx%d grid. Press q or ESC.\n",
                 n, cols, rows);
    while (true) {
        composite.setTo(cv::Scalar(0, 0, 0));
        for (int i = 0; i < n; ++i) {
            const int x0 = (i % cols) * kCellW;
            const int y0 = (i / cols) * kCellH;
            cv::Rect roi(x0, y0, kCellW, kCellH);
            cv::Mat  tile;
            {
                std::lock_guard<std::mutex> lk(channels[i]->mtx);
                if (!channels[i]->error.empty()) {
                    tile = placeholderTile(i, channels[i]->error);
                } else if (channels[i]->frame.empty()) {
                    tile = placeholderTile(i, "warming up");
                } else {
                    tile = channels[i]->frame.clone();
                }
            }
            tile.copyTo(composite(roi));
        }
        cv::imshow(p.windowTitle, composite);
        int key = cv::waitKey(16) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    stop.store(true);
    for (auto& ch : channels) if (ch->worker.joinable()) ch->worker.join();
    cv::destroyAllWindows();
    return 0;
}
