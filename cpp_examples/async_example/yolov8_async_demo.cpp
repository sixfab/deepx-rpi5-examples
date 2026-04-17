// ============================================================
// yolov8_async_demo.cpp  --  YOLOv8 async producer/consumer demo
// Two threads: producer (read+letterbox+RunAsync) and the main
// thread (Wait+decode+draw+imshow). Geometry travels with each
// request so multiple frames can be in flight without stomping
// on each other. HighGUI stays on the main thread where the
// window was created. Queue depth 2 keeps the NPU fed without
// memory bloat. Trade-off: higher throughput, extra latency.
// Usage: ./yolov8_async_demo [--model path] [--source video|camera]
//                           [--path file] [--conf 0.3] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/highgui.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <thread>

namespace {

struct PendingRequest {
    int                 reqId;
    cv::Mat             frame;      // original BGR frame (owns pixels)
    cv::Mat             inputTensor;// keeps letterbox buffer alive until NPU done
    float               gain;
    cv::Point2f         pad;
    cv::Size            origSize;
};

class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : m_cap(cap) {}

    bool push(PendingRequest&& item) {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_notFull.wait(lk, [&]{ return m_q.size() < m_cap || m_stop; });
        if (m_stop) return false;
        m_q.push(std::move(item));
        m_notEmpty.notify_one();
        return true;
    }

    bool pop(PendingRequest& out) {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_notEmpty.wait(lk, [&]{ return !m_q.empty() || m_done; });
        if (m_q.empty()) return false;
        out = std::move(m_q.front());
        m_q.pop();
        m_notFull.notify_one();
        return true;
    }

    void markDone()  { std::lock_guard<std::mutex> lk(m_mtx); m_done = true; m_notEmpty.notify_all(); }
    void requestStop(){ std::lock_guard<std::mutex> lk(m_mtx); m_stop = true; m_notFull.notify_all(); m_notEmpty.notify_all(); }

private:
    size_t                      m_cap;
    std::queue<PendingRequest>  m_q;
    std::mutex                  m_mtx;
    std::condition_variable     m_notFull, m_notEmpty;
    bool                        m_done = false;
    bool                        m_stop = false;
};

} // namespace

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    int inputH = p.inputHeight, inputW = p.inputWidth;
    auto engine = sdk::loadEngine(p.modelPath, inputH, inputW);
    if (!engine) { std::fprintf(stderr, "[ERROR] Failed to load model\n"); return 1; }

    InputSource source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    BoundedQueue queue(/*cap=*/2);
    std::atomic<bool> stopFlag{false};

    std::thread producer([&]() {
        cv::Mat frame;
        while (!stopFlag.load()) {
            if (!source.read(frame) || frame.empty()) break;
            PendingRequest r;
            r.frame    = frame.clone();
            auto lb    = sdk::letterbox(r.frame, inputH, inputW);
            r.inputTensor = lb.image;
            r.gain     = lb.gain;
            r.pad      = lb.pad;
            r.origSize = r.frame.size();
            r.reqId    = engine->RunAsync(r.inputTensor.data, nullptr, nullptr);
            if (!queue.push(std::move(r))) break;
        }
        queue.markDone();
    });

    std::fprintf(stderr, "[INFO] Async pipeline started. Press 'q' or ESC to quit.\n");
    cv::namedWindow(p.windowTitle, cv::WINDOW_NORMAL);

    auto tLast = std::chrono::steady_clock::now();
    double emaFps = 0.0;

    PendingRequest r;
    while (queue.pop(r)) {
        auto outs = engine->Wait(r.reqId);
        auto dets = ppu::decodeYolov8Ppu(outs, inputW, inputH, p.confThreshold, p.iouThreshold);

        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        for (const auto& d : dets) boxes.push_back(d.box);
        auto boxesPx = sdk::unletterboxBoxes(boxes, r.gain, r.pad, r.origSize);
        for (size_t i = 0; i < dets.size(); ++i) {
            scores.push_back(dets[i].score);
            classIds.push_back(dets[i].classId);
        }
        vis::drawDetections(r.frame, boxesPx, scores, classIds, labs);

        auto tNow = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(tNow - tLast).count();
        tLast = tNow;
        if (dt > 0.0) emaFps = emaFps ? 0.9 * emaFps + 0.1 / dt : 1.0 / dt;
        if (p.showFps) vis::drawFps(r.frame, emaFps);

        cv::imshow(p.windowTitle, r.frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { stopFlag.store(true); queue.requestStop(); break; }
    }

    stopFlag.store(true);
    queue.requestStop();
    if (producer.joinable()) producer.join();
    cv::destroyAllWindows();
    return 0;
}
