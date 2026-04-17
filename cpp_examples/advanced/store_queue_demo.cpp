// store_queue_demo.cpp  --  YOLOv8 PPU + centroid tracker +
// multi-zone wait-time colouring. Mirrors python_examples/advanced/
// store_queue_analysis_demo.py exactly: a person inside *any* configured
// region starts a wall-clock timer on entry and is boxed in green /
// yellow / red based on how long they have been waiting. Leaving all
// regions resets the timer so a later re-entry counts from zero.

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "tracker.h"
#include "visualizer.h"
#include "zone_utils.h"

#include <dxrt/dxrt_api.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kPersonClassId = 0;

using Clock     = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct QueueState {
    bool      inQueue    = false;
    TimePoint enterTime  = {};
};

struct Thresholds {
    float greenSec  = 7.0f;
    float yellowSec = 15.0f;
};

struct ExtraParams {
    Thresholds thresh;
    int   maxMissingFrames = 10;
    float maxDistance      = 0.1f;
};

// Parse the three fields the shared config loader doesn't know about.
ExtraParams loadExtras(const std::string& configPath) {
    ExtraParams e;
    namespace fs = std::filesystem;
    if (configPath.empty() || !fs::exists(configPath)) return e;
    try {
        std::ifstream in(configPath);
        nlohmann::json j; in >> j;
        if (auto it = j.find("wait_thresholds"); it != j.end() && it->is_object()) {
            if (auto g = it->find("green");  g != it->end() && g->is_number())
                e.thresh.greenSec  = g->get<float>();
            if (auto y = it->find("yellow"); y != it->end() && y->is_number())
                e.thresh.yellowSec = y->get<float>();
        }
        if (auto it = j.find("max_missing_frames"); it != j.end() && it->is_number())
            e.maxMissingFrames = it->get<int>();
        if (auto it = j.find("max_distance"); it != j.end() && it->is_number())
            e.maxDistance = it->get<float>();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[WARN] extras parse: %s\n", ex.what());
    }
    return e;
}

cv::Scalar waitColor(float seconds, const Thresholds& t) {
    if (seconds < t.greenSec)  return cv::Scalar(  0, 255,   0);   // green
    if (seconds < t.yellowSec) return cv::Scalar(  0, 255, 255);   // yellow
    return                             cv::Scalar(  0,   0, 255);  // red
}

void drawPersonBox(cv::Mat& frame, const cv::Rect2f& box,
                   const std::string& label, const cv::Scalar& color)
{
    cv::rectangle(frame, box, color, 2);

    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.55;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(label, font, scale, thickness, &baseline);

    const int x1 = static_cast<int>(box.x);
    const int y1 = static_cast<int>(box.y);
    const int labelY = (y1 - ts.height - 6 > 0) ? (y1 - 6) : (y1 + ts.height + 6);
    cv::rectangle(frame,
                  cv::Point(x1, labelY - ts.height - 4),
                  cv::Point(x1 + ts.width + 4, labelY + 4),
                  color, cv::FILLED);
    cv::putText(frame, label, cv::Point(x1 + 2, labelY),
                font, scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
}

class Detector {
public:
    explicit Detector(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold)
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load model");
    }
    std::vector<ppu::Detection> infer(const cv::Mat& bgr) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        auto dets  = ppu::decodeYolov8Ppu(outs, m_inputW, m_inputH, m_conf, m_iou);
        std::vector<cv::Rect2f> boxes;
        for (const auto& d : dets) boxes.push_back(d.box);
        auto px = sdk::unletterboxBoxes(boxes, lb.gain, lb.pad, bgr.size());
        for (size_t i = 0; i < dets.size(); ++i) dets[i].box = px[i];
        return dets;
    }
private:
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    int   m_inputH, m_inputW;
    float m_conf, m_iou;
};

bool insideAnyRegion(const cv::Point2f& p,
                     const std::vector<std::vector<cv::Point2f>>& zones)
{
    for (const auto& z : zones) {
        if (zone::pointInPolygon(p, z)) return true;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const std::string configPath = resolveConfigPath(argc, argv);
    auto p      = loadConfig(configPath, argc, argv);
    auto extras = loadExtras(configPath);
    auto labs   = sdk::loadLabels(p.labelPath, labels::COCO80);

    if (p.regions.empty()) {
        std::fprintf(stderr,
            "[WARN] No valid queue regions in config "
            "(need >=3 points per region) — defaulting to full frame.\n");
        p.regions = {{{0.f,0.f},{1.f,0.f},{1.f,1.f},{0.f,1.f}}};
    }

    Detector        model(p);
    CentroidTracker tracker(extras.maxMissingFrames, extras.maxDistance);
    InputSource     source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    std::unordered_map<int, QueueState> queueState;

    runDemo(source, [&](cv::Mat& frame) {
        const float w = static_cast<float>(frame.cols);
        const float h = static_cast<float>(frame.rows);
        const auto  now = Clock::now();

        std::vector<std::vector<cv::Point2f>> zonesPx;
        zonesPx.reserve(p.regions.size());
        for (const auto& r : p.regions) zonesPx.push_back(zone::toPixels(r, frame.size()));

        auto dets = model.infer(frame);

        std::vector<cv::Rect2f>   personBoxes;
        std::vector<cv::Point2f>  centroidsNorm;
        std::vector<cv::Point2f>  centroidsPx;
        personBoxes.reserve(dets.size());
        centroidsNorm.reserve(dets.size());
        centroidsPx.reserve(dets.size());
        for (const auto& d : dets) {
            if (d.classId != kPersonClassId) continue;
            personBoxes.push_back(d.box);
            const float cx = d.box.x + d.box.width  * 0.5f;
            const float cy = d.box.y + d.box.height * 0.5f;
            centroidsPx.emplace_back(cx, cy);
            centroidsNorm.emplace_back(cx / w, cy / h);
        }

        auto assignments = tracker.update(centroidsNorm);

        int queueCount = 0;
        struct DrawItem { cv::Rect2f box; std::string label; cv::Scalar color; };
        std::vector<DrawItem> drawList;
        drawList.reserve(personBoxes.size());

        for (size_t i = 0; i < personBoxes.size(); ++i) {
            auto it = assignments.find(static_cast<int>(i));
            if (it == assignments.end()) continue;
            const int trackId = it->second;

            QueueState& st = queueState[trackId];
            const bool inside = insideAnyRegion(centroidsPx[i], zonesPx);

            if (inside && !st.inQueue) {
                st.inQueue   = true;
                st.enterTime = now;
            } else if (!inside && st.inQueue) {
                st.inQueue   = false;
                st.enterTime = {};
            }

            DrawItem item;
            item.box = personBoxes[i];
            if (st.inQueue) {
                const float secs = std::chrono::duration<float>(now - st.enterTime).count();
                item.color = waitColor(secs, extras.thresh);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "In Queue %d: %ds",
                              trackId, static_cast<int>(secs));
                item.label = buf;
                ++queueCount;
            } else {
                item.color = cv::Scalar(255, 255, 255);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "Person %d", trackId);
                item.label = buf;
            }
            drawList.push_back(std::move(item));
        }

        // Prune queue entries for tracks the tracker has dropped.
        const auto& active = tracker.tracks();
        for (auto it = queueState.begin(); it != queueState.end(); ) {
            if (!active.count(it->first)) it = queueState.erase(it);
            else                          ++it;
        }

        // Draw order: zones (translucent) -> people -> overlay text.
        const cv::Scalar zoneColor(255, 200, 0);  // BGR: warm-cyan
        for (const auto& zpx : zonesPx) {
            zone::drawFilledPolygon(frame, zpx, zoneColor, /*alpha=*/0.2f);
        }

        for (const auto& d : drawList) {
            drawPersonBox(frame, d.box, d.label, d.color);
        }

        char buf[48];
        std::snprintf(buf, sizeof(buf), "Queue Count: %d", queueCount);
        const auto font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.7;
        const int thickness = 2;
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(buf, font, scale, thickness, &baseline);
        const int x = 10, y = 60;
        cv::rectangle(frame,
                      cv::Point(x - 6, y - ts.height - 6),
                      cv::Point(x + ts.width + 6, y + baseline),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, buf, cv::Point(x, y),
                    font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }, cfg);

    return 0;
}
