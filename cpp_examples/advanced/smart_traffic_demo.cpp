// smart_traffic_demo.cpp  --  YOLOv8 PPU + centroid tracker + line
// crossing counter. Mirrors SmartTrafficAdapter.cpp.

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
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

namespace {

// COCO: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.
const std::unordered_set<int> kVehicleClassIds = {1, 2, 3, 5, 7};

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

} // namespace

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    if (p.lines.empty()) {
        std::fprintf(stderr, "[WARN] No counting line — using horizontal mid-line.\n");
        p.lines.emplace_back(cv::Point2f(0.1f, 0.5f), cv::Point2f(0.9f, 0.5f));
    }

    Detector        model(p);
    CentroidTracker tracker(/*maxMissed=*/15, /*maxDistance=*/0.1f);
    InputSource     source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    // Per-track previous centroid + "has crossed" flag — both keyed by track id.
    std::unordered_map<int, cv::Point2f> prevCentroids;
    std::unordered_set<int> crossed;
    int totalCount = 0;

    runDemo(source, [&](cv::Mat& frame) {
        const float w = static_cast<float>(frame.cols);
        const float h = static_cast<float>(frame.rows);
        const cv::Point2f lineAPx(p.lines[0].first.x * w,  p.lines[0].first.y  * h);
        const cv::Point2f lineBPx(p.lines[0].second.x * w, p.lines[0].second.y * h);

        auto dets = model.infer(frame);

        std::vector<cv::Rect2f>  boxes;
        std::vector<float>       scores;
        std::vector<int>         classIds;
        std::vector<cv::Point2f> centroids;
        for (const auto& d : dets) {
            if (!kVehicleClassIds.count(d.classId)) continue;
            boxes.push_back(d.box);
            scores.push_back(d.score);
            classIds.push_back(d.classId);
            centroids.emplace_back((d.box.x + d.box.width  * 0.5f) / w,
                                   (d.box.y + d.box.height * 0.5f) / h);
        }

        auto assignments = tracker.update(centroids);

        // Check crossings BEFORE overwriting prev centroids.
        for (const auto& kv : assignments) {
            const int idx = kv.first, id = kv.second;
            const auto& curr = centroids[idx];
            auto it = prevCentroids.find(id);
            if (it != prevCentroids.end() && !crossed.count(id)) {
                if (zone::segmentCrossesLine(it->second, curr,
                                             p.lines[0].first, p.lines[0].second)) {
                    ++totalCount;
                    crossed.insert(id);
                }
            }
            prevCentroids[id] = curr;
        }

        // Drop state for ids the tracker has pruned.
        const auto& active = tracker.tracks();
        for (auto it = prevCentroids.begin(); it != prevCentroids.end(); ) {
            if (!active.count(it->first)) it = prevCentroids.erase(it);
            else                          ++it;
        }

        zone::drawCountingLine(frame, lineAPx, lineBPx, totalCount, 0);
        vis::drawDetections(frame, boxes, scores, classIds, labs);
    }, cfg);

    return 0;
}
