// ============================================================
// trespassing_demo.cpp  --  YOLOv8 PPU + polygon intrusion test.
// Raises an alert banner and tints the zone red whenever any
// person's foot (box bottom-center) sits inside the configured
// polygon. Mirrors TrespassingAdapter.cpp.
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"
#include "zone_utils.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/imgproc.hpp>
#include <cstdio>

namespace {

constexpr int kPersonClassId = 0;

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

void drawAlertBanner(cv::Mat& frame) {
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 40),
                  cv::Scalar(0, 0, 180), cv::FILLED);
    cv::putText(frame, "!! TRESPASSING DETECTED !!", cv::Point(10, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

} // namespace

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    if (p.regions.empty()) {
        std::fprintf(stderr, "[WARN] No regions in config — full-frame fallback.\n");
        p.regions = {{{0.f,0.f},{1.f,0.f},{1.f,1.f},{0.f,1.f}}};
    }

    Detector    model(p);
    InputSource source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        const auto polygonPx = zone::toPixels(p.regions[0], frame.size());

        auto dets = model.infer(frame);

        bool alert = false;
        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        for (const auto& d : dets) {
            boxes.push_back(d.box);
            scores.push_back(d.score);
            classIds.push_back(d.classId);

            if (d.classId != kPersonClassId) continue;
            const cv::Point2f foot(d.box.x + d.box.width * 0.5f,
                                   d.box.y + d.box.height);
            if (zone::pointInPolygon(foot, polygonPx)) { alert = true; continue; }
            // Person larger than the zone: foot may fall outside the polygon
            // while the polygon sits inside the bbox. Treat that as intrusion.
            for (const auto& v : polygonPx) {
                if (d.box.contains(v)) { alert = true; break; }
            }
        }

        zone::drawZoneOverlay(frame, polygonPx, alert);
        vis::drawDetections(frame, boxes, scores, classIds, labs);
        if (alert) drawAlertBanner(frame);
    }, cfg);

    return 0;
}
