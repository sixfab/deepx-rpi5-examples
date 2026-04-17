// ============================================================
// people_tracking_demo.cpp  --  YOLOv5/v8 PPU detection +
// centroid tracker. Each person receives a persistent integer
// ID drawn above the detection box. Mirrors PeopleTrackingAdapter.cpp.
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "tracker.h"
#include "visualizer.h"

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
        m_anchors = ppu::defaultYolov5Anchors();
    }

    std::vector<ppu::Detection> infer(const cv::Mat& bgr) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        auto dets  = ppu::decodeYolov5Ppu(outs, m_inputW, m_inputH,
                                          m_conf, m_iou, m_anchors);
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
    std::vector<ppu::AnchorLayer> m_anchors;
};

} // namespace

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    Detector       model(p);
    CentroidTracker tracker(/*maxMissed=*/10, /*maxDistance=*/0.1f);
    InputSource    source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        auto dets = model.infer(frame);
        const float w = static_cast<float>(frame.cols);
        const float h = static_cast<float>(frame.rows);

        std::vector<cv::Rect2f> personBoxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        std::vector<cv::Point2f> centroids;
        for (const auto& d : dets) {
            if (d.classId != kPersonClassId) continue;
            personBoxes.push_back(d.box);
            scores.push_back(d.score);
            classIds.push_back(d.classId);
            centroids.emplace_back((d.box.x + d.box.width  * 0.5f) / w,
                                   (d.box.y + d.box.height * 0.5f) / h);
        }

        auto assignments = tracker.update(centroids);
        vis::drawDetections(frame, personBoxes, scores, classIds, labs);

        for (const auto& kv : assignments) {
            const int i = kv.first, id = kv.second;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "ID %d", id);
            const cv::Point pt(
                static_cast<int>(personBoxes[i].x),
                static_cast<int>(personBoxes[i].y + personBoxes[i].height + 18));
            cv::putText(frame, buf, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(frame, buf, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }, cfg);

    return 0;
}
