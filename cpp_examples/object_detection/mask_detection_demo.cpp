// ============================================================
// mask_detection_demo.cpp  --  Mask / No-Mask Detection
// Uses a YOLOv5 PPU model — anchor-based decode on the PPU,
// plain BBOX stream on the host.
// Usage: ./mask_detection_demo [--model path] [--source video|camera]
//                              [--path file] [--conf 0.4] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>

class MaskDetector {
public:
    explicit MaskDetector(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold),
          m_anchors(ppu::defaultYolov5Anchors())
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load mask-detection model");
    }

    std::vector<ppu::Detection> infer(const cv::Mat& bgr) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);

        auto dets = ppu::decodeYolov5Ppu(outs, m_inputW, m_inputH,
                                         m_conf, m_iou, m_anchors);

        std::vector<cv::Rect2f> boxes;
        boxes.reserve(dets.size());
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

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::MASK_CLASSES);

    MaskDetector model(p);
    InputSource  source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        auto dets = model.infer(frame);
        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        boxes.reserve(dets.size());
        scores.reserve(dets.size());
        classIds.reserve(dets.size());
        for (const auto& d : dets) {
            boxes.push_back(d.box);
            scores.push_back(d.score);
            classIds.push_back(d.classId);
        }
        vis::drawDetections(frame, boxes, scores, classIds, labs);
    }, cfg);

    return 0;
}
