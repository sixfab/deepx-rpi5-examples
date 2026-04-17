// ============================================================
// body_pose_demo.cpp  --  YOLO-Pose Body Pose Estimation
// 17 COCO keypoints per person, PPU post-processing.
// Usage: ./body_pose_demo [--model path] [--source video|camera] [--path file]
//                        [--conf 0.3] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>

class BodyPoseDetector {
public:
    explicit BodyPoseDetector(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold),
          m_anchors(ppu::defaultPoseAnchors())
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load body pose model");
    }

    std::vector<ppu::PoseDetection> infer(const cv::Mat& bgr) {
        auto lb     = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId  = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs   = m_engine->Wait(reqId);

        auto dets = ppu::decodePosePpu(outs, m_inputW, m_inputH,
                                       m_conf, m_iou, m_anchors);

        // Unletterbox boxes + keypoints with the same gain/pad.
        std::vector<cv::Rect2f> boxes;
        boxes.reserve(dets.size());
        for (const auto& d : dets) boxes.push_back(d.box);
        auto boxesPx = sdk::unletterboxBoxes(boxes, lb.gain, lb.pad, bgr.size());

        for (size_t i = 0; i < dets.size(); ++i) {
            dets[i].box = boxesPx[i];
            dets[i].keypoints = sdk::unletterboxPoints(
                dets[i].keypoints, lb.gain, lb.pad, bgr.size());
        }
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
    const std::vector<std::string> labs = {"person"};

    BodyPoseDetector model(p);
    InputSource      source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
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
        for (const auto& d : dets) {
            boxes.push_back(d.box);
            scores.push_back(d.score);
            classIds.push_back(d.classId);
        }
        vis::drawDetections(frame, boxes, scores, classIds, labs, 0.0f);
        for (const auto& d : dets) {
            vis::drawBodyPose(frame, d.keypoints, d.keypointScores);
        }
    }, cfg);

    return 0;
}
