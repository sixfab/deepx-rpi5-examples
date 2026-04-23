// ============================================================
// face_emotion_demo.cpp  --  Two-stage face emotion pipeline
// Stage 1: YOLOv5-Face detector (float tensor, 5 landmarks)
// Stage 2: Per-face crop -> emotion classifier (224x224 RGB,
//          softmax + argmax over N emotion classes).
// Usage: ./face_emotion_demo [--model detPath] [--source video|camera]
//                           [--path file] [--conf 0.5] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

static const std::vector<std::string> EMOTION_LABELS = {
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
};

class FaceEmotionDetector {
public:
    explicit FaceEmotionDetector(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold),
          m_clsH(224), m_clsW(224)
    {
        m_det = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_det) throw std::runtime_error("Failed to load face detection model");
        if (p.modelPath2.empty())
            throw std::runtime_error("face_emotion_demo requires model_path_2");
        m_cls = sdk::loadEngine(p.modelPath2, m_clsH, m_clsW);
        if (!m_cls) throw std::runtime_error("Failed to load emotion classifier");
    }

    struct Result {
        cv::Rect2f box;
        std::vector<cv::Point2f> landmarks;
        std::string label;
    };

    std::vector<Result> infer(const cv::Mat& bgr) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_det->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_det->Wait(reqId);
        auto dets  = ppu::decodeYolov5Face(outs, m_conf, m_iou);

        std::vector<cv::Rect2f> boxes;
        for (const auto& d : dets) boxes.push_back(d.box);
        auto boxesPx = sdk::unletterboxBoxes(boxes, lb.gain, lb.pad, bgr.size());

        std::vector<Result> out;
        for (size_t i = 0; i < dets.size(); ++i) {
            Result r;
            r.box       = boxesPx[i];
            r.landmarks = sdk::unletterboxPoints(dets[i].landmarks, lb.gain, lb.pad, bgr.size());
            r.label     = classify(bgr, r.box);
            out.push_back(std::move(r));
        }
        return out;
    }

private:
    std::string classify(const cv::Mat& bgr, const cv::Rect2f& box) {
        cv::Rect roi(std::max(0, (int)box.x), std::max(0, (int)box.y),
                     std::min((int)box.width,  bgr.cols - (int)box.x),
                     std::min((int)box.height, bgr.rows - (int)box.y));
        if (roi.width < 10 || roi.height < 10) return "";

        cv::Mat resized;
        cv::resize(bgr(roi), resized, cv::Size(m_clsW, m_clsH));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        auto reqId = m_cls->RunAsync(resized.data, nullptr, nullptr);
        auto outs  = m_cls->Wait(reqId);
        if (outs.empty()) return "";

        const auto& t = outs[0];
        int n = 1; for (auto s : t->shape()) n *= s;
        const float* data = static_cast<const float*>(t->data());

        float mx = data[0];
        for (int i = 1; i < n; ++i) if (data[i] > mx) mx = data[i];
        std::vector<float> pr(n); float s = 0;
        for (int i = 0; i < n; ++i) { pr[i] = std::exp(data[i] - mx); s += pr[i]; }
        int best = 0;
        for (int i = 1; i < n; ++i) if (pr[i] > pr[best]) best = i;
        if (best >= (int)EMOTION_LABELS.size()) return "";
        return EMOTION_LABELS[best] + " " + std::to_string((int)(pr[best] / s * 100)) + "%";
    }

    std::unique_ptr<dxrt::InferenceEngine> m_det, m_cls;
    int   m_inputH, m_inputW;
    float m_conf, m_iou;
    int   m_clsH, m_clsW;
};

int main(int argc, char** argv) {
    auto p = loadConfig("", argc, argv);

    FaceEmotionDetector model(p);
    InputSource         source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        for (const auto& r : model.infer(frame)) {
            cv::rectangle(frame, r.box, cv::Scalar(0, 255, 0), 2);
            if (!r.label.empty())
                cv::putText(frame, r.label,
                            cv::Point((int)r.box.x, std::max(15, (int)r.box.y - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            vis::drawFaceLandmarks(frame, r.box, r.landmarks);
        }
    }, cfg);

    return 0;
}
