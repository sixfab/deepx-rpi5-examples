// ============================================================
// yolov8seg_demo.cpp  --  YOLOv8 Instance Segmentation
// Two-tensor model:
//   outputs[0]: detection head, float [1, 4 + 80 + 32, N]
//   outputs[1]: proto basis,    float [1, 32, mh, mw]
// Decode = transpose detection rows, per-class NMS, then
// matmul mask coefs × proto basis + sigmoid to generate masks.
// Usage: ./yolov8seg_demo [--model path] [--source video|camera]
//                        [--path file] [--conf 0.3] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "sdk_utils.h"
#include "seg_decode.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <cstdio>

namespace {
constexpr int kNumClasses = 80;
constexpr int kMaskDim    = 32;
} // namespace

class YOLOv8Segmentor {
public:
    struct SegResult {
        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        std::vector<cv::Mat>    masks;
    };

    explicit YOLOv8Segmentor(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold)
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load YOLOv8-seg model");
    }

    SegResult infer(const cv::Mat& bgr) {
        SegResult r;
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        if (outs.size() < 2) return r;

        const auto& detSh = outs[0]->shape();
        const auto& prSh  = outs[1]->shape();
        if (detSh.size() < 3 || prSh.size() < 4) return r;

        // Detection tensor: pick the axis whose size == 4 + 80 + 32 as the
        // channel axis. The other is the anchor count.
        const int dA = static_cast<int>(detSh[detSh.size() - 2]);
        const int dB = static_cast<int>(detSh[detSh.size() - 1]);
        int channels, numAnchors;
        if (dA == 4 + kNumClasses + kMaskDim) { channels = dA; numAnchors = dB; }
        else if (dB == 4 + kNumClasses + kMaskDim) { channels = dB; numAnchors = dA; }
        else return r;

        const float* detPtr = static_cast<const float*>(outs[0]->data());
        // decodeYolov8Seg assumes channel-first (C, N). If the tensor is
        // (N, C) we transpose on the fly into a temporary buffer.
        std::vector<float> transposed;
        const float* usePtr = detPtr;
        if (dA != channels) {
            transposed.resize(static_cast<size_t>(channels) * numAnchors);
            for (int n = 0; n < numAnchors; ++n) {
                for (int c = 0; c < channels; ++c) {
                    transposed[c * numAnchors + n] = detPtr[n * channels + c];
                }
            }
            usePtr = transposed.data();
        }

        auto dets = seg::decodeYolov8Seg(usePtr, kNumClasses, kMaskDim,
                                         numAnchors, m_inputW, m_inputH,
                                         m_conf, m_iou);
        if (dets.empty()) return r;

        // Proto tensor shape: [1, maskDim, mh, mw].
        const int mh = static_cast<int>(prSh[prSh.size() - 2]);
        const int mw = static_cast<int>(prSh[prSh.size() - 1]);
        const float* protoPtr = static_cast<const float*>(outs[1]->data());

        r.masks = seg::generateMasks(protoPtr, kMaskDim, mh, mw,
                                     dets, m_inputW, m_inputH,
                                     lb.gain, lb.pad, bgr.size());

        std::vector<cv::Rect2f> boxesModel;
        boxesModel.reserve(dets.size());
        for (const auto& d : dets) boxesModel.push_back(d.box);
        r.boxes = sdk::unletterboxBoxes(boxesModel, lb.gain, lb.pad, bgr.size());
        r.scores.reserve(dets.size());
        r.classIds.reserve(dets.size());
        for (const auto& d : dets) {
            r.scores.push_back(d.score);
            r.classIds.push_back(d.classId);
        }
        return r;
    }

private:
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    int   m_inputH, m_inputW;
    float m_conf, m_iou;
};

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    YOLOv8Segmentor model(p);
    InputSource     source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        auto r = model.infer(frame);
        // Masks first so the boxes/labels render on top of the overlay.
        vis::drawSegMasks(frame, r.masks, r.classIds);
        vis::drawDetections(frame, r.boxes, r.scores, r.classIds, labs);
    }, cfg);

    return 0;
}
