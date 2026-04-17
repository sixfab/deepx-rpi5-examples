// ============================================================
// yolo26seg_demo.cpp  --  YOLOv26 Instance Segmentation
// YOLOv26-seg has NMS baked into the detection head, so each row
// of outputs[0] is already [x1, y1, x2, y2, score, class_id,
// mask_coef_0..mask_coef_M-1] — no transpose, no NMS on the host.
// The proto basis in outputs[1] is identical to YOLOv8-seg, so
// mask generation is shared via seg::generateMasks().
// Usage: ./yolo26seg_demo [--model path] [--source video|camera]
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
constexpr int kMaskDim = 32;
constexpr int kDetCols = 6;  // x1, y1, x2, y2, score, class_id
} // namespace

class YOLO26Segmentor {
public:
    struct SegResult {
        std::vector<cv::Rect2f> boxes;
        std::vector<float>      scores;
        std::vector<int>        classIds;
        std::vector<cv::Mat>    masks;
    };

    explicit YOLO26Segmentor(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold)
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load YOLOv26-seg model");
    }

    SegResult infer(const cv::Mat& bgr) {
        SegResult r;
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        if (outs.size() < 2) return r;

        const auto& detSh = outs[0]->shape();
        const auto& prSh  = outs[1]->shape();
        if (detSh.size() < 2 || prSh.size() < 4) return r;

        // Detection tensor is row-major [1, N, 6+maskDim] — no transpose needed.
        const int rowCols = kDetCols + kMaskDim;
        const int dA = static_cast<int>(detSh[detSh.size() - 2]);
        const int dB = static_cast<int>(detSh[detSh.size() - 1]);
        int numRows;
        if (dB == rowCols) numRows = dA;
        else if (dA == rowCols) numRows = dB;    // unexpected but handle it
        else return r;

        const float* detPtr = static_cast<const float*>(outs[0]->data());
        auto dets = seg::decodeYolo26Seg(detPtr, kMaskDim, numRows,
                                         m_inputW, m_inputH, m_conf);
        if (dets.empty()) return r;

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
    float m_conf;
};

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    YOLO26Segmentor model(p);
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
        vis::drawSegMasks(frame, r.masks, r.classIds);
        vis::drawDetections(frame, r.boxes, r.scores, r.classIds, labs);
    }, cfg);

    return 0;
}
