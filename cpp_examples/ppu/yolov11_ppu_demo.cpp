// ============================================================
// yolov11_ppu_demo.cpp  --  Educational PPU demo for YOLOv11
//
// Same inference path as object_detection/yolov11_demo, but with
// extra logging of the first-frame tensor(s) so you can see what
// the Post-Processing Unit actually returns. YOLOv11 shares the
// anchor-free PPU layout with YOLOv8:
//
//   tensor[0].type == dxrt::DataType::BBOX
//   tensor[0].shape == {1, N, 1}  where each elem holds:
//     cx (f32), cy (f32), w (f32), h (f32), score (f32), label (u32)
//
// The PPU performs NMS + anchor-free decode on-chip, removing
// thousands of CPU ops per frame relative to the raw-float path.
// Usage: ./yolov11_ppu_demo [--model path] [--source video|camera]
//                          [--path file] [--conf 0.3] [--iou 0.45]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "ppu_decode.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <cstdio>

class YOLOv11PpuDetector {
public:
    explicit YOLOv11PpuDetector(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold)
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load YOLOv11 model");
    }

    std::vector<ppu::Detection> infer(const cv::Mat& bgr) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_engine->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);

        if (m_firstFrame) {
            std::fprintf(stderr, "[PPU] first frame: %zu output tensor(s)\n", outs.size());
            for (size_t i = 0; i < outs.size(); ++i) {
                const auto& t = outs[i];
                std::fprintf(stderr, "[PPU]   tensor[%zu] shape=[", i);
                const auto& sh = t->shape();
                for (size_t k = 0; k < sh.size(); ++k)
                    std::fprintf(stderr, "%s%ld", k ? "," : "", (long)sh[k]);
                std::fprintf(stderr, "] type=%d elem_size=%u\n",
                             (int)t->type(), (unsigned)t->elem_size());
            }
            m_firstFrame = false;
        }

        auto dets = ppu::decodeYolov8Ppu(outs, m_inputW, m_inputH, m_conf, m_iou);

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
    bool  m_firstFrame = true;
};

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::COCO80);

    YOLOv11PpuDetector model(p);
    InputSource        source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
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
        vis::drawDetections(frame, boxes, scores, classIds, labs);
    }, cfg);

    return 0;
}
