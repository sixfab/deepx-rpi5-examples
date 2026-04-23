// ============================================================
// deeplabv3_demo.cpp  --  DeepLabV3+ Semantic Segmentation
// Single-tensor output: [1, numClasses, H, W] class logits.
// Uses DIRECT resize (not letterbox) because a letterbox would
// distort the class-map geometry for semantic segmentation.
// Usage: ./deeplabv3_demo [--model path] [--source video|camera]
//                        [--path file]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "sdk_utils.h"
#include "seg_decode.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/imgproc.hpp>

class DeepLabV3Segmentor {
public:
    explicit DeepLabV3Segmentor(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth)
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load DeepLabV3+ model");
    }

    cv::Mat infer(const cv::Mat& bgr) {
        // Direct stretch-resize + BGR->RGB — the Cityscapes reference does the
        // same, and letterbox padding would distort the per-pixel label map.
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(m_inputW, m_inputH),
                   0, 0, cv::INTER_LINEAR);

        auto reqId = m_engine->RunAsync(resized.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        if (outs.empty()) return {};

        const auto& sh = outs[0]->shape();
        // Expect [1, numClasses, H, W]. Read the class axis from shape[1].
        if (sh.size() < 4) return {};
        const int numClasses = static_cast<int>(sh[1]);
        const int h = static_cast<int>(sh[2]);
        const int w = static_cast<int>(sh[3]);
        const float* logits = static_cast<const float*>(outs[0]->data());
        return seg::decodeDeeplabV3(logits, numClasses, h, w);
    }

private:
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    int m_inputH, m_inputW;
};

int main(int argc, char** argv) {
    auto p       = loadConfig("", argc, argv);
    auto palette = labels::cityscapesPalette();

    DeepLabV3Segmentor model(p);
    InputSource        source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        cv::Mat labelMap = model.infer(frame);
        vis::drawSemanticSeg(frame, labelMap, palette);
    }, cfg);

    return 0;
}
