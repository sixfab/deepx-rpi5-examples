// ============================================================
// efficientnet_demo.cpp  --  EfficientNet-B0 ImageNet Classifier
// Output: [1, 1000] probabilities. Direct resize (no letterbox).
// ImageNet labels are loaded from a file if present, otherwise we
// fall back to "class_N" strings so the demo still produces output.
// Usage: ./efficientnet_demo [--model path] [--source video|camera]
//                           [--path file]
// ============================================================

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "label_sets.h"
#include "sdk_utils.h"
#include "visualizer.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace {
constexpr int kTopK = 5;
} // namespace

class EfficientNetClassifier {
public:
    EfficientNetClassifier(const DemoParams& p, std::vector<std::string> labels)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_labels(std::move(labels))
    {
        m_engine = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_engine) throw std::runtime_error("Failed to load EfficientNet model");
    }

    std::vector<std::pair<std::string, float>> infer(const cv::Mat& bgr) {
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(m_inputW, m_inputH),
                   0, 0, cv::INTER_LINEAR);

        auto reqId = m_engine->RunAsync(resized.data, nullptr, nullptr);
        auto outs  = m_engine->Wait(reqId);
        if (outs.empty()) return {};

        const auto& sh = outs[0]->shape();
        int numClasses = 1;
        for (size_t i = 0; i < sh.size(); ++i) numClasses *= static_cast<int>(sh[i]);
        const float* probs = static_cast<const float*>(outs[0]->data());

        std::vector<int> idx(numClasses);
        for (int i = 0; i < numClasses; ++i) idx[i] = i;
        const int k = std::min(kTopK, numClasses);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        std::vector<std::pair<std::string, float>> out;
        out.reserve(k);
        for (int i = 0; i < k; ++i) {
            const int cid = idx[i];
            std::string name = (cid >= 0 && cid < static_cast<int>(m_labels.size()))
                                   ? m_labels[cid]
                                   : ("class_" + std::to_string(cid));
            out.emplace_back(std::move(name), probs[cid]);
        }
        return out;
    }

private:
    std::unique_ptr<dxrt::InferenceEngine> m_engine;
    int m_inputH, m_inputW;
    std::vector<std::string> m_labels;
};

int main(int argc, char** argv) {
    auto p    = loadConfig("", argc, argv);
    auto labs = sdk::loadLabels(p.labelPath, labels::imagenet1000());

    EfficientNetClassifier model(p, labs);
    InputSource            source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }

    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        vis::drawClassification(frame, model.infer(frame));
    }, cfg);

    return 0;
}
