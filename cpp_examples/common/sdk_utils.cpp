#include "sdk_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <opencv2/imgproc.hpp>

namespace sdk {

void initDevice() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        try {
            dxrt::DevicePool::GetInstance().InitCores();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ERROR] DevicePool::InitCores failed: %s\n", e.what());
        }
    });
}

std::unique_ptr<dxrt::InferenceEngine> loadEngine(
    const std::string& modelPath, int& inputH, int& inputW)
{
    initDevice();

    if (!std::filesystem::exists(modelPath)) {
        std::fprintf(stderr, "[ERROR] Model file not found: %s\n", modelPath.c_str());
        return nullptr;
    }

    std::unique_ptr<dxrt::InferenceEngine> engine;
    try {
        dxrt::InferenceOption option;
        engine = std::make_unique<dxrt::InferenceEngine>(modelPath, option);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] Failed to load model %s: %s\n",
                     modelPath.c_str(), e.what());
        return nullptr;
    }

    try {
        auto inputs = engine->GetInputs();
        if (!inputs.empty()) {
            const auto& shape = inputs[0].shape();
            // DX-COM compiled models are NHWC => [1, H, W, C].
            if (shape.size() == 4) {
                inputH = static_cast<int>(shape[1]);
                inputW = static_cast<int>(shape[2]);
            }
        }
    } catch (...) {
        // Fall back to whatever the caller pre-populated (e.g. from config).
    }

    std::printf("[INFO] Loaded model: %s\n", modelPath.c_str());
    std::printf("[INFO] Input size  : %dx%d\n", inputW, inputH);
    return engine;
}

LetterboxResult letterbox(const cv::Mat& bgrFrame, int targetH, int targetW) {
    LetterboxResult out;
    const int srcW = bgrFrame.cols;
    const int srcH = bgrFrame.rows;

    out.gain = std::min(static_cast<float>(targetW) / srcW,
                        static_cast<float>(targetH) / srcH);

    const int newW = static_cast<int>(std::round(srcW * out.gain));
    const int newH = static_cast<int>(std::round(srcH * out.gain));
    const int dw = (targetW - newW) / 2;
    const int dh = (targetH - newH) / 2;
    out.pad = cv::Point2f(static_cast<float>(dw), static_cast<float>(dh));

    cv::Mat resized;
    if (srcW != newW || srcH != newH) {
        cv::resize(bgrFrame, resized, cv::Size(newW, newH),
                   0, 0, cv::INTER_LINEAR);
    } else {
        resized = bgrFrame;
    }

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       dh, targetH - newH - dh,
                       dw, targetW - newW - dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::cvtColor(padded, out.image, cv::COLOR_BGR2RGB);
    return out;
}

std::vector<cv::Rect2f> unletterboxBoxes(
    const std::vector<cv::Rect2f>& boxes,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize)
{
    std::vector<cv::Rect2f> out;
    out.reserve(boxes.size());

    const float srcW = static_cast<float>(originalSize.width);
    const float srcH = static_cast<float>(originalSize.height);

    for (const auto& b : boxes) {
        float x1 = (b.x - pad.x) / gain;
        float y1 = (b.y - pad.y) / gain;
        float x2 = ((b.x + b.width)  - pad.x) / gain;
        float y2 = ((b.y + b.height) - pad.y) / gain;

        x1 = std::clamp(x1, 0.0f, srcW - 1.0f);
        y1 = std::clamp(y1, 0.0f, srcH - 1.0f);
        x2 = std::clamp(x2, 0.0f, srcW - 1.0f);
        y2 = std::clamp(y2, 0.0f, srcH - 1.0f);

        out.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }
    return out;
}

std::vector<cv::Point2f> unletterboxPoints(
    const std::vector<cv::Point2f>& points,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize)
{
    std::vector<cv::Point2f> out;
    out.reserve(points.size());
    const float srcW = static_cast<float>(originalSize.width);
    const float srcH = static_cast<float>(originalSize.height);
    for (const auto& p : points) {
        float x = std::clamp((p.x - pad.x) / gain, 0.0f, srcW - 1.0f);
        float y = std::clamp((p.y - pad.y) / gain, 0.0f, srcH - 1.0f);
        out.emplace_back(x, y);
    }
    return out;
}

std::vector<std::string> loadLabels(
    const std::string& path,
    const std::vector<std::string>& fallback)
{
    if (path.empty() || !std::filesystem::exists(path)) {
        return fallback;
    }
    std::ifstream in(path);
    if (!in.is_open()) return fallback;

    std::vector<std::string> result;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                 line.back() == ' '  || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) result.push_back(line);
    }
    return result.empty() ? fallback : result;
}

cv::Scalar colorForClass(int classId) {
    // Same deterministic hash the existing demo uses, in BGR.
    int r = (classId * 50) % 255;
    int g = (classId * 80 + 50) % 255;
    int b = (classId * 120 + 100) % 255;
    return cv::Scalar(b, g, r);
}

} // namespace sdk
