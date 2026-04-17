#include "reid_utils.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace reid {

void l2Normalize(std::vector<float>& v) {
    float sq = 0.0f;
    for (float x : v) sq += x * x;
    const float n = std::sqrt(sq);
    if (n > 1e-6f) for (auto& x : v) x /= n;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

std::vector<float> extractEmbedding(
    dxrt::InferenceEngine& engine,
    const cv::Mat& bgrFrame,
    const cv::Rect2f& box,
    int inputH, int inputW,
    int minCropSize)
{
    const int x  = std::max(0, static_cast<int>(box.x));
    const int y  = std::max(0, static_cast<int>(box.y));
    const int w  = std::min(static_cast<int>(box.width),  bgrFrame.cols - x);
    const int h  = std::min(static_cast<int>(box.height), bgrFrame.rows - y);
    if (w < minCropSize || h < minCropSize) return {};

    cv::Mat resized;
    cv::resize(bgrFrame(cv::Rect(x, y, w, h)), resized, cv::Size(inputW, inputH));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    auto reqId = engine.RunAsync(resized.data, nullptr, nullptr);
    auto outs  = engine.Wait(reqId);
    if (outs.empty()) return {};

    const auto& t = outs[0];
    int n = 1; for (auto s : t->shape()) n *= s;
    if (n <= 0) return {};
    const float* data = static_cast<const float*>(t->data());

    std::vector<float> emb(data, data + n);
    l2Normalize(emb);
    return emb;
}

} // namespace reid
