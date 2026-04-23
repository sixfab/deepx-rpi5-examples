#include "seg_decode.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace seg {

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Per-class NMS over SegDetection. Mirrors the template in ppu_decode.cpp so
// we keep mask coefficients paired with their boxes.
std::vector<SegDetection> nmsByClass(std::vector<SegDetection>& dets, float iouThresh) {
    if (dets.empty()) return {};
    std::sort(dets.begin(), dets.end(),
              [](const SegDetection& a, const SegDetection& b) {
                  return a.score > b.score;
              });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<SegDetection> kept;
    kept.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(std::move(dets[i]));
        const auto& bi = kept.back().box;
        const float ai = bi.width * bi.height;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j] || kept.back().classId != dets[j].classId) continue;
            const auto& bj = dets[j].box;
            const float x1 = std::max(bi.x, bj.x);
            const float y1 = std::max(bi.y, bj.y);
            const float x2 = std::min(bi.x + bi.width,  bj.x + bj.width);
            const float y2 = std::min(bi.y + bi.height, bj.y + bj.height);
            const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            const float uni = ai + bj.width * bj.height - inter + 1e-6f;
            if (inter / uni > iouThresh) suppressed[j] = true;
        }
    }
    return kept;
}

} // namespace

std::vector<SegDetection> decodeYolov8Seg(
    const float* detPtr,
    int numClasses, int maskDim, int numAnchors,
    int inputW, int inputH,
    float confThresh, float iouThresh)
{
    (void)inputW; (void)inputH;
    std::vector<SegDetection> out;
    if (!detPtr || numAnchors <= 0) return out;

    const int channels = 4 + numClasses + maskDim;

    // The tensor is stored channel-first: val(c, a) = detPtr[c * numAnchors + a].
    auto at = [&](int c, int a) -> float {
        return detPtr[c * numAnchors + a];
    };

    out.reserve(64);
    for (int a = 0; a < numAnchors; ++a) {
        float best = 0.0f;
        int bestId = 0;
        for (int c = 0; c < numClasses; ++c) {
            const float s = at(4 + c, a);
            if (s > best) { best = s; bestId = c; }
        }
        if (best < confThresh) continue;

        const float cx = at(0, a), cy = at(1, a);
        const float bw = at(2, a), bh = at(3, a);

        SegDetection d;
        d.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
        d.score = best;
        d.classId = bestId;
        d.maskCoeffs.resize(maskDim);
        for (int m = 0; m < maskDim; ++m) {
            d.maskCoeffs[m] = at(4 + numClasses + m, a);
        }
        out.push_back(std::move(d));
    }
    (void)channels;
    return nmsByClass(out, iouThresh);
}

std::vector<SegDetection> decodeYolo26Seg(
    const float* detPtr,
    int maskDim, int numRows,
    int inputW, int inputH,
    float confThresh)
{
    (void)inputW; (void)inputH;
    std::vector<SegDetection> out;
    if (!detPtr || numRows <= 0) return out;

    // YOLOv26-seg row: [x1, y1, x2, y2, score, class_id, coef0..coefM-1].
    const int rowCols = 6 + maskDim;
    out.reserve(64);
    for (int r = 0; r < numRows; ++r) {
        const float* row = detPtr + r * rowCols;
        const float score = row[4];
        if (score < confThresh) continue;

        const float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        const int classId = static_cast<int>(row[5]);

        SegDetection d;
        d.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        d.score = score;
        d.classId = classId;
        d.maskCoeffs.assign(row + 6, row + 6 + maskDim);
        out.push_back(std::move(d));
    }
    // NMS already baked in by the YOLOv26 head — no extra NMS pass.
    return out;
}

std::vector<cv::Mat> generateMasks(
    const float* protoPtr,
    int maskDim, int maskH, int maskW,
    const std::vector<SegDetection>& dets,
    int inputW, int inputH,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize)
{
    std::vector<cv::Mat> out;
    if (dets.empty() || !protoPtr) return out;

    const int N = static_cast<int>(dets.size());
    const int M = maskH * maskW;

    // Flatten protos to a (maskDim, M) cv::Mat so we can matmul.
    // (Non-owning view — protoPtr's storage outlives the call.)
    cv::Mat protoMat(maskDim, M,  CV_32F,
                     const_cast<float*>(protoPtr));
    cv::Mat coefMat(N, maskDim,  CV_32F);
    for (int i = 0; i < N; ++i) {
        std::copy(dets[i].maskCoeffs.begin(),
                  dets[i].maskCoeffs.end(),
                  coefMat.ptr<float>(i));
    }
    cv::Mat masksFlat = coefMat * protoMat;            // (N, M) logits

    // Convert to sigmoid probabilities in-place.
    for (int i = 0; i < N; ++i) {
        float* p = masksFlat.ptr<float>(i);
        for (int j = 0; j < M; ++j) p[j] = sigmoid(p[j]);
    }

    const int srcW = originalSize.width;
    const int srcH = originalSize.height;
    const int unpadH = static_cast<int>(std::round(srcH * gain));
    const int unpadW = static_cast<int>(std::round(srcW * gain));
    const int padTop  = static_cast<int>(std::round(pad.y));
    const int padLeft = static_cast<int>(std::round(pad.x));

    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        // Reshape one row to (maskH, maskW) and upsample to model input size.
        cv::Mat mask(maskH, maskW, CV_32F, masksFlat.ptr<float>(i));
        cv::Mat scaled;
        cv::resize(mask, scaled, cv::Size(inputW, inputH),
                   0, 0, cv::INTER_LINEAR);

        // Zero everything outside the detection's box so masks don't bleed.
        const cv::Rect2f& b = dets[i].box;
        int bx1 = std::max(0, static_cast<int>(b.x));
        int by1 = std::max(0, static_cast<int>(b.y));
        int bx2 = std::min(inputW, static_cast<int>(b.x + b.width));
        int by2 = std::min(inputH, static_cast<int>(b.y + b.height));
        if (bx2 < bx1) bx2 = bx1;
        if (by2 < by1) by2 = by1;

        if (by1 > 0) scaled.rowRange(0, by1).setTo(0);
        if (by2 < inputH) scaled.rowRange(by2, inputH).setTo(0);
        if (bx1 > 0) scaled.colRange(0, bx1).setTo(0);
        if (bx2 < inputW) scaled.colRange(bx2, inputW).setTo(0);

        // Strip letterbox padding, then resize back to the source frame.
        int cropW = std::max(1, std::min(unpadW, inputW - padLeft));
        int cropH = std::max(1, std::min(unpadH, inputH - padTop));
        cv::Mat cropped = scaled(cv::Rect(padLeft, padTop, cropW, cropH));
        cv::Mat full;
        cv::resize(cropped, full, cv::Size(srcW, srcH),
                   0, 0, cv::INTER_LINEAR);

        // Binarise at 0.5 — same threshold the Python draw_masks uses.
        cv::Mat binary;
        cv::threshold(full, binary, 0.5f, 255.0f, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8U);
        out.push_back(std::move(binary));
    }
    return out;
}

cv::Mat decodeDeeplabV3(
    const float* logitsPtr,
    int numClasses, int h, int w)
{
    cv::Mat labels(h, w, CV_8U, cv::Scalar(0));
    if (!logitsPtr || numClasses <= 0) return labels;

    const int plane = h * w;
    for (int i = 0; i < plane; ++i) {
        float best = logitsPtr[i];
        int bestC = 0;
        for (int c = 1; c < numClasses; ++c) {
            const float v = logitsPtr[c * plane + i];
            if (v > best) { best = v; bestC = c; }
        }
        labels.data[i] = static_cast<uint8_t>(bestC);
    }
    return labels;
}

} // namespace seg
