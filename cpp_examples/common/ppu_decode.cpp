#include "ppu_decode.h"

#include <algorithm>
#include <cstddef>
#include <dxrt/datatype.h>

namespace ppu {

namespace {

// Per-class NMS over normalized xyxy Rect2f boxes.
template <typename T>
std::vector<T> nmsByClass(std::vector<T>& dets, float iouThresh) {
    if (dets.empty()) return {};
    std::sort(dets.begin(), dets.end(),
              [](const T& a, const T& b) { return a.score > b.score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<T> kept;
    kept.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);

        const auto& bi = dets[i].box;
        const float ai = bi.width * bi.height;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j] || dets[i].classId != dets[j].classId) continue;

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

std::vector<Detection> decodeYolov8Ppu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh)
{
    std::vector<Detection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::BBOX) {
        return out;
    }

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num = static_cast<int>(shape[1]);
    auto* raw = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        const auto& b = raw[i];
        if (b.score < confThresh) continue;

        // YOLOv8 PPU output: pixel coords in model-input space, centre form.
        (void)inputW; (void)inputH;
        const float x1 = b.x - b.w * 0.5f;
        const float y1 = b.y - b.h * 0.5f;

        Detection d;
        d.box = cv::Rect2f(x1, y1, b.w, b.h);
        d.score = b.score;
        d.classId = static_cast<int>(b.label);
        out.push_back(d);
    }

    return nmsByClass(out, iouThresh);
}

std::vector<Detection> decodeYolov5Ppu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers)
{
    std::vector<Detection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::BBOX) {
        return out;
    }

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num = static_cast<int>(shape[1]);
    auto* raw = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        const auto& b = raw[i];
        if (b.score < confThresh) continue;

        const int li = b.layer_idx;
        if (li < 0 || li >= static_cast<int>(anchorLayers.size())) continue;

        const auto& layer = anchorLayers[li];
        const int stride = layer.stride;
        int ai = b.box_idx;
        if (ai < 0) ai = 0;
        if (ai >= static_cast<int>(layer.anchors.size())) {
            ai = static_cast<int>(layer.anchors.size()) - 1;
        }
        const auto [aw, ah] = layer.anchors[ai];

        (void)inputW; (void)inputH;
        const float cx = (b.x * 2.0f - 0.5f + b.grid_x) * stride;
        const float cy = (b.y * 2.0f - 0.5f + b.grid_y) * stride;
        const float w  = (b.w * b.w * 4.0f) * aw;
        const float h  = (b.h * b.h * 4.0f) * ah;
        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;

        Detection d;
        d.box = cv::Rect2f(x1, y1, w, h);
        d.score = b.score;
        d.classId = static_cast<int>(b.label);
        out.push_back(d);
    }

    return nmsByClass(out, iouThresh);
}

std::vector<Detection> decodeYolov8Float(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh)
{
    (void)inputW; (void)inputH;
    std::vector<Detection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::FLOAT) return out;

    const auto& sh = outputs[0]->shape();
    if (sh.size() < 2) return out;
    const int dimA = static_cast<int>(sh[sh.size() - 2]);
    const int dimB = static_cast<int>(sh[sh.size() - 1]);

    // Smaller axis holds channels (4 + numClasses); larger is anchor count.
    int C, N;
    bool cFirst;
    if (dimA <= dimB) { C = dimA; N = dimB; cFirst = true; }
    else              { C = dimB; N = dimA; cFirst = false; }
    const int numClasses = C - 4;
    if (numClasses <= 0) return out;

    const float* f = static_cast<const float*>(outputs[0]->data());
    auto at = [&](int row, int col) -> float {
        return cFirst ? f[row * N + col] : f[col * C + row];
    };

    out.reserve(64);
    for (int a = 0; a < N; ++a) {
        float best = 0.f;
        int   bestId = 0;
        for (int c = 0; c < numClasses; ++c) {
            const float s = at(4 + c, a);
            if (s > best) { best = s; bestId = c; }
        }
        if (best < confThresh) continue;

        const float cx = at(0, a), cy = at(1, a);
        const float bw = at(2, a), bh = at(3, a);
        Detection d;
        d.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
        d.score = best;
        d.classId = bestId;
        out.push_back(d);
    }
    return nmsByClass(out, iouThresh);
}

std::vector<Detection> decodeYolov5Float(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    float objThresh)
{
    (void)inputW; (void)inputH;
    std::vector<Detection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::FLOAT) return out;

    const auto& sh = outputs[0]->shape();
    if (sh.size() < 2) return out;
    const int dimA = static_cast<int>(sh[sh.size() - 2]);
    const int dimB = static_cast<int>(sh[sh.size() - 1]);

    // Smaller axis holds channels (5 + numClasses); larger is anchor count.
    int C, N;
    bool cFirst;
    if (dimA <= dimB) { C = dimA; N = dimB; cFirst = true; }
    else              { C = dimB; N = dimA; cFirst = false; }
    const int numClasses = C - 5;
    if (numClasses <= 0) return out;

    const float* f = static_cast<const float*>(outputs[0]->data());
    auto at = [&](int row, int col) -> float {
        return cFirst ? f[row * N + col] : f[col * C + row];
    };

    out.reserve(64);
    for (int a = 0; a < N; ++a) {
        const float obj = at(4, a);
        if (obj < objThresh) continue;

        float best = 0.f;
        int   bestId = 0;
        for (int c = 0; c < numClasses; ++c) {
            const float s = at(5 + c, a);
            if (s > best) { best = s; bestId = c; }
        }
        const float score = obj * best;
        if (score < confThresh) continue;

        const float cx = at(0, a), cy = at(1, a);
        const float bw = at(2, a), bh = at(3, a);
        Detection d;
        d.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
        d.score = score;
        d.classId = bestId;
        out.push_back(d);
    }
    return nmsByClass(out, iouThresh);
}

std::vector<AnchorLayer> defaultYolov5Anchors() {
    return {
        {8,  {{10.0f, 13.0f}, {16.0f, 30.0f},  {33.0f, 23.0f}}},
        {16, {{30.0f, 61.0f}, {62.0f, 45.0f},  {59.0f, 119.0f}}},
        {32, {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}},
    };
}

std::vector<PoseDetection> decodePosePpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers)
{
    std::vector<PoseDetection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::POSE) {
        return out;
    }

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num = static_cast<int>(shape[1]);
    auto* raw = static_cast<dxrt::DevicePose_t*>(outputs[0]->data());

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        const auto& p = raw[i];
        if (p.score < confThresh) continue;

        const int li = p.layer_idx;
        if (li < 0 || li >= static_cast<int>(anchorLayers.size())) continue;
        const auto& layer = anchorLayers[li];
        const int stride = layer.stride;
        int ai = p.box_idx;
        if (ai < 0) ai = 0;
        if (ai >= static_cast<int>(layer.anchors.size())) {
            ai = static_cast<int>(layer.anchors.size()) - 1;
        }
        const auto [aw, ah] = layer.anchors[ai];

        (void)inputW; (void)inputH;
        const float cx = (p.x * 2.0f - 0.5f + p.grid_x) * stride;
        const float cy = (p.y * 2.0f - 0.5f + p.grid_y) * stride;
        const float w  = (p.w * p.w * 4.0f) * aw;
        const float h  = (p.h * p.h * 4.0f) * ah;
        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;

        PoseDetection d;
        d.box = cv::Rect2f(x1, y1, w, h);
        d.score = p.score;
        d.classId = 0;

        d.keypoints.reserve(17);
        d.keypointScores.reserve(17);
        for (int k = 0; k < 17; ++k) {
            const float lx = (p.grid_x - 0.5f + p.kpts[k][0] * 2.0f) * stride;
            const float ly = (p.grid_y - 0.5f + p.kpts[k][1] * 2.0f) * stride;
            d.keypoints.emplace_back(lx, ly);
            d.keypointScores.push_back(p.kpts[k][2]);
        }
        out.push_back(std::move(d));
    }

    return nmsByClass(out, iouThresh);
}

std::vector<AnchorLayer> defaultPoseAnchors() {
    return {
        {8,  {{19.0f, 27.0f},  {44.0f, 40.0f},  {38.0f, 94.0f}}},
        {16, {{96.0f, 68.0f},  {86.0f, 152.0f}, {180.0f, 137.0f}}},
        {32, {{140.0f, 301.0f}, {303.0f, 264.0f}, {238.0f, 542.0f}}},
        {64, {{436.0f, 615.0f}, {739.0f, 380.0f}, {925.0f, 792.0f}}},
    };
}

std::vector<PoseDetection> decodeHandPosePpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers,
    int kptCount)
{
    (void)inputW; (void)inputH;
    std::vector<PoseDetection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::POSE) return out;

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num = static_cast<int>(shape[1]);
    auto* bytes = static_cast<uint8_t*>(outputs[0]->data());

    constexpr int headerSize = 28;
    const int kptsSize   = kptCount * 3 * sizeof(float);
    const int rawSize    = headerSize + kptsSize;
    const int stride     = ((rawSize + 31) / 32) * 32;   // 32-byte align

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        uint8_t* elem = bytes + i * stride;

        const float x  = *reinterpret_cast<float*>(elem + 0);
        const float y  = *reinterpret_cast<float*>(elem + 4);
        const float w  = *reinterpret_cast<float*>(elem + 8);
        const float h  = *reinterpret_cast<float*>(elem + 12);
        const uint8_t gY = *(elem + 16);
        const uint8_t gX = *(elem + 17);
        const uint8_t bIdx = *(elem + 18);
        const uint8_t lIdx = *(elem + 19);
        const float score = *reinterpret_cast<float*>(elem + 20);

        if (score < confThresh) continue;
        if (lIdx >= anchorLayers.size()) continue;
        const auto& layer = anchorLayers[lIdx];
        const int st = layer.stride;
        if (bIdx >= layer.anchors.size()) continue;
        const auto [aw, ah] = layer.anchors[bIdx];

        const float cx = (x * 2.0f - 0.5f + gX) * st;
        const float cy = (y * 2.0f - 0.5f + gY) * st;
        const float bw = (w * w * 4.0f) * aw;
        const float bh = (h * h * 4.0f) * ah;

        PoseDetection d;
        d.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
        d.score = score;
        d.classId = 0;

        const float* kpts = reinterpret_cast<const float*>(elem + headerSize);
        d.keypoints.reserve(kptCount);
        d.keypointScores.reserve(kptCount);
        for (int k = 0; k < kptCount; ++k) {
            const float lx = (gX - 0.5f + kpts[k * 3 + 0] * 2.0f) * st;
            const float ly = (gY - 0.5f + kpts[k * 3 + 1] * 2.0f) * st;
            d.keypoints.emplace_back(lx, ly);
            d.keypointScores.push_back(kpts[k * 3 + 2]);
        }
        out.push_back(std::move(d));
    }
    return nmsByClass(out, iouThresh);
}

std::vector<AnchorLayer> defaultHandPoseAnchors() {
    return defaultPoseAnchors();
}

std::vector<PoseDetection> decodePoseFloat(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    int kptCount)
{
    (void)inputW; (void)inputH;
    std::vector<PoseDetection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::FLOAT) return out;

    const auto& sh = outputs[0]->shape();
    if (sh.size() < 2) return out;

    // Accept [1, C, N] (YOLOv8/v11-pose, C-first) or [1, N, C] (YOLOv5-pose).
    const int dimA = static_cast<int>(sh[sh.size() - 2]);
    const int dimB = static_cast<int>(sh[sh.size() - 1]);
    const int withVis = 5 + kptCount * 3;
    const int noVis   = 5 + kptCount * 2;

    int C = 0, N = 0;
    bool cFirst = true;
    if (dimA == withVis || dimA == noVis) {
        C = dimA; N = dimB; cFirst = true;
    } else if (dimB == withVis || dimB == noVis) {
        C = dimB; N = dimA; cFirst = false;
    } else {
        return out;
    }
    const int kpStride = (C == withVis) ? 3 : 2;

    const float* f = static_cast<const float*>(outputs[0]->data());
    auto at = [&](int row, int col) -> float {
        return cFirst ? f[row * N + col] : f[col * C + row];
    };

    out.reserve(64);
    for (int a = 0; a < N; ++a) {
        const float score = at(4, a);
        if (score < confThresh) continue;

        const float cx = at(0, a), cy = at(1, a);
        const float bw = at(2, a), bh = at(3, a);

        PoseDetection d;
        d.box = cv::Rect2f(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
        d.score = score;
        d.classId = 0;
        d.keypoints.reserve(kptCount);
        d.keypointScores.reserve(kptCount);
        for (int k = 0; k < kptCount; ++k) {
            const int base = 5 + k * kpStride;
            const float kx = at(base, a);
            const float ky = at(base + 1, a);
            const float kv = (kpStride == 3) ? at(base + 2, a) : 1.0f;
            d.keypoints.emplace_back(kx, ky);
            d.keypointScores.push_back(kv);
        }
        out.push_back(std::move(d));
    }
    return nmsByClass(out, iouThresh);
}

std::vector<FaceDetection> decodeYolov5Face(
    const dxrt::TensorPtrs& outputs,
    float confThresh, float iouThresh)
{
    std::vector<FaceDetection> out;
    if (outputs.empty()) return out;

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num  = static_cast<int>(shape[1]);
    const int elem = (shape.size() >= 3) ? static_cast<int>(shape[2]) : 16;
    // Layout: [cx, cy, w, h, obj, lm0_x, lm0_y, ..., lm4_x, lm4_y, cls_score]
    if (elem < 16) return out;

    const float* data = static_cast<const float*>(outputs[0]->data());
    out.reserve(num);

    for (int i = 0; i < num; ++i) {
        const float* d = data + i * elem;
        const float obj = d[4];
        if (obj < confThresh) continue;
        const float cls = d[15];
        const float score = obj * cls;
        if (score < confThresh) continue;

        const float cx = d[0], cy = d[1], w = d[2], h = d[3];

        FaceDetection fd;
        fd.box = cv::Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
        fd.score = score;
        fd.landmarks.reserve(5);
        for (int k = 0; k < 5; ++k) {
            fd.landmarks.emplace_back(d[5 + k * 2], d[5 + k * 2 + 1]);
        }
        out.push_back(std::move(fd));
    }
    return nmsByClass(out, iouThresh);
}

std::vector<FaceDetection> decodeScrfdPpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<int>& strides)
{
    (void)inputW; (void)inputH;
    std::vector<FaceDetection> out;
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::FACE) return out;

    const auto& shape = outputs[0]->shape();
    if (shape.size() < 2) return out;
    const int num = static_cast<int>(shape[1]);
    auto* raw = static_cast<dxrt::DeviceFace_t*>(outputs[0]->data());

    out.reserve(num);
    for (int i = 0; i < num; ++i) {
        const auto& f = raw[i];
        if (f.score < confThresh) continue;

        const int li = f.layer_idx;
        if (li < 0 || li >= static_cast<int>(strides.size())) continue;
        const float s = static_cast<float>(strides[li]);

        // SCRFD distance-from-centre decode: x/y/w/h are reused as l/t/r/b.
        const float l = f.x, t = f.y, r = f.w, b = f.h;
        const float gx = static_cast<float>(f.grid_x);
        const float gy = static_cast<float>(f.grid_y);

        const float x1 = (gx - l) * s;
        const float y1 = (gy - t) * s;
        const float x2 = (gx + r) * s;
        const float y2 = (gy + b) * s;

        FaceDetection d;
        d.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        d.score = f.score;
        d.classId = 0;
        d.landmarks.reserve(5);
        for (int k = 0; k < 5; ++k) {
            const float kx = (gx + f.kpts[k][0]) * s;
            const float ky = (gy + f.kpts[k][1]) * s;
            d.landmarks.emplace_back(kx, ky);
        }
        out.push_back(std::move(d));
    }
    return nmsByClass(out, iouThresh);
}

} // namespace ppu
