#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace seg {

struct SegDetection {
    cv::Rect2f box;                  // xyxy in model-input pixel coords
    float score;
    int classId;
    std::vector<float> maskCoeffs;   // 32 prototype coefficients (YOLOv8/v26-seg)
};

// Decode YOLOv8-seg detection output.
// detPtr: raw float pointer, shape [1, 4 + numClasses + maskDim, N] (C-first).
// Per-anchor row layout after transpose: [cx, cy, w, h, cls_0..cls_{C-1},
// mask_coef_0..mask_coef_{M-1}]. Confidence gate + per-class NMS applied.
std::vector<SegDetection> decodeYolov8Seg(
    const float* detPtr,
    int numClasses, int maskDim, int numAnchors,
    int inputW, int inputH,
    float confThresh, float iouThresh);

// Decode YOLOv26-seg detection output. YOLOv26 ships with NMS baked in and
// emits rows of [x1, y1, x2, y2, score, class_id, mask_coef_0..].
// detPtr: raw float pointer, shape [1, N, 6 + maskDim].
std::vector<SegDetection> decodeYolo26Seg(
    const float* detPtr,
    int maskDim, int numRows,
    int inputW, int inputH,
    float confThresh);

// Combine prototype masks with per-detection coefficients to produce one
// binary mask per detection in original-frame pixel space.
//   protoPtr : [maskDim, maskH, maskW] (float, batch-1 squeezed)
//   dets     : output of decodeYolov8Seg / decodeYolo26Seg (model-input coords)
//   gain/pad : letterbox parameters (from sdk::letterbox)
std::vector<cv::Mat> generateMasks(
    const float* protoPtr,
    int maskDim, int maskH, int maskW,
    const std::vector<SegDetection>& dets,
    int inputW, int inputH,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize);

// Decode DeepLabV3+ semantic-segmentation logits.
// logitsPtr: raw float pointer, shape [1, numClasses, H, W].
// Returns a CV_8U label map (H x W) containing argmax class ids.
cv::Mat decodeDeeplabV3(
    const float* logitsPtr,
    int numClasses, int h, int w);

} // namespace seg
