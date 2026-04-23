#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <dxrt/dxrt_api.h>

namespace ppu {

struct Detection {
    cv::Rect2f box;   // pixel coords in model-input space, xywh form
    float score;
    int classId;
};

struct PoseDetection {
    cv::Rect2f box;                       // pixel coords in model-input space, xywh
    float score;
    int classId;
    std::vector<cv::Point2f> keypoints;   // pixel coords in model-input space
    std::vector<float> keypointScores;
};

struct AnchorLayer {
    int stride;
    std::vector<std::pair<float, float>> anchors;  // (w, h) per anchor index
};

// Decode a YOLOv8 (anchor-free) PPU output tensor.
// outputs: result of engine->Wait()/Run(); expects first tensor of type BBOX.
// conf/iou thresholds are applied with per-class NMS.
std::vector<Detection> decodeYolov8Ppu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh);

// Decode a YOLOv5/v7 (anchor-based) PPU output tensor.
// anchorLayers: one entry per stride, with 3 (w, h) anchors each.
std::vector<Detection> decodeYolov5Ppu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers);

// Decode a default (non-PPU) YOLOv8/v11 FLOAT tensor.
// Shape: [1, 4+C, N] (C-first) or [1, N, 4+C] — auto-detected.
// Per row: [cx, cy, w, h, cls_0, cls_1, ...] with NO objectness channel.
// Coords already in letterboxed pixel space.
std::vector<Detection> decodeYolov8Float(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh);

// Decode a default (non-PPU) YOLOv5/v7 FLOAT tensor.
// Shape: [1, N, 5+C] or [1, 5+C, N] — auto-detected.
// Per row: [cx, cy, w, h, obj, cls_0, cls_1, ...]. Final conf = obj * max(cls).
// Coords already in letterboxed pixel space (detect head baked in).
std::vector<Detection> decodeYolov5Float(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    float objThresh = 0.25f);

// Default YOLOv5s anchors — strides 8/16/32, matches `YoloPostProcess` defaults.
std::vector<AnchorLayer> defaultYolov5Anchors();

// Decode a YOLO-pose PPU POSE output tensor (17 COCO body keypoints).
// Only kptCount == 17 is supported by the hardware DevicePose_t struct.
std::vector<PoseDetection> decodePosePpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers);

// Default 4-layer anchors used by the body-pose model (strides 8/16/32/64).
std::vector<AnchorLayer> defaultPoseAnchors();

// Decode a YOLO-hand-pose PPU POSE output tensor for any keypoint count
// (21 for hand). Parses the raw byte stream because DevicePose_t is
// hardcoded to 17 kpts and must be stride-aligned for N != 17.
// Layout (byte offsets inside each element):
//     0: x(f32) 4: y(f32) 8: w(f32) 12: h(f32)
//    16: grid_y(u8) 17: grid_x(u8) 18: box_idx(u8) 19: layer_idx(u8)
//    20: score(f32) 24: label(u32) 28: kpts[N][3](f32)
// Each element is rounded up to the next 32-byte multiple.
std::vector<PoseDetection> decodeHandPosePpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<AnchorLayer>& anchorLayers,
    int kptCount);

// Default 4-layer anchors used by the hand-pose model (same strides as body-pose).
std::vector<AnchorLayer> defaultHandPoseAnchors();

// Decode a YOLO-pose FLOAT tensor (NOT a PPU struct output). Used by models
// that bypass the PPU and emit a raw [1, C, N] or [1, N, C] float grid, where
//   C = 5 + kptCount * 3    (with per-kpt visibility)
// or
//   C = 5 + kptCount * 2    (x/y only)
// Each row: [cx, cy, w, h, obj, kpt0_x, kpt0_y, (kpt0_v,) ... kptK-1_*]
// All coords already in model-input pixel space — no anchor math is applied.
std::vector<PoseDetection> decodePoseFloat(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    int kptCount);

struct FaceDetection {
    cv::Rect2f box;                       // pixel coords in model-input space, xywh
    float score;
    int classId = 0;                      // always 0 — the template NMS needs it
    std::vector<cv::Point2f> landmarks;   // 5 face landmarks (left-eye, right-eye,
                                          //   nose, left-mouth, right-mouth) in
                                          //   pixel coords of model-input space
};

// Decode a YOLOv5-Face float tensor output (NOT a PPU struct output).
// Expects shape [1, N, 16] where each row is:
//   [cx, cy, w, h, obj_conf, cls_conf, lm0x, lm0y, ... lm4x, lm4y]
// Values are assumed pre-decoded by the model / dxrt, so no sigmoid or
// anchor math is applied here — this matches the layout seen in
// FaceAlignmentAdapter.cpp.
std::vector<FaceDetection> decodeYolov5Face(
    const dxrt::TensorPtrs& outputs,
    float confThresh, float iouThresh);

// Decode an SCRFD PPU output tensor. The PPU returns a DataType::FACE stream
// whose elements reuse the DeviceFace_t struct layout (x,y,w,h,
// grid_y, grid_x, box_idx, layer_idx, score, kpts[5][2]) but with field
// semantics specific to SCRFD:
//   - x/y/w/h are reused as (l, t, r, b) distances-from-centre in grid units
//   - layer_idx indexes into the `strides` vector (typically {8, 16, 32})
// Returns face detections in model-input pixel-space coordinates, with 5
// facial landmarks per face and class id 0.
std::vector<FaceDetection> decodeScrfdPpu(
    const dxrt::TensorPtrs& outputs,
    int inputW, int inputH,
    float confThresh, float iouThresh,
    const std::vector<int>& strides = {8, 16, 32});

} // namespace ppu
