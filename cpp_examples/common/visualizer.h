#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace vis {

// Draw axis-aligned detection boxes with an optional translucent fill and
// "classname 87%" style labels. boxes are pixel coords in frame space.
void drawDetections(
    cv::Mat& frame,
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& classIds,
    const std::vector<std::string>& labels,
    float alpha = 0.3f);

// Draw 17-keypoint COCO body pose skeleton. keypoints are in pixel coords.
// Low-score keypoints are skipped when score < kptThresh.
void drawBodyPose(
    cv::Mat& frame,
    const std::vector<cv::Point2f>& keypoints,
    const std::vector<float>& scores,
    float kptThresh = 0.3f);

// Draw 21-keypoint hand skeleton. keypoints are in pixel coords.
void drawHandPose(
    cv::Mat& frame,
    const std::vector<cv::Point2f>& keypoints,
    const std::vector<float>& scores,
    float kptThresh = 0.3f);

// Draw 5-point face landmarks (eyes, nose, mouth corners).
void drawFaceLandmarks(
    cv::Mat& frame,
    const cv::Rect2f& box,
    const std::vector<cv::Point2f>& landmarks);

// Overlay "FPS: NN.N" in the top-left corner.
void drawFps(cv::Mat& frame, double fps);

// Draw instance segmentation masks as coloured overlays, one colour per class.
// masks: CV_8U binary masks (0 or 255) already resized to the frame size.
// classIds: class index per mask, used to pick the overlay colour.
// alpha: overlay opacity (0 disables the blend and skips all drawing).
void drawSegMasks(
    cv::Mat& frame,
    const std::vector<cv::Mat>& masks,
    const std::vector<int>& classIds,
    float alpha = 0.4f);

// Draw a semantic segmentation label map as a translucent coloured overlay.
// labelMap: CV_8U argmax result; resized to the frame size with nearest-neighbor
//           to preserve sharp class boundaries.
// palette:  BGR colour per class index.
// alpha:    overlay opacity.
void drawSemanticSeg(
    cv::Mat& frame,
    const cv::Mat& labelMap,
    const std::vector<cv::Scalar>& palette,
    float alpha = 0.55f);

// Draw a top-K classification result as a text block in the top-left corner.
// Each entry is rendered as "<pct>%  <label>" on its own line.
void drawClassification(
    cv::Mat& frame,
    const std::vector<std::pair<std::string, float>>& results);

} // namespace vis
