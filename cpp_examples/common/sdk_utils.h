#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <dxrt/dxrt_api.h>

namespace sdk {

struct LetterboxResult {
    cv::Mat image;      // padded & resized, RGB
    float gain;         // uniform scale factor applied to width/height
    cv::Point2f pad;    // (pad_x, pad_y) in pixels (left, top)
};

// Initialize the DeepX device pool. Safe to call repeatedly.
void initDevice();

// Load a .dxnn inference engine and populate input spatial dims via the
// reference-parameters. Falls back to the caller's pre-set values when the
// engine cannot report them. Prints a friendly error and returns nullptr on
// failure.
std::unique_ptr<dxrt::InferenceEngine> loadEngine(
    const std::string& modelPath, int& inputH, int& inputW);

// Resize + pad a BGR frame into the (targetH, targetW) model input.
// Padding value is 114 (the YOLO convention). Output is RGB.
LetterboxResult letterbox(const cv::Mat& bgrFrame, int targetH, int targetW);

// Map model-input-space boxes back to original-frame pixel coords.
// boxes: xywh in pixel coords of the (padded) model input tensor.
std::vector<cv::Rect2f> unletterboxBoxes(
    const std::vector<cv::Rect2f>& boxes,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize);

// Map model-input-space points back to original-frame pixel coords.
// points: (x, y) in pixel coords of the (padded) model input tensor.
std::vector<cv::Point2f> unletterboxPoints(
    const std::vector<cv::Point2f>& points,
    float gain, cv::Point2f pad,
    const cv::Size& originalSize);

// Read one label per line from path. Returns fallback when the file is
// missing or empty so demos still run out of the box.
std::vector<std::string> loadLabels(
    const std::string& path,
    const std::vector<std::string>& fallback);

// Stable BGR colour per class id for drawing.
cv::Scalar colorForClass(int classId);

} // namespace sdk
