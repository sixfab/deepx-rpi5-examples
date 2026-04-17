#include "visualizer.h"

#include <cstdio>
#include <opencv2/imgproc.hpp>

#include "sdk_utils.h"

namespace vis {

namespace {

// COCO 17-keypoint skeleton — matches VideoWidget::paintEvent exactly.
// Keypoint indices:
//   0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear,
//   5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist,
//   11 L-hip, 12 R-hip, 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle.
const std::vector<std::pair<int,int>> COCO_SKELETON = {
    {5, 7},  {7, 9},  {6, 8},   {8, 10},    // arms
    {11,13}, {13,15}, {12,14},  {14,16},    // legs
    {5, 6},  {11,12}, {5, 11},  {6, 12},    // body cross-connections
    {0, 1},  {0, 2},  {1, 3},   {2, 4},     // face
};

// Keypoints classified by body side for L=green / R=red / centre=yellow colouring.
constexpr int LEFT_KP_BITS  = (1<<1)|(1<<3)|(1<<5)|(1<<7)|(1<<9)
                              |(1<<11)|(1<<13)|(1<<15);
constexpr int RIGHT_KP_BITS = (1<<2)|(1<<4)|(1<<6)|(1<<8)|(1<<10)
                              |(1<<12)|(1<<14)|(1<<16);

enum class Side { CENTRE, LEFT, RIGHT };
inline Side sideOf(int kp) {
    if (LEFT_KP_BITS  & (1<<kp)) return Side::LEFT;
    if (RIGHT_KP_BITS & (1<<kp)) return Side::RIGHT;
    return Side::CENTRE;
}

// BGR
const cv::Scalar COLOR_LEFT (  0, 255,   0);   // green
const cv::Scalar COLOR_RIGHT(  0,   0, 255);   // red
const cv::Scalar COLOR_CENTRE( 0, 255, 255);   // yellow

// 21-keypoint hand skeleton (MediaPipe-style finger chains).
const std::vector<std::pair<int,int>> HAND_SKELETON = {
    {0,1}, {1,2}, {2,3}, {3,4},           // thumb
    {0,5}, {5,6}, {6,7}, {7,8},           // index
    {0,9}, {9,10}, {10,11}, {11,12},      // middle
    {0,13}, {13,14}, {14,15}, {15,16},    // ring
    {0,17}, {17,18}, {18,19}, {19,20},    // pinky
    {5,9}, {9,13}, {13,17},               // palm bridges
};

void drawLabel(cv::Mat& frame, const std::string& text,
               int x, int y, const cv::Scalar& color)
{
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.5;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, font, scale, thickness, &baseline);

    const int labelY = (y - ts.height - 6 > 0) ? (y - 6) : (y + ts.height + 6);
    cv::rectangle(frame,
                  cv::Point(x, labelY - ts.height - 4),
                  cv::Point(x + ts.width + 4, labelY + 4),
                  color, cv::FILLED);
    cv::putText(frame, text, cv::Point(x + 2, labelY),
                font, scale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
}

} // namespace

void drawDetections(cv::Mat& frame,
                    const std::vector<cv::Rect2f>& boxes,
                    const std::vector<float>& scores,
                    const std::vector<int>& classIds,
                    const std::vector<std::string>& labels,
                    float alpha)
{
    const size_t n = std::min({boxes.size(), scores.size(), classIds.size()});

    // Translucent fill layer first.
    if (alpha > 0.0f) {
        cv::Mat overlay = frame.clone();
        for (size_t i = 0; i < n; ++i) {
            const cv::Scalar c = sdk::colorForClass(classIds[i]);
            cv::rectangle(overlay, boxes[i], c, cv::FILLED);
        }
        cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame);
    }

    for (size_t i = 0; i < n; ++i) {
        const cv::Scalar c = sdk::colorForClass(classIds[i]);
        cv::rectangle(frame, boxes[i], c, 2);

        std::string text;
        if (classIds[i] >= 0 && classIds[i] < static_cast<int>(labels.size())) {
            text = labels[classIds[i]];
        }
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d%%",
                      static_cast<int>(scores[i] * 100.0f + 0.5f));
        if (!text.empty()) text += " ";
        text += buf;

        drawLabel(frame, text,
                  static_cast<int>(boxes[i].x),
                  static_cast<int>(boxes[i].y),
                  c);
    }
}

namespace {

void drawSkeleton(cv::Mat& frame,
                  const std::vector<cv::Point2f>& kp,
                  const std::vector<float>& scores,
                  const std::vector<std::pair<int,int>>& edges,
                  float kptThresh,
                  const cv::Scalar& lineColor,
                  const cv::Scalar& pointColor)
{
    for (const auto& [a, b] : edges) {
        if (a >= static_cast<int>(kp.size()) || b >= static_cast<int>(kp.size())) continue;
        if (!scores.empty() &&
            (scores[a] < kptThresh || scores[b] < kptThresh)) continue;
        cv::line(frame, kp[a], kp[b], lineColor, 2, cv::LINE_AA);
    }
    for (size_t i = 0; i < kp.size(); ++i) {
        if (!scores.empty() && scores[i] < kptThresh) continue;
        cv::circle(frame, kp[i], 3, pointColor, -1, cv::LINE_AA);
    }
}

} // namespace

void drawBodyPose(cv::Mat& frame,
                  const std::vector<cv::Point2f>& keypoints,
                  const std::vector<float>& scores,
                  float kptThresh)
{
    for (const auto& [a, b] : COCO_SKELETON) {
        if (a >= static_cast<int>(keypoints.size()) ||
            b >= static_cast<int>(keypoints.size())) continue;
        if (!scores.empty() &&
            (scores[a] < kptThresh || scores[b] < kptThresh)) continue;

        const Side sa = sideOf(a), sb = sideOf(b);
        cv::Scalar col = COLOR_CENTRE;
        if (sa == Side::LEFT  && sb == Side::LEFT)  col = COLOR_LEFT;
        else if (sa == Side::RIGHT && sb == Side::RIGHT) col = COLOR_RIGHT;
        cv::line(frame, keypoints[a], keypoints[b], col, 2, cv::LINE_AA);
    }
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (!scores.empty() && scores[i] < kptThresh) continue;
        cv::Scalar col = COLOR_CENTRE;
        if (sideOf(static_cast<int>(i)) == Side::LEFT)  col = COLOR_LEFT;
        else if (sideOf(static_cast<int>(i)) == Side::RIGHT) col = COLOR_RIGHT;
        cv::circle(frame, keypoints[i], 4, col, -1, cv::LINE_AA);
    }
}

void drawHandPose(cv::Mat& frame,
                  const std::vector<cv::Point2f>& keypoints,
                  const std::vector<float>& scores,
                  float kptThresh)
{
    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (!scores.empty() && scores[i] < kptThresh) continue;
        cv::circle(frame, keypoints[i], 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
}

void drawFaceLandmarks(cv::Mat& frame,
                       const cv::Rect2f& box,
                       const std::vector<cv::Point2f>& landmarks)
{
    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
    // Canonical order: L-eye, R-eye, nose, L-mouth-corner, R-mouth-corner.
    static const cv::Scalar kFaceLmColors[5] = {
        cv::Scalar(  0, 255,   0),   // L-eye   — green
        cv::Scalar(255,   0,   0),   // R-eye   — blue
        cv::Scalar(  0, 255, 255),   // nose    — yellow
        cv::Scalar(  0,   0, 255),   // L-mouth — red
        cv::Scalar(  0, 128, 255),   // R-mouth — orange
    };
    for (size_t i = 0; i < landmarks.size() && i < 5; ++i) {
        cv::circle(frame, landmarks[i], 3, kFaceLmColors[i], -1, cv::LINE_AA);
    }
}

void drawSegMasks(cv::Mat& frame,
                  const std::vector<cv::Mat>& masks,
                  const std::vector<int>& classIds,
                  float alpha)
{
    if (masks.empty() || alpha <= 0.0f) return;
    const size_t n = std::min(masks.size(), classIds.size());
    for (size_t i = 0; i < n; ++i) {
        const cv::Mat& m = masks[i];
        if (m.empty() || m.size() != frame.size()) continue;
        const cv::Scalar c = sdk::colorForClass(classIds[i]);
        // Per-pixel blend only where the mask is active — match the Python
        // draw_masks helper which reads mask > 0.5 as a binary region.
        for (int y = 0; y < frame.rows; ++y) {
            const uchar* mp = m.ptr<uchar>(y);
            cv::Vec3b* fp = frame.ptr<cv::Vec3b>(y);
            for (int x = 0; x < frame.cols; ++x) {
                if (!mp[x]) continue;
                for (int k = 0; k < 3; ++k) {
                    fp[x][k] = static_cast<uchar>(
                        fp[x][k] * (1.0f - alpha) + c[k] * alpha);
                }
            }
        }
    }
}

void drawSemanticSeg(cv::Mat& frame,
                     const cv::Mat& labelMap,
                     const std::vector<cv::Scalar>& palette,
                     float alpha)
{
    if (labelMap.empty() || palette.empty()) return;
    cv::Mat resized;
    if (labelMap.size() != frame.size()) {
        cv::resize(labelMap, resized, frame.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        resized = labelMap;
    }
    cv::Mat coloured(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    const int n = static_cast<int>(palette.size());
    for (int y = 0; y < resized.rows; ++y) {
        const uchar* lp = resized.ptr<uchar>(y);
        cv::Vec3b* cp = coloured.ptr<cv::Vec3b>(y);
        for (int x = 0; x < resized.cols; ++x) {
            int cls = lp[x];
            if (cls < 0 || cls >= n) continue;
            const cv::Scalar& s = palette[cls];
            cp[x] = cv::Vec3b(
                static_cast<uchar>(s[0]),
                static_cast<uchar>(s[1]),
                static_cast<uchar>(s[2]));
        }
    }
    cv::addWeighted(frame, 1.0 - alpha, coloured, alpha, 0.0, frame);
}

void drawClassification(cv::Mat& frame,
                        const std::vector<std::pair<std::string, float>>& results)
{
    if (results.empty()) return;

    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.6;
    const int thickness = 1;
    const int lineH = 22;
    const int padX  = 6;
    const int padY  = 6;

    std::vector<std::string> lines;
    lines.reserve(results.size());
    int maxW = 0;
    for (const auto& [label, prob] : results) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%5.1f%%  %s", prob * 100.0f, label.c_str());
        lines.emplace_back(buf);
        const cv::Size ts = cv::getTextSize(lines.back(), font, scale, thickness, nullptr);
        if (ts.width > maxW) maxW = ts.width;
    }

    const int x0 = 10;
    const int y0 = 10;
    const int boxW = maxW + padX * 2;
    const int boxH = lineH * static_cast<int>(lines.size()) + padY * 2;

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay,
                  cv::Point(x0, y0),
                  cv::Point(x0 + boxW, y0 + boxH),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame);

    for (size_t i = 0; i < lines.size(); ++i) {
        const int ty = y0 + padY + static_cast<int>(i + 1) * lineH - 6;
        cv::putText(frame, lines[i],
                    cv::Point(x0 + padX, ty),
                    font, scale, cv::Scalar(255, 255, 255),
                    thickness, cv::LINE_AA);
    }
}

void drawFps(cv::Mat& frame, double fps) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);

    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.7;
    const int thickness = 2;
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(buf, font, scale, thickness, &baseline);

    const int pad = 6;
    const int x = 10;
    const int y = 10 + ts.height;
    cv::rectangle(frame,
                  cv::Point(x - pad, y - ts.height - pad),
                  cv::Point(x + ts.width + pad, y + baseline),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, buf, cv::Point(x, y),
                font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

} // namespace vis
