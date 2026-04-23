#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace zone {

// Ray-cast point-in-polygon test. polygon points and p share the same
// coordinate system (either normalized or pixel — the test is topological).
bool pointInPolygon(const cv::Point2f& p,
                    const std::vector<cv::Point2f>& polygon);

// Segment-segment crossing test. Returns true if p1->p2 actually crosses
// the segment a->b. Ports SmartTrafficAdapter::checkLineCrossing verbatim:
// cross-product sign check on both segments.
bool segmentCrossesLine(const cv::Point2f& p1, const cv::Point2f& p2,
                        const cv::Point2f& a,  const cv::Point2f& b);

// Semi-transparent filled polygon overlay with a crisp outline on top.
// polygon in pixel coords. alertActive flips the colour from cyan to red.
void drawZoneOverlay(cv::Mat& frame,
                     const std::vector<cv::Point2f>& polygon,
                     bool alertActive,
                     float alpha = 0.25f);

// Same as drawZoneOverlay but with an explicit BGR colour, no alert flag.
void drawFilledPolygon(cv::Mat& frame,
                       const std::vector<cv::Point2f>& polygon,
                       const cv::Scalar& colorBgr,
                       float alpha = 0.25f);

// Counting line with "IN: x  OUT: y" labels. a, b in pixel coords.
void drawCountingLine(cv::Mat& frame,
                      const cv::Point2f& a, const cv::Point2f& b,
                      int countIn, int countOut);

// Zone statistics card (label + people count + wait-time estimate).
// labelPos is the top-left of the card in pixel coords.
void drawZoneStats(cv::Mat& frame,
                   const cv::Point2f& labelPos,
                   const std::string& label,
                   int count, float waitSec);

// Convert normalized points to pixel coords using the given frame size.
std::vector<cv::Point2f> toPixels(const std::vector<cv::Point2f>& normPts,
                                  const cv::Size& frameSize);

} // namespace zone
