#include "zone_utils.h"

#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace zone {

bool pointInPolygon(const cv::Point2f& p,
                    const std::vector<cv::Point2f>& polygon)
{
    if (polygon.size() < 3) return false;
    // Ray-cast — same formula as MultiChannelAdapter::isPointInPolygon.
    bool inside = false;
    const size_t n = polygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const auto& a = polygon[i];
        const auto& b = polygon[j];
        if (((a.y > p.y) != (b.y > p.y)) &&
            (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) {
            inside = !inside;
        }
    }
    return inside;
}

namespace {
inline float cross(const cv::Point2f& o,
                   const cv::Point2f& a,
                   const cv::Point2f& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}
}

bool segmentCrossesLine(const cv::Point2f& p1, const cv::Point2f& p2,
                        const cv::Point2f& a,  const cv::Point2f& b)
{
    const float cp1 = cross(a, b, p1);
    const float cp2 = cross(a, b, p2);
    if ((cp1 > 0 && cp2 < 0) || (cp1 < 0 && cp2 > 0)) {
        const float cp3 = cross(p1, p2, a);
        const float cp4 = cross(p1, p2, b);
        if ((cp3 > 0 && cp4 < 0) || (cp3 < 0 && cp4 > 0)) {
            return true;
        }
    }
    return false;
}

void drawFilledPolygon(cv::Mat& frame,
                       const std::vector<cv::Point2f>& polygon,
                       const cv::Scalar& colorBgr,
                       float alpha)
{
    if (polygon.size() < 3) return;
    std::vector<cv::Point> pts;
    pts.reserve(polygon.size());
    for (const auto& pt : polygon) pts.emplace_back(cvRound(pt.x), cvRound(pt.y));

    cv::Mat overlay = frame.clone();
    const cv::Point* ppt[1] = { pts.data() };
    int npt[] = { static_cast<int>(pts.size()) };
    cv::fillPoly(overlay, ppt, npt, 1, colorBgr);
    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame);
    cv::polylines(frame, ppt, npt, 1, /*closed=*/true, colorBgr, 2, cv::LINE_AA);
}

void drawZoneOverlay(cv::Mat& frame,
                     const std::vector<cv::Point2f>& polygon,
                     bool alertActive,
                     float alpha)
{
    // BGR: cyan when idle, red when alarm-triggered.
    const cv::Scalar color = alertActive ? cv::Scalar(0, 0, 200)
                                         : cv::Scalar(200, 200, 0);
    drawFilledPolygon(frame, polygon, color, alpha);
}

void drawCountingLine(cv::Mat& frame,
                      const cv::Point2f& a, const cv::Point2f& b,
                      int countIn, int countOut)
{
    cv::line(frame, a, b, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    // Small arrow midpoint -> perpendicular, to hint at "in" direction.
    const cv::Point2f mid{(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
    cv::Point2f dir{b.x - a.x, b.y - a.y};
    const float len = std::hypot(dir.x, dir.y);
    if (len > 1e-3f) {
        dir.x /= len; dir.y /= len;
        const cv::Point2f perp{-dir.y, dir.x};
        const cv::Point2f tip{mid.x + perp.x * 24.0f, mid.y + perp.y * 24.0f};
        cv::arrowedLine(frame, mid, tip, cv::Scalar(0, 255, 0), 2,
                        cv::LINE_AA, 0, 0.4);
    }

    (void)countOut;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Count: %d", countIn);
    cv::putText(frame, buf, cv::Point(10, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, buf, cv::Point(10, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

void drawZoneStats(cv::Mat& frame,
                   const cv::Point2f& labelPos,
                   const std::string& label,
                   int count, float waitSec)
{
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s: %d (~%.1fs)", label.c_str(), count, waitSec);

    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.55;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(buf, font, scale, thickness, &baseline);

    const int pad = 4;
    const cv::Point tl(static_cast<int>(labelPos.x),
                       static_cast<int>(labelPos.y));
    cv::rectangle(frame,
                  cv::Point(tl.x - pad, tl.y - pad),
                  cv::Point(tl.x + ts.width + pad, tl.y + ts.height + baseline + pad),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, buf, cv::Point(tl.x, tl.y + ts.height),
                font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

std::vector<cv::Point2f> toPixels(const std::vector<cv::Point2f>& normPts,
                                  const cv::Size& frameSize)
{
    std::vector<cv::Point2f> out;
    out.reserve(normPts.size());
    for (const auto& n : normPts) {
        out.emplace_back(n.x * frameSize.width, n.y * frameSize.height);
    }
    return out;
}

} // namespace zone
