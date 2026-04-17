// osnet_reid_demo.cpp — YOLOv5 PPU person detection + OSNet appearance Re-ID.
// CentroidTracker combines spatial distance with cosine similarity on 512-d
// embeddings for stable IDs. Embeddings run every reidInterval frames.

#include "config_loader.h"
#include "demo_runner.h"
#include "input_source.h"
#include "ppu_decode.h"
#include "reid_utils.h"
#include "sdk_utils.h"
#include "tracker.h"

#include <dxrt/dxrt_api.h>
#include <opencv2/imgproc.hpp>
#include <cstdio>

namespace {

constexpr int kPersonClassId = 0;
constexpr int kReidInputH    = 256;      // OSNet portrait: H then W
constexpr int kReidInputW    = 128;
constexpr int kMinCropSize   = 20;

class ReIDTracker {
public:
    explicit ReIDTracker(const DemoParams& p)
        : m_inputH(p.inputHeight), m_inputW(p.inputWidth),
          m_conf(p.confThreshold), m_iou(p.iouThreshold),
          m_reidH(kReidInputH), m_reidW(kReidInputW)
    {
        m_det = sdk::loadEngine(p.modelPath, m_inputH, m_inputW);
        if (!m_det) throw std::runtime_error("Failed to load detector");
        if (p.modelPath2.empty())
            throw std::runtime_error("osnet_reid_demo requires model_path_2 (OSNet)");
        int rh = m_reidH, rw = m_reidW;
        m_reid = sdk::loadEngine(p.modelPath2, rh, rw);
        if (!m_reid) throw std::runtime_error("Failed to load OSNet embedding model");
        m_anchors = ppu::defaultYolov5Anchors();
    }

    struct Result { cv::Rect2f box; float score; int trackId; };

    std::vector<Result> infer(const cv::Mat& bgr, int reidInterval) {
        auto lb    = sdk::letterbox(bgr, m_inputH, m_inputW);
        auto reqId = m_det->RunAsync(lb.image.data, nullptr, nullptr);
        auto outs  = m_det->Wait(reqId);
        auto dets  = ppu::decodeYolov5Ppu(outs, m_inputW, m_inputH,
                                          m_conf, m_iou, m_anchors);

        std::vector<cv::Rect2f> rawBoxes;
        for (const auto& d : dets) rawBoxes.push_back(d.box);
        auto px = sdk::unletterboxBoxes(rawBoxes, lb.gain, lb.pad, bgr.size());

        std::vector<cv::Rect2f> personBoxes; std::vector<float> scores;
        for (size_t i = 0; i < dets.size(); ++i) {
            if (dets[i].classId != kPersonClassId) continue;
            personBoxes.push_back(px[i]); scores.push_back(dets[i].score);
        }

        // Empty embeddings on skipped frames -> tracker uses spatial-only match.
        const bool runReid = (m_frameIdx++ % std::max(1, reidInterval)) == 0;
        std::vector<std::vector<float>> embeddings(personBoxes.size());
        if (runReid) {
            for (size_t i = 0; i < personBoxes.size(); ++i)
                embeddings[i] = reid::extractEmbedding(
                    *m_reid, bgr, personBoxes[i], m_reidH, m_reidW, kMinCropSize);
        }

        std::vector<cv::Point2f> centroids;
        const float W = static_cast<float>(bgr.cols), H = static_cast<float>(bgr.rows);
        centroids.reserve(personBoxes.size());
        for (const auto& b : personBoxes)
            centroids.emplace_back((b.x + b.width * 0.5f) / W,
                                   (b.y + b.height * 0.5f) / H);

        auto assignments = m_tracker.update(centroids, embeddings, /*appearanceW=*/0.4f);

        std::vector<Result> out;
        out.reserve(personBoxes.size());
        for (size_t i = 0; i < personBoxes.size(); ++i) {
            Result r{personBoxes[i], scores[i], -1};
            auto it = assignments.find(static_cast<int>(i));
            if (it != assignments.end()) r.trackId = it->second;
            out.push_back(r);
        }
        return out;
    }

private:
    std::unique_ptr<dxrt::InferenceEngine> m_det, m_reid;
    int   m_inputH, m_inputW;
    float m_conf, m_iou;
    int   m_reidH, m_reidW;
    std::vector<ppu::AnchorLayer> m_anchors;
    CentroidTracker m_tracker{/*maxMissed=*/30, /*maxDistance=*/0.15f};
    int   m_frameIdx = 0;
};

} // namespace

int main(int argc, char** argv) {
    auto p = loadConfig("", argc, argv);

    ReIDTracker model(p);
    InputSource source(parseSourceType(p.sourceType), p.sourcePath, p.cameraIndex);
    if (!source.isOpened()) {
        std::fprintf(stderr, "[ERROR] Failed to open source: %s\n", p.sourcePath.c_str());
        return 1;
    }
    DemoConfig cfg;
    cfg.windowTitle = p.windowTitle;
    cfg.showFps     = p.showFps;

    runDemo(source, [&](cv::Mat& frame) {
        for (const auto& r : model.infer(frame, /*reidInterval=*/3)) {
            const cv::Scalar color = sdk::colorForClass(r.trackId);
            cv::rectangle(frame, r.box, color, 2);
            char buf[32]; std::snprintf(buf, sizeof(buf), "ID %d", r.trackId);
            const cv::Point pt(static_cast<int>(r.box.x),
                               std::max(15, static_cast<int>(r.box.y) - 6));
            cv::putText(frame, buf, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(frame, buf, pt, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv::LINE_AA);
        }
    }, cfg);

    return 0;
}
