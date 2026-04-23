#pragma once

#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <dxrt/dxrt_api.h>

// Helpers for person Re-ID with an OSNet-family embedding model.
// The model ingests a 128x256 (W x H) RGB crop and emits a [1, 512]
// feature vector that is expected to be L2-normalised for cosine matching.
namespace reid {

// In-place L2 normalisation. No-op for zero vectors.
void l2Normalize(std::vector<float>& v);

// Cosine similarity between two vectors. Assumes both are already L2-normalised;
// returns 0 if sizes differ or either side is empty.
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

// Crop `box` from `bgrFrame`, resize to (inputW x inputH), swap to RGB, and run
// the engine. Returns an L2-normalised embedding, or an empty vector if the
// crop is too small or the engine produces no output.
// minCropSize is measured in source-frame pixels (applied to both dimensions).
std::vector<float> extractEmbedding(
    dxrt::InferenceEngine& engine,
    const cv::Mat& bgrFrame,
    const cv::Rect2f& box,
    int inputH, int inputW,
    int minCropSize = 20);

} // namespace reid
