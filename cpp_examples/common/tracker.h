#pragma once

#include <map>
#include <vector>
#include <opencv2/core.hpp>

// Greedy nearest-neighbor centroid tracker.
// Ported verbatim from PeopleTrackingAdapter.cpp (the same algorithm is
// used by SmartTrafficAdapter and StoreQueueAnalysisAdapter).
// All centroids are in normalized [0, 1] frame coordinates so the
// distance threshold is resolution-agnostic.
class CentroidTracker {
public:
    struct TrackedObject {
        int                id;
        cv::Point2f        centroid;
        int                missedFrames;
        std::vector<float> embedding;   // empty unless the embedding-aware update is used
    };

    explicit CentroidTracker(int maxMissed = 10, float maxDistance = 0.1f);

    // Match the given centroids against existing tracks. Unmatched centroids
    // spawn new tracks; tracks missing for > maxMissed frames are pruned.
    // Returns a map from input-index -> assigned track id.
    std::map<int, int> update(const std::vector<cv::Point2f>& centroids);

    // Embedding-aware variant. Match cost combines spatial distance (normalised
    // by maxDistance) and appearance distance (1 - cosine_similarity), weighted
    // by appearanceWeight. Detections with empty embeddings fall back to pure
    // spatial matching. Matched tracks EMA their embedding (0.9 old + 0.1 new,
    // re-normalised).
    std::map<int, int> update(const std::vector<cv::Point2f>& centroids,
                              const std::vector<std::vector<float>>& embeddings,
                              float appearanceWeight = 0.4f);

    // Currently-active tracks (including those pending a prune).
    const std::map<int, TrackedObject>& tracks() const { return m_tracks; }

    void reset();

private:
    int                            m_nextId = 0;
    int                            m_maxMissed;
    float                          m_maxDistance;
    std::map<int, TrackedObject>   m_tracks;
};
