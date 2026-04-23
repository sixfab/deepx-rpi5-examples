#include "tracker.h"

#include <cmath>
#include <limits>
#include <set>

CentroidTracker::CentroidTracker(int maxMissed, float maxDistance)
    : m_maxMissed(maxMissed), m_maxDistance(maxDistance) {}

namespace {

float cosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;   // assumes both are L2-normalised
}

void emaUpdate(std::vector<float>& dst, const std::vector<float>& src) {
    if (src.empty()) return;
    if (dst.size() != src.size()) { dst = src; return; }
    float sumSq = 0.0f;
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = 0.9f * dst[i] + 0.1f * src[i];
        sumSq += dst[i] * dst[i];
    }
    const float norm = std::sqrt(sumSq);
    if (norm > 1e-6f) for (auto& v : dst) v /= norm;
}

} // namespace

void CentroidTracker::reset() {
    m_tracks.clear();
    m_nextId = 0;
}

std::map<int, int> CentroidTracker::update(
    const std::vector<cv::Point2f>& centroids)
{
    // Step 1: age every existing track. Matched tracks below will have
    // missedFrames reset back to 0.
    for (auto& kv : m_tracks) kv.second.missedFrames++;

    std::map<int, int> assignments;     // input index -> track id
    std::set<size_t>   matched;          // detection indices already claimed

    // Step 2: greedy match — for each existing track pick the closest
    // unmatched detection within m_maxDistance. Outer loop is tracks
    // (matches PeopleTrackingAdapter.cpp lines 50-74).
    for (auto& kv : m_tracks) {
        auto& track = kv.second;

        int    bestIdx  = -1;
        float  bestDist = m_maxDistance;

        for (size_t i = 0; i < centroids.size(); ++i) {
            if (matched.count(i)) continue;
            const float dx = track.centroid.x - centroids[i].x;
            const float dy = track.centroid.y - centroids[i].y;
            const float d  = std::sqrt(dx*dx + dy*dy);
            if (d < bestDist) {
                bestDist = d;
                bestIdx  = static_cast<int>(i);
            }
        }

        if (bestIdx >= 0) {
            track.centroid    = centroids[bestIdx];
            track.missedFrames = 0;
            matched.insert(bestIdx);
            assignments[bestIdx] = track.id;
        }
    }

    // Step 3: every unmatched detection becomes a new track.
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (matched.count(i)) continue;
        const int newId = m_nextId++;
        m_tracks[newId] = TrackedObject{newId, centroids[i], 0};
        assignments[static_cast<int>(i)] = newId;
    }

    // Step 4: prune stale tracks.
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->second.missedFrames > m_maxMissed) it = m_tracks.erase(it);
        else                                       ++it;
    }

    return assignments;
}

std::map<int, int> CentroidTracker::update(
    const std::vector<cv::Point2f>& centroids,
    const std::vector<std::vector<float>>& embeddings,
    float appearanceWeight)
{
    for (auto& kv : m_tracks) kv.second.missedFrames++;

    std::map<int, int> assignments;
    std::set<size_t>   matched;
    const float        spatialW = 1.0f - appearanceWeight;

    // Greedy match per track. Cost = w_s * (d_spatial / maxDistance)
    //                              + w_a * (1 - cosine_sim).
    // Gating: the spatial term alone must remain below 1.0, i.e. raw distance
    // < m_maxDistance — a mostly-occluded detection can't teleport across the
    // frame just because its embedding matches.
    for (auto& kv : m_tracks) {
        auto& track = kv.second;

        int   bestIdx  = -1;
        float bestCost = std::numeric_limits<float>::max();

        for (size_t i = 0; i < centroids.size(); ++i) {
            if (matched.count(i)) continue;
            const float dx = track.centroid.x - centroids[i].x;
            const float dy = track.centroid.y - centroids[i].y;
            const float d  = std::sqrt(dx*dx + dy*dy);
            if (d >= m_maxDistance) continue;           // spatial gate

            const float sNorm = d / m_maxDistance;
            float cost;
            const bool hasEmb = i < embeddings.size() &&
                                !embeddings[i].empty() && !track.embedding.empty();
            if (hasEmb) {
                const float appDist = 1.0f - cosineSim(track.embedding, embeddings[i]);
                cost = spatialW * sNorm + appearanceWeight * appDist;
            } else {
                cost = sNorm;                            // spatial-only fallback
            }
            if (cost < bestCost) { bestCost = cost; bestIdx = static_cast<int>(i); }
        }

        if (bestIdx >= 0) {
            track.centroid     = centroids[bestIdx];
            track.missedFrames = 0;
            if (bestIdx < (int)embeddings.size())
                emaUpdate(track.embedding, embeddings[bestIdx]);
            matched.insert(bestIdx);
            assignments[bestIdx] = track.id;
        }
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        if (matched.count(i)) continue;
        const int newId = m_nextId++;
        TrackedObject t{newId, centroids[i], 0, {}};
        if (i < embeddings.size()) t.embedding = embeddings[i];
        m_tracks[newId] = std::move(t);
        assignments[static_cast<int>(i)] = newId;
    }

    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->second.missedFrames > m_maxMissed) it = m_tracks.erase(it);
        else                                       ++it;
    }

    return assignments;
}
