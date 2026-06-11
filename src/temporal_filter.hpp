// adas-multistream — temporal confirmation filter.
//
// A single frame cannot tell a real object from a flicker: night-time
// light blobs score like cars for one or two frames and vanish. Real
// objects persist. This filter only *confirms* a detection after it has
// been matched in `min_hits` frames, and keeps a confirmed track alive
// through up to `max_misses` dropped frames (occlusion, threshold
// jitter).
//
// This is the standard first line of defense in ADAS perception stacks
// (usually called track confirmation / M-of-N gating). The cost is
// latency: a new real object becomes visible only after min_hits frames
// (min_hits=3 @ 30 fps ≈ 67 ms).
//
// Matching is greedy IoU within the same class, in the detector's input
// pixel space (64x64) — cheap enough to be negligible next to decode.

#pragma once

#include "visionpipe/model/detection.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace adas {

class TemporalFilter {
public:
    TemporalFilter(int min_hits, int max_misses, float iou_match)
        : min_hits_(min_hits), max_misses_(max_misses), iou_match_(iou_match) {}

    // Feed one frame's (post-NMS) detections; returns only the confirmed
    // ones. Pass-through when min_hits <= 1.
    std::vector<visionpipe::model::Detection>
    update(std::vector<visionpipe::model::Detection> dets) {
        if (min_hits_ <= 1) return dets;

        std::vector<bool> det_used(dets.size(), false);
        std::vector<bool> track_hit(tracks_.size(), false);

        // Greedy: strongest detection picks its best free track first.
        std::vector<std::size_t> order(dets.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) {
                      return dets[a].score > dets[b].score;
                  });

        for (std::size_t oi : order) {
            float best_iou = iou_match_;
            std::size_t best_t = tracks_.size();
            for (std::size_t t = 0; t < tracks_.size(); ++t) {
                if (track_hit[t]) continue;
                if (tracks_[t].det.cls_id != dets[oi].cls_id) continue;
                const float v = iou(tracks_[t].det, dets[oi]);
                if (v >= best_iou) { best_iou = v; best_t = t; }
            }
            if (best_t < tracks_.size()) {
                tracks_[best_t].det = dets[oi];   // follow the object
                tracks_[best_t].hits += 1;
                tracks_[best_t].misses = 0;
                track_hit[best_t] = true;
                det_used[oi] = true;
            }
        }

        // Unmatched detections found a new candidate track.
        for (std::size_t i = 0; i < dets.size(); ++i) {
            if (!det_used[i]) tracks_.push_back(Track{dets[i], 1, 0});
        }

        // Unmatched tracks age; stale ones die.
        for (std::size_t t = 0; t < track_hit.size(); ++t) {
            if (!track_hit[t]) tracks_[t].misses += 1;
        }
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                          [&](const Track& tr) {
                              return tr.misses > max_misses_;
                          }),
                      tracks_.end());

        std::vector<visionpipe::model::Detection> confirmed;
        for (const auto& tr : tracks_) {
            if (tr.hits >= min_hits_ && tr.misses == 0) confirmed.push_back(tr.det);
        }
        return confirmed;
    }

private:
    struct Track {
        visionpipe::model::Detection det;
        int hits{1};
        int misses{0};
    };

    static float iou(const visionpipe::model::Detection& a,
                     const visionpipe::model::Detection& b) noexcept {
        const float ax1 = a.cx - a.w * 0.5f, ay1 = a.cy - a.h * 0.5f;
        const float ax2 = a.cx + a.w * 0.5f, ay2 = a.cy + a.h * 0.5f;
        const float bx1 = b.cx - b.w * 0.5f, by1 = b.cy - b.h * 0.5f;
        const float bx2 = b.cx + b.w * 0.5f, by2 = b.cy + b.h * 0.5f;
        const float iw = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
        const float ih = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
        const float inter = iw * ih;
        const float uni = a.w * a.h + b.w * b.h - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    }

    int   min_hits_;
    int   max_misses_;
    float iou_match_;
    std::vector<Track> tracks_;
};

}  // namespace adas
