// adas-multistream — per-stream measurement.
//
// Records per-frame latencies for one camera stream and reduces them to
// the numbers an SoC platform team actually asks for: throughput, mean,
// and tail latency (p50/p95/p99).

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace adas {

struct LatencySummary {
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
    double max_ms{0.0};
};

class StreamStats {
public:
    explicit StreamStats(std::string name) : name_(std::move(name)) {}

    void record_frame(double decode_ms, double infer_ms,
                      std::size_t raw_detections, std::size_t confirmed_detections) {
        decode_ms_.push_back(decode_ms);
        infer_ms_.push_back(infer_ms);
        total_raw_ += raw_detections;
        total_confirmed_ += confirmed_detections;
    }

    const std::string& name() const noexcept { return name_; }
    std::size_t frames() const noexcept { return infer_ms_.size(); }

    double avg_raw() const noexcept {
        return frames() == 0 ? 0.0
            : static_cast<double>(total_raw_) / static_cast<double>(frames());
    }
    double avg_confirmed() const noexcept {
        return frames() == 0 ? 0.0
            : static_cast<double>(total_confirmed_) / static_cast<double>(frames());
    }

    LatencySummary decode_summary() const { return summarize(decode_ms_); }
    LatencySummary infer_summary()  const { return summarize(infer_ms_); }

private:
    static LatencySummary summarize(const std::vector<double>& samples) {
        LatencySummary s;
        if (samples.empty()) return s;
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0.0;
        for (double v : sorted) sum += v;
        s.mean_ms = sum / static_cast<double>(sorted.size());
        s.p50_ms  = percentile(sorted, 0.50);
        s.p95_ms  = percentile(sorted, 0.95);
        s.p99_ms  = percentile(sorted, 0.99);
        s.max_ms  = sorted.back();
        return s;
    }

    // Nearest-rank percentile on a pre-sorted vector.
    static double percentile(const std::vector<double>& sorted, double p) {
        const auto n = sorted.size();
        auto rank = static_cast<std::size_t>(p * static_cast<double>(n - 1) + 0.5);
        if (rank >= n) rank = n - 1;
        return sorted[rank];
    }

    std::string name_;
    std::vector<double> decode_ms_;
    std::vector<double> infer_ms_;
    std::size_t total_raw_{0};
    std::size_t total_confirmed_{0};
};

}  // namespace adas
