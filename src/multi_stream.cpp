// adas-multistream — N virtual cameras, one shared detector.
//
// Open-loop replay: each video file plays the role of one vehicle camera
// (front / rear / left / right). All streams share a single TinyDetector
// instance — the "one NPU" — while every stream keeps its own memory
// arenas, so one stream's allocation behavior can never disturb another's.
//
//   drive_front.mp4 ─► VideoFileSource ─► preprocess ─┐
//   drive_rear.mp4  ─► VideoFileSource ─► preprocess ─┤  TinyDetector
//   drive_left.mp4  ─► VideoFileSource ─► preprocess ─┼─ (shared)
//   drive_right.mp4 ─► VideoFileSource ─► preprocess ─┘     │
//                                                  decode → NMS → stats
//
// Scheduling: dynamic work-stealing. Every stream is placed in one shared
// queue; each NPU repeatedly pulls the next ready stream, runs one frame,
// and returns it to the back of the queue. A stream is never bound to a
// fixed NPU, so a fast NPU simply processes more frames than a slow one and
// the load balances itself. Per-stream decode and inference latency are
// recorded separately so the report shows where time actually goes.
//
// Usage:
//   multi_stream <out_dir> <video1> [video2 ...]
//       [--max-frames N]     stop each stream after N frames (default: all)
//       [--threshold T]      base detection score threshold (default 0.5)
//       [--cls name=T]       per-class threshold override, repeatable
//                            (e.g. --cls person=0.35 --cls car=0.6);
//                            classes without an override use --threshold
//       [--snapshot N]       write one annotated PPM per stream at frame N
//                            (default 60; pass -1 to disable)
//       [--weights F.vpnw]   load trained weights (default: synthesized)
//       [--temporal N]       only report detections confirmed in N
//                            consecutive frames (0 = off, default).
//                            Flickering false positives die; persistent
//                            objects pay N-1 frames of latency.
//
// Per-class thresholds are applied here, in the application: the SDK's
// decode provides the mechanism (one floor threshold), the app owns the
// policy (which classes deserve a lower bar). Decoding happens at the
// minimum of all thresholds, then each detection is re-checked against
// its own class's threshold before NMS.

#include "stream_stats.hpp"
#include "temporal_filter.hpp"

#include "visionpipe/model/detection.hpp"
#include "visionpipe/model/tiny_detector.hpp"
#include "visionpipe/model/weight_file.hpp"
#include "visionpipe/runtime/allocator.hpp"
#include "visionpipe/runtime/tensor.hpp"
#include "visionpipe/vision/draw.hpp"
#include "visionpipe/vision/ppm.hpp"
#include "visionpipe/vision/preprocess.hpp"
#include "visionpipe/vision/video_file_source.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace visionpipe;
namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

constexpr int kInW = static_cast<int>(model::TinyDetector::kInW);
constexpr int kInH = static_cast<int>(model::TinyDetector::kInH);
constexpr int kGW  = static_cast<int>(model::TinyDetector::kGridW);
constexpr int kGH  = static_cast<int>(model::TinyDetector::kGridH);

// Must match the training-side CLASS_NAMES in
// visionpipe-npu/tools/train_tiny_detector.py.
const std::vector<std::string> kClassNames = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
};

// Everything one camera stream owns. The arenas are deliberately
// per-stream: on a real SoC this is the partitioning that stops one
// camera's traffic from fragmenting another's memory.
struct StreamCtx {
    StreamCtx(const std::string& path, std::size_t scratch_bytes,
              std::int64_t head_ch, int temporal_hits)
        : source(path, vision::VideoFileOptions{vision::PixelFormat::kRgb24, /*loop=*/false}),
          io_arena(1 * 1024 * 1024),
          scratch(scratch_bytes),
          stats(fs::path(path).stem().string()),
          filter(temporal_hits, /*max_misses=*/2, /*iou_match=*/0.3f),
          small_rgb(static_cast<std::size_t>(kInW * kInH * 3)) {
        auto* in_buf = io_arena.allocate(
            static_cast<std::size_t>(3 * kInH * kInW) * sizeof(float),
            alignof(std::max_align_t));
        input = runtime::make_tensor(in_buf, {1, 3, kInH, kInW}, runtime::DType::kFloat32);
        auto* out_buf = io_arena.allocate(
            static_cast<std::size_t>(head_ch * kGH * kGW) * sizeof(float),
            alignof(std::max_align_t));
        output = runtime::make_tensor(out_buf, {1, head_ch, kGH, kGW}, runtime::DType::kFloat32);
    }

    vision::VideoFileSource    source;
    runtime::ArenaAllocator    io_arena;
    runtime::ArenaAllocator    scratch;
    adas::StreamStats          stats;
    adas::TemporalFilter       filter;
    std::vector<std::uint8_t>  small_rgb;
    std::vector<std::uint8_t>  annotated;
    runtime::Tensor            input{};
    runtime::Tensor            output{};
    std::int64_t               frames_done{0};
};

// Per-NPU (worker) accounting. Each slot is written by exactly one worker
// thread and read back only after join, so no locking is needed.
struct WorkerStat {
    std::int64_t frames  = 0;     // frames this NPU actually processed
    double       busy_ms = 0.0;   // wall time it spent decoding + inferring
};

// Dynamic stream scheduler — the work-stealing replacement for static
// `stream i → NPU i % N` placement. The unit of work is a *stream*, not a
// frame: each stream is stateful (decoder position, temporal tracks) and must
// never be touched by two NPUs at once. Any idle NPU pulls the next ready
// stream, runs one frame, and returns it to the back of the queue; a stream
// that hits EOF is retired instead. A fast NPU therefore drains more frames
// than a slow one — the load balances itself with nothing pinned to one NPU.
// The internal mutex also publishes one worker's writes to a StreamCtx before
// the next worker that picks it up can observe them.
class StreamScheduler {
public:
    explicit StreamScheduler(const std::vector<StreamCtx*>& initial)
        : ready_(initial.begin(), initial.end()),
          remaining_(initial.size()) {}

    // Pop the next ready stream. Blocks while the queue is momentarily empty
    // but streams are still in flight; returns nullptr once every stream has
    // been retired — the signal for a worker to exit.
    StreamCtx* take() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !ready_.empty() || remaining_ == 0; });
        if (ready_.empty()) return nullptr;
        StreamCtx* s = ready_.front();
        ready_.pop_front();
        return s;
    }

    // Return a stream that still has frames left, for any worker to pick up.
    void put_back(StreamCtx* s) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            ready_.push_back(s);
        }
        cv_.notify_one();
    }

    // Retire a finished stream (EOF / frame cap). When the last one finishes,
    // wake every worker so blocked ones see remaining_ == 0 and exit.
    void retire() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            --remaining_;
        }
        cv_.notify_all();
    }

private:
    std::mutex              mu_;
    std::condition_variable cv_;
    std::deque<StreamCtx*>  ready_;
    std::size_t             remaining_;
};

// BLIP-caption-style bottom bar: black band + one line of scene summary,
// e.g. "drive right  frame 300  objects 6  car 3 motorcycle 2 person 1  infer 0.33 ms".
void draw_analysis_bar(std::uint8_t* buf, int width, int height, int stride,
                       const std::string& stream_name, std::int64_t frame_idx,
                       const std::vector<model::Detection>& dets,
                       double infer_ms) {
    std::vector<int> count(kClassNames.size(), 0);
    for (const auto& d : dets) {
        const auto k = static_cast<std::size_t>(d.cls_id);
        if (k < count.size()) ++count[k];
    }
    char text[256];
    int n = std::snprintf(text, sizeof(text), "%s  frame %lld  objects %zu  ",
                          stream_name.c_str(),
                          static_cast<long long>(frame_idx), dets.size());
    for (std::size_t k = 0; k < count.size(); ++k) {
        if (count[k] > 0 && n < static_cast<int>(sizeof(text))) {
            n += std::snprintf(text + n, sizeof(text) - static_cast<std::size_t>(n),
                               "%s %d  ", kClassNames[k].c_str(), count[k]);
        }
    }
    if (n < static_cast<int>(sizeof(text))) {
        std::snprintf(text + n, sizeof(text) - static_cast<std::size_t>(n),
                      " infer %.2f ms", infer_ms);
    }

    // Bar geometry scales with the frame so 1080p and 576p look alike.
    const int scale = std::max(2, height / 360);     // 1080p → 3
    const int text_h = 7 * scale;
    const int pad = 4 * scale;
    const int bar_h = text_h + 2 * pad;
    vision::draw_filled_rect(buf, width, height, stride,
                             0, height - bar_h, width - 1, height - 1,
                             0, 0, 0);
    vision::draw_label(buf, width, height, stride,
                       pad, height - bar_h + pad, text,
                       255, 255, 255, scale);
}

void print_report(const std::vector<std::unique_ptr<StreamCtx>>& streams,
                  const std::vector<WorkerStat>& workers,
                  double wall_s) {
    std::int64_t total_frames = 0;
    for (const auto& s : streams) total_frames += s->frames_done;

    std::printf("\n");
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf(" adas-multistream report — %zu streams, %zu NPUs, %.1f s wall, %lld frames total\n",
                streams.size(), workers.size(), wall_s,
                static_cast<long long>(total_frames));
    std::printf(" aggregate throughput: %.1f fps (all streams combined)\n",
                static_cast<double>(total_frames) / wall_s);
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf(" %-14s %7s %9s │ %-25s │ %-25s │ %s\n",
                "stream", "frames", "fps", "decode ms (avg/p95/p99)",
                "infer ms (avg/p95/p99)", "det/frame raw→conf");
    std::printf("────────────────────────────────────────────────────────────────────────────\n");
    for (const auto& s : streams) {
        const auto dec = s->stats.decode_summary();
        const auto inf = s->stats.infer_summary();
        const double fps = static_cast<double>(s->frames_done) / wall_s;
        std::printf(" %-14s %7lld %8.1f  │ %7.2f / %6.2f / %6.2f │ %7.2f / %6.2f / %6.2f │ %5.1f → %5.1f\n",
                    s->stats.name().c_str(),
                    static_cast<long long>(s->frames_done), fps,
                    dec.mean_ms, dec.p95_ms, dec.p99_ms,
                    inf.mean_ms, inf.p95_ms, inf.p99_ms,
                    s->stats.avg_raw(), s->stats.avg_confirmed());
    }

    // Per-NPU work-stealing balance. Under static placement these frame
    // counts were fixed by the stream→NPU map (and idle time on the fast NPU
    // was invisible); with work-stealing a faster NPU grabs more, so the
    // columns show how evenly the load actually spread and how busy each
    // NPU stayed (util% = busy / wall).
    if (workers.size() > 1) {
        std::printf("──────────────────────────────────────────────────────────────────────────\n");
        std::printf(" %-6s %9s %12s %8s\n", "npu", "frames", "busy ms", "util%");
        for (std::size_t n = 0; n < workers.size(); ++n) {
            const double util = wall_s > 0.0
                ? 100.0 * workers[n].busy_ms / (wall_s * 1000.0) : 0.0;
            std::printf(" %-6zu %9lld %12.1f %7.1f\n",
                        n, static_cast<long long>(workers[n].frames),
                        workers[n].busy_ms, util);
        }
    }
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <out_dir> <video1> [video2 ...] "
            "[--max-frames N] [--threshold T] [--snapshot N]\n", argv[0]);
        return 2;
    }

    const std::string out_dir = argv[1];
    std::vector<std::string> videos;
    std::vector<std::pair<std::string, float>> cls_overrides;
    std::string  weights_path;
    std::int64_t max_frames     = -1;
    float        threshold      = 0.5f;
    std::int64_t snapshot_frame = 60;
    int          temporal_hits  = 0;
    int          num_npus       = 1;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--max-frames" && i + 1 < argc)    max_frames = std::atoll(argv[++i]);
        else if (a == "--threshold" && i + 1 < argc) threshold = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--snapshot" && i + 1 < argc)  snapshot_frame = std::atoll(argv[++i]);
        else if (a == "--weights" && i + 1 < argc)   weights_path = argv[++i];
        else if (a == "--temporal" && i + 1 < argc)  temporal_hits = std::atoi(argv[++i]);
        else if (a == "--npus" && i + 1 < argc)      num_npus = std::atoi(argv[++i]);
        else if (a == "--cls" && i + 1 < argc) {
            const std::string kv = argv[++i];
            const auto eq = kv.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "--cls expects name=threshold, got '%s'\n", kv.c_str());
                return 2;
            }
            cls_overrides.emplace_back(kv.substr(0, eq),
                                       static_cast<float>(std::atof(kv.c_str() + eq + 1)));
        }
        else videos.push_back(a);
    }
    if (videos.empty()) {
        std::fprintf(stderr, "no input videos\n");
        return 2;
    }
    fs::create_directories(out_dir);

    // --- Simulated NPUs ---
    // One NPU = one detector instance (its own weight copy, modeling the
    // NPU's private SRAM) + one worker thread (independent compute).
    if (num_npus < 1) num_npus = 1;
    std::vector<std::unique_ptr<runtime::ArenaAllocator>> npu_weight_arenas;
    std::vector<std::unique_ptr<model::TinyDetector>> npus;
    {
        std::unique_ptr<model::WeightFile> wf;
        if (weights_path.empty()) {
            std::printf("weights: synthesized (untrained — boxes are not meaningful)\n");
        } else {
            std::printf("weights: %s\n", weights_path.c_str());
            wf = std::make_unique<model::WeightFile>(weights_path);
        }
        for (int n = 0; n < num_npus; ++n) {
            npu_weight_arenas.push_back(
                std::make_unique<runtime::ArenaAllocator>(1 * 1024 * 1024));
            npus.push_back(wf
                ? std::make_unique<model::TinyDetector>(*wf, *npu_weight_arenas.back(), 1)
                : std::make_unique<model::TinyDetector>(*npu_weight_arenas.back(), 1));
        }
    }
    model::TinyDetector& detector0 = *npus.front();
    const std::size_t scratch_bytes = detector0.recommended_scratch_bytes();

    std::printf("npus: %d (1 detector instance + 1 worker thread each)\n", num_npus);
    if (detector0.num_classes() > 0) {
        std::printf("classes: %lld\n",
                    static_cast<long long>(detector0.num_classes()));
    }

    // Per-class thresholds: every class starts at the base threshold,
    // --cls overrides take precedence. decode runs at the minimum so no
    // class is filtered before its own threshold gets a say.
    std::vector<float> cls_threshold(kClassNames.size(), threshold);
    for (const auto& [name, t] : cls_overrides) {
        const auto it = std::find(kClassNames.begin(), kClassNames.end(), name);
        if (it == kClassNames.end()) {
            std::fprintf(stderr, "--cls: unknown class '%s'\n", name.c_str());
            return 2;
        }
        cls_threshold[static_cast<std::size_t>(it - kClassNames.begin())] = t;
        std::printf("threshold: %s = %.2f (base %.2f)\n", name.c_str(),
                    static_cast<double>(t), static_cast<double>(threshold));
    }
    const float decode_threshold =
        *std::min_element(cls_threshold.begin(), cls_threshold.end());

    if (temporal_hits > 1) {
        std::printf("temporal: confirm after %d consecutive frames "
                    "(~%.0f ms latency @ 30 fps)\n",
                    temporal_hits, (temporal_hits - 1) * 1000.0 / 30.0);
    }

    // --- One context per virtual camera. Placement is no longer static:
    //     every stream goes into a shared scheduler and any NPU may run it. ---
    std::vector<std::unique_ptr<StreamCtx>> streams;
    for (std::size_t i = 0; i < videos.size(); ++i) {
        auto ctx = std::make_unique<StreamCtx>(videos[i], scratch_bytes,
                                               detector0.head_channels(),
                                               temporal_hits);
        const auto info = ctx->source.info();
        std::printf("stream %-14s %dx%d @ %.1f fps\n",
                    ctx->stats.name().c_str(), info.width, info.height, info.fps);
        streams.push_back(std::move(ctx));
    }
    std::printf("scheduling: dynamic work-stealing across %d NPU%s\n",
                num_npus, num_npus == 1 ? "" : "s");

    // --- Shared scheduler: every stream is eligible for any NPU ---
    std::vector<StreamCtx*> all;
    all.reserve(streams.size());
    for (auto& sp : streams) all.push_back(sp.get());
    StreamScheduler sched(all);

    std::mutex io_mu;
    std::vector<WorkerStat> worker_stats(static_cast<std::size_t>(num_npus));

    // Run exactly one frame of stream `s` on NPU `npu_id`'s detector.
    // Returns false when the stream has reached EOF or its frame cap, i.e.
    // the caller should retire it instead of returning it to the queue.
    auto process_one = [&](StreamCtx& s, model::TinyDetector& npu_det,
                           int npu_id) -> bool {
        if (max_frames >= 0 && s.frames_done >= max_frames) return false;

        const auto t_dec = Clock::now();
        auto frame = s.source.next_frame();
        if (!frame) return false;
        const double decode_ms = ms_since(t_dec);

        const auto t_inf = Clock::now();
        vision::resize_rgb24_nn(frame->data, frame->width, frame->height,
                                frame->stride,
                                s.small_rgb.data(), kInW, kInH, kInW * 3);
        vision::rgb24_to_nchw_fp32(s.small_rgb.data(), kInW, kInH, kInW * 3,
                                   s.input);
        npu_det.forward(s.input, s.scratch, s.output);
        auto dets = model::decode_tiny_detector_output(s.output, /*batch_index=*/0,
                                                       kInW, kInH, decode_threshold);
        // App policy: each class clears its own bar before NMS.
        dets.erase(std::remove_if(dets.begin(), dets.end(),
                       [&](const model::Detection& d) {
                           const auto k = static_cast<std::size_t>(d.cls_id);
                           return k < cls_threshold.size() &&
                                  d.score < cls_threshold[k];
                       }),
                   dets.end());
        dets = model::nms(std::move(dets), /*iou_threshold=*/0.45f);
        const std::size_t raw_count = dets.size();
        dets = s.filter.update(std::move(dets));
        const double infer_ms = ms_since(t_inf);

        s.stats.record_frame(decode_ms, infer_ms, raw_count, dets.size());

        if (s.frames_done == snapshot_frame) {
            const float sx = static_cast<float>(frame->width)  / static_cast<float>(kInW);
            const float sy = static_cast<float>(frame->height) / static_cast<float>(kInH);
            for (auto& d : dets) { d.cx *= sx; d.cy *= sy; d.w *= sx; d.h *= sy; }
            const auto bytes = static_cast<std::size_t>(frame->stride) *
                               static_cast<std::size_t>(frame->height);
            s.annotated.assign(bytes, 0u);
            std::memcpy(s.annotated.data(), frame->data, bytes);
            vision::draw_detections_labeled(s.annotated.data(),
                                            frame->width, frame->height,
                                            frame->stride, dets, kClassNames,
                                            /*thickness=*/3, /*label_scale=*/3);
            draw_analysis_bar(s.annotated.data(),
                              frame->width, frame->height, frame->stride,
                              s.stats.name(), s.frames_done, dets, infer_ms);
            const std::string path = out_dir + "/" + s.stats.name() + "_snapshot.ppm";
            vision::write_ppm(path, s.annotated.data(),
                              frame->width, frame->height, frame->stride);
            std::lock_guard<std::mutex> lk(io_mu);
            std::printf("  [npu%d:%s] snapshot @ frame %lld → %s (%zu dets)\n",
                        npu_id, s.stats.name().c_str(),
                        static_cast<long long>(s.frames_done),
                        path.c_str(), dets.size());
        }

        ++s.frames_done;
        return true;
    };

    // --- NPU worker: pull any ready stream, run one frame, return it ---
    auto run_worker = [&](int npu_id) {
        model::TinyDetector& npu_det = *npus[static_cast<std::size_t>(npu_id)];
        WorkerStat& ws = worker_stats[static_cast<std::size_t>(npu_id)];
        for (;;) {
            StreamCtx* s = sched.take();
            if (!s) break;                       // all streams retired → exit
            const auto t0 = Clock::now();
            const bool more = process_one(*s, npu_det, npu_id);
            ws.busy_ms += ms_since(t0);
            if (more) {
                ++ws.frames;
                sched.put_back(s);
            } else {
                sched.retire();
            }
        }
    };

    // --- Launch one thread per NPU; all share the stream scheduler ---
    const auto t_start = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_npus));
    for (int n = 0; n < num_npus; ++n) workers.emplace_back(run_worker, n);
    for (auto& w : workers) w.join();
    const double wall_s = ms_since(t_start) / 1000.0;

    print_report(streams, worker_stats, wall_s);
    return 0;
}
