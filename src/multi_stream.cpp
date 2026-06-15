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
// Scheduling: a two-stage work-stealing pipeline. Decode (slow, ~3.3 ms of
// H.264 software decode) and inference (fast, ~0.33 ms on the NPU) run in
// separate thread pools — `--decoders M` decode threads feed `--npus N` NPU
// threads through a bounded per-stream buffer — so the two stages overlap
// instead of running back-to-back on one thread. Each stage steals work
// independently; within a stage a stream is single-flight, so its stateful
// decoder and temporal filter stay safe and frames stay in order. Per-stream
// decode and inference latency are recorded separately so the report shows
// where time actually goes.
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
#include <optional>
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

// One decoded + preprocessed frame, handed from a decode thread to an NPU
// thread. The frame data is *copied out* of the source's FrameView (which is
// non-owning and only valid until the next decode) so the decoder can race
// ahead while the NPU is still inferring an earlier frame. Preprocessing
// (resize → NCHW fp32) runs in the decode stage, so only the small input
// tensor crosses threads — never the full 1080p frame, except on the rare
// snapshot frame which also carries an original-resolution copy to annotate.
struct PipeFrame {
    std::vector<float> input;            // NCHW fp32, ready for forward()
    std::int64_t       frame_idx{0};     // = frames_decoded at decode time
    double             decode_ms{0.0};   // measured in the decode stage

    bool                      want_snapshot{false};
    std::vector<std::uint8_t> full_rgb;  // original frame, snapshot only
    int                       full_w{0}, full_h{0}, full_stride{0};
};

// Free-list of input buffers shared by all threads. Every PipeFrame.input is
// the same size (3·kInH·kInW floats), so a decoder borrows one instead of
// heap-allocating per frame and the NPU returns it after inference. Same idea
// as a framebuffer cache: pool the per-frame scratch, pay the malloc
// once. The free list self-sizes to the number of frames in flight, then
// stops growing.
class BufferPool {
public:
    std::vector<float> acquire(std::size_t n) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!free_.empty()) {
                std::vector<float> buf = std::move(free_.back());
                free_.pop_back();
                buf.resize(n);     // no realloc once capacity is reached
                return buf;
            }
        }
        return std::vector<float>(n);
    }
    void release(std::vector<float>&& buf) {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push_back(std::move(buf));
    }

private:
    std::mutex                      mu_;
    std::vector<std::vector<float>> free_;
};

// Everything one camera stream owns. The arenas are deliberately
// per-stream: on a real SoC this is the partitioning that stops one
// camera's traffic from fragmenting another's memory.
//
// A stream flows decode → buffer → infer. The decode side touches `source`
// and `small_rgb`; the infer side touches `scratch`, `output`, `filter`,
// `stats`. Those sets are disjoint, so a decode thread and an NPU thread may
// work the *same* stream at once (frame N+1 decoding while frame N infers).
// The only shared field is `buffer`, owned by the Pipeline's lock.
struct StreamCtx {
    StreamCtx(const std::string& path, std::size_t scratch_bytes,
              std::int64_t head_ch, int temporal_hits)
        : source(path, vision::VideoFileOptions{vision::PixelFormat::kRgb24, /*loop=*/false}),
          io_arena(1 * 1024 * 1024),
          scratch(scratch_bytes),
          stats(fs::path(path).stem().string()),
          filter(temporal_hits, /*max_misses=*/2, /*iou_match=*/0.3f),
          small_rgb(static_cast<std::size_t>(kInW * kInH * 3)) {
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
    std::vector<std::uint8_t>  small_rgb;     // decode-stage scratch
    runtime::Tensor            output{};
    std::int64_t               frames_decoded{0};   // decode side only
    std::int64_t               frames_done{0};      // infer side only

    // --- Pipeline scheduling state, guarded by Pipeline::mu_ ---
    std::deque<PipeFrame> buffer;        // decoded, waiting for an NPU
    bool decoding{false};                // a decode thread holds it now
    bool inferring{false};               // an NPU thread holds it now
    bool decode_done{false};             // EOF or frame cap reached
    bool retired{false};                 // fully drained, counted out
};

// Per-thread accounting (decode pool and NPU pool alike). Each slot is
// written by exactly one thread and read back only after join — no locking.
struct WorkerStat {
    std::int64_t frames  = 0;     // frames this thread handled
    double       busy_ms = 0.0;   // wall time it spent doing real work
};

// Two-stage work-stealing pipeline. Decode (slow, CPU/HW-decoder-modelled)
// and inference (fast, NPU) run in separate thread pools so they overlap;
// a bounded per-stream buffer lets the decoder run ahead without unbounded
// memory or losing frame order. Each stage independently steals work:
//
//   decode-eligible: not finished, not already decoding, buffer has room
//   infer-eligible : buffer non-empty, not already inferring
//
// Per stage a stream is single-flight (the flags), so the stateful decoder
// and temporal filter are never touched concurrently and frames stay in
// order; across stages the same stream may decode and infer at once. The
// shared mutex also publishes one thread's writes before another observes
// them. Streams are few, so eligibility is found by a short linear scan
// rather than maintaining separate ready-queues.
class Pipeline {
public:
    Pipeline(std::vector<StreamCtx*> streams, std::size_t buf_cap,
             std::int64_t max_frames)
        : streams_(std::move(streams)), cap_(buf_cap),
          max_frames_(max_frames), active_(streams_.size()) {}

    // Claim a stream to decode one frame into, or nullptr when all done.
    StreamCtx* acquire_decode() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            for (auto* s : streams_) {
                if (decode_eligible(*s)) { s->decoding = true; return s; }
            }
            if (active_ == 0) return nullptr;
            cv_.wait(lk);
        }
    }

    // Hand back a decoded frame (nullopt = the decode just hit EOF).
    void complete_decode(StreamCtx* s, std::optional<PipeFrame> pf) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            s->decoding = false;
            if (pf) {
                s->buffer.push_back(std::move(*pf));
                ++s->frames_decoded;
                if (max_frames_ >= 0 && s->frames_decoded >= max_frames_)
                    s->decode_done = true;
            } else {
                s->decode_done = true;
            }
            maybe_retire(*s);
        }
        cv_.notify_all();
    }

    // Claim a stream's oldest decoded frame, or nullopt when all done.
    std::optional<std::pair<StreamCtx*, PipeFrame>> acquire_infer() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            for (auto* s : streams_) {
                if (infer_eligible(*s)) {
                    s->inferring = true;
                    PipeFrame pf = std::move(s->buffer.front());
                    s->buffer.pop_front();
                    lk.unlock();
                    cv_.notify_all();   // freed a buffer slot → decode may resume
                    return std::make_pair(s, std::move(pf));
                }
            }
            if (active_ == 0) return std::nullopt;
            cv_.wait(lk);
        }
    }

    void complete_infer(StreamCtx* s) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            s->inferring = false;
            ++s->frames_done;
            maybe_retire(*s);
        }
        cv_.notify_all();
    }

private:
    bool decode_eligible(const StreamCtx& s) const {
        return !s.decoding && !s.decode_done && s.buffer.size() < cap_;
    }
    bool infer_eligible(const StreamCtx& s) const {
        return !s.inferring && !s.buffer.empty();
    }
    void maybe_retire(StreamCtx& s) {
        if (!s.retired && s.decode_done && s.buffer.empty() &&
            !s.decoding && !s.inferring) {
            s.retired = true;
            --active_;
        }
    }

    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<StreamCtx*> streams_;
    std::size_t             cap_;
    std::int64_t            max_frames_;
    std::size_t             active_;
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

// One per-thread utilization block (used for the decode pool and the NPU
// pool). util% = time spent doing real work / wall time.
void print_pool(const char* label, const std::vector<WorkerStat>& pool,
                double wall_s) {
    std::printf("──────────────────────────────────────────────────────────────────────────\n");
    std::printf(" %-6s %9s %12s %8s\n", label, "frames", "busy ms", "util%");
    for (std::size_t n = 0; n < pool.size(); ++n) {
        const double util = wall_s > 0.0
            ? 100.0 * pool[n].busy_ms / (wall_s * 1000.0) : 0.0;
        std::printf(" %-6zu %9lld %12.1f %7.1f\n",
                    n, static_cast<long long>(pool[n].frames),
                    pool[n].busy_ms, util);
    }
}

void print_report(const std::vector<std::unique_ptr<StreamCtx>>& streams,
                  const std::vector<WorkerStat>& decoders,
                  const std::vector<WorkerStat>& npus,
                  double wall_s) {
    std::int64_t total_frames = 0;
    for (const auto& s : streams) total_frames += s->frames_done;

    std::printf("\n");
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf(" adas-multistream report — %zu streams, %zu decoders, %zu NPUs, "
                "%.1f s wall, %lld frames total\n",
                streams.size(), decoders.size(), npus.size(), wall_s,
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

    // Per-pool work-stealing balance. The decode pool is the bottleneck
    // (each decoder ~3.3 ms/frame); the NPU pool is cheap (~0.33 ms) and
    // should sit at low util unless decoders outnumber NPUs — that gap is
    // the whole point of splitting the stages.
    print_pool("dec", decoders, wall_s);
    print_pool("npu", npus, wall_s);
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
    int          num_decoders   = -1;   // -1 → default to num_npus

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--max-frames" && i + 1 < argc)    max_frames = std::atoll(argv[++i]);
        else if (a == "--threshold" && i + 1 < argc) threshold = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--snapshot" && i + 1 < argc)  snapshot_frame = std::atoll(argv[++i]);
        else if (a == "--weights" && i + 1 < argc)   weights_path = argv[++i];
        else if (a == "--temporal" && i + 1 < argc)  temporal_hits = std::atoi(argv[++i]);
        else if (a == "--npus" && i + 1 < argc)      num_npus = std::atoi(argv[++i]);
        else if (a == "--decoders" && i + 1 < argc)  num_decoders = std::atoi(argv[++i]);
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

    if (num_decoders < 1) num_decoders = num_npus;   // default: one per NPU
    std::printf("npus: %d (1 detector instance + 1 worker thread each)\n", num_npus);
    std::printf("decoders: %d (separate decode pool feeding the NPUs)\n", num_decoders);
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
    std::printf("scheduling: decode/infer pipeline — %d decoder%s ⇄ %d NPU%s\n",
                num_decoders, num_decoders == 1 ? "" : "s",
                num_npus, num_npus == 1 ? "" : "s");

    // --- The pipeline: decode pool and NPU pool steal from one coordinator ---
    constexpr std::size_t kBufCap = 3;   // frames a decoder may run ahead per stream
    std::vector<StreamCtx*> all;
    all.reserve(streams.size());
    for (auto& sp : streams) all.push_back(sp.get());
    Pipeline pipe(all, kBufCap, max_frames);

    std::mutex io_mu;
    BufferPool input_pool;   // reused NCHW input buffers (no per-frame malloc)
    std::vector<WorkerStat> decoder_stats(static_cast<std::size_t>(num_decoders));
    std::vector<WorkerStat> npu_stats(static_cast<std::size_t>(num_npus));

    // Decode stage: decode one frame and preprocess it (resize → NCHW) so only
    // the small input tensor — never the full 1080p frame — crosses to the NPU.
    auto run_decoder = [&](int dec_id) {
        WorkerStat& ws = decoder_stats[static_cast<std::size_t>(dec_id)];
        for (;;) {
            StreamCtx* s = pipe.acquire_decode();
            if (!s) break;                       // all streams retired → exit

            const auto t0 = Clock::now();
            auto frame = s->source.next_frame();
            std::optional<PipeFrame> pf;
            if (frame) {
                PipeFrame f;
                f.frame_idx = s->frames_decoded;
                vision::resize_rgb24_nn(frame->data, frame->width, frame->height,
                                        frame->stride,
                                        s->small_rgb.data(), kInW, kInH, kInW * 3);
                f.input = input_pool.acquire(static_cast<std::size_t>(3 * kInH * kInW));
                runtime::Tensor in = runtime::make_tensor(
                    f.input.data(), {1, 3, kInH, kInW}, runtime::DType::kFloat32);
                vision::rgb24_to_nchw_fp32(s->small_rgb.data(), kInW, kInH,
                                           kInW * 3, in);
                if (f.frame_idx == snapshot_frame) {
                    f.want_snapshot = true;
                    f.full_w = frame->width; f.full_h = frame->height;
                    f.full_stride = frame->stride;
                    const auto bytes = static_cast<std::size_t>(frame->stride) *
                                       static_cast<std::size_t>(frame->height);
                    f.full_rgb.assign(frame->data, frame->data + bytes);
                }
                f.decode_ms = ms_since(t0);
                pf = std::move(f);
                ++ws.frames;
            }
            ws.busy_ms += ms_since(t0);
            pipe.complete_decode(s, std::move(pf));
        }
    };

    // Infer stage: run the NPU on one buffered frame, then post-process.
    auto run_npu = [&](int npu_id) {
        model::TinyDetector& npu_det = *npus[static_cast<std::size_t>(npu_id)];
        WorkerStat& ws = npu_stats[static_cast<std::size_t>(npu_id)];
        for (;;) {
            auto job = pipe.acquire_infer();
            if (!job) break;                     // all streams retired → exit
            StreamCtx& s = *job->first;
            PipeFrame& f = job->second;

            const auto t0 = Clock::now();
            runtime::Tensor in = runtime::make_tensor(
                f.input.data(), {1, 3, kInH, kInW}, runtime::DType::kFloat32);
            npu_det.forward(in, s.scratch, s.output);
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
            const double infer_ms = ms_since(t0);

            s.stats.record_frame(f.decode_ms, infer_ms, raw_count, dets.size());

            if (f.want_snapshot) {
                const float sx = static_cast<float>(f.full_w) / static_cast<float>(kInW);
                const float sy = static_cast<float>(f.full_h) / static_cast<float>(kInH);
                for (auto& d : dets) { d.cx *= sx; d.cy *= sy; d.w *= sx; d.h *= sy; }
                vision::draw_detections_labeled(f.full_rgb.data(),
                                                f.full_w, f.full_h, f.full_stride,
                                                dets, kClassNames,
                                                /*thickness=*/3, /*label_scale=*/3);
                draw_analysis_bar(f.full_rgb.data(), f.full_w, f.full_h, f.full_stride,
                                  s.stats.name(), f.frame_idx, dets, infer_ms);
                const std::string path = out_dir + "/" + s.stats.name() + "_snapshot.ppm";
                vision::write_ppm(path, f.full_rgb.data(),
                                  f.full_w, f.full_h, f.full_stride);
                std::lock_guard<std::mutex> lk(io_mu);
                std::printf("  [npu%d:%s] snapshot @ frame %lld → %s (%zu dets)\n",
                            npu_id, s.stats.name().c_str(),
                            static_cast<long long>(f.frame_idx),
                            path.c_str(), dets.size());
            }

            input_pool.release(std::move(f.input));   // back to the pool for reuse
            ws.busy_ms += ms_since(t0);
            ++ws.frames;
            pipe.complete_infer(&s);
        }
    };

    // --- Launch both pools; they overlap through the shared Pipeline ---
    const auto t_start = Clock::now();
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_decoders + num_npus));
    for (int n = 0; n < num_decoders; ++n) threads.emplace_back(run_decoder, n);
    for (int n = 0; n < num_npus; ++n)     threads.emplace_back(run_npu, n);
    for (auto& t : threads) t.join();
    const double wall_s = ms_since(t_start) / 1000.0;

    print_report(streams, decoder_stats, npu_stats, wall_s);
    return 0;
}
