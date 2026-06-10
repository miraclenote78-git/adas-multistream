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
// Scheduling: round-robin. Every live stream gets exactly one frame per
// scheduling cycle; a stream that hits EOF drops out and the others keep
// going. Per-stream decode and inference latency are recorded separately
// so the report shows where time actually goes.
//
// Usage:
//   multi_stream <out_dir> <video1> [video2 ...]
//       [--max-frames N]     stop each stream after N frames (default: all)
//       [--threshold T]      detection score threshold (default 0.5)
//       [--snapshot N]       write one annotated PPM per stream at frame N
//                            (default 60; pass -1 to disable)

#include "stream_stats.hpp"

#include "visionpipe/model/detection.hpp"
#include "visionpipe/model/tiny_detector.hpp"
#include "visionpipe/runtime/allocator.hpp"
#include "visionpipe/runtime/tensor.hpp"
#include "visionpipe/vision/draw.hpp"
#include "visionpipe/vision/ppm.hpp"
#include "visionpipe/vision/preprocess.hpp"
#include "visionpipe/vision/video_file_source.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
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
constexpr int kCh  = static_cast<int>(model::TinyDetector::kHeadCh);

// Everything one camera stream owns. The arenas are deliberately
// per-stream: on a real SoC this is the partitioning that stops one
// camera's traffic from fragmenting another's memory.
struct StreamCtx {
    explicit StreamCtx(const std::string& path, std::size_t scratch_bytes)
        : source(path, vision::VideoFileOptions{vision::PixelFormat::kRgb24, /*loop=*/false}),
          io_arena(1 * 1024 * 1024),
          scratch(scratch_bytes),
          stats(fs::path(path).stem().string()),
          small_rgb(static_cast<std::size_t>(kInW * kInH * 3)) {
        auto* in_buf = io_arena.allocate(
            static_cast<std::size_t>(3 * kInH * kInW) * sizeof(float),
            alignof(std::max_align_t));
        input = runtime::make_tensor(in_buf, {1, 3, kInH, kInW}, runtime::DType::kFloat32);
        auto* out_buf = io_arena.allocate(
            static_cast<std::size_t>(kCh * kGH * kGW) * sizeof(float),
            alignof(std::max_align_t));
        output = runtime::make_tensor(out_buf, {1, kCh, kGH, kGW}, runtime::DType::kFloat32);
    }

    vision::VideoFileSource    source;
    runtime::ArenaAllocator    io_arena;
    runtime::ArenaAllocator    scratch;
    adas::StreamStats          stats;
    std::vector<std::uint8_t>  small_rgb;
    std::vector<std::uint8_t>  annotated;
    runtime::Tensor            input{};
    runtime::Tensor            output{};
    bool                       alive{true};
    std::int64_t               frames_done{0};
};

void print_report(const std::vector<std::unique_ptr<StreamCtx>>& streams,
                  double wall_s) {
    std::int64_t total_frames = 0;
    for (const auto& s : streams) total_frames += s->frames_done;

    std::printf("\n");
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf(" adas-multistream report — %zu streams, %.1f s wall, %lld frames total\n",
                streams.size(), wall_s, static_cast<long long>(total_frames));
    std::printf(" aggregate throughput: %.1f fps (all streams combined)\n",
                static_cast<double>(total_frames) / wall_s);
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf(" %-14s %7s %9s │ %-25s │ %-25s │ %s\n",
                "stream", "frames", "fps", "decode ms (avg/p95/p99)",
                "infer ms (avg/p95/p99)", "det/frame");
    std::printf("────────────────────────────────────────────────────────────────────────────\n");
    for (const auto& s : streams) {
        const auto dec = s->stats.decode_summary();
        const auto inf = s->stats.infer_summary();
        const double fps = static_cast<double>(s->frames_done) / wall_s;
        std::printf(" %-14s %7lld %8.1f  │ %7.2f / %6.2f / %6.2f │ %7.2f / %6.2f / %6.2f │ %6.1f\n",
                    s->stats.name().c_str(),
                    static_cast<long long>(s->frames_done), fps,
                    dec.mean_ms, dec.p95_ms, dec.p99_ms,
                    inf.mean_ms, inf.p95_ms, inf.p99_ms,
                    s->stats.avg_detections());
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
    std::int64_t max_frames     = -1;
    float        threshold      = 0.5f;
    std::int64_t snapshot_frame = 60;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--max-frames" && i + 1 < argc)    max_frames = std::atoll(argv[++i]);
        else if (a == "--threshold" && i + 1 < argc) threshold = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--snapshot" && i + 1 < argc)  snapshot_frame = std::atoll(argv[++i]);
        else videos.push_back(a);
    }
    if (videos.empty()) {
        std::fprintf(stderr, "no input videos\n");
        return 2;
    }
    fs::create_directories(out_dir);

    // --- The "one NPU": a single shared detector ---
    runtime::ArenaAllocator weights(1 * 1024 * 1024);
    model::TinyDetector detector(weights, /*batch=*/1);
    const std::size_t scratch_bytes = detector.recommended_scratch_bytes();

    // --- One context per virtual camera ---
    std::vector<std::unique_ptr<StreamCtx>> streams;
    for (const auto& v : videos) {
        auto ctx = std::make_unique<StreamCtx>(v, scratch_bytes);
        const auto info = ctx->source.info();
        std::printf("stream %-14s %dx%d @ %.1f fps\n",
                    ctx->stats.name().c_str(), info.width, info.height, info.fps);
        streams.push_back(std::move(ctx));
    }

    // --- Round-robin scheduling loop ---
    const auto t_start = Clock::now();
    std::size_t live = streams.size();
    while (live > 0) {
        for (auto& sp : streams) {
            StreamCtx& s = *sp;
            if (!s.alive) continue;
            if (max_frames >= 0 && s.frames_done >= max_frames) {
                s.alive = false; --live; continue;
            }

            const auto t_dec = Clock::now();
            auto frame = s.source.next_frame();
            if (!frame) { s.alive = false; --live; continue; }
            const double decode_ms = ms_since(t_dec);

            const auto t_inf = Clock::now();
            vision::resize_rgb24_nn(frame->data, frame->width, frame->height,
                                    frame->stride,
                                    s.small_rgb.data(), kInW, kInH, kInW * 3);
            vision::rgb24_to_nchw_fp32(s.small_rgb.data(), kInW, kInH, kInW * 3,
                                       s.input);
            detector.forward(s.input, s.scratch, s.output);
            auto dets = model::decode_tiny_detector_output(s.output, /*batch_index=*/0,
                                                           kInW, kInH, threshold);
            dets = model::nms(std::move(dets), /*iou_threshold=*/0.45f);
            const double infer_ms = ms_since(t_inf);

            s.stats.record_frame(decode_ms, infer_ms, dets.size());

            if (s.frames_done == snapshot_frame) {
                const float sx = static_cast<float>(frame->width)  / static_cast<float>(kInW);
                const float sy = static_cast<float>(frame->height) / static_cast<float>(kInH);
                for (auto& d : dets) { d.cx *= sx; d.cy *= sy; d.w *= sx; d.h *= sy; }
                const auto bytes = static_cast<std::size_t>(frame->stride) *
                                   static_cast<std::size_t>(frame->height);
                s.annotated.assign(bytes, 0u);
                std::memcpy(s.annotated.data(), frame->data, bytes);
                vision::draw_detections(s.annotated.data(),
                                        frame->width, frame->height, frame->stride,
                                        dets, /*thickness=*/3);
                const std::string path = out_dir + "/" + s.stats.name() + "_snapshot.ppm";
                vision::write_ppm(path, s.annotated.data(),
                                  frame->width, frame->height, frame->stride);
                std::printf("  [%s] snapshot @ frame %lld → %s (%zu dets)\n",
                            s.stats.name().c_str(),
                            static_cast<long long>(s.frames_done),
                            path.c_str(), dets.size());
            }

            ++s.frames_done;
        }
    }
    const double wall_s = ms_since(t_start) / 1000.0;

    print_report(streams, wall_s);
    return 0;
}
