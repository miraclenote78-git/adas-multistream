# adas-multistream

Multi-camera ADAS **open-loop replay** demo on top of the
[visionpipe-npu](../visionpipe-npu) runtime.

Four driving videos play the role of four vehicle cameras
(front / rear / left / right). All streams share **one detector instance**
— the "one NPU" — while each stream keeps its **own memory arenas**, so
the demo exercises exactly the questions a multi-camera SoC platform team
lives with:

- Who gets the NPU next? (scheduling — decode/infer work-stealing pipeline)
- How is memory partitioned per stream? (per-stream `ArenaAllocator`)
- Where does the time go? (decode vs inference, mean and tail latency)

```
drive_front.mp4 ─► VideoFileSource ─► preprocess ─┐
drive_rear.mp4  ─► VideoFileSource ─► preprocess ─┤   TinyDetector
drive_left.mp4  ─► VideoFileSource ─► preprocess ─┼─► (shared, 1 instance)
drive_right.mp4 ─► VideoFileSource ─► preprocess ─┘        │
                                            decode → NMS → per-stream stats
```

This mirrors how the automotive industry validates perception stacks:
recorded camera footage is replayed against the target compute platform
(open-loop replay) before any closed-loop simulation or road testing.

## Build

Requires the `visionpipe-npu` source tree as a sibling directory (or pass
`-DVISIONPIPE_DIR=...`) and its dependencies (ffmpeg dev libs).

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
TD=../visionpipe-npu/test_data
./build/multi_stream output \
    $TD/drive_front.mp4 $TD/drive_rear.mp4 \
    $TD/drive_left.mp4  $TD/drive_right.mp4
```

Options:

| flag | meaning | default |
|---|---|---|
| `--max-frames N` | stop each stream after N frames | all |
| `--threshold T` | base detection score threshold | 0.5 |
| `--cls name=T` | per-class threshold override, repeatable (e.g. `--cls person=0.35`) | base |
| `--snapshot N` | write annotated PPM per stream at frame N (-1 = off) | 60 |
| `--weights F.vpnw` | load trained weights | synthesized (untrained) |
| `--temporal N` | confirm detections only after N consecutive matched frames (0 = off) | 0 |
| `--npus N` | number of NPU (inference) threads, each its own detector instance | auto: one per stream (capped at cores) |
| `--decoders N` | number of decode threads feeding the NPUs | = `--npus` |

### Decode/infer pipeline (`--npus N`, `--decoders M`)

One NPU = one detector instance (its own weight copy, modeling the NPU's
private SRAM) + one inference thread. **Decode and inference run in separate
thread pools**: `M` decode threads (slow H.264 software decode + preprocess,
~3.3 ms/frame) feed `N` NPU threads (inference, ~0.33 ms) through a bounded
per-stream buffer, so the two stages overlap instead of running back-to-back
on one thread (`Pipeline` in `src/multi_stream.cpp`).

Both pools steal work independently. Within a stage a stream is single-flight
(its stateful decoder / temporal tracks are never touched concurrently and
frames stay in order); *across* stages the same stream may decode frame N+1
while the NPU infers frame N — that overlap is the point. Streams are still
the unit of work, never raw frames. The report's per-pool `frames` / `util%`
columns show where the time goes.

Measured (same 4 videos, trained weights, temporal 2, no snapshots):

| config | aggregate fps | NPU util | what it shows |
|---|---|---|---|
| `--npus 1 --decoders 1` | 347 | 11% | decode saturates (100%), the NPU sits idle |
| `--npus 1 --decoders 4` | **1035** | 69% | **+3× from decode threads alone — one NPU is enough** |
| `--npus 4 --decoders 4` | 1039 | 19%×4 | four NPUs barely used; throughput is decode-bound |

The headline: **`--npus 1 --decoders 4` (1035 fps) ≈ `--npus 4 --decoders 4`
(1039 fps)** — one NPU fed by four decoders matches four NPUs, because decode,
not inference, was the wall. At `--npus 1 --decoders 1` the NPU runs at 11%
util (decode is ~9× the work: 8.3 s vs 0.9 s over the run), exactly the 10×
decode-dominates ratio the single-thread report predicted.

Two lessons made measurable:

1. **Pipelining alone doesn't beat the bottleneck — parallel decode does.**
   Overlapping the 0.33 ms infer behind the 3.3 ms decode only hides the small
   stage; the decode wall stays. The win comes from scaling the decode pool
   independently of the NPU count, which the split makes possible. On a real
   SoC the ultimate fix is a hardware video decoder block in front of the NPU.
2. **Decode parallelism is capped by stream count, not thread count.** With 4
   streams a stream is single-flight per stage, so at most 4 decode threads do
   useful work (extra decoders idle); the decoder util spread (67–100%) is this
   concurrency ceiling plus per-stream content asymmetry, not a scheduling bug.
   All threads also share one memory subsystem, so per-stream decode latency
   rises slightly under load — exactly like chiplets sharing an interconnect.

Per-class thresholds are an *application policy*: the SDK decodes at the
minimum threshold (mechanism), then each detection must clear its own
class's bar before NMS. Useful when classes have asymmetric confidence
distributions — e.g. `--threshold 0.55 --cls person=0.35` keeps night-time
"car" false positives out while letting weaker-but-real person
detections through.

### Temporal confirmation (`--temporal N`)

A threshold is a 1-D knob and cannot separate overlapping score
distributions (a night-time light-blob "car 0.63" vs a distant real
"car 0.58"). The time axis can: real objects persist, flickers don't.
`--temporal N` runs a tiny greedy-IoU tracker per stream
(`src/temporal_filter.hpp`) and only reports detections matched in N
consecutive frames (track survives 2 missed frames).

Measured (threshold 0.55, person 0.35, motorcycle/bicycle 0.4):

| setting | night FP (rear, frame 300) | city scooter (right) | det/frame raw→conf |
|---|---|---|---|
| off | `car 0.63` visible | kept | 8.3 → 8.3 |
| `--temporal 2` ⭐ | **gone** | kept | 8.3 → 7.6 |
| `--temporal 3` | gone | **lost** (fast motion breaks IoU match at 8×8 grid) | 8.3 → 7.0 |

Trade-offs: confirmation adds (N-1) frames of detection latency
(~33 ms/frame @ 30 fps), and large near-field objects move too fast for
plain IoU association at this grid resolution — the textbook next step
is motion-compensated matching (constant-velocity prediction, i.e. a
minimal Kalman filter).

### Real detection

Train the detector first (in the SDK tree — YOLOv8n pseudo-labels the
test videos, the PyTorch twin of TinyDetector learns from them):

```bash
cd ../visionpipe-npu
python3 tools/train_tiny_detector.py training_output \
    test_data/drive_front.mp4 test_data/drive_rear.mp4 \
    test_data/drive_left.mp4  test_data/drive_right.mp4 \
    test_data/vtest.avi
cd ../adas-multistream

./build/multi_stream output_trained \
    $TD/drive_front.mp4 $TD/drive_rear.mp4 \
    $TD/drive_left.mp4  $TD/drive_right.mp4 \
    --weights ../visionpipe-npu/training_output/tiny_detector_trained.vpnw \
    --snapshot 300
```

With trained weights, detections follow scene content
(same throughput — weights don't change the compute):

| stream | scene | det/frame |
|---|---|---|
| drive_right | busy city street | **12.4** |
| drive_rear | night highway | 1.0 |
| drive_front | country road | 0.4 |
| drive_left | empty coastal road | **0.0** |

The final report shows, per stream: frames processed, effective fps,
decode latency (avg/p95/p99), inference latency (avg/p95/p99) and average
detections per frame — plus the aggregate throughput across all streams.

## Test videos

`drive_*.mp4` live in `visionpipe-npu/test_data/` (Pexels License,
re-encoded to a uniform 1080p / 30 fps / 24 s / 720 frames so the
cross-stream comparison is fair). See that directory's README for sources.

## Measured results (Apple Silicon Mac mini, Release build)

4 streams × 1080p30 × 720 frames each (2,880 frames total), the
`--npus 1 --decoders 1` baseline (one decode thread ⇄ one NPU thread; the
default auto-picks one NPU per stream — 4 here — shown in the pipeline
section above):

```
══════════════════════════════════════════════════════════════════════════
 adas-multistream report — 4 streams, 1 decoders, 1 NPUs, 8.4 s wall, 2880 frames total
 aggregate throughput: 343.6 fps (all streams combined)
══════════════════════════════════════════════════════════════════════════
 stream          frames       fps │ decode ms (avg/p95/p99)   │ infer ms (avg/p95/p99)    │ det/frame raw→conf
────────────────────────────────────────────────────────────────────────────
 drive_front        720     85.9  │    3.38 /   4.72 /   5.44 │    0.32 /   0.36 /   0.49 │   0.0 →   0.0
 drive_rear         720     85.9  │    2.22 /   2.82 /   3.58 │    0.33 /   0.38 /   0.64 │   0.0 →   0.0
 drive_left         720     85.9  │    3.27 /   4.82 /   5.47 │    0.32 /   0.36 /   0.49 │   0.0 →   0.0
 drive_right        720     85.9  │    2.76 /   4.31 /   4.70 │    0.31 /   0.32 /   0.35 │   0.0 →   0.0
──────────────────────────────────────────────────────────────────────────
 dec       frames      busy ms    util%
 0           2880       8375.9    99.9
──────────────────────────────────────────────────────────────────────────
 npu       frames      busy ms    util%
 0           2880        930.8    11.1
══════════════════════════════════════════════════════════════════════════
```

The `util%` columns already tell the story: even with one decoder and one
NPU, the decode thread is pinned at 100% while the NPU sits at 11% — decode
is ~9× the work. Adding decode threads (`--decoders 4`) is what scales
throughput; see the pipeline section above.

Key observations:

1. **Real-time headroom**: every stream sustains ~81 fps — 2.7× above the
   30 fps capture rate. Four 1080p cameras fit comfortably on one node.
2. **Decode dominates**: H.264 1080p software decode costs ~2.3–3.3 ms/frame
   while TinyDetector inference costs ~0.33 ms. On a real SoC this is the
   argument for a hardware video decoder block in front of the NPU.
3. **Content asymmetry shows up in decode**: the visually busy daytime
   front camera decodes slowest (3.33 ms avg), the dark rear camera
   fastest (2.18 ms) — bitrate follows scene complexity, so per-stream
   tail latency differs even with identical resolutions.

## Notes

- Without `--weights`, `TinyDetector` runs deterministic synthesized
  weights — boxes are not meaningful, but the runtime load is identical.
  With a trained `.vpnw` (see above) the same binary detects real
  vehicles/people.
- Scheduling is a two-stage work-stealing pipeline: `--decoders M` decode
  threads and `--npus N` NPU threads steal streams from a shared `Pipeline`
  (see above). Planned extensions: priority scheduling (front camera first),
  frame dropping under overload, and routing each NPU's frames through one
  `CommandQueue`.
