# adas-multistream

Multi-camera ADAS **open-loop replay** demo on top of the
[visionpipe-npu](../visionpipe-npu) runtime.

Four driving videos play the role of four vehicle cameras
(front / rear / left / right). All streams share **one detector instance**
— the "one NPU" — while each stream keeps its **own memory arenas**, so
the demo exercises exactly the questions a multi-camera SoC platform team
lives with:

- Who gets the NPU next? (scheduling — round-robin)
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
| `--npus N` | number of simulated NPUs (1 detector instance + 1 worker thread each) | 1 |

### Multi-NPU simulation (`--npus N`)

One NPU = one detector instance (its own weight copy, modeling the NPU's
private SRAM) + one worker thread (independent compute). Streams are **not**
pinned to an NPU: they all sit in one shared scheduler and each NPU pulls the
next ready stream, runs one frame, and returns it to the back of the queue
(`StreamScheduler` in `src/multi_stream.cpp`). A stream is the unit of work —
never a single frame — because each stream is stateful (decoder position,
temporal tracks) and must never be touched by two NPUs at once. A faster NPU
simply drains more frames, so the load balances itself with no hand-tuned
placement. The report's per-NPU `frames` / `util%` columns show the result.

Measured (same 4 videos, trained weights, temporal 2, no snapshots):

| config | aggregate fps | scaling | per-NPU balance |
|---|---|---|---|
| `--npus 1` | 318 | 1.00× | baseline |
| `--npus 2` | **614** | 1.93× | frames 1446/1434, util 100%/99% — even |
| `--npus 4` | 1042 | 3.28× | frames 615…823, capped by 4-stream concurrency |

For comparison, the earlier static `i mod N` placement gave 534 fps at
`--npus 2` (imbalanced: one NPU got both slow-decode streams) and only
reached 566 after the videos were *hand-reordered* into slow+fast pairs.
Work-stealing hits 614 with no reordering — the placement problem disappears.

Two lessons made measurable:

1. **Work-stealing beats static placement for free** — under static `i mod N`,
   wall time was gated by the slowest NPU and the operator had to reorder
   streams to balance it (the stream-to-eFPGA SoC assignment problem). Pulling
   from a shared queue removes that knob: `--npus 2` jumps 566 → 614 and the
   two NPUs land at 100% / 99% utilization.
2. **Imperfect scaling is shared-resource contention** — per-stream decode
   latency still rises slightly with more threads (3.28 → 3.48 ms) because all
   "NPUs" share one memory subsystem, exactly like chiplets sharing an
   interconnect. At `--npus 4` there are only 4 streams for 4 NPUs, so a fast
   NPU that finishes has nothing to steal until a stream is returned — the
   util spread (69–100%) is this concurrency ceiling, not placement.

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

4 streams × 1080p30 × 720 frames each (2,880 frames total), single thread,
round-robin scheduling, shared TinyDetector:

```
══════════════════════════════════════════════════════════════════════════
 adas-multistream report — 4 streams, 8.9 s wall, 2880 frames total
 aggregate throughput: 325.0 fps (all streams combined)
══════════════════════════════════════════════════════════════════════════
 stream          frames       fps │ decode ms (avg/p95/p99)   │ infer ms (avg/p95/p99)
────────────────────────────────────────────────────────────────────────────
 drive_front        720     81.2  │    3.33 /   4.57 /   5.25 │    0.36 /   0.35 /   0.42
 drive_rear         720     81.2  │    2.18 /   2.72 /   3.26 │    0.32 /   0.34 /   0.40
 drive_left         720     81.2  │    2.66 /   3.80 /   4.44 │    0.32 /   0.35 /   0.41
 drive_right        720     81.2  │    2.78 /   4.29 /   4.91 │    0.32 /   0.35 /   0.40
══════════════════════════════════════════════════════════════════════════
```

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
- Scheduling is dynamic work-stealing: N NPU threads pull streams from a
  shared `StreamScheduler` (see `--npus N` above). Planned extensions:
  priority scheduling (front camera first), frame dropping under
  overload, and routing each NPU's frames through one `CommandQueue`.
