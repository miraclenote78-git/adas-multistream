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
| `--threshold T` | detection score threshold | 0.5 |
| `--snapshot N` | write annotated PPM per stream at frame N (-1 = off) | 60 |

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

- The detector is `TinyDetector` with deterministic synthesized weights —
  the boxes are not semantically meaningful. The subject of this demo is
  the **runtime behavior under multi-stream load**, not detection quality.
  Real weights load via `WeightFile` without touching this code.
- Scheduling is round-robin in a single thread. Planned extensions:
  priority scheduling (front camera first), frame dropping under
  overload, and a worker-thread pool feeding one `CommandQueue`.
