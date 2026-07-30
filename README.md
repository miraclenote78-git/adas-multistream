# adas-multistream

[visionpipe-npu](../visionpipe-npu) 런타임 위에서 동작하는 멀티카메라 ADAS
**open-loop replay** 데모.

주행 영상 4개가 차량 카메라 4대(전방/후방/좌측/우측) 역할을 한다. 모든
스트림이 **검출기 인스턴스 1개**("NPU 1개")를 공유하되, 스트림마다 **자기만의
메모리 아레나**를 유지한다 — 멀티카메라 SoC 플랫폼 팀이 매일 마주하는
바로 그 질문들을 재현하기 위해서다:

- 다음 NPU는 누가 쓰나? (스케줄링 — decode/inference work-stealing 파이프라인)
- 스트림별 메모리는 어떻게 분할하나? (스트림별 `ArenaAllocator`)
- 시간은 어디서 새나? (decode vs inference, 평균·tail 레이턴시)

```
drive_front.mp4 ─► VideoFileSource ─► preprocess ─┐
drive_rear.mp4  ─► VideoFileSource ─► preprocess ─┤   TinyDetector
drive_left.mp4  ─► VideoFileSource ─► preprocess ─┼─► (공유, 인스턴스 1개)
drive_right.mp4 ─► VideoFileSource ─► preprocess ─┘        │
                                            decode → NMS → 스트림별 통계
```

이는 자동차 업계가 인지(perception) 스택을 검증하는 방식과 같다:
녹화된 카메라 영상을 타깃 컴퓨팅 플랫폼에서 재생하며 검증(open-loop
replay)한 뒤에야 closed-loop 시뮬레이션·실차 테스트로 넘어간다.

## 빌드

`visionpipe-npu` 소스 트리가 형제 디렉터리에 있어야 하며(또는
`-DVISIONPIPE_DIR=...`로 지정), 그 의존성(ffmpeg dev 라이브러리)이 필요하다.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 실행

```bash
TD=../visionpipe-npu/test_data
./build/multi_stream output \
    $TD/drive_front.mp4 $TD/drive_rear.mp4 \
    $TD/drive_left.mp4  $TD/drive_right.mp4
```

옵션:

| 플래그 | 의미 | 기본값 |
|---|---|---|
| `--max-frames N` | 각 스트림을 N프레임에서 중단 | 전체 |
| `--threshold T` | 기본 검출 점수 임계값 | 0.5 |
| `--cls name=T` | 클래스별 임계값 오버라이드, 반복 지정 가능 (예: `--cls person=0.35`) | 기본값 |
| `--snapshot N` | N번째 프레임에서 스트림별 주석 PPM 저장 (-1 = 끔) | 60 |
| `--weights F.vpnw` | 학습된 가중치 로드 | 합성 가중치(미학습) |
| `--temporal N` | N프레임 연속 매칭된 검출만 확정 (0 = 끔) | 0 |
| `--npus N` | NPU(inference) 스레드 수, 각자 검출기 인스턴스 보유 | 자동: 스트림당 1개 (코어 수 상한) |
| `--decoders N` | NPU에 프레임을 공급하는 decode 스레드 수 | = `--npus` |

### Decode/inference 파이프라인 (`--npus N`, `--decoders M`)

NPU 1개 = 검출기 인스턴스 1개(자기 가중치 사본 보유 — NPU의 전용 SRAM을
모델링) + inference 스레드 1개. **Decode와 inference는 별도 스레드 풀에서
돈다**: `M`개의 decode 스레드(느린 H.264 소프트웨어 디코드 + 전처리,
~3.3 ms/frame)가 스트림별 유한 버퍼를 통해 `N`개의 NPU 스레드(inference,
~0.33 ms)에 프레임을 공급한다. 두 단계가 한 스레드에서 순차 실행되는 대신
겹쳐서 돈다 (`src/multi_stream.cpp`의 `Pipeline`).

두 풀은 각자 독립적으로 work-stealing한다. 한 단계 안에서 스트림은
single-flight다(상태를 가진 디코더/temporal 트랙은 동시 접근되지 않고
프레임 순서가 유지됨). 반면 단계를 *가로질러서는* 같은 스트림이 프레임 N을
inference하는 동안 프레임 N+1을 decode할 수 있다 — 이 겹침이 핵심이다.
작업 단위는 여전히 스트림이지, 개별 프레임이 아니다. 리포트의 풀별
`frames` / `util%` 열이 시간이 어디에 쓰이는지 보여준다.

실측 (같은 영상 4개, 학습 가중치, temporal 2, 스냅샷 없음):

| 설정 | 합산 fps | NPU 가동률 | 의미 |
|---|---|---|---|
| `--npus 1 --decoders 1` | 347 | 11% | decode 포화(100%), NPU는 놀고 있음 |
| `--npus 1 --decoders 4` | **1035** | 69% | **decode 스레드만으로 +3배 — NPU는 1개면 충분** |
| `--npus 4 --decoders 4` | 1039 | 19%×4 | NPU 4개는 거의 안 쓰임; 처리량은 decode에 묶임 |

핵심: **`--npus 1 --decoders 4`(1035 fps) ≈ `--npus 4 --decoders 4`
(1039 fps)** — decoder 4개가 공급하는 NPU 1개가 NPU 4개와 맞먹는다. 병목이
inference가 아니라 decode였기 때문이다. `--npus 1 --decoders 1`에서 NPU
가동률은 11%(decode가 ~9배의 일: 실행 전체에서 8.3 s vs 0.9 s) — 단일
스레드 리포트가 예측한 "decode 10배 지배" 비율 그대로다.

측정으로 증명된 교훈 두 가지:

1. **파이프라이닝만으로는 병목을 못 이긴다 — decode 병렬화가 이긴다.**
   0.33 ms inference를 3.3 ms decode 뒤에 숨기는 건 작은 단계를 감출 뿐,
   decode 벽은 그대로다. 승부는 decode 풀을 NPU 개수와 독립적으로 확장하는
   데서 나며, 단계 분리가 그걸 가능하게 한다. 실제 SoC에서의 최종 해법은
   NPU 앞단의 하드웨어 비디오 디코더 블록이다.
2. **Decode 병렬성의 상한은 스레드 수가 아니라 스트림 수다.** 스트림이
   4개면 단계당 스트림이 single-flight라 decode 스레드는 최대 4개만 유효
   (초과분은 유휴). 디코더 가동률 편차(67–100%)는 이 동시성 상한 +
   스트림별 콘텐츠 비대칭이지, 스케줄링 버그가 아니다. 모든 스레드가
   메모리 서브시스템 하나를 공유하므로 부하가 오르면 스트림별 decode
   레이턴시가 약간 상승한다 — 인터커넥트를 공유하는 칩렛과 정확히 같은 현상.

클래스별 임계값은 *애플리케이션 정책*이다: SDK는 최소 임계값으로
디코드하고(메커니즘), 각 검출은 NMS 전에 자기 클래스의 기준을 넘어야 한다.
클래스별 신뢰도 분포가 비대칭일 때 유용하다 — 예:
`--threshold 0.55 --cls person=0.35`는 야간 "car" 오검출은 걸러내면서
약하지만 진짜인 person 검출은 통과시킨다.

### Temporal 확정 (`--temporal N`)

임계값은 1차원 손잡이라 겹치는 점수 분포를 분리하지 못한다(야간 불빛 덩어리
"car 0.63" vs 멀리 있는 진짜 "car 0.58"). 시간 축은 가능하다: 진짜 물체는
지속되고, 깜빡임은 사라진다. `--temporal N`은 스트림별로 작은 greedy-IoU
트래커(`src/temporal_filter.hpp`)를 돌려 N프레임 연속 매칭된 검출만
보고한다(트랙은 2프레임 누락까지 생존).

실측 (threshold 0.55, person 0.35, motorcycle/bicycle 0.4):

| 설정 | 야간 오검출 (후방, frame 300) | 도심 스쿠터 (우측) | det/frame raw→conf |
|---|---|---|---|
| 끔 | `car 0.63` 보임 | 유지 | 8.3 → 8.3 |
| `--temporal 2` ⭐ | **사라짐** | 유지 | 8.3 → 7.6 |
| `--temporal 3` | 사라짐 | **놓침** (빠른 움직임이 8×8 그리드에서 IoU 매칭을 깸) | 8.3 → 7.0 |

트레이드오프: 확정에 (N-1)프레임의 검출 지연이 추가되고(30 fps 기준
~33 ms/frame), 근거리 대형 물체는 이 그리드 해상도의 단순 IoU 연관으로는
너무 빨리 움직인다 — 교과서적 다음 단계는 움직임 보상 매칭
(등속 예측, 즉 최소 Kalman 필터)이다.

### 실제 검출

먼저 검출기를 학습시킨다 (SDK 트리에서 — YOLOv8n이 테스트 영상에
의사 라벨을 달고, TinyDetector의 PyTorch 쌍둥이가 그걸 학습):

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

학습 가중치를 쓰면 검출이 장면 내용을 따라간다
(처리량은 동일 — 가중치는 연산량을 바꾸지 않는다):

| 스트림 | 장면 | det/frame |
|---|---|---|
| drive_right | 붐비는 도심 거리 | **12.4** |
| drive_rear | 야간 고속도로 | 1.0 |
| drive_front | 시골길 | 0.4 |
| drive_left | 빈 해안도로 | **0.0** |

최종 리포트는 스트림별로 처리 프레임 수, 유효 fps, decode 레이턴시
(avg/p95/p99), inference 레이턴시(avg/p95/p99), 프레임당 평균 검출 수 —
그리고 전 스트림 합산 처리량을 보여준다.

## 테스트 영상

`drive_*.mp4`는 `visionpipe-npu/test_data/`에 있다 (Pexels License,
스트림 간 비교가 공정하도록 1080p / 30 fps / 24 s / 720프레임으로 통일
재인코딩). 출처는 해당 디렉터리의 README 참고.

## 실측 결과 (Apple Silicon Mac mini, Release 빌드)

4 스트림 × 1080p30 × 각 720프레임(총 2,880프레임),
`--npus 1 --decoders 1` 베이스라인(decode 스레드 1 ⇄ NPU 스레드 1;
기본값은 스트림당 NPU 1개 자동 선택 — 여기선 4개, 위 파이프라인 섹션 참고):

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

`util%` 열이 이미 결론을 말해준다: decoder 1개 + NPU 1개인데도 decode
스레드는 100%에 못박혀 있고 NPU는 11%에 머문다 — decode가 ~9배의 일이다.
처리량을 키우는 건 decode 스레드 추가(`--decoders 4`)다. 위 파이프라인
섹션 참고.

핵심 관찰:

1. **실시간 여유**: 모든 스트림이 ~81 fps 유지 — 30 fps 촬영 속도의
   2.7배. 1080p 카메라 4대가 노드 하나에 여유 있게 들어간다.
2. **Decode가 지배한다**: H.264 1080p 소프트웨어 디코드는 ~2.3–3.3 ms/frame,
   TinyDetector inference는 ~0.33 ms. 실제 SoC에서 NPU 앞단에 하드웨어
   비디오 디코더 블록을 두는 근거가 바로 이것이다.
3. **콘텐츠 비대칭이 decode에 드러난다**: 시각적으로 복잡한 주간 전방
   카메라가 가장 느리게(평균 3.33 ms), 어두운 후방 카메라가 가장 빠르게
   (2.18 ms) 디코드된다 — 비트레이트가 장면 복잡도를 따라가므로, 해상도가
   같아도 스트림별 tail 레이턴시는 다르다.

## 참고

- `--weights` 없이 실행하면 `TinyDetector`는 결정적 합성 가중치로 돈다 —
  박스는 의미 없지만 런타임 부하는 동일하다. 학습된 `.vpnw`(위 참고)를
  쓰면 같은 바이너리가 실제 차량/사람을 검출한다.
- 스케줄링은 2단계 work-stealing 파이프라인이다: `--decoders M` decode
  스레드와 `--npus N` NPU 스레드가 공유 `Pipeline`에서 스트림을 가져간다
  (위 참고). 계획된 확장: 우선순위 스케줄링(전방 카메라 우선), 과부하 시
  프레임 드롭, 각 NPU의 프레임을 하나의 `CommandQueue`로 라우팅.
