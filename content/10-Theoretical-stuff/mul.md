# Matrix Multiplication: Numerical Computing에서 GPU, HPC, Quantization까지

---

## 1. Introduction

Matrix multiplication은 현대 컴퓨팅에서 가장 중요한 primitive 중 하나다.

$$
C = AB
$$

는 종이 위에서는 한 줄짜리 정의지만, 실제 컴퓨터 시스템에서는 다음과 같은 요소들이 동시에 얽힌다.

- numerical stability (floating-point accumulation order)
- cache locality (data reuse, blocking)
- SIMD / SIMT execution
- memory bandwidth vs compute throughput
- Tensor Core utilization
- quantization scaling
- kernel fusion

특히 현대 AI/HPC 시스템에서는 거의 모든 핵심 workload가 결국 **GEMM (General Matrix Multiplication)** 으로 귀결된다.

- Transformer attention (QK^T, attention·V)
- MLP / Linear projection
- Convolution lowering (im2col + GEMM)
- Embedding projection
- HPC linear solver (LU, QR, Cholesky)
- Scientific simulation (FEM stiffness assembly)

따라서 GEMM은 단순한 수학 연산이 아니라, 하드웨어와 numerical method, 그리고 system software가 만나는 지점이다.

---

## 2. Numerical Matrix Multiplication

### 2.1 Definition

수학적 정의는 다음과 같다.

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}
$$

즉 A의 i번째 row와 B의 j번째 column의 dot product다. 이 정의에는 두 가지 자유도가 있다.

- **summation order**: k loop의 순서 (k를 안쪽에 두느냐, ij를 안쪽에 두느냐)
- **accumulation precision**: 합산을 어떤 정밀도로 수행할 것인가

수학적으로는 모두 동일하지만, 부동소수점 환경에서는 결과가 달라진다.

### 2.2 Computational Complexity

Naive 알고리즘:

$$
T(N) = O(N^3)
$$

하지만 이론적으로는 더 낮은 bound가 있다.

| Algorithm | Complexity |
|---|---|
| Naive | $O(N^3)$ |
| Strassen (1969) | $O(N^{2.807})$ |
| Coppersmith–Winograd (1990) | $O(N^{2.376})$ |
| Alman–Williams (2021) | $O(N^{2.3728596})$ |

matrix multiplication exponent $\omega$의 정확한 값은 여전히 미해결 문제이며 $\omega \ge 2$가 자명한 lower bound다.

그러나 실무에서는 **거의 항상 $O(N^3)$ 알고리즘이 사용된다.** 이유는:

- Strassen 계열은 numerical stability가 떨어진다 (error bound가 worse)
- recursion overhead, 작은 matrix에서 break-even 점이 너무 크다
- 무엇보다 modern hardware는 dense regular한 $O(N^3)$ pattern에 극단적으로 최적화되어 있다

즉 알고리즘 복잡도 자체보다, **arithmetic intensity와 hardware match**가 실측 성능을 결정한다.

### 2.3 Numerical Stability

Floating-point 환경에서 $\sum_k A_{ik}B_{kj}$는 결합법칙이 성립하지 않는다.

$$
(a + b) + c \ne a + (b + c) \quad \text{in IEEE 754}
$$

따라서 accumulation 순서가 곧 결과를 바꾼다. 주요 issue:

- **catastrophic cancellation**: 부호 다른 큰 값들의 합
- **swamping**: 작은 값이 큰 누적합에 흡수되어 사라짐
- **error accumulation**: $O(K)$에 비례한 unit roundoff 누적

이 때문에 GEMM 구현은 다음과 같은 전략을 쓴다.

- **higher-precision accumulator**: FP16 input → FP32 accumulator, FP8 input → FP32 accumulator
- **pairwise / tree reduction**: 순차 합산 대신 트리 형태로 합산해 error를 $O(\log K)$로 감소
- **Kahan summation**: compensation 변수를 두어 lost low bits를 보정

Tensor Core가 input precision과 accumulator precision을 분리하여 정의하는 이유가 바로 이것이다.

---

## 3. CPU에서의 Matrix Multiplication

초기 GEMM은 CPU 기반으로 발전했다. 핵심 최적화는 결국 **"memory hierarchy를 어떻게 활용할 것인가"** 라는 한 가지 질문으로 수렴한다.

### 3.1 Cache Blocking

Naive 구현:

```cpp
for (int i = 0; i < M; ++i)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      C[i][j] += A[i][k] * B[k][j];
```

이 코드는 cache locality가 매우 나쁘다.

- `A[i][k]`: row major면 sequential access (good)
- `B[k][j]`: column 방향 stride access (bad, cache line waste)
- `C[i][j]`: outer loop에서 register reuse는 가능하나, 큰 N에서는 L1 miss

따라서 실제 구현은 multi-level blocking을 한다.

```
for (jc = 0; jc < N; jc += Nc)       # L3 blocking
  for (kc = 0; kc < K; kc += Kc)     # L2 blocking
    pack B -> Bp
    for (ic = 0; ic < M; ic += Mc)   # L2 blocking
      pack A -> Ap
      for (jr ...)                   # L1 blocking
        for (ir ...)
          microkernel(Ap, Bp, C)     # register blocking
```

각 level별 tile size는 해당 cache 크기에 맞춰 정해진다.

- $M_c \times K_c$ tile of A → L2
- $K_c \times N_c$ tile of B → L3
- innermost microkernel → register

이 구조가 BLIS, OpenBLAS, MKL의 공통된 5-loop framework다.

### 3.2 SIMD와 FMA

현대 CPU는 SIMD instruction으로 한 cycle에 여러 element를 처리한다.

- AVX2: 256-bit (FP32 × 8, FP64 × 4)
- AVX-512: 512-bit (FP32 × 16, FP64 × 8)
- ARM NEON / SVE
- AMX (Intel Advanced Matrix Extensions): 2D tile register

특히 **FMA (Fused Multiply-Add)** 는 GEMM의 기본 단위다.

$$
d = a \cdot b + c \quad \text{(single rounding)}
$$

분리된 multiply+add 대비:
- throughput 2배 (한 instruction)
- rounding error 1회로 감소 (better numerical accuracy)

### 3.3 Goto's Algorithm

Kazushige Goto가 2008년 논문에서 정립한 high-performance GEMM 구조다. 핵심은:

- innermost microkernel은 register-resident
- L1/L2/L3 각각에 정확히 fit하는 tile size 선택
- A, B를 미리 packing하여 streaming access 보장

오늘날 OpenBLAS, BLIS, MKL의 GEMM은 모두 이 구조의 변형이다.

### 3.4 Microkernel

가장 안쪽 loop는 assembly나 intrinsic으로 직접 작성된 **microkernel**이다. 보통:

- $m_r \times n_r$ output tile (예: 6×16, 8×8)을 register에 hold
- $K_c$ 동안 누적
- pipelining으로 FMA latency hiding

이 microkernel 한 줄의 효율이 전체 GEMM 성능의 70% 이상을 결정한다.

---

## 4. BLAS와 GEMM

Matrix multiplication 인터페이스는 **BLAS (Basic Linear Algebra Subprograms)** 로 표준화되었다.

### 4.1 BLAS Levels

BLAS는 arithmetic intensity 기준으로 세 level로 나뉜다.

| Level | 연산 | 예시 | Intensity |
|---|---|---|---|
| 1 | vector–vector | `axpy`, `dot` | $O(1)$ FLOP/Byte |
| 2 | matrix–vector | `gemv` | $O(1)$ FLOP/Byte |
| 3 | matrix–matrix | `gemm` | $O(N)$ FLOP/Byte |

**Level 3만이 arithmetic intensity가 매트릭스 크기에 비례해 증가**한다. 즉 cache reuse가 가능하다. 이것이 GEMM이 hardware acceleration의 sweet spot인 근본적인 이유다.

### 4.2 Canonical GEMM Form

BLAS Level 3의 핵심 primitive는 단순한 $C = AB$가 아니라:

$$
C \leftarrow \alpha A B + \beta C
$$

여기서 $A, B, C$는 matrix, $\alpha, \beta$는 scalar이다.

### 4.3 α와 β

- **α**: GEMM 결과 scaling
- **β**: 기존 output accumulation 가중치
  - $\beta = 0$: overwrite ($C = \alpha AB$)
  - $\beta = 1$: accumulate ($C \mathrel{+}= \alpha AB$)
  - 그 외: weighted update

### 4.4 왜 이런 형태를 쓰는가

별도 kernel chain:

```
tmp = A * B       # GEMM
tmp = alpha * tmp # scale
C   = beta * C    # scale
C   = C + tmp     # add
```

은 매 단계마다 global memory traffic이 발생한다. 반면 fused form은 microkernel의 register accumulator 위에서:

```
acc = alpha * acc + beta * C   # in-register
store(acc)
```

한 번의 store로 끝난다.

즉 α, β는 단순한 수학적 일반화가 아니라 **memory traffic을 줄이기 위한 hardware-aware 설계**다.

---

## 5. GPU로의 이동

CPU는:

- low latency, complex control flow
- 작은 batch, branchy code

에 강하지만, **dense linear algebra의 massive parallelism**에는 한계가 있다.

GPU는 정반대 trade-off를 택했다.

- 수천 개의 thread를 동시에 실행
- 높은 arithmetic throughput (TFLOPS)
- 높은 memory bandwidth (HBM, TB/s 단위)
- 그 대신 single-thread latency는 길고, branch divergence에 취약

GEMM은 다음 특성 덕분에 GPU와 완벽히 맞는다.

- **regular, dense, predictable**: branch가 없다
- **arithmetic intensity가 높다**: bandwidth로 throughput을 채울 수 있다
- **massive data parallelism**: tile 단위로 independent

---

## 6. GPU Architecture와 Matrix Multiplication

GPU GEMM은 hardware의 multi-level hierarchy에 정확히 mapping된다.

| Hardware | Software | 역할 |
|---|---|---|
| GPC / SM | grid / thread block | global tile 분할 |
| sub-core | warp (32 threads) | warp-level MMA |
| ALU / Tensor Core | thread | element-level operation |
| HBM | global memory | bulk data |
| L2 | (shared) | 재사용 |
| SMEM | shared memory | tile staging |
| Register file | register | accumulator |

### 6.1 Hierarchical Tiling

GPU GEMM은 보통 세 level의 tiling을 한다.

- **CTA tile** (예: 128×128): thread block 하나가 담당
- **Warp tile** (예: 64×32): warp 하나가 담당
- **Thread tile** (예: 8×8): thread 하나가 register에 hold

각 level은 그 level의 storage에 fit하도록 설계된다.

### 6.2 Shared Memory and Bank Conflicts

Global memory (HBM) 접근은 latency가 길다 (수백 cycle). 따라서:

- A tile, B tile을 SMEM으로 staging
- 동일 tile을 여러 warp가 reuse

SMEM은 32개의 bank로 나뉘는데, 같은 bank의 다른 word에 동시 접근하면 **bank conflict**가 일어나 직렬화된다. 이를 피하기 위해:

- **swizzling**: address bit 재배치로 conflict 회피
- **padding**: 의도적으로 stride에 +1을 더해 conflict 분산
- **ldmatrix / stmatrix** instruction (Ampere+): hardware-level conflict-free load

### 6.3 Register Accumulation

실제 누적은 register에서 일어난다.

$$
\text{Acc}_{\text{reg}} \mathrel{+}= A_{\text{smem}} \cdot B_{\text{smem}}
$$

최종 store 직전까지 accumulator는 register-resident다. 이게 가능한 이유는 GPU의 register file이 매우 크기 때문이다 (SM당 256KB regs on A100/H100).

### 6.4 Software Pipelining (Double Buffering)

Tensor Core가 tile k를 계산하는 동안, 다음 tile k+1을 SMEM으로 미리 fetch한다.

```
stage 0: load tile_0 -> smem_A0, smem_B0
loop k:
  prefetch tile_{k+1} -> smem_A1, smem_B1   # async
  compute tile_k from smem_A0, smem_B0       # Tensor Core
  swap buffers
```

이를 통해 memory latency가 compute로 가려진다 (latency hiding).

### 6.5 Async Copy and TMA

- Ampere: `cp.async` — global → shared를 register를 거치지 않고 비동기로 copy
- Hopper: **TMA (Tensor Memory Accelerator)** — 2D/3D tile 전체를 hardware engine이 비동기 transfer. thread가 직접 address를 만들지 않아도 됨

이러한 기능은 software pipelining의 효율을 비약적으로 끌어올렸다.

---

## 7. Tensor Core와 MMA

현대 NVIDIA GPU는 Tensor Core를 제공한다. Tensor Core는 다음을 **단일 hardware instruction**으로 수행한다.

$$
D = A B + C
$$

이를 MMA (Matrix Multiply-Accumulate)라 부른다.

### 7.1 Tensor Core 세대별 변화

| 세대 | GPU | 추가된 정밀도 / 기능 |
|---|---|---|
| 1st | Volta (V100) | FP16 → FP32 acc, 4×4×4 |
| 2nd | Turing (T4) | + INT8, INT4 |
| 3rd | Ampere (A100) | + BF16, TF32, `cp.async`, sparsity |
| 4th | Hopper (H100) | + FP8 (E4M3, E5M2), `wgmma`, TMA |
| 5th | Blackwell (B200) | + FP4, microscaling (MX) format, 2nd-gen Transformer Engine |

### 7.2 대표 Instruction

- `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` (Ampere)
- `wgmma.mma_async.sync.aligned.m64n128k16` (Hopper, warp-group async MMA)
- `ldmatrix` / `stmatrix`: Tensor Core register layout에 맞는 SMEM load

이 instruction들은 PTX 레벨에서 호출되며, CUTLASS는 이를 C++ template으로 wrapping한다.

### 7.3 Input과 Accumulator 정밀도 분리

Tensor Core의 핵심 설계는 **input precision과 accumulator precision의 분리**다.

| Input | Accumulator |
|---|---|
| FP16 | FP32 |
| BF16 | FP32 |
| TF32 | FP32 |
| FP8 (E4M3/E5M2) | FP32 |
| INT8 | INT32 |
| FP4 (MX) | FP32 |

이는 §2.3의 numerical stability 이슈와 직접 연관된다. low-precision input의 dynamic range 문제는 scaling으로, accumulation error는 wider accumulator로 분리해 해결한다.

---

## 8. GEMM Libraries

GPU/CPU에서 GEMM은 거의 항상 library를 통해 호출된다.

### 8.1 OpenBLAS / BLIS / MKL

CPU 기반 optimized BLAS implementation.

- **OpenBLAS**: Goto's algorithm 기반, multi-arch (x86, ARM, RISC-V)
- **BLIS**: Goto의 후신, microkernel만 arch별로 작성하면 framework가 처리
- **Intel MKL**: 닫힌 소스, Intel CPU에서 최적

공통 특징: AVX/AVX-512 활용, multi-threaded, NUMA-aware.

### 8.2 cuBLAS / cuBLASLt

NVIDIA의 공식 GPU BLAS library.

- **cuBLAS**: 전통적인 BLAS API
- **cuBLASLt**: light-weight 확장, epilogue fusion, mixed precision, batched/strided GEMM에 특화

핵심 기능:
- Tensor Core 자동 활용
- heuristic + autotuning으로 kernel 선택
- mixed precision (FP16 in, FP32 acc)
- batched / strided batched GEMM

대부분의 DL framework (PyTorch, TensorFlow)가 backend로 사용한다.

### 8.3 CUTLASS

NVIDIA의 CUDA C++ template GEMM framework.

cuBLAS보다 한 단계 낮은 추상 level에서 작동하며, GEMM을 **building block의 조합**으로 표현한다.

- threadblock / warp / instruction shape를 template parameter로 지정
- custom epilogue (bias, activation, quantization 등)
- custom data layout
- Tensor Core kernel generation

TensorRT, Triton의 일부 backend, 수많은 custom inference kernel이 CUTLASS 위에서 만들어진다.

### 8.4 Triton, ThunderKittens

최근에는 Python DSL로 GPU kernel을 작성하는 흐름이 강해졌다.

- **Triton** (OpenAI): block-level abstraction, autotuning이 내장
- **ThunderKittens** (Stanford): tile-level primitive를 C++로 노출, Hopper에 특화

CUTLASS만큼의 fine control은 어려우나, 생산성이 훨씬 높다.

---

## 9. Epilogue Fusion

현대 GPU GEMM에서 핵심은 **단순 multiplication이 아니다.** 실제 canonical form은:

$$
D = \phi\!\left(\alpha A B + \beta C + b\right)
$$

여기서:
- $b$: bias vector
- $\phi$: activation function (ReLU, GeLU, SiLU, …)

### 9.1 왜 Fusion이 중요한가

별도 kernel 구성:

```
GEMM
  -> global store
  -> bias add kernel
  -> global store
  -> activation kernel
  -> global store
```

은 매 단계 global memory를 왕복한다. arithmetic intensity가 매우 낮아진다.

반면 fused epilogue:

```
register accumulator
  -> +bias
  -> activation
  -> single store
```

는 한 번의 store로 끝난다. 효과는:

- bandwidth 감소
- kernel launch overhead 감소
- L2/HBM traffic 감소
- end-to-end latency 감소

### 9.2 일반적인 Epilogue Pattern

| Pattern | 예시 |
|---|---|
| linear + bias | `out = AB + b` |
| linear + bias + activation | `out = ReLU(AB + b)` |
| linear + residual | `out = AB + x` (skip connection) |
| quantize | `out = quantize(AB * s)` |
| dequant + matmul + requant | INT8 inference 전체 흐름 |

CUTLASS, cuBLASLt, TensorRT 모두 이런 epilogue 조합을 first-class로 지원한다.

---

## 10. HPC에서의 Matrix Multiplication

HPC (High Performance Computing) 영역에서도 GEMM은 가장 중요한 kernel이다.

대표 workload:
- CFD (Computational Fluid Dynamics)
- FEM (Finite Element Method)
- Weather / climate simulation
- Molecular dynamics
- Quantum chemistry (DFT, CCSD)
- LINPACK (Top500 ranking benchmark)

LINPACK이 사실상 dense LU decomposition이고, LU의 핵심이 panel-trailing matrix update GEMM이라는 점에서, **Top500 순위는 사실상 distributed GEMM 성능 순위**다.

### 10.1 Arithmetic Intensity와 Roofline

Arithmetic intensity:

$$
I = \frac{\text{FLOP}}{\text{Byte}}
$$

GEMM의 intensity는 tile 크기에 비례해 증가한다 ($O(N)$). 따라서 충분히 큰 tile에서는 **bandwidth가 아닌 compute가 bottleneck**이 된다.

**Roofline model**:

$$
\text{Perf} = \min(\text{Peak FLOPS},\ I \cdot \text{Peak Bandwidth})
$$

GEMM은 거의 유일하게 roofline의 compute-bound 영역에 진입하는 kernel이다.

### 10.2 Distributed GEMM

초대형 GEMM은 multi-node로 분산된다.

분산 알고리즘:
- **SUMMA** (Scalable Universal Matrix Multiplication Algorithm): 2D process grid에서 broadcast 기반
- **2.5D / 3D algorithm**: memory를 더 써서 communication을 줄임
- **Cannon's algorithm**: 2D torus에서 systolic-like shift

통신 stack:
- MPI (Message Passing Interface)
- NCCL (NVIDIA Collective Communications Library)
- NVLink, NVSwitch (intra-node)
- InfiniBand, RoCE (inter-node)

수십만 GPU 규모에서는 GEMM 자체보다 **communication overlap**이 성능을 결정한다.

---

## 11. Quantization

최근 AI inference에서 가장 중요한 주제 중 하나는 quantization이다. 핵심 아이디어는:

> **고정밀 weight/activation을 저비트 정수 또는 저정밀 부동소수점으로 근사하여 메모리·연산을 줄인다.**

### 11.1 Quantization 기본 식

대칭 (symmetric) quantization:

$$
x_q = \text{round}(x / s_x), \qquad x \approx s_x \cdot x_q
$$

비대칭 (asymmetric) quantization:

$$
x_q = \text{round}(x / s_x) + z_x, \qquad x \approx s_x (x_q - z_x)
$$

여기서:
- $s_x$: scale (per-tensor, per-channel, per-token, per-group 등)
- $z_x$: zero-point (offset)

### 11.2 Granularity

| Granularity | 적용 단위 | 특징 |
|---|---|---|
| per-tensor | 전체 tensor 하나의 scale | 단순, error 큼 |
| per-channel | output channel별 scale | conv/linear에 표준 |
| per-token | activation token별 scale | LLM activation에 유리 |
| per-group | weight를 group으로 나눠 scale | GPTQ, AWQ 등 |

### 11.3 Quantized GEMM

INT8 GEMM의 흐름:

$$
\text{Acc}_{\text{int32}} = A_q B_q
$$

이후 dequantization:

$$
Y = s_A s_B \cdot \text{Acc}_{\text{int32}}
$$

즉 외부에서 보면 BLAS의 α와 정확히 같은 위치다.

$$
\alpha = s_A s_B
$$

asymmetric의 경우 zero-point 항이 추가된다.

$$
A_q B_q = (s_A^{-1}(A - z_A))(s_B^{-1}(B - z_B))
$$

를 전개하면 cross-term이 생기는데, 이를 미리 계산해 bias로 흡수하는 것이 일반적인 구현이다.

### 11.4 Calibration과 PTQ / QAT

- **PTQ (Post-Training Quantization)**: 학습된 모델에 calibration data를 흘려 scale을 결정
- **QAT (Quantization-Aware Training)**: 학습 중에 fake-quant를 삽입
- **GPTQ, AWQ, SmoothQuant**: LLM weight quantization에 특화된 알고리즘

---

## 12. Q/DQ Pipeline

Modern inference engine은 다음 pipeline을 single fused kernel로 수행한다.

```
Quantize (FP -> INT8)
  -> INT8 MMA (Tensor Core)
  -> INT32 accumulation
  -> Dequant scaling (* s_A s_B)
  -> +Bias
  -> Activation (ReLU/GeLU)
  -> Requantize (FP -> INT8 for next layer)
```

이 전체가 §9의 epilogue fusion 안에 들어간다. TensorRT, cuBLASLt, CUTLASS의 INT8 GEMM이 이 형태다.

핵심 통찰:
- INT8 MMA 자체는 Tensor Core가 처리 (compute-bound)
- 앞뒤의 quant/dequant/bias/activation은 epilogue에서 register-level fusion
- 이 둘을 합쳐야 INT8의 이론적 효율이 실현된다

### 12.1 양자화 모델의 Graph 표현

위 pipeline은 runtime의 모습이다. 그렇다면 학습이 끝난 양자화 모델은 어떻게 표현되어 이 pipeline까지 도달할까? "어디에서 어떻게 quantize/dequantize가 일어나는가"의 의미를 model representation에 인코딩하는 방식에는 두 가지 큰 패러다임이 있다.

#### Graph-level: Explicit Q/DQ Node

가장 보편적인 방식. fp graph에 `Quantize`와 `DeQuantize` node를 명시적으로 끼워 넣는다.

```
DQ(A_q) -> Linear -> Q
DQ(W_q) ----^
```

이후 graph optimizer가 패턴을 인식해 fused integer op로 rewrite한다.

```
   ↓ (graph rewrite)

QLinear(A_q, s_A, z_A, W_q, s_W, z_W, s_Y, z_Y) -> Y_q
```

이 rewrite는 두 단계로 구성된다.

1. **selector**: subgraph pattern을 찾고, fusion이 의미적으로 안전한지 (수학적 등가성, dtype 호환성, transpose flag 등) 검사
2. **action**: 검사를 통과한 subgraph를 fused op로 교체

selector가 보수적이지 못해 안전하지 않은 케이스를 통과시키면, kernel 자체와 무관하게 **graph 변환만으로 wrong output**이 발생한다. 따라서 graph rewriter의 fail-safe 원칙은:

> 수학적 등가가 보존되지 않는 fusion은, 빠른 잘못된 답보다 느린 정확한 답으로 fallback해야 한다.

PyTorch에서 이 패러다임은 두 세대로 구현되어 있다.

- **FX Graph Mode Quantization** (`torch.ao.quantization`): `prepare_fx` / `convert_fx`가 FX symbolic trace 위에서 Q/DQ node를 자동 삽입
- **PT2E (PyTorch 2 Export Quantization)** — 현재 권장 path: `torch.export` 기반. Q/DQ가 ATen op로 graph에 박힌다:

```
torch.ops.quantized_decomposed.quantize_per_tensor
torch.ops.quantized_decomposed.dequantize_per_tensor
```

다른 framework도 동일 구조다 — ONNX의 QDQ format, TVM의 `qnn` dialect, MLIR의 quant dialect 모두 graph-level Q/DQ 표현이다.

#### Tensor-level: Quantized Tensor Subclass

새로운 흐름. **graph topology는 손대지 않고, tensor에 quantization metadata를 담는다.**

```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig())
```

내부적으로 weight tensor가 `AffineQuantizedTensor`, `Float8Tensor` 같은 subclass로 교체된다. graph는 여전히 `F.linear(x, w)` 형태이고, PyTorch dispatcher가 `__torch_function__` / `__torch_dispatch__` hook으로 quantized kernel을 호출한다.

대표 구현 (`torchao`):

- `AffineQuantizedTensor`: int4 / int8 weight, per-tensor / per-channel / per-group granularity
- `Float8Tensor`: FP8 training/inference, per-tensor 또는 per-row scaling
- MX format, NVFP4 같은 차세대 format도 동일 subclass 구조로 통합 중

장점:
- graph가 깨끗 — 선형대수 의미가 그대로 노출되어 분석/디버깅 용이
- LLM weight-only quantization, FP8 training처럼 **dtype 중심 사고**에 자연스럽다
- `torch.compile` (Inductor)이 specialization 시점에 dequant + matmul을 single kernel로 fuse

vLLM의 LLM serving quantization, ExecuTorch on-device deployment의 backbone이 torchao다.

#### 두 패러다임의 비교

| 축 | Graph-level (Q/DQ node) | Tensor-level (subclass) |
|---|---|---|
| 추상화 위치 | graph topology | tensor type system |
| 표현 매체 | Q/DQ op (node) | quantized tensor subclass |
| 대표 구현 | ONNX QDQ, PT2E, TVM QNN | torchao |
| Static export | 자연스러움 | export 시 Q/DQ로 materialize 필요 |
| Fusion 주체 | graph rewriter / backend lowering | runtime dispatch + `torch.compile` |
| 강점 영역 | AOT compiler, interchange format | LLM weight-only, FP8 training |

#### 공통 도착지

두 패러다임은 **언제, 어디에서 quantization 의미를 materialize하느냐**만 다를 뿐, 결국 §12 서두의 동일한 fused low-precision GEMM kernel로 lowering된다. 그 kernel 내부의 수학 (§11의 $\alpha = s_A s_B$, accumulator precision 분리, scaling layout)은 양쪽 모두 동일하다.

graph optimizer의 책임은 이 lowering이 의미를 보존하도록 보장하는 것이다 — 어느 패러다임이든.

---

## 13. FP8 시대

Hopper (H100) 이후 FP8이 핵심 정밀도가 되었다. FP8은 두 가지 format으로 정의된다.

| Format | Exponent | Mantissa | 특징 |
|---|---|---|---|
| E4M3 | 4 bits | 3 bits | dynamic range 좁음, precision 높음. forward에 사용 |
| E5M2 | 5 bits | 2 bits | dynamic range 넓음, precision 낮음. backward (gradient)에 사용 |

### 13.1 왜 Scaling이 필수인가

FP8은 dynamic range가 매우 좁다 (E4M3 기준 약 $\pm 448$). 따라서 raw FP32/BF16 값을 그대로 cast하면 saturate 또는 underflow가 발생한다.

해결: per-tensor 또는 per-block scale을 곱해 FP8 표현 가능 범위로 normalize.

$$
\tilde{A} = s_A \cdot A_{fp8}, \qquad \tilde{B} = s_B \cdot B_{fp8}
$$

GEMM 결과:

$$
C = s_A s_B (A_{fp8} B_{fp8})
$$

즉 **scaling이 곧 numerical correctness의 일부**가 된다. INT8과 본질적으로 같은 수학 구조다.

### 13.2 Delayed Scaling과 Transformer Engine

Per-tensor scale을 실시간으로 계산하면 reduction이 필요해 overhead가 크다. NVIDIA Transformer Engine은:

- 직전 N step의 amax를 추적 (rolling window)
- 다음 step의 scale을 이 history에서 예측

하는 **delayed scaling** 전략을 쓴다.

### 13.3 Microscaling (MX) Format

Blackwell (B200)에서 도입된 **MX format**(OCP 표준)은 일정 크기의 block마다 하나의 scale을 갖는다.

$$
A_{\text{block}} = s_{\text{block}} \cdot \{a_0, a_1, \ldots, a_{B-1}\}
$$

- per-tensor보다 훨씬 fine한 granularity
- 그러나 per-element scale보다 훨씬 적은 metadata
- FP4 (MXFP4), FP6 (MXFP6), FP8 (MXFP8), INT8 (MXINT8) 모두에 적용 가능

이는 §11.2의 per-group quantization을 hardware native로 끌어올린 것이다.

표준 OCP MX format의 핵심 사양:

| 항목 | 값 |
|---|---|
| Block size | 32 elements |
| Block scale type | **E8M0** (8-bit, exponent only, power-of-2) |
| Element formats | FP4(E2M1), FP6(E2M3/E3M2), FP8(E4M3/E5M2), INT8 |

E8M0 scale은 부호·mantissa가 없는 순수 exponent다. 즉 $2^k$ 형태로만 scaling이 가능하다 — 가벼운 대신 표현력이 제한된다.

### 13.4 NVFP4: NVIDIA의 자체 FP4 Format

Blackwell은 OCP MX 외에도 NVIDIA 자체 format인 **NVFP4**를 hardware native로 지원한다. 핵심 차이는 **two-level scaling**과 **더 작은 block**이다.

| 항목 | MXFP4 (OCP) | NVFP4 (NVIDIA) |
|---|---|---|
| Element format | FP4 (E2M1) | FP4 (E2M1) |
| Block size | 32 | **16** |
| Per-block scale | E8M0 (power-of-2) | **E4M3** (FP8) |
| Per-tensor scale | 없음 | **FP32** (second level) |
| Scaling 표현력 | $2^k$ 만 | 임의 실수 (FP8 정밀도) |

수식으로:

$$
x \approx s_{\text{tensor}}^{\text{(fp32)}} \cdot s_{\text{block}}^{\text{(e4m3)}} \cdot x^{\text{(fp4)}}
$$

장점:
- **block size 16**: MX의 32보다 더 fine → outlier에 강함
- **E4M3 block scale**: power-of-2가 아닌 실수 scale 가능 → quantization error 감소
- **FP32 per-tensor scale**: 전체 dynamic range를 추가로 흡수

단점:
- metadata overhead가 MX보다 크다 (block당 8-bit vs 8-bit지만 block 수는 2배)
- format이 NVIDIA 독점 → portability는 떨어짐

### 13.5 MX vs NVFP4: 실무적 의미

LLM inference 관점에서:

- **MXFP4**: 표준화되어 있어 ecosystem 호환성이 좋다. 하지만 power-of-2 scale 때문에 sensitive layer (attention output projection 등)에서 accuracy drop이 보고된다
- **NVFP4**: NVIDIA 측 벤치마크상 FP8 대비 accuracy 손실이 매우 작다고 보고됨. TensorRT-LLM, Transformer Engine이 이를 native로 활용
- 둘 다 GEMM 자체는 **FP4 input × FP4 input → FP32 accumulator**의 동일한 Tensor Core pipeline을 사용. 차이는 dequant scaling 단계에서 epilogue가 어떤 scale을 곱하느냐일 뿐

즉 §11의 quantization framework, §9의 epilogue fusion, §13의 scaling이 모두 한 점으로 모이며, MX/NVFP4는 그 위에서 **metadata layout과 scale 표현 방식의 trade-off**일 뿐이다.

---

## 14. Compiler와 Kernel Fusion

현대 compiler stack:

- **TensorRT** (NVIDIA, inference)
- **XLA** (Google, JAX/TF)
- **TVM** (Apache, multi-target)
- **Triton** (OpenAI, kernel DSL)
- **torch.compile / Inductor** (PyTorch)
- **MLIR / IREE**

이들은 더 이상 단순히 cuBLAS를 호출하지 않는다. 대신:

```
GEMM + Bias + Activation + Quantization + LayerNorm
```

전체를 분석해, **하나의 fused kernel**로 code generation한다.

이 단계에서 결정되는 요소:
- tile size (M, N, K block)
- pipeline depth (몇 stage prefetch)
- precision (FP16/BF16/FP8 선택)
- epilogue 구조
- shared memory layout / swizzling
- async copy 사용 여부

즉 modern matrix multiplication은 더 이상 단순한 linear algebra primitive가 아니라, **compiler가 search space에서 생성하는 program**이다.

---

## 15. Modern Perspective

현대 AI/HPC 시스템에서 matrix multiplication은 사실상 **Universal Compute Primitive**의 지위에 있다.

- Transformer attention → batched GEMM
- MLP / FFN → GEMM
- Conv → im2col + GEMM (또는 implicit GEMM)
- Embedding lookup + projection → sparse GEMM
- HPC solver → distributed GEMM
- Scientific simulation → batched small GEMM

결국 거의 모든 compute-intensive workload는 어떤 형태의 GEMM pipeline으로 환원된다.

따라서 hardware (Tensor Core, TMA, MX format), software (CUTLASS, Triton, compiler), numerical method (scaling, accumulator precision)가 모두 동일한 한 점, GEMM을 향해 수렴한다.

---

## 16. Conclusion

Matrix multiplication은 단순한 다음 정의가 아니다.

$$
C = AB
$$

현대 시스템에서 GEMM은 다음 요소들이 모두 결합된 핵심 computational primitive다.

- cache hierarchy 활용 (multi-level blocking)
- SIMD / SIMT execution model
- Tensor Core MMA
- epilogue fusion
- quantization scaling
- mixed precision (input vs accumulator)
- distributed reduction
- compiler-generated kernel

특히 AI 시대에 들어서면서:

- Tensor Core
- low-precision arithmetic (INT8, FP8, FP4)
- fused epilogue
- quantized / scaled GEMM
- microscaling format

이 중심이 되며, matrix multiplication은 이제 단순한 수학 연산이 아니라 **programmable compute pipeline**으로 진화했다.

GEMM을 깊게 이해한다는 것은, 사실상 modern computing system의 단면을 이해한다는 것과 같다.
