# NVIDIA GPU ISA - Warp 레벨과 Uniform 연산

> 원문: https://zhuanlan.zhihu.com/p/712357647

**목차**
- SIMT 체계에서의 Tensor 계산
- Warp 레벨 명령
  - DMMA
  - HMMA
  - IMMA
  - SHFL 등
- Uniform 명령
- 정리
- 참고

이전 글들에서 Load, 부동소수, 정수, 비트·논리 명령을 다뤘습니다. 이들은 논리적으로 단일 스레드 단위로 동작합니다. NVIDIA GPU는 그 외에도 **warp 단위** 로만 이해·실행 가능한 명령들을 제공합니다. 또 warp 레벨 Uniform 레지스터와 대응 명령으로 효율을 끌어올립니다. 본 글은 그 두 부류의 명령을 다룹니다.

## SIMT 체계에서의 Tensor 계산

NVIDIA GPU는 SIMT(Single Instruction Multiple Thread) 모델을 제공하며, 프런트엔드 언어는 CUDA. 프로그래머의 사고:

- 병렬화 가능한 문제를 더 작은 원자적 계산 단위로 분할
- CUDA의 단일 thread로 해당 계산을 작성

이때 집중하는 건 국소적·원자적 단일 thread의 실행 로직입니다. 다음 단계로 thread block, grid block을 구성해 가로로 확장합니다.

![Figure 1. 문제 분할 → 하드웨어 공급](images/v2-1ecf83d6170568c2e18c89384cc86a58_1440w.jpg)
*Figure 1. 문제를 작게 쪼개 하드웨어에 공급*

CUDA에선 단일 thread 로직을 쓰지만 하드웨어가 실제로 스케줄하는 최소 단위는 **warp**(32 thread). 32 thread는 동시에 시작·종료되며 lock-step. Volta 이전엔 32 thread가 같은 PC를 공유했고, Volta 이후에는 각자 독립 PC가 있어 락스텝으로 인한 데드락 문제를 회피할 수 있습니다. 그렇다고 thread가 자유롭게 독립 실행되는 건 아니고, 경합 구간 이후엔 동기 지점을 두어 warp의 효율적 실행을 유지합니다.

SIMT 의미상 각 thread는 자신의 사적 저장 공간(분할된 레지스터)을 가집니다. lane-0, lane-1, lane-31은 warp의 첫째·둘째·마지막 thread. 스케줄러가 실행 가능한 warp를 골라 `FADD R3 R2 R1` 같은 명령을 SIMD EU에 보내면 EU가 32 lane의 작업을 병렬로 끝냅니다. lane 결과는 각자의 R3에 저장되고 상호 영향 없음.

![Figure 2. Lane은 독립 논리 유닛](images/v2-3c17849403a071b8d87ceaa0e5c5a41c_1440w.jpg)
*Figure 2. lane은 레지스터를 갖는 독립 논리 유닛*

SIMT 프레임워크에서 레지스터 자원은 thread별로 분할되며, 논리적으로 thread는 서로 독립적이라 영향이 없습니다. 단순하고 효율적이며 전통 CUDA 프로그래밍은 이 모델 위에서 이뤄집니다.

딥러닝 폭발과 Google TPU에 대응해 NVIDIA는 Volta부터 **Tensor Core** 를 도입했습니다. Tensor Core는 정해진 규격의 행렬 곱을 짧은 명령 주기에 완수해 GPU 연산력을 비약적으로 높였습니다. SIMT 프로그래밍 모델도 확장됐습니다. Tensor Core는 한 lane이 자기만의 저장과 단순 로직을 갖는 식과는 다릅니다. 실행에 warp 내 모든 lane이 참여해야 하며, 각 lane이 일부 데이터만 입력하고, 결과도 각 lane의 레지스터에 분산 저장됩니다. lane 간 데이터 공유·교환이 도입됩니다. 하드웨어에 레지스터 파일은 원래 하나이므로 (lane "사적"은 논리 매핑일 뿐) 별 일이 아니지만, 소프트웨어의 SIMT 추상엔 이질적입니다. lane 단위의 사적·독립 계산 논리를 깨고, 모든 thread가 데이터를 동시에 공급하는 식으로 사고를 바꿔야 합니다. 각 lane은 fragment에 불과하고 Tensor Core가 계산한 뒤 결과를 다시 lane들에 채워 줍니다. lane-i의 입력이 lane-j(i≠j)의 출력에 영향. 이런 부류의 명령을 **Warp-Level 명령** 이라 부릅니다.

![Figure 3. Tensor Core는 SIMT의 단순 확장이 아님](images/v2-ba8210edea4170cf2bf736eb94767d66_1440w.jpg)
*Figure 3. Tensor Core는 lane 간 레지스터 자원을 공유*

`FADD`는 전통 SIMT — lane은 사적이고 독립. `HMMA`류는 warp 레벨 — warp 내 모든 lane이 참여, 출력이 lane 사적 데이터에 한정되지 않음.

## Warp 레벨 명령

핵심: Tensor Core 행렬 곱-누적 명령.

```
DMMA, HMMA, IMMA
```

### DMMA

DMMA(Double Matrix Multiply Accumulate). FP64 행렬 곱-누적. 그림 4는 DMMA 계산의 논리 공간과 레지스터 분포. lane-10은 A의 (2행 2열) 원소를 제공하고 B의 (2행 2열) 원소를 제공한 뒤, Tensor Core가 D = AB + C를 계산해 D의 (2행 4·5열)을 lane-10이 받습니다. D의 각 원소는 해당 행·열에 필요한 모든 데이터가 동원되므로 lane-10만의 결과로 결정되지 않습니다.

![Figure 4. DMMA의 레지스터 분포](images/v2-01b4f239a9c50d645194037d18158bfd_1440w.jpg)
*Figure 4. Tensor Core DMMA*

Ampere의 DMMA 예:

```
DMMA.884 R64 R96 R90 R64
```

`884`는 mnk가 8·8·4, 즉 `D[8×8] = A[8×4] · B[4×8] + C[8×8]`. R64, R65, R66, R67이 연속 출력 (D의 double 2개). R96, R97이 A의 double 1개. R90, R91이 B의 double 1개. Modifier로 A/B 음수화, reuse 가능:

```
DMMA.884 R64 -R96 R90 R64
DMMA.884 R64 R96 -R90 R64
DMMA.884 R64 -R96 -R90 R64
DMMA.884 R64 R96.reuse R90 R64
DMMA.884 R64 R96 R90.reuse R64
```

위 레지스터 분포에 맞춰 NVIDIA는 `LDSM` 명령을 제공합니다. ([B42](../B42_gpu_isa_load_cache/README.md) 참고)

### HMMA

HMMA(Half Matrix Multiply Accumulate). 반정밀(half/bfloat16/tfloat32) 행렬 곱-누적.

```
HMMA.16816.F16
HMMA.16816.F32
HMMA.16816.F32.BF16
HMMA.1688.F16
HMMA.1688.F32
HMMA.1688.F32.BF16
HMMA.1688.F32.TF32
```

`16816`은 mnk = 16·8·16, `1688`은 16·8·8. 타입 modifier — `F16`(ABCD 모두 half), `F32`(누산기 float32, AB는 half), `F32.BF16`(누산기 float32, AB는 bfloat16), `F32.TF32`(누산기 float32, AB는 tfloat32). 자세한 레지스터 분포는 PTX 매뉴얼.

### IMMA

IMMA(Integer Matrix Multiply Accumulate). 정수 행렬 곱-누적.

```
IMMA.16816.S8.S8
IMMA.16832.S8.S8.SAT
IMMA.8816.S8.S8.SAT
```

누산기는 int32. `S8.S8`은 A/B가 signed int8. `SAT`는 누산기 오버플로 시 int32 max/min으로 포화. A/B에 unsigned 8-bit, signed 4-bit, unsigned 4-bit, 1-bit도 가능하고 mnk가 조정됩니다. cute의 MMA 추상도 참고.

### SHFL 등

행렬 곱-누적 외 warp 레벨 명령:

```
SHFL
VOTE / VOTEU
REDUX
WARPSYNC
```

**SHFL** (SHuFfLe): warp 내 lane 간 레지스터 데이터 교환. 정렬·reduce 등에 유용.

```
SHFL.BFLY  SHFL.DOWN  SHFL.IDX  SHFL.UP
```

각 modifier는 XOR/Down/지정 인덱스/Up 방식의 교환. CUDA의 `__shfl_sync` 류 함수에 대응.

**VOTE**: warp의 lane bool reduce. `Any`(하나라도 True면 True), `All`(모두 True). `VOTEU`는 Warp Uniform 레지스터까지 활용.

Ampere부터는 정수 reduce도 가능 — **REDUX**:

```
REDUX.MAX.S32
REDUX.MIN.S32
REDUX.OR
```

```cpp
int32_t ret = V[lane0];
for (int ilane = 1; ilane < 32; ++ilane) ret = max(ret, V[ilane]);
```

CUDA의 `__reduce_add_sync` 류 또는 PTX `redux.sync`로 트리거.

**WARPSYNC**: 분기 실행한 lane들을 동기화해 실행 정렬, 후속 명령 효율을 보존.

## Uniform 명령

Uniform 레지스터는 warp 레벨이라 모든 lane이 같은 상태를 공유합니다. 그림 5에서 `UR1, UR2`는 lane 단위 범용 레지스터와 달리 warp 전체의 공통 상태를 표현합니다.

![Figure 5. Uniform 레지스터와 명령](images/v2-937422fe3d7aea650477b7e93365286c_1440w.jpg)
*Figure 5. Uniform Register and Uniform Instruction*

Uniform 명령은 제어 흐름과 보조 계산이지 핵심 계산은 아닙니다. NVIDIA는 정수·논리 관련 Uniform 명령만 제공하고, 부동소수는 없습니다. 사용자가 CUDA/PTX로 직접 트리거할 수 없고 컴파일러만 사용합니다(AMD에선 Scalar 레지스터에 해당). 컴파일러는 실행 흐름 분석 후 warp 일관 경로에 Uniform 레지스터·명령을 적용할지 자동 결정합니다.

Uniform 레지스터·명령을 활용하면 범용 레지스터 사용을 줄여 occupancy를 올리고 효율을 끌어올릴 수 있습니다. 흔히 보이는 Uniform 명령:

```
UFLO UIADD3 UIMAD UISETP ULDC ULEA ULOP3 UMOV UPLOP3 UPOPC UPRMT USEL USGXT USHF
```

이름은 정수·비트·논리 명령에 `U` 접두를 붙인 형태이며 의미는 동일합니다.

## 정리

SIMT 프로그래밍 체계와 자원 분할-실행 관계를 짚고, NVIDIA GPU의 warp 레벨 명령을 살펴봤습니다. warp 내 모든 lane이 협력해 데이터·연산을 표현하는 DMMA/HMMA/IMMA의 lane 분포를 자세히 봤고, warp 상태를 다루는 Uniform 레지스터·명령도 정리했습니다. SIMT 체계에서 Tensor 계산이 어떻게 협력하는지 이해하는 데 도움이 되길 바랍니다.

## 참고

- reed: NVIDIA GPU ISA - 부동소수 연산
- reed: NVIDIA GPU ISA - 레지스터
- reed: NVIDIA GPU ISA - 정수 연산
- https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functions
- TensorCore `ldmatrix` 명령의 장점은?
- reed: cute Swizzle
