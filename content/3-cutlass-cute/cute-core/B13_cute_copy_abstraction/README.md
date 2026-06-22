# CuTe의 Copy 추상

> 원문: https://zhuanlan.zhihu.com/p/666232173

앞서 CuTe의 **MMA 추상**을 다뤘습니다. MMA로 Tensor Core를 활용해 레지스터 상에서 `D = A × B + C`를 완성할 수 있습니다. GPU 프로그래밍에서 입력 데이터는 일반적으로 **global memory**에 저장되므로, global memory 데이터를 Tensor Core MMA 계산에 필요한 레지스터로 **효율적으로 옮기는 문제**가 생깁니다.

데이터 이동의 수학적 기술은 `D = S`입니다. D는 대상 Tensor(Destination), S는 원본 Tensor(Source)이며, 보통 서로 다른 저장 계층에 위치하고 각자의 Layout 기술을 가집니다. 본 글은 이 문제를 중심으로 **CuTe의 데이터 이동 관련 데이터 구조 추상**을 소개합니다. 구성: (1) CUDA GPU의 저장 계층과 행렬 계산 관련 `ldmatrix` 명령, (2) CuTe의 Copy 관련 데이터 구조·함수의 전체 추상과 관계, (3) 각 데이터 구조의 핵심 함수·멤버, (4) 정리.

## CUDA의 저장 계층과 데이터 로드 경로

![Figure 1. GPU 저장 계층과 데이터 이동 경로](images/v2-0429920b4041feaf68e0d10bbf94b119_1440w.jpg)

그림 1처럼 전형적인 GPU 저장 계층:

- **Off-chip**: global memory. 데이터센터급 A100에서 80GB, HBM2e, 최대 2TB/s
- **L2 Cache**: A100-80GB 기준 80MB, 최대 20TB/s
- **On-chip (SM 내)**: shared memory + L1 data cache. 둘이 192KB를 공유하며 크기 조정 가능, shared memory는 최대 164KB
- **Register File**: Tensor Core·CUDA Core에 가장 가까움. 계산 유닛은 레지스터에서만 데이터를 가져옴(Ampere 이전. Hopper의 Tensor Core는 shared memory 데이터를 직접 읽을 수 있음). GPU에서 가장 빠른 저장. 스레드당 최대 255개 32비트 레지스터

계산용 원본 데이터는 global memory에서 출발해 **세 경로** 중 하나로 Tensor Core/CUDA Core에 도달:

1. **경로 1**: global → L2 → shared memory(L1 bypass) → 레지스터
2. **경로 2**: global → L2 → L1 → shared memory → 레지스터
3. **경로 3**: global → L2 → L1 → 레지스터

경로 1·2는 Ampere 이후만 지원. Ampere 이전은 global → 레지스터는 경로 3만 가능하며, global → shared는 먼저 경로 3으로 레지스터에 로드 후 경로 4로 shared memory에 저장해야 했음.

사용자가 프로그래밍 제어할 수 있는 부분은 **global memory·shared memory·레지스터**. L1·L2는 캐시 구조로 bypass 여부를 제어 가능하며, PTX 명령 수정자로 L2 프리페치 동작도 제어 가능.

## 효율적인 ldmatrix 명령

행렬 계산 최적화의 핵심은 **데이터 블록화로 데이터 재사용**을 구현하는 것입니다. 재사용으로 저수준 저장 접근을 줄이고 접근 효율과 전체 성능을 향상. GPU에서 프로그래밍 가능한 재사용은 **shared memory** 영역에서 일어납니다. 사용자는 일부 데이터를 shared memory에 로드해 재사용하고, shared memory → 레지스터 이동을 구현하여 더 효율적 계산을 가능하게 합니다.

앞선 MMA 장에서 Tensor Core 기초를 다뤘습니다. Tensor Core 어셈블리 명령을 자세히 보면, **warp 내 참여 스레드는 행렬의 일부 데이터만 보유**하며 이 데이터는 스레드 전용 레지스터에 저장됩니다(SIMT 아키텍처에서 레지스터는 스레드 전용으로 간주). warp 내 모든 스레드의 레지스터가 모여 **행렬 계산 데이터 전체**를 구성합니다. 이는 NVIDIA가 SIMT에서 warp level Tensor Core 계산을 구현한 혁신적 실천입니다.

그림 2처럼, SIMT 관점에서 각 스레드가 두 원소(예: float16 두 개, 한 레지스터에 표현)를 보유하고, warp의 32 스레드가 총 64 원소를 모아 `8×8` warp-level 소행렬을 형성해 Tensor Core 계산에 공급합니다.

![Figure 2. SIMT 레지스터 협동으로 warp-level 행렬 구성](images/v2-cc05e3dc0eb99235fd060c2ff45aa860_1440w.jpg)

스레드들이 제공하는 레지스터로 warp-level 행렬 표현·저장을 완성하고, Tensor Core로 레지스터 상 고효율 행렬 계산이 가능합니다.

shared memory → 레지스터 로드는 SIMT 관점의 **LDS(load shared)** 로도 가능하지만, 데이터가 서로 다른 스레드의 레지스터에 분산되어 있어 연속성이 나쁩니다. 극한 성능을 위해 NVIDIA는 **Turing부터 `ldmatrix`** 명령을 제공합니다.

그림 3은 SIMT 방식 로드와 `ldmatrix` 협동 방식 로드의 비교입니다. `ldmatrix`는 스레드가 shared memory 주소(16B 데이터)를 제공하면 데이터를 warp 내 각 스레드의 레지스터로 분배해 **SIMT 레지스터 경계를 넘는 쓰기**를 실현합니다. SIMT 로드는 더 좁은 비트 폭만 가능해 같은 데이터량을 위해 더 많은 명령이 필요합니다. `ldmatrix`는 **warp 당 `16B × 32 = 512B`** 를 단일 명령으로 로드할 수 있어 `16×16 float16` 행렬을 한 명령으로 읽을 수 있습니다. 명령 수 감소·스케줄링 효율 향상에 더해 **전치(transpose) 결합 능력**도 제공합니다.

`ldmatrix`로 warp-level shared memory → 레지스터 이동을 구현할 수 있으며, CuTe는 이 이동에 대한 추상 능력을 자연스럽게 제공합니다.

![Figure 3. SIMT 로드 vs ldmatrix 협동 로드](images/v2-c1031c4aa65e40d119c601740b9afd1c_1440w.jpg)

## CuTe Copy 추상과 관계

MMA와 유사하게, CuTe는 데이터 이동에 대해 다음 구조·함수를 제공합니다: `CopyOperation`, `Copy_Traits`, `Copy_Atom`, `TiledCopy`, `ThrCopy`, `cute::copy`.

- **`CopyOperation`**: 명령 레벨 데이터 이동 캡슐화. NVIDIA는 아키텍처·저장 계층 조합마다 다른 명령(`ldmatrix`, `LDS`, Ampere의 `cp.async` 등)을 제공. 사용자는 HW 지원·대상 계층에 맞춰 이미 제공된 Operation을 선택
- **`Copy_Traits`**: MMA_Traits와 유사하게 `CopyOperation`이 제공하지 않지만 `Copy_Atom`이 필요한 정보를 채우는 **다리** 역할
- **`Copy_Atom`**: 명령 레벨 **분할 불가 이동 능력**
- **`TiledCopy`**: Copy_Atom을 반복해 더 큰 블록 이동 능력으로 확장(실행 스레드를 늘리거나 여러 번 복사)
- **`ThrCopy`**: TiledCopy는 **논리적 표현**이지만 커널 실행 시 CUDA 패러다임에 맞춰 **스레드 레벨 명령**으로 써야 함. ThrCopy는 `threadIdx.x`로 큰 Tensor를 분할해 현재 스레드가 `D = S` 이동을 위해 할 일을 얻음
- **`cute::copy`**: ThrCopy가 제공한 스레드 작업을 실제 이동 명령으로 트리거

![Figure 4. CuTe Copy 핵심 구조와 상호 관계](images/v2-6dd2070aa1e70515090e6956735c0a4c_1440w.jpg)

그림 4처럼 하드웨어·이동 방향 위에 `CopyOperation` 명령 추상을 제공하고, 그 위에 `D = S` 이동 논리 추상(`Copy_Atom` + `TiledCopy`)을, 그 위에 스레드별 작업을 분할하여 `cute::copy`로 트리거해 모든 스레드가 함께 Tensor → Tensor 이동을 완성합니다.

## CopyOperation

Operation은 **특정 HW가 지원하는 이동 능력**을 캡슐화합니다. 일반적으로 PTX(또는 CUDA)로 구현되며, 명령 세트의 이동 능력을 추상화하고 원본·대상 데이터 타입과 개수를 정의하며, 프레임워크 계층이 호출할 `copy` 함수를 제공합니다. 예시: 원본 레지스터가 `uint128_t`(128비트), 대상이 `uint32_t`:

```cpp
struct SM75_U32x1_LDSM_N {
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  void copy(uint128_t const& smem_src, uint32_t& dst) {
    asm volatile ("ldmatrix.sync. ...\n");
  }
};
```

## Copy_Traits

traits는 CopyOperation의 부족한 정보를 보충. 예를 들어 operation 수행에 필요한 스레드 수, 원본·대상 데이터 Layout 배치를 제공합니다. 스레드와 데이터의 저장 관계(스레드 번호·레지스터 번호 → 데이터 논리 위치)를 기술하며, 스레드 레벨 분할 시 retile 능력을 위한 `RefLayout`도 제공:

```cpp
struct Copy_Traits<SM75_U32x1_LDSM_N> {
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

## Copy_Atom

Atom은 Operation과 Traits를 캡슐화·추상화합니다. 내부 데이터 타입을 정의해 TiledCopy·ThrCopy 작업 분해 시 정보를 제공. Traits에서 스레드·Layout 정보를 상속받고, `call` 메서드로 하위 명령을 호출하는 진입점을 제공:

```cpp
struct Copy_Atom<Copy_Traits<Args...>, T>
  : Copy_Traits<Args...>
{
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = T;

  void call(Tensor<TS,SLayout> const& src, Tensor<TD,DLayout>& dst);
};
```

## TiledCopy

Tiled 추상은 Atom 능력을 반복해 **더 큰 블록 이동 능력**을 얻습니다. Atom 반복은 **스레드-저장 Layout**을 제공하는 방식, 또는 Atom 능력과 MMA의 `tiled_mma`를 결합한 방식(예: `make_tiled_copy_A/B/C`)으로 구현 가능합니다. MMA는 이미 `D = A × B + C` 계산에 필요한 데이터 분할 능력을 제공하므로 이를 활용. 단 이 함수들은 **레지스터 표현** 대상입니다.

TiledCopy의 핵심 함수는 `get_slice`·`get_thread_slice`로, 논리 Tensor 이동 능력을 스레드 id에 따라 각 스레드 Layout 기술 이동 작업으로 얻습니다. 반환 객체는 **ThrCopy**:

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord  [Need not be 2D...]
          class ShapeTile_MN>   // coord space
struct TiledCopy : Copy_Atom {
  ThrCopy get_slice(ThrIdx const& thr_idx);
  ThrCopy get_thread_slice(ThrIdx const& thr_idx);
};

CUTE_HOST_DEVICE
auto make_tiled_copy_A(Copy_Atom<Args...> const& copy_atom,
                  TiledMMA           const& tiled_mma)
```

## ThrCopy

ThrCopy는 **스레드 레벨 이동 추상**이며 `TiledCopy::get_slice`로 얻습니다. 핵심 함수는 `partition_S/D`·`retile_S/D` — S·D는 각각 source·destination.

- `partition`: 큰 논리 Tensor를 현재 스레드용으로 분할해 소스 Tensor·대상 Tensor를 얻음
- `retile`: 입력이 이미 현재 스레드의 private 데이터이지만 이동이 요구하는 형상에 맞지 않을 때, 이동이 지원하는 형상으로 변환

```cpp
template <class TiledCopy, class ThrIdx>
struct ThrCopy {
  auto partition_S(Tensor&& stensor);
  auto partition_D(Tensor&& dtensor);
  auto retile_S(Tensor&& stensor);
  auto retile_D(Tensor&& stensor);
};
```

## cute::copy

`copy` 함수는 **실제 이동 실행 함수**. 호출 시 스레드 레벨 이동이 발생하고, 스레드 명령 실행으로 src → dst 이동을 완료해 논리적 `D = S`를 구현합니다. 블록 단위 이동 시 경계 처리가 필요한 경우 `copy_if`로 일부 데이터 이동을 mask하여 불법 접근을 방지할 수 있습니다.

```cpp
void copy(TiledCopy const& copy, Tensor const& src, Tensor& dst);
void copy_if(TiledCopy const& copy, PrdTensor const& pred, Tensor const& src, Tensor& dst);
```

`cute::copy`와 다른 컴포넌트 관계:

| 기능 | 구성 요소 |
|---|---|
| Copy 명령 + 저장 타입 | `CopyOperation` |
| 논리 타입·형상 요구 | `Copy_Traits` |
| 원자 능력 | `Copy_Atom` |
| 블록 능력(여러 원자) | `TiledCopy` |
| 스레드 레벨 능력 | `ThrCopy` |
| 데이터 분할 API | `ThrCopy::partition_S/D()`, `ThrCopy::retile_S/D()` |
| 실행 트리거 | `cute::copy(tiled_copy, thr_s, thr_d)` |

## 정리

CuTe는 `D = S` Tensor 이동을 수행할 Copy 능력을 제공합니다. 명령·어댑터 계층·원자 능력·블록 능력·스레드 레벨 추상이 각각 `CopyOperation`·`Copy_Traits`·`Copy_Atom`·`TiledCopy`·`ThrCopy`·`cute::copy`로 형성됩니다. 이 추상들의 도움으로 **저수준 명령 세부에 과하게 신경 쓰지 않고** 저장 유닛 간 논리 Tensor 이동을 구현할 수 있습니다. 이 능력은 MMA와 함께 CuTe 행렬 곱의 기반을 구성합니다. Copy와 MMA 추상으로 **논리 계층에서 `D = S`와 `D = A × B + C`** 를 구성할 수 있습니다.

여기까지 CuTe의 **Layout·Tensor·MMA·Copy** 추상을 다뤘습니다. 이제 GEMM(General Matrix Multiplication) 문제를 풀 수 있습니다:

- Tensor·Layout으로 계산 행렬 블록 분할
- Copy로 블록 A·B 데이터를 global → 레지스터 로드
- MMA로 Tensor Core에서 레지스터 상 소행렬 곱 수행
- 다시 Copy로 레지스터 결과를 global에 쓰기 → 완전한 GEMM 완성

다음 글에서는 **Layout·Tensor·MMA·Copy 능력을 활용해 간단한 행렬 곱을 완성**합니다.

## 참고

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
- https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/copy.hpp
- tensorcore의 ldmatrix 명령 장점
