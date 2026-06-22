# CuTe의 Tensor

> 원문: https://zhuanlan.zhihu.com/p/663093816

앞선 글들은 CuTe Layout과 그 대수·기하 해석을 다뤘습니다. Layout은 데이터 배치와 저장 위치 관계를 기술하지만 **저장 자체는 지정하지 않습니다**. Tensor는 Layout 위에 저장을 포함한 것 — 즉 **Tensor = Layout + storage**. 저장은 포인터가 가리키는 데이터이거나 스택 상의 데이터(GPU에선 레지스터)일 수 있습니다.

CuTe의 Tensor는 딥러닝 프레임워크(PyTorch·TensorFlow 등)의 Tensor와 다릅니다. 딥러닝 프레임워크의 Tensor는 **데이터 실체**를 강조하며 Tensor 간 연산으로 새 데이터 실체를 생성합니다. 반면 CuTe의 Tensor는 **Tensor의 분해·조합** 같은 작업이 주이고 이런 작업은 대개 **Layout 변환**(논리 층위의 데이터 조직 형식)입니다. 저수준 데이터 실체는 보통 변경되지 않습니다. 즉 딥러닝 프레임워크의 Tensor는 Tensor로 새 Tensor를 만들지만, CuTe의 Tensor는 **데이터 표현 형식의 변환**이고 저수준 데이터는 보통 불변 — 표현 형식만 Layout 연산으로 바뀝니다. 프레임워크 Tensor는 데이터 실체, CuTe Tensor는 **기술 실체**에 가깝습니다.

본 글은 CuTe Tensor의 자주 쓰는 메서드와 벡터 합 예제로 사용법을 보여주고, Tensor의 특성·사용을 정리합니다.

## Tensor의 생성

```cpp
// 스택 객체: 타입과 Layout을 동시에 지정, layout은 반드시 static shape
Tensor make_tensor<T>(Layout layout);

// 힙 객체: pointer와 Layout 지정, layout은 동적·정적 모두 가능
Tensor make_tensor(Pointer pointer, Layout layout);

// 스택 객체, tensor의 layout은 반드시 static
Tensor make_tensor_like(Tensor tensor);

// 스택 객체, tensor의 layout은 반드시 static
Tensor make_fragment_like(Tensor tensor);
```

두 가지 주요 생성 방식: **(1) 스택 객체**(위 형식 1), **(2) 힙 객체**(pointer 지정). pointer는 `make_gmem_ptr`·`make_smem_ptr` 등으로 생성합니다. 스택 객체는 반드시 static이며, 힙 객체는 동적·정적 모두 가능. 동적 스택 구조는 존재하지 않습니다.

| | Static | Dynamic |
|---|---|---|
| **Heap (non-owning)** | `make_tensor(ptr, make_shape(Int<M>{}, Int<N>{}))`, `make_tensor_like(tensor)`, `make_fragment_like(tensor)` | `make_tensor(ptr, make_shape(M, N))` |
| **Stack (owning)** | `make_tensor(make_shape(Int<M>{}, Int<N>{}))`, `make_tensor_like(tensor)`, `make_fragment_like(tensor)` | NA |

## 차원 정보 조회

전역 `size` 함수로 tensor 원소 수를 얻을 수 있으며 정수 타입과 비교 연산도 지원. 멤버 함수·전역 함수로 각 속성 조회:

```cpp
// 멤버 함수
Tensor::layout();
Tensor::shape();
Tensor::stride();
Tensor::size();

// 전역 함수, 전체 정보 또는 <>로 특정 차원
auto cute::layout<>(Tensor tensor);
auto cute::shape<>(Tensor tensor);
auto cute::stride<>(Tensor tensor);
auto cute::size<>(Tensor tensor);
auto cute::rank<>(Tensor tensor); // (1, (2, 3)) => rank 2
auto cute::depth<>(Tensor tensor);
```

## Tensor 접근 `operator()` / `operator[]`

괄호 연산자로 데이터 읽기·쓰기 접근 가능. 좌표는 1차원 또는 계층 표현:

```cpp
Tensor tensor = make_tensor(ptr, layout);
auto coord = make_coord(irow, icol);

tensor(0) = 1;
tensor(1, 2) = 100;
tensor(coord) = 200;
```

`data` 함수로 저장 공간 주소를 직접 얻을 수도 있습니다.

```cpp
Tensor::data();
```

## Tensor의 Slice

`_`로 특정 축을 선택. Layout 표현과 동일:

```cpp
Tensor tensor = make_tensor(ptr, make_shape(M, N, K)); // MxNxK
Tensor tensor1 = tensor(_, _, 3); // MxN, k=3
```

## Tensor의 Take

`take` 함수로 `[B, E)` 범위 축의 데이터를 추출:

```cpp
Tensor tensor = make_tensor(ptr, make_shape(M, N, K));
Tensor tensor1 = take<0, 1>(tensor);
```

## Tensor의 flatten

계층 layout을 한 층으로 펼침(M·N·K를 MNK로 펼치는 것이 아님):

```cpp
Tensor tensor = ...;  // M, N, K
Tensor tensor1 = flatten(tensor);  // M, N, K
```

## Tensor의 층 병합 coalesce

계층에서 공간상 연속 가능한 좌표(stride상 틈이 없는)가 있으면 병합:

```cpp
Tensor tensor = make_tensor(ptr, make_shape(M, N));
Tensor tensor1 = coalesce(tensor);
```

## 주 축 계층화 group_modes

`[B, E)` 구간의 주 축을 새 계층으로 묶음:

```cpp
Tensor tensor = ...; // 1, 2, 3, 4
Tensor tensor1 = group_modes<B, E>(tensor); // B=1, E=3 => 1, (2, 3), 4
```

## Tensor 분할 logical_divide / tiled_divide / zipped_divide

`divide`는 tile 크기에 따라 tensor를 분할. 자세한 의미는 앞선 [《CuTe Layout의 대수와 기하 해석》](../B10_cute_layout_algebra_geometry/README.md)의 나눗셈 설명 참고.

```cpp
Tensor tensor = ...;
Tensor tensor1 = logical_divide(tensor, tile);
Tensor tensor2 = zipped_divide(tensor, tile);
Tensor tensor3 = tiled_divide(tensor, tile);
```

## Layout의 곱 logical / zipped / tiled / blocked / raked

Tensor에는 곱이 정의되지 않고 **Layout에만** 정의됨. "반복"을 표현:

```cpp
Layout layout = ...;
Tile tile = ...;
Layout tensor1 = logical_product(layout, tile);
Layout tensor2 = zipped_product(layout, tile);
Layout tensor3 = tiled_product(layout, tile);
Layout tensor4 = blocked_product(layout, tile);
Layout tensor5 = raked_product(layout, tile);
```

## Tensor의 국소 블록 local_tile

사용자가 자주 쓰는 중요 함수. `tile`로 tensor를 분할하고 `coord`로 블록을 선택. 아래는 MNK 텐서를 `2×3×4` 블록으로 분할하고 `(1, 2, 3)` 블록을 취하는 예:

```cpp
Tensor tensor = make_tensor(ptr, make_shape(M, N, K));
Tensor tensor1 = local_tile(tensor, make_shape(2, 3, 4), make_coord(1, 2, 3));
```

![Figure 1. local tile의 기하 해석](images/v2-97d7483ee5a4ca0d5fb5657d0754c194_1440w.jpg)

그림 1처럼 A Tensor는 4행 6열 row-major. tile `2×2`, `local_tile`이 A를 tile 단위로 분할한 후 `(1, 1)` 블록을 선택해 우하단 결과를 얻습니다.

## Tensor의 국소 데이터 추출 local_partition

`local_partition`은 `local_tile`과 유사. 먼저 Tensor를 tile 크기로 분할한 뒤, **각 블록에서 coordinate가 지정하는 원소만 추출**해 새 블록을 구성합니다. 그림 2 참고.

```cpp
Tensor tensor = make_tensor(ptr, ...);
Tensor tensor1 = local_partition(tensor, tile);
```

![Figure 2. local partition의 기하 해석](images/v2-940f54c0095cfc2860f772d985cbb726_1440w.jpg)

## 데이터 타입 변환 recast

Tensor가 특정 타입 데이터를 표현할 때, 이를 **재해석**해 새 tensor를 형성. C++ `reinterpret_cast`와 유사:

```cpp
Tensor tensor = make_tensor<float>(make_shape(...));
Tensor tensor1 = recast<NewType>(tensor);
```

## 내용 채우기 fill / 지우기 clear

`clear()`·`fill()`로 원소 단위 채움·비움:

```cpp
Tensor tensor = make_tensor(...);

clear(tensor);  // T{} 기본 생성자로 할당
fill(tensor, value);  // value로 채움
```

## 선형 조합 axpby

두 tensor의 `y = a*x + b*y` 선형 조합:

```cpp
Tensor x = make_tensor(...);
Tensor y = make_tensor_like(x);

axpby(a, x, b, y);
```

## 출력 print

전역 `print`로 tensor 디버깅·표시 가능. `print`는 저장 위치·shape·stride를, `print_tensor`는 추가로 각 값까지 출력:

```cpp
Tensor tensor = make_tensor(...);
print(tensor);
print_tensor(tensor);
```

## 특수 행렬

형상만 있고 타입은 없는 tensor 구성. 특정 변환용:

```cpp
Tensor tensor = make_identity_tensor(shape);
```

## Tensor로 Vector Add 구현 예제

위의 Tensor 메서드들을 복습했으니, half 타입 `z = a*x + b*y + c` 벡터 연산을 구현하는 CUDA 커널을 예로 봅니다.

CUDA 경험이 풍부하면 아래 최적화 수단으로 효율적 구현이 가능합니다:

- **한 스레드가 여러 데이터 처리** — 데이터 프리페치·명령 병렬로 로드 효율과 파이프라인 향상
- **global memory를 대용량 단위로 읽기·쓰기** — I/O 명령 수 감소, 스케줄 오버헤드 감소
- **Half2 타입 사용** — half 타입 PRMT 변환·오버헤드 감소
- **FMA(fused multiply-add)** — FMUL·FADD 명령 수 감소, 정밀도 향상

구체적으로 아래와 같이 구현합니다.

```cpp
// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) { // 비정렬 처리는 생략
    return;
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // 두 개의 hfma2 명령
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}
```

코드 해설:

- **template 행**: 컴파일 타임 상수로 스레드당 8 원소 처리. 런타임 상수는 레지스터 주소 지정 불가 → Local Memory 문제 야기. 스레드당 8 원소, `sizeof(half)=2` → 스레드당 `8×2=16B` 데이터. 이 크기는 **LDG.128** 한 명령으로 global → 레지스터 로드 가능
- **`__global__`** 로 CUDA 커널임을 선언. `const` 등으로 입출력 힌트
- **`using`**: CuTe 네임스페이스 도입
- **`idx` / `if`**: `threadIdx`·`blockIdx`·`blockDim`으로 스레드 위치 결정
- **`tz`·`tx`·`ty`**: `make_tensor`로 커널 인자의 raw pointer와 shape 정보를 tensor로 포장
- **`tzr`·`txr`·`tyr`**: `local_tile`로 블록 분할·선택(`idx`). 이후 국소 tensor만 신경쓰면 됨. `Int<>{}`로 shape를 컴파일 타임 상수화 → 런타임 상수로 인한 Local Memory 방지
- **`txR`·`tyR`·`tzR`**: `make_tensor_like`로 스택 tensor 정의(GPU에선 레지스터)
- **`copy`**: CuTe `copy` 함수로 global → 레지스터 로드 → **LDG.128** 생성
- **`half2 a2, b2, c2`**: 계수 a·b·c를 반복해 half2로 구성 → **HFMA2** 명령 활용
- **`recast`**: 연속된 half → half2로 변환 → 더 효율적인 HFMA2
- **`#pragma unroll` + for**: `z = a*x + b*y + c` 계산. **괄호**로 두 HFMA2 명령으로 표현하도록 강제. 괄호 없으면 `HMUL2 + HMUL2 + HADD2 + HADD2`로 컴파일됨(곱셈은 결합법칙 성립 안 함, IEEE가 코드 작성 순서대로 계산 강제)
- 마지막으로 결과를 half로 cast 백, `copy`로 global memory에 저장

**주의할 점**: `Tensor tz;`·`Tensor tzr;` 등의 행은 Tensor 생성처럼 보여도 **실제 global memory 읽기·쓰기는 일어나지 않습니다**(복사 없음). Layout으로 tensor를 표현·변환할 뿐 데이터 실체는 이동하지 않음. **실제 데이터 읽기·쓰기는 `copy` 시점**에만 발생합니다. 이것이 도입부에서 말한 "CuTe의 Tensor와 딥러닝 프레임워크의 Tensor가 다르다"의 의미입니다. 대부분의 경우 CuTe에서 Tensor는 **논리적 의미와 변환**을 사용할 뿐 실질적 데이터 이동을 트리거하지 않습니다.

또한 Tensor 의미와 도구로 로직을 더 형상화해 사고를 돕지만, **CUDA 최적화의 사고·기교가 Tensor 도입으로 쉬워지거나 어려워지지는 않습니다**. Tensor는 표현 도구일 뿐, 깊이 있는 최적화는 여전히 경험의 도전입니다.

## 정리와 논의

본 글은 CuTe Tensor의 자주 쓰는 메서드를 소개하고 벡터 곱·합 연산으로 사용 예를 보였습니다. Tensor는 데이터 표현에 대한 훌륭한 추상으로, 사고 모델을 더 추상화·단순화해 프로그래밍 난도를 낮춥니다. CuTe의 Tensor는 **표현과 변환**이 주이고, Tensor로 Tensor를 생성하는 것은 Layout 표현의 변환일 뿐 데이터 실체 이동을 일으키지 않습니다(예: `local_tile`). Tensor 표현은 편의를 주지만 **표현의 효율·편의**에 국한되며, 실제 프로그램 최적화에 대한 추가 통찰은 제공하지 않습니다. 최적화는 여전히 다른 경로로 얻어야 합니다.

그럼에도 **"표현과 추상"은 지극히 중요**합니다. Galois가 '군'이라는 표현 도구 없이는 다항식 근 문제를 풀기 어려웠고, 양진녕이 '군'이 없었다면 양–밀스 이론을 구상하기 어려웠던 것처럼 말입니다.

추상과 도구는 우리가 더 높은 차원에서 사고하게 해줍니다. 이어지는 글에서는 Tensor(Layout)라는 추상 위에 **MMA(matrix multiply accumulate)** 와 **COPY**를 소개합니다.
