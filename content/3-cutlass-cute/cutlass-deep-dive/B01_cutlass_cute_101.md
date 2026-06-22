# CUTLASS CuTe 101

> 원문: https://zhuanlan.zhihu.com/p/660379052

NVIDIA GPU의 기능이 점점 복잡해지면서 CUDA 작성도 어려워지고 있습니다. 이제 다들 CUTLASS 같은 공식 툴 라이브러리를 학습해야 할 때가 되었습니다. 최신 CUTLASS에서는 **CuTe**를 새로 제안했고, FlashAttention 2에서도 폭넓게 활용되고 있습니다. 다만 NVIDIA의 다른 툴과 마찬가지로 CuTe 공식 튜토리얼은 내용이 조각조각 흩어져 있어서, 본 글은 튜토리얼에서 충분히 설명되지 않은 부분을 조금 보충하며 CuTe의 기본 개념을 간단히 정리합니다. 오류가 있다면 지적 환영합니다.

공식 튜토리얼 권장(태그 v3.2.0 기준)처럼, 아래 GEMM 예제가 CuTe 학습의 출발점으로 적합합니다:
https://github.com/NVIDIA/cutlass/blob/v3.2.0/examples/cute/tutorial/sgemm_nt_1.cu

이 예제를 따라 설명하겠습니다.

코드를 보기 전에 CuTe의 핵심 개념 몇 가지를 짚고 갑니다: **tensor, layout, shape, stride**. 이들 사이에는 매우 단순한 두 가지 지칭 관계가 있습니다.

```
tensor = ptr + layout
layout = shape + stride
```

즉 **tensor**는 메모리 포인터(ptr, 실제로는 GPU 메모리지만 편의상 "메모리"로 통칭)와 그 포인터에 대응하는 **메모리 배치 정보(layout)** 의 조합입니다. **배치(layout)** 는 tensor의 **형상(shape)**, 즉 몇 차원이고 각 차원의 크기가 얼마이며 각 차원의 인접 원소 사이의 간격(**stride**)이 얼마인지를 뜻합니다.

간격이란? 예를 들어 2차원 tensor A의 형상이 M×N이라고 할 때 원소가 모두 연속 배치되어 있고 column-major라면, CuTe 표기로 shape은 `(M, N)`, stride는 `(1, M)` 입니다. 왜냐하면 같은 열에서 인접 행 사이의 거리는 1이고, 같은 행에서 인접 열 사이의 거리는 M이기 때문입니다.

예제 코드로 돌아오면, $C_{M \times N} = A_{M \times K} B_{K \times N}$ 행렬 곱을 구현합니다. M = N = 5120, K = 4096으로 하드코딩되어 있습니다. 알고리즘은 매우 일반적입니다. block size를 256(16×16)으로 하고, 각 thread가 C의 8×8 블록을 계산합니다. 즉 한 block이 128×128 C를 처리합니다. 구체 흐름:

- 레지스터에 8×8 누산기(accumulator) 할당
- A·B 일부를 global memory → shared memory로 복사
- shared memory의 A·B로 행렬 곱, 결과를 누산기에 누적
- 2·3을 128×128 관련 A·B 연산이 끝날 때까지 반복
- 누산기 결과를 global memory의 C에 가산

앞의 CuTe 개념들은 바로 이 "각 block·각 thread가 어떤 메모리에 접근할지"를 정의하는 데 사용됩니다.

A·B·C가 모두 column-major라면, 대응하는 CuTe tensor는 다음과 같습니다.

```cpp
// Represent the full tensors
auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), make_stride(Int<1>{}, M));  // (M,K)
auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), make_stride(Int<1>{}, N));  // (N,K)
auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), make_stride(Int<1>{}, M));  // (M,N)
```

> `Int<1>()`은 컴파일 타임에 확정되는 1을 나타냅니다.

한 block이 128×128 C를 처리하므로 M 128개와 N 128개를 처리합니다. K 차원에서 block이 매번 8개를 처리한다고 가정하면 다음 세 상수가 정해집니다.

```cpp
// Define block sizes (static)
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<  8>{};
```

그러면 kernel의 `dimBlock`·`dimGrid`도 정해집니다.

```cpp
dim3 dimBlock(size(tC));
dim3 dimGrid(ceil_div(size(M), size(bM)),
             ceil_div(size(N), size(bN)));
gemm_device<<< dimGrid, dimBlock, 0, stream >>>(...)
```

> `dimBlock`은 1차원이지만, CuTe 내장 indexing으로 2차원 효과를 발휘합니다.

`bM`·`bN`·`bK`는 "한 block의 한 번 루프에서 처리하는 A의 `bM × bK`, B의 `bN × bK`"를 의미합니다. A·B의 shared memory도 이 형상대로 할당하므로, 대응 CuTe tensor는:

```cpp
// Shared memory buffers
__shared__ TA smemA[bM * bK];
__shared__ TB smemB[bN * bK];
auto sA = make_tensor(make_smem_ptr(smemA), make_layout(make_shape(bM,bK)));  // (BLK_M,BLK_K)
auto sB = make_tensor(make_smem_ptr(smemB), make_layout(make_shape(bN,bK)));  // (BLK_N,BLK_K)
```

> shared memory는 먼저 수동으로 1차원으로 할당해야 하며, tensor의 기본 stride는 column-major입니다.

이제 한 block 안 thread들의 역할, 즉 256개 thread의 layout을 정의합니다. `bK=8`이므로 복사 시 이 차원에서 최대 8개 thread만 쓸 수 있습니다. A·B·C의 thread layout은:

```cpp
// Define the thread layouts (static)
auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));
auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
```

tensor와 layout이 정의되면, 몇몇 함수로 각 thread에 해당하는 메모리를 꺼낼 수 있습니다. 먼저 각 block의 global memory 영역:

```cpp
// Get the appropriate blocks for this thread block --
// potential for thread block locality
auto blk_shape = make_shape(bM, bN, bK);
auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);            // (m,n,k)

auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, _,_1>{});  // (BLK_M,BLK_K,k)
auto gB = local_tile(mB, blk_shape, blk_coord, Step< _,_1,_1>{});  // (BLK_N,BLK_K,k)
auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1,_1, _>{});  // (BLK_M,BLK_N)
```

여기는 조금 복잡하지만 실제로는 2단계입니다. 첫째 **`Step`이 작용**하여 `blk_shape`·`blk_coord`에서 `Step`이 `_`인 위치의 값을 제거합니다. 위 코드는 다음과 동치입니다.

```cpp
auto gA = local_tile(mA, make_shape(bM, bK), make_coord(blockIdx.x, _));  // (BLK_M,BLK_K,k)
auto gB = local_tile(mB, make_shape(bN, bK), make_coord(blockIdx.y, _));  // (BLK_N,BLK_K,k)
auto gC = local_tile(mC, make_shape(bM, bN), make_coord(blockIdx.x, blockIdx.y));  // (BLK_M,BLK_N)
```

그럼 나머지는 무엇을 의미할까요? `local_tile`은 먼저 tensor 형상을 여러 tile로 **reshape**한 다음, `coord`(coordinate) 위치의 tensor를 꺼냅니다.

구체적으로 첫 줄은 `mA`를 `(M, K)`에서 `(bM, bK, M/bM, K/bK)`로 reshape한 뒤 `mA[:, :, blockIdx.x, :]`를 취한 것과 같습니다(`coord`의 `_`는 파이썬의 `:`에 해당). 결과 `gA`의 형상은 `(bM, bK, K/bK)`입니다. 같은 식으로 `gB`는 `(bN, bK, K/bK)`, `gC`는 `(bM, bN)`이고, 모두 현재 block이 접근할 global memory 영역입니다.

> CuTe의 장점 하나는 이런 reshape·indexing을 따라 **stride를 자동 조정**하며 **실제 메모리 복사는 전혀 일어나지 않는다**는 점입니다. `gA`·`gB`·`gC`의 stride를 계산해 공식 튜토리얼과 일치하는지 확인해보세요.

이제 각 thread가 접근할 영역을 봅니다. 여기서는 `local_tile`과 유사한 `local_partition` API를 사용합니다.

```cpp
auto tAgA = local_partition(gA, tA, threadIdx.x);  // (THR_M,THR_K,k)
auto tAsA = local_partition(sA, tA, threadIdx.x);  // (THR_M,THR_K)

auto tBgB = local_partition(gB, tB, threadIdx.x);  // (THR_N,THR_K,k)
auto tBsB = local_partition(sB, tB, threadIdx.x);  // (THR_N,THR_K)
```

`local_tile`과 유사하게 먼저 변환이 일어납니다. `threadIdx.x`를 `tA`·`tB`에 맞춰 2차원 coord로 조정합니다(이 변환된 인덱스를 `(x, y)`라고 부릅시다). 그 다음 `local_partition`은 두 번째 인자로 reshape한 tile을 취하되, 이번엔 **tile 위에서** 인덱스를 적용합니다.

구체적으로 첫 줄은 `gA`를 `(bM, bK, K/bK)`에서 `(tA[0], tA[1], bM/tA[0], bK/tA[1], K/bK)`로 reshape한 뒤 `gA[x, y, :, :, :]`를 취해 `tAgA`의 형상은 `(bM/tA[0], bK/tA[1], K/bK)`가 됩니다. 이는 **현재 thread가 접근할 global memory**입니다. 유사하게 `tAsA`는 `(bM/tA[0], bK/tA[1])`, `tBgB`는 `(bN/tB[0], bK/tB[1], K/bK)`, `tBsB`는 `(bN/tB[0], bK/tB[1])`입니다. `tXgX`는 thread가 접근할 global memory, `tXsX`는 thread가 접근할 shared memory입니다.

각 thread의 C 접근에도 유사한 기법이 쓰입니다.

```cpp
// Partition sA (M,K) by the rows of tC
auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});  // (THR_M,BLK_K)
// Partition sB (N,K) by the cols of tC
auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});  // (THR_N,BLK_K)
// Partition gC (M,N) by the tile of tC
auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});  // (THR_M,THR_N)
```

여러분도 직접 유도해보면 좋습니다. 각 thread가 얼마만큼의 C에 접근할까요? 특히 `tCgC`의 형상은 8×8이 맞을까요?

누산기 할당에는 새로운 API가 쓰입니다.

```cpp
// Allocate the accumulators -- same size as the projected data
auto tCrC = make_fragment_like(tCgC);  // (THR_M,THR_N)
```

여기서는 `TC a[8x8]` 같은 로컬 배열을 할당해 레지스터를 바로 쓸 수 있게 합니다.

메모리 사용 방식이 명확해지면, 남은 루프에서 CuTe의 `gemm` 함수를 호출하는 부분은 매우 단순해지므로 여기서는 자세히 다루지 않습니다.

이상으로 GEMM 예제를 통해 CuTe의 메모리 관련 기본 개념과 사용법을 간단히 훑어보았습니다. CuTe 학습에 도움이 되기를 바랍니다.
