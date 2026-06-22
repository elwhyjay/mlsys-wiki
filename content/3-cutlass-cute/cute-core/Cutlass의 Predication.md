> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역해 이 공개 계정에 게시합니다. 원문 주소: https://veitner.bearblog.dev/predication-in-cutlass/

# Cutlass의 Predication

Cutlass 문서의 CuTe 부분은 predication(https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0y_predication.md)이라는 주제를 간단히 언급하지만, 완전한 코드 예시는 제공하지 않습니다. 이 블로그 글에서는 CuTe 프로그램에서 predication을 사용해 적절한 boundary check를 수행하는 방법을 설명합니다.

## 소개

우리는 CuTe tutorial(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/tiled_copy.cu)의 kernel 하나에서 시작합니다. 이 kernel은 효율적인 tiled copy를 수행합니다. predication이라는 주제를 논의하기 전에 먼저 non-vectorized version의 tiled copy를 간단히 보겠습니다.

```c++
/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  # SEE BELOW
}

/// Main function
int main(int argc, char** argv) {
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  int M = 32768;
  int N = 16384;

  auto tensor_shape = make_shape(M, N);

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  Tensor tensor_S =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                  make_layout(tensor_shape));
  Tensor tensor_D =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                  make_layout(tensor_shape));

  auto block_shape = make_shape(Int<256>{}, Int<128>{});

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape);  // ((M, N), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape);  // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout =
      make_layout(make_shape(Int<32>{}, Int<8>{}));  // (32,8) -> thr_idx

  dim3 gridDim(
      size<1>(tiled_tensor_D),
      size<2>(tiled_tensor_D));  // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  copy_kernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                     thr_layout);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
              << std::endl;
    return -1;
  }

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i
                << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}
```

이 예시는 cutlass repo의 CuTe tutorial에서 가져온 것입니다. 이것들은 main function에서 kernel을 호출하기 전에 수행하는 단계입니다. 아래에서 순서대로 설명하겠습니다.

```c++
  Tensor tensor_S =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())),
                  make_layout(tensor_shape));
  Tensor tensor_D =
      make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())),
                  make_layout(tensor_shape));
```

단순히 tensor를 초기화합니다.

```c++
  auto block_shape = make_shape(Int<256>{}, Int<128>{});

  
  if ((size<0>(tensor_shape) % size<0>(block_shape)) ||
      (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape."
              << std::endl;
    return -1;
  }

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape);  
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape);  
```

여기서는 tensor를 간단히 tile로 나눕니다. 이는 `(M,N) -> ((blkM, blkN), ceil(M/blkM), ceil(N/blkN)`로 변환합니다. 즉 초기 matrix를 shape가 `(blkM, blkN)`인 더 작은 matrix들로 나눕니다. shape의 마지막 두 차원은 x와 y 차원에서 만든 block 개수에 대응합니다. 위 예시에서는 `(32768, 16384) -> ((256, 128), 128, 128)`이 됩니다.

```c++
 Layout thr_layout =
      make_layout(make_shape(Int<32>{}, Int<8>{}));  

  dim3 gridDim(
      size<1>(tiled_tensor_D),
      size<2>(tiled_tensor_D));  
  dim3 blockDim(size(thr_layout));

  copy_kernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                     thr_layout);
```

여기서는 thread layout을 만듭니다. 이는 block을 kernel 안의 thread들로 다시 나누며, 다음과 같습니다. `(256, 128) -> (256/32, 128/8) = (8, 16)` 그 다음 tiled layout이 주는 block 수, 즉 x 방향 256개 block과 y 방향 128개 block, 그리고 각 block에 필요한 thread 수, 즉 `32 * 8 = 256`으로 kernel을 launch합니다.

predication이 없는 kernel은 다음과 같습니다.

```c++
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) {
  using namespace cute;
  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)

  Tensor fragment = make_tensor_like(thr_tile_S);  // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}
```

이 kernel은 전체 matrix block을 가져온 뒤 위에서 설명한 것처럼 local partition을 만듭니다. 그 다음 각 thread는 `GMEM -> RMEM -> GMEM`으로 이 원소들을 복사합니다. 이 과정은 매우 효율적이며 H100에서 `~3 TB/s` bandwidth를 달성할 수 있습니다. 서로 다른 block size를 조정해 이 값을 더 높일 수도 있지만, 이는 이 글의 초점이 아닙니다.

## 왜 predication이 필요한가?

차원이 `(M, N) = (32768 + 1, 16384 + 1)`인 matrix를 처리하고 싶다고 생각해 봅시다. 위 kernel은 동작하지 않습니다. 왜일까요? tiling 결과가 `(32768 + 1, 16384 + 1) -> ((256, 128), 129, 129)`가 되기 때문입니다. 문제는 마지막 block에서 복사해서는 안 되는 데이터를 복사하려 한다는 것입니다. x 방향 마지막 block에는 처리해야 할 원소가 하나뿐이고, y 방향 마지막 block도 마찬가지입니다. 이 block들의 전체 thread block을 복사하고 싶지는 않습니다. M과 N을 조정해 프로그램을 실행해 보면 다음과 같은 오류를 얻을 수 있습니다.

```c++
CUDA Runtime error: an illegal memory access was encountered
terminate called after throwing an instance of 'thrust::THRUST_200700_900_NS::system::system_error'
  what():  CUDA free failed: cudaErrorIllegalAddress: an illegal memory access was encountered
Aborted (core dumped)
```

CUDA kernel을 사용해 본 사람에게는 놀랍지 않을 것입니다. 우리는 자주 적절한 boundary check를 해야 합니다.

## CuTe로 predication 사용하기

이 글에서는 위 문제를 해결하는 kernel을 제시합니다.

```c++
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel_predicate(TensorS S, TensorD D, ThreadLayout, int M,
                                      int N) {
  using namespace cute;

  Tensor tile_S = S(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x,
                    blockIdx.y);  // (BlockShape_M, BlockShape_N)

  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{},
                                      threadIdx.x);  // (ThrValM, ThrValN)

  auto identity_tensor = make_identity_tensor(make_shape(
      size<0>(tile_S), size<1>(tile_S)));  // (BlockShape_M, BlockShape_N)
  auto thread_identity_tensor = local_partition(
      identity_tensor, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  Tensor fragment = make_tensor_like(thr_tile_S);  // (ThrValM, ThrValN)
  auto predicator = make_tensor<bool>(
      make_shape(size<0>(fragment), size<1>(fragment)));  // (ThrValM, ThrValN)

  CUTE_UNROLL
  for (int i = 0; i < size<0>(predicator); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(predicator); ++j) {
      auto thread_identity = thread_identity_tensor(i, j);
      int global_row = blockIdx.x * size<0>(tile_S) + get<0>(thread_identity);
      int global_col = blockIdx.y * size<1>(tile_S) + get<1>(thread_identity);
      predicator(i, j) = (global_row < M) && (global_col < N);
    }
  }

  // Copy from GMEM to RMEM and from RMEM to GMEM with predicate
  copy_if(predicator, thr_tile_S, fragment);
  copy_if(predicator, fragment, thr_tile_D);
}
```

이 kernel은 predication 없는 버전과 매우 비슷합니다. predication logic은 Lei Mao(https://leimao.github.io/article/CuTe-Matrix-Transpose/)에게서 빌려온 것입니다. 그는 matrix transpose 작업에서 비슷한 기법을 사용했습니다. 그의 블로그 글은 매우 잘 쓰였으므로 강력히 추천합니다. 이제 block tile 차원으로 나누어떨어지지 않는 matrix를 처리하려면 어떤 변경이 필요한지 설명하겠습니다.

```c++
  auto identity_tensor = make_identity_tensor(make_shape(
      size<0>(tile_S), size<1>(tile_S)));  // (BlockShape_M, BlockShape_N)
  auto thread_identity_tensor = local_partition(
      identity_tensor, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
```

복사할 tensor와 완전히 같은 tiling을 가진 identity tensor를 만듭니다. 이 identity tensor는 단순히 `(x,y)->(x,y)`로 mapping합니다.


```c++
  auto predicator = make_tensor<bool>(
      make_shape(size<0>(fragment), size<1>(fragment)));  // (ThrValM, ThrValN)
```

predication matrix를 초기화합니다. 이 matrix는 boundary `[0, M] x [0, N]` 안에 있는 모든 tuple `(x,y)`에 대해 1이 됩니다. 즉 문제의 원소가 matrix 안에 있을 때 1이 됩니다.


```c++
 CUTE_UNROLL
  for (int i = 0; i < size<0>(predicator); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(predicator); ++j) {
      auto thread_identity = thread_identity_tensor(i, j);
      int global_row = blockIdx.x * size<0>(tile_S) + get<0>(thread_identity);
      int global_col = blockIdx.y * size<1>(tile_S) + get<1>(thread_identity);
      predicator(i, j) = (global_row < M) && (global_col < N);
    }
  }
```

모든 thread block을 순회합니다. 각 tuple `(i, j)`에 대응하는 row와 column을 계산합니다. 이는 `blockIdx`에 해당 차원의 block tile 길이를 곱하고, thread tiling으로 생긴 offset을 더하면 간단히 구현할 수 있습니다.

```c++
  copy_if(predicator, thr_tile_S, fragment);
  copy_if(predicator, fragment, thr_tile_D);
```

이는 matrix boundary 안에 있는 원소만 복사합니다. kernel은 shape가 `(M, N) = (32768 + 1, 16384 + 1)`인 matrix도 기쁘게 복사하고, 오류 없이 올바른 결과를 줍니다. block tile 차원으로 나누어떨어지는 matrix의 경우 성능은 위 copy kernel과 비슷합니다. 제 추측으로는 compiler가 predication을 제거할 수 있다는 것을 인식하기 때문입니다. 나누어떨어지지 않는 matrix에서는 warp divergence 때문에 kernel이 약간 덜 최적화됩니다.

이 글이 CuTe의 predication을 더 잘 이해하는 데 도움이 되기를 바랍니다.
