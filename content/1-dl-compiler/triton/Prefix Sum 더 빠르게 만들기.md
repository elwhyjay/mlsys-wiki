> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역해 이 공개 계정에 게시합니다. 원문 주소: https://veitner.bearblog.dev/making-prefix-sum-really-fast/

# Prefix Sum 더 빠르게 만들기

2025년 4월 13일

이 블로그 글에서는 tiled prefix sum 작업을 최적화하는 방법을 보여줍니다. tiled prefix sum은 다음과 같이 동작합니다.

vector `v`가 주어지면, 이 vector를 여러 block으로 나눕니다. 각 block 내부에서 prefix sum 연산을 수행합니다. 간단한 예로 각 block에 4개 원소가 있고 입력 vector가 `v = [0, 1, 2, 3, 4, 5, 6, 7]`라면, 출력 vector는 `o = [0, 1, 3, 6, 4, 9, 15, 22]`가 됩니다.

이는 GPU에서 prefix sum 작업의 기본 building block입니다.

## 알고리즘

이 알고리즘을 이해하려면 slide 21(https://safari.ethz.ch/projects_and_seminars/fall2022/lib/exe/fetch.php?media=p_s-hetsys-fs2022-meeting10-aftermeeting.pdf)을 참고하세요. 이 그림은 우리가 실행하려는 알고리즘을 명확하게 설명합니다. 알고리즘은 여러 stage를 포함합니다. 각 stage에서 더하는 두 원소 사이의 stride를 두 배로 늘립니다. 마지막 stage가 끝나면 cumulative sum vector를 결과로 얻습니다.

## Naive baseline 구현

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_0(int *input, int *output) {
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  output[gtid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    int tmp;
    if (tid >= offset) {
      tmp = output[gtid - offset];
    }
    __syncthreads();

    if (tid >= offset && gtid < numElements) {
      output[gtid] += tmp;
    }
    __syncthreads();
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_0(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_0<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

이 알고리즘은 위 그림에 표시된 기능을 구현합니다. 각 stage에서 offset을 두 배로 늘립니다. thread block의 중간 위치에 도달하고, 누적 원소 사이의 거리가 thread block size의 절반이 될 때까지 계속합니다. race condition을 피하려면 `__syncthreads`를 사용해야 한다는 점에 주의하세요. 이 두 synchronization barrier가 없으면 두 array element가 동시에 읽히고 쓰이는 상황이 생길 수 있습니다.

간단한 CPU 구현으로 프로그램 correctness를 확인하는 것은 좋은 습관입니다.

```c++
emplate <int threadsPerBlock, int numElements>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i & (threadsPerBlock - 1)) == 0)) {
      output[i] = input[i] + output[i - 1];
    } else {
      output[i] = input[i];
    }
  }
}
```

이 알고리즘은 올바른 결과를 줍니다. 아쉽게도 성능은 이상적이지 않습니다. 전역 메모리에 자주 접근하기 때문입니다. bandwidth를 계산해 성능을 측정할 수 있습니다. 우리는 N번의 read/write 작업을 수행했고, 여기서 `N = 1 << 30 = 2**30`입니다. 위 kernel의 측정 성능은 다음과 같습니다.

```c++
Bandwidth: 823.944 GB/s
Efficiency: 0.24968
```

## Shared memory 사용

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_1(int *input, int *output) {
  extern __shared__ int buffer[threadsPerBlock];

  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  buffer[tid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    int tmp;
    if (tid >= offset) {
      tmp = buffer[tid - offset];
    }
    __syncthreads();

    if (tid >= offset && gtid < numElements) {
      buffer[tid] += tmp;
    }
    __syncthreads();
  }

  if (gtid < numElements) {
    output[gtid] = buffer[tid];
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_1(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_1<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

이 kernel은 위 kernel과 매우 비슷합니다. 주요 차이는 여기서 shared memory를 사용한다는 점입니다. 원소에 자주 접근해야 한다면 shared memory는 global memory보다 훨씬 저렴합니다. 성능은 다음과 같습니다.

```c++
Bandwidth: 1288.72 GB/s
Efficiency: 0.390522
```

## Double buffer 사용

```c++
template <int threadsPerBlock, int numElements>
__global__ void kernel_2(int *input, int *output) {
  __shared__ int _buffer_one[threadsPerBlock];
  __shared__ int _buffer_two[threadsPerBlock];

  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * threadsPerBlock + tid;

  int *buffer_one = _buffer_one;
  int *buffer_two = _buffer_two;

  buffer_one[tid] = input[gtid];
  __syncthreads();

#pragma unroll
  for (unsigned int offset = 1; offset <= threadsPerBlock / 2; offset <<= 1) {
    if (tid >= offset) {
      buffer_two[tid] = buffer_one[tid] + buffer_one[tid - offset];
    } else {
      buffer_two[tid] = buffer_one[tid];
    }
    __syncthreads();

    int *tmp = buffer_one;
    buffer_one = buffer_two;
    buffer_two = tmp;
  }

  if (gtid < numElements) {
    output[gtid] = buffer_one[tid];
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_2(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_2<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

이 kernel은 double buffer를 사용합니다. shared memory 안에 두 array를 초기화하고, 각 stage마다 buffer를 교환합니다. 이 방법의 장점은 synchronization barrier 하나를 아낄 수 있다는 것입니다. 이제 접근할 array가 두 개이므로 race condition이 발생하지 않도록 보장할 수 있기 때문입니다. 이 kernel의 성능은 다음과 같습니다.

```c++
Bandwidth: 1616.71 GB/s
Efficiency: 0.489913
```

## Warp primitive 사용

CUDA는 warp primitive를 제공합니다. 그중 하나가 `__shfl_up_sync`이며, 위 그림의 작업을 정확히 수행하므로 우리 연산에 매우 적합합니다. 이 블로그 글(https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)에서 더 자세히 알 수 있습니다. 이를 사용해 kernel 성능을 더 높일 수 있습니다.

```c++
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define WARP_MASK (WARP_SIZE - 1)
__device__ inline int lane_id(void) { return threadIdx.x & WARP_MASK; }
__device__ inline int warp_id(void) { return threadIdx.x >> LOG_WARP_SIZE; }
// Warp scan
__device__ __forceinline__ int warp_scan(int val) {
  int x = val;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, offset);
    if (lane_id() >= offset) x += y;
  }
  return x - val;
}

template <int threadsPerBlock>
__device__ int block_scan(int in) {
  __shared__ int sdata[threadsPerBlock >> LOG_WARP_SIZE];
  // A. Exclusive scan within each warp
  int warpPrefix = warp_scan(in);
  // B. Store in shared memory
  if (lane_id() == WARP_SIZE - 1) sdata[warp_id()] = warpPrefix + in;
  __syncthreads();
  // C. One warp scans in shared memory
  if (threadIdx.x < WARP_SIZE)
    sdata[threadIdx.x] = warp_scan(sdata[threadIdx.x]);
  __syncthreads();
  // D. Each thread calculates its final value
  int thread_out_element = warpPrefix + sdata[warp_id()];
  return thread_out_element;
}

template <int threadsPerBlock, int numElements>
__global__ void kernel_3(int *input, int *output) {
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int val = input[gtid];
  int result = block_scan<threadsPerBlock>(val);
  if (gtid < numElements) {
    output[gtid] = result + val;
  }
}

template <int threadsPerBlock, int numElements>
void launch_kernel_3(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  kernel_3<threadsPerBlock, numElements>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

이 kernel의 성능은 더 향상됩니다. warp primitive를 사용해 warp-level reductions를 매우 효율적으로 수행하기 때문입니다. 자세한 설명은 다음 영상(https://www.youtube.com/watch?v=SG0gvcbf2eo)을 참고하세요. 새 성능은 다음과 같습니다.

```c++
Bandwidth: 1976.42 GB/s
Efficiency: 0.598916
```

## thread당 작업량 늘리기

위 kernel들, 또는 적어도 그 변형들은 잘 알려져 있으며, 인터넷에서 많은 설명을 찾을 수 있습니다. 마지막 kernel은 잘 기록되어 있지 않습니다. 저는 이 간단한 기술에 대한 참고 자료를 온라인에서 찾지 못했지만, peak 성능에 가까워지는 핵심입니다.

주의: 이 기술은 thread coarsening이라고 불리며, GPU mode discord 서버의 ngc92가 알려주었습니다. PPMP 책에서 더 자세히 배울 수 있습니다(https://www.sciencedirect.com/science/article/abs/pii/B9780323912310000227).

```c++
template <int threadsPerBlock, int numElements, int batchSize>
__global__ void kernel_4(int *input, int *output) {
  int reductions[batchSize];
  int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_sum = 0;
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      total_sum += input[idx];
      reductions[i] = total_sum;
    }
  }
  int reduced_total_sum = block_scan<threadsPerBlock>(total_sum);
#pragma unroll
  for (int i = 0; i < batchSize; i++) {
    const int idx = gtid * batchSize + i;
    if (idx < numElements) {
      output[idx] = reduced_total_sum + reductions[i];
    }
  }
}

template <int threadsPerBlock, int numElements, int batchSize>
void launch_kernel_4(int *input, int *output) {
  const int numBlocks = (numElements + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  kernel_4<threadsPerBlock, numElements, batchSize>
      <<<numBlocks, threadsPerBlock>>>(input, output);
}
```

여기서 `block_scan`은 이전과 같습니다. 차이는 이제 thread마다 여러 원소를 처리한다는 점입니다.

다른 batchSize를 사용하도록 validation function을 조정합니다.

```c++
template <int threadsPerBlock, int numElements, int batchSize>
void cpu_scan(int *input, int *output) {
  output[0] = input[0];
  for (int i = 1; i < numElements; i++) {
    if (!((i % (threadsPerBlock * batchSize)) == 0)) {
      output[i] = input[i] + output[i - 1];
    } else {
      output[i] = input[i];
    }
  }
}
```

결과가 여전히 올바르다는 것을 보여줍니다. 참고로 batchSize가 2^n 형태라면 위의 bit operation을 사용해 modulo 연산을 수행할 수 있습니다.

우리는 현재 thread에 속한 원소들에 대해 먼저 간단한 sequential scan을 수행하는 방식으로 이를 구현합니다. 그런 다음 block이 이 합들에 대해 scan을 수행합니다. 이후 reduced sum을 reduction 부분에 더해 output을 씁니다. 이 과정은 위의 warp scan hierarchy와 전체 prefix sum을 위해 수행한 작업과 비슷합니다. 더 자세한 설명은 위 강의를 다시 참고하세요. 최종 kernel 성능은 다음과 같습니다.

```shell
Bandwidth: 3056.53 GB/s
Efficiency: 0.926221
```

block과 batchsize를 조정해 GPU 성능을 더 짜낼 수도 있지만, 블로그 글을 간결하게 유지하기 위해 여기서 멈춥니다. 예를 들어 batch data를 load할 때 더 적은 명령을 사용하도록 `int4`를 사용할 수도 있지만, 제 실험에서는 성능에 큰 영향을 주지 않았습니다. 성능을 더 높이는 추가 기법이 있다면 알려 주세요. 이 글을 재미있게 읽었기를 바랍니다. 위에서 언급한 강의(https://www.youtube.com/watch?v=SG0gvcbf2eo)는 매우 도움이 되었고 prefix sum을 더 잘 이해하게 해주었습니다. CUDA에 대해 더 이야기하고 싶다면 Linkedin(https://www.linkedin.com/in/simon-veitner-174a681b6/)으로 연락할 수 있습니다. 여러분의 생각을 듣고 싶습니다. 모든 코드는 제 github repo(https://github.com/simveit/effective_scan)에서 찾을 수 있습니다.
