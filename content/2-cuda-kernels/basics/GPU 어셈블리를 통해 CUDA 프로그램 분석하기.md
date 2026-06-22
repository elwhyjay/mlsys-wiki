> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역하여 이 public account에 게시한 것이다. 원문 주소: https://veitner.bearblog.dev/analyze-cuda-programs-by-looking-at-gpu-assembly/

# GPU Assembly를 보며 CUDA 프로그램 분석하기

2025년 4월 21일

이 글은 SASS code를 분석해 memory-bound CUDA program의 성능을 높이는 방법에 관한 짧은 note다.

## 간단한 vector copy

vector를 copy하는 다음 두 program을 살펴보자.

```c++
#define threadsPerBlock 1024

__global__ void vectorCopy(float *input, float *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < N) {
    output[i] = input[i];
  }
}

__global__ void vectorCopyVectorized(float4 *input, float4 *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < (N >> 2)) {
    output[i] = input[i];
  }
}
```

결과를 보면 $N = 1 << 30$ 일 때 vectorized version의 성능은 $\frac{3.94361ms-2.802ms}{2.802ms} = 41\%$ 더 빠르다.

## 분석

그 이유를 이해하려면 각 kernel의 SASS code를 보는 것이 도움이 된다. godbolt(https://godbolt.org/)를 사용하거나 NVIDIA NCU tool로 kernel을 분석해 SASS code를 얻을 수 있다. godbolt에서는 적절한 nvcc version을 선택하고 compiler command parameter를 `-arch sm_90 -use_fast_math -O3`처럼 설정해야 한다. 적절한 architecture를 고르는 것이 중요하다.

그 다음 SASS code를 볼 수 있다. NVIDIA document(https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm)는 SASS를 low-level assembly language로 설명한다. 이는 binary microcode로 compile되어 NVIDIA GPU hardware에서 native하게 실행된다.

아래는 H100에서 이 두 kernel의 SASS code다.

```shell
vectorCopy(float*, float*, int):

 LDC R1, c[0x0][0x28] 

 S2R R7, SR_TID.X 

 ULDC UR4, c[0x0][0x220] 

 S2R R0, SR_CTAID.X 

 LEA R7, R0, R7, 0xa 

 ISETP.GE.AND P0, PT, R7, UR4, PT 

 @P0 EXIT 

 LDC.64 R2, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R4, c[0x0][0x218] 

 IMAD.WIDE R2, R7, 0x4, R2 

 LDG.E R3, desc[UR4][R2.64] 

 IMAD.WIDE R4, R7, 0x4, R4 

 STG.E desc[UR4][R4.64], R3 

 EXIT 

```

```shell
vectorCopyVectorized(float4*, float4*, int):

 LDC R1, c[0x0][0x28] 

 S2R R7, SR_TID.X 

 ULDC UR4, c[0x0][0x220] 

 USHF.R.S32.HI UR4, URZ, 0x2, UR4 

 S2R R0, SR_CTAID.X 

 LEA R7, R0, R7, 0xa 

 ISETP.GE.AND P0, PT, R7, UR4, PT 

 @P0 EXIT 

 LDC.64 R4, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R2, c[0x0][0x218] 

 IMAD.WIDE R4, R7, 0x10, R4 

 LDG.E.128 R8, desc[UR4][R4.64] 

 IMAD.WIDE R2, R7, 0x10, R2 

 STG.E.128 desc[UR4][R2.64], R8 

 EXIT 
```

vectorized version에는 instruction이 하나 더 있다. 이는 `N / 4 = N >> 2`를 계산하기 위해 bit shift를 수행하기 때문이다. kernel에 `N/4`를 전달하면 이 instruction(`USHF.R.S32.HI UR4, URZ, 0x2, UR4`)을 최적화해 제거할 수 있다. 그러면 instruction 하나가 줄지만 큰 차이는 없다. 따라서 이어지는 분석에서는 이 shift operation을 무시한다.

logic에서 흥미로운 부분, 즉 copy를 수행하는 부분은 다음과 같다.

```shell
LDC.64 R2, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R4, c[0x0][0x218] 

 IMAD.WIDE R2, R7, 0x4, R2 

 LDG.E R3, desc[UR4][R2.64] 

 IMAD.WIDE R4, R7, 0x4, R4 

 STG.E desc[UR4][R4.64], R3
```

vectorized version은 다음과 같다.

```shell
LDC.64 R4, c[0x0][0x210] 

 ULDC.64 UR4, c[0x0][0x208] 

 LDC.64 R2, c[0x0][0x218] 

 IMAD.WIDE R4, R7, 0x10, R4 

 LDG.E.128 R8, desc[UR4][R4.64] 

 IMAD.WIDE R2, R7, 0x10, R2 

 STG.E.128 desc[UR4][R2.64], R8 
```

`LDG.E/STG.E` 대신 `LDG.E.128/STG.E.128`을 사용함으로써 32bit가 아니라 128bit를 load/store한다는 것을 볼 수 있다. 이는 같은 수의 instruction이 필요하지만 필요한 block 수는 훨씬 적다는 뜻이다.

이를 이해하기 위해 다음을 비교해보자.

```c++
template <int threadsPerBlock>
__global__ void vectorCopy(float *input, float *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < N) {
    output[i] = input[i];
  }
}

template <int threadsPerBlock>
void launchVectorCopy(float *input, float *output, int N) {
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorCopy<threadsPerBlock>
      <<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
```

그리고:

```c++
template <int threadsPerBlock>
__global__ void vectorCopyVectorized(float4 *input, float4 *output, int N) {
  const int i = threadIdx.x + blockIdx.x * threadsPerBlock;

  if (i < (N >> 2)) {
    output[i] = input[i];
  }
}

template <int threadsPerBlock>
void launchVectorCopyVectorized(float *input, float *output, int N) {
  const int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
  vectorCopyVectorized<threadsPerBlock><<<blocksPerGrid, threadsPerBlock>>>(
      reinterpret_cast<float4 *>(input), reinterpret_cast<float4 *>(output), N);
}

```

`N = 1 << 30`과 `threadsPerBlock = 1 << 10`을 취하면, 첫 번째 version에서는 `1048576`개의 block을 launch하고, 두 번째 version에서는 `262144`개의 block을 launch한다. shift instruction의 cost를 무시하거나 위에서 말한 것처럼 단순히 제거하면, 두 번째 kernel이 훨씬 빠른 이유를 이해할 수 있다. 훨씬 적은 instruction을 실행하기 때문이다. 즉 non-vectorized version에서 load하는 block 수의 일부만 launch한다.

이 blog post가 vectorized load와 store를 더 잘 이해하는 데 도움이 되기를 바란다. code는 내 github(https://github.com/simveit/vector_copy_vectorized)에서 찾을 수 있다. makefile에는 NVIDIA Nsight Compute에서 사용하는 kernel analysis command도 포함되어 있다. 이 blog post가 마음에 들었다면 Linkedin에서 나와 연결할 수 있다. 나는 CUDA와 일반적인 MLSys에 관한 생각을 교류하는 것을 좋아한다.


