# Tensor-015 CUDA Green Context

- 원문 제목: CUDA Green Context
- 저자: ZhaB
- 계정: zartbot
- 발행일: 2025년 5월 7일 11:12

CUDA Green Context[1]는 기존 Context의 lightweight alternative다. PaaS/MaaS에서 여러 model inference를 수행하거나 일부 operator를 유연하게 호출할 때 장점이 있을 수 있다. 하지만 공식 문서에는 example code가 없어 보인다. 그래서 오늘 아침 A10 card 하나를 찾아 잠깐 test해 보았다.

## 1. 기존 Context

먼저 test용 kernel 몇 개를 만들고 cubin file을 생성해 서로 다른 Context가 load할 수 있게 한다. kernel test code는 다음과 같다:

```c++
//kernel_1.cu
#include <cuda_runtime.h>
#include <iostream>

const int SIZE_N = 32 *512;

__global__ void kernel_1(float *data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float v = data[idx];
    int i = 0;
    for (int j = 0; i < 10240; ++j)
    {
        i = j % SIZE_N;
        v += logf(data[idx+i]);
        data[idx+i] = v;
    }
}

__global__ void kernel_2(float *data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float v = data[idx];
    int i = 0;
    for (int j = 0; i < 10240; ++j)
    {
        i = j % SIZE_N;
        v += logf(data[idx+i]);
        data[idx+i] = v;
    }
}

__global__ void kernel_3(float *data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float v = data[idx];
    int i = 0;
    for (int j = 0; i < 10240; ++j)
    {
        i = j % SIZE_N;
        v += expf(data[idx+i]);
        data[idx+i] = v;
    }
}

__global__ void kernel_4(float *data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float v = data[idx];
    int i = 0;
    for (int j = 0; i < 10240; ++j)
    {
        i = j % SIZE_N;

        v += data[idx+i];
        data[idx+i] = v;
    }
    data[idx] = v;
}
```

cubin을 생성하고 function signature를 얻는다.

```c++
 nvcc -arch=sm_86 -ptx kernel_1.cu -o kernel_1.ptx
 nvcc -arch=sm_86  kernel_1.ptx -cubin -o kernel_1.cubin

 more kernel_1.ptx  | grep ".globl"
        // .globl       _Z8kernel_1Pf
        // .globl       _Z8kernel_2Pf
        // .globl       _Z8kernel_3Pf
        // .globl       _Z8kernel_4Pf
```

기존 context 사용 방식은 다음과 같다:

```c++
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

const int GRID_SIZE = 32;
const int BLOCK_SIZE = 512;
const int CTX_NUM = 4;

#define CHECK_CUDA(func)                                                  \
    {                                                                     \
        CUresult status = (func);                                         \
        if (status != CUDA_SUCCESS)                                       \
        {                                                                 \
            std::printf("CUDA API failed at line %d with error:  (%d)\n", \
                        __LINE__, status);                                \
            return EXIT_FAILURE;                                          \
        }                                                                 \
    }

int main()
{
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    CUmodule module[CTX_NUM];
    CUfunction kernel[CTX_NUM];
    CUcontext ctx[CTX_NUM];
    CUstream stream[CTX_NUM];
    float *data[CTX_NUM];
    const char *func_name[CTX_NUM] = {"_Z8kernel_1Pf", "_Z8kernel_2Pf", "_Z8kernel_3Pf", "_Z8kernel_4Pf"};

    for (int i = 0; i < CTX_NUM; ++i)
    {
        //Context 생성
        cuCtxCreate(&ctx[i], 0, dev);
        cuCtxSetCurrent(ctx[i]);
        //Module load
        cuModuleLoad(&module[i], "kernel_1.cubin");

        //function 가져오기
        CHECK_CUDA(cuModuleGetFunction(&kernel[i], module[i], func_name[i]));

        cudaMalloc(&data[i], sizeof(float) * GRID_SIZE * BLOCK_SIZE);

        //cudastream 생성
        cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < CTX_NUM; ++i)
    {
        //cudastream을 통해 kernel launch
        void *kernelParams[] = {(void *)&data[i]};
        cuLaunchKernel((CUfunction)kernel[i], GRID_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, stream[i], kernelParams, 0);
    }
    for (int i = 0; i < CTX_NUM; ++i)
    {
        cuStreamSynchronize(stream[i]);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    printf("Normal Elapsed: %5.3lf ms\n", msec);
}
```

실행 시간

```c++
nvcc -arch=sm_86  -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcuda  ctx.cu

Normal Elapsed: 9.172 ms
```

## 2. Green Context

Green Context 생성은 상대적으로 조금 복잡하다. 코드는 다음과 같다:

```c++
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

const int GRID_SIZE = 32;
const int BLOCK_SIZE = 512;
const int CTX_NUM = 4;

#define CHECK_CUDA(func)                                                  \
    {                                                                     \
        CUresult status = (func);                                         \
        if (status != CUDA_SUCCESS)                                       \
        {                                                                 \
            std::printf("CUDA API failed at line %d with error:  (%d)\n", \
                        __LINE__, status);                                \
            return EXIT_FAILURE;                                          \
        }                                                                 \
    }

int main()
{
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    /*
    (1) Start with an initial set of resources, for example via cuDeviceGetDevResource. Only SM type is supported today.
    (2) Partition this set of resources by providing them as input to a partition API, for example: cuDevSmResourceSplitByCount.
    (3) Finalize the specification of resources by creating a descriptor via cuDevResourceGenerateDesc.
    (4) Provision the resources and create a green context via cuGreenCtxCreate.
    */

    CUdevResource resource;
    cuDeviceGetDevResource(dev, &resource, CU_DEV_RESOURCE_TYPE_SM);

    //SM resource를 최소 80% 점유
    unsigned int minCount;
    minCount = (unsigned int)((float)resource.sm.smCount * 0.8f);

    unsigned int split_group = CTX_NUM;

    //SM resource 기반 allocation
    CUdevResource split_resource[CTX_NUM];
    cuDevSmResourceSplitByCount(split_resource, &split_group, &resource, 0, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, minCount);

    //resource descriptor 생성
    CUdevResourceDesc split_desc[CTX_NUM];
    cuDevResourceGenerateDesc(split_desc, split_resource, split_group);

    CUgreenCtx gctx[CTX_NUM];
    CUstream gstream[CTX_NUM];

    CUmodule module[CTX_NUM];
    CUfunction kernel[CTX_NUM];
    float *data[CTX_NUM];
    const char *func_name[CTX_NUM] = {"_Z8kernel_1Pf", "_Z8kernel_2Pf", "_Z8kernel_3Pf", "_Z8kernel_4Pf"};

    for (int i = 0; i < CTX_NUM; ++i)
    {
        //resource descriptor에 따라 green context와 cuda stream을 생성하고 Module/Function load
        cuGreenCtxCreate(&gctx[i], split_desc[i], dev, CU_GREEN_CTX_DEFAULT_STREAM);
        cuGreenCtxStreamCreate(&gstream[i], gctx[i], CU_STREAM_NON_BLOCKING, 0);

        //SetCurrentContext 전에 Green context를 기존 Context로 변환해야 함
        CUcontext ctx;
        cuCtxFromGreenCtx (&ctx,gctx[i]);
        cuCtxSetCurrent(ctx);

        //Module load
        cuModuleLoad(&module[i], "kernel_1.cubin");
        CHECK_CUDA(cuModuleGetFunction(&kernel[i], module[i], func_name[i]));
        cudaMalloc(&data[i], sizeof(float) * GRID_SIZE * BLOCK_SIZE);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < CTX_NUM; ++i)
    {
        //green context에 대응하는 cuda stream을 사용해 kernel 호출
        void *kernelParams[] = {(void *)&data[i]};
        cuLaunchKernel((CUfunction)kernel[i], GRID_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, gstream[i], kernelParams, 0);
    }
    for (int i = 0; i < CTX_NUM; ++i)
    {
        cuStreamSynchronize(gstream[i]);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec;
    cudaEventElapsedTime(&msec, start, end);

    printf("Green Context Elapsed: %5.3lf ms\n", msec);
}
```

test 실행 시간:

```c++
vcc -arch=sm_86  -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lcuda  green_ctx.cu
Green Context Elapsed: 3.254 ms
```

Normal Context 대비 성능이 3배 빨랐다...

참고 자료

[1]

CUDA Green Context: *https://docs.nvidia.com/cuda/cuda-driver-api/group\_\_CUDA\_\_GREEN\_\_CONTEXTS.html*
