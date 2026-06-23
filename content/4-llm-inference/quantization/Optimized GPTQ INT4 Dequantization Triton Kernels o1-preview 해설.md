# Optimized GPTQ INT4 Dequantization Triton Kernels o1-preview 해설

## 0x0. 서문

[번역: GPU에서 GPTQ Triton dequantization kernel을 어떻게 가속할까](https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg)에서 PyTorch official은 GPTQ INT4 dequantization Triton Kernels를 최적화하는 일련의 방법을 제시했습니다. 예를 들면 L2 Cache(Block swizzled), vectorized read, SplitK optimization으로 Warp Stalling을 개선하는 방법입니다. 여기서는 현재 가장 advanced한 o1-preview model을 사용해 이 Triton code implementation을 다시 분석해 보고, 현재 가장 advanced한 model이 Triton kernel을 읽는 능력이 어떤지 살펴보겠습니다.

## 0x1. Prerequisite

위 Blog 외에도, 제가 예전에 Triton MatMul tutorial을 공부할 때 여기서 언급한 L2 Cache optimization도 학습했고, 그 내용을 [BBuf의 CUDA 노트 13, OpenAI Triton 입문 노트 1](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ) 글에 기록했습니다. 이 Block swizzled가 matrix multiplication에 어떤 영향을 주는지 깊이 이해하고 싶다면 읽어 보세요.

[CUDA-MODE 강의 노트 7강: Quantization Cuda vs Triton](https://mp.weixin.qq.com/s/1gCgpp49NF7sDw__EpO-nw)이라는 CUDA-MODE 학습 노트에서는 PyTorch Core Developer가 large model quantization 방법을 비교적 포괄적으로 설명하고, CUDA/Triton으로 이런 인기 있는 quantization kernel을 개발할 때의 장단점을 비교했습니다. LLM quantization background가 부족한 독자도 이 노트를 읽어 볼 수 있습니다.

## 0x2. Code link

Triton으로 구현한 GPTQ INT4 dequantization Kernel code link: https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py

## 0x3. o1-preview-128k 기반 code analysis

가장 advanced한 o1-preview-128k model로 이 Triton code를 분석해 봅니다. [번역: GPU에서 GPTQ Triton dequantization kernel을 어떻게 가속할까](https://mp.weixin.qq.com/s/CX6lPJOVYRPlpFS_WbGbmg)에서 언급한 세 가지 핵심 optimization point를 짚을 수 있는지 보겠습니다.

prompt는 다음과 같습니다.

```shell
아래 code implementation을 자세히 분석해 주세요. 모든 줄에 comment를 달아야 하며, 한국어로 답변해 주세요. +code...
```

o1-preview-128k가 준 답변을 붙여 보겠습니다.

```python
# PyTorch library import
import torch
# Triton library import
import triton
# Triton에서 language module을 import하고 tl로 줄여 쓴다.
from triton import language as tl

# @triton.jit decorator를 사용해 function을 Triton kernel로 compile한다.
@triton.jit()
def swizzle_tile(pid,
                 m, n,
                 block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    # M direction의 grid size를 ceil division으로 계산
    grid_m = tl.cdiv(m, block_m)
    # N direction의 grid size를 ceil division으로 계산
    grid_n = tl.cdiv(n, block_n)

    # 각 group의 width, 즉 group 안 program count 계산
    width = group_m * grid_n
    # current group ID 계산. pid는 program의 global ID
    group_id = pid // width
    # current group size 계산. out of bounds 방지
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    # current program의 M dimension block index pid_m 계산
    pid_m = group_id * group_m + (pid % group_size)
    # current program의 N dimension block index pid_n 계산
    pid_n = (pid % width) // group_size

    # 후속 계산에 사용할 pid_m과 pid_n 반환
    return pid_m, pid_n

# K dimension blocked computation(Split-K)을 지원하는 matrix multiplication Triton kernel 정의
@triton.jit()
def matmul_split_k_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          stride_scales_g, stride_scales_n,
                          stride_zeros_g, stride_zeros_n,
                          groupsize,
                          m, n, k,
                          block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
                          group_m: tl.constexpr, split_k: tl.constexpr):
    # 첫 번째 dimension(M*N dimension)의 current program ID를 얻는다.
    pid = tl.program_id(0)
    # K dimension의 program ID를 얻는다.
    pid_k = tl.program_id(1)
    # K dimension 전체 block 수를 ceil division으로 계산
    total_blocks_k = tl.cdiv(k, block_k * split_k)

    # custom swizzle_tile function으로 current program에 대응하는 block index 계산
    pid_m, pid_n = swizzle_tile(pid,
                                m, n,
                                block_m, block_n, group_m)

    # current program의 M, N, K dimension element offset 계산
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)

    # offs_m과 offs_n의 contiguous/alignment를 보장
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    # matrix A와 matrix B의 current block pointer 계산
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    # scales와 zeros pointer 계산
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    # compressed representation에서 실제 value를 추출하기 위한 shift amount 계산
    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    # accumulator를 0으로 initialize. shape는 (block_m, block_n), dtype은 float32
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    # K dimension의 모든 block을 순회
    for k in range(0, total_blocks_k):
        # global memory에서 matrix A와 matrix B의 current block을 load
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # 대응하는 scales와 zeros를 얻기 위한 current group ID 계산
        g_id = (k * split_k + pid_k) // (groupsize // block_k)

        # 대응하는 scales load
        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        # 대응하는 zeros load
        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        # zeros 처리, 실제 offset value 추출
        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        # matrix B 처리, 실제 value를 추출하고 dequantize
        b = (b >> shifter[:, None]) & 0xF
        b = b * scales[None, :] - zeros[None, :]

        # matrix multiplication 수행 및 result accumulate
        acc += tl.dot(a, b)
        # pointer update, 다음 block data로 이동
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k // 8) * split_k * stride_bk

    # accumulator dtype을 float16으로 변환
    acc = acc.to(tl.float16)

    # output matrix C의 element offset 재계산
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)

    # output matrix C의 pointer position 계산
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # atomic add로 computation result를 global memory의 matrix C에 write
    tl.atomic_add(c_ptrs, acc, sem='release')

# matrix multiplication wrapper function 정의. parameter 설정 후 kernel 호출
def matmul_split_k(a, b, scales, zeros):
    # matrix A shape, m rows k columns
    m, k = a.shape
    # matrix B shape, _ rows n columns
    _, n = b.shape

    # quantization group size
    quant_groupsize = 128
    # block size 정의
    block_m = 16
    block_n = 32
    block_k = 128
    # group마다 M dimension block count
    group_m = 8
    # stage count와 warp count, performance optimization용
    num_stages = 3
    num_warps = 4
    # K dimension split count
    split_k = 4

    # total block count를 ceil division으로 계산
    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    # M*N dimension의 total program count 계산
    total_programs_mn = total_blocks_m * total_blocks_n
    # K dimension의 program count
    total_programs_k = split_k

    # kernel grid size 정의
    grid = (total_programs_mn, total_programs_k)

    # problem scale과 block information 출력
    print(f"problem m size: {m}, tile size m: {block_m}, total blocks m: {total_blocks_m}")
    print(f"problem n size: {n}, tile size n: {block_n}, total blocks n: {total_blocks_n}")
    print(f"problem k size: {k}, tile size k: {block_k}, total thread blocks k: {split_k}")

    # total thread block count 출력
    print(f"total thread blocks k: {k}, total thread blocks m and total thread blocks n = {total_blocks_m=} x {total_blocks_n} = {total_programs_mn}")
    print(f"{total_programs_mn=}, {total_programs_k=}")

    # output matrix C initialize, dtype은 float16
    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    # Triton kernel function 호출. parameter와 kernel execution config 전달
    k = matmul_split_k_kernel[grid](a, b, c, scales, zeros,
                                    a.stride(0), a.stride(1),
                                    b.stride(0), b.stride(1),
                                    c.stride(0), c.stride(1),
                                    scales.stride(0), scales.stride(1),
                                    zeros.stride(0), zeros.stride(1),
                                    quant_groupsize,
                                    m, n, k,
                                    block_m, block_n, block_k,
                                    group_m, split_k, num_stages=num_stages, num_warps=num_warps)

    # kernel performance information 출력. register usage, spills, shared memory usage 포함
    print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    # kernel assembly code 등 정보를 text file에 write해 debugging과 analysis에 사용
    with open('matmul_split_k.txt', 'w') as f:
        print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        print("IR", k.asm['ttir'], file=f)
        print("TTGIR", k.asm['ttgir'], file=f)
        print("PTX", k.asm['ptx'], file=f)
        print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)

    # computation result matrix C 반환
    return c

# specified shape와 dtype의 tensor를 생성하는 helper function 정의
def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # dtype이 int32이면 random integer로 tensor를 채운다.
        res = torch.randint(low=-2147483648, high=2147483647, size=(M, N), dtype=dtype, device="cuda")
    else:
        # 그렇지 않으면 tensor를 만들고 normal distribution random number로 채운다.
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    # generated tensor 반환
    return res

# main function, program entry point
if __name__ == '__main__':
    # matrix size 정의
    m = 16
    k = 4096
    n = 4096
    # quantization group size 정의
    groupsize = 128
    # quantization group count 계산
    g = k // groupsize

    # matrix A 생성, shape (m, k), dtype float16
    a = make_tensor(m, k, dtype=torch.float16)
    # matrix B 생성, shape (k//8, n), dtype int32
    b = make_tensor(k // 8, n, dtype=torch.int32)
    # matrix C 생성, shape (m, n), dtype float16
    c = make_tensor(m, n, dtype=torch.float16)
    # zeros와 scales 생성. quantization/dequantization에 사용
    zeros = make_tensor(g, n // 8, torch.int32)
    scales = make_tensor(g, n, torch.float16)

    # matrix multiplication function 호출해 result 계산
    split_k_output = matmul_split_k(a, b, scales, zeros)
    # computation result shape와 일부 data 출력
    print(f"{split_k_output.shape=}, {split_k_output[0][0:4]}")
```

code flow 자체에서는 큰 문제를 찾기 어렵습니다. 하지만 `swizzle_tile`이라는 optimization level이 높은 trick에 대해서는 AI가 아직 제대로 이해하지 못합니다. 이 function 설명은 위에서 생성된 comment를 무시하고 [BBuf의 CUDA 노트 13, OpenAI Triton 입문 노트 1](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ)을 참고하세요. 그리고 code에서 다소 뜬금없어 보이는 부분은 아래 몇 줄입니다. INT32->INT4와 INT4->FP16 dequantization이 관련됩니다.

```python
zeros = (zeros >> zeros_shifter) & 0xF
zeros = (zeros + 1) * scales

b = (b >> shifter[:, None]) & 0xF
b = b * scales[None, :] - zeros[None, :]
```

o1-preview-128k에게 이 몇 줄을 어떻게 이해해야 하는지, 자세한 설명을 요청했습니다.

![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/001.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/002.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/003.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/004.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/005.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/006.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/007.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/008.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/009.png)

o1-preview-128k는 이 몇 줄의 code를 완전히 이해했고, 그 뒤의 mathematical principle도 정확히 복원할 수 있었습니다. 매우 훌륭합니다.

다음으로 vectorized read optimization을 o1-preview-128k가 제대로 이해할 수 있는지 보겠습니다.

![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/010.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/011.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/012.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/013.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/014.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/015.png)
![](img/optimized-gptq-int4-dequantization-triton-kernels-o1-preview-analysis-2ca28258/016.png)

o1-preview-128k는 이 optimization을 완전히 이해했습니다. 또한 예시와 diagram까지 들어 vectorized read 원리를 설명하고, address calculation을 단순화할 수 있다는 점도 지적했습니다.

## 0x4. 정리

위에서 보듯 L2 Cache, vectorized read, SplitK 측면에서 o1-preview-128k model은 이런 optimization의 역할을 이해할 수 있습니다. 다만 L2 Cache optimization 측면에서 o1-preview-128k model이 준 설명만으로는 이 Block swizzle 원리를 완전히 이해했다고 말할 수 없습니다. 이 optimization은 여전히 Triton documentation이나 [BBuf의 CUDA 노트 13, OpenAI Triton 입문 노트 1](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ)을 참고해 이해해야 합니다. 전반적으로 우리는 large model을 사용해 code를 더 잘 읽고 뒤의 원리를 탐구하는 데 도움을 받을 수 있으며, 이는 확실히 productivity revolution이라고 할 만합니다. 최근 Cursor의 인기도 이를 보여줍니다. 하지만 특히 전문 영역의 code에서는 가장 좋은 code reading experience를 얻기 위해 여전히 가장 advanced한 large model이 필요합니다. 관심 있는 독자는 다른 large model이 위 code를 어떻게 설명하는지도 시도해 볼 수 있습니다.
