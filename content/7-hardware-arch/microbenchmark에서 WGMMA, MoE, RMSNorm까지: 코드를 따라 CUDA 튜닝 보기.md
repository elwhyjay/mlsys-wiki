> 이 글은 GPT5.4 Medium Thinking이 생성했다

# microbenchmark에서 WGMMA, MoE, RMSNorm까지: 코드를 따라 CUDA 튜닝 보기

여기의 자료는 주로 네 부분으로 이루어진다. `YHs_Sample` 안의 microbenchmark와 손으로 작성한 `GEMM`, `mma_vs_wgmma.cu`, `sgl-kernel`의 추론 operator 묶음, 그리고 `hp_rms_norm`이다.

## 먼저 머신을 정확히 파악하라. 그렇지 않으면 뒤에서 쉽게 감으로 튜닝하게 된다

나는 `YHs_Sample/cuda/microbenchmark` 이 코드 묶음을 꽤 좋아한다. 목표가 아주 단순하기 때문이다. 먼저 머신을 측정하고, 비즈니스 kernel 작성은 서두르지 않는다. 이 순서는 사실 많은 사람이 생각하는 것보다 더 중요하다. DRAM, L2, shared memory 각각의 경계조차 감이 없으면, 뒤에서 profiler를 볼 때 대개 느낌으로만 추측하게 된다.

예를 들어 `dram_bandwidth.cu`는 대충 손으로 바로 칠 수 있는 평범한 copy kernel을 쓰지 않고, 벡터화된 `ld.global.cs.v4.b32`와 `st.global.cs.v4.b32`를 직접 사용한다.

```cpp
__device__ __forceinline__
uint4 ldg_cs(const void *ptr) {
    uint4 ret;
    asm volatile (
        "ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
        : "l"(ptr)
    );
    return ret;
}

__device__ __forceinline__
void stg_cs(const uint4 &reg, void *ptr) {
    asm volatile (
        "st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
        : : "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr)
    );
}
```

여기서 `v4.b32`는 아주 직관적이다. 한 번에 16바이트를 옮겨 memory access를 최대한 그럴듯하게 만든다. `.cs`도 그냥 붙인 것이 아니다. cache 오염을 최대한 줄이려는 것이다. 쉽게 말해 이 benchmark가 측정하려는 것은 cache가 도와서 보기 좋게 만든 결과가 아니라, "streaming global memory access"에 비교적 가까운 경로다.

host 쪽 구성도 문제를 잘 보여 준다.

```cpp
const int MEMORY_OFFSET = (1u << 20) * 16;
const int BENCH_ITER = 100;
...
cudaMalloc(&ws, size_in_byte + MEMORY_OFFSET * BENCH_ITER);
...
for (int i = BENCH_ITER - 1; i >= 0; --i) {
    read_kernel<BLOCK, LDG_UNROLL><<<grid, BLOCK>>>(ws + i * MEMORY_OFFSET, nullptr);
}
```

이 부분에서 가장 중요한 것은 `MEMORY_OFFSET * BENCH_ITER`이다. 각 반복이 의도적으로 서로 다른 주소 구간을 치게 해서 같은 cache line이 반복적으로 hit되는 일을 줄인다. 다시 말해 실제로 DRAM 상한 쪽에 가까워지려는 것이다. `copy_kernel`에서 `grid / 2`를 쓰는 것도 마찬가지다. 작성 방식이 이상해서가 아니라 copy는 읽기와 쓰기 두 경로를 동시에 차지하므로 주소도 나누어야 하기 때문이다.

바꿔 말하면 이 benchmark는 단순히 "대역폭을 한번 측정"하는 것이 아니라, 읽기, 쓰기, 복사 세 경로를 의도적으로 분리한다. 이 습관은 중요하다. 많은 사람이 benchmark를 만들 때 마지막에 총 숫자 하나만 보고하는 것을 좋아하지만, 그 숫자는 대개 어떤 문제도 설명하지 못한다.

`l2cache_latency.cu`는 다른 사고방식을 따른다. 여기서 관심 있는 것은 throughput이 아니라 한 번의 access가 정확히 몇 cycle이 걸리는가이므로, 진짜 dependency chain을 구성해야 한다.

```cpp
template <int ROUND>
__global__ __launch_bounds__(32, 1)
void l2_latency_kernel(const uint32_t *stride,
                       uint32_t *ret,
                       uint32_t *clk) {
    const char *ldg_ptr = reinterpret_cast<const char *>(stride + threadIdx.x);
    uint32_t val;

    asm volatile (
        "ld.global.cg.b32 %0, [%1];\n"
        : "=r"(val)
        : "l"(ldg_ptr)
        : "memory"
    );

    ldg_ptr += val;
    ...
    for (int i = 0; i < ROUND; ++i) {
        asm volatile (
            "ld.global.cg.b32 %0, [%1];\n"
            : "=r"(val)
            : "l"(ldg_ptr)
            : "memory"
        );
        ldg_ptr += val;
    }
}
```

이 코드에서 `.cg`, `ldg_ptr += val`, 앞쪽의 warmup은 사실 아주 구체적인 세 가지 일을 한다.

- access를 최대한 global cache 경로로 제한한다.
- 매 load가 이전 결과에 의존하게 보장해서 하드웨어가 latency를 숨길 여지를 주지 않는다.
- 먼저 TLB와 cache를 데운 뒤 steady state를 측정한다.

내 생각에 이런 코드에서 가장 가치 있는 부분은 마지막에 `l2 cache latency xxx cycles` 한 줄을 출력한다는 점이 아니라, 질문을 어떻게 던져야 하는지 가르쳐 준다는 점이다. 대역폭과 지연은 같은 것이 아니고, throughput과 단일 비용도 같은 것이 아니다. 먼저 질문을 제대로 던지지 않으면, 뒤에서 kernel을 튜닝할 때 현상을 거꾸로 보기 쉽다.

shared memory 쪽 두 benchmark도 마찬가지다. `smem_latency.cu`는 dependency chain 위의 shared latency를 측정하고, `smem_bandwidth.cu`는 `st.shared.v4.b32` 같은 경로가 throughput을 얼마나 높게 밀어 올릴 수 있는지 측정한다. 여기까지 오면 사실 아주 실용적인 판단 프레임워크가 이미 생긴다.

어떤 kernel의 DRAM throughput이 이런 benchmark보다 훨씬 낮다면, 보통 Tensor Core를 먼저 탓할 필요는 없다. 대부분 access 방식, coalescing 정도, 또는 중간 데이터 이동에 문제가 생긴 것이다. shared memory 명령이 많은데 throughput이 여전히 올라오지 않는다면, 그때는 bank, thread map, LSU issue, barrier를 보는 편이 보통 더 얻는 것이 많다.

## 손으로 작성한 GEMM에서 진짜 어려운 것은 언제나 FMA 몇 줄이 아니다

`YHs_Sample/cuda/gemm/sgemm.cu`는 아주 전형적인 구식 고성능 `FP32 GEMM`이다. 나는 오히려 이런 코드가 CUDA를 이해하는 데 더 도움이 된다고 생각한다. 문법적 sugar가 많지 않고, 문제가 전부 눈앞에 펼쳐져 있기 때문이다. tile을 어떻게 나눌지, shared memory를 어떻게 배치할지, double buffering을 어떻게 돌릴지, register pressure를 어떻게 제어할지가 모두 드러난다.

주 kernel은 이렇게 생겼다.

```cpp
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                             const float *B,
                             float *C,
                             uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             uint32_t A_ldg_step,
                             uint32_t B_ldg_step) {
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    ...
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    ...
}
```

내가 처음 이런 kernel을 봤을 때는 항상 FMA loop에 주의를 빼앗겼다. 나중에야 진짜 봐야 할 것은 바깥의 그다지 "눈에 띄지 않는" 것들이라는 사실을 알게 되었다.

`__launch_bounds__(256, 2)`는 먼저 제약을 하나 건다. block은 256개 thread이고, 작성자는 각 SM이 적어도 두 block의 동시성을 유지하기를 바란다. 이는 사실 이미 register 예산을 암시한다. 이어서 `24KB` shared memory는 아무렇게나 맞춘 것이 아니다. `16KB A + 8KB B`가 `128x128x8` 이 tile과 대응된다. 더 안쪽을 보면 `A_frag[2][8]`, `B_frag[2][8]`가 double buffering을 register 레벨까지 직접 가져온다.

또 이 두 줄이 있다.

```cpp
A_sts_addr ^= 0x2000;
B_sts_addr ^= 0x1000;
```

이런 작성법은 조금 딱딱해 보이지만 매우 대표적이다. shared memory layout이 일단 규칙적으로 정리되면 buffer 전환은 한 번의 XOR로 퇴화할 수 있다. 겉보기에는 integer 명령 몇 개를 아낀 것에 불과하지만, 실제로는 한 가지 사실을 말한다. kernel이 이미 이 정도까지 작성되면 주소 갱신 자체도 따로 고려할 가치가 있다.

계산 부분은 오히려 매우 소박하다.

```cpp
for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
        C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
    }
}
```

바로 이렇게 소박하기 때문에 GEMM에서 가장 까다로운 부분이 multiply-add가 아니라 "다음 라운드에서 쓸 데이터를 어떻게 미리 제자리에 보내 둘 것인가"라는 점을 더 잘 볼 수 있다. 많은 경우 코드를 한참 작성한 뒤의 성능 차이는 main loop 몇 줄에서 나오지 않고, shared memory layout, thread map, buffer 전환, writeback 재배열 같은 더 자잘한 부분에서 나온다.

## `cp.async`가 진짜 바꾼 것은 kernel을 작성하는 사고방식이다

`ampere_sgemm.cu`를 앞의 코드와 나란히 놓고 보면 차이가 매우 분명하다. 여기서는 더 이상 전통적으로 global에서 register로 load한 뒤 shared memory에 store하는 방식이 아니라, global에서 shared로의 이동을 `cp.async`에 직접 맡긴다.

```cpp
__global__ __launch_bounds__(256)
void ampere_sgemm_128x256x8_kernel(
        const float *A,
        const float *B,
        float *C,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t B_ldg_step) {
    __shared__ __align__(16 * 1024) char smem[32 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);
    ...
}
```

여기서 block tile은 이미 `128x256x8`까지 커졌고, shared memory도 `32KB`가 되었다. 이는 단순히 tile을 키운 것이 아니라, `cp.async`가 더 깊은 prefetch pipeline을 가치 있게 만들었기 때문이다.

감싼 그 명령이 중요하다.

```cpp
asm volatile (
    "{.reg .pred p;\n"
    " setp.ne.b32 p, %2, 0;\n"
    " @p cp.async.ca.shared.global.L2::128B [%0], [%1], 4;}\n"
    : : "r"(smem_addr), "l"(gmem_ptr), "r"((int)guard)
);
```

이것은 적어도 세 가지를 설명한다. 첫째, global에서 shared로의 이동이 더 이상 일반 register를 우회하지 않는다. 둘째, 이 경로가 이미 L2 sector granularity와 뚜렷하게 맞춰지기 시작했다. 셋째, boundary handling도 async copy 안으로 눌러 넣어 별도의 느린 경로를 다시 나눌 필요가 없다.

그래서 `Ampere` 이후 많은 kernel의 핵심 문제는 더 이상 "shared memory에 double buffering이 필요한가"가 아니라, "이동 queue를 몇 stage로 배치할지, 언제 wait할지, 언제 다음 beat로 전환할지"가 되었다. 이런 코드를 많이 읽고 나면, FMA unroll 횟수를 먼저 보는 대신 pipeline 깊이, async stage, wait group을 자연스럽게 신경 쓰게 된다.

## `Hopper`에 이르면 이동, 동기화, 계산을 함께 봐야 한다

`mma_vs_wgmma.cu`는 이 변화를 읽기에 매우 적합하다. 전통적인 `warp-level MMA`와 `SM90` 위의 `WGMMA + TMA`를 같은 곳에 놓았기 때문이다. 비교해서 보면 많은 것이 단번에 분명해진다.

`WGMMA` 경로의 뼈대는 다음과 같다.

```cpp
__global__ void wgmma_kernel(
    TiledMma mma,
    TensorA gA,
    TensorB gB,
    TensorC gC,
    TensorD gD,
    CUTLASS_GRID_CONSTANT TmaA const tmaA,
    CUTLASS_GRID_CONSTANT TmaB const tmaB) {
    extern __shared__ __align__(128) uint8_t shared_memory[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory)), SharedMemoryALayout{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<T_IN*>(shared_memory) + cosize(SharedMemoryALayout{})), SharedMemoryBLayout{});
    uint64_t* mbar = reinterpret_cast<uint64_t*>(shared_memory + sizeof(T_IN) * (cosize(SharedMemoryALayout{}) + cosize(SharedMemoryBLayout{})));
    ...
}
```

이동은 thread가 직접 load하는 것이 아니라 TMA에 맡긴다.

```cpp
ProducerBarType::arrive_and_expect_tx(mbar, tma_transaction_bytes);
copy(tmaA.with(*mbar), tAgA, tAsA);
copy(tmaB.with(*mbar), tBgB, tBsB);
ProducerBarType::wait(mbar, 0);
```

계산할 때도 더 이상 일반 warp granularity가 아니다.

```cpp
ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
...
gemm(mma, tCrA(_, _, _), tCrB(_, _, _), tCrAcc);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

이런 코드는 나중에 `Hopper`에 대한 이해를 매우 구체적으로 만들어 주었다. 이제 단순히 thread 무리가 행렬곱을 하는 것이 아니라, 세 가지 일을 어떻게 안정적으로 이어 붙일지 배치하는 것이다.

- tensor 단위 이동을 언제 시작할지
- transaction이 언제 실제로 visible해지는지
- warp-group이 언제 이 데이터를 먹기 시작할지

여기서 가장 과소평가되기 쉬운 것은 동기화 의미론이다. `wait(mbar, 0)`가 기다리는 것은 "다른 thread가 도착했다"가 아니라 이번 transaction이 실제로 완료되었다는 것이다. 이는 `__syncthreads()`와 전혀 같은 일이 아니다.

이 사고를 따라 더 내려가면, 이른바 `pingpong schedule`도 사실 그렇게 신비롭지 않다. 쉽게 말해 이동, main loop, 마무리가 서로 발을 밟지 않게 하는 것이다. 현재 stage가 계산 중이고, 다음 stage가 이동 중이며, 그보다 앞 stage가 결과를 내보내는 식으로 세 일이 맞물리면 Tensor Core가 굶기 쉽지 않다.

## 실제 추론 시스템에서 어려운 지점은 대개 그 고빈도 작은 operator 묶음에 있다

GEMM만 계속 보면, 행렬곱만 충분히 빠르면 시스템도 거의 괜찮아질 것이라는 착각을 하기 쉽다. `sgl-kernel/csrc/common_extension.cc`는 거의 한눈에 이 착각을 깨뜨린다. 여기서 등록되는 것은 단일 대형 operator가 아니라, 진짜 hot primitive의 연속이다.

```cpp
m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

m.def(
    "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
    " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
m.impl("sgl_per_token_group_quant_8bit_v2", torch::kCUDA, &sgl_per_token_group_quant_8bit_v2);

m.def(
    "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
    "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
    "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
    "expert_offsets, Tensor workspace) -> ()");
m.impl("fp8_blockwise_scaled_grouped_mm", torch::kCUDA, &fp8_blockwise_scaled_grouped_mm);
```

이 등록표 자체가 이미 많은 것을 설명한다. 예를 들어 `FP8 group GEMM`은 단일 행렬곱이 아니다. 앞에는 quantization, scale layout, grouped dispatch가 있다. 또 `enable_pdl`이 `RMSNorm` 같은 고빈도 operator에 붙어 있다는 것은 launch 방식과 scheduling latency가 이미 성능 변수로 취급되고 있음을 보여 준다. MoE 쪽은 더 분명하다. `problem_sizes`, `expert_offsets` 같은 매개변수가 등장하면, bottleneck이 단지 계산에만 있는 것이 아니라 "서로 다른 shape의 문제를 어떻게 한 경로 안에 batch로 넣을 것인가"에도 있다는 것을 알 수 있다.

그래서 현대 추론에서의 CUDA 튜닝은 많은 경우 "가장 빠른 kernel" 하나를 쫓는 것이 아니라, 전체 hot path를 고치는 일이다. quantization이 조금 느리면 뒤의 grouped MM이 idle 상태로 기다린다. RMSNorm이 중간 결과를 한 번 더 memory에 떨어뜨리면 전체 layer decode가 길어진다. MoE dispatch를 잘 배치하지 못하면 main kernel도 조각난 문제를 잔뜩 떠안는다.

## `RMSNorm` 같은 operator는 작아 보여도 실제로 작성하면 전혀 작지 않다

`RMSNorm`은 매우 쉽게 과소평가된다. 처음 보면 많은 사람이 reduce 한 번 하고 scale을 곱하는 것뿐이라고 생각한다. 하지만 실제 추론에 넣으면 곧바로 탄탄한 hot spot임을 알게 된다. 매 layer마다 수행해야 하고, token 단위로 고빈도 호출되며, hidden size도 흔히 `8192`, `16384`이다.

`sgl-kernel`은 이 문제를 매우 실질적으로 처리한다. fused 경로를 바로 앞에 배치한다.

```cpp
void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl) {
    CHECK_INPUT(input);
    ...
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
        cudaError_t status = norm::FusedAddRMSNorm(
            static_cast<c_type*>(input.data_ptr()),
            static_cast<c_type*>(residual.data_ptr()),
            static_cast<c_type*>(weight.data_ptr()),
            batch_size,
            hidden_size,
            input.stride(0),
            residual.stride(0),
            eps,
            enable_pdl,
            torch_current_stream);
        TORCH_CHECK(
            status == cudaSuccess, "FusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
        return true;
    });
}
```

이 구현은 핵심을 이미 매우 분명히 말하고 있다. residual add와 RMSNorm을 따로 하지 않는 이유는 중간 결과를 읽고 쓰는 한 라운드를 줄이기 위해서다. 이런 고빈도이면서 memory access 비중이 낮지 않은 operator에서는 이 일이 내부에서 산술 명령 몇 개를 더 아끼는 것보다 더 가치 있는 경우가 많다.

benchmark 쪽에서도 이것이 실제 호출과 동떨어진 극단적으로 단순한 측정 방식이 아님을 볼 수 있다.

```python
def rmsnorm_sglang(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    if residual is not None:
        sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps, enable_pdl=enable_pdl)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, weight, eps, out=out, enable_pdl=enable_pdl)
        output = out
    ...
```

먼저 shape를 2차원으로 정리한 뒤 fused와 non-fused 경로를 각각 탄다. 본질적으로 실제 사용 방식에 따라 서로 다른 구현을 비교하는 것이다. 여기서 진짜 눈여겨볼 변수는 세 가지뿐이다. reduce를 어느 단계까지 수행하는가, weight와 input이 충분히 넓은가, residual add를 자연스럽게 fuse했는가.

## `hp_rms_norm`의 사고방식은 완성도가 높다. 어떤 loop 한 조각만 빠르게 쓴 것이 아니다

나는 `hp_rms_norm` 이 구현이 꽤 흥미롭다고 생각한다. 내부 loop에 작은 수정을 하는 데 그치지 않고, type, shared memory, persistent CTA, occupancy까지 함께 설계하기 때문이다.

처음에 type layer를 보면 이런 느낌이 바로 온다. `half2`에서 멈추지 않고 `16B`와 `32B` 경로를 바로 나누었다.

```cpp
union U32B_bf162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __nv_bfloat162 real_type[8];
};

union U32B_f162{
#if __CUDACC_VER_MAJOR__ >= 13
  longlong4_32a memory_type;
#else
  longlong4 memory_type;
#endif
  __half2 real_type[8];
};
```

trait도 그에 맞춰 따라간다.

```cpp
template<> struct UVTypeTrait<__nv_bfloat16, 32> {
  using U = U32B_bf162;
#if __CUDACC_VER_MAJOR__ >= 13
  using V = longlong4_32a;
#else
  using V = longlong4;
#endif
};
```

이는 작성자가 처음부터 `32B` granularity의 vectorization을 진지하게 추구했다는 것을 보여 준다. 그것을 그냥 손쉬운 최적화로 여긴 것이 아니다. `B200`처럼 memory access 조직에 더 민감한 플랫폼에서는 이런 경로가 자주 분수령이 된다.

가장 소박한 `rms_norm_vector_reg_kernel`을 다시 보면, 이미 residual add, 제곱합 누적, 결과 writeback을 한데 묶고 있다.

```cpp
const V* p = reinterpret_cast<const V*>(input) + token_id * vec_hidden_dim;
u.memory_type = p[threadIdx.x];
V* p_res = reinterpret_cast<V*>(residual) + token_id * vec_hidden_dim;
u_res.memory_type = p_res[threadIdx.x];
...
float2 inp_res = make_float2(val.x + res.x, val.y + res.y);
acc_square.x += inp_res.x * inp_res.x;
acc_square.y += inp_res.y * inp_res.y;
u.real_type[i] = __float22half2_rn(inp_res);
...
p_res[threadIdx.x] = u.memory_type;
```

진짜로 계층을 끌어올리는 것은 `rms_norm_vector_reg_shm_kernel`이다. 먼저 `weight`를 비동기로 shared memory에 옮긴다.

```cpp
__shared__ barrier mbarrier;
...
if (threadIdx.x == 0) {
  init(&mbarrier, blockDim.x);
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
}
__syncthreads();

if (threadIdx.x == 0) {
  cuda::memcpy_async(
      shared_memory,
      weight,
      cuda::aligned_size_t<16>(vec_hidden_dim * VEC_SIZE_IN_BYTE),
      mbarrier
  );
}
...
barrier::arrival_token arrival_token = mbarrier.arrive();
```

첫 번째 token은 이 transaction을 명확히 한 번 기다린다.

```cpp
if (token_id == static_cast<int>(blockIdx.x)) {
  mbarrier.wait(std::move(arrival_token));
}
```

이후에는 persistent 스타일의 token loop에 들어간다.

```cpp
while (token_id < tokens) {
    ...
    token_id += static_cast<int>(gridDim.x);
}
```

이 코드 조각들을 함께 보면 매우 분명하다. 이 구현이 진짜로 하려는 것은 특정 `RMSNorm` 한 번을 조금 빠르게 계산하는 데 그치지 않는다. weight의 lifetime을 길게 만들고, shared memory staging이 가치 있게 하며, CTA를 persistent하게 앞으로 밀어붙이는 것이다. 즉 weight 재사용, vectorization, token lifetime, concurrency를 동시에 처리하고 있다.

## `8192`와 `16384` 두 hidden size는 많은 경우 같은 급의 문제가 아니다

많은 사람이 hidden size를 볼 때 16384는 그저 8192의 두 배라고 무의식적으로 생각한다. 하지만 이런 kernel에서는 보통 일이 그렇게 선형적이지 않다.

이 부분을 보자.

```cpp
int iteration = (vec_hidden_dim + threads - 1) / threads;
...
V* p_shm = reinterpret_cast<V*>(shared_memory + VEC_SIZE_IN_BYTE * vec_hidden_dim);
for (int i = 1; i < iteration; i++) {
  auto offset = threadIdx.x + i * threads;
  if (offset < vec_hidden_dim) {
    ...
    p_shm[shm_offset] = tmp.memory_type;
  }
}
```

`vec_hidden_dim`이 커지면 `iteration`도 따라 증가하고, shared memory에 입력을 임시 저장하는 영역도 팽창한다. 뒤쪽 reduce도 더 무거워진다.

```cpp
float warp_sum = cooperative_groups::reduce(
  cg_warp,
  acc_square.x + acc_square.y,
  cooperative_groups::plus<float>()
);

float cta_sum = cooperative_groups::reduce(
  cg_warp,
  threadIdx.x < NUM_WARPS ? buffer[threadIdx.x] : 0.0f,
  cooperative_groups::plus<float>()
);
```

이때 문제는 보통 "원소 수가 두 배"인 것에 그치지 않는다. 여러 일이 함께 변한다. vector load가 많아지고, shared memory footprint가 커지고, block 내 동기화 빈도가 높아지고, register lifetime이 길어지며, occupancy도 계단식으로 떨어질 수 있다. 많은 kernel이 여기서 먼저 무너지는 것은 산술이 아니라 자원 균형이다.

## dynamic shared memory를 켤 가치가 있는지는 occupancy와 함께 계산해야 한다

`hp_rms_norm`에서 가장 배울 만한 점은 shared memory 사용량을 고정으로 써 두지 않고, runtime이 "목표 concurrency에서 dynamic shared memory를 얼마나 더 열 수 있는지"를 명확히 계산하게 한다는 것이다.

```cpp
cudaFuncAttributes kernel_attr;
AT_CUDA_CHECK(cudaFuncGetAttributes(&kernel_attr, kernel_ptr));
AT_CUDA_CHECK(cudaFuncSetAttribute(
  kernel_ptr,
  cudaFuncAttributeMaxDynamicSharedMemorySize,
  at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin - kernel_attr.sharedSizeBytes
));

size_t smem_size;
AT_CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel_ptr, num_ctas_per_sm, num_threads));
```

grid도 아무렇게나 설정하지 않고, persistent CTA 수에 맞춰 직접 설정한다.

```cpp
uint persistent_ctas =
  at::cuda::getCurrentDeviceProperties()->multiProcessorCount * num_ctas_per_sm;

dim3 grid(persistent_ctas, 1, 1);
```

이런 코드는 사실 매우 실질적인 trade-off를 수행한다. shared memory를 조금 더 크게 열면 local reuse가 더 좋아질 가능성은 물론 있다. 하지만 그것이 occupancy와 맞바꿀 가치가 있는가? 이 질문은 많은 경우 "shared memory가 global load를 몇 번 줄였는가"보다 더 중요하다.

나는 적지 않은 kernel에서 local하게 보면 shared memory staging이 아름다운데, 실제로 실행하면 occupancy가 먼저 떨어지고 마지막에는 latency가 오히려 더 뚜렷하게 드러나는 경우를 보았다. 문제는 shared memory 자체가 아니라, 그것을 전체 SM resource model 안에 다시 넣어 보지 않았다는 데 있다.

## 마지막에는 결국 시스템으로 돌아가야 한다. 단일 대형 kernel만 봐서는 안 된다

`sgemm`이나 `wgmma`만 보면 CUDA 튜닝이 주로 한두 개의 대형 kernel을 쫓는 일이라고 생각하기 쉽다. 하지만 실제 추론에서는 많은 시간이 고빈도 중소 operator 묶음에 흩어져 있다. 특히 quantization, RMSNorm, MoE routing, RoPE, cache 이동 같은 경로가 그렇다. `common_extension.cc` 안의 인터페이스들이 함께 놓여 있는 모습은 사실 이 현실을 이미 보여 준다.

`fused_add_rmsnorm`, `sgl_per_token_group_quant_8bit_v2`, `fp8_blockwise_scaled_grouped_mm`, `moe_fused_gate`, `fast_topk_transform_fused`는 각각만 보면 "가장 눈부신" kernel이라고 할 수는 없다. 하지만 path 안에서 어느 하나라도 빠지면 시스템 성능은 한 조각 새어 나간다.

그래서 지금의 내 CUDA 튜닝 이해는 처음 배울 때와 꽤 달라졌다. 예전에는 어떤 명령이 새로운지, 어떤 kernel의 peak가 더 높은지에 더 신경 썼다. 지금은 오히려 세 가지를 더 신경 쓴다.

- 이 bottleneck이 대역폭인지, latency인지, 아니면 scheduling cadence인지
- 데이터가 너무 많이 옮겨졌는지, 또는 충분히 매끄럽게 옮겨지지 않았는지
- 단일 kernel 바깥에서 전체 hot path가 이어질 수 있는지

이 코드 묶음을 따라 보고 나면, 마지막에 남는 것은 사실 어떤 "비밀" 하나가 아니라 꽤 성실한 판단 프레임워크다. 먼저 머신을 파악하고, 그다음 데이터 흐름을 보고, 새 아키텍처의 이동과 동기화 의미론을 본 뒤, 마지막으로 문제를 시스템 안에 다시 놓는다. 이렇게 걷다 보면 산발적으로 보이던 많은 기법, 즉 `cp.async`, `TMA`, `WGMMA`, `FP8 grouped GEMM`, `persistent CTA`, `enable_pdl`이 그렇게 흩어져 있지 않다. 그것들은 모두 한 가지 일을 한다. 계산 유닛이 데이터를 기다리며 비어 있지 않게 만드는 것이다.

나는 이것이 CUDA 튜닝에서 가장 화려하지는 않지만 가장 잘 통하는 경험에 꽤 가까운 부분이라고 생각한다.
