# Nvidia Tensor Core - WMMA API 프로그래밍 입문

> 원문: https://zhuanlan.zhihu.com/p/620766588

## 1. WMMA (Warp-level Matrix Multiply Accumulate) API

Compute Capability 7.0 이상의 CUDA 장치에서는 CUDA C++ API로 Tensor Core를 호출할 수 있으며, `D = A*B + C` 형태의 혼합 정밀도 행렬 곱을 지원합니다.

```cpp
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

- **fragment**: Tensor Core 데이터 저장 클래스. `matrix_a`, `matrix_b`, `accumulator` 지원
- **load_matrix_sync**: Tensor Core 데이터 로드 API. global/shared memory에서 fragment로 로드
- **store_matrix_sync**: Tensor Core 결과 저장 API. fragment에서 global/shared memory로 저장
- **fill_fragment**: fragment를 상수로 채우는 API
- **mma_sync**: Tensor Core 행렬 곱 계산 API. `D = A*B + C` 또는 `C = A*B + C` 지원

## 2. 예제

m16n16k16을 예로 HGEMM(`C = A*B`)을 구현합니다. 여기서 행렬 A(M×K, row-major), B(K×N, col-major), C(M×N, row-major)의 정밀도는 모두 FP16입니다. 먼저 CUDA Core로 naive HGEMM을 작성하는 방법을 봅니다.

### 2.1 CUDA Core

각 스레드가 행렬 C의 한 원소를 계산하는 방식으로 naive 커널을 구성합니다. 현재 스레드가 처리할 C 원소의 좌표를 결정한 뒤, K 차원을 순회하며 필요한 A·B 원소를 global memory에서 레지스터로 직접 로드해 계산하고, 결과를 레지스터에서 C에 직접 써냅니다. 모든 block의 계산이 끝나면 C가 완성됩니다. 기술 난이도가 높은 코드는 아니며 단순 비교를 위한 예시입니다. 소스는 `cuda_hgemm`에 있습니다.

```cpp
__global__ void simtNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= M && col >= N) {
        return;
    }

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += __half2float(A[row * K + i]) * __half2float(B[i + col * K]);
    }

    C[row * N + col] = __float2half(tmp);
}

void simtNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(16, 16);
    dim3 grid(div_ceil(N, block.x), div_ceil(M, block.y));

    simtNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
```

### 2.2 Tensor Core

이번에는 WMMA API로 naive 커널을 구성해봅니다(`cuda-sample` 참고). CUDA Core 버전과 다른 점은, WMMA는 **각 warp이 C의 `WMMA_M × WMMA_N` 크기 타일 하나를 처리**하도록 구성해야 한다는 것입니다. Tensor Core의 연산 단위는 warp 레벨이며, 계산 대상인 행렬 원소도 2차원이기 때문입니다.

이후 흐름은 CUDA Core naive와 동일합니다. 현재 warp이 처리할 C 타일의 좌표를 결정하고, 타일 계산에 필요한 fragment를 선언한 뒤, `WMMA_K` 간격으로 K를 순회하며 global memory에서 A·B 타일을 fragment로 로드해 계산합니다. 마지막으로 결과를 fragment에서 C로 직접 써냅니다. 모든 block이 완료되면 C가 완성됩니다.

주의할 점은 `load_matrix_sync`와 `store_matrix_sync` 모두 **stride 단위로** 행렬 원소에 접근한다는 것입니다. 소스는 `cuda_hgemm`에 있습니다.

```cpp
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

using namespace nvcuda;

__global__ void wmmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    wmmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
```

### 2.3 차이점

위 두 naive 커널 코드로부터 CUDA Core와 Tensor Core 호출의 차이를 정리하면 다음과 같습니다.

- **연산 단위**: CUDA Core는 스레드 단위, Tensor Core는 warp 단위
- **연산 차원**: CUDA Core는 1차원 원소 단위, Tensor Core는 2차원 타일 단위
- **자료 구조 의존성**: WMMA로 Tensor Core를 호출하려면 데이터 저장 클래스 `fragment`가 필요하나, CUDA Core는 별도 자료 구조 없음

## 3. 하위 레벨 코드

위 WMMA naive 커널을 RTX A6000(sm_86, CUDA 11.3)에서 컴파일했을 때 대응하는 PTX와 SASS를 살펴봅니다.

### 3.1 PTX

덤프한 PTX 코드는 아래와 같습니다. 생각보다 단순하지 않습니다.

```
.visible .entry _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm(
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_0,
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_1,
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_2,
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_3,
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_4,
.param .u64 _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_5
)
{
.reg .pred %p<8>;
.reg .b16 %rs<2>;
.reg .f32 %f<2>;
.reg .b32 %r<44>;
.reg .b64 %rd<36>;

ld.param.u64 %rd14, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_0];
ld.param.u64 %rd15, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_1];
ld.param.u64 %rd16, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_2];
ld.param.u64 %rd19, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_3];
ld.param.u64 %rd17, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_4];
ld.param.u64 %rd18, [_Z15wmmaNaiveKernelPK6__halfS1_PS_mmm_param_5];
shr.u64 %rd1, %rd18, 4;
mov.u32 %r15, %ctaid.y;
shl.b32 %r16, %r15, 4;
cvt.u64.u32 %rd2, %r16;
mov.u32 %r17, %ctaid.x;
shl.b32 %r18, %r17, 4;
cvt.u64.u32 %rd3, %r18;
setp.ge.u64 %p1, %rd2, %rd19;
setp.ge.u64 %p2, %rd3, %rd17;
and.pred %p3, %p1, %p2;
@%p3 bra $L__BB0_5;

and.b64 %rd4, %rd18, 15;
setp.ne.s64 %p4, %rd4, 0;
mov.f32 %f1, 0f00000000;

    { cvt.rn.f16.f32 %rs1, %f1;}

    mov.b32 %r40, {%rs1, %rs1};
selp.b64 %rd20, -1, 0, %p4;
setp.eq.s64 %p5, %rd1, %rd20;
mov.u32 %r41, %r40;
mov.u32 %r42, %r40;
mov.u32 %r43, %r40;
@%p5 bra $L__BB0_4;

mul.lo.s64 %rd21, %rd2, %rd18;
cvt.u32.u64 %r2, %rd18;
selp.u64 %rd22, 1, 0, %p4;
add.s64 %rd35, %rd1, %rd22;
mul.lo.s64 %rd23, %rd3, %rd18;
cvta.to.global.u64 %rd24, %rd15;
shl.b64 %rd25, %rd23, 1;
add.s64 %rd34, %rd24, %rd25;
cvta.to.global.u64 %rd26, %rd14;
shl.b64 %rd27, %rd21, 1;
add.s64 %rd33, %rd26, %rd27;
mov.u32 %r41, %r40;
mov.u32 %r42, %r40;
mov.u32 %r43, %r40;

$L__BB0_3:
wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%r19, %r20, %r21, %r22, %r23, %r24, %r25, %r26}, [%rd33], %r2;
wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%r27, %r28, %r29, %r30, %r31, %r32, %r33, %r34}, [%rd34], %r2;
wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {%r43, %r42, %r41, %r40}, {%r19, %r20, %r21, %r22, %r23, %r24, %r25, %r26}, {%r27, %r28, %r29, %r30, %r31, %r32, %r33, %r34}, {%r43, %r42, %r41, %r40};
add.s64 %rd34, %rd34, 32;
add.s64 %rd33, %rd33, 32;
add.s64 %rd35, %rd35, -1;
setp.ne.s64 %p7, %rd35, 0;
@%p7 bra $L__BB0_3;

$L__BB0_4:
mul.lo.s64 %rd28, %rd2, %rd17;
add.s64 %rd29, %rd28, %rd3;
cvta.to.global.u64 %rd30, %rd16;
shl.b64 %rd31, %rd29, 1;
add.s64 %rd32, %rd30, %rd31;
cvt.u32.u64 %r35, %rd17;
wmma.store.d.sync.aligned.row.m16n16k16.global.f16 [%rd32], {%r43, %r42, %r41, %r40}, %r35;

$L__BB0_5:
ret;

}
```

다만 우리가 주목할 부분은 WMMA 관련 PTX 명령이며, 아래와 같습니다. 바로 NVIDIA가 제공하는 WMMA PTX 명령으로 Tensor Core를 호출하고 있습니다. 즉 WMMA API로 작성하든 WMMA PTX 명령으로 직접 작성하든, **하위 레벨에서 큰 차이가 없습니다**.

```
wmma.load.a.sync.aligned.row.m16n16k16.global.f16
wmma.load.b.sync.aligned.col.m16n16k16.global.f16
wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16
wmma.store.d.sync.aligned.row.m16n16k16.global.f16
```

### 3.2 SASS

대응하는 SASS 코드를 더 덤프하면 아래와 같습니다. 역시 단순하지 않습니다.

```
        Function : _Z15wmmaNaiveKernelPK6__halfS1_PS_mmm
    .headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R32, SR_CTAID.X ;
        /*0020*/                   IMAD.MOV.U32 R0, RZ, RZ, c[0x0][0x188] ;
        /*0030*/                   S2R R18, SR_CTAID.Y ;
        /*0040*/                   IMAD.SHL.U32 R32, R32, 0x10, RZ ;
        /*0050*/                   IMAD.SHL.U32 R18, R18, 0x10, RZ ;
        /*0060*/                   ISETP.GE.U32.AND P0, PT, R32, c[0x0][0x180], PT ;
        /*0070*/                   ISETP.GE.U32.AND P1, PT, R18, c[0x0][0x178], PT ;
        /*0080*/                   ISETP.GE.U32.AND.EX P0, PT, RZ, c[0x0][0x184], PT, P0 ;
        /*0090*/                   ISETP.GE.U32.AND.EX P1, PT, RZ, c[0x0][0x17c], PT, P1 ;
        /*00a0*/               @P0 EXIT P1 ;
        /*00b0*/                   LOP3.LUT P0, RZ, R0.reuse, 0xf, RZ, 0xc0, !PT ;
        /*00c0*/                   IMAD.MOV.U32 R3, RZ, RZ, 0x4 ;
        /*00d0*/                   ULDC.64 UR4, c[0x0][0x118] ;
        /*00e0*/                   IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x18c] ;
        /*00f0*/                   LOP3.LUT P0, RZ, RZ, c[0x0][0x18c], RZ, 0xc0, P0 ;
        /*0100*/                   CS2R R20, SRZ ;
        /*0110*/                   SHF.R.U64 R3, R0, R3, c[0x0][0x18c] ;
        /*0120*/                   CS2R R16, SRZ ;
        /*0130*/                   SEL R2, RZ, 0xffffffff, !P0 ;
        /*0140*/                   SHF.R.U32.HI R5, RZ, 0x4, R5 ;
        /*0150*/                   ISETP.NE.U32.AND P1, PT, R3, R2, PT ;
        /*0160*/                   ISETP.NE.AND.EX P1, PT, R5, R2, PT, P1 ;
        /*0170*/              @!P1 BRA 0xbc0 ;
        /*0180*/                   SEL R2, RZ, 0x1, !P0 ;
        /*0190*/                   IMAD.WIDE.U32 R6, R18, c[0x0][0x188], RZ ;
        /*01a0*/                   CS2R R16, SRZ ;
        /*01b0*/                   IADD3 R2, P0, R2, R3, RZ ;
        /*01c0*/                   IMAD.WIDE.U32 R8, R32, c[0x0][0x188], RZ ;
        ...
        /*0490*/                   HMMA.16816.F16 R12, R8.reuse, R12, R16 ;
        ...
        /*04c0*/                   HMMA.16816.F16 R24, R8, R24, R20 ;
        ...
        /*0530*/                   HMMA.16816.F16 R16, R8.reuse, R16, R12 ;
        /*0560*/                   HMMA.16816.F16 R24, R8, R14, R24 ;
        /*0670*/                   HMMA.16816.F16 R16, R8.reuse, R20, R16 ;
        /*0680*/                   HMMA.16816.F16 R24, R8, R26, R24 ;
        /*06a0*/                   HMMA.16816.F16 R16, R12.reuse, R28, R16 ;
        /*06b0*/                   HMMA.16816.F16 R20, R12, R22, R24 ;
        ...
        /*0930*/                   HMMA.16816.F16 R16, R8.reuse, R22, R16 ;
        /*0940*/                   HMMA.16816.F16 R20, R8, R24, R20 ;
        /*0960*/                   HMMA.16816.F16 R16, R12.reuse, R26, R16 ;
        /*0970*/                   HMMA.16816.F16 R20, R12, R28, R20 ;
        ...
        /*0b90*/                   HMMA.16816.F16 R16, R8.reuse, R12, R16 ;
        /*0ba0*/                   HMMA.16816.F16 R20, R8, R28, R20 ;
        ...
        /*0cb0*/                   STG.E [R4.64], R16 ;
        /*0cc0*/                   STG.E [R2.64], R17 ;
        /*0cd0*/                   STG.E [R4.64+0x10], R20 ;
        /*0ce0*/                   STG.E [R2.64+0x10], R21 ;
        /*0cf0*/                   EXIT ;
```

(전체 SASS 덤프는 분량이 많아 핵심만 발췌했습니다.)

여기서도 주목할 것은 WMMA 관련 SASS 명령입니다. **WMMA 16×16×16은 하위 레벨에서 두 개의 HMMA 16×8×16 명령으로 구현**됨을 확인할 수 있습니다. SASS 명령 또한 NVIDIA가 제공하는 Tensor Core 호출 방식 중 하나입니다.

```
HMMA.16816.F16
```

[Nvidia Tensor Core 초탐](../B66_tensor_core_intro/README.md)에서 언급한 Tensor Core 호출 방식 네 가지 중 세 가지(WMMA API, WMMA PTX, SASS)를 여기서 다뤘고, 나머지 하나인 MMA PTX는 후속 글에서 다룹니다. MMA16816 PTX 명령의 하위 구현도 바로 HMMA16816입니다.

## 4. 기타

### 4.1 HGEMM 최적화

WMMA API 학습의 목표는 Tensor Core를 호출해 HGEMM을 최적화하는 것입니다. cuBLAS 대비 WMMA 기반 구현의 성능은 오픈소스 `cuda_hgemm` 코드를 참고하세요.
