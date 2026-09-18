

## `nvfp4_quant_entry.cu` 구현

```c++
// NVFP4 지원이 활성화되어 있는지 확인한다 (SM100 아키텍처에서만 사용 가능)
#if defined ENABLE_NVFP4 && ENABLE_NVFP4

// SM100 아키텍처 전용 FP4 양자화 함수 선언
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,        // 양자화된 FP4 텐서 출력
    torch::Tensor const& input,   // 입력 FP16/BF16 텐서
    torch::Tensor& output_sf,     // 출력 스케일 팩터
    torch::Tensor const& input_sf // 입력 스케일 팩터
);

// SM100 아키텍처 전용 expert 모델 FP4 양자화 함수 선언
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                              // 양자화된 FP4 텐서 출력
    torch::Tensor& output_scale,                        // 출력 스케일 팩터
    torch::Tensor const& input,                         // 입력 텐서
    torch::Tensor const& input_global_scale,            // 입력 global 스케일 팩터
    torch::Tensor const& input_offset_by_experts,       // expert별 offset
    torch::Tensor const& output_scale_offset_by_experts // 출력 스케일 팩터 offset
);

// SM100 아키텍처 전용 SiLU 활성화 + 곱셈 + expert 모델 FP4 양자화 함수 선언
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                       // 양자화된 FP4 텐서 출력
    torch::Tensor& output_scale,                 // 출력 스케일 팩터
    torch::Tensor const& input,                  // 입력 텐서
    torch::Tensor const& input_global_scale,     // 입력 global 스케일 팩터
    torch::Tensor const& mask,                   // mask 텐서 (게이팅에 사용)
    bool use_silu_and_mul                        // SiLU 활성화와 곱셈을 사용할지 여부
);

#endif

// 범용 FP4 양자화 인터페이스 함수
void scaled_fp4_quant(
    torch::Tensor& output,        // 양자화된 FP4 텐서 출력
    torch::Tensor const& input,   // 입력 FP16/BF16 텐서
    torch::Tensor& output_sf,     // 출력 스케일 팩터
    torch::Tensor const& input_sf // 입력 스케일 팩터
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // NVFP4를 지원하면 SM100 전용 구현을 호출한다
  return scaled_fp4_quant_sm100a(output, input, output_sf, input_sf);
#endif
  // NVFP4를 지원하지 않으면 미구현 에러를 던진다
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization");
}

// 범용 expert 모델 FP4 양자화 인터페이스 함수
void scaled_fp4_experts_quant(
    torch::Tensor& output,                              // 양자화된 FP4 텐서 출력
    torch::Tensor& output_scale,                        // 출력 스케일 팩터
    torch::Tensor const& input,                         // 입력 텐서
    torch::Tensor const& input_global_scale,            // 입력 global 스케일 팩터
    torch::Tensor const& input_offset_by_experts,       // expert별 offset
    torch::Tensor const& output_scale_offset_by_experts // 출력 스케일 팩터 offset
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // NVFP4를 지원하면 SM100 전용 구현을 호출한다
  return scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, input_offset_by_experts, output_scale_offset_by_experts);
#endif
  // NVFP4를 지원하지 않으면 미구현 에러를 던진다
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}

// 범용 SiLU 활성화 + 곱셈 + expert 모델 FP4 양자화 인터페이스 함수
void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output,                       // 양자화된 FP4 텐서 출력
    torch::Tensor& output_scale,                 // 출력 스케일 팩터
    torch::Tensor const& input,                  // 입력 텐서
    torch::Tensor const& input_global_scale,     // 입력 global 스케일 팩터
    torch::Tensor const& mask,                   // mask 텐서 (게이팅에 사용)
    bool use_silu_and_mul                        // SiLU 활성화와 곱셈을 사용할지 여부
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // NVFP4를 지원하면 SM100 전용 구현을 호출한다
  return silu_and_mul_scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, mask, use_silu_and_mul);
#endif
  // NVFP4를 지원하지 않으면 미구현 에러를 던진다
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}
```

## `nvfp4_quant.cuh` 구현

```c++
// 필요한 헤더 파일 포함
#include <cuda.h>           // CUDA 런타임 API
#include <cuda_fp8.h>       // FP8 데이터 타입 지원
#include <cutlass/arch/config.h>  // CUTLASS 아키텍처 설정

// 타입 변환기: 단정밀도와 배정밀도 벡터 타입 사이를 변환한다 (half과 bfloat16에 적용)
template <typename T>
struct TypeConverter {
  using Type = half2;  // 기본값은 half2 타입으로 변환
};  // 범용성을 유지한다

// half2 -> half 특수화
template <>
struct TypeConverter<half2> {
  using Type = half;
};

// half -> half2 특수화
template <>
struct TypeConverter<half> {
  using Type = half2;
};

// __nv_bfloat162 -> __nv_bfloat16 특수화
template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

// __nv_bfloat16 -> __nv_bfloat162 특수화
template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// 스레드 하나가 처리하는 element 개수
#define ELTS_PER_THREAD 8

// FP4 변환 관련 상수
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;  // 스레드 하나가 변환하는 FP4 element 개수
constexpr int CVT_FP4_SF_VEC_SIZE = 16;     // FP4 스케일 팩터 벡터 크기

// float32 값 8개를 e2m1 값 8개로 변환한다 (하나의 uint32_t로 표현)
// e2m1은 FP4 포맷이다: 지수 2비트, 가수 1비트
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  // 여기서 사용하는 PTX 명령은 sm100a/sm103a 아키텍처 지원이 필요하다
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"                                    // 8비트 register 선언
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"     // float32 두 개를 e2m1x2 포맷으로 변환
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"     // rn=round to nearest, satfinite=포화 유한값
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"         // 4개 바이트를 32비트 값으로 패킹
      "}"
      : "=r"(val)                                            // 출력: 32비트 register
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),  // 입력: float register 8개
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
#else
  return 0;  // 지원하지 않는 아키텍처에서는 0을 반환
#endif
}

// float2 값 4개를 e2m1 값 8개로 변환한다 (하나의 uint32_t로 표현)
// 위 함수의 float2 벡터 버전이다
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  // 여기서 사용하는 PTX 명령은 sm100a/sm103a 아키텍처 지원이 필요하다
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y),  // float2의 x, y 성분에 접근
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// 빠른 역수 근사 계산 (flush-to-zero 모드)
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// FP4 양자화에서 스케일 팩터의 출력 offset 주소를 계산한다
// SFType: 스케일 팩터 타입, CVT_FP4_NUM_THREADS_PER_SF: 스케일 팩터 하나당 스레드 수
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numCols, SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // SM100+ 아키텍처에서만 지원
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 스레드 한 쌍이 스케일 팩터 하나를 global memory에 기록한다
  // TODO: shared memory에 임시로 모아 packed STG.32 명령을 지원하도록 할 것
  // 이 방식이 스레드 4개의 STG.8보다 나은가?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // 스케일 팩터 벡터 인덱스 (K 차원에서 element 16개가 스케일 팩터 하나를 공유한다)
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // 스케일 팩터 레이아웃: [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 대응하는 인덱스: [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // M 차원의 tile 인덱스를 계산
    int32_t mTileIdx = mIdx / (32 * 4);
    // 스케일 팩터 벡터 크기는 16이다
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;  // 올림 나눗셈
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    // K 차원의 tile 인덱스와 stride를 계산
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile 레이아웃 [32, 4]는 column-major다
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // 전역 offset을 계산
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// 16바이트 packed 데이터 타입 정의
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // 타입 변환기로 대응하는 벡터 타입을 얻는다
};

// FP8 e4m3 포맷 특수화 버전
template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];  // FP8x2 element 8개, 총 16개의 FP8 값
};

```

## `nvfp4_quant_kernels.cu` 구현

```c++
// PackedVec을 FP4 포맷으로 양자화하여 uint32_t로 출력한다
// Type: 입력 데이터 타입(half/bfloat16), UE8M0_SF: UE8M0 포맷 스케일 팩터를 사용할지 여부
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // SM100+ 아키텍처에서만 지원
  // 로컬 8개 값 중 절댓값 최댓값을 구한다
  auto localMax = __habs2(vec.elts[0]);

  // 로컬 최댓값 계산 (루프 언롤링 최적화)
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // half2의 절댓값 최댓값을 계산
  }

  // warp shuffle로 16개 값 전체의 절댓값 최댓값을 구한다 (스레드 두 개가 협력)
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 최종 절댓값 최댓값을 얻는다
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 스케일 팩터 SF를 계산한다 (벡터 최댓값 / e2m1 최댓값)
  // e2m1 포맷의 최댓값 = 6.0
  // TODO: 계산용 데이터 타입으로 half를 사용해 성능을 높일 것
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 스케일 팩터의 8비트 표현
  uint8_t fp8SFVal;
  
  // 템플릿 파라미터에 따라 스케일 팩터 포맷을 선택한다
  if constexpr (UE8M0_SF) {
    // UE8M0 포맷 사용 (지수 8비트, 가수 0비트)
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // E4M3 포맷 사용 (지수 4비트, 가수 3비트)
    // 여기서 SFValue는 항상 양수이므로 E4M3는 UE4M3와 동일하다
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  
  // 출력 스케일 팩터를 계산한다
  // 수식: final_scale = 1 / (fp32(fp8(SFValue * SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  // 출력 포인터가 주어졌다면 스케일 팩터를 global memory에 기록한다 (8비트 저장)
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 입력 데이터를 float2 배열로 변환한다
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 입력 타입에 따라 변환한다
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 출력 스케일 팩터를 적용한다
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // e2m1 값(FP4 포맷)으로 변환한다
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 패킹된 e2m1 값을 반환한다
  return e2m1Vec;
#else
  return 0;  // 지원하지 않는 아키텍처에서는 0을 반환
#endif
}

// FP16/BF16을 FP4로 변환하는 CUDA kernel
// 기본값으로 UE4M3 포맷 스케일 팩터를 사용한다
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+ 아키텍처에서의 launch bound 최적화
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,        // 입력 행렬의 행 수
    int32_t numCols,        // 입력 행렬의 열 수
    Type const* in,         // 입력 데이터 포인터
    float const* SFScale,   // global 스케일 팩터
    uint32_t* out,          // 출력 FP4 데이터 포인터
    uint32_t* SFout         // 출력 스케일 팩터 포인터
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // SF에 적용할 global 스케일 팩터를 가져온다
  // 주의: SFScale은 다음 GEMM의 alpha와 같다. 즉 (448.f / (Alpha_A / 6.f))
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // 입력 텐서의 행/열 루프 처리
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
      // 입력 offset을 계산
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 출력 텐서의 offset을 얻는다
      // element 8개가 하나의 uint32_t로 패킹되므로 inOffset과 동일하다
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // 스케일 팩터 출력 주소를 계산
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      // FP16에서 FP4로의 변환을 수행
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

// FP4 양자화의 host 측 호출 함수 템플릿
template <typename T>
void invokeFP4Quantization(
    int m,                      // 행렬의 행 수
    int n,                      // 행렬의 열 수
    T const* input,             // 입력 데이터 포인터
    float const* SFScale,       // global 스케일 팩터
    int64_t* output,            // 출력 FP4 데이터
    int32_t* SFOuput,           // 출력 스케일 팩터
    bool useUE8M0,              // UE8M0 포맷 사용 여부
    int multiProcessorCount,    // SM 개수
    cudaStream_t stream         // CUDA stream
) {
  // grid와 block 크기 설정
  // 스레드 하나가 값 8개를 변환한다
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // SM당 block 수를 구한다 (SM을 충분히 활용할 수 있다고 가정)
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // 변환 kernel을 실행한다
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// 함수 템플릿 명시적 인스턴스화 - half 타입
template void invokeFP4Quantization(
    int m,
    int n,
    half const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

// 함수 템플릿 명시적 인스턴스화 - bfloat16 타입
template void invokeFP4Quantization(
    int m,
    int n,
    __nv_bfloat16 const* input,
    float const* SFScale,
    int64_t* output,
    int32_t* SFOuput,
    bool useUE8M0,
    int multiProcessorCount,
    cudaStream_t stream);

// 현재 GPU의 multiprocessor 개수를 얻는다 (static 캐시로 최적화)
inline int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // 현재 CUDA 디바이스 ID를 얻는다
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // 현재 디바이스의 multiprocessor 개수를 얻는다
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

    return count;  // static 변수를 초기화한다
  }();

  return multi_processor_count;  // 이후 호출에서는 캐시된 값을 반환한다
}

// SM100 아키텍처 전용 FP4 양자화 구현 함수
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,          // 출력 FP4 텐서
    torch::Tensor const& input,     // 입력 FP16/BF16 텐서
    torch::Tensor& output_sf,       // 출력 스케일 팩터 텐서
    torch::Tensor const& input_sf   // 입력 스케일 팩터 텐서
) {
  // SM 아키텍처 버전을 확인한다
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 입력 텐서의 차원을 얻는다
  int32_t m = input.size(0);  // 행 수
  int32_t n = input.size(1);  // 열 수

  // 열 수가 반드시 16의 배수인지 확인한다 (FP4 패킹 요구사항)
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  // GPU의 multiprocessor 개수를 얻는다
  int multiProcessorCount = getMultiProcessorCount();

  // 데이터 포인터를 얻는다
  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  // CUDA 디바이스 guard와 stream을 설정한다
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // 현재는 e8m0 스케일 팩터 포맷을 지원하지 않는다
  bool useUE8M0 = false;

  // 입력 데이터 타입에 따라 해당 양자화 함수로 dispatch한다
  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      std::cerr << "Observing: " << input.scalar_type() << " for the input datatype which is invalid";
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
}
```

## `nvfp4_expert_quant.cu` 구현

```c++
// expert 모델 전용 FP16에서 FP4로의 양자화 함수 (기본 버전과 핵심 로직이 동일하다)
// Type: 입력 데이터 타입(half/bfloat16), UE8M0_SF: UE8M0 포맷 스케일 팩터를 사용할지 여부
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // SM100+ 아키텍처에서만 지원
  // 로컬 8개 값 중 절댓값 최댓값을 구한다
  auto localMax = __habs2(vec.elts[0]);

  // 로컬 최댓값 계산 (루프 언롤링 최적화)
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // half2의 절댓값 최댓값을 계산
  }

  // warp shuffle로 16개 값 전체의 절댓값 최댓값을 구한다 (스레드 두 개가 협력)
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 최종 절댓값 최댓값을 얻는다
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 스케일 팩터 SF를 계산한다 (벡터 최댓값 / e2m1 최댓값)
  // e2m1 포맷의 최댓값 = 6.0
  // TODO: 계산용 데이터 타입으로 half를 사용해 성능을 높일 것
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 스케일 팩터의 8비트 표현
  uint8_t fp8SFVal;
  
  // 템플릿 파라미터에 따라 스케일 팩터 포맷을 선택한다
  if constexpr (UE8M0_SF) {
    // float32에서 8비트 지수부를 추출한다
    // float 32비트 = 부호 1비트 + 지수 8비트 + 가수 23비트
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // 다시 fp32 포맷으로 변환한다
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // 여기서 SFValue는 항상 양수이므로 E4M3는 UE4M3와 동일하다
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // 다시 fp32 포맷으로 변환한다
    SFValue = float(tmp);
  }
  
  // 출력 스케일 팩터를 계산한다
  // 수식: final_scale = 1 / (fp32(fp8(SFValue * SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

  // 출력 포인터가 주어졌다면 스케일 팩터를 global memory에 기록한다 (8비트 저장)
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 입력 데이터를 float2 배열로 변환한다
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 입력 타입에 따라 변환한다
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 출력 스케일 팩터를 적용한다
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // e2m1 값(FP4 포맷)으로 변환한다
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 패킹된 e2m1 값을 반환한다
  return e2m1Vec;
#else
  return 0;  // 지원하지 않는 아키텍처에서는 0을 반환
#endif
}

// SiLU 활성화 함수 구현: silu(x) = x / (1 + exp(-x))
// Swish 활성화 함수라고도 하며 Transformer 모델에서 널리 쓰인다
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

// SiLU 활성화 함수와 곱셈을 융합한 연산
// expert 모델에서는 보통 게이팅 연산이 필요하다: silu(x) * y
template <class Type>
inline __device__ void silu_and_mul(PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  float2 x[CVT_FP4_ELTS_PER_THREAD / 2];  // x 벡터의 float2 값을 저장
  float2 y[CVT_FP4_ELTS_PER_THREAD / 2];  // y 벡터의 float2 값을 저장

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 입력 타입에 따라 변환하고 계산한다
    if constexpr (std::is_same_v<Type, half>) {
      x[i] = __half22float2(x_vec.elts[i]);     // half2 -> float2
      y[i] = __half22float2(y_vec.elts[i]);     // half2 -> float2
      x[i].x = silu(x[i].x) * y[i].x;           // x 성분에 silu를 적용하고 y를 곱한다
      x[i].y = silu(x[i].y) * y[i].y;           // y 성분에 silu를 적용하고 y를 곱한다
      x_vec.elts[i] = __float22half2_rn(x[i]);  // float2 -> half2 (round to nearest)
    } else {
      x[i] = __bfloat1622float2(x_vec.elts[i]);     // bfloat162 -> float2
      y[i] = __bfloat1622float2(y_vec.elts[i]);     // bfloat162 -> float2
      x[i].x = silu(x[i].x) * y[i].x;               // x 성분에 silu를 적용하고 y를 곱한다
      x[i].y = silu(x[i].y) * y[i].y;               // y 성분에 silu를 적용하고 y를 곱한다
      x_vec.elts[i] = __float22bfloat162_rn(x[i]);  // float2 -> bfloat162 (round to nearest)
    }
  }
}

// expert 모델 FP4 양자화 kernel (동적 expert 탐색과 SiLU 활성화 지원)
// 기본값으로 UE4M3 포맷 스케일 팩터를 사용한다
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+ 아키텍처에서의 launch bound 최적화
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 입력 행렬의 행 수
    int32_t numCols,                        // 입력 행렬의 열 수
    Type const* in,                         // 입력 데이터 포인터
    float const* SFScale,                   // expert별 스케일 팩터 배열
    uint32_t* out,                          // 출력 FP4 데이터 포인터
    uint32_t* SFout,                        // 출력 스케일 팩터 포인터
    uint32_t* input_offset_by_experts,      // 입력에서 expert별 offset
    uint32_t* output_scale_offset_by_experts, // 출력 스케일 팩터에서 expert별 offset
    int32_t* mask,                          // mask 배열 (조기 종료에 사용)
    int n_experts,                          // expert 개수
    bool low_latency                        // 저지연 모드 활성화 여부
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 스레드와 데이터 인덱스를 계산한다
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  
  // TODO(kaixih@nvidia): 현재는 mask가 silu_and_mul과 함께 사용된다고 가정한다
  // 앞으로 더 범용적인 mask 동작이 필요할 수 있다. silu인 경우 입력의 마지막 차원이 두 배가 된다
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

  // 전역 스레드 하나가 element 하나를 처리한다
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 현재 전역 스레드가 처리해야 할 행과 열을 계산한다
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // expert 개수에 따라 서로 다른 전략으로 expert 내 인덱스를 찾는다
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      // expert 개수가 적은 경우: 선형 탐색
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);    // 현재 offset을 캐시 로드
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);   // 다음 offset을 캐시 로드
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // expert 개수가 많은 경우: 청크 단위 벡터화 로드 최적화
      // 로컬 배열 크기를 17로 잡은 것은 register 제약 때문이다
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        // int4 벡터화 로드로 offset 16개를 읽는다 (한 번에 uint32_t 4개씩 로드)
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

        // 로드한 offset 16개를 검사한다
#pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

    // mask를 사용할 때의 조기 종료
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 입력 offset을 계산하고 데이터를 로드한다
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // mask를 사용하면 SiLU 활성화와 곱셈 융합 연산을 수행한다
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 출력 텐서의 offset을 얻는다
    // element 8개가 하나의 uint32_t로 패킹되므로 inOffset과 동일하다
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // SF에 적용할 global 스케일 팩터를 가져온다
    // 주의: SFScale은 다음 GEMM의 alpha와 같다. 즉 (448.f / (Alpha_A / 6.f))
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 현재 expert의 스케일 팩터 출력 주소를 계산한다
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 실제 output_scales 차원은 패딩된 numCols로부터 계산된다
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 스케일 팩터 출력 주소를 계산한다
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // FP16에서 FP4로의 변환을 수행한다
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// expert 전용 FP4 양자화 kernel (스레드에서 expert로의 정적 매핑)
// 기본값으로 UE4M3 포맷 스케일 팩터를 사용한다
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_expert(  // SM100+ 아키텍처에서의 launch bound 최적화
#else
cvt_fp16_to_fp4_expert(
#endif
    int32_t numRows,            // 입력 행렬의 행 수
    int32_t numCols,            // 입력 행렬의 열 수
    Type const* in,             // 입력 데이터 포인터
    float const* SFScale,       // expert별 스케일 팩터 배열
    uint32_t* out,              // 출력 FP4 데이터 포인터
    uint32_t* SFout,            // 출력 스케일 팩터 포인터
    int32_t* mask,              // mask 배열 (조기 종료에 사용)
    bool use_silu_and_mul,      // SiLU 활성화와 곱셈을 사용할지 여부
    int n_experts               // expert 개수
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 스레드에서 expert로의 정적 매핑을 계산한다
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;      // expert마다 배정되는 기본 스레드 수
  int remainder = (gridDim.x * blockDim.x) % n_experts;   // 남는 스레드 수
  int expert_idx;         // 현재 스레드가 담당하는 expert 인덱스
  int tid_in_expert;      // expert 내에서의 스레드 로컬 인덱스
  int actual_stride;      // 실제 stride
  
  // 스레드 수가 expert 수로 나누어떨어지지 않는 경우를 처리한다
  if (remainder > 0) {
    int bound = remainder * (stride + 1);  // 앞쪽 remainder개 expert에 스레드를 하나씩 더 배정한다
    if (tid < bound) {
      // 앞쪽 expert에는 각각 (stride + 1)개의 스레드를 배정한다
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      // 뒤쪽 expert에는 각각 stride개의 스레드를 배정한다
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    // 스레드 수가 expert 수로 나누어떨어지는 경우
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  
  // expert별 데이터 차원을 계산한다
  int m = numRows / n_experts;                    // expert별 행 수
  int padded_m = (m + (128 - 1)) / 128 * 128;     // 128의 배수로 패딩

  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // TODO(kaixih@nvidia): 현재는 mask가 silu_and_mul과 함께 사용된다고 가정한다
  // 앞으로 더 범용적인 mask 동작이 필요할 수 있다. silu인 경우 입력의 마지막 차원이 두 배가 된다
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_silu_and_mul ? colsPerRow * 2 : colsPerRow;

  // 전역 스레드 하나가 element 하나를 처리하되, 현재 expert에 배정된 데이터만 처리한다
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow; 
       globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
    // 현재 전역 스레드가 처리해야 할 행과 열을 계산한다
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // expert 내 행 인덱스를 계산한다
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    // mask를 사용할 때의 조기 종료
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      break;  // 현재 expert의 유효 데이터는 모두 처리했다
    }

    // 입력 offset을 계산하고 데이터를 로드한다
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // SiLU 활성화와 곱셈을 사용하면 융합 연산을 수행한다
    if (use_silu_and_mul) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 출력 텐서의 offset을 얻는다
    // element 8개가 하나의 uint32_t로 패킹되므로 inOffset과 동일하다
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // SF에 적용할 global 스케일 팩터를 가져온다
    // 주의: SFScale은 다음 GEMM의 alpha와 같다. 즉 (448.f / (Alpha_A / 6.f))
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 현재 expert의 스케일 팩터 출력 주소를 계산한다
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 실제 output_scales 차원은 패딩된 numCols로부터 계산된다
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;

    // 스케일 팩터 출력 주소를 계산한다
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // FP16에서 FP4로의 변환을 수행한다
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// 작업량이 큰 경우에 최적화한 FP4 양자화 kernel (LARGE_M_TOPK = true)
// shared memory와 이진 탐색으로 대규모 expert 탐색을 최적화한다
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 4) cvt_fp16_to_fp4(  // 점유율을 높이기 위해 더 큰 block 크기 사용
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 입력 행렬의 행 수
    int32_t numCols,                        // 입력 행렬의 열 수
    Type const* in,                         // 입력 데이터 포인터
    float const* SFScale,                   // expert별 스케일 팩터 배열
    uint32_t* out,                          // 출력 FP4 데이터 포인터
    uint32_t* SFout,                        // 출력 스케일 팩터 포인터
    uint32_t* input_offset_by_experts,      // 입력에서 expert별 offset
    uint32_t* output_scale_offset_by_experts, // 출력 스케일 팩터에서 expert별 offset
    int32_t* mask,                          // mask 배열 (조기 종료에 사용)
    int n_experts                           // expert 개수
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");
  extern __shared__ uint32_t shared_input_offsets[];  // shared memory에 두는 expert offset 배열

  // 이후의 expert 탐색을 빠르게 하기 위해 입력 offset을 shared memory로 로드한다
  // expert 개수가 4보다 크면 int4 벡터화 로드를 사용해 명령 수를 아낀다
  // expert 개수가 4보다 작으면 그대로 읽는다
  if constexpr (SMALL_NUM_EXPERTS) {
    // expert 개수가 적은 경우: 스칼라 로드 사용
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    // expert 개수가 많은 경우: 벡터화 로드 사용 (한 번에 uint32_t 4개씩 로드)
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) = 
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    // 스레드 0이 마지막 offset 로드를 담당한다
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();  // 모든 스레드가 shared memory 로드를 마쳤음을 보장한다

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  bool use_mask = mask != nullptr;
  int actualColsPerRow = use_mask ? colsPerRow * 2 : colsPerRow;

  // 전역 스레드 하나가 element 하나를 처리한다
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 현재 전역 스레드가 처리해야 할 행과 열을 계산한다
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 이진 탐색으로 expert를 찾는다. m_topk가 큰 경우 성능이 더 좋다
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // shared memory를 통해 이진 탐색을 수행한다
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // offset을 얻는다: shared_input_offsets[i]는 input_offset_by_experts[i]에 대응한다
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        // 대응하는 expert를 찾았다
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;  // 왼쪽 절반에서 탐색
      } else {
        left = mid + 1;   // 오른쪽 절반에서 탐색
      }
    }

    // mask를 사용할 때의 조기 종료
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 입력 offset을 계산하고 데이터를 로드한다
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // mask를 사용하면 SiLU 활성화와 곱셈 융합 연산을 수행한다
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 출력 텐서의 offset을 얻는다
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // global 스케일 팩터를 가져온다
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    // 현재 expert의 스케일 팩터 출력 주소를 계산한다
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 스케일 팩터 출력 주소를 계산한다
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // FP16에서 FP4로의 변환을 수행한다
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// expert 모델 FP4 양자화의 범용 구현 함수
// 파라미터 설정에 따라 최적의 kernel 실행 전략을 선택한다
template <typename T>
void quant_impl(
    void* output,                       // 출력 FP4 데이터
    void* output_scale,                 // 출력 스케일 팩터
    void* input,                        // 입력 데이터
    void* input_global_scale,           // 입력 global 스케일 팩터
    void* input_offset_by_experts,      // expert 입력 offset
    void* output_scale_offset_by_experts, // expert 출력 스케일 팩터 offset
    void* mask,                         // mask 배열
    bool use_silu_and_mul,              // SiLU 활성화와 곱셈을 사용할지 여부
    int m_topk,                         // 입력 행 수 (top-k로 선택된 행 수)
    int k,                              // 입력 열 수
    int n_experts,                      // expert 개수
    cudaStream_t stream                 // CUDA stream
) {
  // TODO: 중복 조회를 피하도록 multiProcessorCount를 캐싱해야 한다
  int device;
  cudaGetDevice(&device);
  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

  // grid와 block 크기 설정
  // 스레드 하나가 값 8개를 변환한다
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  
  // SM당 block 수를 구한다 (SM을 충분히 활용할 수 있다고 가정)
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x), multiProcessorCount * numBlocksPerSM));
  
  // 점유율을 최적화하기 위해 grid와 block 크기를 동적으로 조정한다
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  // TODO(kaixih@nvidia): 임의의 grid 크기를 허용하도록 제약을 완화해야 한다
  // mask를 사용하면 전용 expert kernel을 사용한다
  if (mask != nullptr) {
    grid.x = (grid.x + n_experts - 1) / n_experts * n_experts;  // grid 크기가 expert 수의 배수가 되도록 보장한다
    cvt_fp16_to_fp4_expert<T, false><<<grid, block, 0, stream>>>(
        m_topk,
        k,
        reinterpret_cast<T*>(input),
        reinterpret_cast<float*>(input_global_scale),
        reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(output_scale),
        reinterpret_cast<int32_t*>(mask),
        use_silu_and_mul,
        n_experts);
    return;
  }

  // 각 block이 반복 실행해야 하는 횟수를 계산한다
  int const blockRepeat = (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  
  if (blockRepeat > 1) {
    // 작업량이 큰 경우: shared memory로 최적화한 kernel 사용
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    if (n_experts >= 4) {
      // expert 개수가 많은 경우: 벡터화 로드 사용
      cvt_fp16_to_fp4<T, false, false><<<grid, block, shared_mem_size, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    } else {
      // expert 개수가 적은 경우: 스칼라 로드 사용
      cvt_fp16_to_fp4<T, false, true><<<grid, block, shared_mem_size, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts);
    }
  } else {
    // 작업량이 작은 경우: 저지연으로 최적화한 kernel 사용 (shared memory 없음)
    if (n_experts >= 16) {
      // expert 개수가 많은 경우: register 최적화 사용
      cvt_fp16_to_fp4<T, false, false><<<grid, block, 0, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
          /* bool low_latency */ true);
    } else {
      // expert 개수가 적은 경우: 선형 탐색 사용
      cvt_fp16_to_fp4<T, false, true><<<grid, block, 0, stream>>>(
          m_topk, k,
          reinterpret_cast<T*>(input),
          reinterpret_cast<float*>(input_global_scale),
          reinterpret_cast<uint32_t*>(output),
          reinterpret_cast<uint32_t*>(output_scale),
          reinterpret_cast<uint32_t*>(input_offset_by_experts),
          reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
          reinterpret_cast<int32_t*>(mask),
          n_experts,
          /* bool low_latency */ true);
    }
  }
}

// Avoid redefinition warnings
#undef CHECK_CONTIGUOUS
#undef CHECK_TH_CUDA
#undef CHECK_INPUT

/*Quantization entry for fp4 experts quantization*/
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, m) \
  CHECK_TH_CUDA(x, m);    \
  CHECK_CONTIGUOUS(x, m);

// constexpr auto FP8 = at::ScalarType::Float8_e4m3fn;
constexpr auto HALF = at::ScalarType::Half;
constexpr auto BF16 = at::ScalarType::BFloat16;
constexpr auto FLOAT = at::ScalarType::Float;
constexpr auto INT = at::ScalarType::Int;
constexpr auto UINT8 = at::ScalarType::Byte;

// SM100 아키텍처 전용 expert 모델 FP4 양자화 PyTorch 인터페이스 함수
// 표준 expert 모델 양자화에 사용한다 (SiLU 활성화를 포함하지 않는다)
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                          // 출력 FP4 텐서 [m_topk, k/2]
    torch::Tensor& output_scale,                    // 출력 스케일 팩터 텐서
    torch::Tensor const& input,                     // 입력 FP16/BF16 텐서 [m_topk, k]
    torch::Tensor const& input_global_scale,        // 입력 global 스케일 팩터 [n_experts]
    torch::Tensor const& input_offset_by_experts,   // expert 입력 offset [n_experts+1]
    torch::Tensor const& output_scale_offset_by_experts // expert 출력 스케일 팩터 offset [n_experts+1]
) {
  // SM 아키텍처 버전을 확인한다
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 모든 입력 텐서의 기본 속성을 검증한다
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts must be a CUDA tensor");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts must be a CUDA tensor");

  // 텐서 차원을 검증한다
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  // 텐서 데이터 타입을 검증한다
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output은 uint8이다 (nvfp4 값 두 개가 하나의 uint8로 패킹된다)
  // output_scale은 int32다 (fp8 값 네 개가 하나의 int32로 패킹된다)
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 텐서 shape과 크기 제약을 검증한다
  const int BLOCK_SIZE = 16;  // FP4 양자화의 block 크기
  auto m_topk = input.size(0);
  auto k = input.size(1);
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4는 저장 공간을 절반만 차지한다
  
  // 스케일 팩터 텐서의 크기를 검증한다
  int scales_k = k / BLOCK_SIZE;
  // 4는 nvidia nvfp4의 swizzle 요구사항을 의미한다
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4는 fp8 값 네 개가 하나의 int32로 패킹됨을 의미한다
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // CUDA 디바이스와 stream을 설정한다
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 입력 데이터 타입에 따라 해당 구현으로 dispatch한다
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask (mask 없음)
        false,    // use_silu_and_mul (SiLU 활성화를 사용하지 않음)
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask (mask 없음)
        false,    // use_silu_and_mul (SiLU 활성화를 사용하지 않음)
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}

// SM100 아키텍처 전용 SiLU 활성화 + 곱셈 + expert 모델 FP4 양자화 PyTorch 인터페이스 함수
// SiLU 활성화 함수를 포함하는 expert 모델 양자화에 사용한다
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                      // 출력 FP4 텐서 [m_topk, k/2]
    torch::Tensor& output_scale,                // 출력 스케일 팩터 텐서
    torch::Tensor const& input,                 // 입력 FP16/BF16 텐서 [m_topk, k*2 or k]
    torch::Tensor const& input_global_scale,    // 입력 global 스케일 팩터 [n_experts]
    torch::Tensor const& mask,                  // mask 텐서 [n_experts]
    bool use_silu_and_mul                       // SiLU 활성화와 곱셈을 사용할지 여부
) {
  // SM 아키텍처 버전을 확인한다
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 모든 입력 텐서의 기본 속성을 검증한다
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(mask, "mask must be a CUDA tensor");

  // 텐서 차원을 검증한다
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);

  // 텐서 데이터 타입을 검증한다
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(mask.scalar_type() == INT);
  // output은 uint8이다 (nvfp4 값 두 개가 하나의 uint8로 패킹된다)
  // output_scale은 int32다 (fp8 값 네 개가 하나의 int32로 패킹된다)
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 텐서 shape과 크기 제약을 검증한다
  const int BLOCK_SIZE = 16;  // FP4 양자화의 block 크기
  auto m_topk = input.size(0);
  auto k_by_2 = input.size(1);
  auto k = k_by_2;
  
  // SiLU 활성화와 곱셈을 사용하면 입력 차원이 두 배가 된다 (게이팅 벡터를 포함하기 때문)
  if (use_silu_and_mul) {
    TORCH_CHECK(k_by_2 % 2 == 0, "k must be a multiple of 2");
    k = k_by_2 / 2;
  }
  
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(mask.size(0) == n_experts);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4는 저장 공간을 절반만 차지한다
  
  // 스케일 팩터 텐서의 크기를 검증한다
  int scales_k = k / BLOCK_SIZE;
  // 4는 nvidia nvfp4의 swizzle 요구사항을 의미한다
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4는 fp8 값 네 개가 하나의 int32로 패킹됨을 의미한다
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // CUDA 디바이스와 stream을 설정한다
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 입력 데이터 타입에 따라 해당 구현으로 dispatch한다
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts (expert offset 없음)
        nullptr,  // output_scale_offset_by_experts (expert offset 없음)
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts (expert offset 없음)
        nullptr,  // output_scale_offset_by_experts (expert offset 없음)
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}
```