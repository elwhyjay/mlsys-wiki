# SM100 NVFP4 Quant 구현 노트

## `nvfp4_quant_entry.cu` 구현

```c++
// 관련 내용여부관련 내용사용NVFP4지원（관련 내용에서SM100관련 내용상가능사용）
#if defined ENABLE_NVFP4 && ENABLE_NVFP4

// SM100관련 내용사용의FP4관련 내용함수관련 내용
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,        // 출력관련 내용후의FP4텐서
    torch::Tensor const& input,   // 입력의FP16/BF16텐서
    torch::Tensor& output_sf,     // 출력관련 내용
    torch::Tensor const& input_sf // 입력관련 내용
);

// SM100관련 내용사용의expert모델FP4관련 내용함수관련 내용
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                              // 출력관련 내용후의FP4텐서
    torch::Tensor& output_scale,                        // 출력관련 내용
    torch::Tensor const& input,                         // 입력텐서
    torch::Tensor const& input_global_scale,            // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    torch::Tensor const& input_offset_by_experts,       // expert관련 내용
    torch::Tensor const& output_scale_offset_by_experts // 출력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
);

// SM100관련 내용사용의SiLU이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+ +expert)모델FP4관련 내용함수관련 내용
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                       // 출력관련 내용후의FP4텐서
    torch::Tensor& output_scale,                 // 출력관련 내용
    torch::Tensor const& input,                  // 입력텐서
    torch::Tensor const& input_global_scale,     // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    torch::Tensor const& mask,                   // mask텐서（사용된다gating）
    bool use_silu_and_mul                        // 여부관련 내용사용SiLU관련 내용와관련 내용
);

#endif

// 관련 내용사용FP4관련 내용인터페이스함수
void scaled_fp4_quant(
    torch::Tensor& output,        // 출력관련 내용후의FP4텐서
    torch::Tensor const& input,   // 입력의FP16/BF16텐서
    torch::Tensor& output_sf,     // 출력관련 내용
    torch::Tensor const& input_sf // 입력관련 내용
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 만약지원NVFP4，호출한다SM100관련 내용사용구현
  return scaled_fp4_quant_sm100a(output, input, output_sf, input_sf);
#endif
  // 만약아니지원NVFP4，관련 내용구현관련 내용
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 quantization");
}

// 관련 내용사용expert모델FP4관련 내용인터페이스함수
void scaled_fp4_experts_quant(
    torch::Tensor& output,                              // 출력관련 내용후의FP4텐서
    torch::Tensor& output_scale,                        // 출력관련 내용
    torch::Tensor const& input,                         // 입력텐서
    torch::Tensor const& input_global_scale,            // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    torch::Tensor const& input_offset_by_experts,       // expert관련 내용
    torch::Tensor const& output_scale_offset_by_experts // 출력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 만약지원NVFP4，호출한다SM100관련 내용사용구현
  return scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, input_offset_by_experts, output_scale_offset_by_experts);
#endif
  // 만약아니지원NVFP4，관련 내용구현관련 내용
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}

// 관련 내용사용SiLU이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+ +expert)모델FP4관련 내용인터페이스함수
void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output,                       // 출력관련 내용후의FP4텐서
    torch::Tensor& output_scale,                 // 출력관련 내용
    torch::Tensor const& input,                  // 입력텐서
    torch::Tensor const& input_global_scale,     // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    torch::Tensor const& mask,                   // mask텐서（사용된다gating）
    bool use_silu_and_mul                        // 여부관련 내용사용SiLU관련 내용와관련 내용
) {
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
  // 만약지원NVFP4，호출한다SM100관련 내용사용구현
  return silu_and_mul_scaled_fp4_experts_quant_sm100a(
      output, output_scale, input, input_global_scale, mask, use_silu_and_mul);
#endif
  // 만약아니지원NVFP4，관련 내용구현관련 내용
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled nvfp4 experts quantization kernel");
}
```

## `nvfp4_quant.cuh` 구현

```c++
// 관련 내용의관련 내용파일관련 내용
#include <cuda.h>           // CUDA이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row API)
#include <cuda_fp8.h>       // FP8관련 내용지원
#include <cutlass/arch/config.h>  // CUTLASS관련 내용설정

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다에서관련 내용와이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용된다half와bfloat16）
template <typename T>
struct TypeConverter {
  using Type = half2;  // 기본관련 내용로half2관련 내용
};  // 관련 내용사용관련 내용

// half2 -> half 의관련 내용
template <>
struct TypeConverter<half2> {
  using Type = half;
};

// half -> half2 의관련 내용
template <>
struct TypeConverter<half> {
  using Type = half2;
};

// __nv_bfloat162 -> __nv_bfloat16 의관련 내용
template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

// __nv_bfloat16 -> __nv_bfloat162 의관련 내용
template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// 각개thread관련 내용의관련 내용개수
#define ELTS_PER_THREAD 8

// FP4관련 내용관련관련 내용
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;  // 각개thread관련 내용의FP4관련 내용개수
constexpr int CVT_FP4_SF_VEC_SIZE = 16;     // FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다크기

// 할 것이다8개float32관련 내용로8개e2m1관련 내용로관련 내용개uint32_t）
// e2m1이다FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (2 1)
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  // 관련 내용사용의PTX이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (sm100a/sm103a)지원
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg.b8 byte0;\n"                                    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (8 register)
      ".reg.b8 byte1;\n"
      ".reg.b8 byte2;\n"
      ".reg.b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"     // 관련 내용개float32로e2m1x2관련 내용
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"     // rn=round to nearest, satfinite=관련 내용와있다관련 내용
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"         // 할 것이다4개관련 내용로32관련 내용
      "}"
: "=r"(val)                                            // 출력：32이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (register)
: "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),  // 입력：8개floatregister
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
#else
  return 0;  // 아니지원의관련 내용반환한다0
#endif
}

// 할 것이다4개float2관련 내용로8개e2m1관련 내용로관련 내용개uint32_t）
// 관련 내용이다위함수의float2관련 내용버전
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  // 관련 내용사용의PTX이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (sm100a/sm103a)지원
#if CUTLASS_ARCH_MMA_SM100A_ENABLED || CUTLASS_ARCH_MMA_SM103A_ENABLED
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg.b8 byte0;\n"
      ".reg.b8 byte1;\n"
      ".reg.b8 byte2;\n"
      ".reg.b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
: "=r"(val)
: "f"(array[0].x), "f"(array[0].y),  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (float2)의x와y관련 내용
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// 빠른이 부분은 원문의 해당 기술 설명을 이어서 서술한다계산（flush-to-zero모드）
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n": "=f"(b): "f"(a));
  return b;
}

// 계산FP4관련 내용중관련 내용의출력관련 내용
// SFType: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CVT_FP4_NUM_THREADS_PER_SF:)각개관련 내용의thread관련 내용
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numCols, SFType* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 관련 내용에서SM100+관련 내용상지원
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 관련 내용대해thread할 것이다관련 내용개관련 내용쓰기global memory
  // TODO: 통해shared memory관련 내용로지원관련 내용의STG.32관련 내용
  // 관련 내용여부관련 내용 (4)개thread의STG.8더좋은？
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다인덱스（K차원상16개이 부분은 원문의 해당 기술 설명을 이어서 서술한다개관련 내용
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 대응인덱스：[mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // 계산M차원의tile인덱스
    int32_t mTileIdx = mIdx / (32 * 4);
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다크기로16
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;  // 관련 내용상관련 내용
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    // 계산K차원의tile인덱스와관련 내용
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile관련 내용[32, 4]이다column관련 내용의
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride + outerMIdx * outerMStride +
                       innerMIdx * innerMStride + innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
#endif
  return nullptr;
}

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // 관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다얻는다대응의관련 내용
};

// FP8 e4m3관련 내용의관련 내용버전
template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];  // 8개FP8x2이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)개FP8관련 내용
};

```

## `nvfp4_quant_kernels.cu` 구현

```c++
// 할 것이다PackedVec관련 내용로FP4관련 내용그리고출력로uint32_t
// Type: 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (half/bfloat16), UE8M0_SF: 여부관련 내용사용UE8M0관련 내용의관련 내용
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 관련 내용에서SM100+관련 내용상지원
  // 얻는다관련 내용 (8)개관련 내용중의관련 내용대해관련 내용큰관련 내용
  auto localMax = __habs2(vec.elts[0]);

  // 계산관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다최적화）
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 계산half2의관련 내용대해관련 내용큰관련 내용
  }

  // 통해warp shuffle얻는다관련 내용있다16개관련 내용중의관련 내용대해관련 내용큰관련 내용개thread관련 내용
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 얻는다관련 내용의관련 내용대해관련 내용큰관련 내용
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SF)큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (/ e2m1)큰관련 내용
  // e2m1관련 내용의관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (= 6.0)
  // TODO: 관련 내용사용half관련 내용로계산관련 내용로높인다성능
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 관련 내용의8관련 내용
  uint8_t fp8SFVal;
  
  // 관련 내용파라미터선택이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  if constexpr (UE8M0_SF) {
    // 관련 내용사용UE8M0이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (8 0)
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // 관련 내용사용E4M3이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (4 3)
    // 여기SFValue관련 내용이다관련 내용그래서E4M3이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (UE4M3)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }
  
  // 계산출력관련 내용
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: final_scale = 1 / fp32 fp8 SFValue)* SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue!= 0? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)): 0.0f;

  // 만약관련 내용출력포인터，할 것이다관련 내용쓰기global memory（8관련 내용
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 할 것이다입력관련 내용로float2배열
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 관련 내용입력관련 내용수행한다관련 내용
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 응용출력관련 내용
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 관련 내용로e2m1이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FP4)
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 반환한다관련 내용의e2m1관련 내용
  return e2m1Vec;
#else
  return 0;  // 아니지원의관련 내용반환한다0
#endif
}

// FP16/BF16까지FP4관련 내용의CUDAkernel
// 기본관련 내용사용UE4M3관련 내용의관련 내용
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+관련 내용상의시작관련 내용최적화
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,        // 입력matrixrow관련 내용
    int32_t numCols,        // 입력matrixcolumn관련 내용
    Type const* in,         // 입력관련 내용포인터
    float const* SFScale,   // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    uint32_t* out,          // 출력FP4관련 내용포인터
    uint32_t* SFout         // 출력관련 내용포인터
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다할 것이다응용관련 내용 (SF)
  // 주의：SFScale와하관련 내용개GEMM의alpha이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.f / Alpha_A / 6.f)
  float const SFScaleVal = SFScale == nullptr? 1.0f: SFScale[0];

  // 입력텐서의row/column관련 내용
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; colIdx += blockDim.x) {
      // 계산입력관련 내용
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 얻는다출력텐서관련 내용
      // 와inOffset관련 내용왜냐하면8개관련 내용로관련 내용개uint32_t
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // 계산관련 내용출력관련 내용
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numCols, SFout);

      // 실행한다FP16까지FP4의관련 내용
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif
}

// FP4관련 내용의관련 내용호출한다함수관련 내용
template <typename T>
void invokeFP4Quantization(
    int m,                      // matrixrow관련 내용
    int n,                      // matrixcolumn관련 내용
    T const* input,             // 입력관련 내용포인터
    float const* SFScale,       // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    int64_t* output,            // 출력FP4관련 내용
    int32_t* SFOuput,           // 출력관련 내용
    bool useUE8M0,              // 여부관련 내용사용UE8M0관련 내용
    int multiProcessorCount,    // SM개수
    cudaStream_t stream         // CUDA관련 내용
) {
  // 관련 내용와block크기설정
  // 각개thread관련 내용 (8)개관련 내용
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // 얻는다각개SM의block관련 내용가능로관련 내용사용SM）
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // 시작이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다함수이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (- half)
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

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다함수이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (- bfloat16)
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

// 얻는다현재GPU의많은관련 내용개수（관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (cache)최적화）
inline int getMultiProcessorCount() {
  static int multi_processor_count = []() {
    int device_id = 0;
    int count = 0;

    // 얻는다현재CUDA관련 내용 (ID)
    CHECK_CUDA_SUCCESS(cudaGetDevice(&device_id));

    // 얻는다현재관련 내용의많은관련 내용개수
    CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_id));

    return count;  // 초기화관련 내용변수
  }();

  return multi_processor_count;  // 후관련 내용호출한다반환한다cache관련 내용
}

// SM100관련 내용사용의FP4관련 내용구현함수
void scaled_fp4_quant_sm100a(
    torch::Tensor& output,          // 출력FP4텐서
    torch::Tensor const& input,     // 입력FP16/BF16텐서
    torch::Tensor& output_sf,       // 출력관련 내용텐서
    torch::Tensor const& input_sf   // 입력관련 내용텐서
) {
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SM)버전
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 얻는다입력텐서차원
  int32_t m = input.size(0);  // row관련 내용
  int32_t n = input.size(1);  // column관련 내용

  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)반드시이다16의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FP4)
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  // 얻는다GPU많은관련 내용개수
  int multiProcessorCount = getMultiProcessorCount();

  // 얻는다관련 내용포인터
  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUDA)와관련 내용
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // 현재아니지원e8m0이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  bool useUE8M0 = false;

  // 관련 내용입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다까지관련 내용의관련 내용함수
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
// expert모델관련 내용사용의FP16까지FP4관련 내용함수（와관련 내용버전관련 내용의관련 내용
// Type: 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (half/bfloat16), UE8M0_SF: 여부관련 내용사용UE8M0관련 내용의관련 내용
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)  // 관련 내용에서SM100+관련 내용상지원
  // 얻는다관련 내용 (8)개관련 내용중의관련 내용대해관련 내용큰관련 내용
  auto localMax = __habs2(vec.elts[0]);

  // 계산관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다최적화）
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 계산half2의관련 내용대해관련 내용큰관련 내용
  }

  // 통해warp shuffle얻는다관련 내용있다16개관련 내용중의관련 내용대해관련 내용큰관련 내용개thread관련 내용
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // 얻는다관련 내용의관련 내용대해관련 내용큰관련 내용
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SF)큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (/ e2m1)큰관련 내용
  // e2m1관련 내용의관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (= 6.0)
  // TODO: 관련 내용사용half관련 내용로계산관련 내용로높인다성능
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 관련 내용의8관련 내용
  uint8_t fp8SFVal;
  
  // 관련 내용파라미터선택이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  if constexpr (UE8M0_SF) {
    // 부터float32중이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (8)
    // float 32이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (= 1 + 8 + 23)
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (fp32)
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // 여기SFValue관련 내용이다관련 내용그래서E4M3이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (UE4M3)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (fp32)
    SFValue = float(tmp);
  }
  
  // 계산출력관련 내용
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: final_scale = 1 / fp32 fp8 SFValue)* SFScaleVal)) / SFScaleVal)
  float outputScale =
      SFValue!= 0? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)): 0.0f;

  // 만약관련 내용출력포인터，할 것이다관련 내용쓰기global memory（8관련 내용
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 할 것이다입력관련 내용로float2배열
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 관련 내용입력관련 내용수행한다관련 내용
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);        // half2 -> float2
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);    // bfloat162 -> float2
    }
    // 응용출력관련 내용
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 관련 내용로e2m1이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FP4)
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // 반환한다관련 내용의e2m1관련 내용
  return e2m1Vec;
#else
  return 0;  // 아니지원의관련 내용반환한다0
#endif
}

// SiLU관련 내용함수구현：silu(x) = x / (1 + exp(-x))
// 도관련 내용로Swish관련 내용함수，에서Transformer모델중관련 내용사용
__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

// SiLU관련 내용함수와관련 내용의융합관련 내용
// 대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)모델，관련 내용수행한다gating이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (silu x)* y
template <class Type>
inline __device__ void silu_and_mul(PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  float2 x[CVT_FP4_ELTS_PER_THREAD / 2];  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (x)의float2관련 내용
  float2 y[CVT_FP4_ELTS_PER_THREAD / 2];  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (y)의float2관련 내용

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    // 관련 내용입력관련 내용수행한다관련 내용와계산
    if constexpr (std::is_same_v<Type, half>) {
      x[i] = __half22float2(x_vec.elts[i]);     // half2 -> float2
      y[i] = __half22float2(y_vec.elts[i]);     // half2 -> float2
      x[i].x = silu(x[i].x) * y[i].x;           // 대해x관련 내용응용silu그리고관련 내용로y
      x[i].y = silu(x[i].y) * y[i].y;           // 대해y관련 내용응용silu그리고관련 내용로y
      x_vec.elts[i] = __float22half2_rn(x[i]);  // float2 -> half2 (round to nearest)
    } else {
      x[i] = __bfloat1622float2(x_vec.elts[i]);     // bfloat162 -> float2
      y[i] = __bfloat1622float2(y_vec.elts[i]);     // bfloat162 -> float2
      x[i].x = silu(x[i].x) * y[i].x;               // 대해x관련 내용응용silu그리고관련 내용로y
      x[i].y = silu(x[i].y) * y[i].y;               // 대해y관련 내용응용silu그리고관련 내용로y
      x_vec.elts[i] = __float22bfloat162_rn(x[i]);  // float2 -> bfloat162 (round to nearest)
    }
  }
}

// expert모델FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)지원이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)와SiLU관련 내용
// 기본관련 내용사용UE4M3관련 내용의관련 내용
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4(  // SM100+관련 내용상의시작관련 내용최적화
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 입력matrixrow관련 내용
    int32_t numCols,                        // 입력matrixcolumn관련 내용
    Type const* in,                         // 입력관련 내용포인터
    float const* SFScale,                   // 각개expert의관련 내용배열
    uint32_t* out,                          // 출력FP4관련 내용포인터
    uint32_t* SFout,                        // 출력관련 내용포인터
    uint32_t* input_offset_by_experts,      // 각개expert에서입력중의관련 내용
    uint32_t* output_scale_offset_by_experts, // 각개expert에서출력관련 내용중의관련 내용
    int32_t* mask,                          // mask배열（사용된다관련 내용
    int n_experts,                          // expert개수
    bool low_latency                        // 여부관련 내용사용낮은latency모드
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 계산thread와관련 내용인덱스
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  
  // TODO(kaixih@nvidia): 현재이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (mask)와silu_and_mul관련 내용사용
  // 관련 내용와서가능가능관련 내용더관련 내용사용의maskrow로。에서silu관련 내용하，입력의마지막으로관련 내용된다관련 내용
  bool use_mask = mask!= nullptr;
  int actualColsPerRow = use_mask? colsPerRow * 2: colsPerRow;

  // 각개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)개관련 내용
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 계산현재이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)의row와column
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)개수관련 내용사용아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의인덱스
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      // 작은expert개수：관련 내용
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);    // cache로드현재관련 내용
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);   // cache로드하관련 내용개관련 내용
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // 큰expert개수：이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (blockvectorization)로드최적화
      // 관련 내용배열크기관련 내용로17이다왜냐하면register관련 내용
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        // 관련 내용사용int4vectorization로드16개관련 내용각관련 내용로드4개uint32_t）
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(&input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

        // 관련 내용로드의16개관련 내용
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

    // 관련 내용사용mask관련 내용의관련 내용
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 계산입력관련 내용그리고로드관련 내용
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 만약관련 내용사용mask，실행한다SiLU관련 내용와관련 내용융합관련 내용
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 얻는다출력텐서관련 내용
    // 와inOffset관련 내용왜냐하면8개관련 내용로관련 내용개uint32_t
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다할 것이다응용관련 내용 (SF)
    // 주의：SFScale와하관련 내용개GEMM의alpha이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.f / Alpha_A / 6.f)
    float const SFScaleVal = SFScale == nullptr? 1.0f: SFScale[expert_idx];

    // 계산현재expert의관련 내용출력관련 내용
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 관련 내용의output_scales차원부터관련 내용의numCols계산관련 내용
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 계산관련 내용출력관련 내용
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 실행한다FP16까지FP4의관련 내용
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// expert관련 내용사용FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel thread)까지expert관련 내용
// 기본관련 내용사용UE4M3관련 내용의관련 내용
template <class Type, bool UE8M0_SF = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(512, 4) cvt_fp16_to_fp4_expert(  // SM100+관련 내용상의시작관련 내용최적화
#else
cvt_fp16_to_fp4_expert(
#endif
    int32_t numRows,            // 입력matrixrow관련 내용
    int32_t numCols,            // 입력matrixcolumn관련 내용
    Type const* in,             // 입력관련 내용포인터
    float const* SFScale,       // 각개expert의관련 내용배열
    uint32_t* out,              // 출력FP4관련 내용포인터
    uint32_t* SFout,            // 출력관련 내용포인터
    int32_t* mask,              // mask배열（사용된다관련 내용
    bool use_silu_and_mul,      // 여부관련 내용사용SiLU관련 내용와관련 내용
    int n_experts               // expert개수
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");

  // 계산thread까지expert의관련 내용
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = (gridDim.x * blockDim.x) / n_experts;      // 각개expert할당의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)
  int remainder = (gridDim.x * blockDim.x) % n_experts;   // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)
  int expert_idx;         // 현재thread담당의expert인덱스
  int tid_in_expert;      // thread에서expert관련 내용의관련 내용인덱스
  int actual_stride;      // 관련 내용
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)아니가능이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의관련 내용
  if (remainder > 0) {
    int bound = remainder * (stride + 1);  // 전remainder개expert많은할당관련 내용개thread
    if (tid < bound) {
      // 전관련 내용의expert，각개할당(stride + 1)개thread
      expert_idx = tid / (stride + 1);
      tid_in_expert = tid % (stride + 1);
      actual_stride = stride + 1;
    } else {
      // 후관련 내용의expert，각개할당stride개thread
      expert_idx = remainder + (tid - bound) / stride;
      tid_in_expert = (tid - bound) % stride;
      actual_stride = stride;
    }
  } else {
    // thread관련 내용가능이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의관련 내용
    expert_idx = tid / stride;
    tid_in_expert = tid % stride;
    actual_stride = stride;
  }
  
  // 계산각개expert의관련 내용차원
  int m = numRows / n_experts;                    // 각개expert의row관련 내용
  int padded_m = (m + (128 - 1)) / 128 * 128;     // 관련 내용까지128의관련 내용

  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  // TODO(kaixih@nvidia): 현재이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (mask)와silu_and_mul관련 내용사용
  // 관련 내용와서가능가능관련 내용더관련 내용사용의maskrow로。에서silu관련 내용하，입력의마지막으로관련 내용된다관련 내용
  bool use_mask = mask!= nullptr;
  int actualColsPerRow = use_silu_and_mul? colsPerRow * 2: colsPerRow;

  // 각개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)개관련 내용만관련 내용할당관련 내용현재expert의관련 내용
  for (int globalIdx = tid_in_expert + expert_idx * m * colsPerRow; 
       globalIdx < (expert_idx + 1) * m * colsPerRow;
       globalIdx += actual_stride) {
    // 계산현재이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)의row와column
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 계산expert관련 내용의row인덱스
    int rowIdx_in_expert = rowIdx - expert_idx * m;

    // 관련 내용사용mask관련 내용의관련 내용
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      break;  // 현재expert의있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    }

    // 계산입력관련 내용그리고로드관련 내용
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 만약관련 내용사용SiLU관련 내용와관련 내용실행한다융합관련 내용
    if (use_silu_and_mul) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 얻는다출력텐서관련 내용
    // 와inOffset관련 내용왜냐하면8개관련 내용로관련 내용개uint32_t
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다할 것이다응용관련 내용 (SF)
    // 주의：SFScale와하관련 내용개GEMM의alpha이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.f / Alpha_A / 6.f)
    float const SFScaleVal = SFScale == nullptr? 1.0f: SFScale[expert_idx];

    // 계산현재expert의관련 내용출력관련 내용
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    // 관련 내용의output_scales차원부터관련 내용의numCols계산관련 내용
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + expert_idx * padded_m * numCols_SFout;

    // 계산관련 내용출력관련 내용
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 실행한다FP16까지FP4의관련 내용
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// 큰관련 내용최적화버전의FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel LARGE_M_TOPK = true)
// 관련 내용사용shared memory와관련 내용최적화큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)
template <class Type, bool UE8M0_SF = false, bool SMALL_NUM_EXPERTS = false>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(1024, 4) cvt_fp16_to_fp4(  // 더큰의block크기로높인다관련 내용사용관련 내용
#else
cvt_fp16_to_fp4(
#endif
    int32_t numRows,                        // 입력matrixrow관련 내용
    int32_t numCols,                        // 입력matrixcolumn관련 내용
    Type const* in,                         // 입력관련 내용포인터
    float const* SFScale,                   // 각개expert의관련 내용배열
    uint32_t* out,                          // 출력FP4관련 내용포인터
    uint32_t* SFout,                        // 출력관련 내용포인터
    uint32_t* input_offset_by_experts,      // 각개expert에서입력중의관련 내용
    uint32_t* output_scale_offset_by_experts, // 각개expert에서출력관련 내용중의관련 내용
    int32_t* mask,                          // mask배열（사용된다관련 내용
    int n_experts                           // expert개수
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD, "Vec size is not matched.");
  extern __shared__ uint32_t shared_input_offsets[];  // shared memory중의expert관련 내용배열

  // 할 것이다입력관련 내용로드까지shared memory중로가속후관련 내용의expert관련 내용
  // 만약expert개수큰관련 내용 (4)사용vectorizationint4로드로이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  // 만약expert개수작은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (4)읽기
  if constexpr (SMALL_NUM_EXPERTS) {
    // 작은expert개수：관련 내용사용관련 내용로드
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    // 큰expert개수：관련 내용사용vectorization로드（각관련 내용로드4개uint32_t）
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) = 
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    // thread0담당로드마지막으로관련 내용개관련 내용
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();  // 보장관련 내용있다thread완료shared memory로드

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  bool use_mask = mask!= nullptr;
  int actualColsPerRow = use_mask? colsPerRow * 2: colsPerRow;

  // 각개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)개관련 내용
  for (int globalIdx = tid; globalIdx < numRows * colsPerRow; globalIdx += gridDim.x * blockDim.x) {
    // 계산현재이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)의row와column
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // 관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)에서큰m_topk관련 내용하성능더좋은
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // 통해shared memory수행한다관련 내용
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (shared_input_offsets)[i]대응input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        // 관련 내용까지대응의expert
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;  // 에서관련 내용부분관련 내용
      } else {
        left = mid + 1;   // 에서관련 내용부분관련 내용
      }
    }

    // 관련 내용사용mask관련 내용의관련 내용
    if (use_mask && rowIdx_in_expert >= mask[expert_idx]) {
      continue;
    }

    // 계산입력관련 내용그리고로드관련 내용
    int64_t inOffset = rowIdx * actualColsPerRow + colIdx;
    PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
    
    // 만약관련 내용사용mask，실행한다SiLU관련 내용와관련 내용융합관련 내용
    if (use_mask) {
      PackedVec in_vec_mul = reinterpret_cast<PackedVec const*>(in)[inOffset + colsPerRow];
      silu_and_mul(in_vec, in_vec_mul);
    }

    // 얻는다출력텐서관련 내용
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    float const SFScaleVal = SFScale == nullptr? 1.0f: SFScale[expert_idx];

    // 계산현재expert의관련 내용출력관련 내용
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numCols_padded = (numCols + factor - 1) / factor * factor;
    int numCols_SFout = numCols_padded / CVT_FP4_SF_VEC_SIZE / 4;
    uint32_t* SFout_in_expert = SFout + output_scale_offset_by_experts[expert_idx] * numCols_SFout;

    // 계산관련 내용출력관련 내용
    auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
        rowIdx_in_expert, colIdx, numCols, SFout_in_expert);

    // 실행한다FP16까지FP4의관련 내용
    out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
  }
#endif
}

// expert모델FP4관련 내용의관련 내용사용구현함수
// 관련 내용아니관련 내용의파라미터설정선택관련 내용의kernel시작관련 내용
template <typename T>
void quant_impl(
    void* output,                       // 출력FP4관련 내용
    void* output_scale,                 // 출력관련 내용
    void* input,                        // 입력관련 내용
    void* input_global_scale,           // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    void* input_offset_by_experts,      // expert입력관련 내용
    void* output_scale_offset_by_experts, // expert출력이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    void* mask,                         // mask배열
    bool use_silu_and_mul,              // 여부관련 내용사용SiLU관련 내용와관련 내용
    int m_topk,                         // 입력row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (top-k)선택의row관련 내용
    int k,                              // 입력column관련 내용
    int n_experts,                      // expert개수
    cudaStream_t stream                 // CUDA관련 내용
) {
  // TODO: multiProcessorCount이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (cache)로이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  int device;
  cudaGetDevice(&device);
  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

  // 관련 내용와block크기설정
  // 각개thread관련 내용 (8)개관련 내용
  int const workSizePerRow = k / ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  
  // 얻는다각개SM의block관련 내용가능로관련 내용사용SM）
  int const numBlocksPerSM = 2048 / block.x;
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x), multiProcessorCount * numBlocksPerSM));
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다와block크기로최적화관련 내용사용관련 내용
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  // TODO(kaixih@nvidia): 이 부분은 원문의 해당 기술 설명을 이어서 서술한다로이 부분은 원문의 해당 기술 설명을 이어서 서술한다크기
  // 만약관련 내용사용mask，관련 내용사용관련 내용의expertkernel
  if (mask!= nullptr) {
    grid.x = (grid.x + n_experts - 1) / n_experts * n_experts;  // 보장관련 내용크기이다expert관련 내용의관련 내용
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

  // 계산각개block관련 내용실행한다의관련 내용
  int const blockRepeat = (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  
  if (blockRepeat > 1) {
    // 큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용shared memory최적화의kernel
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    if (n_experts >= 4) {
      // 큰expert개수：관련 내용사용vectorization로드
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
      // 작은expert개수：관련 내용사용관련 내용로드
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
    // 작은이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용낮은latency최적화의kernel（없음shared memory）
    if (n_experts >= 16) {
      // 큰expert개수：관련 내용사용register최적화
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
      // 작은expert개수：관련 내용사용관련 내용
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

// SM100관련 내용사용의expert모델FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (PyTorch)인터페이스함수
// 사용된다관련 내용의expert모델관련 내용아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SiLU)
void scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                          // 출력FP4텐서 [m_topk, k/2]
    torch::Tensor& output_scale,                    // 출력관련 내용텐서
    torch::Tensor const& input,                     // 입력FP16/BF16텐서 [m_topk, k]
    torch::Tensor const& input_global_scale,        // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다[n_experts]
    torch::Tensor const& input_offset_by_experts,   // expert입력관련 내용[n_experts+1]
    torch::Tensor const& output_scale_offset_by_experts // expert출력이 부분은 원문의 해당 기술 설명을 이어서 서술한다[n_experts+1]
) {
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SM)버전
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 검증관련 내용있다입력텐서의관련 내용
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts must be a CUDA tensor");
  CHECK_INPUT(output_scale_offset_by_experts, "output_scale_offset_by_experts must be a CUDA tensor");

  // 검증텐서차원
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  // 검증텐서관련 내용
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == INT);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == INT);
  // output이다uint8（관련 내용개nvfp4관련 내용로관련 내용개uint8）
  // output_scale이다int32（관련 내용개fp8관련 내용로관련 내용개int32）
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 검증텐서shape와크기관련 내용
  const int BLOCK_SIZE = 16;  // FP4관련 내용의block크기
  auto m_topk = input.size(0);
  auto k = input.size(1);
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  
  // 검증관련 내용텐서의크기
  int scales_k = k / BLOCK_SIZE;
  // 4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (nvidia nvfp4)의swizzle관련 내용
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4관련 내용 (4)개fp8관련 내용로관련 내용개int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUDA)와관련 내용
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 관련 내용입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다까지관련 내용의구현
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask（없음mask）
        false,    // use_silu_and_mul（아니관련 내용사용SiLU관련 내용
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(),
        nullptr,  // mask（없음mask）
        false,    // use_silu_and_mul（아니관련 내용사용SiLU관련 내용
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}

// SM100관련 내용사용의SiLU이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+ +expert)모델FP4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (PyTorch)인터페이스함수
// 사용된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SiLU)함수의expert모델관련 내용
void silu_and_mul_scaled_fp4_experts_quant_sm100a(
    torch::Tensor& output,                      // 출력FP4텐서 [m_topk, k/2]
    torch::Tensor& output_scale,                // 출력관련 내용텐서
    torch::Tensor const& input,                 // 입력FP16/BF16텐서 [m_topk, k*2 or k]
    torch::Tensor const& input_global_scale,    // 입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다[n_experts]
    torch::Tensor const& mask,                  // mask텐서 [n_experts]
    bool use_silu_and_mul                       // 여부관련 내용사용SiLU관련 내용와관련 내용
) {
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SM)버전
  auto sm_version = getSMVersion();
  TORCH_CHECK(sm_version == 100 || sm_version == 103, "fp4_quant is only supported on sm100a/sm103a");

  // 검증관련 내용있다입력텐서의관련 내용
  CHECK_INPUT(output, "output must be a CUDA tensor");
  CHECK_INPUT(output_scale, "output_scale must be a CUDA tensor");
  CHECK_INPUT(input, "input must be a CUDA tensor");
  CHECK_INPUT(input_global_scale, "input_global_scale must be a CUDA tensor");
  CHECK_INPUT(mask, "mask must be a CUDA tensor");

  // 검증텐서차원
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);

  // 검증텐서관련 내용
  TORCH_CHECK(input.scalar_type() == HALF || input.scalar_type() == BF16);
  TORCH_CHECK(input_global_scale.scalar_type() == FLOAT);
  TORCH_CHECK(mask.scalar_type() == INT);
  // output이다uint8（관련 내용개nvfp4관련 내용로관련 내용개uint8）
  // output_scale이다int32（관련 내용개fp8관련 내용로관련 내용개int32）
  TORCH_CHECK(output.scalar_type() == UINT8);
  TORCH_CHECK(output_scale.scalar_type() == INT);

  // 검증텐서shape와크기관련 내용
  const int BLOCK_SIZE = 16;  // FP4관련 내용의block크기
  auto m_topk = input.size(0);
  auto k_by_2 = input.size(1);
  auto k = k_by_2;
  
  // 만약관련 내용사용SiLU관련 내용와관련 내용입력차원된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (gating)
  if (use_silu_and_mul) {
    TORCH_CHECK(k_by_2 % 2 == 0, "k must be a multiple of 2");
    k = k_by_2 / 2;
  }
  
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(mask.size(0) == n_experts);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);  // FP4관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  
  // 검증관련 내용텐서의크기
  int scales_k = k / BLOCK_SIZE;
  // 4이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (nvidia nvfp4)의swizzle관련 내용
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4관련 내용 (4)개fp8관련 내용로관련 내용개int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);

  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUDA)와관련 내용
  auto in_dtype = input.dtype();
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  
  // 관련 내용입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다까지관련 내용의구현
  if (in_dtype == at::ScalarType::Half) {
    quant_impl<half>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts（없음expert관련 내용
        nullptr,  // output_scale_offset_by_experts（없음expert관련 내용
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else if (in_dtype == at::ScalarType::BFloat16) {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(),
        output_scale.data_ptr(),
        input.data_ptr(),
        input_global_scale.data_ptr(),
        nullptr,  // input_offset_by_experts（없음expert관련 내용
        nullptr,  // output_scale_offset_by_experts（없음expert관련 내용
        mask.data_ptr(),
        use_silu_and_mul,
        m_topk, k, n_experts, stream);
  } else {
    TORCH_CHECK(false, "Expected input data type to be half or bfloat16");
  }
}
```