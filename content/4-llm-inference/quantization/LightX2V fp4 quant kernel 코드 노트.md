# 0x0. 서론

이 노트는 예전에 LightX2V 안의 `lightx2v_kernel`이 FP4 양자화 GEMM을 어떻게 하는지 파악하려고 쓴 코드 리딩 노트다.

주로 2가지 문제를 설명한다:
- **인터페이스와 제약**: 어떤 shape / 정렬이 강한 제약인지, scale factor 텐서가 도대체 어떤 layout인지.
- **kernel 핵심 경로**: 양자화 kernel이 `fp16/bf16 -> fp4 + fp8 sf`를 어떻게 만드는지, 그리고 GEMM이 CUTLASS의 Block Scaled Tensor Core에 어떻게 데이터를 넘기는지.

프로젝트 주소: https://github.com/ModelTC/LightX2V/blob/main/lightx2v_kernel

# 0x1. 인터페이스 사용

## 0x1.1 Python 인터페이스

LightX2V kernel은 간결한 Python 인터페이스를 제공하는데, 주로 양자화 함수와 행렬 곱셈 함수다.

### NVFP4 인터페이스

```python
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

# 양자화 함수
def scaled_nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    입력 텐서를 FP4 형식으로 양자화한다
    
    Args:
        input: 입력 텐서, shape은 (m, n), dtype은 fp16/bf16
        input_global_scale: 전역 스케일 factor, 스칼라 텐서
    
    Returns:
        output: 양자화된 텐서, shape은 (m, n//2), dtype은 uint8(fp4 두 개를 패킹)
        output_scale: 양자화 factor, shape은 (rounded_m, rounded_k), dtype은 float8_e4m3fn
                     여기서 rounded_m = ((m + 128 - 1) // 128) * 128
                          rounded_k = (n // 16 + 4 - 1) // 4
    """

# 행렬 곱셈 함수
def cutlass_scaled_nvfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """
    FP4 행렬 곱셈을 수행한다: D = alpha * A @ B^T + bias
    
    Args:
        mat_a: 행렬 A, shape은 (m, k//2), 이미 양자화됨
        mat_b: 행렬 B, shape은 (n, k//2), 이미 양자화됨
        scales_a: A의 양자화 factor
        scales_b: B의 양자화 factor
        alpha: 스케일 factor
        bias: 선택적인 bias 항, shape은 (1, n)
    
    Returns:
        out: 출력 텐서, shape은 (m, n), dtype은 bfloat16
    """
```

### MXFP4 인터페이스

```python
from lightx2v_kernel.gemm import scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm

# 양자화 함수
def scaled_mxfp4_quant(input: torch.Tensor):
    """
    입력 텐서를 MXFP4 형식으로 양자화한다
    
    Args:
        input: 입력 텐서, shape은 (m, n), dtype은 fp16/bf16
    
    Returns:
        output: 양자화된 텐서, shape은 (m, n//2), dtype은 uint8
        output_scale: 양자화 factor, dtype은 float8_e8m0fnu
    """

# 행렬 곱셈 함수(인터페이스는 NVFP4와 유사)
def cutlass_scaled_mxfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """MXFP4 행렬 곱셈"""
```

## 0x1.2 `scaled_nvfp4_quant` 함수 분석

이 함수를 끝까지 읽어 보니 기억해 둘 만한 것은 몇 가지뿐이다(나머지 세부 사항은 코드를 직접 보는 편이 더 확실하다).

- **양자화 granularity**: 마지막 차원을 따라 per-group으로 하며, NVFP4의 기본값은 `16`이다.
- **출력 형태**: FP4 값 두 개를 하나의 `uint8`로 패킹하므로 마지막 차원이 `n//2`가 된다.
- **scale factor**: 각 group마다 하나의 scale이 대응되고, scale 자체도 양자화되며(NVFP4는 `float8_e4m3fn`), swizzled layout으로 저장된다(이는 이후 Block Scaled Tensor Core가 읽기 편하게 하기 위한 것이다).
- **padding/정렬**: scale tensor의 shape은 단순한 `(m, n/16)`이 아니라 round-up이 적용된 형태다(뒤에서 GEMM이 엄격하게 검증한다).

호출 체인만 확인하고 싶다면:

```python
torch.ops.lightx2v_kernel.scaled_nvfp4_quant_sm120.default(output, input, output_scale, input_global_scale)
```

이 단계가 하는 일은 다음과 같이 요약할 수 있다:
- `fp16/bf16` 입력을 읽는다
- group 내에서 absmax를 구한다
- `fp8` sf를 하나 기록한다(swizzled)
- 데이터를 스케일링한 뒤 `fp4`로 변환하고 패킹한다

(Python wrapper 안의 `torch.empty/torch.zeros/view` 같은 세부 사항은 글에서 굳이 펼치지 않는다. 정말 대조해 봐야 할 때는 소스 코드를 직접 보는 편이 빠르다.)

## 0x1.3 사용 예시

아래는 완전한 사용 예시로, NVFP4로 가중치 양자화와 추론을 어떻게 하는지 살펴본다:

```python
import torch
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

class MMWeightFp4:
    """FP4 양자화를 사용하는 행렬 곱셈 래퍼 클래스"""
    
    def __init__(self, weight, bias):
        # 가중치를 로드하고 양자화한다
        self.load_fp4_weight(weight, bias)
        # 활성값의 최대값을 캘리브레이션한다
        self.calibrate_x_absmax()

    @torch.no_grad()
    def apply(self, input_tensor):
        """양자화 행렬 곱셈을 수행한다"""
        # 입력을 양자화한다
        input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
            input_tensor, self.input_global_scale
        )
        # 행렬 곱셈을 수행한다
        output_tensor = cutlass_scaled_nvfp4_mm(
            input_tensor_quant, 
            self.weight, 
            input_tensor_scale, 
            self.weight_scale, 
            alpha=self.alpha, 
            bias=self.bias
        )
        return output_tensor

    @torch.no_grad()
    def load_fp4_weight(self, weight, bias):
        """가중치를 양자화한다"""
        # 가중치의 전역 스케일 factor를 계산한다
        # 2688.0 = 6.0 * 448.0, 여기서 6.0은 FP4 최대값, 448.0은 FP8(E4M3) 최대값이다
        self.weight_global_scale = (
            2688.0 / torch.max(torch.abs(weight))
        ).to(torch.float32)
        
        # 가중치를 양자화한다
        self.weight, self.weight_scale = scaled_nvfp4_quant(
            weight, self.weight_global_scale
        )
        self.bias = bias

    def calibrate_x_absmax(self):
        """입력 활성값의 최대값을 캘리브레이션한다"""
        # 이 값은 캘리브레이션 데이터셋을 통해 결정해야 한다
        x_absmax = ...
        self.x_absmax = torch.tensor(
            x_absmax, dtype=torch.float32, device=self.weight.device
        )
        # 입력의 전역 스케일 factor를 계산한다
        self.input_global_scale = (
            2688.0 / self.x_absmax
        ).to(torch.float32)
        # 최종 alpha 값을 계산한다
        self.alpha = 1.0 / (
            self.input_global_scale * self.weight_global_scale
        )

# 사용 예시
input_tensor = ...
weight = ...
m, k = input_tensor.shape
n = weight.shape[0]
bias = None

# 양자화 행렬 곱셈 객체를 만든다
mm = MMWeightFp4(weight, bias)

# 추론을 수행한다
output = mm.apply(input_tensor)
print(f"Output shape: {output.shape}")
```

---

# 0x2. 양자화 원리

양자화 원리 부분은 주로 https://github.com/ModelTC/LightX2V/tree/main/lightx2v_kernel 의 문서를 수정해서 가져온 것이다.

## 0x2.1 NVFP4 양자화 원리

#### 3.1.1 데이터 형식

NVFP4는 E2M1 형식(부호 비트 1개 + 지수 비트 2개 + 가수 비트 1개)을 사용하며, 부동소수점 수의 계산식은 다음과 같다:

```
ans = (-1)^s * 2^(p-b) * (1 + d1/2)
```

여기서:
- `s`: 부호 비트
- `p`: 지수 비트의 값(0-3)
- `b = 2^(e-1) - 1 = 2^(2-1) - 1 = 1`(bias 값)
- `d1`: 가수 비트의 값(0 또는 1)

**NVFP4의 특별한 점**:
- inf와 nan 표현을 없앴다
- 최대값을 ±6.0까지 표현할 수 있다(표준 E2M1은 ±3.0까지만 표현 가능)
- 0000은 +0, 1000은 -0을 나타낸다
- 0001은 0.5, 1001은 -0.5를 나타낸다

전체 E2M1 값 표:

| E2M1 | 0000 | 0001 | 0010 | 0011 | 0100 | 0101 | 0110 | 0111 | 1000 | 1001 | 1010 | 1011 | 1100 | 1101 | 1110 | 1111 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 값   | +0   | 0.5  | 1.0  | 1.5  | 2.0  | 3.0  | 4.0  | 6.0  | -0   | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |

#### 3.1.2 양자화 과정

NVFP4는 **Per-Group 양자화**를 채택하며, 양자화 granularity는 원소 16개가 한 group이다. 양자화 factor는 FP8(E4M3) 형식으로 저장한다.

**양자화 단계**:

한 묶음의 데이터 `X`가 주어졌을 때, `Xg`는 한 group의 데이터(원소 16개)를 나타낸다고 하자.

1. **scale1 계산**(각 group의 원본 scale):
   ```
   scale1 = max(abs(Xg)) / 6.0
   ```
   여기서 6.0은 NVFP4의 최대값이다.

2. **scale 양자화**(scale을 FP8로 양자화):
   ```
   global_scale = 6.0 * 448.0 / max(abs(X))
   scale2 = global_scale * scale1
   scale2 = max(abs(Xg)) / max(abs(X)) * 448.0
   ```
   이때 scale2는 FP8(E4M3)의 범위(최대값 448.0)로 스케일링되고, 그다음 FP8로 양자화된다:
   ```
   scale2_fp8 = quant_fp8(scale2)
   ```

3. **데이터 X 양자화**:
   ```
   scale2_fp32 = cvt2fp32(scale2_fp8)
   Xquant = quant_fp4(X * global_scale / scale2_fp32)
   ```
   근사적으로 다음과 같다:
   ```
   Xquant ≈ quant_fp4(X / scale1)
   ```

4. **FP4 행렬 곱셈**:
   ```
   ans = Aquant * Bquant * Ascale2 * Bscale2 / Aglobal_scale / Bglobal_scale
   ```
   단순화하면:
   ```
   ans ≈ Aquant * Bquant * Ascale1 * Bscale1
   ```

**핵심 포인트**:
- Weight와 Activation 모두 Per-Group 양자화를 사용하며, group size는 16이다
- 양자화 scale은 FP8(E4M3) 형식으로 저장한다
- scale 자체를 양자화해야 하는데, 이것이 흔히 쓰이는 W8A8-INT8 양자화와의 주된 차이다

## 0x2.2 MX-Formats 양자화 원리

#### 3.2.1 데이터 형식과 양자화 factor

MX-Formats(Microscaling Formats)는 OCP(Open Compute Project)가 정의한 표준화된 마이크로스케일링 부동소수점 형식이다.

**소스 데이터 형식**: fp16/bf16

**타깃 데이터 형식**: mxfp4/6/8

**양자화 factor 데이터 형식**: E8M0
- E8M0는 fp32와 수치 범위가 동일하다
- rounding을 거치면 양자화 factor를 그대로 저장할 수 있다
- 단점: 가수의 손실이 정밀도에 영향을 준다

**양자화 granularity**: [1×32]
- 원소 32개마다 하나의 양자화 factor를 공유한다

**양자화 차원**:
- K 차원을 따라 양자화한다(GEMM의 K 차원)

#### 3.2.2 Rounding과 Clamp

CUDA는 PTX 명령이나 내장 함수를 통해 Rounding과 Clamp 연산을 효율적으로 처리한다.

예를 들어 `cvt.rn.satfinite.e2m1x2.f32`는 fp32 타입 입력 두 개를 fp4 타입 출력 두 개로 변환할 수 있다:
- **Rounding 모드**: `rn`(round-to-nearest-even)
- **Clamp 모드**: `satfinite`(타깃 범위 내의 최대 유한값으로 클램핑하며, 무한과 NaN은 제외)

#### 3.2.3 데이터 레이아웃과 양자화 factor 레이아웃

**데이터 레이아웃**:
- MXFP4: fp4 값 두 개를 하나의 uint8로 패킹
- MXFP6: fp6 값 4개마다 uint8 3개로 패킹
- MXFP8: uint8을 그대로 사용해 저장

**양자화 factor 레이아웃**:
Cutlass Block Scaled GEMMs는 행렬 연산 가속을 만족시키기 위해 양자화 factor 레이아웃에 특별한 swizzle 요구사항을 둔다. 레이아웃 형식은 다음과 같다:
```
[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
```

#### 3.2.4 MX-Formats와 NVFP4의 차이

| 특성 | NVFP4 | MX-Formats |
|------|-------|------------|
| 양자화 granularity | 원소 16개 | 원소 32개 |
| 양자화 factor 형식 | FP8(E4M3) | FP8(E8M0) |
| scale 양자화 필요 여부 | 필요함 | 불필요 |
| 전역 스케일 factor | 필요함 | 불필요 |

---

# 0x3. 코드 구현

## 0x3.1 NVFP4 양자화 구현

### 핵심 데이터 구조

```cpp
// 타입 변환기: Type과 Type2 사이의 변환에 사용한다(half <-> half2, bfloat16 <-> bfloat162)
template <typename T>
struct TypeConverter {
  using Type = half2;  // 기본값
};

template <>
struct TypeConverter<half> {
  using Type = half2;  // half는 half2에 대응
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;  // bfloat16은 bfloat162에 대응
};

// 패킹 벡터: 16바이트의 패킹 데이터 타입
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // Type2 4개, 총 8개 원소
};
```

#### 4.1.2 FP32에서 E2M1로의 변환

이것이 양자화의 핵심 연산이며, PTX 인라인 어셈블리로 효율적인 변환을 구현한다:

```cpp
// float2 값 4개(총 8개의 float)를 8개의 e2m1 값으로 변환한다(uint32_t 1개로 패킹)
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{"
      ".reg .b8 byte0;"           // 8-bit 레지스터 4개를 정의한다
      ".reg .b8 byte1;"
      ".reg .b8 byte2;"
      ".reg .b8 byte3;"
      // 각 명령은 float32 2개를 e2m1 2개(총 1바이트)로 변환한다
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;"  // array[0].y, array[0].x -> byte0
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;"  // array[1].y, array[1].x -> byte1
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;"  // array[2].y, array[2].x -> byte2
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;"  // array[3].y, array[3].x -> byte3
      // 4바이트를 uint32_t 1개로 패킹한다
      "mov.b32 %0, {byte0, byte1, byte2, byte3};"
      "}"
      : "=r"(val)  // 출력: val
      : "f"(array[0].x), "f"(array[0].y),  // 입력: float 8개
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
}
```

**핵심 포인트**:
- `cvt.rn.satfinite.e2m1x2.f32`: PTX 명령으로, float32 2개를 e2m1 2개로 변환한다
- `rn`: round-to-nearest-even(가장 가까운 짝수로 반올림)
- `satfinite`: 유한값 범위로 포화시키며, inf와 nan은 제외한다
- 변환 명령 4개 + 패킹 명령 1개로 값 8개의 변환을 효율적으로 끝낸다

### 빠른 역수 계산

```cpp
// PTX 명령으로 빠른 근사 역수를 구현한다
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  // rcp.approx.ftz.f32: 빠른 근사 역수, flush-to-zero
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}
```

**장점**:
- 표준적인 `1.0f / a`보다 훨씬 빠르다

참고: CUDA PTX ISA 문서에 `rcp.approx`와 수식자(`ftz` 포함)에 대한 정의가 있다
https://docs.nvidia.com/cuda/parallel-thread-execution/

### 양자화 factor 레이아웃 계산

Cutlass Block Scaled GEMM은 양자화 factor가 특별한 swizzled 레이아웃을 사용할 것을 요구한다:

```cpp
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int numCols, SFType* SFout) {
  
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 특정 스레드만 SF를 기록한다(CVT_FP4_NUM_THREADS_PER_SF개 스레드마다 SF 하나를 기록)
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF 벡터 인덱스(K 차원에서 원소 16개마다 SF 하나를 공유)
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF 레이아웃: [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 인덱스: [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // M 차원의 tile 인덱스를 계산한다
    int32_t mTileIdx = mIdx / (32 * 4);  // 각 M tile은 128행을 포함한다
    int factor = CVT_FP4_SF_VEC_SIZE * 4;  // 16 * 4 = 64
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;  // M tile의 스트라이드

    // K 차원의 tile 인덱스를 계산한다
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;  // K tile의 스트라이드

    // M tile 내부 레이아웃은 열 우선(column major) [32, 4]이다
    int32_t outerMIdx = (mIdx % 32);  // 바깥쪽 M 인덱스(0-31)
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;  // 안쪽 M 인덱스(0-3)
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);  // 안쪽 K 인덱스(0-3)
    int64_t innerKStride = 1;

    // 전역 오프셋을 계산한다
    int64_t SFOffset = mTileIdx * mTileStride + 
                       kTileIdx * kTileStride + 
                       outerMIdx * outerMStride +
                       innerMIdx * innerMStride + 
                       innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}
```

레이아웃 설명:
- SF 레이아웃은 5차원 구조를 채택한다: `[numMTiles, numKTiles, 32, 4, 4]`
- M 차원은 128행을 하나의 tile로 나눈다(32×4)
- K 차원은 원소 64개를 하나의 tile로 나눈다(16×4)
- M tile 내부는 열 우선 레이아웃을 채택한다
- 이 레이아웃은 Tensor Core의 접근 패턴을 최적화한다

### 핵심 양자화 Kernel

이것이 양자화를 수행하는 주요 kernel 함수다:

```cpp
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(256, 6) cvt_fp16_to_fp4(
    int32_t numRows, int32_t numCols, Type const* in, 
    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = 
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);  // 16 / 8 = 2
  
  // 전역 스케일 factor를 가져온다
  // SFScale은 다음 GEMM의 alpha와 동일하다. 즉 (448.0 / (Alpha_A / 6.0))
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // 입력 텐서의 행/열 루프
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; 
         colIdx += blockDim.x) {
      
      // 입력 데이터를 읽는다(16바이트, 원소 8개)
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 출력 오프셋(원소 8개를 uint32_t 1개로 패킹)
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // SF 출력 주소를 가져온다
      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
          rowIdx, colIdx, numCols, SFout);

      // 양자화를 수행한다
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
}
```

Kernel 설정:
- `__launch_bounds__(256, 6)`: block당 256개 스레드, SM당 최대 6개 block
- 각 스레드는 원소 8개를 처리한다
- grid-stride loop로 모든 행을 처리한다

### Warp 레벨 양자화 함수

이것이 warp 내에서 양자화를 수행하는 핵심 함수다:

```cpp
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  
  // 1. 로컬 최대값을 계산한다(각 스레드가 원소 8개를 처리)
  auto localMax = __habs2(vec.elts[0]);  // 절대값을 취한다
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 쌍 단위로 비교한다
  }

  // 2. Warp 내 리덕션으로 원소 16개의 최대값을 얻는다(스레드 2개)
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 양자화 factor(SF)를 계산한다
  // vecMax / 6.0이 원본 scale이고, SFScaleVal을 곱한 뒤 FP8로 양자화한다
  float SFValue = SFScaleVal * (vecMax * 0.16666666666666666f);  // 0.1666... = 1/6
  
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
    // E8M0 형식을 사용한다
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // E4M3 형식을 사용한다(기본값)
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }

  // 4. 출력 스케일 factor를 계산한다
  // 최종 데이터 = 원본 데이터 * outputScale, 그다음 FP4로 양자화한다
  float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValue) : 0.0f;

  // 5. 양자화 factor를 전역 메모리에 기록한다
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6. 입력 데이터를 float으로 변환하고 스케일링한다
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // 7. e2m1 값으로 변환한다
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  return e2m1Vec;
}
```

핵심 단계:
1. **로컬 최대값 계산**: `__habs2`와 `__hmax2`로 벡터화 연산을 수행한다
2. **Warp 리덕션**: `__shfl_xor_sync`로 warp 내에서 데이터를 교환해 원소 16개의 최대값을 얻는다
3. **양자화 factor 계산**: `vecMax / 6.0 * SFScaleVal`을 구한 뒤 FP8로 양자화한다
4. **출력 스케일링**: `outputScale = SFScaleVal / SFValue`를 계산해 원본 데이터를 스케일링하는 데 사용한다
5. **데이터 변환**: fp16/bf16을 float으로 변환하고 outputScale을 곱한 뒤 다시 e2m1로 변환한다

### 호스트 측 호출 인터페이스

```cpp
void scaled_nvfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, 
    torch::Tensor& output_sf, torch::Tensor const& input_sf) {
  
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  // N 차원이 반드시 16의 배수인지 확인한다
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount = getMultiProcessorCount();

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  bool useUE8M0 = false;  // 기본적으로 E4M3을 사용한다

  // 입력 타입에 따라 디스패치한다
  switch (input.scalar_type()) {
    case torch::kHalf: {
      auto input_ptr = reinterpret_cast<half const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, 
                           sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    case torch::kBFloat16: {
      auto input_ptr = reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr());
      invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr, 
                           sf_out, useUE8M0, multiProcessorCount, stream);
      break;
    }
    default: {
      throw std::runtime_error("Unsupported input data type for quantize_to_fp4.");
    }
  }
}
```

---

## 0x3.2 양자화 Kernel 스레드 모델

GEMM 구현을 보기 전에 양자화 kernel의 스레드 모델을 먼저 살펴보자. 이것은 성능을 이해하는 데 중요하다.

### NVFP4 양자화 Kernel 스레드 모델

#### Grid와 Block 설정

```cpp
// 각 스레드는 원소 8개를 처리한다
dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));

// SM당 block 수
int const numBlocksPerSM = 1536 / block.x;

// Grid 크기
dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));
```

설정 분석:

1. **Block 크기 계산**:
   - `block.x = min(n / 8, 256)`
   - 각 스레드가 원소 8개를 처리하므로 `n / 8`개의 스레드가 필요하다
   - block당 최대 256개 스레드(`__launch_bounds__(256, 6)`의 제한을 받는다)

2. **SM당 Block 수**:
   - `numBlocksPerSM = 1536 / block.x`
   - 여기서의 목적은 `__launch_bounds__`와 맞춰서 SM을 최대한 채우고, 병렬도 부족으로 인한 idle을 줄이는 것이다.

3. **Grid 크기**:
   - `grid.x = min(m, multiProcessorCount * numBlocksPerSM)`
   - 각 block은 한 행의 데이터를 처리한다
   - grid-stride loop로 모든 행을 처리한다

#### Kernel 실행 모델

```cpp
__launch_bounds__(256, 6)  // block당 256 스레드, SM당 최대 6개 block
cvt_fp16_to_fp4(...) {
  // Grid-stride loop로 행을 처리한다
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    // Block-stride loop로 열을 처리한다
    for (int colIdx = threadIdx.x; colIdx < numCols / 8; colIdx += blockDim.x) {
      // 각 스레드는 원소 8개를 처리한다
      // ...
    }
  }
}
```

실행 흐름:

1. **행 단위 병렬**(Grid 차원):
   - 각 block이 한 행 또는 여러 행을 담당한다
   - grid-stride loop를 사용한다: `rowIdx += gridDim.x`
   - 모든 행이 처리되도록 보장한다

2. **열 단위 병렬**(Block 차원):
   - 각 스레드가 연속된 원소 8개를 담당한다
   - block-stride loop를 사용한다: `colIdx += blockDim.x`
   - 스레드 0은 원소 [0-7], [256×8-256×8+7], ... 를 처리한다
   - 스레드 1은 원소 [8-15], [256×8+8-256×8+15], ... 를 처리한다

3. **Warp 레벨 협력**:
   - 스레드 2개(원소 16개)마다 scale factor 하나를 계산한다
   - `__shfl_xor_sync`로 warp 내 리덕션을 수행한다
   - 스레드 32개(한 warp)가 원소 256개를 처리하며 scale factor 16개를 생성한다

#### 메모리 접근 패턴

전역 메모리 읽기(Coalesced):
```
Warp 0 (Threads 0-31):
  Thread 0:  input[row][0:8] 읽기
  Thread 1:  input[row][8:16] 읽기
  ...
  Thread 31: input[row][248:256] 읽기

(여기서의 접근 패턴은 전형적인 연속 주소 읽기이며, 목표는 메모리 트랜잭션을 최대한 병합하는 것이다. 최종적으로 “완벽”한지 여부는 실제 stride/정렬에 따라 달라진다.)
```

전역 메모리 쓰기:
```
양자화 데이터(FP4 2개마다 uint8 1개로 패킹):
  Thread 0:  output[row][0] 쓰기
  Thread 1:  output[row][1] 쓰기
  ...

Scale factors(swizzled layout):
  Thread 0:  SF[swizzled_offset] 쓰기 → 1 byte (FP8)
  Thread 2:  SF[swizzled_offset] 쓰기 → 1 byte
  ...(스레드 2개마다 SF 하나를 기록)
```

### MXFP4 양자화 Kernel 스레드 모델

MXFP4의 스레드 모델은 NVFP4와 비슷하며, 주된 차이는 양자화 granularity에 있다.

#### 핵심 차이

```cpp
// NVFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 16;  // 원소 16개/그룹
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 16 / 8 = 2;  // 스레드 2개/SF

// MXFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 32;  // 원소 32개/그룹
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 32 / 8 = 4;  // 스레드 4개/SF
```

영향:

1. **Warp 리덕션 횟수**:
   - NVFP4: `__shfl_xor_sync(mask, val, 1)` 1회(스레드 2개 리덕션)
   - MXFP4: `__shfl_xor_sync` 2회(스레드 4개 리덕션)
   ```cpp
   // MXFP4는 추가 리덕션 단계가 필요하다
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
   ```

2. **Scale Factor 밀도**:
   - NVFP4: 원소 16개마다 SF 1개 → 원소 256개마다 SF 16개
   - MXFP4: 원소 32개마다 SF 1개 → 원소 256개마다 SF 8개

3. **메모리 접근 패턴**:
   - 동일한 coalesced 읽기 패턴
   - 다른 SF 쓰기 패턴(더 희소하다)

---

## 0x3.3 NVFP4 행렬 곱셈 구현

NVFP4의 행렬 곱셈은 CUTLASS 3.x의 Block Scaled GEMM을 기반으로 한다.

### GEMM 설정 구조

```cpp
struct Fp4GemmSm120 {
    // A 행렬 설정
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4 타입
    using LayoutATag = cutlass::layout::RowMajor;                   // 행 우선
    static constexpr int AlignmentA = 32;                           // 정렬 요구사항: 원소 32개

    // B 행렬 설정
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;                // 열 우선
    static constexpr int AlignmentB = 32;

    // C/D 행렬 설정
    using ElementD = cutlass::bfloat16_t;                           // 출력 타입
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    
    // 누산기 설정
    using ElementAccumulator = float;                               // 내부 누산은 float 사용
    using ArchTag = cutlass::arch::Sm120;                           // Blackwell 아키텍처
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; // Block Scaled Tensor Op

    // 성능 설정
    using ThreadBlockShape = Shape<_128,_128,_128>;                 // Tile 크기: 128×128×128
    using ClusterShape = Shape<_1,_1,_1>;                           // Cluster 크기

    // Epilogue 설정: per-column bias 지원
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    // Collective Epilogue 구축
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        EVTOp
    >::CollectiveOp;

    // Collective Mainloop 구축
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // GEMM Kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
```

설정 설명:
- **ElementA/B**: `nv_float4_t<float_e2m1_t>`로 NVFP4 타입을 표현한다
- **AlignmentA/B**: 원소 32개 정렬로, 효율적인 메모리 접근을 보장한다
- **ThreadBlockShape**: 128×128×128의 tile 크기로, 레지스터 사용량과 shared memory의 균형을 맞춘다
- **OpClassBlockScaledTensorOp**: Block Scaled Tensor Core 연산을 사용한다
- **EVTOp**: per-column bias를 지원하는 epilogue 융합 연산

### 파라미터 구축 함수

```cpp
typename Fp4GemmSm120::Gemm::Arguments args_from_options_nvfp4_nvfp4(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    at::Tensor const& alpha, c10::optional<torch::Tensor> const& bias,
    int64_t M, int64_t N, int64_t K) {
  
  using Sm1xxBlkScaledConfig = 
      typename Fp4GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  
  // stride를 계산한다
  auto stride_A = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideD{}, {m, n, 1});

  // scale factor의 layout을 계산한다
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  if (bias) {
    // bias가 있는 경우
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {// Epilogue arguments
       {},
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    
    // fusion 파라미터를 설정한다
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Fp4GemmSm120::Gemm::ElementC const*>(
        bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    
    return arguments;
  } else {
    // bias가 없는 경우(비슷하므로 bias 설정은 생략)
    // ...
  }
}
```

핵심 포인트:
- stride와 scale factor layout을 계산한다
- CUTLASS Arguments 구조체를 구축한다
- bias가 있는 경우와 없는 경우 두 가지를 지원한다
- epilogue fusion을 사용해 bias 덧셈을 GEMM 안으로 융합한다

### GEMM 실행 함수

```cpp
void runGemmNvfp4Sm120(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    at::Tensor const& alpha, c10::optional<torch::Tensor> const& bias,
    int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  
  typename Fp4GemmSm120::Gemm gemm;

  // 파라미터를 구축한다
  auto arguments = args_from_options_nvfp4_nvfp4(
      D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  
  // workspace를 할당한다
  size_t workspace_size = Fp4GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  // 실행 가능한지 확인한다
  CUTLASS_CHECK(gemm.can_implement(arguments));
  
  // 초기화
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  
  // 실행
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
```

### 호스트 측 인터페이스

```cpp
void cutlass_scaled_nvfp4_mm_sm120(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, c10::optional<torch::Tensor> const& bias) {

  // 입력 검사
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1], "a and b shapes cannot be multiplied");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;  // FP4 두 개가 uint8 하나로 패킹되기 때문이다

  // 정렬 검사
  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // rounded 크기를 계산한다
  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);  // k/16은 scale factor의 개수다

  // scale factor의 크기를 검사한다
  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "scale_a must be padded and swizzled to shape (", rounded_m, "x", rounded_k, ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "scale_b must be padded and swizzled to shape (", rounded_n, "x", rounded_k, ")");

  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  runGemmNvfp4Sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
}
```

---

## 0x3.4 MXFP4 양자화 구현

MXFP4의 양자화 구현은 NVFP4와 비슷하고, 주된 차이는 다음과 같다:

### 주요 차이

```cpp
// NVFP4 vs MXFP4의 핵심 차이

// 1. 양자화 granularity
constexpr int CVT_FP4_SF_VEC_SIZE_NVFP4 = 16;  // NVFP4: 원소 16개가 한 그룹
constexpr int CVT_FP4_SF_VEC_SIZE_MXFP4 = 32;  // MXFP4: 원소 32개가 한 그룹

// 2. 양자화 factor 형식
// NVFP4: E4M3을 사용하며 global_scale이 필요하다
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);

// MXFP4: E8M0을 사용하며 global_scale이 필요 없다
__nv_fp8_e8m0 tmp;
tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);

// 3. 출력 스케일 계산
// NVFP4: global_scale을 고려해야 한다
float outputScale = SFScaleVal * reciprocal_approximate_ftz(SFValue);

// MXFP4: SF의 역수를 그대로 사용한다
float outputScale = reciprocal_approximate_ftz(SFValue);
```

### MXFP4 Warp 양자화 함수

```cpp
template <class Type>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, uint8_t* SFout) {
  
  // 1. 로컬 최대값을 계산한다(스레드마다 원소 8개)
  auto localMax = __habs2(vec.elts[0]);
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // 2. Warp 내 리덕션으로 원소 32개의 최대값을 얻는다(스레드 4개)
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);  // 추가 리덕션 1회
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 양자화 factor를 계산한다(6.0으로 바로 나누며, global_scale이 필요 없다)
  float SFValue = vecMax * 0.16666666666666666f;
  
  // 4. E8M0으로 양자화한다
  uint8_t fp8SFVal;
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);
  fp8SFVal = tmp.__x;

  // 5. 출력 스케일을 계산한다(global_scale이 필요 없다)
  float outputScale = SFValue != 0 ? reciprocal_approximate_ftz(SFValue) : 0.0f;

  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6-7. 변환과 양자화(NVFP4와 동일)
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  
  #pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
  return e2m1Vec;
}
```

**핵심 차이**:
- **리덕션 횟수**: MXFP4는 `__shfl_xor_sync`가 두 번 필요하다(원소 32개에는 스레드 4개가 필요하다)
- **global_scale 없음**: MXFP4는 `vecMax / 6.0`을 바로 SF로 사용한다
- **E8M0 형식**: `__nv_cvt_float_to_e8m0`으로 변환한다

---

## 0x3.5 MXFP4 행렬 곱셈 구현

MXFP4의 행렬 곱셈 설정은 NVFP4와 비슷하고, 주된 차이는 다음과 같다:

### GEMM 설정 차이

```cpp
struct Mxfp4GemmSm120 {
    // A 행렬 설정
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  // mx_float4_t를 사용한다
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128;  // 더 큰 정렬 요구사항: 원소 128개

    // B 행렬 설정
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128;

    // 나머지 설정은 NVFP4와 동일하다
    // ...
};
```

**핵심 차이**:
- **ElementA/B**: `nv_float4_t`가 아니라 `mx_float4_t`를 사용한다
- **AlignmentA/B**: 원소 128개 정렬(MXFP4의 group size가 32이므로 group 4개에 해당)
- **Scale Factor 타입**: `float_ue4m3_t`가 아니라 `float_ue8m0_t`를 사용한다

### 정렬 검사 차이

```cpp
void cutlass_scaled_mxfp4_mm_sm120(...) {
  // ...
  
  auto const k = A.sizes()[1] * 2;
  
  // MXFP4는 더 엄격한 정렬이 필요하다
  constexpr int alignment = 128;  // NVFP4는 32
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // Scale factor의 계산도 다르다
  int rounded_k = round_up(k / 32, 4);  // MXFP4: k/32, NVFP4: k/16
  
  // ...
}
```

---

# 0x4. LightX2V 프로젝트에서의 실제 사용

## 0x4.1 통합 방식

LightX2V 프로젝트는 `MMWeight` 클래스 체계를 통해 lightx2v_kernel을 통합하여, 모델 가중치의 양자화와 추론 가속을 구현했다.

### 양자화 연산자 임포트

```python
# lightx2v/common/ops/mm/mm_weight.py
try:
    from lightx2v_kernel.gemm import (
        cutlass_scaled_mxfp4_mm,
        cutlass_scaled_mxfp6_mxfp8_mm,
        cutlass_scaled_mxfp8_mm,
        cutlass_scaled_nvfp4_mm,
        scaled_mxfp4_quant,
        scaled_mxfp6_quant,
        scaled_mxfp8_quant,
        scaled_nvfp4_quant,
    )
except ImportError:
    # lightx2v_kernel이 설치되어 있지 않으면 None을 사용한다
    scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm = None, None
    scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm = None, None
    scaled_mxfp6_quant, cutlass_scaled_mxfp6_mxfp8_mm = None, None
    scaled_mxfp8_quant, cutlass_scaled_mxfp8_mm = None, None
```

## 0x4.2 NVFP4 양자화 가중치 클래스

LightX2V는 NVFP4 양자화 가중치를 관리하기 위해 `MMWeightNvfp4` 클래스를 구현했다.

### 클래스 정의

```python
@MM_WEIGHT_REGISTER("nvfp4")
class MMWeightNvfp4(MMWeightQuantNvfp4Template):
    """
    NVFP4 양자화 가중치 클래스
    - Weight: NVFP4 형식
    - Act: NVFP4 동적 양자화
    - Kernel: lightx2v_kernel
    """
    
    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
        )
        # 양자화 함수를 설정한다
        self.load_func = self.load_nvfp4
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_nvfp4
```

### 가중치 로딩

가중치를 로딩할 때는 다음 데이터를 로드해야 한다:

```python
def _get_cuda_tensor_pair(self, source, is_lazy):
    # 1. 양자화된 가중치를 로드한다
    weight = source.get_tensor(self.weight_name).to(AI_DEVICE)
    
    # 2. 가중치의 scale factors를 로드한다
    scale = source.get_tensor(self.weight_scale_name).to(AI_DEVICE)
    
    # 3. input_global_scale을 계산하거나 로드한다
    if self.input_absmax_name in source:
        # 캘리브레이션 데이터로부터 계산한다
        input_absmax = source.get_tensor(self.input_absmax_name)
        input_global_scale = (2688.0 / input_absmax).to(torch.float32)
        weight_global_scale = source.get_tensor(self.weight_global_scale_name)
        alpha = 1.0 / (input_global_scale * weight_global_scale)
    else:
        # 그대로 로드한다
        input_global_scale = source.get_tensor(self.input_global_scale_name)
        alpha = source.get_tensor(self.alpha_name)
    
    return weight, scale, input_global_scale, alpha
```

핵심 파라미터 설명:
- `weight`: 양자화된 가중치, shape은 `(out_features, in_features//2)`, dtype은 `uint8`
- `scale`: 가중치의 scale factors, dtype은 `float8_e4m3fn`
- `input_global_scale`: 입력의 전역 스케일 factor로, 활성값 양자화에 사용된다
- `alpha`: 출력 스케일 factor, `alpha = 1.0 / (input_global_scale * weight_global_scale)`

### 추론 과정

```python
def apply(self, input_tensor):
    # 1. 입력 활성값을 양자화한다
    # input_tensor: (batch_size, in_features), dtype=bfloat16
    input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
    # input_tensor_quant: (batch_size, in_features//2), dtype=uint8
    # input_tensor_scale: (batch_size, in_features//16), dtype=float8_e4m3fn
    
    # 2. 양자화 행렬 곱셈을 수행한다
    output_tensor = cutlass_scaled_nvfp4_mm(
        input_tensor_quant,      # 양자화된 입력
        self.weight,             # 양자화된 가중치
        input_tensor_scale,      # 입력의 scale factors
        self.weight_scale,       # 가중치의 scale factors
        alpha=self.alpha,        # 출력 스케일 factor
        bias=self.bias,          # 선택적인 bias
    )
    # output_tensor: (batch_size, out_features), dtype=bfloat16
    
    return output_tensor
```

### 활성값 양자화 함수

```python
def act_quant_nvfp4(self, x):
    """
    입력 활성값에 대해 NVFP4 양자화를 수행한다
    
    Args:
        x: 입력 텐서, shape=(batch_size, in_features), dtype=bfloat16
    
    Returns:
        input_tensor_quant: 양자화된 텐서, shape=(batch_size, in_features//2)
        input_tensor_scale: scale factors,shape=(batch_size, in_features//16)
    """
    input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
        x, 
        self.input_global_scale
    )
    return input_tensor_quant, input_tensor_scale
```

## 0x4.3 완전한 추론 흐름

아래는 완전한 추론 흐름의 예시다:

```python
# 1. 양자화 가중치 객체를 생성한다
mm_weight = MMWeightNvfp4(
    weight_name="transformer.blocks.0.attn.qkv.weight",
    bias_name="transformer.blocks.0.attn.qkv.bias",
    lazy_load=True,
    lazy_load_file="/path/to/quantized_model",
)

# 2. 양자화 가중치를 로드한다
mm_weight.load(weight_dict)

# 3. 가중치를 GPU로 올린다
mm_weight.to_cuda()

# 4. 추론
input_tensor = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device="cuda")
output_tensor = mm_weight.apply(input_tensor)
# 출력 shape은 대응하는 Linear의 out_features에 따라 결정된다

# 5. 추론이 끝나면 CPU로 오프로드할 수 있다
mm_weight.to_cpu()
```

## 0x4.4 양자화 모델 변환

LightX2V는 모델 양자화 변환 도구를 제공하며, FP16/BF16 모델을 NVFP4 양자화 모델로 변환할 수 있다.

### 가중치 양자화

```python
# tools/convert/quant/quant.py
def quantize_weight_nvfp4(weight, calib_data):
    """
    가중치를 NVFP4 형식으로 양자화한다
    
    Args:
        weight: 원본 가중치, shape=(out_features, in_features), dtype=bfloat16
        calib_data: 캘리브레이션 데이터로, input_global_scale 계산에 사용된다
    
    Returns:
        quantized_weight: 양자화된 가중치
        weight_scale: 가중치의 scale factors
        input_global_scale: 입력의 전역 스케일 factor
        weight_global_scale: 가중치의 전역 스케일 factor
    """
    # 1. input_global_scale을 계산한다
    input_absmax = calib_data.abs().max()
    input_global_scale = 2688.0 / input_absmax
    
    # 2. 가중치를 양자화한다
    weight = weight.to("cuda").to(torch.bfloat16)
    quantized_weight, weight_scale = scaled_nvfp4_quant(
        weight, 
        torch.tensor(input_global_scale, device="cuda")
    )
    
    # 3. weight_global_scale을 계산한다
    weight_absmax = weight.abs().max()
    weight_global_scale = 2688.0 / weight_absmax
    
    return quantized_weight, weight_scale, input_global_scale, weight_global_scale
```

### 양자화 모델 저장

```python
def save_quantized_model(model, output_path):
    """양자화된 모델을 저장한다"""
    state_dict = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'mm_weight') and isinstance(module.mm_weight, MMWeightNvfp4):
            # 양자화 가중치를 저장한다
            state_dict[f"{name}.weight"] = module.mm_weight.weight
            state_dict[f"{name}.weight_scale"] = module.mm_weight.weight_scale
            state_dict[f"{name}.input_global_scale"] = module.mm_weight.input_global_scale
            state_dict[f"{name}.weight_global_scale"] = module.mm_weight.weight_global_scale
            
            if module.mm_weight.bias is not None:
                state_dict[f"{name}.bias"] = module.mm_weight.bias
    
    # safetensors를 사용해 저장한다
    from safetensors.torch import save_file
    save_file(state_dict, output_path)
```


## 0x4.5 실제 응용 시나리오

### 비디오 생성 모델 가속

LightX2V 프로젝트는 주로 비디오 생성 모델(예: Wan2.2, HunyuanVideo)의 추론 가속에 사용된다:

```python
# 예시: Wan2.2 모델의 Transformer block
class TransformerBlock:
    def __init__(self):
        # QKV projection은 NVFP4 양자화를 사용한다
        self.qkv = MMWeightNvfp4(
            weight_name="transformer.blocks.0.attn.qkv.weight",
            bias_name="transformer.blocks.0.attn.qkv.bias",
        )
        
        # MLP는 NVFP4 양자화를 사용한다
        self.mlp_fc1 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc1.weight",
            bias_name="transformer.blocks.0.mlp.fc1.bias",
        )
        self.mlp_fc2 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc2.weight",
            bias_name="transformer.blocks.0.mlp.fc2.bias",
        )
    
    def forward(self, x):
        # 1. QKV projection (양자화 가속)
        qkv = self.qkv.apply(x)  # (B, L, 3*D)
        
        # 2. Attention (FP16/BF16)
        attn_out = self.attention(qkv)
        
        # 3. MLP (양자화 가속)
        mlp_out = self.mlp_fc2.apply(
            F.gelu(self.mlp_fc1.apply(attn_out))
        )
        
        return mlp_out
```

# 0x5. 요약

That's all.

