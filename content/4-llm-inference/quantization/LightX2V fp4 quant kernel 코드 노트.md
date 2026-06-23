# LightX2V fp4 quant kernel 코드 노트

## 0x0. 머리말

관련 내용노트이다이전에는위해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (LightX2V)`lightx2v_kernel` 관련 내용하다 FP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (GEMM)쓰기의읽다관련 내용노트。

주요관련 내용 (2)개문제：
- **인터페이스와관련 내용**：이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (shape /)정렬이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale factor)의텐서까지관련 내용이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layout)
- **kernel 핵심관련 내용**：이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)`fp16/bf16 -> fp4 + fp8 sf`，로및 GEMM 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUTLASS)의 Block Scaled Tensor Core。

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)https://github.com/ModelTC/LightX2V/blob/main/lightx2v_kernel

## 0x1. 인터페이스관련 내용사용

### 0x1.1 Python 인터페이스

LightX2V kernel 관련 내용의 Python 인터페이스，주요관련 내용이다관련 내용함수와matrix multiplication함수。

#### NVFP4 인터페이스

```python
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

# 관련 내용함수
def scaled_nvfp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    """
    할 것이다입력텐서관련 내용로 FP4 관련 내용
    
    Args:
        input: 입력텐서，shape 로 (m, n)，dtype 로 fp16/bf16
        input_global_scale: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다텐서
    
    Returns:
        output: 관련 내용후의텐서，shape 로 (m, n//2)，dtype 로 uint8（관련 내용개 fp4 관련 내용
        output_scale: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (shape)로 (rounded_m, rounded_k)，dtype 로 float8_e4m3fn
                     여기서 rounded_m = ((m + 128 - 1) // 128) * 128
                          rounded_k = (n // 16 + 4 - 1) // 4
    """

# matrix multiplication함수
def cutlass_scaled_nvfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """
    실행한다 FP4 matrix multiplication：D = alpha * A @ B^T + bias
    
    Args:
        mat_a: matrix A，shape 로 (m, k//2)，관련 내용
        mat_b: matrix B，shape 로 (n, k//2)，관련 내용
        scales_a: A 의관련 내용
        scales_b: B 의관련 내용
        alpha: 관련 내용
        bias: 가능관련 내용의bias이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (shape)로 (1, n)
    
    Returns:
        out: 출력텐서，shape 로 (m, n)，dtype 로 bfloat16
    """
```

#### MXFP4 인터페이스

```python
from lightx2v_kernel.gemm import scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm

# 관련 내용함수
def scaled_mxfp4_quant(input: torch.Tensor):
    """
    할 것이다입력텐서관련 내용로 MXFP4 관련 내용
    
    Args:
        input: 입력텐서，shape 로 (m, n)，dtype 로 fp16/bf16
    
    Returns:
        output: 관련 내용후의텐서，shape 로 (m, n//2)，dtype 로 uint8
        output_scale: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (dtype)로 float8_e8m0fnu
    """

# matrix multiplication함수（인터페이스와 NVFP4 관련 내용
def cutlass_scaled_mxfp4_mm(mat_a, mat_b, scales_a, scales_b, alpha, bias=None):
    """MXFP4 matrix multiplication"""
```

### 0x1.2 `scaled_nvfp4_quant` 함수관련 내용

이함수필자는읽다하와서，이 부분은 원문의 해당 기술 설명을 이어서 서술한다의이 부분은 원문의 해당 기술 설명을 이어서 서술한다하의관련 내용가서보다코드더관련 내용

- **관련 내용**：관련 내용마지막으로관련 내용하다 per-group，NVFP4 기본이다 `16`。
- **출력관련 내용**：FP4 관련 내용개이 부분은 원문의 해당 기술 설명을 이어서 서술한다개 `uint8`，그래서마지막으로관련 내용된다관련 내용`n//2`。
- **scale factor**：각개 group 대응관련 내용개 scale，scale 관련 내용된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (NVFP4)이다 `float8_e4m3fn`），그리고관련 내용사용 swizzled layout 관련 내용이다위해후이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Block Scaled Tensor Core)읽다관련 내용와서관련 내용
- **padding/정렬**：scale tensor 의 shape 아니이다관련 내용의 `(m, n/16)`，관련 내용이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (round-up)의버전（후이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (GEMM)된다하다관련 내용

만약사용자는만이다관련 내용확인호출한다관련 내용

```python
torch.ops.lightx2v_kernel.scaled_nvfp4_quant_sm120.default(output, input, output_scale, input_global_scale)
```

관련 내용하다의관련 내용가능로관련 내용
- 읽다 `fp16/bf16` 입력
- group 관련 내용하다 absmax
- 쓰기관련 내용`fp8` 의 sf（swizzled）
- 관련 내용후관련 내용`fp4` 그리고관련 내용

（Python wrapper 관련 내용`torch.empty/torch.zeros/view` 의관련 내용필자는관련 내용아니에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다대해관련 내용의관련 내용보다관련 내용더빠른。）

### 0x1.3 관련 내용사용예제

아래이다관련 내용개완전한의관련 내용사용관련 내용보다보다관련 내용사용 NVFP4 수행한다weight관련 내용와관련 내용

```python
import torch
from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm

class MMWeightFp4:
    """관련 내용사용 FP4 관련 내용의matrix multiplication관련 내용
    
    def __init__(self, weight, bias):
        # 로드그리고이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
        self.load_fp4_weight(weight, bias)
        # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다의관련 내용큰관련 내용
        self.calibrate_x_absmax()

    @torch.no_grad()
    def apply(self, input_tensor):
        """실행한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (matrix multiplication)
        # 관련 내용입력
        input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
            input_tensor, self.input_global_scale
        )
        # 실행한다matrix multiplication
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
        """이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
        # 계산weight의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
        # 2688.0 = 6.0 * 448.0，여기서 6.0 이다 FP4 관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.0)이다 FP8(E4M3) 관련 내용큰관련 내용
        self.weight_global_scale = (
            2688.0 / torch.max(torch.abs(weight))
        ).to(torch.float32)
        
        # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
        self.weight, self.weight_scale = scaled_nvfp4_quant(
            weight, self.weight_global_scale
        )
        self.bias = bias

    def calibrate_x_absmax(self):
        """관련 내용입력관련 내용의관련 내용큰관련 내용
        # 이관련 내용통해이 부분은 원문의 해당 기술 설명을 이어서 서술한다와서관련 내용
        x_absmax =...
        self.x_absmax = torch.tensor(
            x_absmax, dtype=torch.float32, device=self.weight.device
        )
        # 계산입력의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
        self.input_global_scale = (
            2688.0 / self.x_absmax
        ).to(torch.float32)
        # 계산관련 내용의 alpha 관련 내용
        self.alpha = 1.0 / (
            self.input_global_scale * self.weight_global_scale
        )

# 관련 내용사용예제
input_tensor =...
weight =...
m, k = input_tensor.shape
n = weight.shape[0]
bias = None

# 생성한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (matrix multiplication)대해관련 내용
mm = MMWeightFp4(weight, bias)

# 실행한다관련 내용
output = mm.apply(input_tensor)
print(f"Output shape: {output.shape}")
```

---

## 0x2. 관련 내용

관련 내용부분주요부터 https://github.com/ModelTC/LightX2V/tree/main/lightx2v_kernel 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다와서。

### 0x2.1 NVFP4 관련 내용

##### 3.1.1 관련 내용

NVFP4 관련 내용사용 E2M1 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1 + 2 + 1)의계산관련 내용로：

```
ans = (-1)^s * 2^(p-b) * (1 + d1/2)
```

여기서：
- `s`：관련 내용
- `p`：관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (0-3)
- `b = 2^(e-1) - 1 = 2^(2-1) - 1 = 1`（bias관련 내용
- `d1`：관련 내용의관련 내용 (0)또는 1）

**NVFP4 의관련 내용**：
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (inf)와 nan 의관련 내용
- 관련 내용큰관련 내용가능로관련 내용까지 ±6.0（이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (E2M1)만가능관련 내용까지 ±3.0）
- 0000 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+0 1000 -0)
- 0001 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (0.5 1001 -0.5)

완전한의 E2M1 관련 내용

| E2M1 | 0000 | 0001 | 0010 | 0011 | 0100 | 0101 | 0110 | 0111 | 1000 | 1001 | 1010 | 1011 | 1100 | 1101 | 1110 | 1111 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 관련 내용| +0   | 0.5  | 1.0  | 1.5  | 2.0  | 3.0  | 4.0  | 6.0  | -0   | -0.5 | -1.0 | -1.5 | -2.0 | -3.0 | -4.0 | -6.0 |

##### 3.1.2 관련 내용

NVFP4 관련 내용사용 **Per-Group 관련 내용**，관련 내용로 16 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용 FP8(E4M3) 관련 내용

**관련 내용**：

이 부분은 원문의 해당 기술 설명을 이어서 서술한다`X`，관련 내용`Xg` 관련 내용개 group 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)개관련 내용

1. **계산 scale1**（각개 group 의원본 scale）：
   ```
   scale1 = max(abs(Xg)) / 6.0
   ```
   여기서 6.0 이다 NVFP4 의관련 내용큰관련 내용

2. **이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale)**（할 것이다 scale 관련 내용까지 FP8）：
   ```
   global_scale = 6.0 * 448.0 / max(abs(X))
   scale2 = global_scale * scale1
   scale2 = max(abs(Xg)) / max(abs(X)) * 448.0
   ```
   이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale2)까지 FP8(E4M3) 의관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.0)그다음관련 내용까지 FP8：
   ```
   scale2_fp8 = quant_fp8(scale2)
   ```

3. **이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (X)**：
   ```
   scale2_fp32 = cvt2fp32(scale2_fp8)
   Xquant = quant_fp4(X * global_scale / scale2_fp32)
   ```
   관련 내용
   ```
   Xquant ≈ quant_fp4(X / scale1)
   ```

4. **FP4 matrix multiplication**：
   ```
   ans = Aquant * Bquant * Ascale2 * Bscale2 / Aglobal_scale / Bglobal_scale
   ```
   관련 내용로：
   ```
   ans ≈ Aquant * Bquant * Ascale1 * Bscale1
   ```

**핵심관련 내용**：
- Weight 와 Activation 모두관련 내용사용 Per-Group 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (group size)로 16
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale)사용 FP8(E4M3) 관련 내용
- 관련 내용대해 scale 관련 내용수행한다관련 내용이다와자주 쓰는 W8A8-INT8 관련 내용의주요관련 내용

### 0x2.2 MX-Formats 관련 내용

##### 3.2.1 관련 내용와관련 내용

MX-Formats（Microscaling Formats）이다 OCP（Open Compute Project）관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다

**이 부분은 원문의 해당 기술 설명을 이어서 서술한다**：fp16/bf16

**이 부분은 원문의 해당 기술 설명을 이어서 서술한다**：mxfp4/6/8

**이 부분은 원문의 해당 기술 설명을 이어서 서술한다**：E8M0
- E8M0 와 fp32 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (rounding)후가능이 부분은 원문의 해당 기술 설명을 이어서 서술한다
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다의관련 내용된다관련 내용

**관련 내용**：[1×32]
- 각 32 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다개관련 내용

**관련 내용차원**：
- 관련 내용 (K)차원이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (GEMM)의 K 차원）

##### 3.2.2 Rounding 와 Clamp

CUDA 통해 PTX 관련 내용또는관련 내용함수높은관련 내용완료 Rounding 와 Clamp 관련 내용

관련 내용`cvt.rn.satfinite.e2m1x2.f32` 가능로할 것이다관련 내용개 fp32 관련 내용의입력관련 내용로관련 내용개 fp4 관련 내용의출력：
- **Rounding 모드**：`rn`（round-to-nearest-even）
- **Clamp 모드**：`satfinite`（관련 내용까지이 부분은 원문의 해당 기술 설명을 이어서 서술한다의관련 내용큰있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다없음관련 내용와 NaN）

##### 3.2.3 관련 내용와이 부분은 원문의 해당 기술 설명을 이어서 서술한다

**관련 내용**：
- MXFP4：관련 내용개 fp4 관련 내용로관련 내용개 uint8
- MXFP6：각 4 개 fp6 관련 내용로 3 개 uint8
- MXFP8：관련 내용사용 uint8 관련 내용

**이 부분은 원문의 해당 기술 설명을 이어서 서술한다**：
Cutlass Block Scaled GEMMs 대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다있다관련 내용의 swizzle 관련 내용로이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (matrix)가속。관련 내용로：
```
[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
```

##### 3.2.4 MX-Formats 와 NVFP4 의관련 내용

| 관련 내용| NVFP4 | MX-Formats |
|------|-------|------------|
| 관련 내용| 16 개관련 내용| 32 개관련 내용|
| 이 부분은 원문의 해당 기술 설명을 이어서 서술한다| FP8(E4M3) | FP8(E8M0) |
| 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale)| 이다 | 관련 내용|
| 이 부분은 원문의 해당 기술 설명을 이어서 서술한다| 관련 내용| 아니관련 내용|

---

## 0x3. 코드구현

### 0x3.1 NVFP4 관련 내용구현

#### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다

```cpp
// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용된다에서 Type 와 Type2 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (half <-)> half2, bfloat16 <-> bfloat162）
template <typename T>
struct TypeConverter {
  using Type = half2;  // 기본
};

template <>
struct TypeConverter<half> {
  using Type = half2;  // half 대응 half2
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;  // bfloat16 대응 bfloat162
};

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];  // 4 개 Type2，관련 내용 (8)개관련 내용
};
```

##### 4.1.2 FP32 까지 E2M1 의관련 내용

관련 내용이다관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용 PTX inline관련 내용구현높은관련 내용

```cpp
// 할 것이다 4 개 float2 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (8)개 float）관련 내용로 8 개 e2m1 관련 내용로 1 개 uint32_t）
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{"
      ".reg.b8 byte0;"           // 관련 내용 (4)개 8-bit register
      ".reg.b8 byte1;"
      ".reg.b8 byte2;"
      ".reg.b8 byte3;"
      // 각관련 내용할 것이다 2 개 float32 관련 내용로 2 개 e2m1（관련 내용 (1)개관련 내용
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;"  // array[0].y, array[0].x -> byte0
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;"  // array[1].y, array[1].x -> byte1
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;"  // array[2].y, array[2].x -> byte2
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;"  // array[3].y, array[3].x -> byte3
      // 할 것이다 4 개관련 내용로 1 개 uint32_t
      "mov.b32 %0, {byte0, byte1, byte2, byte3};"
      "}"
: "=r"(val)  // 출력：val
: "f"(array[0].x), "f"(array[0].y),  // 입력：8 개 float
        "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y),
        "f"(array[3].x), "f"(array[3].y));
  return val;
}
```

**핵심관련 내용**：
- `cvt.rn.satfinite.e2m1x2.f32`：PTX 관련 내용할 것이다 2 개 float32 관련 내용로 2 개 e2m1
- `rn`：round-to-nearest-even（최근관련 내용
- `satfinite`：관련 내용와까지있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (inf)와 nan
- 4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+ 1)높은관련 내용완료 8 개관련 내용의관련 내용

#### 빠른관련 내용계산

```cpp
// 관련 내용사용 PTX 관련 내용구현빠른이 부분은 원문의 해당 기술 설명을 이어서 서술한다
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  // rcp.approx.ftz.f32：빠른이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (flush-to-zero)
  asm volatile("rcp.approx.ftz.f32 %0, %1;": "=f"(b): "f"(a));
  return b;
}
```

**관련 내용**：
- 관련 내용의 `1.0f / a` 빠른관련 내용많은

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUDA PTX ISA)대해 `rcp.approx` 로및이 부분은 원문의 해당 기술 설명을 이어서 서술한다`ftz`）있다관련 내용
https://docs.nvidia.com/cuda/parallel-thread-execution/

#### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다계산

Cutlass Block Scaled GEMM 이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용관련 내용의 swizzled 관련 내용

```cpp
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int numCols, SFType* SFout) {
  
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  // 만있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)쓰기 SF（각 CVT_FP4_NUM_THREADS_PER_SF 개thread쓰기관련 내용개 SF）
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF 관련 내용인덱스（K 차원각 16 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다개 SF）
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF 관련 내용[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // 인덱스：[mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // 계산 M 차원의 tile 인덱스
    int32_t mTileIdx = mIdx / (32 * 4);  // 각개 M tile 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (128 row)
    int factor = CVT_FP4_SF_VEC_SIZE * 4;  // 16 * 4 = 64
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;  // M tile 의관련 내용

    // 계산 K 차원의 tile 인덱스
    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;  // K tile 의관련 내용

    // M tile 관련 내용이다column관련 내용[32, 4]
    int32_t outerMIdx = (mIdx % 32);  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer M)인덱스（0-31）
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer M)인덱스（0-3）
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer K)인덱스（0-3）
    int64_t innerKStride = 1;

    // 계산관련 내용
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

관련 내용설명：
- SF 관련 내용사용 5 관련 내용 (:)`[numMTiles, numKTiles, 32, 4, 4]`
- M 차원관련 내용로 128 row관련 내용개 tile(32×4)
- K 차원관련 내용로 64 개관련 내용개 tile(16×4)
- M tile 관련 내용사용column관련 내용
- 관련 내용최적화 Tensor Core 의관련 내용모드

#### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Kernel)

관련 내용이다실행한다관련 내용의주요 kernel 함수：

```cpp
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(256, 6) cvt_fp16_to_fp4(
    int32_t numRows, int32_t numCols, Type const* in, 
    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = 
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);  // 16 / 8 = 2
  
  // 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  // SFScale 와하관련 내용개 GEMM 의 alpha 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (448.0 / Alpha_A / 6.0)
  float const SFScaleVal = SFScale == nullptr? 1.0f: SFScale[0];

  // 입력텐서의row/column관련 내용
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD; 
         colIdx += blockDim.x) {
      
      // 읽기입력이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16 8)개관련 내용
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      
      // 출력관련 내용 (8)개관련 내용로 1 개 uint32_t）
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      // 얻는다 SF 출력관련 내용
      auto sf_out = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
          rowIdx, colIdx, numCols, SFout);

      // 실행한다관련 내용
      out_pos = cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
}
```

Kernel 설정:
- `__launch_bounds__(256, 6)`: 각개 block 256개thread,각개 SM 관련 내용많은 6개 block
- 각개thread관련 내용 (8)개관련 내용
- 관련 내용사용 grid-stride loop 관련 내용있다row

#### Warp 관련 내용함수

관련 내용이다에서 warp 관련 내용실행한다관련 내용의관련 내용함수：

```cpp
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  
  // 1. 계산관련 내용큰관련 내용각개thread관련 내용 (8)개관련 내용
  auto localMax = __habs2(vec.elts[0]);  // 관련 내용대해관련 내용
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));  // 관련 내용대해관련 내용
  }

  // 2. Warp 관련 내용얻는다 16 개관련 내용의관련 내용큰관련 내용 (2)개thread）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SF)
  // vecMax / 6.0 이다원본 scale，관련 내용로 SFScaleVal 후관련 내용까지 FP8
  float SFValue = SFScaleVal * (vecMax * 0.16666666666666666f);  // 0.1666... = 1/6
  
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
    // 관련 내용사용 E8M0 관련 내용
    __nv_fp8_e8m0 tmp;
    tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
    SFValue = static_cast<float>(tmp);
    fp8SFVal = tmp.__x;
  } else {
    // 관련 내용사용 E4M3 관련 내용기본）
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
  }

  // 4. 계산출력관련 내용
  // 관련 내용의관련 내용 (=)원본관련 내용* outputScale，그다음관련 내용까지 FP4
  float outputScale = SFValue!= 0? SFScaleVal * reciprocal_approximate_ftz(SFValue): 0.0f;

  // 5. 쓰기관련 내용까지global memory
  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6. 관련 내용입력관련 내용까지 float 그리고관련 내용
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

  // 7. 관련 내용로 e2m1 관련 내용
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  return e2m1Vec;
}
```

핵심관련 내용 (:)
1. **관련 내용큰관련 내용계산**：관련 내용사용 `__habs2` 와 `__hmax2` 수행한다vectorization관련 내용
2. **Warp 관련 내용**：관련 내용사용 `__shfl_xor_sync` 에서 warp 이 부분은 원문의 해당 기술 설명을 이어서 서술한다얻는다 16 개관련 내용의관련 내용큰관련 내용
3. **관련 내용계산**：`vecMax / 6.0 * SFScaleVal`，그다음관련 내용까지 FP8
4. **출력관련 내용**：계산 `outputScale = SFScaleVal / SFValue`，사용된다관련 내용원본관련 내용
5. **관련 내용**：할 것이다 fp16/bf16 관련 내용로 float，관련 내용로 outputScale，다시관련 내용로 e2m1

#### 관련 내용호출한다인터페이스

```cpp
void scaled_nvfp4_quant_sm120(
    torch::Tensor& output, torch::Tensor const& input, 
    torch::Tensor& output_sf, torch::Tensor const& input_sf) {
  
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  // 관련 내용 (N)차원반드시이다 16 의관련 내용
  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");

  int multiProcessorCount = getMultiProcessorCount();

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  
  at::cuda::CUDAGuard device_guard{(char)input.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  bool useUE8M0 = false;  // 기본관련 내용사용 E4M3

  // 관련 내용입력관련 내용
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

### 0x3.2 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Kernel thread)모델

에서보다 GEMM 구현이전에는,관련 내용하이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)의thread모델,관련 내용대해관련 내용성능관련 내용

#### NVFP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Kernel thread)모델

##### Grid 와 Block 설정

```cpp
// 각개thread관련 내용 (8)개관련 내용
dim3 block(std::min(int(n / ELTS_PER_THREAD), 256));

// 각개 SM 의 block 개수
int const numBlocksPerSM = 1536 / block.x;

// Grid 크기
dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));
```

설정관련 내용 (:)

1. **Block 크기계산**：
   - `block.x = min(n / 8, 256)`
   - 각개thread관련 내용 (8)개관련 내용그래서관련 내용`n / 8` 개thread
   - 관련 내용많은 256 개thread/block（관련 내용`__launch_bounds__(256, 6)` 관련 내용

2. **각개 SM 의 Block 개수**：
   - `numBlocksPerSM = 1536 / block.x`
   - 여기의관련 내용의관련 내용이다：관련 내용`__launch_bounds__`，이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SM)줄인다왜냐하면그리고row관련 내용아니관련 내용의관련 내용

3. **Grid 크기**：
   - `grid.x = min(m, multiProcessorCount * numBlocksPerSM)`
   - 각개 block 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row)
   - 관련 내용사용 grid-stride loop 관련 내용있다row

##### Kernel 실행한다모델

```cpp
__launch_bounds__(256, 6)  // 각개 block 256 thread，각개 SM 관련 내용많은 6 개 block
cvt_fp16_to_fp4(...) {
  // Grid-stride loop 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row)
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    // Block-stride loop 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)
    for (int colIdx = threadIdx.x; colIdx < numCols / 8; colIdx += blockDim.x) {
      // 각개thread관련 내용 (8)개관련 내용
      //...
    }
  }
}
```

실행한다관련 내용 (:)

1. **row관련 내용그리고row**（Grid 차원）：
   - 각개 block 담당관련 내용 (row)또는많은row
   - 관련 내용사용 grid-stride loop：`rowIdx += gridDim.x`
   - 관련 내용있다row모두관련 내용

2. **column관련 내용그리고row**（Block 차원）：
   - 각개thread담당 8 개관련 내용
   - 관련 내용사용 block-stride loop：`colIdx += blockDim.x`
   - thread 0 관련 내용[0-7], [256×8-256×8+7],...
   - thread 1 관련 내용[8-15], [256×8+8-256×8+15],...

3. **Warp 관련 내용**：
   - 각 2 개thread（16 개관련 내용계산관련 내용개 scale factor
   - 관련 내용사용 `__shfl_xor_sync` 수행한다 warp 관련 내용
   - 32 개thread（관련 내용개 warp）이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (256)개관련 내용생성한다 16 개 scale factors

##### 관련 내용모드

global memory읽기(Coalesced):
```
Warp 0 (Threads 0-31):
  Thread 0:  읽기 input[row][0:8]
  Thread 1:  읽기 input[row][8:16]
...
  Thread 31: 읽기 input[row][248:256]

（여기의관련 내용모드이다관련 내용의관련 내용읽기，관련 내용이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다그리고；관련 내용이다아니이다“관련 내용로사용자는관련 내용의 stride/정렬로관련 내용
```

global memory쓰기:
```
관련 내용각 2 개 FP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1)개 uint8）：
  Thread 0:  쓰기 output[row][0]
  Thread 1:  쓰기 output[row][1]
...

Scale factors（swizzled layout）：
  Thread 0:  쓰기 SF[swizzled_offset] → 1 byte (FP8)
  Thread 2:  쓰기 SF[swizzled_offset] → 1 byte
...（각 2 개thread쓰기관련 내용개 SF）
```

#### MXFP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Kernel thread)모델

MXFP4 의thread모델와 NVFP4 차이아니많은,주요차이관련 내용에서관련 내용

##### 핵심차이관련 내용

```cpp
// NVFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 16;  // 16 개관련 내용 (/)
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 16 / 8 = 2;  // 2 개thread/SF

// MXFP4
constexpr int CVT_FP4_SF_VEC_SIZE = 32;  // 32 개관련 내용 (/)
constexpr int CVT_FP4_NUM_THREADS_PER_SF = 32 / 8 = 4;  // 4 개thread/SF
```

관련 내용 (:)

1. **Warp 관련 내용**：
   - NVFP4：1 관련 내용`__shfl_xor_sync(mask, val, 1)`（2 개thread관련 내용
   - MXFP4：2 관련 내용`__shfl_xor_sync`（4 개thread관련 내용
   ```cpp
   // MXFP4 관련 내용의관련 내용
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
   localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
   ```

2. **Scale Factor 관련 내용**：
   - NVFP4：각 16 개관련 내용 (1)개 SF → 각 256 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)개 SF
   - MXFP4：각 32 개관련 내용 (1)개 SF → 각 256 개관련 내용 (8)개 SF

3. **관련 내용모드**：
   - 관련 내용의 coalesced 읽기모드
   - 아니관련 내용의 SF 쓰기모드（더관련 내용

---

### 0x3.3 NVFP4 matrix multiplication구현

NVFP4 의matrix multiplication기반으로 CUTLASS 3.x 의 Block Scaled GEMM。

#### GEMM 설정관련 내용

```cpp
struct Fp4GemmSm120 {
    // A matrix설정
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // NVFP4 관련 내용
    using LayoutATag = cutlass::layout::RowMajor;                   // row관련 내용
    static constexpr int AlignmentA = 32;                           // 정렬이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (32)개관련 내용

    // B matrix설정
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;                // column관련 내용
    static constexpr int AlignmentB = 32;

    // C/D matrix설정
    using ElementD = cutlass::bfloat16_t;                           // 출력관련 내용
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    
    // 관련 내용설정
    using ElementAccumulator = float;                               // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용 float
    using ArchTag = cutlass::arch::Sm120;                           // Blackwell 관련 내용
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; // Block Scaled Tensor Op

    // 성능설정
    using ThreadBlockShape = Shape<_128,_128,_128>;                 // Tile 크기：128×128×128
    using ClusterShape = Shape<_1,_1,_1>;                           // Cluster 크기

    // Epilogue 설정：지원 per-column bias
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Collective Epilogue)
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

    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Collective Mainloop)
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

설정설명:
- **ElementA/B**：관련 내용사용 `nv_float4_t<float_e2m1_t>` 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (NVFP4)
- **AlignmentA/B**：32 개관련 내용정렬，보장높은관련 내용의관련 내용
- **ThreadBlockShape**：128×128×128 의 tile 크기，이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (register)사용와shared memory
- **OpClassBlockScaledTensorOp**：관련 내용사용 Block Scaled Tensor Core 관련 내용
- **EVTOp**：지원 per-column bias 의 epilogue 융합관련 내용

#### 파라미터관련 내용함수

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
  
  // 계산 stride
  auto stride_A = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideD{}, {m, n, 1});

  // 계산 scale factor 의 layout
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m, n, k, 1));

  if (bias) {
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bias)의관련 내용
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
    
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (fusion)파라미터
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Fp4GemmSm120::Gemm::ElementC const*>(
        bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    
    return arguments;
  } else {
    // 아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bias)의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bias)
    //...
  }
}
```

핵심관련 내용 (:)
- 계산 stride 와 scale factor layout
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CUTLASS Arguments)
- 지원이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bias)와아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bias)
- 관련 내용사용 epilogue fusion 할 것이다 bias 추가융합까지 GEMM 중

#### GEMM 실행한다함수

```cpp
void runGemmNvfp4Sm120(
    at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
    at::Tensor const& A_sf, at::Tensor const& B_sf,
    at::Tensor const& alpha, c10::optional<torch::Tensor> const& bias,
    int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  
  typename Fp4GemmSm120::Gemm gemm;

  // 관련 내용파라미터
  auto arguments = args_from_options_nvfp4_nvfp4(
      D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  
  // 할당 workspace
  size_t workspace_size = Fp4GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  // 관련 내용여부가능로실행한다
  CUTLASS_CHECK(gemm.can_implement(arguments));
  
  // 초기화
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  
  // 실행한다
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}
```

#### 관련 내용인터페이스

```cpp
void cutlass_scaled_nvfp4_mm_sm120(
    torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& A_sf, torch::Tensor const& B_sf,
    torch::Tensor const& alpha, c10::optional<torch::Tensor> const& bias) {

  // 입력관련 내용
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
  auto const k = A.sizes()[1] * 2;  // 왜냐하면관련 내용개 FP4 관련 내용로관련 내용개 uint8

  // 정렬관련 내용
  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // 계산 rounded 관련 내용
  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);  // k/16 이다 scale factor 의개수

  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scale factor)의관련 내용
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

### 0x3.4 MXFP4 관련 내용구현

MXFP4 의관련 내용구현와 NVFP4 차이아니많은,주요차이관련 내용에서관련 내용 (:)

#### 주요차이관련 내용

```cpp
// NVFP4 vs MXFP4 의핵심차이관련 내용

// 1. 관련 내용
constexpr int CVT_FP4_SF_VEC_SIZE_NVFP4 = 16;  // NVFP4: 16 개관련 내용
constexpr int CVT_FP4_SF_VEC_SIZE_MXFP4 = 32;  // MXFP4: 32 개관련 내용

// 2. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
// NVFP4: 관련 내용사용 E4M3，이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (global_scale)
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);

// MXFP4: 관련 내용사용 E8M0，아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (global_scale)
__nv_fp8_e8m0 tmp;
tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);

// 3. 출력관련 내용계산
// NVFP4: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (global_scale)
float outputScale = SFScaleVal * reciprocal_approximate_ftz(SFValue);

// MXFP4: 관련 내용사용 SF 의관련 내용
float outputScale = reciprocal_approximate_ftz(SFValue);
```

#### MXFP4 Warp 관련 내용함수

```cpp
template <class Type>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, uint8_t* SFout) {
  
  // 1. 계산관련 내용큰관련 내용각개thread 8 개관련 내용
  auto localMax = __habs2(vec.elts[0]);
  
  #pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // 2. Warp 관련 내용얻는다 32 개관련 내용의관련 내용큰관련 내용 (4)개thread）
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 3. 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다로 6.0，아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (global_scale)
  float SFValue = vecMax * 0.16666666666666666f;
  
  // 4. 관련 내용까지 E8M0
  uint8_t fp8SFVal;
  __nv_fp8_e8m0 tmp;
  tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
  SFValue = static_cast<float>(tmp);
  fp8SFVal = tmp.__x;

  // 5. 계산출력관련 내용아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (global_scale)
  float outputScale = SFValue!= 0? reciprocal_approximate_ftz(SFValue): 0.0f;

  if (SFout) {
    *SFout = fp8SFVal;
  }

  // 6-7. 관련 내용와관련 내용와 NVFP4 관련 내용
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

**핵심차이관련 내용**：
- **관련 내용**：MXFP4 관련 내용`__shfl_xor_sync`（32 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (4)개thread）
- **없음 global_scale**：MXFP4 관련 내용사용 `vecMax / 6.0` 관련 내용로 SF
- **E8M0 관련 내용**：관련 내용사용 `__nv_cvt_float_to_e8m0` 관련 내용

---

### 0x3.5 MXFP4 matrix multiplication구현

MXFP4 의matrix multiplication설정와 NVFP4 차이아니많은,주요차이관련 내용 (:)

#### GEMM 설정차이관련 내용

```cpp
struct Mxfp4GemmSm120 {
    // A matrix설정
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  // 관련 내용사용 mx_float4_t
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128;  // 더큰의정렬이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (128)개관련 내용

    // B matrix설정
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128;

    // 관련 내용설정와 NVFP4 관련 내용
    //...
};
```

**핵심차이관련 내용**：
- **ElementA/B**：관련 내용사용 `mx_float4_t` 관련 내용아니이다 `nv_float4_t`
- **AlignmentA/B**：128 개관련 내용정렬（MXFP4 의 group size 이다 32，4 개 group）
- **Scale Factor 관련 내용**：관련 내용사용 `float_ue8m0_t` 관련 내용아니이다 `float_ue4m3_t`

#### 정렬관련 내용차이관련 내용

```cpp
void cutlass_scaled_mxfp4_mm_sm120(...) {
  //...
  
  auto const k = A.sizes()[1] * 2;
  
  // MXFP4 관련 내용더관련 내용의정렬
  constexpr int alignment = 128;  // NVFP4 이다 32
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  // Scale factor 의계산도아니관련 내용
  int rounded_k = round_up(k / 32, 4);  // MXFP4: k/32，NVFP4: k/16
  
  //...
}
```

---

## 0x4. LightX2V 관련 내용중의관련 내용사용

### 0x4.1 관련 내용

LightX2V 관련 내용통해 `MMWeight` 관련 내용와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (lightx2v_kernel),구현모델weight의관련 내용와관련 내용가속。

#### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (operator)

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
    # 만약관련 내용있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (lightx2v_kernel),관련 내용사용 None
    scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm = None, None
    scaled_mxfp4_quant, cutlass_scaled_mxfp4_mm = None, None
    scaled_mxfp6_quant, cutlass_scaled_mxfp6_mxfp8_mm = None, None
    scaled_mxfp8_quant, cutlass_scaled_mxfp8_mm = None, None
```

### 0x4.2 NVFP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)

LightX2V 구현 `MMWeightNvfp4` 관련 내용와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (NVFP4)의weight。

#### 관련 내용

```python
@MM_WEIGHT_REGISTER("nvfp4")
class MMWeightNvfp4(MMWeightQuantNvfp4Template):
    """
    NVFP4 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
    - Weight: NVFP4 관련 내용
    - Act: NVFP4 관련 내용
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
        # 관련 내용함수
        self.load_func = self.load_nvfp4
        self.weight_need_transpose = True
        self.act_quant_func = self.act_quant_nvfp4
```

#### weight로드

weight로드관련 내용로드로하관련 내용 (:)

```python
def _get_cuda_tensor_pair(self, source, is_lazy):
    # 1. 로드관련 내용후의weight
    weight = source.get_tensor(self.weight_name).to(AI_DEVICE)
    
    # 2. 로드weight의 scale factors
    scale = source.get_tensor(self.weight_scale_name).to(AI_DEVICE)
    
    # 3. 계산또는로드 input_global_scale
    if self.input_absmax_name in source:
        # 부터관련 내용계산
        input_absmax = source.get_tensor(self.input_absmax_name)
        input_global_scale = (2688.0 / input_absmax).to(torch.float32)
        weight_global_scale = source.get_tensor(self.weight_global_scale_name)
        alpha = 1.0 / (input_global_scale * weight_global_scale)
    else:
        # 관련 내용로드
        input_global_scale = source.get_tensor(self.input_global_scale_name)
        alpha = source.get_tensor(self.alpha_name)
    
    return weight, scale, input_global_scale, alpha
```

핵심파라미터설명:
- `weight`: 관련 내용후의weight,shape 로 `(out_features, in_features//2)`,dtype 로 `uint8`
- `scale`: weight의 scale factors,dtype 로 `float8_e4m3fn`
- `input_global_scale`: 입력의이 부분은 원문의 해당 기술 설명을 이어서 서술한다,사용된다관련 내용
- `alpha`: 출력관련 내용,`alpha = 1.0 / (input_global_scale * weight_global_scale)`

#### 관련 내용

```python
def apply(self, input_tensor):
    # 1. 관련 내용입력관련 내용
    # input_tensor: (batch_size, in_features), dtype=bfloat16
    input_tensor_quant, input_tensor_scale = self.act_quant_func(input_tensor)
    # input_tensor_quant: (batch_size, in_features//2), dtype=uint8
    # input_tensor_scale: (batch_size, in_features//16), dtype=float8_e4m3fn
    
    # 2. 실행한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (matrix multiplication)
    output_tensor = cutlass_scaled_nvfp4_mm(
        input_tensor_quant,      # 관련 내용후의입력
        self.weight,             # 관련 내용후의weight
        input_tensor_scale,      # 입력의 scale factors
        self.weight_scale,       # weight의 scale factors
        alpha=self.alpha,        # 출력관련 내용
        bias=self.bias,          # 가능관련 내용의 bias
    )
    # output_tensor: (batch_size, out_features), dtype=bfloat16
    
    return output_tensor
```

#### 관련 내용함수

```python
def act_quant_nvfp4(self, x):
    """
    대해입력관련 내용수행한다 NVFP4 관련 내용
    
    Args:
        x: 입력텐서,shape=(batch_size, in_features), dtype=bfloat16
    
    Returns:
        input_tensor_quant: 관련 내용후의텐서,shape=(batch_size, in_features//2)
        input_tensor_scale: scale factors,shape=(batch_size, in_features//16)
    """
    input_tensor_quant, input_tensor_scale = scaled_nvfp4_quant(
        x, 
        self.input_global_scale
    )
    return input_tensor_quant, input_tensor_scale
```

### 0x4.3 완전한의관련 내용

아래이다관련 내용개완전한의관련 내용예제:

```python
# 1. 생성한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)대해관련 내용
mm_weight = MMWeightNvfp4(
    weight_name="transformer.blocks.0.attn.qkv.weight",
    bias_name="transformer.blocks.0.attn.qkv.bias",
    lazy_load=True,
    lazy_load_file="/path/to/quantized_model",
)

# 2. 로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
mm_weight.load(weight_dict)

# 3. 할 것이다weight로드까지 GPU
mm_weight.to_cuda()

# 4. 관련 내용
input_tensor = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device="cuda")
output_tensor = mm_weight.apply(input_tensor)
# 출력 shape 관련 내용대응 Linear 의 out_features

# 5. 관련 내용완료후가능로관련 내용까지 CPU
mm_weight.to_cpu()
```

### 0x4.4 관련 내용모델관련 내용

LightX2V 관련 내용모델이 부분은 원문의 해당 기술 설명을 이어서 서술한다,가능로할 것이다 FP16/BF16 모델관련 내용로 NVFP4 관련 내용모델。

#### weight관련 내용

```python
# tools/convert/quant/quant.py
def quantize_weight_nvfp4(weight, calib_data):
    """
    할 것이다weight관련 내용로 NVFP4 관련 내용
    
    Args:
        weight: 원본weight,shape=(out_features, in_features), dtype=bfloat16
        calib_data: 관련 내용,사용된다계산 input_global_scale
    
    Returns:
        quantized_weight: 관련 내용후의weight
        weight_scale: weight의 scale factors
        input_global_scale: 입력의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
        weight_global_scale: weight의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    """
    # 1. 계산 input_global_scale
    input_absmax = calib_data.abs().max()
    input_global_scale = 2688.0 / input_absmax
    
    # 2. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
    weight = weight.to("cuda").to(torch.bfloat16)
    quantized_weight, weight_scale = scaled_nvfp4_quant(
        weight, 
        torch.tensor(input_global_scale, device="cuda")
    )
    
    # 3. 계산 weight_global_scale
    weight_absmax = weight.abs().max()
    weight_global_scale = 2688.0 / weight_absmax
    
    return quantized_weight, weight_scale, input_global_scale, weight_global_scale
```

#### 저장관련 내용모델

```python
def save_quantized_model(model, output_path):
    """저장관련 내용후의모델"""
    state_dict = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'mm_weight') and isinstance(module.mm_weight, MMWeightNvfp4):
            # 저장이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)
            state_dict[f"{name}.weight"] = module.mm_weight.weight
            state_dict[f"{name}.weight_scale"] = module.mm_weight.weight_scale
            state_dict[f"{name}.input_global_scale"] = module.mm_weight.input_global_scale
            state_dict[f"{name}.weight_global_scale"] = module.mm_weight.weight_global_scale
            
            if module.mm_weight.bias is not None:
                state_dict[f"{name}.bias"] = module.mm_weight.bias
    
    # 관련 내용사용 safetensors 저장
    from safetensors.torch import save_file
    save_file(state_dict, output_path)
```


### 0x4.5 관련 내용응용관련 내용

#### 관련 내용생성한다모델가속

LightX2V 관련 내용주요사용된다관련 내용생성한다모델(이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Wan2.2), HunyuanVideo)의관련 내용가속:

```python
# 예제: Wan2.2 모델의 Transformer block
class TransformerBlock:
    def __init__(self):
        # QKV projection 관련 내용사용 NVFP4 관련 내용
        self.qkv = MMWeightNvfp4(
            weight_name="transformer.blocks.0.attn.qkv.weight",
            bias_name="transformer.blocks.0.attn.qkv.bias",
        )
        
        # MLP 관련 내용사용 NVFP4 관련 내용
        self.mlp_fc1 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc1.weight",
            bias_name="transformer.blocks.0.mlp.fc1.bias",
        )
        self.mlp_fc2 = MMWeightNvfp4(
            weight_name="transformer.blocks.0.mlp.fc2.weight",
            bias_name="transformer.blocks.0.mlp.fc2.bias",
        )
    
    def forward(self, x):
        # 1. QKV projection (관련 내용가속)
        qkv = self.qkv.apply(x)  # (B, L, 3*D)
        
        # 2. Attention (FP16/BF16)
        attn_out = self.attention(qkv)
        
        # 3. MLP (관련 내용가속)
        mlp_out = self.mlp_fc2.apply(
            F.gelu(self.mlp_fc1.apply(attn_out))
        )
        
        return mlp_out
```

## 0x5. 정리

That's all.

