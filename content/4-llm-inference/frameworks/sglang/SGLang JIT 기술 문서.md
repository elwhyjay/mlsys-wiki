> 이 글의 원문은 https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model-note/sglang%26lightllm/SGLang_JIT_%E6%8A%80%E6%9C%AF%E6%96%87%E6%A1%A3.md 에 있습니다.

## 제목

SGLang JIT Kernel 소개

## 0x0. 서문

이전에는 SGLang에서 CUDA Kernel을 개발하려면 sgl-kernel에 cuda kernel을 작성하고, pybind 인터페이스를 export하고, operator를 등록하고, cmakelists를 수정하는 흐름을 거쳐야 했습니다. 개발이 끝나 main에 merge된 뒤에도 sgl-kernel release를 한 번 거쳐야 사용할 수 있었기 때문에 개발 흐름이 꽤 번거로웠습니다. 또한 templated kernel이 점점 많아지면서 컴파일 시간이 길어졌고, 머신을 바꾸거나 Docker 환경을 바꿔 처음부터 컴파일하면 1시간이 걸리기도 해 소프트웨어 개발 속도에 큰 영향을 주었습니다. 반복 속도를 높이기 위해 최근 몇 달 동안 우리는 TVM-FFI 기반으로 JIT kernel 개발 방식을 탐색하기 시작했고, Kernel iteration 속도를 크게 높였습니다. 이제 개발자는 더 이상 많은 컴파일 시간을 들일 필요가 없고, Kernel apply와 SGLang python 코드를 함께 업데이트할 수 있습니다. 번거로운 sgl-kernel release 과정을 거치지 않고도 현재 개발 중인 Kernel을 LLM/Diffusion/VLM 같은 모델에 빠르게 사용해 성능 이득을 얻을 수 있습니다.

이 글은 SGLang JIT kernel을 이해하거나 개발하고 싶은 개발자를 대상으로, JIT Kernel의 메커니즘, 추상화, 새 Kernel 추가 흐름을 정리합니다.

- 메커니즘: JIT kernel이 Python 호출에서 런타임 컴파일을 거쳐 CUDA kernel launch까지 어떻게 이어지는가
- 추상화: `jit_kernel` 코드가 어떤 공통 시설을 제공하는가(검증, 오류 위치 찾기, kernel launch, vectorization)
- 흐름: `add_constant`와 `fused_add_rmsnorm`을 예로 C++/CUDA 코드, Python wrapper, test/benchmark를 어떻게 구성하는가

Q&A:

- 커뮤니티는 sgl-kernel에서 JIT Kernel로의 migration 계획을 시작했습니다. 설명이 필요한 점은 이 계획이 sgl-kernel에서 template가 매우 큰 몇 개의 gemm kernel만 제거해 sgl-kernel wheel package 크기를 줄이는 것이며, sgl-kernel의 기존 모든 AOT kernel을 제거하는 것은 아니라는 점입니다. sgl-kernel의 AOT kernel에 의존하는 사용자나 프로젝트는 이 과정에서 기본적으로 영향을 받지 않습니다. migration 계획에 관심이 있거나 기여하고 싶다면 이 ISSUE에 댓글을 남겨 주세요. https://github.com/sgl-project/sglang/issues/17865
- JIT Kernel의 기반 시설, 특히 TVM-FFI 연결 부분은 mini-sglang의 작성자 https://github.com/DarkSharpness 가 주도해 완성했고, 계속 유지보수 중입니다.
- CUTE DSL 환경도 SGLang 환경에 들어왔으므로, CuteDSL로 고성능 kernel을 구현한 뒤 JIT Kernel 아래에 두는 것도 환영합니다. 예: https://github.com/sgl-project/sglang/pull/14717

> 이 글의 거의 절반은 공식 문서에서 번역해 온 내용입니다. 이 문서도 함께 보기를 권합니다. https://docs.sglang.io/developer_guide/development_jit_kernel_guide.html

## 0x1. 환경 설정

JIT kernel 개발에는 `clangd`를 language server로 사용하는 것을 추천합니다. Ubuntu/Debian에서는 [apt.llvm.org](https://apt.llvm.org/)에서 clangd를 다운로드할 수 있습니다. VS Code 사용자는 `clangd` extension 설치를 권장합니다.

모든 JIT 관련 파일은 `python/sglang/jit_kernel`에 있습니다. Ahead-of-Time 컴파일(AOT) 방식의 `sgl-kernel`과 달리 JIT kernel은 런타임에 컴파일되므로 정적인 `compile_commands.json`을 생성할 수 없습니다.

`clangd`가 code completion을 지원하도록 하려면 현재 디렉터리에서 다음을 실행합니다.

```bash
python -m sglang.jit_kernel
```

`.clangd` 설정 파일이 생성된 뒤 clangd language server를 재시작하면 모든 JIT kernel 파일을 인식할 수 있습니다.

## 0x2. 디렉터리 구조

```
python/sglang/jit_kernel/
├── __main__.py                         # Generate .clangd for clangd completion/navigation.
├── utils.py                            # load_jit/cache_once/make_cpp_args
├── csrc/
│   ├── add_constant.cuh                # Simple example
│   └── elementwise/
│       └── fused_add_rmsnorm.cuh       # Advanced example
├── include/sgl_kernel/                 # Shared C++/CUDA abstractions
│   ├── utils.h                         # DebugInfo/RuntimeCheck/PanicError/irange
│   ├── utils.cuh                       # LaunchKernel/PDL/RuntimeDeviceCheck
│   ├── tensor.h                        # TensorMatcher for shape/stride/dtype/device validation
│   ├── type.cuh                        # dtype_trait/cast/packed_t
│   ├── vec.cuh                         # AlignedVector, up to 32B=256-bit vectorization
│   └── runtime.cuh                     # runtime: get_cc_major/get_sm_count, etc.
├── add_constant.py                     # Python API for add_constant
├── norm.py                             # Python API for fused_add_rmsnorm
├── tests/
│   ├── test_add_constant.py
│   └── test_fused_add_rmsnorm.py
└── benchmark/
    └── bench_fused_add_rmsnorm.py
```

C++ 구현은 `python/sglang/jit_kernel/csrc`에 있고, 재사용 함수는 `python/sglang/jit_kernel/include`에 있습니다.

Python 인터페이스는 `python/sglang/jit_kernel`에 정의됩니다. 효율적인 foreign language binding에는 tvm-ffi(https://github.com/apache/tvm-ffi)를 사용합니다. 보통 `tvm::ffi::TensorView`만으로도 Python에서 PyTorch Tensor를 전달하기에 충분합니다.

## 0x3. JIT 경로: Python 호출에서 CUDA kernel launch까지

대응 구현은 `python/sglang/jit_kernel/utils.py`에 있습니다.

### cache_once: JIT module의 cache 방식

`functools.lru_cache`가 아니라 커스텀 `cache_once`를 사용합니다. 후자는 `torch.compile`과 호환되지 않기 때문입니다. 기능은 다음과 같습니다.

- (파라미터 → 컴파일 결과)를 cache로 사용
- 같은 파라미터 조합은 한 번만 컴파일하고, 이후에는 이미 로드된 module을 재사용

### make_cpp_args: Python 파라미터를 C++ template 파라미터로 바꾸기

`make_cpp_args`는 `int/float/bool/torch.dtype`을 C++ template parameter 문자열로 바꿉니다. 예를 들어 dtype은 `bf16_t/fp16_t/fp32_t` 같은 type alias로 매핑됩니다.

### load_jit: 런타임 컴파일 + symbol export + load

`load_jit`의 핵심 흐름은 세 단계입니다.

1. `cuda_files=["add_constant.cuh"]`를 `#include "absolute path"`로 바꿔 compilation unit에 붙입니다.
2. `TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, (symbol))`로 C++ symbol을 export합니다.
3. `tvm_ffi.cpp.load_inline`을 호출해 컴파일하고 로드한 뒤 `Module`을 반환합니다.

`args`는 template instantiation에 쓰일 뿐 아니라 module의 고유 이름(`sgl_kernel_jit_${args_joined}`)에도 들어갑니다. 서로 다른 variant의 `args`는 반드시 구분되어야 하며, 그렇지 않으면 cache가 충돌합니다.

`cuda_wrappers=[("func", "cpp_func")]`를 통해 C++ 함수를 export하고, Python에서는 `module.func`로 호출합니다.

## 0x4. 공통 추상화: kernel 작성 시 최대한 재사용해야 할 기반 시설

`python/sglang/jit_kernel/include/sgl_kernel/`가 제공하는 기반 시설은 반복적이고 오류가 나기 쉽고 문제 해결 효율에 영향을 주는 boilerplate 코드를 수렴합니다.

### irange(utils.h): 정수 범위 iteration

PyTorch와 비슷하게 정수 범위를 나타내는 `irange` 함수를 제공합니다.

```cpp
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) {        // [0, 100)
    // do something
  }
  for (auto i : host::irange(0, 100)) {     // [0, 100)
    // do something
  }
}
```

### RuntimeCheck / RuntimeDeviceCheck(utils.h & utils.cuh)

JIT kernel debug 비용은 보통 파라미터 전달 오류에서 옵니다. 예를 들어 shape/dtype/stride/device가 맞지 않는 경우입니다.

- `RuntimeCheck`: 조건이 만족되지 않으면 예외를 던지고, 파일명과 줄 번호를 포함하며, 오류 보고를 위한 optional parameter를 지원합니다.
- `RuntimeDeviceCheck`: 최근 kernel launch 상태를 검증합니다.

```cpp
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test(int hidden_size, int elements_in_vec) {
  using namespace host;
  
  RuntimeCheck(hidden_size % elements_in_vec == 0,
               "hidden_size=", hidden_size,
               " is not aligned to elements_in_vec=", elements_in_vec);
  
  RuntimeDeviceCheck();
  RuntimeDeviceCheck(cudaGetLastError());  // Explicitly pass cudaError_t.
}
```

이런 검사는 host entry(`::run`) 안에서 수행하는 것을 권장합니다. kernel launch 이후 silent wrong보다 위치를 찾기 쉽습니다.

### TensorMatcher(tensor.h)

`TensorMatcher`는 entry 검증에 사용되며 누락을 줄이고 완전한 오류 정보를 제공합니다.

특징:

- **기본 contiguous**: `with_strides(...)`를 쓰지 않으면 `view.is_contiguous()`를 요구합니다.
- **완전한 오류 정보**: 실패 시 실제 tensor의 `shape/strides/dtype/device`와 root cause를 출력합니다.
- **Symbolic 변수**: 모든 검증에서 같은 값으로 해석되어야 하며, `.unwrap()`으로 matching value를 가져옵니다.
- **유연한 matching**: size 또는 stride에 `-1`을 전달하면 임의 값을 matching합니다.

설정:

- `with_strides`: 생략하면 tensor contiguous를 기대합니다.
- `with_dtype`: template parameter로 허용할 data type을 제한합니다.
- `with_device`: template parameter로 허용할 device type을 제한합니다.
- `with_xxx`에 전달한 값은 강제 동등성 검사를 수행합니다.

```cpp
#include <sgl_kernel/tensor.h>

using namespace host;

void check(const tvm::ffi::TensorView input,
           const tvm::ffi::TensorView residual,
           const tvm::ffi::TensorView weight) {
  auto N = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_size"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({N, D})                       // input: [N, D]
      .with_strides({D, 1})                   // Require the last dimension to be contiguous.
      .with_dtype<bf16_t, fp16_t>(dtype)      // Restrict allowed dtype.
      .with_device<kDLCUDA, kDLCPU>(device)   // Restrict allowed device.
      .verify(input);

  TensorMatcher({N, D})                       // residual: [N, D]
      .with_strides({D, 1})
      .with_dtype<bf16_t, fp16_t>(dtype)      // dtype must match input.
      .with_device<kDLCUDA, kDLCPU>(device)   // device must match input.
      .verify(residual);

  TensorMatcher({D})                          // weight: [D]
      .with_dtype<bf16_t, fp16_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(weight);
  
  size_t num_tokens = N.unwrap();
  size_t hidden_size = D.unwrap();
}
```

주의: `TensorMatcher`는 temporary expression입니다. 변수에 저장하지 마세요. `TensorMatcher` chain 끝에 `//`를 추가하면 올바른 indentation을 강제할 수 있습니다.

### LaunchKernel(utils.cuh)

`LaunchKernel`은 다음 세부 사항을 통일해서 처리합니다.

- `DLDevice`에서 현재 stream을 해석합니다. PyTorch stream semantics와 맞춥니다.
- `cudaLaunchKernelEx`로 launch합니다.
- 자동으로 `RuntimeDeviceCheck(...)`를 수행합니다. `cudaGetLastError()` 검사와 같습니다.
- optional `.enable_pdl(true/false)`로 PDL 활성화 여부를 제어합니다.

`LaunchKernel::resolve_device`는 PyTorch에서 현재 `cudaStream`을 가져옵니다.

```cpp
#include <sgl_kernel/utils.cuh>
#include <dlpack/dlpack.h>

using namespace host;

__global__ void kernel(float* x) { /* ... */ }

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // Assume it is initialized correctly.
  
  // Method 1: Launch directly from DLDevice.
  LaunchKernel(num_blocks, num_threads, dev)(kernel, x);
  
  // Method 2: Resolve stream explicitly, then launch.
  cudaStream_t stream = LaunchKernel::resolve_device(dev);
  LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel, x);
}
```

### AlignedVector(vec.cuh)

`AlignedVector<T, N>`은 vectorized memory access에 쓰이는 aligned POD wrapper입니다. `fused_add_rmsnorm`이 B200에서 32B 경로를 사용할 때 이에 의존합니다.

핵심 제약: `sizeof(T) * N <= 32`, 즉 최대 32바이트(256-bit)입니다.

```cpp
#include <sgl_kernel/vec.cuh>

using device::AlignedVector;

__global__ void vec_ldg_stg(const half* src, half* dst) {
  // 32B: half(2B) * 16 = 32B
  using vec_t = AlignedVector<half, 16>;
  vec_t v;
  v.load(src, /*offset=*/blockIdx.x);
  v.store(dst, /*offset=*/blockIdx.x);
}
```

vectorization의 전제 조건은 **주소 정렬**과 **길이의 나누어떨어짐**입니다. `fused_add_rmsnorm`의 host entry는 이에 대응하는 검사를 수행합니다.

- `elements_in_vec = max_vec_size_byte / sizeof(DType)`
- `RuntimeCheck(hidden_size % elements_in_vec == 0, ...)`

## 0x5. 전체 예시: add_constant(end-to-end 흐름)

`add_constant` kernel은 입력 tensor의 각 원소에 상수를 더합니다.

Python 인터페이스 개념:

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### STEP 1: C++ kernel 작성

`python/sglang/jit_kernel/csrc/add_constant.cuh` 파일을 만들고 상수를 template parameter로 전달합니다.

```cpp
#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // LaunchKernel
#include <sgl_kernel/utils.h>    // div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors.
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 1D tensor, must be contiguous.
      .with_dtype<int32_t>()          // Must be int32.
      .with_device<kDLCUDA>(device_)  // Must be on CUDA device.
      .verify(dst)                    // Check tensor dst.
      .verify(src);                   // Check tensor src.

  // 2. Extract parameters and prepare kernel launch.
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  
  RuntimeCheck(num_elements > 0, 
               "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. Launch kernel and automatically check error code.
  LaunchKernel(grid_size, kBlockSize, device)(
      add_constant_kernel<kConstant>,
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace
```

### STEP 2: Python 인터페이스 생성

`python/sglang/jit_kernel/add_constant.py` 파일을 만듭니다.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst
```

### STEP 3: kernel 사용

```python
from sglang.jit_kernel.add_constant import add_constant

x = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device='cuda')
y = add_constant(x, 10)
# y = tensor([11, 12, 13, 14], device='cuda:0')
```

전체 예시는 `python/sglang/jit_kernel/tests/test_add_constant.py`를 참고하세요.

## 0x6. 고급 예시: fused_add_rmsnorm

### Motivation

FlashInfer 버전과 비교하면 주요 차이는 두 가지입니다.

1. **`inp+res`를 shared memory에 저장하고 다시 읽는 일을 피합니다.**
2. **B200에서 256-bit LDG(32B vectorization)를 사용합니다.**

### Modifications

도입 시 수정한 파일:

- `python/sglang/jit_kernel/norm.py`
- `python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh`
- `python/sglang/jit_kernel/tests/test_fused_add_rmsnorm.py`
- `python/sglang/jit_kernel/include/sgl_kernel/vec.cuh`

### Python wrapper

module을 컴파일하고 cache하며, export된 symbol을 호출하는 Python 함수를 제공합니다(`python/sglang/jit_kernel/norm.py`).

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_add_rmsnorm_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_add_rmsnorm",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm.cuh"],
        cuda_wrappers=[("fused_add_rmsnorm", f"FusedAddRMSNormKernel<{args}>::run")],
    )


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    module = _jit_fused_add_rmsnorm_module(input.dtype)
    module.fused_add_rmsnorm(input, residual, weight, eps)
```

설명:

- `cuda_files`는 `csrc/` 아래 구현 파일을 가리킵니다. 경로는 `csrc/` 기준 상대 경로입니다.
- `cuda_wrappers`는 `FusedAddRMSNormKernel<...>::run`을 Python 쪽의 `module.fused_add_rmsnorm`으로 export합니다.
- `@cache_once`는 같은 dtype의 module이 한 번만 컴파일/로드되도록 보장합니다.

### CUDA/C++ 구현

구현 파일: `python/sglang/jit_kernel/csrc/elementwise/fused_add_rmsnorm.cuh`.

구조는 두 계층으로 나뉩니다.

1. **host entry**: `FusedAddRMSNormKernel<DType>::run`  
   `TensorMatcher` 검증을 수행하고, 아키텍처에 따라 16B/32B vectorization을 선택하고, threads를 계산한 뒤 마지막에 `LaunchKernel(...).enable_pdl(false)(...)`로 launch합니다.
2. **device kernel**: `fused_add_rmsnorm_reg_kernel<DType, 16/32>`  
   핵심 작업:
   - `residual <- input + residual`, in-place로 다시 씁니다.
   - 같은 `inp+res`를 사용해 제곱합을 계산해 `rsqrt`를 얻고, RMSNorm을 적용해 `input`에 in-place로 다시 씁니다.

shared memory 왕복을 피합니다. `inp+res`를 residual에 직접 다시 쓰고, 동시에 register 경로에서 RMSNorm 출력을 계속 완성합니다. 중간값은 shared에 쓰지 않습니다.

B200은 256-bit vectorization을 사용합니다. host entry는 `get_cc_major`로 아키텍처를 판단하고, `cc_major >= 10`이면 32B vectorization 경로를 선택합니다. 하위 vectorization은 `device::AlignedVector`에 의존합니다.

## 0x7. 요약

이 글에서는 SGLang JIT kernel의 메커니즘, 추상화, 개발 흐름을 정리했습니다. SGLang에서 JIT kernel을 개발하려는 개발자에게 도움이 되기를 바랍니다.
