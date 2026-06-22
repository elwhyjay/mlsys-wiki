# OpenAI/Triton MLIR 1장: Triton DSL

## 본 글은 GiantPandaCV에 최초 게시되었으며, 저자의 허락 없이 무단 전재를 금합니다

### 머리말

지난 1장의 반응이 꽤 좋았고, 많은 분들이 Triton의 구체적인 최적화에는 어떤 것이 있는지, 왜 cuBLAS보다 좋은 성능을 낼 수 있는지 빨리 다음 글을 보고 싶다고 메시지를 주셨습니다. 너무 조급해하지 않으셔도 됩니다. 사실 이것이 제가 이 시리즈 글을 쓰기로 한 본래 취지이기도 합니다. 여러분과 함께 Triton의 [DSL](<https://zhida.zhihu.com/search?content_id=227774756&content_type=Article&match_order=1&q=DSL&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiJEU0wiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjc3NzQ3NTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.pA1p6BkNfGzSCnFUdj9V66SugFg5iRWnkKbJ6Z3sVUU&zhida_source=entity>) frontend부터 최종 machine code 생성까지의 흐름을 한 단계씩 명확하게 이해해 나가면서, 컴파일이 고성능 컴퓨팅에서 어떤 역할을 하는지 보여드리려 합니다. 우선 OpenAI가 Triton에 대해 내세우는 광고 문구를 보겠습니다.

"An open-source python-like programming language which enables researchers with no [CUDA](<https://zhida.zhihu.com/search?content_id=227774756&content_type=Article&match_order=1&q=CUDA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiJDVURBIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjI3Nzc0NzU2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.lQ6cJcDg7J_IuI6m01NLuzNiPTNnFfPugFIbR4Dzu_0&zhida_source=entity>) experience to write highly efficient GPU code -- most of the time on par with what an expert would be able to produce"

홍보 문구가 정말 강력합니다. Triton은 완전한 오픈 소스 컴파일 흐름으로서 python 기반의 frontend를 제공하는데, 여기서는 이를 DSL이라고 부르겠습니다. 이것이 바로 이 글에서 소개할 주된 내용입니다. DSL은 Domain Specific Language의 약자입니다. 왜 DSL을 설계할까요? 사실 DSL을 설계하는 목적은 해당 도구를 사용하는 사람이 낮은 입문 비용으로 그 도구나 소프트웨어 스택이 가져다주는 성능 향상을 체험할 수 있도록 하기 위함입니다. [PyTorch](<https://zhida.zhihu.com/search?content_id=227774756&content_type=Article&match_order=1&q=PyTorch&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiJQeVRvcmNoIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjI3Nzc0NzU2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.AKUbOYqHxKXa_-il20nWYWxpzr7nyMt2tuoSlzndcFY&zhida_source=entity>), TensorFlow, MXNet, Taichi, [TVM](<https://zhida.zhihu.com/search?content_id=227774756&content_type=Article&match_order=1&q=TVM&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiJUVk0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjc3NzQ3NTYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.0fl8bvUB5DycX1DkCCepgq3j06iFn8y0Ttvppq8X58E&zhida_source=entity>) 등에 비유할 수 있는데, 이것들은 모두 사용자에게 비교적 명확한 python api를 제공하고, 사용자는 이 python api 사용 규약과 일반적인 python 개발 흐름을 학습하는 데 일정한 시간만 들이면, 소프트웨어나 framework의 내부 디테일을 완전히 파악하지 않은 채로도 최상의 개발 경험을 얻을 수 있습니다. 물론 여기서 DSL의 설계 철학에 대해 좀 더 이야기하자면, 제가 지난 몇 년간 소프트웨어 시스템 개발과 실전을 통해 얻은 경험으로 볼 때, DSL 설계에서 가장 중시되는 것은 유연성, 즉 프로그래밍 언어 설계 관련 논문에서 자주 언급되는 flexibility입니다. 유연한 프로그래밍 방식은 사용자에게 차별화된 사용 경험을 제공할 수 있습니다. 딥러닝 알고리즘 종사자에게 가장 흔한 예는 사실 PyTorch의 등장입니다. caffe와 TensorFlow 1.x가 가져온 프로그래밍 방식의 전복적 변화에서, 전자는 imperative한 프로그래밍 패러다임으로 사용자에게 단순하고 제어가 쉬운 개발 경험을 제공하여 사용자의 디버깅과 기존 코드와의 상호 작용을 편리하게 하고, model 구축 과정에서 언제든지 기존의 일부 계산 그래프를 시각화하고 실행할 수 있게 하며, 더 많은 최적화 디테일을 숨겨 줍니다. 반면 후자는 declarative한 프로그래밍 패러다임으로, 사용자가 먼저 특정 placeholder api를 통해 완전한 계산 그래프를 구축한 다음, 해당 계산 그래프에 대해 전역 범위의 최적화를 수행하도록 합니다. 이렇게 하면 자연스럽게 더 많은 최적화 여지가 생깁니다. 그러나 이것이 가져오는 문제 또한 분명한데, 경험이 부족한 사용자의 경우 프로그래밍 중 버그가 발생했을 때 구체적인 문제를 빠르게 찾아내기가 매우 어렵다는 것입니다. 자, 다시 Triton으로 돌아오면, Triton이 우리에게 제공하는 프로그래밍 패러다임은 어떤 것일까요?

Triton이 우리에게 제공하는 것은 일종의 imperative한 프로그래밍 패러다임에 더 가깝지만, Triton이 매번 조작할 수 있는 단위는 Block 수준입니다. 어떤 분들은 Block이 무엇이냐고 물으실 수 있습니다. 여기서의 Block 개념은 CUDA 프로그래밍의 thread-Block 개념과 동일합니다. 즉, 우리가 CUDA 코드를 작성할 때 thread-Block 안의 각 thread를 정밀하게 프로그래밍해야 하는 것과 같습니다. 좀 더 깊이 들어가면, 사실 현재 TVM과 같은 codegen 도구나 cutlass와 같은 템플릿 라이브러리가 for-loop tiling을 수행하는 과정에서, inter-level 수준, 즉 thread-Block의 concurrent 실행 측면의 최적화는 이미 매우 잘하고 있습니다. 그러나 각 thread-Block 내부의 intra-level 수준 병렬화에 대해서는 여전히 많은 최적화 여지가 있는데, memory coalescing, shared memory의 sync 및 bank conflict 처리, 그리고 더 미세한 register-level의 Tensor Core 스케줄링 등이 포함됩니다. 위와 같은 최적화는, 매우 노련한 고성능 엔지니어가 아니거나 GPU 아키텍처와 CUDA 설계에 대한 깊은 연구와 경험이 없다면, 짧은 시간 내에 cuBLAS에 필적하는 고성능 op 라이브러리를 작성하기란 매우 어렵습니다. 동시에 우리는 PyTorch를 작성하듯이 Triton이 제공하는 DSL을 통해 원하는 op를 완벽하게 정의한 다음, 이를 다른 기존 framework의 backend에 임베드하여 codegen으로 사용할 수 있습니다. 그래서 Triton의 포지셔닝에 대해, 저는 이를 python DSL로 고성능 GPU op를 생성하는 만능 칼(스위스 군용 칼)로 자리매김하는 것이 더 적절하다고 생각합니다. 물론 Triton의 사용에는 일정한 진입 장벽이 있어서, 만약 여러분이 이전에 CUDA나 OpenCL 같은 GPU 가속기를 프로그래밍하는 언어를 작성해 본 적이 없다면, Triton을 학습하는 것만으로 cuBLAS에 어느 정도 필적하는 코드를 바로 작성하는 것은 다소 어렵다고 봅니다. 여기서 NVIDIA의 그래픽 카드를 예로 들면, Triton의 포지셔닝은 이전에 CUDA 최적화에 대한 어느 정도의 기초가 있고, 더 많은 고급 최적화 디테일은 숨기고 python 수준의 기술만으로 전체 알고리즘 흐름을 명확하게 정의한 채, 컴파일과 고급 최적화 디테일은 codegen에 맡기고자 하는 사용자에게 더 적합하다고 느껴집니다.

이렇게 길게 이야기했는데, 정리하자면 [Triton DSL](<https://zhida.zhihu.com/search?content_id=227774756&content_type=Article&match_order=1&q=Triton+DSL&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiJUcml0b24gRFNMIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjI3Nzc0NzU2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.VLTc39MbcCeIOC9-S7SuYeO-2MLWx-MOqJ3_ah1eHn8&zhida_source=entity>)이 사용자에게 어떤 일을 도와줄 수 있을까요?

  * Embedded In Python: python의 데코레이터를 사용하여 최적화하려는 kernel을 정의
  * Pointer Arithmetics: pointer arithmetic 방식으로 DRAM 위의 다차원 데이터를 조작
  * Optimizing Compiler: Block 단위의 프로그래밍 방식을 통해 사용자에게 더 많은 최적화 디테일을 숨기고, 이러한 최적화 작업을 컴파일러에 맡김



* * *

### Triton DSL 기초

Triton 공식은 DSL에 대해 PyTorch, TensorFlow 또는 TVM 같은 도구처럼 비교적 상세한 설명과 소개를 가지고 있지 않아서, 초보자가 입문하기에는 다소 진입 장벽이 있습니다. DSL에 대한 공식 문서는 다음 주소에 있습니다.

[](<https://link.zhihu.com/?target=https%3A//triton-lang.org/main/python-api/triton.language.html>)

제가 Triton으로 2차 개발을 진행하는 과정에서 어떤 부분은 이미 구식이 된 반면 문서가 미처 갱신되지 않은 것을 발견했기에, 여기서는 현재 Triton의 main 브랜치 코드를 기준으로 소개하겠습니다. Triton이라는 프로그래밍 언어와 관련된 대부분의 내용은 /python/triton 디렉터리에 위치합니다. 해당 디렉터리 아래의 compiler, language, runtime은 Triton DSL이 구체적인 workload를 기술하고, 중간 코드 생성, 그리고 최종적으로 autotuning을 통해 최적의 구현을 찾아내는 과정을 정의합니다. Triton DSL을 사용하기 위해서는 가장 처음에 다음과 같이 Triton을 개발 환경에 import 해야 합니다. 이는 이전에 PyTorch를 작성할 때 사용하던 import torch와 유사합니다.
    
    
    import triton
    import triton.language as tl

자, 다음으로 일단 tl을 import 하고 나면 Triton DSL을 사용해 다양한 workload를 구축할 수 있습니다. tl의 모든 연산은 python/triton/language/__init__.py의 __all__에서 확인할 수 있으며, 총 95개의 자주 쓰이는 연산이 정의되어 있습니다.
    
    
    __all__ = [
        "abs",
        "advance",
        "arange",
        "argmin",
        "argmax",
        "atomic_add",
        "atomic_and",
        "atomic_cas",
        "atomic_max",
        "atomic_min",
        "atomic_or",
        "atomic_xchg",
        "atomic_xor",
        "bfloat16",
        "block_type",
        "broadcast",
        "broadcast_to",
        "builtin",
        "cat",
        "cdiv",
        "constexpr",
        "cos",
        "debug_barrier",
        "device_assert",
        "device_print",
        "dot",
        "dtype",
        "exp",
        "expand_dims",
        "extra",
        "fdiv",
        "float16",
        "float32",
        "float64",
        "float8e4",
        "float8e5",
        "full",
        "function_type",
        "int1",
        "int16",
        "int32",
        "int64",
        "int8",
        "ir",
        "math",
        "load",
        "log",
        "make_block_ptr",
        "max",
        "max_contiguous",
        "maximum",
        "min",
        "minimum",
        "multiple_of",
        "num_programs",
        "pair_uniform_to_normal",
        "philox",
        "philox_impl",
        "pi32_t",
        "pointer_type",
        "program_id",
        "rand",
        "rand4x",
        "randint",
        "randint4x",
        "randn",
        "randn4x",
        "ravel",
        "reduce",
        "reshape",
        "sigmoid",
        "sin",
        "softmax",
        "sqrt",
        "static_range",
        "static_assert",
        "static_print",
        "store",
        "sum",
        "swizzle2d",
        "tensor",
        "trans",
        "triton",
        "uint16",
        "uint32",
        "uint32_to_uniform_float",
        "uint64",
        "uint8",
        "umulhi",
        "view",
        "void",
        "where",
        "xor_sum",
        "zeros",
        "zeros_like",
    ]

그리고 triton의 모든 연산은 /python/triton/__init__.py에서 확인할 수 있으며, 총 19개의 자주 쓰이는 연산이 정의되어 있습니다.
    
    
    __all__ = [
        "autotune",
        "cdiv",
        "CompilationError",
        "compile",
        "Config",
        "heuristics",
        "impl",
        "jit",
        "JITFunction",
        "KernelInterface",
        "language",
        "MockTensor",
        "next_power_of_2",
        "ops",
        "OutOfResources",
        "reinterpret",
        "runtime",
        "TensorWrapper",
        "testing",
        "program_ids_from_grid",
    ]
    

다음으로, 이 95+19개의 자주 쓰이는 연산을 통해 어떻게 "행렬 곱셈"에 대한 완전한 최적화 흐름을 정의할 수 있는지 이야기해 보겠습니다.

* * *

### Triton DSL로 행렬 곱셈 구현하기

먼저 CUDA의 kernel을 작성하는 흐름과 유사하게, 우선 연산에 필요한 입력 tensor와 출력 tensor를 정의한 다음, kernel을 launch하여 계산을 수행하고, 마지막으로 계산 결과와 golden data를 비교하여 단위 테스트를 진행합니다.

### 0x0 kernel 정의 준비 작업
    
    
    def matmul(a, b):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            ACTIVATION=activation
        )
        return c

위 코드 조각에서 우리가 비교적 낯설게 느끼는 부분은 아마도 다음의 grid와 matmul_kernel에 대한 정의일 것입니다.
    
    
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            ACTIVATION=activation
        )

여기는 CUDA 프로그래밍에서 main 함수에서 작성하는 kernel을 launch하는 부분에 그대로 비유할 수 있습니다. 다음 코드와 유사합니다.
    
    
        dim3 block(BLOCK_SIZE_M, BLOCK_SIZE_N);  
        dim3 grid((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);
        matmul_kernel<<<grid,block>>>(Ad, Bd, Cd, M, N, K);

여기서 grid는 각 grid에 포함된 thread-Block의 개수를 나타내고, block은 각 thread-Block이 launch하는 thread의 개수를 나타냅니다. 위 Triton 프로그램에서 matmul_kernel<<< >>> 뒤에서, 우리는 본질적으로 "BLOCK_SIZE_M"과 "BLOCK_SIZE_N" 이 두 차원을 합쳐서, 이후에 한 묶음의 id를 사용해 접근하도록 준비한 것입니다. triton.cdiv는 나눗셈 연산을 의미합니다. 다음으로, 가장 중요한 matmul_kernel이 어떻게 정의되어 있는지 살펴보겠습니다.

### 0x1 Triton Kernel 작성하기
    
    
    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetics` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        # You can fuse arbitrary activation functions here
        # while the accumulator is still in FP32!
        c = accumulator.to(tl.float16)
    
        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

위 코드는 matmul_kernel의 구체적인 구현 디테일에 해당하며, 우리는 이를 세 부분으로 나누어 학습할 수 있습니다.

첫 번째 부분으로, 먼저 matmul_kernel의 입력 매개변수에 어떤 것이 있는지 살펴보겠습니다. 우선 Triton에서 kernel을 정의할 때는 @triton.jit으로 데코레이션해야 합니다. a_ptr, b_ptr, c_ptr은 입력 tensor와 출력 tensor에 대응하는 시작 주소를 가리키며, M, N, K는 계산에 필요한 tensor의 차원이 각각 [M, K] x [K, N]임을 나타냅니다. stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn은 각각 a, b, c 이 세 tensor에 대해 한 element를 접근할 때 이동해야 하는 step의 크기를 나타냅니다. 그리고 뒤이어 나오는 BLOCK_SIZE_M, BLOCK_SIZE_N 등 tl.constexpr로 정의된 변수들은 모두 autotuning 시스템에서 열거 가능한 knob에 해당합니다. autotvm을 사용해 본 적이 있다면 그리 낯설지 않을 것입니다.

두 번째 부분은 id를 출력 tensor의 각 block에 매핑하는 것인데, 이 부분은 tutorial에서 L2 Cache 적중률을 높이기 위함이라고 설명하고 있습니다. 글에서 OpenAI는 "super-grouping"이라는 명칭을 사용하여 한 block 안에 포함된 block의 개수를 표현합니다. 사실 super-grouping의 원리는 매우 간단합니다. 다음 그림을 보겠습니다.

![이미지](images/img_01.png)

AxB=C 연산을 진행할 때, A의 데이터를 load할 때 row-major 방식으로 한 번에 9개의 block을 읽고, 그렇게 해서 C 행렬의 첫 번째 행 결과를 얻으려 한다면, C의 저장 방식 또한 row-major 방식이라고 했을 때, 원하는 결과를 얻기까지 총 9+81=90번의 block load 연산과 9번의 block write 연산이 필요합니다. 그러나 "super-grouping" 방식을 채택한다면, 동일하게 C 행렬에서 9번의 block write 연산을 얻기 위해서 A 행렬에 대해서는 9*3번의 load 연산을 진행하고, B 행렬에 대해서도 마찬가지로 9*3번의 load 연산을 진행하므로, block 전체에 대한 load 연산은 27+27=54번이 됩니다. 두 가지를 비교해 보면, 첫 번째 방식은 총 90번의 load + 9번의 write를 진행한 데 반해, 두 번째인 super-grouping 기법은 54번의 load와 9번의 write를 진행한 것이 됩니다. 게다가 OpenAI는 비고에서 A100에서 220TFLOPS에서 245TFLOPS로 향상시킬 수 있다고 설명하고 있습니다. 추후에 이 기법에 대해서는 별도의 장을 마련하여 소개하고 테스트를 진행할 수 있을 것입니다.

세 번째 부분은 비교적 일반적인 내용으로, CUDA 프로그래밍에 대응하면 사실 어떻게 Triton DSL을 통해 각 block을 접근하고, 그런 다음 accumulator 변수를 통해 tl.dot(a, b)의 결과를 누적할지를 탐색하는 것입니다. mask의 역할은 반복(iteration) 과정에서 경계를 넘어가는지 판단하여, 경계 범위를 초과하면 해당 block을 0으로 설정하는 것입니다. 마지막으로 결과를 비트 단위로 대응하는 c 행렬에 다시 써 주면 해당 작업이 완료됩니다.

### 0x2 단위 테스트

단위 테스트의 작성은 자명합니다. 이는 Triton으로 생성한 코드와 PyTorch의 torch.mm으로 계산한 결과가 일치하는지 비교하기 위함입니다.
    
    
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

* * *

### Triton의 autotuning

여기서는 Triton의 autotuning 기법에 대해 너무 깊이 해설하지 않고, 단지 몇 가지 작은 실험을 통해 서로 다른 search space를 정의하는 것이 matmul의 최종 TFLOPS를 상당히 향상시킬 수 있음을 보이는 데 그치겠습니다. Triton의 autotuning, 그리고 효율적인 search space를 어떻게 정의해야 하는지에 대해서는 추후의 내용에서 자세히 설명하겠습니다. 모든 실험은 NVIDIA 3090 GPU에서, batch = 1, datatype = fp16으로 진행했습니다.

OpenAI가 제공하는 기본 autotuning 공간에서는 다음과 같습니다.
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )

![이미지](images/img_02.png)

대응하는 튜닝 공간을 조정해 보면 다음과 같습니다.
    
    
    @triton.autotune(
        configs=[ 
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )

![이미지](images/img_03.png)

search space를 계속해서 조정하면 다음과 같습니다.
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )

![이미지](images/img_04.png)

한 단계 더 수정해 보면 다음과 같습니다.
    
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K'],
    )

![이미지](images/img_05.png)

위의 간단한 실험을 통해 알 수 있듯이, 비교적 좋은 TFLOPS 수치를 얻으려면 "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages", "num_warps" 모두에 대해 적절한 조정이 필요하며, 그래야 cuBLAS에 필적하거나 그 이상의 성능 상한선을 얻을 수 있습니다.  


* * *

### 정리

위에서 Triton DSL에 대한 해설과 Triton DSL을 통해 행렬 곱셈 연산을 완성하는 과정을 살펴보면서, 사용자가 기본적인 python 문법과 PyTorch 작성법만 알고, 거기에 이전에 CUDA를 사용한 경험을 가져와 PyTorch와 매우 유사한 api 몇 가지를 사용하기만 하면, NVIDIA의 그래픽 카드에서 Triton을 사용해 cuBLAS에 필적하는 성능의 고성능 op를 매우 손쉽게 생성할 수 있음을 알 수 있습니다. 만약 여러분이 Triton으로 matmul과 flashAttention을 능숙하게 작성할 수 있다면, 딥러닝의 대부분의 op를 Triton을 통해 손쉽게 cover할 수 있을 것입니다. 이후의 튜토리얼에서는 Triton이 MLIR을 사용해 리팩토링되는 과정에서 채택한 일부 엔지니어링 측면의 구성과, Triton 자체의 내부 설계가 어떻게 NVIDIA의 cuBLAS에 필적하는 고성능 알고리즘 라이브러리를 생성할 수 있게 하는지에 초점을 맞추겠습니다.
