# GTX 4090의 Attention 기묘한 현상

## 문제 1: GTX 4090의 Attention 기묘한 현상

최근 HunyuanVideo의 Flash Attention 연산자의 MFU를 탐색하고 싶어서, HunyuanVideo가 FlashAttention을 호출할 때의 q, k, v shape를 기록했다. 그런 다음 FlashAttention MFU를 테스트하는 스크립트를 작성했다. 구체적으로는 다음과 같다.

```python
import torch
import math
import time
from flash_attn import flash_attn_varlen_func

# 결과 재현성을 보장하기 위해 random seed 설정
torch.manual_seed(0)

# 새 파라미터 설정
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
batch_size = 1
dropout_p = 0.0
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 입력 데이터 생성
q = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 누적 sequence length 설정
cu_seqlens_q = torch.tensor([0, context_length], dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q.clone()

# 이론 FLOPS 계산
def calculate_attention_flops():
    return context_length * context_length * headdim * nheads * 4

# GPU 이론 peak FLOPS 얻기(사용하는 GPU 모델에 따라 조정해야 할 수 있음)
def get_gpu_peak_flops():
    # 아래는 몇 가지 일반적인 GPU의 이론 peak FLOPS(BF16)
    # A100: 312 TFLOPS
    # A6000: 142 TFLOPS
    # GTX 4090: 165 TFLOPS
    # 실제 GPU에 맞게 조정해야 한다.
    return 165 * 1024 * 1024 * 1024 # GTX 4090

# warmup
torch.cuda.nvtx.range_push("warmup")
for _ in range(10):
    _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        context_length,
        context_length,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
torch.cuda.nvtx.range_pop()
# 시간 측정과 성능 테스트
torch.cuda.synchronize()
start_time = time.time()

num_iters = 200
torch.cuda.nvtx.range_push("test")
for i in range(num_iters):
    torch.cuda.nvtx.range_push(f"iter_{i}")
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        context_length,
        context_length,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
end_time = time.time()

# 성능 지표 계산
elapsed_time = end_time - start_time
avg_time = elapsed_time / num_iters
flops = calculate_attention_flops()
flops_per_sec = flops / (avg_time*1000)
theoretical_peak_flops = get_gpu_peak_flops()
mfu = flops_per_sec / theoretical_peak_flops

print(f"\n성능 테스트 결과:")
print(f"평균 실행 시간: {avg_time*1000:.2f} ms")
print(f"MFU: {mfu*100:.2f}%")
```

구체적으로 나는 A800과 GTX 4090에서 테스트를 실행했다. `def get_gpu_peak_flops():` 함수 안은 실제 GPU 모델에 맞게 조정해야 한다. GTX 4090의 bf16 peak FLOPS는 165 TFLOPS이고, A800의 bf16 peak FLOPS는 312 TFLOPS다. 각각 실행한 테스트 결과는 다음과 같다.

- GTX 4090

성능 테스트 결과:
평균 실행 시간: 967.51 ms
MFU: 89.69%

- A800:

성능 테스트 결과:
평균 실행 시간: 698.68 ms
MFU: 65.68%

기묘한 점은 GTX 4090의 MFU가 거의 90%에 도달했다는 것이다. 느낌상 매우 이상하다.

그래서 이 API를 한 번만 호출하고, 아래 스크립트로 nsight compute profile을 수행했다.

```python
import torch
import math
import time
from flash_attn import flash_attn_varlen_func

# 결과 재현성을 보장하기 위해 random seed 설정
torch.manual_seed(0)

# 새 파라미터 설정
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
batch_size = 1
dropout_p = 0.0
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 입력 데이터 생성
q = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(context_length, nheads, headdim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# 누적 sequence length 설정
cu_seqlens_q = torch.tensor([0, context_length], dtype=torch.int32, device="cuda")
cu_seqlens_k = cu_seqlens_q.clone()

torch.cuda.synchronize()
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    context_length,
    context_length,
    dropout_p=dropout_p,
    softmax_scale=softmax_scale,
    causal=causal,
)
torch.cuda.synchronize()
print(out.shape)
```

나는 4090과 A800 모두에서 nsight compute로 profile 결과를 수집했다. Speed Of Light는 A800이 70%+의 SM 이용률에 도달했음을 보여줬지만, 4090의 SM 이용률은 40%+에 불과했다. 이는 계산된 MFU 결과와 일치하지 않는다. ncu의 RoofLine을 확인해 보니 이 kernel은 두 플랫폼 모두에서 memory bound 구간에 놓여 있었다. 이어서 Memory Workloads Analysis도 확인했는데, 이 kernel이 A800에서는 HBM에서 16GB 데이터를 읽고 GTX 4090에서는 HBM에서 6.2GB 데이터를 읽는 것을 발견했다. 이론적으로 QKV를 읽는 데는 2GB도 필요하지 않은데, 왜 차이가 이렇게 큰지 모르겠다. 이것도 매우 이상한 문제라는 느낌이다.

## 문제 2: FlashInfer 성능 문제(A800)

아래는 위 입력 설정에서 flashinfer를 측정하는 스크립트이며, `flashinfer.single_prefill_with_kv_cache`를 호출했다.

```python
import torch
import math
import time
import flashinfer

# 결과 재현성을 보장하기 위해 random seed 설정
torch.manual_seed(0)

# 파라미터 설정
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
qo_len = context_length
kv_len = context_length
num_qo_heads = nheads
num_kv_heads = nheads  # 비 MQA/GQA 모드이며, kv_heads는 qo_heads와 같다.
head_dim = headdim
batch_size = 1
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 입력 데이터 생성
q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# attention mask 생성 - non-causal 상황에서는 all-1 행렬 사용
mask = torch.full((qo_len, kv_len), True, device="cuda")

# 이론 FLOPS 계산
def calculate_attention_flops():
    return qo_len * kv_len * head_dim * num_qo_heads * 4

# GPU 이론 peak FLOPS 얻기(A100을 예로 듦)
def get_gpu_peak_flops():
    return 312 * 1024 * 1024 * 1024  # A100 BF16 TFLOPS

# warmup
print("warmup 시작...")
for _ in range(10):
    _ = flashinfer.single_prefill_with_kv_cache(
        q, k, v, 
        causal=False,
        allow_fp16_qk_reduction=True
    )

# 시간 측정과 성능 테스트
torch.cuda.synchronize()
start_time = time.time()

num_iters = 20
print(f"\n{num_iters}회 iteration 테스트 시작...")

for i in range(num_iters):
    print(f"{i+1}번째 iteration")
    out = flashinfer.single_prefill_with_kv_cache(
        q, k, v,
        custom_mask=mask,
        causal=False,
        allow_fp16_qk_reduction=True
    )
    torch.cuda.synchronize()

torch.cuda.synchronize()
end_time = time.time()

# 성능 지표 계산
elapsed_time = end_time - start_time
avg_time = elapsed_time / num_iters
flops = calculate_attention_flops()
flops_per_sec = flops / (avg_time*1000)
theoretical_peak_flops = get_gpu_peak_flops()
mfu = flops_per_sec / theoretical_peak_flops

print(f"\nFlashInfer 성능 테스트 결과:")
print(f"입력 shape:")
print(f"- Query: {q.shape}")
print(f"- Key/Value: {k.shape}")
print(f"평균 실행 시간: {avg_time*1000:.2f} ms")
print(f"TFLOPS: {flops_per_sec/1e12:.2f}")
print(f"MFU: {mfu*100:.2f}%")

# 출력 shape 검증
print(f"\n출력 shape: {out.shape}")
print(f"기대 shape: torch.Size([{qo_len}, {num_qo_heads}, {head_dim}])") 
```

나는 A800에서 테스트했으며, flashinfer의 이 연산자 MFU는 22%에 불과했다. 그래서 ncu profile도 수행했다. 스크립트는 다음과 같다.

```python
import torch
import math
import time
import flashinfer

# 결과 재현성을 보장하기 위해 random seed 설정
torch.manual_seed(0)

# 파라미터 설정
context_length = 111856
hidden_size = 3072
nheads = 24
headdim = 128  # hidden_size // nheads
qo_len = context_length
kv_len = context_length
num_qo_heads = nheads
num_kv_heads = nheads  # 비 MQA/GQA 모드이며, kv_heads는 qo_heads와 같다.
head_dim = headdim
batch_size = 1
causal = False
softmax_scale = 1.0 / math.sqrt(headdim)

# 입력 데이터 생성
q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# attention mask 생성 - non-causal 상황에서는 all-1 행렬 사용
mask = torch.full((qo_len, kv_len), True, device="cuda")

# 시간 측정과 성능 테스트
torch.cuda.synchronize()

out = flashinfer.single_prefill_with_kv_cache(
    q, k, v,
    custom_mask=mask,
    causal=False,
    allow_fp16_qk_reduction=False
)
torch.cuda.synchronize()


print(f"\n출력 shape: {out.shape}")
```

flashinfer의 이 연산자는 이런 상황에서 계산력을 가득 채우지 못하고 compute bound이며, 대량의 비 coalesced memory access가 존재한다. 그리고 Memory Workloads Analysis에서 HBM 읽기 데이터가 놀랍게도 647GB에 도달한 것을 볼 수 있었다.
