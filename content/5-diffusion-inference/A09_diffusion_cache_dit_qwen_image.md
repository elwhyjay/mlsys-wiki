# [Diffusion 추론] cache-dit: Qwen-Image 1.5x 무손실 가속!

> 원문: https://zhuanlan.zhihu.com/p/1938547315221705644

### 0x00 서문

![](images/A09_diffusion_cache_dit_qwen_image/v2-30d5523df5c06c574f4ba944aeedbca3_r.png)
Qwen-Image

본 글은 cache-dit을 사용하여 Qwen-Image에 cache 가속을 적용하는 방법을 보여줍니다. cache-dit: DBCache F8B0 + TaylorSeer + Cache CFG 설정으로 NVIDIA L20에서(cpu offload 필요) 약 **1.5x** 성능 가속이 가능하며 **효과는 거의 무손실**입니다. 전체 코드 링크: cache-dit/examples/run_qwen_image.py

### 0x01 Cache 가속

cache-dit과 dev 버전 diffusers 설치:
```
pip install git+https://github.com/huggingface/diffusers
pip install -U cache-dit
```

바로 예제 코드를 첨부합니다. cache-dit은 Qwen-Image에 **DBCache, TaylorSeer** 및 **cache CFG** 캐시 가속 지원을 제공합니다. 참고: **최신 cache-dit과 최신 diffusers를 설치해야 합니다.**

```python
import os
import time
import torch
import argparse
from diffusers import QwenImagePipeline
from cache_dit


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=4)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=8)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.12)
    parser.add_argument("--warmup-steps", type=int, default=8)
    return parser.parse_args()


args = get_args()
print(args)

pipe = QwenImagePipeline.from_pretrained(
    os.environ.get("QWEN_IMAGE_DIR", "Qwen/Qwen-Image"),
    torch_dtype=torch.bfloat16,
)

if args.cache:
    cache_options = {
        "cache_type": CacheType.DBCache,
        "warmup_steps": args.warmup_steps,
        "max_cached_steps": -1,  # -1은 제한 없음
        "Fn_compute_blocks": args.Fn_compute_blocks,
        "Bn_compute_blocks": args.Bn_compute_blocks,
        "residual_diff_threshold": args.rdt,
        "do_separate_classifier_free_guidance": True,
        "cfg_compute_first": False,
        "enable_taylorseer": args.taylorseer,
        "enable_encoder_taylorseer": args.taylorseer,
        "taylorseer_cache_type": "residual",
        "taylorseer_kwargs": {
            "n_derivatives": args.taylorseer_order,
        },
    }
    cache_type_str = (
        f"DBCACHE_F{args.Fn_compute_blocks}"
        f"B{args.Bn_compute_blocks}W{args.warmup_steps}"
        f"T{int(args.taylorseer)}O{args.taylorseer_order}_"
        f"R{args.rdt}"
    )
    print(f"cache options:\n{cache_options}")
    cache_dit.enable_cache(pipe, **cache_options)
else:
    cache_type_str = "NONE"

# 메모리 절약 활성화
pipe.enable_model_cpu_offload()

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.",
    "zh": "超清，4K，电影级构图",
}

prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee   $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

negative_prompt = " "

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

start = time.time()

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

end = time.time()

if hasattr(pipe.transformer, "_cached_steps"):
    cached_steps = pipe.transformer._cached_steps
    residual_diffs = pipe.transformer._residual_diffs
    print(f"Cache Steps: {len(cached_steps)}, {cached_steps}")
    print(f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}")
if hasattr(pipe.transformer, "_cfg_cached_steps"):
    cfg_cached_steps = pipe.transformer._cfg_cached_steps
    cfg_residual_diffs = pipe.transformer._cfg_residual_diffs
    print(f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps}")
    print(f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}")

time_cost = end - start
save_path = f"qwen-image.{cache_type_str}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
```

실행 스크립트:
```
python3 run_qwen_image.py # baseline
python3 run_qwen_image.py --cache --Fn 8 --Bn 0 --rdt 0.12 --taylorseer --order 2
python3 run_qwen_image.py --cache --Fn 8 --Bn 0 --rdt 0.12 --taylorseer --order 4
```

### 0x02 가속 효과

- Baseline
![](images/A09_diffusion_cache_dit_qwen_image/v2-f19e3ae7e46ea1c979fb9f6341b1f0e6_r.png)
Baseline w/o cache-dit

- cache-dit: DBCache F8B0 + TaylorSeer **O(2)** + Cache CFG, 1.5x Speed Up
![](images/A09_diffusion_cache_dit_qwen_image/v2-c62324074eaa6318fd294e9e99d2fe23_r.png)
cache-dit 1.5x Speed Up: TaylorSeer O(2)

- cache-dit: DBCache F8B0 + TaylorSeer **O(4)** + Cache CFG, 1.5x Speed Up
![](images/A09_diffusion_cache_dit_qwen_image/v2-ae6f74c696abf26ec735cf1707f09036_r.png)
cache-dit 1.5x Speed Up: TaylorSeer O(4)

TaylorSeer 차수가 높을수록 정확도가 좋아지는 것을 볼 수 있습니다. TaylorSeer O(4)의 이미지는 칠판 위에 생성된 내용이 Baseline과 기본적으로 일치하지만, TaylorSeer O(2)는 새로운 환형 조명관을 생성합니다.

### 0x03 워밍업의 영향

대부분의 확산 모델은 디노이징 단계의 **앞부분 스텝이 고노이즈** 단계에 해당하므로, 합리적인 cache 로직은 앞부분의 몇 스텝을 워밍업(즉 cache를 하지 않음)하고, **뒷부분의 저노이즈 스텝에 cache를 적용**하는 것입니다. 실측 결과, warmup_steps > 0을 지정하면 정확도가 크게 향상됩니다. 예를 들어 Qwen-Image 예시에서 warmup_steps=8을 사용할 수 있으며, 가속비에 대한 영향은 크지 않습니다. 결과적으로 "전선"의 생성이 Baseline에 더 가까워졌습니다.

![](images/A09_diffusion_cache_dit_qwen_image/v2-0d2c1ce22797b9ba6e6d045b2784f746_r.png)
DBCache F8B0 + TaylorSeer O(4) + Cache CFG + warmup_steps 8

### 0x04 총결

본 글은 cache-dit을 사용하여 Qwen-Image에 cache 가속을 적용하는 방법을 보여줬습니다. cache-dit: DBCache F8B0 + TaylorSeer + Cache CFG 설정으로 NVIDIA L20에서(cpu offload 필요) 약 **1.5x** 성능 가속, 효과 거의 무손실. 전체 코드 링크: https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py
