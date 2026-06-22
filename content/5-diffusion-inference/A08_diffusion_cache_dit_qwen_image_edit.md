# [Diffusion 추론] cache-dit: Qwen-Image-Edit 2x 무손실 가속!

> 원문: https://zhuanlan.zhihu.com/p/1941503245764792443

### 0x00 서문

![](images/A08_diffusion_cache_dit_qwen_image_edit/v2-9809b6e2381e7db5b47226760ea4811f_r.png)
Qwen-Image-Edit

본 글은 cache-dit을 사용하여 Qwen-Image-Edit에 cache 가속을 적용하는 방법을 보여줍니다. cache-dit: DBCache F8B0 + TaylorSeer + Cache CFG 설정으로 NVIDIA L20에서 실행 시, 약 **2x** 성능 가속이 가능하며 **효과는 거의 무손실**입니다. 전체 코드 링크: cache-dit/examples/run_qwen_image_edit.py, cache-dit 상세 사용 문서는 다음을 참고하세요: An Unified Cache Acceleration Toolbox for DiTs

### 0x01 Cache 가속

cache-dit과 dev 버전 diffusers 설치:
```
pip install git+https://github.com/huggingface/diffusers
pip install -U cache-dit
```

바로 예제 코드를 첨부합니다. cache-dit은 Qwen-Image-Edit에 **DBCache, TaylorSeer** 및 **cache CFG** 캐시 가속 지원을 제공합니다. 참고: **최신 cache-dit과 최신 diffusers를 설치해야 합니다.**

```python
import os
import time
import torch
import argparse

from PIL import Image
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from utils import GiB
import cache_dit


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # 일반 인자
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=4)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=8)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.12)
    parser.add_argument("--warmup-steps", type=int, default=8)
    return parser.parse_args()


args = get_args()
print(args)


pipe = QwenImageEditPipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_EDIT_DIR",
        "Qwen/Qwen-Image-Edit",
    ),
    torch_dtype=torch.bfloat16,
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

if args.cache:
    cache_options = {
        "cache_type": cache_dit.DBCache,
        "warmup_steps": args.warmup_steps,
        "max_cached_steps": -1,  # -1은 제한 없음
        "Fn_compute_blocks": args.Fn_compute_blocks,
        "Bn_compute_blocks": args.Bn_compute_blocks,
        "residual_diff_threshold": args.rdt,
        # CFG: classifier free guidance 여부
        "do_separate_classifier_free_guidance": True,
        "cfg_compute_first": False,
        "enable_taylorseer": args.taylorseer,
        "enable_encoder_taylorseer": args.taylorseer,
        # TaylorSeer cache 타입: hidden_states 또는 residual
        "taylorseer_cache_type": "residual",
        "taylorseer_kwargs": {
            "n_derivatives": args.taylorseer_order,
        },
    }
    cache_type_str = "DBCACHE"
    cache_type_str = (
        f"{cache_type_str}_F{args.Fn_compute_blocks}"
        f"B{args.Bn_compute_blocks}W{args.warmup_steps}"
        f"T{int(args.taylorseer)}O{args.taylorseer_order}_"
        f"R{args.rdt}"
    )

    print(f"cache options:\n{cache_options}")

    cache_dit.enable_cache(
        pipe,
        **cache_options,
    )
else:
    cache_type_str = "NONE"


if torch.cuda.device_count() <= 1:
    # 메모리 절약 활성화
    pipe.enable_model_cpu_offload()


image = Image.open("./data/bear.png").convert("RGB")
prompt = "Only change the bear's color to purple"

if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 8192
    pipe.transformer.compile_repeated_blocks(mode="default")

    # 워밍업
    image = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=" ",
        generator=torch.Generator(device="cpu").manual_seed(0),
        true_cfg_scale=4.0,
        num_inference_steps=50,
    ).images[0]

start = time.time()

image = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=" ",
    generator=torch.Generator(device="cpu").manual_seed(0),
    true_cfg_scale=4.0,
    num_inference_steps=50,
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit.C{int(args.compile)}_{cache_type_str}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
```

실행 스크립트:
```
python3 run_qwen_image_edit.py # baseline
python3 run_qwen_image_edit.py --cache --taylorseer
```

### 0x02 가속 효과

- Baseline
![](images/A08_diffusion_cache_dit_qwen_image_edit/v2-2b840b86fd754dcff141cd729a3e3bfb_r.png)
Baseline w/o cache-dit

- cache-dit: DBCache F8B0 + TaylorSeer **O(4)** + Cache CFG, **2x** Speed Up
![](images/A08_diffusion_cache_dit_qwen_image_edit/v2-e5d16020bf6c4d676e59c2af57376890_r.png)
2x Speed Up w/ cache-dit

가속 후 생성된 내용이 Baseline과 기본적으로 일치하는 것을 볼 수 있습니다. cache_dit.summary()가 cache 통계 정보를 출력합니다:
```
⚡️Cache Steps and Residual Diffs Statistics: QwenImageEditPipeline

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |
|-------------|-----------|-----------|-----------|-----------|-----------|
| 25          | 0.034     | 0.069     | 0.109     | 0.148     | 0.229     |


⚡️CFG Cache Steps and Residual Diffs Statistics: QwenImageEditPipeline

| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |
|-----------------|-----------|-----------|-----------|-----------|-----------|
| 25              | 0.034     | 0.069     | 0.109     | 0.148     | 0.229     |
```

### 0x03 워밍업의 영향

대부분의 확산 모델은 디노이징 단계의 **앞부분 스텝이 고노이즈** 단계에 해당하므로, 합리적인 cache 로직은 앞부분의 몇 스텝을 워밍업(즉 cache를 하지 않음)하고, **뒷부분의 저노이즈 스텝에 cache를 적용**하는 것입니다. 실측 결과, warmup_steps > 0을 지정하면 정확도가 크게 향상됩니다. 예를 들어 Qwen-Image-Edit 예시에서 warmup_steps=8을 사용할 수 있으며, 가속비에 대한 영향은 크지 않습니다.

### 0x04 총결

본 글은 cache-dit을 사용하여 Qwen-Image-Edit에 cache 가속을 적용하는 방법을 보여줬습니다. cache-dit: DBCache F8B0 + TaylorSeer + Cache CFG 설정으로 NVIDIA L20에서 약 **2x** 성능 가속, 효과 거의 무손실. 전체 코드 링크: https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image_edit.py
