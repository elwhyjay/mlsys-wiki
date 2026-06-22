## 0x0. 서문

Cache-DiT는 Vipshop이 오픈소스화한 PyTorch native DiT 추론 가속 엔진입니다(https://github.com/vipshop/cache-dit). 하이브리드 캐시 가속과 병렬화 기술로 DiT 모델 추론을 가속합니다. 이전 버전의 Cache-DiT는 주로 오프라인 추론 시나리오에 초점을 맞췄고, 사용자는 Python script를 작성해 모델을 호출해야 했습니다. 이 방식은 연구와 실험에는 편하지만, production 환경 배포에는 그다지 친화적이지 않습니다.

최근 Cache-DiT에 완전한 Serving 기능을 구현했습니다. 목표는 단순합니다. 사용자가 SGLang을 쓰듯이 한 줄 명령으로 서비스를 시작하고, HTTP API를 통해 모델을 호출해 이미지를 생성할 수 있게 하는 것입니다. 이 기능은 단일 GPU 추론뿐 아니라 Tensor Parallelism과 Context Parallelism 같은 분산 추론 모드도 지원합니다.

이 글에서는 Cache-DiT Serving 구현 과정을 자세히 소개합니다. SGLang의 설계를 어떻게 참고했는지, 분산 시나리오에서 어떤 함정을 만났는지, 최종 해결책은 무엇이었는지 다룹니다.

## 0x1. 왜 Serving이 필요한가

production 환경에서 모델을 배포할 때는 보통 통일된 API 인터페이스를 원합니다. Web frontend, mobile app, 다른 backend service 모두 HTTP request로 이미지를 생성할 수 있고, 하위의 모델 로딩이나 GPU 관리 세부사항은 신경 쓰지 않아도 됩니다.

또한 통일된 서비스는 resource management와 monitoring에도 편리합니다. 예를 들어 동시 요청 수를 제한하거나, 각 request의 latency와 resource usage를 기록하거나, 문제가 생겼을 때 더 쉽게 추적할 수 있습니다.

Cache-DiT Serving은 바로 이런 문제를 해결하기 위해 만들어졌습니다. 사용자가 SGLang의 `sglang.launch_server`처럼 간단히 서비스를 시작하고, 곧바로 HTTP API로 호출할 수 있기를 바랐습니다.

## 0x2. 전체 아키텍처

단일 GPU 모드의 architecture는 비교적 단순합니다. FastAPI server를 시작해 HTTP request를 받고, ModelManager를 호출해 추론을 수행합니다. ModelManager는 모델 로딩, cache 관리, 추론 실행 등을 담당합니다.

하지만 분산 모드에서는 조금 더 복잡합니다. TP와 CP 두 병렬 모드에는 공통점이 있습니다. 모든 rank가 동시에 `pipe()`를 호출해 추론해야 합니다. TP는 NCCL all-reduce로 gradient와 activation을 동기화하고, CP는 all-to-all로 attention의 KV를 교환하기 때문입니다. rank 0만 `pipe()`를 호출하면 다른 rank는 NCCL communication에서 계속 기다리다가 결국 timeout deadlock이 납니다.

따라서 모든 rank의 추론 request를 동기화하는 mechanism이 필요합니다. 가장 단순한 방식은 NCCL broadcast를 사용하는 것입니다. rank 0이 HTTP request를 받은 뒤 request 내용을 다른 모든 rank에 broadcast하고, 모든 rank가 함께 추론을 실행합니다. 실행이 끝나면 rank 0은 결과를 client에 반환하고, 다른 rank는 결과를 버린 뒤 다음 request를 기다립니다.

이 방식은 단순하지만 DiT 모델에는 이미 충분합니다. DiT 모델 추론은 보통 serial하게 진행되며, LLM처럼 복잡한 continuous batching과 scheduling system이 필요하지 않기 때문입니다.

## 0x3. SGLang 설계 참고

Cache-DiT Serving을 구현하는 과정에서는 주로 SGLang의 generate 부분 serving 설계를 참고했습니다. SGLang의 이 부분 구현은 비교적 단순하고 직접적이라 이해하고 참고하기 쉬웠습니다.

구체적으로는 SGLang이 FastAPI로 HTTP interface를 구성하는 방식, command line argument를 파싱하는 방식, request lifecycle을 관리하는 방식을 참고했습니다. 이런 기본 HTTP Server architecture 설계는 SGLang의 `http_server.py`와 `launch_server.py`에 비교적 명확하게 구현되어 있습니다.

분산 추론 부분은 DiT 모델의 특성(한 번에 전체 이미지를 생성하며 token-by-token 생성이 필요 없음)을 고려해 더 단순한 방식을 사용했습니다. NCCL broadcast로 request를 직접 동기화합니다. rank 0이 HTTP server를 실행하고, request를 받으면 NCCL broadcast로 모든 rank에 request를 보냅니다. 그러면 모든 rank가 함께 추론을 실행합니다. 이렇게 하면 기존 분산 환경(torchrun이 이미 process를 관리함)을 활용하면서도 추가적인 process 간 communication overhead를 피할 수 있습니다.

## 0x4. 핵심 구현

전체 구현의 핵심은 분산 환경에서 모든 rank의 추론 request를 어떻게 동기화하느냐입니다. 아래에서 몇 가지 핵심 부분의 구현을 자세히 소개합니다.

먼저 단일 GPU 모드의 architecture는 매우 단순합니다. 아래 그림과 같습니다(Claude 4가 생성).

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request
       ↓
┌─────────────────────┐
│   FastAPI Server    │
│  (Rank 0, Port 8000)│
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│   ModelManager      │
│  - load_model()     │
│  - generate()       │
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│  DiffusionPipeline  │
│  + Cache-DiT        │
└─────────────────────┘
```

### 시작 흐름

서비스 시작 흐름은 직관적입니다. 먼저 command line argument를 파싱하고, 분산 사용 여부에 따라 환경을 초기화합니다. TP 또는 CP를 사용한다면 `torch.distributed.init_process_group`을 호출해 NCCL communication을 초기화해야 합니다.

이후 ModelManager를 만들고 모델을 load합니다. 여기서는 사용자 설정에 따라 cache를 켤지, torch.compile을 사용할지 등을 결정합니다.

마지막으로 병렬 유형에 따라 시작 방식을 선택합니다. 분산 모드(TP/CP)이면 rank 0은 HTTP server를 시작하고, 다른 rank는 worker loop를 실행하며 request를 기다립니다. 단일 GPU 모드이면 HTTP server를 바로 시작합니다.

### Broadcast 동기화 mechanism

이 부분이 전체 구현의 핵심입니다. 모든 rank가 prompt, width, height, seed 등 완전히 같은 request parameter를 사용하도록 보장해야 합니다.

구현 방식은 직접적입니다. rank 0이 먼저 pickle로 request object를 byte stream으로 serialize한 뒤, NCCL broadcast로 모든 rank에 보냅니다. 다른 rank가 몇 byte를 받아야 하는지 알 수 있도록 먼저 size tensor를 broadcast하고, 그다음 실제 데이터를 broadcast합니다.

모든 rank는 데이터를 받은 뒤 pickle로 deserialize해 같은 request object를 얻고, 함께 `model_manager.generate()`를 호출해 추론을 수행합니다. 실행이 끝나면 rank 0은 결과를 client에 반환하고, 다른 rank는 결과를 버리고 다음 request를 기다립니다.

여기에는 한 가지 세부사항이 있습니다. rank 0도 원래 request object를 직접 쓰지 않고 broadcast된 데이터를 다시 deserialize하게 했습니다. 이렇게 하면 모든 rank가 완전히 같은 object를 사용하도록 보장할 수 있고, pickle serialize/deserialize 차이로 생기는 문제를 피할 수 있습니다.

대략적인 흐름은 다음과 같습니다. Claude 4에게 그리게 했습니다.

```
Rank 0 (HTTP Server)              Rank 1, 2, ... (Workers)
      |                                  |
  FastAPI 시작                          worker loop 실행
      |                                  |
  HTTP request 수신                      broadcast 대기
      |                                  |
  request broadcast ----NCCL-------->   request 수신
      |                                  |
  pipe() 호출 <--------동기 추론------> pipe() 호출
      |              (all-reduce/all-to-all)
      |                                  |
  결과 반환                              결과 폐기
      |                                  |
  다음 request 대기                      다음 broadcast 대기
```


### 난수 생성의 함정

분산 추론에서는 모든 rank가 같은 random seed를 사용해야 합니다. 그렇지 않으면 여러 이상한 문제가 생깁니다. TP 모드에서는 이미지 전체가 흐릿해지고, CP 모드에서는 이미지 하반부가 깨진 내용이 됩니다.

이 문제의 근본 원인은 PyTorch의 CUDA RNG state가 per-device라는 데 있습니다. 즉 cuda:0과 cuda:1에서 같은 seed로 generator를 만들어도, 생성되는 random number sequence는 서로 다릅니다.

해결책은 CPU generator를 사용하는 것입니다. CPU RNG state는 global하므로 모든 rank가 같은 seed로 CPU generator를 만들면 완전히 같은 random number sequence가 생성됩니다. diffusers는 이 random number를 올바른 GPU로 자동 이동하므로 성능 문제를 걱정할 필요가 없습니다.

또한 사용자가 seed를 제공하지 않은 경우, 분산 모드에서는 고정 seed(예: 42)를 자동 생성합니다. 이렇게 하면 사용자가 seed 설정을 잊어도 이미지가 흐려지거나 깨지는 문제가 생기지 않습니다.

### Device 배치의 함정

또 하나 쉽게 밟는 함정은 device placement입니다. CP 모드에서는 all-to-all communication을 위해 모든 tensor가 GPU 위에 있어야 합니다. 하지만 pipeline의 일부 component(VAE, text encoder 등)가 아직 CPU에 있으면 "No backend type associated with device type cpu"라는 오류가 납니다.

해결책은 모든 모드에서 `pipe.to("cuda")`를 호출하는 것입니다. TP 모드에서는 transformer가 이미 여러 GPU에 나뉘어 있지만, 다른 component는 여전히 수동으로 이동해야 합니다. CP 모드에서는 모든 component가 GPU 위에 있어야 합니다.

## 0x5. 밟았던 함정 정리

구현 중에 꽤 많은 함정을 밟았습니다. 여기서는 주요한 몇 가지를 정리합니다.

첫 번째 함정은 TP/CP deadlock입니다. 처음에는 CP 모드에 broadcast mechanism이 필요 없다고 생각했습니다. 보기에는 CP의 forward pattern이 단일 GPU와 같았기 때문입니다. 그런데 서비스를 시작해 보니 rank 0은 추론에서 계속 멈춰 있고 rank 1은 잠들어 있었습니다. 나중에야 CP의 all-to-all communication도 모든 rank가 동시에 참여해야 한다는 것을 깨달았고, 따라서 broadcast로 request를 동기화해야 했습니다.

두 번째 함정은 이미지 흐림과 깨짐입니다. TP 모드에서 생성한 이미지는 전체적으로 흐렸고, CP 모드에서는 이미지 하반부가 깨졌습니다. 한참 debug한 뒤에야 random number generation 문제라는 것을 발견했습니다. 처음에는 GPU generator를 사용했는데, 서로 다른 GPU의 generator는 같은 seed를 써도 서로 다른 random number를 생성한다는 것을 알게 되었습니다. CPU generator로 바꾸니 문제가 해결됐습니다.

세 번째 함정은 rank 0이 broadcast request를 사용하지 않았던 점입니다. 처음에는 rank 0이 원래 request object를 직접 쓰고, 다른 rank만 broadcast로 받은 request를 쓰게 했습니다. 그 결과 CP 모드에서 이미지가 여전히 문제가 있었습니다. 나중에 pickle serialize/deserialize 차이가 원인일 수 있다고 보고 모든 rank가 broadcast request를 사용하도록 바꾸자 정상화되었습니다.

네 번째 함정은 pipeline마다 지원하는 parameter가 다르다는 점입니다. FLUX2 모델을 테스트할 때 `Flux2Pipeline.__call__() got an unexpected keyword argument 'negative_prompt'` 오류가 났습니다. 알고 보니 FLUX2 pipeline은 `negative_prompt` parameter를 지원하지 않는데 코드에서는 기본으로 이 parameter를 넘기고 있었습니다. 해결책은 `inspect.signature`로 pipeline의 `__call__` method가 어떤 parameter를 지원하는지 검사하고, 지원하는 parameter만 넘기는 것입니다. 이렇게 하면 서로 다른 pipeline과 호환할 수 있습니다.

## 0x6. 사용 방법

사용은 매우 간단하며 SGLang 경험과 거의 같습니다.

먼저 Cache-DiT를 clone하고, https://github.com/vipshop/cache-dit/pull/522 이 PR에 대응하는 branch로 전환합니다.

```bash
git clone git@github.com:BBuf/cache-dit.git
cd cache-dit
git checkout try_to_support_serving
pip install -e ".[serving]"
```

이 기능도 곧 mainline에 merge될 것입니다. 아마도요.

그다음 서비스를 시작합니다. 단일 GPU 모드:

```bash
cache-dit-serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --host 0.0.0.0 \
    --port 8000
```

TP 모드(2 GPU):

```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type tp \
    --host 0.0.0.0 \
    --port 8000
```

CP 모드(2 GPU):

```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type ulysses \
    --host 0.0.0.0 \
    --port 8000
```

시작 후에는 HTTP API로 호출할 수 있습니다. Python client 예시:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8000/generate"
data = {
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "seed": 42,
}

response = requests.post(url, json=data)
result = response.json()

# 이미지 decode
image_data = base64.b64decode(result["images"][0])
image = Image.open(BytesIO(image_data))
image.save("output.png")

print(f"Time cost: {result['time_cost']:.2f}s")
```

cURL과 jq도 사용할 수 있습니다.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }' | jq -r '.images[0]' | base64 -d > output.png
```

서비스를 시작한 뒤 `http://localhost:8000/docs`에 접속하면 전체 API 문서(Swagger UI)를 볼 수 있습니다.

전체 command line argument는 get_args 함수(https://github.com/vipshop/cache-dit/pull/522/files#diff-8d807db087ac7dc3923b8b6c6c4af29c87f8f29882f19ca5a9bf33f9b3d608b6R17)를 보면 됩니다. 여기서는 자주 쓰는 몇 가지만 나열합니다.

- `--model-path`: 모델 경로 또는 HuggingFace model ID
- `--cache`: cache acceleration 활성화
- `--parallel-type`: 병렬 유형(tp/ulysses/ring)
- `--compile`: torch.compile 활성화
- `--host`: server address(기본값 0.0.0.0)
- `--port`: server port(기본값 8000)

더 빠른 추론 속도를 원한다면 torch.compile을 켤 수 있습니다. 더 많은 optimization option은 cache-dit framework를 자세히 살펴보세요.

```bash
cache-dit-serve --model-path FLUX.1-dev --cache --compile
```

첫 추론에서는 compile이 진행되어 비교적 느리지만, 이후 추론은 훨씬 빨라집니다.

FLUX.1.dev 예시를 하나 붙입니다.

- server-side log


```markdown
cache-dit-serve --model-path /nas/bbuf/FLUX.1-dev/ --cache --compile
WARNING 12-03 06:50:00 [_attention_dispatch.py:303] Re-registered NATIVE attention backend to enable context parallelism with attn mask. You can disable this behavior by export env: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:50:00 [_attention_dispatch.py:416] Registered new attention backend: _SDPA_CUDNN, to enable context parallelism with attn mask. You can disable it by: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:50:01 [serve.py:107] Initializing model manager...
INFO 12-03 06:50:01 [model_manager.py:68] Initializing ModelManager: model_path=/nas/bbuf/FLUX.1-dev/, device=cuda
INFO 12-03 06:50:01 [serve.py:119] Loading model...
INFO 12-03 06:50:01 [model_manager.py:72] Loading model: /nas/bbuf/FLUX.1-dev/
Loading pipeline components...:   0%|                                                                       | 0/7 [00:00<?, ?it/s]`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.05it/s]
Loading pipeline components...:  29%|██████████████████                                             | 2/7 [00:02<00:05,  1.08s/it]You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.49s/it]
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████| 7/7 [00:08<00:00,  1.28s/it]
INFO 12-03 06:50:10 [model_manager.py:81] Enabling DBCache acceleration
INFO 12-03 06:50:10 [cache_adapter.py:49] FluxPipeline is officially supported by cache-dit. Use it's pre-defined BlockAdapter directly!
INFO 12-03 06:50:10 [functor_flux.py:61] Applied FluxPatchFunctor for FluxTransformer2DModel, Patch: False.
INFO 12-03 06:50:10 [block_adapters.py:147] Found transformer from diffusers: diffusers.models.transformers.transformer_flux enable check_forward_pattern by default.
INFO 12-03 06:50:10 [block_adapters.py:494] Match Block Forward Pattern: ['FluxSingleTransformerBlock', 'FluxTransformerBlock'], ForwardPattern.Pattern_1
INFO 12-03 06:50:10 [block_adapters.py:494] IN:('hidden_states', 'encoder_hidden_states'), OUT:('encoder_hidden_states', 'hidden_states'))
INFO 12-03 06:50:10 [cache_adapter.py:148] Use custom 'enable_separate_cfg' from cache context kwargs: True. Pipeline: FluxPipeline.
INFO 12-03 06:50:10 [cache_adapter.py:307] Collected Context Config: DBCache_F8B0_W8I1M0MC0_R0.08, Calibrator Config: None
INFO 12-03 06:50:10 [pattern_base.py:70] Match Blocks: CachedBlocks_Pattern_0_1_2, for transformer_blocks, cache_context: transformer_blocks_139774198646688, context_manager: FluxPipeline_139774199622368.
INFO 12-03 06:50:10 [model_manager.py:100] Moving pipeline to CUDA
INFO 12-03 06:52:33 [model_manager.py:108] Enabling torch.compile
INFO 12-03 06:52:33 [model_manager.py:112] Model loaded successfully
INFO 12-03 06:52:33 [serve.py:121] Model loaded successfully!
INFO 12-03 06:52:33 [serve.py:125] Starting server at http://0.0.0.0:8000
INFO 12-03 06:52:33 [serve.py:126] API docs at http://0.0.0.0:8000/docs
INFO:     Started server process [1928284]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO 12-03 06:52:45 [model_manager.py:117] Warming up for shape 1024x1024...
  0%|                                                                                                       | 0/4 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:1547: UserWarning: Dynamo detected a call to a `functools.lru_cache` wrapped function.Dynamo currently ignores `functools.lru_cache` and directly traces the wrapped function.`functools.lru_cache` wrapped functions that read outside state may not be traced soundly.
  warnings.warn(
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.51s/it]
INFO 12-03 06:52:53 [model_manager.py:127] Warmup completed for 1024x1024
INFO 12-03 06:52:53 [model_manager.py:137] Generating image: prompt='A beautiful sunset over the ocean...'
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.98it/s]
WARNING 12-03 06:53:00 [summary.py:275] Can't find Context Options for: FluxSingleTransformerBlock
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: FluxSingleTransformerBlock
WARNING 12-03 06:53:00 [summary.py:275] Can't find Context Options for: FluxTransformerBlock
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: FluxTransformerBlock

🤗Context Options: OptimizedModule

{'cache_config': DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold=0.08, max_accumulated_residual_diff_threshold=None, max_warmup_steps=8, warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=-1, enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=None, steps_computation_mask=None, steps_computation_policy='dynamic'), 'name': 'transformer_blocks_139774198646688'}
WARNING 12-03 06:53:00 [summary.py:284] Can't find Parallelism Config for: OptimizedModule

⚡️Cache Steps and Residual Diffs Statistics: OptimizedModule

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 6           | 0.043     | 0.06      | 0.089     | 0.135     | 0.217     | 0.043     | 0.285     |


⚡️CFG Cache Steps and Residual Diffs Statistics: OptimizedModule

| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |
|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 6               | 0.043     | 0.055     | 0.097     | 0.144     | 0.266     | 0.043     | 0.373     |

INFO 12-03 06:53:00 [model_manager.py:183] Image generation completed in 6.55s
INFO:     127.0.0.1:55144 - "POST /generate HTTP/1.1" 200 OK
```

- client-side log



```markdwon
 python -m cache_dit.serve.client \
    --prompt "A beautiful sunset over the ocean" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --output output.png
WARNING 12-03 06:48:43 [_attention_dispatch.py:303] Re-registered NATIVE attention backend to enable context parallelism with attn mask. You can disable this behavior by export env: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
INFO 12-03 06:48:43 [_attention_dispatch.py:416] Registered new attention backend: _SDPA_CUDNN, to enable context parallelism with attn mask. You can disable it by: export CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0.
Generating image: A beautiful sunset over the ocean
Image saved to output.png
Cache stats: {'cache_stats': [{'cache_options': "{'cache_config': DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=8, Bn_compute_blocks=0, residual_diff_threshold=0.08, max_accumulated_residual_diff_threshold=None, max_warmup_steps=8, warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=-1, enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True, num_inference_steps=None, steps_computation_mask=None, steps_computation_policy='dynamic'), 'name': 'transformer_blocks_140121591103456'}", 'cached_steps': [8, 10, 12, 14, 16, 18], 'parallelism_config': None}]}
Time cost: 9.22s
root@264a63f2d86e:/nas/bbuf/cache-dit# curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }' | jq -r '.images[0]' | base64 -d > output.png
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 1666k  100 1666k  100   125   173k     13  0:00:09  0:00:09 --:--:--  379k
```

- 결과

![](img/cache-dit-native-serving-implementation-8046430a/001.jpg)

## 0x7. 정리

Cache-DiT Serving의 구현 목표는 단순합니다. 사용자가 SGLang을 쓰듯 편하게 DiT 모델을 배포할 수 있게 하는 것입니다. SGLang 설계를 참고하고 DiT 모델의 특성에 맞게 단순화해, 가볍지만 기능은 비교적 완전한 추론 서비스를 만들려고 했습니다. 전체 구현의 핵심은 분산 추론의 동기화 mechanism입니다. NCCL broadcast로 request를 동기화해 복잡한 multi-process architecture를 피하면서도 TP와 CP 모드의 correctness를 보장했습니다. 구현 중 많은 함정을 밟았고, 특히 random number generation과 device placement 문제를 해결하는 데 시간이 들었습니다. 현재 이 기능은 정상 동작하며 단일 GPU, TP, CP 세 가지 모드를 지원합니다. 앞으로 더 많은 테스트를 하고 main branch merge를 추진할 예정입니다. server-side profiler 등도 이후 시간을 들여 마무리할 계획입니다.

많이 사용해 보시고 feedback을 주시면 좋겠습니다!


## 참고 자료

- Cache-DiT GitHub: https://github.com/vipshop/cache-dit
- SGLang GitHub: https://github.com/sgl-project/sglang
- Cache-DiT 학습 노트: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model-note/Cache-Dit%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md
