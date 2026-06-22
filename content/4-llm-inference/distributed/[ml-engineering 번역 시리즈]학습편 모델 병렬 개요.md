> 나의 강의 노트이며, 많은 관심 부탁드린다: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 본 문서의 출처: https://github.com/stas00/ml-engineering 。이 문서는 대규모 딥러닝 모델 학습에서의 병렬화 전략을 전면적으로 소개하며, 전통적인 데이터 병렬(DP), ZeRO로 최적화된 데이터 병렬, 텐서 병렬(TP), 파이프라인 병렬(PP) 및 시퀀스 병렬(SP) 등의 방법을 포함한다. 문서는 각 병렬 방식의 작동 원리와, 그것들이 대규모 모델 학습에서의 메모리 제약과 계산 효율 문제를 어떻게 해결하는지를 상세히 설명한다. 특히 ZeRO 병렬 전략에 대해서는 그 구현 원리, 네트워크 대역폭 요구 사항, 그리고 다른 병렬 방식과의 조합 사용을 포함하여 심도 있게 다룬다. 서로 다른 하드웨어 구성(단일 GPU, 단일 노드 다중 GPU, 다중 노드)과 모델 규모에 대해, 문서는 구체적인 병렬화 전략 선택 권고를 제공하여, 실제 응용에서 구체적인 시나리오에 따라 가장 적합한 병렬화 방안을 선택하고 이를 통해 대규모 모델의 효율적 학습을 실현하도록 돕는다.

# 모델 병렬

## 병렬 개요

현대 머신러닝에서는 다양한 병렬 방법이 다음 용도로 사용된다.

1. GPU 메모리 제약 극복. 예를 들어:
   - 초대형 모델 적재 - 예를 들어, t5-11b는 모델 파라미터만으로도 45GB가 필요하다
   - 초장 시퀀스 적재 - 예를 들어,
2. 학습 속도의 현저한 가속 - 1년이 걸리는 학습 시간을 몇 시간으로 단축

먼저 다양한 1차원 병렬 기법과 그 장단점을 심도 있게 논의한 뒤, 그것들을 어떻게 2차원 및 3차원 병렬로 조합하여 더 빠른 학습 속도를 달성하고 더 큰 모델을 지원하는지 살펴본다. 또한 다양한 강력한 대안 방법들도 소개한다.

주요 개념은 다른 어떤 프레임워크에도 적용될 가능성이 높지만, 본 글은 PyTorch 기반의 구현에 중점을 둔다.

가속기 메모리보다 큰 모델의 학습과 추론을 실현하는 데에는 두 가지 주요 방법이 있다.
1. 3D 병렬 - 네트워크 효율은 매우 높지만, 모델링 코드에 큰 간섭을 줄 수 있어 올바르게 작동시키기 위해 더 많은 작업이 필요하다
2. ZeRO 병렬 - 네트워크 효율은 그리 높지 않지만, 모델링 코드를 거의 변경할 필요가 없어 구현이 매우 쉽다.

## 확장성 개념

다음은 본 글에서 나중에 심도 있게 설명할 주요 개념에 대한 간략한 설명이다.

1. 데이터 병렬(DP) - 동일한 설정이 여러 번 복제되며, 각 복제본이 데이터의 일부를 처리한다. 처리 과정은 병렬로 실행되고, 모든 설정은 각 학습 스텝이 끝날 때 동기화된다.

2. 텐서 병렬(TP) - 전체 텐서가 단일 GPU에 상주하는 것이 아니라, 각 텐서가 여러 블록으로 나뉘고 각 분할 조각이 지정된 GPU에 상주한다. 처리 과정에서 각 분할 조각은 서로 다른 GPU에서 개별적으로 병렬 처리되고, 결과는 스텝이 끝날 때 동기화된다. 분할이 수평 차원에서 이루어지므로 이를 수평 병렬이라고 부를 수 있다.

3. 파이프라인 병렬(PP) - 모델이 여러 GPU 사이에서 수직(레이어 단위)으로 분할되어, 하나의 GPU에는 하나 또는 몇 개의 모델 레이어만 배치된다. 각 GPU는 파이프라인의 서로 다른 단계를 병렬로 처리하며, 소규모 배치 데이터를 처리한다.

4. 제로 중복 옵티마이저(ZeRO) - TP와 유사하게 텐서 분할을 수행하지만, 순전파 또는 역전파 계산 시 전체 텐서를 재구성하므로 모델을 수정할 필요가 없다. 또한 제한된 GPU 메모리를 보완하기 위해 다양한 오프로딩 기법을 지원한다. Sharded DDP는 다양한 다른 ZeRO 구현이 사용하는 기초 ZeRO 개념의 또 다른 이름이다.

5. 시퀀스 병렬 - 긴 입력 시퀀스의 학습에는 대량의 GPU 메모리가 필요하다. 이 기법은 단일 시퀀스의 처리를 여러 GPU에 분산시킨다.

6. 전문가 병렬 - 혼합 전문가(MoE)는 분할이 가능하여, 각 전문가가 전용 GPU(또는 여러 GPU)를 갖도록 할 수 있다.

이 논문의 서론 부분은 가장 흔한 병렬 기법에 대해 내가 찾은 설명 중 최고의 설명 중 하나일 것이다: Breadth-First Pipeline Parallelism(https://arxiv.org/abs/2211.05953)。

## 데이터 병렬

### DDP

2개의 GPU를 가진 대부분의 사용자는 이미 `DataParallel`(DP)와 `DistributedDataParallel`(DDP)이 가져다주는 학습 속도 향상을 누리고 있으며, 이 기능들은 사용하기 매우 쉽고 PyTorch의 내장 특성이다.

자세한 정보는 DistributedDataParallel(https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)을 참조하기 바란다

### ZeRO 데이터 병렬

ZeRO로 구동되는 데이터 병렬(ZeRO-DP)은 다음 블로그 게시물의 도표에 묘사되어 있다(https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

![](img/ml-engineering-translation-study-model-overview-a5e81849/001.png)

이 개념은 처음에는 이해하기 어려울 수 있지만 사실 매우 간단하다. 이것은 일반적인 `DataParallel`(DP)과 같으며, 다만 각 GPU가 완전한 모델 파라미터, 그래디언트, 옵티마이저 상태의 복사본을 저장하는 대신 그 중 한 조각만 저장한다는 점이 다르다. 그런 다음 실행 시점에, 어떤 레이어가 완전한 레이어 파라미터를 필요로 할 때 모든 GPU가 동기화하여 서로에게 부족한 부분을 제공한다 - 그것이 전부이다.

이 간단한 3레이어 모델을 생각해 보자. 각 레이어에는 3개의 파라미터가 있다:
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
레이어 La에는 가중치 a0、a1、a2가 있다.

만약 3개의 GPU가 있다면, sharded DDP(= Zero-DP)는 모델을 다음과 같이 3개의 GPU로 분할한다:

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

어떤 의미에서, 전형적인 DNN 그래프를 상상해 본다면, 이것은 텐서 병렬의 수평 분할과 같다. 수직 분할은 전체 레이어 그룹을 서로 다른 GPU에 배치하는 것이다. 그러나 이것은 단지 시작점일 뿐이다.

이제 각 GPU는 DP에서와 마찬가지로 일반적인 mini-batch를 받는다:
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

입력 데이터는 변하지 않는다 - 그것들은 일반 모델에 의해 처리될 것이라고 여겨진다.

먼저, 입력 데이터는 레이어 La로 들어간다.

GPU0에만 집중해 보자: x0은 순전파를 완료하기 위해 a0、a1、a2 파라미터를 필요로 하지만, GPU0에는 a0만 있다 - 그것은 GPU1로부터 a1을, GPU2로부터 a2를 받아 모델의 모든 부분을 함께 조합한다.

동시에, GPU1은 mini-batch x1을 받으며, a1만 가지고 있지만 a0과 a2 파라미터가 필요하므로 GPU0과 GPU2로부터 이를 받는다.

GPU2도 마찬가지로 입력 x2를 받는다. GPU0과 GPU1로부터 a0과 a1을 받고, 자신의 a2로 완전한 텐서를 재구성한다.

3개의 GPU 모두 완전한 텐서를 재구성하고 순전파 계산을 수행한다.

계산이 완료되면, 더 이상 필요하지 않은 데이터는 폐기된다 - 그것은 계산하는 동안에만 사용된다. 재구성은 프리페치를 통해 효율적으로 이루어진다.

전체 과정은 레이어 Lb, 그리고 Lc의 순전파에서 반복되고, 그런 다음 역전파에서는 Lc -> Lb -> La의 순서로 반복된다.

나에게 이것은 효율적인 팀 배낭 무게 분배 전략처럼 들린다:

1. A는 텐트를 운반하는 일을 담당한다
2. B는 스토브를 운반하는 일을 담당한다
3. C는 도끼를 운반하는 일을 담당한다

매일 밤 그들은 자신이 가진 것을 공유하고, 자신이 갖고 있지 않은 것을 다른 사람들로부터 받으며, 아침에는 자신에게 할당된 종류의 장비를 챙겨 다시 길을 떠난다. 이것이 sharded DDP / Zero DP이다.

각자가 자신의 텐트, 스토브, 도끼를 운반해야 하는 단순한 전략과 비교하면, 이 전략은 훨씬 더 효율적이다.

이 주제의 문헌을 읽다 보면 다음 동의어들을 마주칠 수 있다: Sharded、Partitioned。

ZeRO가 모델 가중치를 분할하는 방식을 자세히 관찰하면 - 그것은 나중에 논의할 텐서 병렬과 매우 유사해 보인다. 이는 ZeRO가 다음에 논의할 수직 모델 병렬과 달리, 각 레이어의 가중치에 대해 분할/분배를 수행하기 때문이다.

ZeRO-DP 단계 1+2+3의 구현:
- DeepSpeed(https://www.deepspeed.ai/tutorials/zero/)
- PyTorch(https://pytorch.org/docs/stable/fsdp.html) (처음에는 FairScale(https://github.com/facebookresearch/fairscale/)에서 구현되었고, 이후 PyTorch Core로 upstream되었다)

Deepspeed ZeRO 통합:
- HF Trainer 통합(https://huggingface.co/docs/transformers/main_classes/deepspeed)
- Accelerate(https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- PyTorch Lightning(https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- Determined.AI(https://docs.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html)

FSDP 통합:
- HF Trainer 통합(https://huggingface.co/docs/transformers/main/en/fsdp)
- Accelerate(https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- PyTorch Lightning(https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)

중요 논문:

Deepspeed와 ZeRO 전반:
- ZeRO: 조 단위 파라미터 모델 학습을 위한 메모리 최적화(https://arxiv.org/abs/1910.02054)
- ZeRO-Offload: 십억 규모 모델 학습의 대중화(https://arxiv.org/abs/2101.06840)
- ZeRO-Infinity: 극단적 규모의 딥러닝을 위한 GPU 메모리 벽 돌파(https://arxiv.org/abs/2104.07857)
- ZeRO++: 거대 모델 학습을 위한 극도로 효율적인 집합 통신(https://arxiv.org/abs/2306.10209)
- DeepSpeed Ulysses: 극도로 긴 시퀀스 Transformer 모델 학습을 실현하는 시스템 최적화(https://arxiv.org/abs/2309.14509)
- AMSP: 효율적인 LLM 학습을 위한 ZeRO의 통신 오버헤드 감소(https://arxiv.org/abs/2311.00257)

PyTorch:
- PyTorch FSDP: 완전 분할 데이터 병렬 확장의 경험(https://arxiv.org/abs/2304.11277)

주요 DeepSpeed ZeRO 자료:
- 프로젝트 github(https://github.com/microsoft/deepspeed)
- 사용 문서(https://www.deepspeed.ai/getting-started/)
- API 문서(https://deepspeed.readthedocs.io/en/latest/index.html)
- 블로그 게시물(https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 거대한 전역 배치 크기 문제 극복

만약 1024개의 가속기를 사용한다면, 각 가속기 위의 분할 조각은 매우 작아지고, 마이크로 배치 크기(MBS)를 위한 대량의 유휴 메모리가 생긴다. MBS=32로 설정할 수 있다고 가정하면 - 최종적으로 GBS=32k가 되는데 - 이는 아마도 당신이 원하는 것이 아닐 것이다.

따라서 텐서 병렬을 배포하거나(이는 구현하기 어렵다), 보통 더 간단한 방법인 시퀀스 병렬을 배포해야 한다(https://arxiv.org/abs/2305.14343)。나는 아직 실제로 시도해 보지 않았지만, 지금까지 내가 이해한 바로는:

- Deepspeed ZeRO는 Deepspeed-Ulysses를 사용한다(https://arxiv.org/abs/2309.14509)
- FSDP는 Paged Ring Attention(https://github.com/lucidrains/ring-attention-pytorch) 논문(https://arxiv.org/abs/2402.08268)을 사용한다

이는 텐서 병렬(https://arxiv.org/abs/2305.14343)만큼 효율적이지 않을 수 있다는 점에 유의하라 - 그러나 나는 아직 실제 추가 오버헤드를 알지 못한다.

#### 여러 복제본을 사용하는 ZeRO

기본적으로 ZeRO는 모든 GPU를 사용하여 단일 모델 복제본을 생성한다 - 즉, 모델이 모든 GPU에 분산된다. 이는 다음과 같은 다양한 제약을 초래한다:

1. 전역 배치 크기가 유연하지 않다 - 그것은 항상 `total_gpus*micro_batch_size`의 함수이며 - 대규모 클러스터에서는 거대한 전역 배치 크기를 초래할 수 있어 효율적 수렴에 불리할 수 있다. 물론 매우 작은 마이크로 배치 크기를 사용하여 전역 배치 크기를 제어할 수 있지만, 이는 각 GPU 위의 행렬을 더 작게 만들어 계산 효율을 떨어뜨린다
2. 더 빠른 노드 내 네트워크를 충분히 활용하지 못한다. 더 느린 노드 간 네트워크가 통신의 전체 속도를 규정하기 때문이다.

ZeRO++(https://arxiv.org/abs/2306.10209)는 계층적 가중치 분할(hpZ)을 도입하여 두 번째 제약을 해결한다. 이 방법에서는 전체 모델 가중치를 모든 GPU에 분산하는 대신, 각 모델 복제본이 단일 노드 내로 제한된다. 이는 메모리 사용량을 총 노드 수만큼 증가시키지만, 이제 분할된 구성 요소를 수집하는 2회의 `all_gather` 호출은 더 빠른 노드 내 연결에서 수행된다. 그래디언트를 집계하고 재분배하는 데 사용되는 `reduce_scatter`만 더 느린 노드 간 네트워크에서 수행된다.

첫 번째 제약은 전체 전역 배치 크기가 그대로 유지되므로 완전히 해결되지 않지만, 각 복제본이 더 효율적이고 추가적인 메모리 압박이 각 GPU 위에서 가능한 마이크로 배치 크기를 제한할 수 있으므로, 이는 전체적으로 시스템의 처리량을 향상시킬 것이다.

PyTorch FSDP는 이 기능을 shardingStrategy.HYBRID_SHARD(https://pytorch.org/docs/stable/fsdp.html)에서 구현한다

관련 논문:

- ZeRO++: 거대 모델 학습을 위한 극도로 효율적인 집합 통신(https://arxiv.org/abs/2306.10209)
- PyTorch FSDP: 완전 분할 데이터 병렬 확장의 경험(https://arxiv.org/abs/2304.11277)


#### ZeRO 변종

ZeRO 프로토콜에 대한 수정을 제안한 발표 논문들:

- MiCS: 퍼블릭 클라우드에서 거대 모델 학습의 준선형 확장(https://arxiv.org/abs/2205.00119) (2022)
- AMSP: 고급 모델 상태 분할을 통한 LLM 학습의 슈퍼 확장 실현(https://arxiv.org/abs/2311.00257) (2023)




## 파이프라인 병렬 방법

### 순진한 모델 병렬(수직)

순진한 모델 병렬(MP)은 모델 레이어 그룹을 여러 GPU에 분산하는 것을 의미한다. 그 메커니즘은 비교적 간단하다 - 대상 레이어를 `.to()`를 통해 대상 디바이스로 전환하고, 이제 데이터가 이 레이어들로 드나들 때 데이터를 해당 레이어와 동일한 디바이스로 전환하며, 나머지 부분은 그대로 유지한다.

우리는 이를 수직 MP라고 부르는데, 대부분의 모델이 어떻게 그려지는지 기억한다면, 레이어를 수직으로 분할하기 때문이다. 예를 들어, 아래 그림이 8레이어 모델을 보여준다고 하자:

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
우리는 이를 수직으로 2개 부분으로 분할하여, 제0-3레이어를 GPU0에, 제4-7레이어를 GPU1에 배치한다.

데이터가 제0레이어에서 제1레이어로, 제1레이어에서 제2레이어로, 그리고 제2레이어에서 제3레이어로 전달될 때, 이것은 일반 모델과 같다. 그러나 데이터가 제3레이어에서 제4레이어로 전달되어야 할 때, 그것은 GPU0에서 GPU1로 전송되어야 하며, 이는 통신 오버헤드를 유발한다. 참여하는 GPU가 동일한 계산 노드(예: 동일한 물리 머신)에 있다면 이 복사 속도는 상당히 빠르지만, GPU가 서로 다른 계산 노드(예: 여러 머신)에 있다면 통신 오버헤드가 현저히 증가할 수 있다.

그런 다음 제4레이어에서 제5레이어로 제6레이어로 제7레이어로의 실행은 일반 모델과 같으며, 제7레이어가 완료되면 우리는 보통 데이터를 제0레이어(레이블이 있는 곳)로 되돌려 보내거나, 레이블을 마지막 레이어로 보내야 한다. 이제 손실을 계산할 수 있고, 옵티마이저가 작동을 시작할 수 있다.

문제:
- 주요 결함(그리고 이것이 "순진한" MP라고 불리는 이유)은 어느 시점에서든 단 하나의 GPU만 작동하고 나머지 GPU는 모두 유휴 상태라는 것이다. 따라서 4개의 GPU를 사용한다면, 이는 단일 GPU의 메모리량을 4배로 늘리는 것과 거의 동등하며, 나머지 하드웨어는 무시된다. 또한 디바이스 간 데이터 복사의 오버헤드도 있다. 그래서 4개의 6GB 그래픽카드는 순진한 MP를 사용하여 1개의 24GB 그래픽카드와 동일한 크기의 모델을 수용할 수 있지만, 후자는 데이터 복사 오버헤드가 없으므로 학습을 더 빨리 완료할 것이다. 그러나 예를 들어, 만약 40GB의 그래픽카드를 가지고 있고 45GB 모델을 수용해야 한다면, 4개의 40GB 그래픽카드로 가능하다(다만 그래디언트와 옵티마이저 상태의 존재 때문에 간신히 가능하다)
- 공유 임베딩(Embedding 가중치)은 GPU 사이에서 왔다 갔다 복사해야 할 수 있다.

### 파이프라인 병렬

파이프라인 병렬(PP)은 순진한 MP와 거의 동일하지만, 입력 배치를 마이크로 배치로 분할하고 인위적으로 파이프라인을 생성함으로써 GPU 유휴 문제를 해결하며, 이를 통해 서로 다른 GPU가 동시에 계산 과정에 참여할 수 있다.

아래 GPipe 논문(https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)의 삽화는 순진한 MP(위 그림)와 PP(아래 그림)를 보여준다:

![](img/ml-engineering-translation-study-model-overview-a5e81849/002.png)


가운데 그림에서 PP가 어떻게 GPU 유휴의 사각지대를 줄이는지 쉽게 알 수 있다. 이 유휴 부분은 "버블"이라고 불린다.

그림의 두 부분 모두 pp=4의 병렬성을 보여준다. 즉, 파이프라인에 4개의 GPU가 참여한다. 따라서 4개의 파이프라인 단계의 순방향 경로 F0、F1、F2、F3이 있고, 그다음 역순의 역방향 경로 B3、B2、B1、B0이 있다.

PP는 튜닝할 새로운 하이퍼파라미터 `chunks`를 도입하는데, 이는 동일한 파이프라인 단계를 통해 순서대로 몇 개의 데이터 블록을 보내는지를 정의한다. 예를 들어, 그림에서 `chunks=4`를 볼 수 있다. GPU0은 블록 0、1、2、3에 대해 동일한 순방향 경로(F0,0、F0,1、F0,2、F0,3)를 실행한 다음, 다른 GPU들이 작업을 완료할 때까지 기다리며, 그것들의 작업이 완료되기 시작할 때에야 GPU0은 다시 작동하여 블록 3、2、1、0에 대해 역방향 경로(B0,3、B0,2、B0,1、B0,0)를 실행한다.

개념적으로 이것은 그래디언트 누적 스텝(GAS)과 동일한 개념임에 유의하라. PyTorch는 `chunks`를 사용하고, DeepSpeed는 동일한 하이퍼파라미터를 GAS라고 부른다.

분할로 인해, PP는 마이크로 배치(MBS)의 개념을 도입한다. DP는 전역 데이터 배치 크기를 작은 배치로 나누므로, DP 차수가 4라면 전역 배치 크기 1024는 각각 256인 4개의 작은 배치(1024/4)로 나뉜다. 만약 `chunks`(또는 GAS) 수가 32라면, 최종적으로 마이크로 배치 크기는 8(256/32)이 된다. 각 파이프라인 단계는 한 번에 하나의 마이크로 배치를 처리한다.

DP + PP 설정의 전역 배치 크기를 계산하려면, 우리는 다음을 수행한다: `mbs*chunks*dp_degree`(`8*32*4=1024`)。

이 그림으로 돌아가 보자.

`chunks=1`일 때, 최종적으로 순진한 MP가 되며, 이는 매우 비효율적이다. 그리고 `chunks` 값이 매우 클 때, 매우 작은 마이크로 배치 크기를 얻게 되는데, 이 또한 그다지 효율적이지 않을 수 있다. 따라서 GPU의 가장 효율적인 이용률을 실현할 수 있는 값을 찾기 위해 실험이 필요하다.

그림은 병렬화할 수 없는 "죽음"의 시간 버블(마지막 `forward` 단계가 `backward`가 pipeline을 완료하기를 기다려야 하기 때문)을 보여주지만, 최적의 `chunks` 값을 찾는 목적은 참여하는 모든 GPU의 높은 동시 이용률을 실현하는 것이며, 이는 버블의 크기를 최소화하는 것을 의미한다.

스케줄링의 선택은 효율적인 성능에 매우 중요하며, 발명 순서대로 나열한 가장 흔한 스케줄링 방식은 다음을 포함한다:

- 순차 Gpipe: 파이프라인 병렬을 사용한 거대 신경망의 효율적 학습(https://arxiv.org/abs/1811.06965)
- 교차 1F1B Pipedream: 빠르고 효율적인 파이프라인 병렬 DNN 학습(https://arxiv.org/abs/1806.03377)
- 순환적, 깊이 우선의 Megatron-LM을 사용한 GPU 클러스터에서의 효율적인 대규모 언어 모델 학습(https://arxiv.org/abs/2104.04473)
- 너비 우선 파이프라인 병렬(https://arxiv.org/abs/2211.05953)
- Llama 3 학습은 깊이 우선과 너비 우선 방법을 결합하여 최상의 성능을 얻었으며, 학습 과정에서 전역 배치 크기를 점진적으로 수정할 수 있도록 허용하였는데, 이는 파이프라인 병렬을 사용할 때 보통 매우 어려운 일이다. 모델 확장을 위한 병렬성에 관해서는 《Llama 3 모델 군집》(https://arxiv.org/abs/2407.21783) 제3.3.2절을 참조하라.

여기 교차 파이프라인의 예가 있다:

![parallelism-sagemaker-interleaved-pipeline](img/ml-engineering-translation-study-model-overview-a5e81849/003.png)

여기서, 버블(유휴 시간)은 역전파를 우선 처리함으로써 한층 더 최소화된다.

DeepSpeed、Varuna、SageMaker 등은 모두 이 방식을 사용하였다.

Varuna는 시뮬레이션을 사용하여 가장 효율적인 스케줄링 방식을 발견함으로써 스케줄링을 한층 더 개선한다.

DeepSeek v3(https://arxiv.org/abs/2412.19437) 는 더 효율적인 PP를 도입하였는데, DualPipe를 통해 버블 크기를 줄이고 더 나은 계산과 통신의 중첩을 실현하였다. 구체적인 세부 사항은 논문 제3.2.1절을 참조하라.

![출처: https://arxiv.org/abs/2412.19437](img/ml-engineering-translation-study-model-overview-a5e81849/004.png)

PP 솔루션에는 두 가지 종류가 있다 - 전통적인 Pipeline API와 더 현대적인 솔루션이며, 후자는 프로세스를 부분적으로 또는 완전히 자동화함으로써 최종 사용자가 사용하기 더 쉽게 만든다:

1. 전통적인 Pipeline API 솔루션:
- Megatron-LM
- DeepSpeed
- PyTorch

2. 현대적 솔루션:
- PiPPy
- Varuna
- Sagemaker
- DeepSeek

전통적인 Pipeline API 솔루션의 문제:
- 모델을 대폭 수정해야 한다. Pipeline은 모듈의 정상적인 흐름을 동일 모듈의 `nn.Sequential` 시퀀스로 재작성하도록 요구하므로, 모델의 설계를 변경해야 할 수 있다.
- 현재 Pipeline API는 매우 제한적이다. 만약 Pipeline의 첫 번째 단계에서 전달해야 할 Python 변수들이 한 무더기 있다면, 해결 방법을 찾아야 한다. 현재 pipeline 인터페이스는 단일 Tensor 또는 Tensor 튜플만을 유일한 입력과 출력으로 받는다. 이러한 텐서의 첫 번째 차원은 배치 크기여야 하는데, pipeline이 mini batch를 micro-batch로 나누기 때문이다. 가능한 개선 사항이 여기서 논의되고 있다(https://github.com/pytorch/pytorch/pull/50693)
- pipe 단계 수준의 조건 제어 흐름은 불가능하다 - 예를 들어, T5와 같은 인코더-디코더 모델은 조건부 인코더 단계를 처리하기 위해 특수한 우회 방법이 필요하다.
- 한 모델의 출력이 다른 모델의 입력이 되도록 각 레이어를 배치해야 한다.

나는 아직 Varuna와 SageMaker를 시도해 보지 않았지만, 그들의 논문 보고에 따르면, 그들은 위 문제 목록을 극복하였으며, 사용자의 모델에 대해 아주 작은 변경만 필요하다.

구현:
- Pytorch(https://pytorch.org/docs/stable/pipeline.html) (pytorch-1.8에서 초기 지원되었고, 1.9와 1.10에서 점진적으로 개선되었다). 일부 예시(https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- FairScale(https://fairscale.readthedocs.io/en/latest/tutorials/pipe.html)
- DeepSpeed(https://www.deepspeed.ai/tutorials/pipeline/)
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)에는 내부 구현이 있다 - API는 없다.
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972) - 이는 AWS에서만 사용할 수 있는 독점 솔루션이다.
- OSLO(https://github.com/eleutherAI/Oslo) - 이는 Hugging Face Transformers 기반으로 구현되었다.
- PiPPy(https://github.com/pytorch/pippy) - `torch.fx`를 통한 자동 PP
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)

### 관련 읽을거리

- 파이프라인 병렬: 모델 분할을 통한 분산 학습(https://siboehm.com/articles/22/pipeline-parallel-training)


## 텐서 병렬

텐서 병렬에서는, 각 GPU가 텐서의 한 조각만 처리하고, 완전한 텐서가 필요한 연산에서만 완전한 텐서를 집계한다.

이 절에서는, Megatron-LM(https://github.com/NVIDIA/Megatron-LM) 논문: GPU 클러스터에서의 효율적인 대규모 언어 모델 학습(https://arxiv.org/abs/2104.04473)에서 나온 개념과 도표를 사용한다.

어떤 transformer의 주요 구성 블록이든 완전 연결 레이어 `nn.Linear`이며, 그 뒤에 비선형 활성화 함수 `GeLU`가 따른다.

Megatron 논문의 표기법에 따라, 우리는 내적 부분을 `Y = GeLU(XA)`로 쓸 수 있는데, 여기서 `X`와 `Y`는 입력과 출력 벡터이고, `A`는 가중치 행렬이다.

만약 계산을 행렬 형태로 본다면, 행렬 곱셈이 여러 GPU 사이에서 어떻게 분할될 수 있는지 쉽게 알 수 있다:

![Parallel GEMM](img/ml-engineering-translation-study-model-overview-a5e81849/005.png)


만약 우리가 가중치 행렬 `A`를 열 단위로 `N`개의 GPU에 분할하고, 행렬 곱셈 `XA_1`부터 `XA_n`까지를 병렬로 실행한다면, 우리는 `N`개의 출력 벡터 `Y_1, Y_2, ..., Y_n`을 얻게 되며, 이것들은 독립적으로 `GeLU`에 입력될 수 있다:

![independent GeLU](img/ml-engineering-translation-study-model-overview-a5e81849/006.png)

이 원리를 사용하여, 우리는 임의 깊이의 MLP를 갱신할 수 있으며, 마지막에 분할 조각으로부터 출력 벡터를 재구성하기 전까지는 GPU 사이에서 어떠한 동기화도 필요하지 않다. Megatron-LM 논문 저자들은 이를 위해 유용한 도식을 제공하였다:

![parallel shard processing](img/ml-engineering-translation-study-model-overview-a5e81849/007.png)

multi-head attention 레이어 자체가 여러 개의 독립적인 head를 가지므로, multi-head attention 레이어를 병렬화하는 것은 한층 더 간단하다!

![parallel self-attention](img/ml-engineering-translation-study-model-overview-a5e81849/008.png)

중요 안내: TP는 매우 빠른 네트워크를 필요로 하며, 노드 내 네트워크가 보통 노드 간 네트워크보다 훨씬 빠르므로, 노드를 가로지르는 TP는 권장되지 않는다. 실제로, 만약 한 노드에 4개의 GPU가 있다면, TP의 최고 차수는 4이다. 만약 8 차수의 TP가 필요하다면, 적어도 8개의 GPU를 가진 노드를 사용해야 한다.

중요 안내: TP 차수는 노드를 가로질러서는 안 된다. 예를 들어, 만약 노드에 8개의 gpu가 있다면, TP 차수는 8을 초과해서는 안 된다.

TP는 다른 병렬화 방법과 결합하여 사용할 수 있다.

다른 이름:
- DeepSpeed는 이를 텐서 분할이라고 부른다(https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)

구현:
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)에는 내부 구현이 있는데, 그것이 모델에 매우 특화되어 있기 때문이다
- PyTorch(https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- SageMaker(https://arxiv.org/abs/2111.05972) - 이는 AWS에서만 사용할 수 있는 독점 솔루션이다
- OSLO(https://github.com/eleutherAI/Oslo)는 Transformers 기반으로 텐서 병렬을 구현하였다
- nanotron(https://github.com/huggingface/nanotron)
- parallelformers(https://github.com/tunib-ai/parallelformers)(현재 추론만 지원)
- torchtian(https://github.com/pytorch/torchtitan)

### 비동기 텐서 병렬

TP의 한 가지 결함은 그 통신을 계산과 중첩하기 어렵다는 것이다. PyTorch는 이 문제를 극복하기 위해 비동기 TP(https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)를 사용할 것을 제안하는데, 이는 `all-gather + matmul`의 의존 시퀀스를 일련의 cudaMemcpyAsync 호출과 더 작은 부분 matmul로 분해하며 - 그리고 `torch.compile`을 사용하면 이를 자동으로 처리할 수 있다!

- Megatron-LM도 `--tp-comm-overlap`을 통해 이 기능을 구현하였다.

### 관련 읽을거리
- 텐서 병렬과 시퀀스 병렬: 상세 분석(https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## TP+SP

TP는 동일한 프로세스 그룹에서 SP와 결합하여 사용함으로써 통신 비용을 최소화할 수 있으며, 구체적인 설명은 《대규모 Transformer 모델에서 활성화 재계산 감소》(https://arxiv.org/abs/2205.05198)를 참조하라. 예를 들어, LLM에서 TP는 임베딩, attention, 선형 레이어에 사용되고, dropout과 레이어 정규화에 도달하면 SP로 전환한다.

## DP+PP

아래 DeepSpeed pipeline 튜토리얼(https://www.deepspeed.ai/tutorials/pipeline/)의 도표는 DP를 PP와 어떻게 결합하여 사용하는지를 보여준다.

![dp-pp-2d](img/ml-engineering-translation-study-model-overview-a5e81849/009.png)

여기서 주의해야 할 점은, DP rank 0은 GPU2를 볼 수 없고, DP rank 1은 GPU3을 볼 수 없다는 것이다. DP에게는 GPU 0과 1만 있으며, 마치 2개의 GPU만 있는 것처럼 그것들에게 데이터를 입력한다. GPU0은 PP를 사용하여 "몰래" 일부 부하를 GPU2로 떠넘긴다. GPU1도 GPU3을 이용하여 같은 일을 한다.

각 차원마다 적어도 2개의 GPU가 필요하므로, 여기서는 적어도 4개의 GPU가 필요하다.

구현:
- DeepSpeed(https://github.com/microsoft/DeepSpeed)
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972)
- OSLO(https://github.com/eleutherAI/Oslo)
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)


## DP+PP+TP

더 효율적인 학습을 얻기 위해, 3D 병렬을 사용할 수 있는데, 즉 PP를 TP 및 DP와 결합하여 사용하는 것이다. 이는 아래 그림에서 알 수 있다.

![dp-pp-tp-3d](img/ml-engineering-translation-study-model-overview-a5e81849/010.png)

이 그림은 블로그 게시물 《3D 병렬: 조 단위 파라미터 모델로 확장》(https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)에서 나온 것이며, 이 또한 읽을 가치가 있는 글이다.

각 차원마다 적어도 2개의 GPU가 필요하므로, 여기서는 적어도 8개의 GPU가 필요하다.

구현:
- DeepSpeed(https://github.com/microsoft/DeepSpeed) - DeepSpeed also includes an even more efficient DP, which they call ZeRO-DP.
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Varuna(https://github.com/microsoft/varuna)
- SageMaker(https://arxiv.org/abs/2111.05972)
- OSLO(https://github.com/eleutherAI/Oslo)
- nanotron(https://github.com/huggingface/nanotron)
- torchtitan(https://github.com/pytorch/torchtitan)

## ZeRO DP+PP+TP

DeepSpeed의 주요 특성 중 하나는 ZeRO이며, 이는 DP의 한 확장이다. ZeRO 데이터 병렬에서 이미 논의하였다. 보통 그것은 독립적인 기능이며, PP나 TP가 필요하지 않다. 그러나 그것은 PP 및 TP와 결합하여 사용할 수 있다.

ZeRO-DP가 PP(및 선택적으로 TP)와 결합될 때, 보통 ZeRO stage 1(옵티마이저 분할)만 활성화한다.

이론적으로 ZeRO stage 2(그래디언트 분할)를 파이프라인 병렬과 결합하여 사용할 수 있지만, 이는 성능에 좋지 않은 영향을 미친다. 각 마이크로 배치는 분할 전에 그래디언트를 집계하기 위한 추가적인 reduce-scatter 집합이 필요한데, 이는 잠재적으로 상당한 통신 오버헤드를 증가시킨다. 파이프라인 병렬의 본질로 인해 작은 마이크로 배치를 사용하며, 중점은 산술 강도(마이크로 배치 크기)와 파이프라인 버블 최소화(마이크로 배치 수)의 균형을 맞추려는 데 있다. 따라서 이러한 통신 비용은 해를 끼친다.

또한, PP 때문에 레이어 수가 이미 정상보다 적으므로, 메모리 절약이 크지 않을 것이다. PP는 이미 그래디언트 크기를 "1/PP"만큼 줄였으므로, 그 위에서의 그래디언트 분할 절약은 순수 DP에 비해 그다지 현저하지 않다.

같은 이유로, ZeRO stage 3도 좋은 선택이 아니다 - 더 많은 노드 간 통신이 필요하다.

우리가 ZeRO를 가지고 있으므로, 또 다른 이점은 ZeRO-Offload이다. 이것이 stage 1이므로, 옵티마이저 상태를 CPU로 오프로드할 수 있다.

구현:
- Megatron-DeepSpeed(https://github.com/microsoft/Megatron-DeepSpeed)와 BigScience에서 나온 Megatron-Deepspeed(https://github.com/bigscience-workshop/Megatron-DeepSpeed)이며, 후자는 전자의 분기이다.
- OSLO(https://github.com/eleutherAI/Oslo)

중요 논문:

- DeepSpeed와 Megatron을 사용하여 대규모 생성 언어 모델인 Megatron-Turing NLG 530B 학습(
https://arxiv.org/abs/2201.11990)



## 시퀀스 병렬

DNA 시퀀싱과 같은 머신러닝 작업은 매우 긴 시퀀스 길이(예: 256K)를 학습해야 할 수 있으며, 일반적인 대규모 언어 모델조차 10k 및 그 이상의 시퀀스를 학습해야 할 수 있다.

Self-Attention은 Transformer의 핵심 구성 요소로서, 그 메모리 요구량이 시퀀스 길이와 이차 관계를 가지므로, 시퀀스 길이가 일정 길이에 도달하면 batch size가 1이라도 단일 GPU에 적재되지 못할 수 있어, 시퀀스 차원을 따라 추가적인 분할이 필요하다. 일단 분할이 완료되면, 시퀀스는 임의의 길이가 될 수 있다.

이 병렬 유형은 본 문서에서 설명한 다른 병렬화 유형들과 직교하므로, 다른 어떤 유형과도 조합하여 4D, ZeRO-DP+SP 등의 조합을 형성할 수 있다.

### Deepspeed-Ulysses SP

논문: DeepSpeed Ulysses: 초장 시퀀스 Transformer 모델 학습을 지원하는 시스템 최적화(https://arxiv.org/abs/2309.14509)

이 구현에서는, 2개의 요소가 분할된다:
1. multi-head attention 가중치가 참여하는 GPU 사이에서 분할되어, 각 GPU가 몇 개의 sub-head만 갖도록 한다. 이는 모델 생성/로드 시 이루어진다. 이는 텐서 병렬과 약간 유사하다.
2. 학습 중에, 각 입력 시퀀스가 블록으로 나뉘고, 각 블록이 GPU 중 하나로 보내진다. 이는 ZeRO-3 분할을 연상시키지만, 다만 여기서 분할되는 것은 가중치가 아니라 입력이다.

계산 과정에서, 각 시퀀스 블록은 QKV로 투영된 다음, 각 디바이스에서 완전한 시퀀스의 QKV로 수집되고, 각 디바이스는 자신이 가진 sub-head만 계산한 다음, 다시 MLP 블록의 완전한 attention 출력으로 수집된다.

![deepspeed-ulysses sp](img/ml-engineering-translation-study-model-overview-a5e81849/011.png)

소스 코드(https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses)

그림에서:
1. 입력 시퀀스 N이 P개의 가용 디바이스에 분할된다.
2. 입력 시퀀스의 각 국부 N/P 분할이 query(Q)、key(K)、value(V) 임베딩으로 투영된다.
3. 다음으로, 계산에 참여하는 디바이스 사이의 고도로 최적화된 all-to-all 집합 통신을 통해, 국부 QKV 임베딩이 전역 QKV로 수집된다.
4. 그런 다음 각 attention head에 대해 attention 계산을 수행한다:

![](img/ml-engineering-translation-study-model-overview-a5e81849/012.png)


5. 마지막으로, 또 다른 all-to-all 집합이 attention 계산의 출력 컨텍스트 텐서를 시퀀스(N/P) 병렬로 변환하여, transformer 레이어 블록의 나머지 모듈의 후속 연산(MLP MatMul, 레이어 정규화 등)에 사용하도록 제공한다.

예시: 시퀀스 길이=8K, head 수=128, 단일 노드 GPU 수=8인 경우를 고려해 보자

1. 각 GPU는 원본 시퀀스의 1K 길이 블록을 받는다(`8K/8`)
2. 각 GPU는 16개의 sub-head를 할당받는다(`128/8`) 
3. a. gpu0에서, `forward` 전에, 원본 시퀀스가 8K개의 token으로 다시 수집된다
   b. 앞의 16개 sub-head에 대해 attention 계산을 수행한다
나머지 7개 GPU는 동일한 로직을 실행하며, 각 GPU는 자신의 16개 sub-head에 대해 8k attention을 계산한다

효율적인 통신의 구체적인 세부 사항은 여기서 읽을 수 있다(https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-ulysses#significant-communication-volume-reduction)。

DeepSpeed-Ulysses는 메시지 크기 또는 시퀀스 길이에 비례하여 GPU 수를 늘림으로써 통신량의 일관성을 유지한다.

### Colossal-AI의 시퀀스 병렬

논문: 시스템 관점에서 본 시퀀스 병렬: 장 시퀀스 학습(https://arxiv.org/abs/2105.13120)

Colossal-AI의 시퀀스 병렬 구현은 ring self-attention 메커니즘을 사용하는데, 이는 query 투영이 국부적이고 key와 value 투영이 ring 방식으로 전송되어 전역 attention을 계산하는 ring 통신 집합으로, 통신 복잡도가 메시지 크기 M과 선형 관계를 가지도록 한다.

### Megatron-LM의 시퀀스 병렬

논문: 대규모 Transformer 모델에서 활성화 재계산 감소(https://arxiv.org/abs/2205.05198)

Megatron-LM의 시퀀스 병렬은 그 텐서 병렬과 긴밀하게 통합되어 있다. Megatron-LM은 시퀀스 차원을 따라 시퀀스를 분할하고, allgather와 reduce scatter 집합 연산을 적용하여 attention 계산을 위해 QKV 투영을 집계한다. 계산 디바이스 수와 무관하게, 그 통신량은 메시지 크기(M)와 선형으로 증가한다.

### Ring Attention with Blockwise Transformers

논문: Ring Attention with Blockwise Transformers for Near-Infinite Context(https://arxiv.org/abs/2310.01889)

1. 텐서는 항상 시퀀스 차원을 따라 분할된다: 형상은 (`seq_len // N, d_model`)
2. attention 레이어에서, 각 GPU는 먼저 자신이 가진 분할 조각을 사용하여 계산할 수 있는 attention 점수 부분을 계산한다.
3. 동시에, 다른 시퀀스 블록에서 온 key와 value가 주변으로 전송된다.
4. 다른 블록의 key/value가 가용해지면, 각 GPU는 시퀀스의 이 새로운 조각에서 온 key/value 텐서를 사용하여 attention 계산을 계속한다
5. attention 계산이 완료될 때까지 계속한다.

시퀀스 병렬 구현:
- Megatron-LM(https://github.com/NVIDIA/Megatron-LM)
- Deepspeed(https://github.com/microsoft/DeepSpeed)
- Colossal-AI(https://colossalai.org/)
- torchtitan(https://github.com/pytorch/torchtitan)

PyTorch도 이 기능을 개발 중이며, 이를 컨텍스트 병렬(CP)이라고 부른다.

### DistFlashAttn

DISTFLASHATTN: 장 컨텍스트 LLM 학습을 위한 분산 메모리 효율적 attention(https://arxiv.org/abs/2310.03294)는 시퀀스 병렬을 수행할 때 워커 노드 사이에서 token별 KVQ 계산 부하의 균형을 맞추기 때문에 Ring Attention보다 몇 배 빠르다고 보고되었다.

![](img/ml-engineering-translation-study-model-overview-a5e81849/013.png)

### Related reading

- Tensor Parallelism and Sequence Parallelism: Detailed Analysis(https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## 전문가 병렬

혼합 전문가 모델(MoE)을 사용할 때(특히 추론 과정에서), 각 전문가에게 자신의 가속기를 할당할 수 있다(하나로 부족하면 여러 개를 할당할 수 있다). 이는 병렬화에 또 다른 차원을 더하며, 모든 전문가에 도달할 수 있는 대규모 배치 데이터를 현저히 가속할 수 있다.

자세한 설명은 다음을 참조하라:
- DeepSpeed-MoE: 차세대 AI 규모를 지원하기 위한 혼합 전문가 모델 추론과 학습의 발전(https://arxiv.org/abs/2201.05596)
- 혼합 전문가 모델 해설(https://huggingface.co/blog/moe#parallelism)

## FlexFlow

FlexFlow(https://github.com/flexflow/FlexFlow)는 약간 다른 방식으로 병렬화 문제를 해결한다.

논문: "심층 신경망의 데이터와 모델 병렬을 넘어서" 저자: Zhihao Jia, Matei Zaharia, Alex Aiken(https://arxiv.org/abs/1807.05358)

그것은 샘플-연산자-속성-파라미터의 이 4가지 차원에서 병렬화를 수행한다.

1. 샘플 = 데이터 병렬(샘플 차원 병렬)
2. 연산자 = 단일 연산을 여러 하위 연산으로 병렬화
3. 속성 = 데이터 병렬(길이 차원 병렬)
4. 파라미터 = 모델 병렬(차원 - 수평 또는 수직 - 을 고려하지 않음)

예시:
* 샘플

10개의 배치가 있고, 각 시퀀스 길이가 512라고 가정하자. 만약 우리가 샘플 차원에서 그것들을 2개의 디바이스로 병렬화하면, 10 x 512가 5 x 2 x 512가 된다.

* 연산자

만약 레이어 정규화를 수행한다면, 우리는 먼저 std를 계산한 다음 mean을 계산하고, 그런 다음 데이터를 정규화할 수 있다. 연산자 병렬은 std와 mean을 병렬로 계산할 수 있게 한다. 그래서 만약 우리가 연산자 차원에서 그것들을 2개의 디바이스(cuda:0, cuda:1)로 병렬화하면, 먼저 입력 데이터를 두 디바이스에 복사하고, cuda:0이 std를 계산하며, cuda:1이 동시에 mean을 계산한다.

* 속성

우리는 10개의 배치가 있고, 각 길이가 512이다. 만약 우리가 속성 차원에서 그것들을 2개의 디바이스로 병렬화하면, 10 x 512가 10 x 2 x 256이 된다.

* 파라미터

이는 텐서 모델 병렬 또는 단순한 레이어 단위 모델 병렬과 유사하다.

![](img/ml-engineering-translation-study-model-overview-a5e81849/014.png)

이 프레임워크의 중요성은 (1) GPU/TPU/CPU, (2) RAM/DRAM, (3) 빠른 내부 연결/느린 외부 연결 등의 자원을 처리할 수 있고, 이 모든 자원을 자동으로 최적화하여 어디에서 어떤 병렬화를 사용할지 알고리즘적으로 결정할 수 있다는 데 있다.

매우 중요한 측면 하나는, FlexFlow가 정적이고 고정된 작업 부하를 가진 DNN 병렬화를 최적화하는 데 특화되어 있다는 것이다. 동적 행동을 가진 모델은 서로 다른 반복에서 서로 다른 병렬화 전략을 선호할 수 있기 때문이다.

그래서 이 약속은 매우 매력적이다 - 그것은 선택한 클러스터에서 30분간 시뮬레이션을 실행하고, 이 특정 환경을 활용할 최적의 전략을 제안한다. 만약 어떤 부분을 추가/삭제/교체하면, 그것은 다시 실행되어 계획을 재최적화한다. 그런 다음 학습을 시작할 수 있다. 서로 다른 설정은 각자의 맞춤형 최적화를 갖게 된다.

### 병렬 네트워크 집합

노드 내와 노드 간의 속도가 보통 10배의 차이가 존재하므로, 노드 내와 노드 간 상호작용을 수행할 때 서로 다른 병렬화 기법을 선택하는 것이 매우 중요하다. 예를 들어, TP는 그 거대한 동기화 요구로 인해 항상 노드 내부에 유지되어야 한다. 또한, 최신 AMD MI3** 시리즈와 같은 일부 가속기는 GPU 간 연결 속도가 매우 느려, 이 또한 병렬화의 최적 성능에 영향을 미친다.

여기 유용한 팁이 있다: all-reduce 집합은 두 개의 독립적인 단계로 분해될 수 있다: reduce-scatter와 all-gather。

![출처: https://engineering.fb.com/2021/07/15/open-source/fsdp/attachment/fsdp-graph-2a/](img/ml-engineering-translation-study-model-overview-a5e81849/015.png)

다음은 서로 다른 병렬화 전략이 사용하는 집합 연산의 상세 설명이다:

- DDP: 그래디언트를 위한 1회 all-reduce - 이상적으로는 계산과 중첩 - 총 통신량: 2배 모델 파라미터
- ZeRO-DP ZeRO-1/ZeRO-2: 옵티마이저 상태를 위한 1회 all-gather에 그래디언트를 위한 1회 reduce-scatter 추가 - 총 통신량: 2배 모델 파라미터
- ZeRO-DP ZeRO-3: 가중치를 위한 2회 all-gather(순전파 전과 역전파 전)에 그래디언트를 위한 1회 reduce-scatter 추가 - 총 통신량: 3배 모델 파라미터(DDP 및 ZeRO-1/ZeRO-2보다 1.5배 많음)
- TP: 2회 all-gather와 2회 reduce-scatter
- PP: 2회 송신 + 2회 수신 - 안정 단계에서 계산과 중첩
- SP: 구현에 따라 다름: 은닉층 크기 h, 시퀀스 길이 N, 병렬 차수 P에 대해
    - Megatron-LM: 2회 all-gather와 2회 reduce-scatter, 통신량은 각 Transformer Layer당 `4*N*h`(논문 제3.2절 참조 https://arxiv.org/abs/2309.14509)
    - DeepSpeed Ulysses: 2회 all-to-all 통신, 통신량은 각 Transformer Layer당 `4*N*h/P`(논문 제3.2절 참조 https://arxiv.org/abs/2309.14509)

서로 다른 구현이 서로 다른 통신 패턴을 사용할 수 있음을 발견할 수 있다.

## ZeRO를 사용할 때의 노드 간 속도 요구 사항

ZeRO 확장성 프로토콜은, Deepspeed ZeRO든 PyTorch FSDP든, TP+PP+DP 솔루션보다 더 많은 노드 간 트래픽을 필요로 한다. 때때로 그것은 더 빠른 노드 내 연결을 활용하지 못하므로, 만약 노드 간 네트워크 속도가 느리다면, 비싼 GPU가 통신으로 인해 심각하게 제약될 수 있다.

ZeRO 프로토콜은 통신을 계산과 부분적으로 중첩하므로, 이상적으로는 `통신 시간 <= 계산 시간`에 도달하기를 원한다. 중첩이 완벽하지 않으므로 항상 어느 정도 네트워크 병목이 있겠지만, 우리는 `통신 시간`이 `계산 시간`보다 너무 크지 않도록 보장하고자 한다.

ZeRO-3에서는, `forward`에서 가중치에 대해 `all_gather`를 수행하고, 그런 다음 `backward`에서 가중치에 대해 `all_gather`를 수행하며, 마지막으로 backward에서 그래디언트에 대해 `reduce_scatter`를 수행한다. 총 3회의 전역 집합 호출이 있으며, 매번 보내는 모델 크기에 파라미터당 사용되는 바이트 수를 곱한다. 예를 들어, 10B 파라미터의 bf16 모델은 ZeRO-3 하에서 `10*2*3` = 60GB의 데이터를 보내야 한다.

이에 비해, DistributedDataParallel(DDP)은 단일 `all_reduce` 호출을 사용하지만, 2배의 데이터 전송이 필요하므로, 10B 파라미터의 bf16 모델은 DDP 하에서 `10*2*2` = 40GB의 데이터를 보내야 한다.

ZeRO-1은 옵티마이저 상태만 분할하며, DDP와 마찬가지로 40GB 데이터를 전송해야 한다(1회 `all_gather`와 1회 `reduce_scatter`).

다음은 통신과 계산의 시간(초)을 계산하는 방법이다:

- `통신 시간 = 전송 횟수 * 바이트 수 * 모델 크기(B) / 노드 간 처리량(GBps)`
- `계산 시간 = 계산 횟수 * 바이트 수 * 모델 크기(B) * 시퀀스 길이 * 전역 배치 크기 / (총 GPU 수 * 1e3 * 통신이 없을 때의 TFLOPS)`

계산 시간 공식은 대략적인 추정이며, Transformer 블록 기반의 어떤 모델에도 적용된다. 그것은 작은 계산은 모두 무시하고, 큰 `matmul`만 포함한다.

IDEFICS-80B 학습의 데이터 포인트를 예로 들어 실험해 보자.

우리가 340GBs EFA로 IDEFICS-80B를 학습할 때, A100에서 Deepspeed ZeRO-3를 사용하면 90TFLOPs만 얻을 수 있었고, Megatron의 TP+PP+DP는 150+TFLOPs를 얻을 수 있었다. 그리고 우리가 하나의 언어 모델과 하나의 비전 모델을 기반으로 새로운 모델을 구축하고 있었으므로, 모델의 상당 부분이 동결되어 있었다. 그래서 우리의 승수는 3보다 작았다. 한편, 우리는 메모리를 절약하기 위해 활성화 재계산을 사용했으므로, 이는 모든 모델 가중치를 추가로 전송해야 하며, 또한 nccl이 적절한 반정밀도 reduction을 지원하지 않으므로 우리는 그래디언트 reduction에 fp32를 사용했고, 그래서 실제로 우리의 승수는 3이 아니라 4.5였다.

IDEFICS-80B 학습에 사용한 값:
- `model_size_in_B` = `80`
- `n_bytes` = `2` (bf16은 2바이트)
- `n_transmissions` = `3` (ZeRO-3/FSDP의 경우 1회 reduce_scatter + 2회 all_gather(fwd + bwd)), ZeRO-1은 2(1회 reduce_scatter + 1회 all_gather)
- 또한, IDEFICS-80B의 경우, 우리는 NCCL 누적 손실을 최소화하기 위해 fp32에서 그래디언트를 reduce하기로 결정했으므로, 실제로 우리는 추가 2바이트를 위해 `n_transmissions*n_bytes=3*2+2=4*2`를 가지지만, 모델의 절반이 동결되어 있어 그래디언트의 약 절반만 보내지므로, 우리는 여전히 3의 승수를 가진다.
- `n_passes` = `4` (활성화 재계산 사용), 또는 `3` (사용하지 않음). 모델은 `forward`에서 1회의 계산만 필요하고, `backward`에서 2회가 필요하다(그래디언트가 두 번 계산되기 때문 - 한 번은 입력에 대해, 한 번은 가중치에 대해). 활성화 재계산을 사용할 때는 `forward`를 한 번 더 해야 한다.
- `total_gpus` = `512`
- `global_batch_size` = `3584`
- `seqlen` = `1024`
- `inter-node-throughput_in_GBps` = 42.5 (340Gbps) (AWS EFA v1)
- `tflops_wo_comms`는 통신 오버헤드가 없을 때의 tflops이다. 그것은 도달할 수 없으므로 이론적 피크는 아니지만, A100@BF16의 경우 75%일 수 있으므로 - `312*0.75=234` TFLOPS이다

우리는 `all_reduce_bench.py`(https://github.com/BBuf/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)를 사용하여 340Gbps의 노드 간 네트워크 처리량을 도출하였으며, 그것은 기본적으로 4GB의 페이로드를 사용한다. IDEFICS-80B의 경우, 우리는 80레이어를 가지므로, 각 레이어는 약 1B 파라미터를 가진다. 이는 각 레이어가 bf16 텐서에 대해 2GB 데이터를, fp32 텐서에 대해 4GB 데이터를 보낸다는 것을 의미하며, 이는 네트워크 벤치마크와 일치한다. 만약 당신의 레이어 크기가 훨씬 작다면, 나는 해당 크기에 맞게 벤치마크를 조정할 것을 권한다. 예를 들어, 만약 당신의 레이어 크기가 100M 파라미터에 불과하다면, bf16 텐서의 페이로드는 0.2GB가 될 것이다. 이는 한 자릿수 더 작으므로, 네트워크가 당신에게 더 낮은 대역폭을 줄 수 있으며, 당신은 계산에서 이 값을 사용해야 한다.

참고: 만약 당신의 모델이 부분적으로 동결되어 있다면, 그래디언트를 동기화할 때 더 적은 데이터가 보내질 것이다. IDEFICS에서는, 우리가 모델의 절반 이상이 동결되어 있어, 그래디언트가 reduce될 때 우리는 약 절반의 트래픽만 가졌다.

이는 우리에게 다음을 준다:

- 통신 = `3 * 2 * 80 / 42.5` = 11초
- 계산 = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 250)` = 18초

만약 우리가 IDEFICS-80B의 로그와 대조하면, 매 반복은 약 49초이다.

좋은 소식은 수학적 계산이 정확하다는 것인데, 통신+계산이 측정 시간과 대체로 일치하기 때문이다. 다만

우리는 계산 공식에 우리가 기록한 90 TFLOPS를 입력하여 또 한 번의 온전성 검사를 할 수 있다:

- 계산 = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 90)` = 51초

그래서 49초와 51초는 매우 가깝다. 그러나 이는 아무것도 설명하지 못하는데, 기록된 TFLOPS가 이 공식을 사용하여 계산되었으므로, 당연히 일치해야 하기 때문이다.

가장 좋은 경우, 나는 공식에서 이론적 피크에 가까운 TFLOPS를 사용하고, 시스템에서 실제로 측정된 계산 시간과 대체로 동일한 계산 추정을 얻기를 기대한다. 통신이 계산과 얽혀 있으므로, 우리가 `forward`+`backward`의 벽시계 시간을 측정할 때, 그것은 통신 시간을 포함한다는 점을 기억하라.

결론은 무엇인가? 나는 분명히 여기에 추가적인 숨겨진 병목이 있으므로 더 많은 조사가 필요하다고 생각한다. 나는 더 이상 이 설정에 접근하여 조사할 수 없으므로, 내가 다른 더 큰 모델을 학습할 때, 이 연습을 다시 수행하고 갱신된 수학적 계산을 당신과 공유할 것이다. 그러나 이 연습은 당신에게 무대 뒤에서 일어나는 일과 이 숫자들이 어떻게 함께 작동하는지에 대한 감을 줄 것이다.

또한, 이 논의는 그래디언트 누적 스텝(GAS)을 수학적 계산에 포함하지 않았다. IDEFICS-80B의 경우에는 그것을 사용하지 않았다. 만약 GAS>1이라면, 이론적 계산 시간은 변하지 않지만, 통신 시간은 `3*2*M/GBps`에서 `GAS*3*2*M/GBps`로 변한다. `forward`와 `backward`의 가중치 수집은 `all_gather`를 통해 그래디언트 누적 스텝만큼 많이 발생한다. 이론적으로 그래디언트에 대해서는 한 번만 발생하면 되지만, 각 GPU 위에 수집된 가중치의 중간 그래디언트를 저장할 곳이 없으므로, 그것 또한 GAS 횟수만큼 reduce해야 한다. 이는 ZeRO-2와 ZeRO-3에 적용된다. ZeRO-1의 경우, GAS>1은 추가 통신을 필요로 하지 않는다.

우리는 또한 여기서 잠재적 병목으로서 `DataLoader`를 논의하지 않았지만, 우리의 테스트에서 그것이 1초 미만임을 발견하였으며, 즉 오버헤드가 작다.

통신 수학으로 돌아가서, 우리는 또한 다양한 하드웨어 지연을 고려하지 않았지만, 큰 페이로드를 처리할 때 그것들은 현저한 추가 오버헤드를 더하지 않을 것이다.

이제 당신은 당신의 시스템 네트워크에서 그 많은 GB를 전송하는 데 얼마나 걸리는지 안다. 예를 들어, 만약 네트워크가 우리가 IDEFICS-80B 학습에 사용한 네트워크보다 5배 느리다면, 즉 8.5GBps(68Gbps)라면:

- 통신 = `3 * 2 * 80 / 8.5` = 56초

이는 더 빠른 계산과 비교하면 분명히 거대한 병목이 될 것이다.

만약 네트워크가 5배 빠르다면, 즉 212GBs(1700Gbps)라면:

- 통신 = `3 * 2 * 80 / 212` = 2초

이는 계산 시간에 비해 미미할 것이며, 특히 그 중 일부가 성공적으로 계산과 중첩된다면 그러하다.

또한, Deepspeed 팀은 384개의 V100 GPU(24개의 DGX-2 노드)에서 176B 모델에 대해 경험적 벤치마크를 수행하였고, 다음을 발견하였다:

1. 100 Gbps IB를 사용하면, 각 GPU는 <20 TFLOPs만 가진다(나쁨)
2. 200-400 Gbps IB를 사용하면, 각 GPU가 합리적인 30-40 TFLOPs에 도달한다(괜찮음)
3. 800 Gbps IB의 경우, 각 GPU가 40+ TFLOPs에 도달한다(우수)

상기시켜 두자면, NVIDIA V100의 fp16 피크 TFLOPS는 125 TFLOPS이다(https://www.nvidia.com/en-gb/data-center/tesla-v100/)。

그러나 주의하라 - 이 벤치마크는 V100에 대한 것이다! 그것은 A100보다 2-3배 느리고, H100보다 4-8배 느리다(반정밀도). 그래서 H100 노드의 경우, 통신은 위 표와 반정밀도에서 일치하려면 적어도 4-8배 빨라야 한다. 우리는 더 최신 하드웨어를 사용한 더 많은 벤치마크 테스트가 필요하다.

참고: 2-3배 범위인 것은, 공식 사양이 V100->A100과 A100->H100이 각각 3배의 TFLOPS 증가를 주장하지만, 사용자 벤치마크 테스트가 보고한 차이는 최대 2.5배 개선이기 때문이다.

그들은 또한, 대규모로 학습할 때 각 GPU의 작은 마이크로 배치 크기가 통신 오버헤드를 더 두드러지게 만든다는 점에 주목하였다. 그리고 우리는 좋은 모델 수렴률을 실현하기 위해 전역 배치 크기가 보통 고정되어 있으므로, 마이크로 배치 크기를 늘릴 수 없을 수 있다. 이 문제는 최근 도입된 ZeRO++(https://github.com/BBuf/ml-engineering/blob/master/training/model-parallelism/README.md#zero-with-multiple-replicas)를 통해 해결되었다.

마지막으로, 위 수학적 계산을 수행할 때, 당신은 당신의 설정에서 얻은 실제 대역폭을 알아야 한다 - 이는 페이로드 크기에 따라 변한다 - 페이로드가 클수록 대역폭이 좋아진다. 이 정보를 얻으려면, 당신은 Deepspeed 설정 파일에서 reduction과 프리페치에 각각 사용되는 `reduce_bucket_size`와 `prefetch_bucket_size` 설정을 봐야 한다. 기본값은 0.5B 파라미터이며, 반정밀도에서는 1GB(0.5B x 2바이트), fp32 정밀도를 사용하면 2GB(0.5B x 4바이트)이다. 그래서 실제 처리량을 측정하려면, 당신은 그 페이로드로 `all_reduce` 벤치마크 테스트를 실행하여 보고된 대역폭이 얼마인지 봐야 한다. 그런 다음 그것을 위 계산에 입력할 수 있다.



## 언제 어떤 전략을 사용할 것인가

다음은 매우 대략적인 병렬 전략 사용 안내이다. 각 목록의 첫 번째 항목이 보통 더 빠르다.

**⇨ 단일 GPU**

* 모델이 단일 GPU에 적재될 수 있는 경우:

    1. 정상 사용

* 모델이 단일 GPU에 적재될 수 없는 경우:

    1. ZeRO + CPU 오프로드, 선택적으로 NVMe 사용
    2. 만약 가장 큰 레이어가 단일 GPU에 적재될 수 없다면, 위 방법에 메모리 중심 분할(아래 자세히 참조)을 추가

* 가장 큰 레이어가 단일 GPU에 적재될 수 없는 경우:

1. ZeRO - 메모리 중심 분할(https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)(MCT) 활성화. 그것은 자동 분할과 순차 실행을 통해 임의로 큰 레이어를 실행할 수 있게 한다. MCT는 GPU 위의 활성 파라미터 수를 줄이지만, 활성화 메모리에는 영향을 미치지 않는다. 다만 이러한 요구는 현재 드물며, 사용자가 수동으로 `torch.nn.Linear`를 재작성해야 한다.

**⇨ 단일 노드/다중 GPU**

* 모델이 단일 GPU에 적재될 수 있는 경우:

    1. DDP - 분산 데이터 병렬
    2. ZeRO - 구체적인 상황과 사용하는 설정에 따라, 더 빠를 수도 더 느릴 수도 있다

* 모델이 단일 GPU에 적재될 수 없는 경우:

    1. PP(파이프라인 병렬)
    2. ZeRO
    3. TP(텐서 병렬)

    NVLINK 또는 NVSwitch의 빠른 노드 내 연결이 있는 경우, 이 세 가지 방법의 성능은 대체로 비슷할 것이다. 만약 이것들이 없다면, PP가 TP나 ZeRO보다 빠를 것이다. TP의 차수도 차이를 만들 수 있다. 최적 방안을 찾기 위해 당신의 특정 설정에서 실험해 보는 것이 가장 좋다.

    TP는 거의 항상 단일 노드 내에서 사용된다. 즉 TP 크기 <= 각 노드의 GPU 수.

* 가장 큰 레이어가 단일 GPU에 적재될 수 없는 경우:

    1. 만약 ZeRO를 사용하지 않는다면 - TP를 사용해야 한다. PP 단독으로는 적재할 수 없기 때문이다.
    2. ZeRO를 사용할 때는, 위 "단일 GPU" 절의 동일한 항목을 참조하라

**⇨ 다중 노드/다중 GPU**

* 만약 모델이 단일 노드에 적재될 수 있다면, 먼저 여러 복제본을 사용하는 ZeRO(본 문서에서 여러 복제본을 사용하는 ZeRO를 검색)를 시도해 보라. 이렇게 하면 당신은 더 빠른 노드 내 연결에서 ZeRO를 수행하고, 더 느린 노드 간 연결에서 DDP를 수행하게 되기 때문이다

* 빠른 노드 간 연결이 있을 때:

    1. ZeRO - 모델을 거의 수정할 필요가 없기 때문
    2. PP+TP+DP - 통신은 더 적지만, 모델을 대폭 변경해야 한다

* 느린 노드 간 연결이 있고 GPU 메모리가 여전히 부족할 때:

    1. DP+PP+TP+ZeRO-1


# 병렬 학습에 관한 지후(知乎)의 일부 관련 문헌

- [한 편으로 이해하는 MPI 통신 인터페이스의 특징과 원리](https://zhuanlan.zhihu.com/p/653968730)
- [ring attention + flash attention: 초장 컨텍스트로 가는 길](https://zhuanlan.zhihu.com/p/683714620)
- [대규모 모델 학습의 시퀀스 병렬 쌍벽: DeepSpeed Ulysses & Ring-Attention](https://zhuanlan.zhihu.com/p/689067888)
- [시퀀스 병렬로 대규모 모델 학습하기, 당신이 알아야 할 여섯 가지](https://zhuanlan.zhihu.com/p/698031151)
- [나는 DeepSpeed-Ulysses를 사랑한다: 대규모 모델 시퀀스 병렬 기법 재고찰](https://zhuanlan.zhihu.com/p/703669087)
- [대규모 모델 추론 시퀀스 병렬](https://zhuanlan.zhihu.com/p/703669087)
