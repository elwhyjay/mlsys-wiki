> 내 강의 노트이며, 관심 있으면 팔로우해도 좋다: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode

> 이 문서의 출처: https://github.com/stas00/ml-engineering . 이 문서는 machine learning workload에서 CPU가 어떻게 사용되는지 주로 다룬다. CPU core requirement를 계산하는 방법을 자세히 소개하며, 각 accelerator와 각 DataLoader worker process에 전용 CPU core가 필요하다는 점을 강조한다. 또한 CPU memory 사용도 논의하며, 보통 각 node의 CPU memory는 적어도 GPU memory와 비슷해야 한다고 지적하고, 주요 memory 사용 상황을 열거한다. 그 밖에도 성능에 영향을 줄 수 있는 NUMA affinity, hyper-threading 등의 요소와 mmap mode를 사용할 때 발생할 수 있는 memory usage 오판도 언급한다.

# CPU

이 글을 쓰는 시점에서 machine learning workload는 CPU를 대량으로 사용하지 않는다. 따라서 이 장의 내용은 많지 않다. CPU가 점점 GPU와 비슷한 방향으로 발전함에 따라 이 상황은 바뀔 수 있으므로, CPU의 진화와 함께 이 장도 발전할 것이라고 예상한다.

## 얼마나 많은 CPU Core가 필요한가

Accelerator 1개마다 다음이 필요하다.

1. accelerator에 bind된 process용 CPU core 1개
2. 각 `DataLoader` worker process마다 CPU core 1개. 보통 worker process는 2~4개가 필요하다.

Language model의 경우, 특히 data가 이미 preprocessing되어 있다면 worker process 2개면 보통 충분하다.

동적 변환이 필요하다면, 이는 computer vision model이나 vision-language model에서 자주 나타나며, 3~4개 또는 그 이상의 worker process가 필요할 수 있다.

목표는 accelerator computation을 block하지 않고 `DataLoader`에서 data를 즉시 가져올 수 있게 하는 것이다. 이는 현재 iteration이 실행되는 동안 다음 iteration의 sample batch를 preprocessing해야 함을 뜻한다. 다시 말해, 다음 batch의 data processing 시간이 같은 크기 batch의 단일 accelerator computation iteration 시간보다 길어서는 안 된다.

Preprocessing 외에도 local storage가 아니라 cloud에서 data를 동적으로 가져온다면, accelerator에 data를 공급하는 worker process의 요구를 만족할 만큼 data prefetch 속도가 충분히 빠른지도 확인해야 한다.

이 숫자에 accelerator 수를 곱하고, operating system에 필요한 core 몇 개(예를 들어 4개)를 더한다.

Node에 accelerator 8개가 있고 worker process가 n개라면 `8*(num_workers+1)+4`개의 core가 필요하다. NLP task를 하고 있다면 보통 accelerator마다 worker process 2개가 필요하므로 `8*(2+1)+4` => 28개 CPU core가 된다. CV training을 하고 있고 accelerator마다 worker process가 4개 필요하다고 가정하면 `8(4+1)+4` => 44개 CPU core가 된다.

매우 active한 process 수가 전체 CPU core 수보다 많으면 어떻게 될까? 일부 process는 preempt되어 CPU core가 사용 가능해질 때까지 queue에 들어간다. 어떤 context switch도 반드시 피해야 한다.

하지만 modern cloud service는 보통 50~100개 이상의 CPU core를 제공하므로, 일반적으로 core 부족 문제는 잘 나타나지 않는다.

Async DataLoader(https://github.com/stas00/ml-engineering/tree/master/training/performance#asynchronous-dataloader)도 참고하라.

### CPU Offload

Deepspeed(https://www.deepspeed.ai/tutorials/zero-offload/) 같은 일부 framework는 병목을 만들지 않고 일부 computation work를 CPU로 offload할 수 있다. 이런 경우에는 추가 CPU core가 필요하다.

## NUMA Affinity

NUMA affinity(https://github.com/stas00/ml-engineering/blob/master/training/performance#numa-affinity)를 참고하라.

## Hyper-Threading

Hyper-threading(https://en.wikipedia.org/wiki/Hyper-threading)은 각 physical core를 2개의 virtual core로 가상화해 두 thread가 같은 CPU core를 동시에 사용할 수 있게 하며, CPU core 수를 두 배로 보이게 한다. Workload type에 따라 이 기능은 전체 성능을 향상할 수도 있고 아닐 수도 있다. 이 기술의 발명자인 Intel은 특정 상황에서 30%의 성능 향상이 가능하다고 말한다.

Hyper-thread를 켤지 말지(https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/performance.md#to-enable-hyper-threads-or-not)도 참고하라.

# CPU Memory

이 장은 매우 짧다. 보통 CPU memory에 대해 알아야 할 세부 사항이 많지 않기 때문이다. 이는 좋은 일이다!

대부분 ML workload의 computation은 GPU에서 발생하지만, 보통 각 node의 CPU memory는 적어도 GPU memory만큼은 있어야 한다. 예를 들어 80GB GPU 8개가 있는 H100 node를 사용한다면 GPU memory는 640GB다. 따라서 적어도 그만큼의 CPU memory가 필요하다. 하지만 최근 high-end cloud service package는 보통 1~2TB CPU memory를 갖춘다.

## ML Workload에서 CPU Memory의 용도

- Model weight를 load한다. 단, weight를 GPU로 직접 load하는 경우는 제외한다. 이는 보통 일시적인 memory 사용이며, model이 GPU로 이동하면 0으로 돌아간다.
- Model weight를 저장한다. 어떤 경우에는 각 GPU가 자신의 checkpoint를 disk에 직접 쓰고, 다른 경우에는 disk에 쓰기 전에 model을 CPU에서 다시 조립한다. 이것도 일시적인 memory 사용이다.
- Deepspeed(https://www.deepspeed.ai/tutorials/zero-offload/) 같은 framework를 사용할 때 parameter와 optimizer state offload가 필요할 수 있다. 이 경우 상당한 CPU memory가 필요할 수 있다.
- `forward` pass에서 계산되고 `backward` path에서 필요해지는 activation도 버렸다가 backward propagation 중 다시 계산하는 대신 CPU로 offload하여 불필요한 overhead를 줄일 수 있다.
- `DataLoader`는 보통 CPU memory의 주요 사용자 중 하나이며, 때로는 많은 memory를 사용할 수 있다. 보통 각 node에서 최소 2x8개의 DL worker process가 실행되므로, 각자 일부 data를 보유하는 최소 16개 process를 지원할 enough memory가 필요하다. 예를 들어 cloud에서 data를 streaming하는 경우 data shard가 크면 이 process들이 수백 GB의 CPU memory를 쉽게 소비할 수 있다.
- Software 자체와 dependency library도 일부 CPU memory를 사용하지만, 이 양은 보통 무시할 수 있다.

## 알아야 할 사항

- `DataLoader`가 `mmap` mode에서 HF `datasets`를 사용할 경우 resident memory usage가 많은 CPU memory를 사용하는 것처럼 보일 수 있다. 전체 dataset을 memory에 mapping하려고 하기 때문이다. 하지만 이는 misleading하다. 다른 곳에서 memory가 필요하면 operating system이 필요 없는 mmap'ed page를 system으로 page back하기 때문이다. 관련 내용은 여기(https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/)에서 더 읽을 수 있다. 물론 이 인식은 `mmap`을 사용하는 모든 dataset에 적용되며, HF `datasets`를 예로 든 것은 널리 사용되기 때문이다.

