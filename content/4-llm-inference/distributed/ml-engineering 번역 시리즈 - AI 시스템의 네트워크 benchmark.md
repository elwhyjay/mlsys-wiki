> 내 강의 노트이며, 관심이 있다면 팔로우를 환영한다. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 이 문서의 출처는 https://github.com/stas00/ml-engineering 이다. 이 문서는 주로 대규모 분산 머신러닝 학습에서의 네트워크 benchmark와 최적화를 소개한다. 먼저 네트워크 성능을 테스트하는 몇 가지 도구 script를 소개한다. 여기에는 `all_reduce_bench.py`, `all_gather_object_vs_all_reduce.py`, `all_reduce_latency_comp.py`가 포함된다. 이어서 네트워크 benchmark의 핵심 요구 사항을 논의하며, 재현 가능성의 중요성을 강조한다. 다음으로 네트워크 throughput의 중요성을 자세히 소개한다. 여기에는 결과를 테스트하고 해석하는 방법, 그리고 서로 다른 GPU와 framework가 네트워크 bandwidth에 대해 갖는 요구 사항이 포함된다. 또한 NCCL(NVIDIA Collective Communications Library)의 성능 최적화도 다루며, 중요한 NCCL 환경 변수 몇 가지와 그 역할을 소개한다. 마지막으로 문서는 세 개의 benchmark script인 `all_reduce_bench.py`, `all_reduce_latency_comp.py`, `all_gather_object_vs_all_reduce.py`의 상세 코드를 제공한다. 이 script들은 서로 다른 시나리오에서 네트워크 성능을 테스트하는 데 사용할 수 있다.

# 네트워크 benchmark

**도구**:

- `all_reduce_bench.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py) - 대량 데이터에서 `all_reduce` operation을 수행할 때 실제 네트워크 bandwidth를 benchmark하는 도구다. 실제로 얻는 성능과 홍보된 specification 사이의 차이를 이해하는 데 유용하다.

- `all_gather_object_vs_all_reduce.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_gather_object_vs_all_reduce.py) - process group에서 완료 상태를 수집할 때 `all_gather_object`에서 `all_reduce`로 바꾸면 23배 속도 향상을 얻을 수 있음을 보여주는 빠른 benchmark다. 예를 들어 모든 process가 완료되었는지를 나타내는 flag를 구현할 때 쓸 수 있다. 이 기법은 서로 다른 iteration 수에서 완료될 수 있는 GPU를 동기화할 때 흔히 사용된다. 여러 DP channel에서 inference를 수행할 때 필요하거나, `DataLoader`에서 `StopIteration` event를 동기화하고 싶을 때 필요하다. `all_gather_object_vs_all_gather.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_gather_object_vs_all_gather.py)도 참고하라.

- `all_reduce_latency_comp.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_latency_comp.py) - 4GB 규모의 reduction operation 1회가 4MB 규모의 reduction operation 1000회보다 훨씬 빠르다는 예를 보여준다.



## 핵심 재현성 요구 사항

성공적인 실험 series에서 가장 중요한 요구 사항은 하나 또는 몇 개의 설정 변수만 바꾸면서 실험 환경을 반복적으로 재현할 수 있어야 한다는 것이다.

따라서 어떤 변화가 성능을 높이거나 낮추는지 알아내려면, 사물을 안정적으로 유지할 방법을 찾아야 한다.

예를 들어 네트워크 사용량에 변동이 생기지 않도록 막는 방법을 찾아야 한다. 108B pre-BLOOM experiment(https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)의 성능을 최적화할 때는 이것이 거의 불가능한 일이었다. 공유 node 간 네트워크를 사용했기 때문에 완전히 같은 설정이라도 다른 사용자가 네트워크를 얼마나 사용하는지에 따라 throughput이 달라졌다. 이런 방식으로는 되지 않는다. BLOOM-176B 기간에는 격리된 네트워크가 있는 전용 SLURM partition을 얻었고, 그 네트워크에는 우리 traffic만 있었다. 이런 환경에서 성능 최적화를 하는 것은 그야말로 완벽했다.


## 네트워크 throughput

특정 model size와 framework가 네트워크 bandwidth, throughput, latency에 대해 갖는 요구 사항을 이해하는 것은 매우 중요하다. 네트워크에 충분히 투자하지 않으면 결국 GPU가 idle 상태가 되어 돈과 시간이 낭비된다. 반대로 매우 빠른 네트워크에 과도하게 비용을 지불했지만 GPU가 느리다면, 역시 돈과 시간을 낭비한 것이다.

네트워크가 매우 느리면 학습은 높은 확률로 네트워크에 제한되며, 많은 학습 설정 개선이 성능 향상에 도움이 되지 않는다.

참고: EAI cookbook(https://github.com/EleutherAI/cookbook)에는 각 collective communication operation을 위한 communication benchmark set(https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication)이 포함되어 있으며, node 간 또는 node 내부 네트워크 throughput을 빠르게 측정하는 데 사용할 수 있다.

아래에는 node 간 네트워크 throughput을 빠르게 측정할 수 있는 간단한 all-reduce benchmark가 있다.

`all_reduce_bench.py`(https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py)

일반적으로 적어도 4개 node에서 benchmark하기를 권장한다. 물론 학습 기간에 사용할 모든 node에 이미 접근할 수 있다면 모든 node를 사용해 benchmark하라.

4개 node에서 실행한다.

```
GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py
```

주의:
- rank 0 hostname을 자동으로 얻는 SLURM 환경이 아니라면 `MASTER_ADDR`를 rank 0의 hostname으로 조정해야 한다.

아래는 SLURM 환경에서 4개 node를 사용해 실행하는 방법이다.
```
salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash
srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 --nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend c10d all_reduce_bench.py
```

주의:
- `--cpus-per-task`와 `--partition` parameter를 조정해야 할 수 있다.
- `salloc`은 한 번만 실행하면 되며, 이후 같은 allocation에서 `srun`을 여러 번 반복 실행할 수 있다.

현재 상황에 따라 5Gbps에서 1600Gbps 사이의 결과가 나올 수 있다. 네트워크 제한을 피하기 위한 최소 속도는 특정 학습 framework에 따라 다르지만, 일반적으로 적어도 400Gbps 이상을 원하게 된다. 다만 우리는 50Gbps 네트워크에서 BLOOM을 학습한 적도 있다.

Deepspeed(https://github.com/microsoft/DeepSpeed)에서 ZeRO Stage-3를 사용하는 것처럼 weight와 optimization stage를 shard하는 framework는, data parallel 외에 tensor parallel과 pipeline parallel도 수행하는 Megatron-Deepspeed(https://github.com/bigscience-workshop/Megatron-DeepSpeed) 같은 framework보다 더 많은 네트워크 traffic을 만든다. 후자는 activation만 전송하므로 그렇게 많은 bandwidth가 필요하지 않다. 하지만 설정과 실행이 훨씬 복잡하다.

물론 효율적인 framework는 communication과 computation을 overlap한다. 그래서 한 stage가 데이터를 가져오는 동안 다른 stage는 병렬로 computation을 수행할 수 있다. 따라서 communication overhead가 computation 시간보다 작기만 하면 네트워크 요구 사항은 충족되며, 반드시 매우 뛰어날 필요는 없다.

DeepSpeed ZeRO Stage 3와 V100 GPU로 대규모 학습(64+ GPU)을 수행할 때 합리적인 GPU throughput을 얻으려면 다음과 같다.

1. 100Gbps는 충분하지 않다.
2. 200-400 Gbps는 괜찮다.
3. 800-1000 Gbps가 이상적이다.

전체 세부 정보(https://github.com/microsoft/DeepSpeed/issues/2928#issuecomment-1463041491)

물론 A100 GPU node의 요구 사항은 더 높고, H100의 요구 사항은 더 높다. 하지만 현재 이런 benchmark 정보는 아직 공유되지 않았다.

### 몇 개 node의 benchmark 결과를 여러 node로 추론하기

수백 개 node를 benchmark하는 일은 보통 쉽지 않기 때문에, 우리는 종종 4개 node를 사용해 interconnect 성능을 benchmark하려고 한다. 이것이 40개 또는 400개 node를 사용할 때 올바른 지표를 제공하는지는 확실하지 않아서 여기(https://github.com/NVIDIA/nccl/issues/790)에 질문했고, 다음과 같은 답변을 받았다.

> ring algorithm과 tree algorithm에 대해 대규모 추론을 하는 것은 어렵지 않다. 우리는 `tuning.cc`에 이를 예측하는 함수가 있으며, ring 기반의 linear latency와 tree 기반의 logarithmic latency, 그리고 감소한 bandwidth를 바탕으로 한다. 하지만 scale이 커질수록 실제 성능이 예측과 크게 달라지게 만드는 요인이 많다. 예를 들어 routing이 있다. 또한 IB network에서는 SHARP를 사용할 수 있다는 점에 유의하라. 그러면 scale이 커져도 latency가 기본적으로 변하지 않고 bandwidth도 많이 떨어지지 않으며, 항상 ring 및 tree algorithm보다 낫다.


## 성능을 위한 NCCL 환경 변수

NCCL은 어떤 네트워크가 주어져도 최적 성능을 자동으로 찾아내는 데 뛰어나지만, 때때로 도움이 필요하다. 이 경우 다음 NCCL 환경 변수로 성능을 조정한다. 알아두면 좋을 몇 가지 흔한 변수를 살펴보자. 전체 목록은 여기(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)에서 찾을 수 있다.

### `NCCL_ALGO`

이 변수는 NCCL이 사용할 algorithm을 정의한다. 보통 다음 중 하나다.

1. Tree
2. Ring
3. CollnetDirect와 CollnetChain (IB SHARP)
4. NVLS (NVLink SHARP)

이 NCCL issue(https://github.com/NVIDIA/nccl/issues/790)에서 사용자가 어떻게 최적화해야 하는지 물었고, 답변은 기본적으로 사용자가 아무것도 최적화하려고 해서는 안 된다는 것이었다. NCCL 내부에는 구체적인 상황에 따라 한 algorithm에서 다른 algorithm으로 자동 전환하는 많은 지능형 algorithm이 있기 때문이다.

Sylvain Jeaugey는 다음과 같이 공유했다.

> 예전에는 static threshold가 있었지만, 더 복잡한 tuning system으로 대체되었다. 새 system은 각 algorithm/protocol 조합에 대해 latency와 bandwidth model을 만든다. 조합이 매우 많다. 그리고 size에 따라 어떤 조합이 가장 좋은 성능을 낼지 결정한다. 그래서 더 이상 환경 변수와 static value는 없다. 이는 좋은 일이다. 각 algorithm의 성능은 node 수와 node당 GPU 수에 따라 달라지므로, algorithm/protocol의 2차원 공간을 탐색해야 하며 이것은 쉽지 않다. 언제든 `NCCL_ALGO=TREE`와 `NCCL_ALGO=RING`으로 특정 algorithm을 강제해 어떤 성능이 나오는지, NCCL이 올바른 지점에서 전환하는지 확인할 수 있다. 이해하기 어렵다는 것을 알지만, 이것이 모든 platform과 사용자에서 최상의 성능을 얻기 위해 우리가 찾은 최선의 해결책이다. 사용자가 전환 지점을 수동으로 조정할 필요가 없다. 단점은 무언가를 수동으로 조정하고 싶어도 할 수 없다는 것이다.

`NCCL_ALGO`를 사용한다면 고려할 algorithm을 나열해야 하지만, 그 외에는 제어할 수 없다. 따라서 이것은 특정 algorithm을 사용하지 않도록 보장하고 싶을 때에만 유용하다.

어떤 algorithm이 더 나은지 물었을 때 받은 답변은 다음과 같다.

> 대략적으로 말하면 ring은 peak bandwidth 측면에서 더 좋다. 2개 node의 경우는 제외한다. 반면 tree는 base latency 측면에서 더 좋다. 특히 scale이 커질 때 그렇다. `bandwidth = size / time`이므로, 주어진 data size에서 time을 보든 bandwidth를 보든 peak bandwidth와 base latency의 조합이 된다. 고정 size data에서 scale이 커지면 ring의 base latency가 더 두드러지고, 이때 tree 구조가 더 잘 동작한다.

또 다른 새 algorithm인 `NVLS`가 있다. NVLink SHARP를 사용할 수 있다면 NVLink 자체보다 더 빠르게 실행된다. 예를 들어 NVLink 4.0(450GBps)을 사용하면 all-reduce benchmark에서 480GBps에 도달할 수 있다. 이들은 IB 또는 RoCE(https://github.com/NVIDIA/nccl/issues/1031#issuecomment-1773965518)가 필요한 node 간 version을 개발하고 있다. 이 글을 쓰는 시점에는 이 새 algorithm이 어디에도 문서화되어 있지 않다.

마지막으로 어떤 algorithm이 사용되고 있는지 알고 싶다면, 알 수 없다. 이 답변(https://github.com/NVIDIA/nccl/issues/754#issuecomment-1346163469)을 참고하라. 따라서 어떤 algorithm이 어떤 throughput을 내는지 알고 싶다면 `NCCL_ALGO` 환경 변수를 설정해 모든 algorithm을 명시적으로 시도해야 한다. 그러면 어떤 것이 선택되었는지 알 수 있다. 또는 같은 답변의 제안처럼 NCCL을 수정하고 다시 compile할 수 있지만, production 환경에서는 그렇게 하고 싶지 않을 것이다.


### `NCCL_CROSS_NIC`

`NCCL_CROSS_NIC` 변수는 NCCL이 ring/tree 구조에서 서로 다른 network interface card(NIC)를 사용하도록 허용할지 제어한다. 이렇게 하면 node 간 communication이 서로 다른 node에서 서로 다른 NIC를 사용할 수 있다.

여러 NIC를 사용할 때 node 간 communication 성능을 최대화하기 위해, NCCL은 node 사이에서 같은 NIC를 사용해 communication하려고 한다. 이는 각 node의 각 NIC가 서로 다른 network switch(network rail)에 연결되는 network design에 맞추고, traffic 간섭 위험을 피하기 위한 것이다. 따라서 `NCCL_CROSS_NIC` 설정은 network topology에 따라 달라지며, 특히 network fabric이 rail에 최적화되어 있는지에 따라 달라진다.

NIC가 하나뿐인 system에는 영향이 없다.

허용되는 값은 다음과 같다.

- 0: 같은 ring/tree 구조에는 항상 같은 NIC를 사용해 network rail을 가로지르지 않도록 한다. 각 NIC마다 switch(rail)가 있고 rail 간 연결이 느린 network에 적합하다. 일부 특수한 경우 NCCL이 여전히 cross-rail communication을 유발할 수 있으므로, rail은 여전히 top level에서 연결되어 있어야 한다.
- 1: 같은 ring/tree 구조에 같은 NIC를 사용하려고 시도하지 않는다. node의 모든 NIC가 같은 switch에 연결되는 network에 적합하다. 이런 경우 같은 NIC를 통한 communication을 시도해도 traffic conflict를 피하는 데 도움이 되지 않는다.
- 2: 기본값이다. 같은 ring/tree 구조에 같은 NIC를 사용하려고 시도하지만, 다른 NIC를 사용하는 것이 더 좋은 성능을 가져오면 다른 NIC 사용을 허용한다.

# Benchmark 관련 script

## `all_reduce_bench.py`

코드 경로: https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py

```python
#!/usr/bin/env python

"""

이 program의 최신 version은 https://github.com/stas00/ml-engineering 에서 찾을 수 있다.

이 benchmark는 https://github.com/NVIDIA/nccl-tests 와 매우 비슷하지만, PyTorch 설치만 필요하므로 설정이 더 쉽다.

이 version:
- @jeffra의 gist에서 파생되었다. https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36
- 그리고 이 gist는 https://github.com/NVIDIA/nccl-tests 의 logic에서 파생되었다.
- contributor에는 다음이 포함된다.
  * Indu Thangakrishnan https://github.com/indhub cuda event를 사용해 timing을 올바르게 처리했다.

중요 설명:

- 이 benchmark 실행을 마친 뒤에는 여기에서 설명한 것처럼 https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth algbw가 아니라 busbw 결과에 주목해야 한다.

- NVIDIA/nccl-tests와 비슷하게 이 benchmark는 단방향 bandwidth를 측정한다. 따라서 결과를 광고된 양방향(full duplex) peak throughput이 아니라 단방향 peak throughput과 비교해야 한다.

- 현재 이 benchmark는 4GB payload(M * N * 4)를 테스트한다. target application이 사용하는 payload가 훨씬 작다면, target payload에 맞게 M*N*4를 수정해야 한다. payload를 계산하려면, 각 reduction에서 전송되는 parameter 수에 2(bf16/fp16) 또는 4(fp32)를 곱한다. 예를 들어 reduction 한 번이 1B parameter인 단일 layer이고 bf16 gradient를 사용한다면 payload는 2GB이다. 사용하는 framework(DDP, FSDP, DeepSpeed ZeRO)에 따라 전송하는 message size에 대한 logic이 모두 다르다.

- https://github.com/NVIDIA/nccl-tests 를 계속 실행해야 하는지 알고 싶다면, 나는 ./build/all_reduce_perf -b 4G -e 4G 로 매우 비슷한 결과를 얻는 것을 검증했다. 4개 node에서 mpirun으로 테스트했다. 결과는 비슷하거나 약간 느려야 한다. 이는 blocking 방법을 사용하기 때문이다. 즉, 각 새로운 all_reduce가 완료될 때까지 기다린 다음 다음 all_reduce를 trigger한다. 반면 nccl-tests는 이를 asynchronous 방식으로 trigger한다. nccl-tests에 `-z`를 추가하면 blocking을 simulate할 수 있다.

- 다른 collective operation을 benchmark하려면 nccl-tests를 사용하라. payload range를 테스트하고 싶을 때도 유용하다. 예를 들어 -b 8 -e 4G -f 2를 설정하면 여러 size를 자동으로 테스트한다.

4개 node에서 실행:

GPUS_PER_NODE=8
NNODES=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    all_reduce_bench.py

주의: hostname을 자동으로 가져오는 SLURM 환경이 아니라면 MASTER_ADDR를 node rank 0의 hostname으로 조정하라.

예를 들어 salloc+srun으로 실행하는 예시는 다음과 같다.

salloc --partition=mypartition --nodes=4 --ntasks-per-node=1 --cpus-per-task=48 --gres=gpu:8 --time=1:00:00 bash

srun --gres=gpu:8 --nodes=4 --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=8 \
--nnodes 4 --rdzv_endpoint $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):6000 --rdzv_backend \
c10d all_reduce_bench.py

2개 GPU에서 빠르게 테스트하려면:

python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d \
all_reduce_bench.py

"""

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 5

# 이 simulation은 M * N * 4 size tensor의 payload가 된다.
N = 500000
M = 2000

def timed_allreduce(mat, start_event, end_event):
    dist.barrier()
    start_event.record()
    dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    size = M * N * 4 # 4는 fp32의 4 byte이다.
    # 여기서는 NVIDIA/nccl-tests와 같은 계산 방법을 따른다.
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # 모든 rank의 평균값을 계산한다.
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

def run(local_rank):
    hostname = socket.gethostname()
    is_global_rank_0 = dist.get_rank() == 0

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # warmup iteration을 몇 번 수행한다.
    for i in range(2):
        timed_allreduce(mat, start_event, end_event)

    # 실제 benchmark
    algbw_gather = []
    for i in range(TRIALS):
        if is_global_rank_0:
            print(i+1)
        algbw_gather += timed_allreduce(mat, start_event, end_event)

    algbw = torch.mean(torch.stack(algbw_gather))

    # all-reduce 특유의 2*(n-1)/n busbw correction factor는 여기에서 설명한다.
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw는 hardware 사용 효율을 반영한다.
    n = dist.get_world_size()
    busbw = algbw * (2*(n - 1) / n)

    if is_global_rank_0:
        print(f"all_reduce average bandwidth for {M*N*4/1e9}GB payload ({TRIALS} trials, {n} ranks):\n",
              f"algbw: {algbw/1e9:.3f} GBps ({algbw*8/1e9:.1f} Gbps)\n",
              f"busbw: {busbw/1e9:.3f} GBps ({busbw*8/1e9:.1f} Gbps)\n",
        )

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)
```

## `all_reduce_latency_comp.py`

> 이 script는 4GB 규모의 reduction operation 1회가 4MB 규모의 reduction operation 1000회보다 훨씬 빠르다는 예를 보여준다.

```python
#!/usr/bin/env python

# 이 script는 all_reduce_bench.py에서 파생되었다.
# 하지만 4GB 규모의 reduction operation 1회가 4MB 규모의 reduction operation 1000회보다 훨씬 빠르다는 점을 보여주도록 조정했다.
#
# 8개 GPU에서 실행하는 command:
# python -u -m torch.distributed.run --nproc_per_node=8 all_reduce_latency_comp.py

import os
import socket
import torch
import torch.distributed as dist

TRIALS = 1  # experiment repeat count

# 이 parameter들은 M * N * 4 size tensor의 payload가 된다.
N = 500000
M = 2000

def timed_allreduce(mat, repeat_times, id, start_event, end_event):
    start_event.record()
    for i in range(repeat_times):
        dist.all_reduce(mat)
    end_event.record()

    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000  # seconds로 변환한다.

    size = M * N * 4  # 4는 fp32의 byte 수이다.
    algbw = (size / duration) * 8  # 8은 byte를 bit로 변환한다.
    n = dist.get_world_size()
    # all-reduce 특유의 2*(n-1)/n busbw correction factor는 여기에서 설명한다.
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    # busbw는 hardware 사용 효율을 반영한다.
    busbw = algbw * (2*(n - 1) / n)

    # print가 섞이지 않도록 global rank 0에서 모든 data를 수집하고 결과를 출력한다.
    data = [id, duration, algbw, busbw]
    output = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(data, output, dst=0)
    if dist.get_rank() == 0:
        for data in output:
            id, duration, algbw, busbw = data
            print(f"{id}:\n",
                  f"duration: {duration:.3f} sec\n",
                  f"algbw: {algbw/1e9:.3f} Gbps\n",
                  f"busbw: {busbw / 1e9:.3f} Gbps"
    )

def run(local_rank):
    hostname = socket.gethostname()
    id = f"{hostname}:{local_rank}"
    global_rank = dist.get_rank()

    chunks = 1000
    mat1 = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)  # 4GB tensor
    mat2 = torch.rand(int(N/chunks), M, dtype=torch.float32).cuda(local_rank)  # 4MB tensor

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(TRIALS):
        dist.barrier()  # 모든 process를 동기화한다.

        if global_rank == 0:
            print(f"\n\n\n----------- 1x {N*M*4/1e9}GB ----------------")
        timed_allreduce(mat1, 1, id, start_event, end_event)  # 단일 4GB all-reduce를 테스트한다.

        if global_rank == 0:
            print(f"\n\n\n----------- {chunks}x {(N*M*4/chunks)/1e9}GB ----------------")
        timed_allreduce(mat2, chunks, id, start_event, end_event)  # 1000회의 4MB all-reduce를 테스트한다.

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)  # 현재 process가 사용할 GPU를 설정한다.
    dist.init_process_group(backend)  # distributed environment를 초기화한다.
    fn(local_rank)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])  # local rank를 가져온다.
    print("local_rank: %d" % local_rank)
    init_processes(local_rank=local_rank, fn=run)
```

## `all_gather_object_vs_all_reduce.py`

> process group 사이에서 count를 수집할 때 all_reduce를 사용하면 all_gather_object를 사용할 때보다 23배 빠르다.

```python
#!/usr/bin/env python

#
# process group 사이에서 count를 수집할 때 all_reduce를 사용하면 all_gather_object를 사용할 때보다 23배 빠르다.
#
# 실행 command: python -m torch.distributed.run --nproc_per_node 2 all_gather_object_vs_all_reduce.py
#
# 예시 output:
# all_gather_object=0.26279118900129106
# all_gather_object=0.2628160299973388
# all_reduce       =0.011241967000387376
# all_reduce       =0.011610440000367817

import torch.distributed as dist
import torch
import os

# local process의 rank를 가져온다.
local_rank = int(os.environ["LOCAL_RANK"])
# 현재 process가 사용할 GPU를 설정한다.
torch.cuda.set_device(local_rank)
# distributed environment를 초기화한다.
dist.init_process_group("nccl")
# device를 GPU로 설정한다. 사용할 수 없으면 CPU를 사용한다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# process group의 size와 현재 process의 rank를 가져온다.
world_size = dist.get_world_size()
rank = dist.get_rank()

# 테스트할 tensor와 Python object를 만든다.
flag_pt = torch.tensor(1.0, device=device)
flag_py = 1

def all_gather_object():
    # 수집한 object를 저장할 list를 만든다.
    output_objects = [None for _ in range(world_size)]
    # all_gather_object를 사용해 모든 process의 object를 수집한다.
    dist.all_gather_object(output_objects, flag_py)
    # 수집한 object를 합산한다.
    flag = sum(output_objects)
    return flag

def all_reduce():
    # all_reduce를 사용해 tensor에 sum operation을 수행한다.
    dist.all_reduce(flag_pt, op=dist.ReduceOp.SUM)
    return flag_pt

# 두 함수를 테스트한다.
print(f"all_gather_object: {all_gather_object()}\n")
print(f"all_reduce: {all_reduce()}\n")

import timeit
# timeit module을 사용해 두 함수의 실행 시간을 측정한다. 1000회 실행한다.
print(f'all_gather_object={timeit.Timer("all_gather_object()", globals=globals()).timeit(number=1000)}')
print(f'all_reduce       ={timeit.Timer("all_reduce()"       , globals=globals()).timeit(number=1000)}')
```

# NVLink benchmark 비활성화

wikitext의 작은 sample에서 gpt2 language model을 학습하는 경우를 비교해보자.

결과는 다음과 같다.

| NVlink | 시간 |
| -----  | ---: |
| 예     | 101초 |
| 아니오 | 131초 |

보면 NVLink를 사용해 학습을 완료하는 속도가 약 23% 더 빠르다. 두 번째 benchmark에서는 `NCCL_P2P_DISABLE=1`을 사용해 GPU가 NVLink를 사용하지 않고 PCIe를 사용하도록 알린다.

HF Transformers 예제(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/examples/pytorch/language-modeling/run_clm.py)를 사용할 것이다.

다음은 전체 benchmark code와 output이다.

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

hardware: TITAN RTX 24GB GPU 2개 + NVLink 연결 2개(`nvidia-smi topo -m`에서 `NV2`로 표시됨)
software: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
