> 내 강의 노트이며, 관심이 있다면 팔로우를 환영한다. https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 

> 이 문서의 출처는 https://github.com/stas00/ml-engineering 이다. 이 문서는 주로 NCCL multi-GPU 및 multi-node 연결 문제를 진단하고 해결하는 방법을 소개한다. NCCL debug 정보를 사용해 network interface와 protocol 문제를 식별하는 방법, network interface를 올바르게 설정하는 방법, Docker container에서 NCCL을 사용하는 방법을 자세히 설명한다. 또한 GPU P2P 지원 여부를 확인하는 방법, NCCL call 횟수를 집계하는 방법, 유용한 NCCL debug 환경 변수도 소개한다. 여기에 더해 distributed GPU 설정을 테스트하기 위한 Python script도 제공한다.

# 네트워크 debug

보통 네트워크 문제를 해결하기 위해 network engineer가 될 필요는 없다. 몇 가지 흔한 문제는 아래 주의 사항을 읽는 것만으로 해결할 수 있다.

## 용어표

- OOB: out-of-band, 보통 더 느린 Ethernet NIC
- Bonding: 더 빠른 speed를 얻거나 backup으로 사용하기 위해 여러 NIC를 묶는 것
- IB: InfiniBand, 원래 Mellanox가 개발했고 이후 NVIDIA가 인수했다.
- NIC: network interface card

## NCCL multi-GPU 및 multi-node 연결 문제를 진단하는 방법

이 section은 확실히 빠짐없는 목록은 아니며, 내가 자주 만나는 가장 흔한 setup 문제 몇 가지를 다루기 위한 것이다. 더 복잡한 문제라면 NCCL repository의 Issues(https://github.com/NVIDIA/nccl/issues)를 조사하라. 상황에 맞는 issue를 찾지 못했다면 새 Issue를 제출하라. NCCL에는 짧은 troubleshooting section(https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html)도 있지만, 보통 Issues(https://github.com/NVIDIA/nccl/issues)를 읽으면서 더 많은 것을 배울 수 있다.

network diagnostic 작업에는 full application을 사용하는 대신, 이를 위해 특별히 개발된 design test script인 `torch-distributed-gpu-test.py` (https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py)를 사용할 것을 권장한다. full application은 시작하는 데 오래 걸릴 수 있고 무관한 문제가 있을 수 있다.

먼저 다음 환경 변수를 설정한 뒤 nccl 기반 program을 실행한다.

```
export NCCL_DEBUG=INFO
```
그러면 NCCL 설정과 network traffic에 대한 많은 debug 정보가 출력된다.

예를 들어 위 debug script를 사용하고 8개 GPU가 있는 single node라면 다음처럼 할 수 있다.

```
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

여러 node에서 시작하려면 SLURM이나 Kubernetes 같은 orchestration software를 사용하거나, 각 node에서 수동으로 시작해야 한다. `pdsh`가 매우 도움이 된다. 자세한 내용은 `torch-distributed-gpu-test.py`(https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py)의 설명을 참고하라. 하지만 동작 방식을 이해하기 위해서는 1개 node부터 시작하고, 그다음 2개 node, 이후 더 많은 node로 진행할 것을 권장한다.

이제 program output을 확인하고 다음으로 시작하는 line을 찾는다.
```
NCCL INFO NET/
```
그런 다음 어떤 protocol과 어떤 interface를 사용하는지 확인한다.

예를 들어 이런 output이 있다.
```
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

이는 nccl-fastsocket(https://github.com/google/nccl-fastsocket) transport layer plugin이 사용되었고, 8개의 `ibs*` network interface(NIC card)를 발견했다는 뜻이다. Google Cloud를 사용한다면 이것이 올바르며, NCCL이 이미 올바르게 설정되었을 가능성이 높다. 하지만 InfiniBand(IB)를 사용하면서 위 output을 얻었다면 node 간 speed가 매우 낮을 수 있다. 잘못된 plugin이 활성화되었다는 뜻이기 때문이다.

IB의 경우 보고 싶은 것은 `NET/IB`와 그 IB interface다.
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

여기서는 IB가 collective communication에 사용되고, 8개의 `mlx5_*` interface가 사용되며, OOB도 하나 있음을 볼 수 있다. OOB는 out-of-band communication을 나타낸다. OOB는 connection bootstrap에 사용되며, 보통 더 느린 Ethernet NIC를 사용한다. 때로는 몇 개 NIC가 하나로 bonding(https://wiki.linuxfoundation.org/networking/bonding)되어 있다. interface name 안의 `bond`가 무엇인지 궁금하다면 이것이 그 의미다.

node에 어떤 TCP/IP interface가 있는지 알고 싶다면 node 하나에서 `ifconfig` command를 실행할 수 있다. 보통 비슷한 node는 모두 같은 interface name을 갖지만, 항상 그렇지는 않다.

cluster communication network가 IB라면 `ifconfig` 대신 `ibstat`을 실행해야 한다. 위의 `NCCL INFO NET` 마지막 예시는 다음 output에 대응한다.

```
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

빠른 node 간 connection NIC 외에도 느린 management Ethernet NIC가 하나 있을 가능성이 높다. 몇 개 있을 수도 있다. 이것은 node 설정, shared file system 사용, internet access 등에 사용된다. 따라서 `ifconfig`에는 거의 확실히 추가 NIC가 포함된다. docker network interface, `lo` loopback interface, 기타 interface도 있을 수 있다. 예를 들어 내 desktop에서는 다음 output을 얻을 수 있다.

```
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

이 모든 것을 언급하는 이유는 핵심이 NCCL이 `Using` debug line에서 올바른 interface만 보고하도록 보장하는 것이기 때문이다. 최종적으로 `docker0`, `lo`, `eth0` 같은 interface가 보고된다면, 예를 들면 다음과 같다.

```
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

더 빠른 network interface를 사용할 수 있다면 이것은 대개 원하는 상황이 아니다. 물론 Ethernet NIC가 유일한 선택인 경우에는 위 상황도 괜찮다. 다만 매우 느릴 것이다.

때로 잘못된 interface가 사용되면 application이 직접 hang될 수 있다.

올바른 interface와 잘못된 interface가 동시에 있다면 NCCL은 동작할 수 있지만 speed가 느려진다.

cloud 환경이라면 보통 cloud service provider가 올바른 setup 설명을 제공해야 한다. 제공하지 않았다면 적어도 NCCL 설정에 어떤 network interface를 사용해야 하는지 물어봐야 한다.

NCCL은 어떤 interface를 사용해야 하는지 자동으로 발견하려고 최선을 다하지만, 이것을 올바르게 하지 못한다면 사용할 interface 또는 사용하지 않을 interface를 알려줌으로써 도와줄 수 있다.

- Infiniband를 사용하지 않을 때는 `NCCL_SOCKET_IFNAME`을 사용해 어떤 `ifconfig` interface를 포함하거나 제외할지 지정할 수 있다. 아래 몇 가지 예가 있다.

```
export NCCL_SOCKET_IFNAME=eth:        Use all interfaces starting with eth, e.g. eth0, eth1, …
export NCCL_SOCKET_IFNAME==eth0:      Use only interface eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: Use only interfaces eth0 and eth1
export NCCL_SOCKET_IFNAME=^docker:    Do not use any interface starting with docker
export NCCL_SOCKET_IFNAME=^=docker0:  Do not use interface docker0.
```
전체 문서는 여기(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)에 있다.

- IB RDMA(IB Verbs interface)를 사용할 때는 `NCCL_SOCKET_IFNAME` 대신 `NCCL_IB_HCA` 환경 변수를 사용한다. 이것은 collective communication에 사용할 interface를 선택한다. 예시는 다음과 같다.

```
export NCCL_IB_HCA=mlx5 :               # mlx5로 시작하는 모든 card의 모든 port를 사용한다.
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 : # mlx5_0과 mlx5_1 card의 port 1을 사용한다.
export NCCL_IB_HCA=^=mlx5_1,mlx5_4 :    # mlx5_1과 mlx5_4 card를 사용하지 않는다.
```
전체 문서는 여기(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)에서 찾을 수 있다.

예를 들어 IB를 사용할 때 보통 `mlx5_bond_0` 같은 추가 interface가 몇 개 있으며, 이것을 NCCL communication에 포함하고 싶지 않다. 다음 report는 `[8]mlx5_bond_0:1/RoCE` interface가 잘못 포함되었음을 보여준다. 이는 거의 확실히 낮은 bandwidth를 유발한다.
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```
이 경우 다음 방식으로 제외할 수 있다.
```
export NCCL_IB_HCA=^mlx5_bond_0:1
```
또는 사용하려는 interface를 명확하게 나열할 수도 있다.
```
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

앞에서 말했듯이 IB interconnect를 사용하는 node에서 `ibstat` command를 실행하면 사용할 수 있는 IB interface가 표시된다.

NCCL은 최적 network interface를 자동 선택하려고 하므로, NCCL이 동작하지 않거나 느리게 동작할 때에만 위 작업을 수행하면 된다. 정상적인 상황에서는 NCCL이 바로 동작해야 하며 사용자가 특별한 작업을 할 필요가 없다.

또한 사용하는 cloud service에 따라 provider가 설정해야 하는 일련의 환경 변수를 제공할 가능성이 높다. 그중 일부를 잘못 설정하면 NCCL이 느리게 실행되거나 완전히 동작하지 않을 수 있다.

사용자가 만나는 또 다른 전형적인 문제는 cloud A에서 잘 동작하던 NCCL 설정을 cloud B에서 재사용하려고 할 때다. 보통 이런 설정은 직접 변환되지 않으며, 이전에 설정한 모든 환경 변수를 주의 깊게 제거하고 새 cloud 환경에 맞게 다시 올바르게 설정해야 한다. 같은 cloud service provider를 사용하더라도 서로 다른 instance type을 사용하면 이 문제가 발생할 수 있다. 일부 network 설정은 특정 instance용이어서 다른 곳에서는 제대로 동작하지 않을 수 있기 때문이다.

NCCL을 올바르게 설정했다고 생각되면, 다음 단계는 connection을 benchmark해 그것이 홍보된 speed와 맞는지, 대략 홍보 speed의 80%에 도달하는지 확인하는 것이다. benchmark chapter(https://github.com/stas00/ml-engineering/tree/master/network/benchmarks)로 이동하라.


## Docker container에서 NCCL 사용하기

* docker `run` command에 다음 추가 parameter를 넣어 충분한 resource를 제공한다. `–shm-size=1g –ulimit memlock=-1` 자세한 내용은 (https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#sharing-data)를 참고하라.
* privileged access: 때때로 docker `run` parameter에 `--privileged`를 추가해야 한다.
* Docker image에 올바른 package가 포함되어 있는지 확인한다. 예를 들어 IB를 사용한다면 적어도 `libibverbs1 librdmacm1`을 설치해야 한다.



## P2P 지원 여부를 확인하는 방법

때로 compute node의 GPU가 P2P access(Peer2Peer)를 지원하는지 알아야 한다. P2P를 비활성화하면 보통 node 내부 connection speed가 느려진다.

이 특정 8개 NVIDIA H100 node에서는 P2P가 지원되는 것을 볼 수 있다.

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

반면 이 특정 NVIDIA L4 GPU 2개에서는 P2P가 지원되지 않는다.

```
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

legend에서 볼 수 있듯이 `CNS`는 "chipset not supported"를 뜻한다.

high-end data center GPU를 사용한다면 이런 상황은 드물다. 하지만 일부 low-end data center GPU는 위 L4 예처럼 P2P를 지원하지 않을 수 있다.

consumer GPU의 경우 GPU가 지원되지 않는 이유는 여러 가지일 수 있다. 보통 IOMMU 및/또는 ACS 기능이 활성화되어 있기 때문이다. 때로는 단순히 driver version 문제일 수도 있다. 시간을 들여 검색하면, P2P를 지원하지 않아야 하는 GPU에서 P2P를 활성화하기 위해 driver를 hack한 사례를 찾을 수 있다. 예를 들어 이 4090 P2P support repository(https://github.com/tinygrad/open-gpu-kernel-modules)가 있다.

PCI access control services(ACS)가 활성화되어 있는지 확인하고 비활성화하려면 이 guide(https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#pci-access-control-services-acs)를 따르라.

IOMMU는 BIOS에서 비활성화할 수 있다.

torch로 특정 GPU 사이의 P2P 지원을 확인할 수도 있다. 여기서는 GPU 0과 1을 확인한다.

```python
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```

## NCCL call 횟수를 집계하는 방법

subsystem에 대한 NCCL debug log를 활성화한다. collective communication operation이다.
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

여러 node가 있는 Slurm 환경에서 작업한다면 rank 0에서만 이 작업을 수행하고 싶을 수 있다. 다음과 같다.
```
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

모든 log가 `main_log.txt`로 전송되었다고 가정하면, 다음 방식으로 각 collective communication call의 실행 횟수를 집계할 수 있다.
```
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

먼저 학습의 특정 stage를 분리하는 것이 좋을 수 있다. loading과 saving은 training iteration과 매우 다른 pattern을 갖기 때문이다.

그래서 나는 보통 iteration 하나를 먼저 slice한다. 예를 들어 각 iteration log가 `iteration: ...`로 시작한다면 먼저 다음을 수행한다.
```
csplit main_log.txt '/iteration: /' "{*}"
```
그런 다음 대응하는 iteration의 result file 중 하나를 분석한다. 기본적으로 그 이름은 `xx02`와 비슷하다.


## 유용한 NCCL debug 환경 변수

다음 환경 변수는 hang과 crash 같은 NCCL 관련 문제를 debug할 때 가장 유용하다. 전체 목록은 [여기](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)에서 찾을 수 있다.


### `NCCL_DEBUG`

이것은 네트워크 문제를 debug할 때 가장 흔히 사용하는 환경 변수다.

값:
- `VERSION` - program 시작 시 NCCL version을 출력한다.
- `WARN` - 어떤 NCCL call이 error를 낼 때 명확한 error message를 출력한다.
- `INFO` - debug 정보를 출력한다.
- `TRACE` - 각 call마다 replay 가능한 trace 정보를 출력한다.

예를 들면:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

이것은 NCCL 관련 debug 정보를 많이 출력한다. 보고된 문제가 있다면 그 정보를 online에서 검색할 수 있다.

`NCCL_DEBUG`를 사용할 때는 `NCCL_DEBUG_FILE`이 매우 유용할 것이다. 특히 여러 node를 사용할 때 정보량이 많기 때문이다.



### `NCCL_DEBUG_FILE`

`NCCL_DEBUG` 환경 변수를 사용할 때 모든 NCCL debug log output을 file로 redirect한다.

기본 output은 `stdout`이다. 여러 GPU를 사용할 때는 각 process의 debug 정보를 자체 log file에 저장하는 것이 매우 유용할 수 있으며, 다음처럼 할 수 있다.

```
NCCL_DEBUG_FILE=/path/to/nccl-log.%h.%p.txt
```

- `%h`는 hostname으로 대체된다.
- `%p`는 process PID로 대체된다.

이런 file 수백 개를 한 번에 분석해야 한다면, 다음 shortcut이 유용하다.

- grep으로 특정 match를 검색하고, match를 찾은 file name과 line number를 함께 출력한다.

```
grep -n "Init COMPLETE" nccl-log*
```

- 모든 nccl log file의 마지막 line을 표시하고, 뒤에 각 file 이름을 붙인다.

```
find . -name "nccl*" -exec sh -c 'echo "$(tail -1 "$1") ($1)"' _ {} \;
```



### `NCCL_DEBUG_SUBSYS`

`NCCL_DEBUG_SUBSYS`는 `NCCL_DEBUG`와 함께 사용되며, 후자에게 어떤 subsystem을 표시할지 알려준다. 일반적으로 이 변수를 지정할 필요는 없지만, 때로는 도와주는 developer가 output을 특정 subsystem으로 제한하라고 요청할 수 있다. 예를 들면 다음과 같다.

```
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
```

### `NCCL_P2P_DISABLE`

P2P communication을 비활성화한다. 예를 들어 NVLink가 있더라도 사용되지 않으므로 성능이 크게 낮아진다. 보통 이렇게 하고 싶지는 않지만, 긴급 상황에서는 debug 과정에서 유용할 수 있다.


### `NCCL_SOCKET_IFNAME`

network interface가 여러 개 있고 특정 하나를 선택해 사용하고 싶다면 이 option이 매우 유용하다.

기본적으로 NCCL은 가장 빠른 type의 interface를 사용하려고 하며, 보통 `ib`(InfiniBand)다.

하지만 Ethernet interface를 사용하고 싶다고 가정하면 다음 방식으로 기본 설정을 override할 수 있다.

```
NCCL_SOCKET_IFNAME=eth
```

이 환경 변수는 때때로 connection 문제를 debug하는 데 사용할 수 있다. 예를 들어 어떤 interface 하나가 firewall에 막혔고 다른 interface는 막히지 않아 시도할 수 있는 경우가 있다. 또는 어떤 문제가 network interface와 관련이 있는지 아니면 다른 원인인지 확실하지 않다면 다른 interface를 테스트해 문제가 network에서 오는지 배제하는 데 도움이 될 수 있다.

## 부록: torch-distributed-gpu-test.py

> 코드 위치: https://github.com/stas00/ml-engineering/blob/master/debug/torch-distributed-gpu-test.py

```python
#!/usr/bin/env python

'''
이것은 cluster 안의 모든 GPU(single node 또는 multi node)가 nccl을 통해 서로 통신하고 GPU memory를 allocate할 수 있는지 확인하는 `torch.distributed` diagnostic script이다. NUMA affinity 같은 다른 유용한 정보도 출력한다.

실행하려면 사용 상황에 맞게 process 수와 node 수만 조정하면 된다.

'''
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
'''

custom address와 port를 사용한다면 `--master_addr $MASTER_ADDR --master_port $MASTER_PORT`를 추가해야 할 수 있다.

rdzv API도 사용할 수 있다. `--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d`

`barrier` call에서 hang이 발생한다면 network 문제가 있을 수 있다. 다음 방법으로 debug해볼 수 있다.

'''
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
'''

이것은 내부에서 무슨 일이 일어나는지 알려줄 것이다.

이 script는 SLURM 환경에서 `srun`으로도 실행할 수 있다. 아래는 2개 node, 각 node 8개 GPU에서 실행하는 SLURM script다.

'''
#!/bin/bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # EDIT to the number of nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node for this script
#SBATCH --cpus-per-task=10           # EDIT this to how many cpu cores the node has
#SBATCH --gres=gpu:8                 # EDIT this if it's not an 8-GPUs node setup
#SBATCH --partition=dev              # EDIT to the desired partition name
#SBATCH --time 0:05:00               # 5 min should be enough
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
'''

launcher에 다음 내용을 추가해 모든 log에 `[hostname:rank] ` prefix를 자동으로 붙일 수도 있다. 예를 들어 `--master_addr` 뒤에 넣는다.

--role `hostname -s`: --tee 3


'''

import builtins
import fcntl
import os
import socket
import torch
import torch.distributed as dist

def print(*args, **kwargs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}:{local_rank}]"

try:
    # XXX: possibly change the dist timeout to something much shorter to get this script to fail
    # fast if there is a problem and not wait for the default 30min

    # test distributed
    dist.init_process_group("nccl")

    # global rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # reduction test
    t = torch.ones(1, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f"{gpu} Reduction op=sum result: {t.item()}")

    # test cuda is available and can allocate memory
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    print(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        print(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"device compute capabilities={torch.cuda.get_device_capability()}")
        print(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    print(f"{gpu} is broken (but it could also mean that it failed because another gpu didn't respond)")
    raise
```
