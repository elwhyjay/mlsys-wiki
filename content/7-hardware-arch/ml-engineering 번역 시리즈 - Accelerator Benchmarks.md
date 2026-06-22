> 제 강의 노트입니다. 많은 관심 부탁드립니다: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode
> 이 문서의 출처: https://github.com/stas00/ml-engineering . 이 문서는 "Maximum Achievable Matmul FLOPS(MAMF) Finder"라는 benchmark tool을 소개합니다. 이 tool의 주요 목적은 여러 크기의 matrix multiplication operation을 테스트해 특정 accelerator(NVIDIA, AMD, Intel 등)가 실제로 달성할 수 있는 최대 TFLOPS performance를 찾는 것입니다. 이 tool은 가치가 큽니다. hardware vendor가 theoretical TFLOPS를 공개하더라도 실제 달성 가능한 performance는 보통 theoretical value보다 낮기 때문입니다. 이 tool을 통해 developer는 더 현실적인 performance baseline을 얻고, 자신의 optimization 효과를 더 잘 평가할 수 있습니다. 문서는 quick test, detailed search, specific shape test 등 여러 사용 시나리오를 포함해 tool 사용법을 자세히 소개하고, A100, MI300X, H100 등 여러 accelerator에 대한 optimization suggestion도 제공합니다. 또한 다양한 hardware architecture를 지원하는 전체 Python implementation code도 포함합니다.

# Accelerator Benchmarks

## Maximum Achievable Matmul FLOPS Finder

Maximum Achievable Matmul FLOPS(MAMF) benchmark: mamf-finder.py(https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py)

자세한 discussion과 여러 accelerator data는 Maximum Achievable FLOPS(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-flops)를 참고하세요.

일부 accelerator manufacturer는 theoretical TFLOPS를 공개하지만, 이는 보통 달성할 수 없습니다. 따라서 software를 optimize하려 할 때 현실적인 performance standard가 없습니다. Model FLOPS Utilization(MFU) metric은 실제 달성한 TFLOPS와 theoretical TFLOPS의 ratio를 측정합니다. 보통 MFU가 약 50%에 도달하면 성공으로 여깁니다. 하지만 이것은 진짜 달성 가능한 throughput에서 얼마나 떨어져 있는지는 알려 주지 않습니다.

이 benchmark는 다양한 large matrix multiplication shape를 scan하고, 기록한 최고 achievable TFLOPS를 report합니다. Transformer training과 일부 inference workload는 주로 large matrix multiplication operation이 지배하므로, 각 accelerator에서 측정한 best matrix multiplication TFLOPS를 Maximum Achievable Matmul FLOPS(MAMF)의 rough estimate로 안전하게 사용할 수 있습니다. 이제 기존 MFU 대신 Model Achievable Matmul FLOPS Utilization(MAMFU)을 사용할 수 있습니다.

따라서 이제 training이나 inference에서 측정한 TFLOPS를 현실적인 숫자와 비교할 수 있습니다. 이제 100%에 더 가까워지므로 언제 optimization을 멈춰야 하는지 알기 더 쉽습니다.

현재 지원하는 high-end architecture:

- NVIDIA: V100, A100, H100, ...
- AMD: MI250, MI300X, ...
- Intel Gaudi2+

Fairness note:

- 각 new accelerator를 black box로 보고 best matrix multiplication TFLOPS를 detect하는 더 좋고 효과적인 방법을 찾는다면, 개선 사항과 생성된 log file을 포함한 PR을 보내 주세요.
- 또한 이 benchmark가 best result를 보여주기 위해 special condition, 예를 들어 kernel setting 같은 조건에서 실행되어야 한다는 것을 알고 있다면, 해당 special instruction을 추가하는 PR을 제출해 주세요. 예를 들어 AMD MI300X의 경우 `numa_balancing`을 disable하면 도움이 된다고 들었습니다.

### 특정 architecture 주의 사항

benchmark를 실행하기 전에 best result를 얻기 위해 아래 special setup instruction을 따르세요.

**MI300x**:

더 나은 performance를 위해 `numa_balancing`을 끕니다.

```
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

### Usage examples

아래 range에서 `N`은 reduce dimension이며, `(MxN)*(NxK)=(MxK)`가 됩니다. 우리는 measured highest TFLOPS를 기록한 MxNxK shape를 출력합니다.

기본적으로 각 shape에 대해 50 warmup iteration과 100 measurement iteration을 사용하고, 평균이 아니라 fastest result를 선택합니다. `--num_warmup_iterations`와 `--num_iterations` argument로 각각 iteration 수를 바꿀 수 있습니다.

여기서는 `torch.mm(MxN,NxK) -> MxK`를 실행합니다.

1. Quick run(1분 이내) - maximum achievable result의 80-90%에 도달할 수 있어야 합니다. 빠른 시도에는 적합하지만 high-precision measurement에는 충분하지 않습니다.

```
./mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

2. 더 exhaustive한 search(더 오래 걸림) - 충분히 오래 실행한 뒤 Ctrl-C로 종료해도 그 시점까지의 best result를 얻을 수 있습니다.

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

3. 매우 긴 exhaustive search(며칠 걸릴 수 있음) - 충분히 오래 실행한 뒤 Ctrl-C로 종료하면 그 시점까지의 best result를 얻을 수 있습니다.

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

4. training에서 사용하는 특정 shape를 측정하려면 range가 아니라 exact shape를 사용하세요. 예를 들어 1024x1024x1024를 측정하고 싶다면 다음을 실행할 수 있습니다.

```
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

5. Accelerator-specific range search suggestion

하지만 accelerator마다 best TFLOPS에 도달할 수 있는 shape range가 다른 듯하므로, 모든 accelerator에 적용되는 range를 제안하기는 어렵습니다. 대신 여기서는 experiment와 contributor suggestion에 기반한 몇 가지 제안을 제공합니다.

- **A100** + **MI300X**

```
./mamf-finder.py --m_range 0 5376 256 --n_range 0 5376 256 --k_range 0 5376 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

- **H100**

```
./mamf-finder.py --m_range 0 20480 256 --n_range 0 20480 256 --k_range 0 20480 256 --output_file=$(date +"%Y-%m-%d-%H:%M:%S").txt
```

To understand better which shapes give the highest matmul FLOPS for a particular accelerator, see Vector and matrix size divisibility(../../../training/performance/README.md#vector-and-matrix-size-divisibility).

### Results

제가 현재 수집한 measurement result는 Maximum Achievable Matmul FLOPS comparison table(https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#maximum-achievable-matmul-flops-comparison-table)에서 찾을 수 있습니다. 특정 accelerator에 접근할 수 있으면 제가 직접 benchmark를 실행하고, 접근할 수 없을 때는 열정적인 contributor들이 시간을 들여 data를 얻었습니다. 따라서 이 contributor들(https://github.com/stas00/ml-engineering/blob/master/contributors.md)에게 매우 감사드립니다.

## `mamf-finder.py` code analysis

```python
#!/usr/bin/env python

"""

This is Maximum Achievable Matmul FLOPS (MAMF) Finder

For discussion and multiple important nuances please refer to
https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks#maximum-achievable-matmul-flops-finder

Credits:
- Parts of this benchmark have been derived from https://github.com/EleutherAI/cookbook/tree/main/benchmarks/sizing (highly recommended!)
- Imtiaz Sajwani: HPU porting

"""

from pathlib import Path

import argparse
import datetime
import numpy as np
import os
import platform
import re
import shlex
import signal
import sys
import time
import torch


# HPU 관련 module import 시도
has_hpu = False
try:
    import habana_frameworks.torch as ht
    if torch.hpu.is_available():
        has_hpu = True
except ModuleNotFoundError:
    pass

# 현재 file directory의 absolute path를 얻는다.
file_dir = os.path.abspath(os.path.dirname(__file__))



### Architecture-specific helper classes ###

class Arch:
    def __init__(self):
        self.arch = "unknown"

    def __repr__(self):
        return self.arch

class CUDAArch(Arch):
    """ Shared by CUDA and ROCm: NVIDIA + AMD """
    def __init__(self):
        if torch.version.hip is not None:
            self.arch = "rocm"
        else:
            self.arch = "cuda"

    def device(self):
        return torch.device('cuda:0')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.cuda.get_device_properties(device)

    def compute_info(self):
        if self.arch == "rocm":
            return f"hip={torch.version.hip}, cuda={torch.version.cuda}"
        else:
            return f"cuda={torch.version.cuda}"

    def event(self, enable_timing=True):
        return torch.cuda.Event(enable_timing)

    def synchronize(self):
        torch.cuda.synchronize()

class HPUArch(Arch):
    """ Intel Gaudi* """
    def __init__(self):
        self.arch = "hpu"

    def device(self):
        return torch.device('hpu')

    def name(self):
        return self.arch

    def device_info(self):
        return torch.hpu.get_device_properties(device)

    def compute_info(self):
        return f"hpu={torch.version.hpu}"

    def event(self, enable_timing=True):
        return ht.hpu.Event(enable_timing)

    def synchronize(self):
        ht.hpu.synchronize()


def get_accelerator_arch():
    """
    Returns: CUDAArch or HPUArch object
    """
    # cuda / rocm
    if torch.cuda.is_available():
        return CUDAArch()

    # hpu
    if has_hpu:
        return HPUArch()

    raise ValueError("currently only cuda, rocm and hpu are supported")

# accelerator architecture 얻기
arch = get_accelerator_arch()



### Helper classes ###

class Tee(object):
    def __init__(self, filename, verbose):
        # output file directory가 없으면 생성
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        if self.verbose:
            self.stdout.write(message)
        # console의 `\r`와 `033\[K`를 replace. log file에는 필요 없다.
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()


def print_benchmark_header(dtype, device, notes="None"):
    """benchmark header information 출력"""

    device_info = arch.device_info()
    compute_info = arch.compute_info()

    print(f"""
Benchmark started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}

** Command line:
{sys.executable} {" ".join(map(shlex.quote, sys.argv))}

** Data type: {dtype}

** Platform/device info:
{" ".join(platform.uname())}
{device_info}

** Key software versions:
torch={torch.__version__}
{compute_info}

** Additional notes:
{notes}

{"-" * 80}

""")

# basic GEMM benchmark
def benchmark_mm(m, n, k, dtype, device, num_iterations, num_warmup_iterations):
    start = arch.event(enable_timing=True)
    end = arch.event(enable_timing=True)

    # random matrix 생성
    A = torch.randn(m, n, dtype=dtype, device=device)
    B = torch.randn(n, k, dtype=dtype, device=device)
    C = torch.empty(m, k, dtype=dtype, device=device)

    times = np.zeros(num_iterations+num_warmup_iterations)
    for i in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        arch.synchronize()
        times[i] = start.elapsed_time(end)
    times = times[num_warmup_iterations:]  # warmup iteration time 제거
    elapsed_time = np.amin(times)/1000  # fastest time을 취하고 second로 변환
    tflops = (2 * m * n * k) / (elapsed_time * 10**12)  # TFLOPS 계산
    return tflops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # command-line arguments 설정
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument("--m", nargs="+", type=int, help='first dimension of GEMM, arbitrary number of values')
    m_group.add_argument("--m_range", nargs='+', type=int, help="first dimension of GEMM, [start, stop, step]")

    n_group = parser.add_mutually_exclusive_group(required=True)
    n_group.add_argument("--n", nargs="*", type=int, help='shared dimension of GEMM, arbitrary number of values')
    n_group.add_argument("--n_range", nargs='+', type=int, help="shared dimension of GEMM, [start, stop, step]")

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument("--k", nargs="*", type=int, help='last dimension of GEMM, arbitrary number of values')
    k_group.add_argument("--k_range", nargs='+', type=int, help="last dimension of GEMM, [start, stop, step]")

    parser.add_argument("--num_iterations", type=int, default=100, help='iterations for benchmarking each GEMM')
    parser.add_argument("--num_warmup_iterations", type=int, default=50, help='warmup iterations')
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device to run benchmark")
    parser.add_argument("--output_file", type=str, default=f"{file_dir}/results/mm.out")
    parser.add_argument("--notes", type=str, default="", help="benchmark-specific notes added to output file header")
    parser.add_argument("--verbose", default=True, action=argparse.BooleanOptionalAction, help='also output to stdout and output_file?')
    args = parser.parse_args()

    m = args.m
    n = args.n
    k = args.k

    dtype = torch.bfloat16
    device = arch.device()

    # range argument 처리
    if m is None:
        start, stop, step = args.m_range
        if start == 0:  # dimension cannot be 0
            start = step
        m = np.arange(start, stop, step)
    if n is None:
        start, stop, step = args.n_range
        if start == 0:  # dimension cannot be 0
            start = step
        n = np.arange(start, stop, step)
    if k is None:
        start, stop, step = args.k_range
        if start == 0:  # dimension cannot be 0
            start = step
        k = np.arange(start, stop, step)

    sys.stdout = Tee(args.output_file, args.verbose)
    print_benchmark_header(dtype, device, args.notes)

    # interrupt 시에도 best result를 report하기 위함
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    best_tflops = 0
    best_config = ""
    num_shapes = 0
    start_time = time.time()

    def finish():
        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print("", end="\033[K")
        print(f"Best result is {best_tflops:.1f}TFLOPS @ {best_config} (tried {num_shapes} shapes)")
        print(f"Elapsed time: {time_str}")

    # Note: for MI300X, transposed version seems to work better

    # benchmark할 모든 size를 순회
    for M in m:
        for N in n:
            for K in k:
                num_shapes += 1
                tflops = benchmark_mm(M, N, K, dtype, device, args.num_iterations, args.num_warmup_iterations)
                cur_config = f"{M}x{N}x{K}"
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_config = f"{M}x{N}x{K} (MxNxK)"
                print(f"{num_shapes:>6} | {tflops:6.1f} TFLOPS @ {cur_config:<20} | best: {best_tflops:6.1f} TFLOPS @ {best_config}", end="\r")
    finish()

```
