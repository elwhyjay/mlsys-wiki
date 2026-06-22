# PyTorch TORCH.COMPILE 사용 가이드

> 원문: https://zhuanlan.zhihu.com/p/620163218

## Introduction

torch.compile은 PyTorch 코드를 가속하는 최신 방법입니다! torch.compile은 JIT 방식으로 PyTorch 코드를 최적화된 커널로 컴파일하여 더 빠르게 실행하며, 대부분의 경우 한 줄의 코드만 수정하면 됩니다.

본 글은 torch.compile의 기본 사용법을 소개하고, TorchScript나 FX Tracing 같은 기존 PyTorch 컴파일러 솔루션 대비 torch.compile의 장점을 보여줍니다.

## Basic Usage

torch.compile은 PyTorch 2.0 이상 설치 후 사용 가능하며, GPU에서 실행하려면 Triton 설치가 필요합니다.

```bash
pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu117"
```

torch.compile은 임의의 Python 함수를 전달하면 최적화된 함수를 반환하여 원본 함수를 대체합니다:

```python
import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
```

또는 Python 함수 앞에 `@torch.compile` 데코레이터를 추가할 수 있습니다:

```python
@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))
```

torch.compile은 `torch.nn.Module` 인스턴스도 직접 최적화할 수 있습니다:

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule()
opt_mod = torch.compile(mod)
print(opt_mod(torch.randn(10, 100)))
```

## Demonstrating Speedups

torch.compile로 실제 모델을 가속하는 방법을 시연합니다. 표준 Eager 모드와 ResNet-18 모델의 추론/훈련 성능을 비교합니다.

```python
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import resnet18
def init_model():
    return resnet18().to(torch.float32).cuda()
```

추론 비교:

```python
def evaluate(mod, inp):
    return mod(inp)

model = init_model()
import torch._dynamo
torch._dynamo.reset()

evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")

inp = generate_data(16)[0]
print("eager:", timed(lambda: evaluate(model, inp))[1])
print("compile:", timed(lambda: evaluate_opt(model, inp))[1])
```

torch.compile은 처음에는 모델을 최적화된 커널로 컴파일해야 하므로 eager보다 더 오래 걸립니다. 하지만 모델 구조가 변하지 않으면 재컴파일이 필요 없으므로, 여러 번 실행하면 현저한 성능 향상을 볼 수 있습니다.

가속은 주로 Python 오버헤드와 GPU 읽기/쓰기 감소에서 비롯되므로, 관찰되는 가속은 모델 아키텍처와 배치 크기 등의 요인에 따라 달라질 수 있습니다.

mode 파라미터에 따라 다른 가속 결과를 볼 수 있습니다. torch.compile은 세 가지 모드를 지원합니다:
- **default**: 과도한 컴파일 시간이나 추가 메모리 사용 없이 효율적으로 컴파일하는 프리셋
- **reduce-overhead**: 프레임워크 오버헤드를 크게 줄이지만 약간의 추가 메모리를 소비
- **max-autotune**: 오래 컴파일하여 가능한 가장 빠른 코드 생성 시도

훈련 비교도 유사하게 torch.compile이 첫 번째 반복에서는 더 오래 걸리지만, 이후 반복에서는 eager 대비 현저한 가속을 보여줍니다.

## Comparison to TorchScript and FX Tracing

torch.compile의 장점은 임의의 Python 코드를 처리할 수 있으며 기존 코드에 최소한의 변경만 필요하다는 것입니다.

### 데이터 의존적 제어 흐름 처리

```python
def f1(x, y):
    if x.sum() < 0:
        return -y
    return y
```

- **TorchScript tracing**: f1을 추적하면 실제 제어 흐름 경로만 추적하여 잘못된 결과를 초래합니다.
- **FX tracing**: 데이터 의존적 제어 흐름이 있으면 에러가 발생합니다.
- **torch.compile**: 데이터 의존적 제어 흐름을 올바르게 처리합니다.

```python
torch._dynamo.reset()
compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))   # True
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))  # True
```

TorchScript script는 데이터 의존적 제어 흐름을 처리할 수 있지만, 상당한 코드 변경이 필요하고 지원되지 않는 Python을 사용하면 에러가 발생합니다.

### 타입 어노테이션 불필요

```python
def f2(x, y):
    return x + y

inp1 = torch.randn(5, 5)
inp2 = 3

# TorchScript은 타입 추론 실패로 에러 발생
# torch.compile은 문제없이 처리
compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))  # True
```

### 비-PyTorch 함수 지원

```python
import scipy
def f3(x):
    x = x * 2
    x = scipy.fft.dct(x.numpy())
    x = torch.from_numpy(x)
    x = x * 2
    return x
```

- **TorchScript tracing**: 비-PyTorch 함수 호출 결과를 상수로 취급하여 잘못된 결과 초래
- **TorchScript script / FX tracing**: 비-PyTorch 함수 호출을 허용하지 않아 에러 발생
- **torch.compile**: 비-PyTorch 함수 호출을 쉽게 처리

```python
compile_f3 = torch.compile(f3)
print("compile 3:", test_fns(f3, compile_f3, (inp2,)))  # True
```

## TorchDynamo and FX Graphs

torch.compile의 중요 컴포넌트인 TorchDynamo는 임의의 Python 코드를 즉시 FX Graph로 컴파일하며, 런타임에 Python 바이트코드를 분석하고 PyTorch 연산 호출을 감지하여 FX Graph를 추출합니다.

TorchInductor는 FX Graph를 최적화된 커널로 추가 컴파일합니다. TorchDynamo는 다양한 백엔드를 허용하므로, 커스텀 백엔드를 만들어 FX Graph를 출력하고 확인할 수 있습니다.

```python
from typing import List
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

torch._dynamo.reset()
opt_model = torch.compile(init_model(), backend=custom_backend)
opt_model(generate_data(16)[0])
```

커스텀 백엔드로 TorchDynamo가 데이터 의존적 제어 흐름을 어떻게 처리하는지 확인할 수 있습니다:

```python
def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

opt_bar = torch.compile(bar, backend=custom_backend)
inp1 = torch.randn(10)
inp2 = torch.randn(10)
opt_bar(inp1, inp2)
opt_bar(inp1, -inp2)
```

출력은 TorchDynamo가 3개의 서로 다른 FX Graph를 추출했음을 보여줍니다:
1. `x = a / (torch.abs(a) + 1)`
2. `b = b * -1; return x * b`
3. `return x * b`

TorchDynamo가 지원하지 않는 Python 기능(예: 데이터 의존적 제어 흐름)을 만나면 계산 그래프를 break하고, 기본 Python 인터프리터가 지원하지 않는 코드를 처리하게 한 후 그래프 캡처를 재개합니다.

이것이 TorchDynamo와 이전 PyTorch 컴파일러 솔루션의 주요 차이점입니다. 이전 솔루션은 지원하지 않는 기능을 만나면 에러를 발생시키거나 조용히 실패했지만, TorchDynamo는 계산 그래프를 break합니다.

`torch._dynamo.explain`으로 TorchDynamo가 어디서 Graph를 break했는지 확인할 수 있습니다:

```python
torch._dynamo.reset()
explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose = torch._dynamo.explain(
    bar, torch.randn(10), torch.randn(10)
)
print(explanation_verbose)
```

가속을 최대화하려면 그래프 break를 제한해야 합니다. `fullgraph=True` 파라미터를 사용하면 첫 번째 그래프 break 시 에러를 발생시킬 수 있습니다:

```python
opt_bar = torch.compile(bar, fullgraph=True)
try:
    opt_bar(torch.randn(10), torch.randn(10))
except:
    tb.print_exc()
```

TorchDynamo가 FX Graph만 출력하고 export하려면 `torch._dynamo.export`를 사용할 수 있으며, `fullgraph=True`와 동일한 효과로 Graph break 시 에러가 발생합니다.

## Conclusion

본 튜토리얼에서는 torch.compile의 기본 사용법을 소개하고, Eager 모드 대비 가속을 시연하며, 이전 PyTorch 컴파일러 솔루션과 비교하고, TorchDynamo와 FX Graph의 상호작용을 간략히 살펴보았습니다.
