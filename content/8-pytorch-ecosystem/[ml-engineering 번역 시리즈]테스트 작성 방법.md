> 강의 노트입니다. 팔로우 환영합니다: https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 이 문서의 출처: https://github.com/stas00/ml-engineering

# 테스트 작성 및 실행

참고: 이 문서의 일부 내용은 testing_utils.py(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py)에서 제공하는 기능을 다루며, 해당 기능은 대부분 필자가 HuggingFace에서 근무하는 동안 개발한 것이다.

이 문서는 `pytest`와 `unittest`의 기능을 다루며, 두 가지를 함께 사용하는 방법을 보여준다.


## 테스트 실행

### 모든 테스트 실행

```console
pytest
```
다음 별칭을 사용한다:
```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

이 명령은 pytest에게 다음을 지시한다:

- 경고 비활성화
- `--instafail` 실패 발생 시 마지막이 아닌 즉시 표시
- `-rA` 간단한 테스트 요약 정보 생성

다음 패키지를 설치해야 한다:
```
pip install pytest-instafail
```


### 모든 테스트 목록 가져오기

테스트 스위트의 모든 테스트 표시:

```bash
pytest --collect-only -q
```

특정 테스트 파일의 모든 테스트 표시:

```bash
pytest tests/test_optimization.py --collect-only -q
```

다음 별칭을 사용한다:
```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 단일 테스트 모듈 실행

단일 테스트 모듈 실행:

```bash
pytest tests/utils/test_logging.py
```

### 특정 테스트 실행

`unittest`를 사용할 때 특정 서브 테스트를 실행하려면 이 테스트를 포함하는 `unittest` 클래스 이름을 알아야 한다. 예를 들어:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

여기서:

- `tests/test_optimization.py` - 테스트를 포함하는 파일
- `OptimizationTest` - 테스트 클래스 이름
- `test_adam_w` - 구체적인 테스트 함수 이름

파일에 여러 클래스가 있는 경우 지정한 클래스의 테스트만 실행할 수 있다. 예를 들어:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

해당 클래스의 모든 테스트를 실행한다.

앞서 언급했듯이 다음 명령을 실행하여 `OptimizationTest` 클래스에 포함된 모든 테스트를 확인할 수 있다:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

키워드 표현식으로 테스트를 실행할 수 있다.

`adam`을 포함하는 테스트만 실행:

```bash
pytest -k adam tests/test_optimization.py
```

논리 `and`와 `or`를 사용하여 모든 키워드가 일치하는지 또는 하나만 일치하는지를 지정할 수 있다. `not`을 사용하여 부정할 수 있다.

`adam`을 포함하지 않는 모든 테스트 실행:

```bash
pytest -k "not adam" tests/test_optimization.py
```

두 패턴을 조합할 수도 있다:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

예를 들어 `test_adafactor`와 `test_adam_w`를 실행하려면:

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

여기서는 두 항목 중 하나라도 키워드와 일치하면 포함하기 위해 `or`를 사용한다는 점에 유의하라.

두 패턴을 모두 포함하는 테스트만 포함하려면 `and`를 사용한다:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 수정된 테스트만 실행

pytest-picked(https://github.com/anapaulagomes/pytest-picked)를 사용하면 스테이징되지 않은 파일 또는 현재 브랜치(Git 기준)와 관련된 테스트를 실행할 수 있다. 이는 변경 사항이 무언가를 깨뜨렸는지 빠르게 테스트하는 좋은 방법으로, 건드리지 않은 파일과 관련된 테스트는 실행하지 않는다.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

수정되었지만 아직 커밋되지 않은 파일과 폴더에서 모든 테스트가 실행된다.

### 실패한 테스트 자동 재실행


pytest-xdist(https://github.com/pytest-dev/pytest-xdist)는 실패한 모든 테스트를 감지한 다음 파일을 수정할 때까지 기다렸다가 수정하면 해당 실패한 테스트를 통과할 때까지 지속적으로 재실행하는 매우 유용한 기능을 제공한다. 그러면 수정 후 pytest를 재시작할 필요가 없다. 이 과정은 모든 테스트가 통과할 때까지 반복되며, 그 후 전체 실행이 다시 수행된다.

```bash
pip install pytest-xdist
```

모드 진입: `pytest -f` 또는 `pytest --looponfail`

파일 변경은 `looponfailroots` 루트 디렉토리와 해당 내용(재귀적으로)을 살펴봄으로써 감지된다.
기본값이 작동하지 않으면 `setup.cfg`에서 구성 옵션을 설정하여 프로젝트에서 변경할 수 있다:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

또는 `pytest.ini`/`tox.ini` 파일:

```ini
[pytest]
looponfailroots = transformers tests
```

이렇게 하면 ini 파일 디렉토리를 기준으로 해당 디렉토리에서만 파일 변경을 찾는다.

pytest-watch(https://github.com/joeyespo/pytest-watch)는 이 기능의 또 다른 구현이다.


### 테스트 모듈 건너뛰기

몇 가지를 제외한 모든 테스트 모듈을 실행하려는 경우 실행 시 테스트 목록을 제공하여 제외할 수 있다. 예를 들어 `test_modeling_*.py`를 포함하지 않는 모든 테스트 실행:

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 상태 지우기

CI 빌드 및 격리가 중요한 경우(속도) 캐시를 지워야 한다:

```bash
pytest --cache-clear tests
```

### 병렬로 테스트 실행

앞서 언급했듯이 `make test`는 `pytest-xdist` 플러그인을 통해 병렬로 테스트를 실행한다(`-n X` 인수, 예: `-n 2`는 2개의 병렬 작업 실행).

`pytest-xdist`의 `--dist=` 옵션을 사용하면 테스트를 그룹화하는 방법을 제어할 수 있다. `--dist=loadfile`은 동일한 파일에 있는 테스트를 동일한 프로세스에 배치한다.

테스트 실행 순서가 다르고 예측할 수 없기 때문에 `pytest-xdist`로 테스트 스위트를 실행하면 실패(즉 감지되지 않은 결합된 테스트가 있음을 의미)가 발생하면 pytest-replay(https://github.com/ESSS/pytest-replay)를 사용하여 동일한 순서로 테스트를 재생하면 실패 시퀀스를 최소화하는 데 도움이 될 것이다.

### 테스트 순서 및 반복

테스트를 순서대로, 무작위로 또는 그룹으로 여러 번 반복하는 것은 잠재적인 상호 의존성과 상태 관련 오류(tear down)를 확인하는 좋은 관행이다. 또한 단순한 여러 번 반복은 딥러닝 무작위성에 의해 가려진 일부 문제를 발견하는 데도 도움이 된다.

#### 테스트 반복

- pytest-flakefinder(https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

각 테스트를 여러 번(기본 50번) 실행:

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

참고: 이 플러그인은 `pytest-xdist`의 `-n` 플래그와 함께 작동하지 않는다.

참고: 또 다른 플러그인 `pytest-repeat`가 있지만 `unittest`와 함께 작동하지 않는다.


#### 무작위로 테스트 실행

```bash
pip install pytest-random-order
```

중요: `pytest-random-order`를 설치하면 구성 변경이나 명령줄 옵션 없이 자동으로 테스트가 무작위화된다.

앞서 언급했듯이 이를 통해 결합된 테스트를 감지할 수 있다 - 즉 한 테스트의 상태가 다른 테스트의 상태에 영향을 미치는 경우. `pytest-random-order`가 설치되면 해당 세션에 사용된 랜덤 시드가 출력된다, 예를 들어:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

이 경우 특정 테스트 시퀀스가 실패하면 동일한 시드를 추가하여 재현할 수 있다, 예를 들어:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```
완전히 동일한 테스트 목록(또는 목록이 전혀 없는 경우)을 사용하는 경우에만 완전히 동일한 순서를 재현한다. 목록을 수동으로 좁히기 시작하면 시드에 더 이상 의존할 수 없으므로 실패한 정확한 순서대로 수동으로 나열하고 `--random-order-bucket=none`을 사용하여 pytest가 무작위화하지 않도록 해야 한다, 예를 들어:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

모든 테스트의 무작위화를 비활성화하려면:

```bash
pytest --random-order-bucket=none
```

기본적으로 `--random-order-bucket=module`이 적용되어 모듈 수준에서 파일을 섞는다. `class`, `package`, `global`, `none` 수준에서도 섞을 수 있다. 자세한 내용은 해당 문서(https://github.com/jbasko/pytest-random-order)를 참조하라.

또 다른 무작위화 대안은 `pytest-randomly`(https://github.com/pytest-dev/pytest-randomly)이다. 이 모듈은 매우 유사한 기능/인터페이스를 가지고 있지만 `pytest-random-order`의 버킷 모드가 없다. 설치 후 강제로 자신을 적용하는 문제도 있다.

### 외관 변경

#### pytest-sugar

pytest-sugar(https://github.com/Frozenball/pytest-sugar)는 외관을 개선하고 진행 바를 추가하며 실패한 테스트와 단언을 즉시 표시하는 플러그인이다. 설치 후 자동으로 활성화된다.

```bash
pip install pytest-sugar
```

이 플러그인 없이 테스트를 실행하려면:

```bash
pytest -p no:sugar
```

또는 제거한다.



#### 각 서브 테스트 이름과 진행 상황 보고

단일 또는 테스트 그룹에 대해 `pytest` 사용 (`pip install pytest-pspec` 이후):

```bash
pytest --pspec tests/test_optimization.py
```

#### 실패한 테스트 즉시 표시

pytest-instafail(https://github.com/pytest-dev/pytest-instafail)은 테스트 세션이 끝날 때까지 기다리지 않고 실패와 오류를 표시한다.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### GPU 사용 여부

GPU 설정에서 CPU 모드를 테스트하려면 `CUDA_VISIBLE_DEVICES=""`를 추가한다:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

또는 여러 개의 GPU가 있는 경우 `pytest`로 사용할 GPU를 지정할 수 있다. 예를 들어 두 번째 GPU만 사용하려면(GPU `0`과 `1`이 있는 경우):

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

이는 다른 GPU에서 다른 작업을 실행하려는 경우 매우 유용하다.

일부 테스트는 CPU에서만 실행해야 하고, 일부는 CPU 또는 GPU 또는 TPU에서, 일부는 여러 GPU에서 실행해야 한다. 테스트의 CPU/GPU/TPU 요구사항을 설정하기 위해 다음 skip 데코레이터가 사용된다:

- `require_torch` - 이 테스트는 torch에서만 실행됨
- `require_torch_gpu` - `require_torch`와 동일하지만 최소 1개의 GPU 필요
- `require_torch_multi_gpu` - `require_torch`와 동일하지만 최소 2개의 GPU 필요
- `require_torch_non_multi_gpu` - `require_torch`와 동일하지만 0 또는 1개의 GPU 필요
- `require_torch_up_to_2_gpus` - `require_torch`와 동일하지만 0, 1 또는 2개의 GPU 필요
- `require_torch_tpu` - `require_torch`와 동일하지만 최소 1개의 TPU 필요

GPU 요구사항을 다음 표로 정리한다:


| n gpus | decorator                      |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


예를 들어 2개 이상의 GPU가 있고 pytorch가 설치된 경우에만 실행되는 테스트:

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

이 데코레이터들은 중첩할 수 있다:

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

`@parametrized`와 같은 일부 데코레이터는 테스트 이름을 재작성하기 때문에 `@require_*` skip 데코레이터를 마지막에 나열해야 제대로 작동한다. 올바른 사용 예시:

```python no-style
from testing_utils import require_torch_multi_gpu
from parameterized import parameterized

@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

이 순서 문제는 `@pytest.mark.parametrize`에는 존재하지 않으며, 처음이나 마지막에 배치해도 여전히 작동한다. 단, `unittest`가 아닌 테스트에만 적용된다.

테스트에서:

- 사용 가능한 GPU 수:

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 분산 훈련

`pytest`는 분산 훈련을 직접 처리할 수 없다. 그렇게 시도하면 - 서브 프로세스가 올바른 일을 하지 않고 최종적으로 자신이 `pytest`라고 생각하여 루프에서 테스트 스위트를 실행하기 시작한다. 그러나 일반 프로세스를 시작하고 여러 워커 프로세스를 시작하여 IO 파이프를 관리하면 작동한다.

이를 사용하는 일부 테스트:

- test_trainer_distributed.py(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- test_deepspeed.py(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

실행 지점으로 바로 이동하려면 해당 테스트에서 `execute_subprocess_async` 호출을 검색하면 `testing_utils.py`(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py)에서 찾을 수 있다.

이 테스트를 보려면 최소 2개의 GPU가 필요하다:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

(`RUN_SLOW`는 HF Transformers가 무거운 테스트를 일반적으로 건너뛰는 데 사용하는 특수 데코레이터이다)

### 출력 캡처

테스트 실행 중 `stdout`과 `stderr`로 전송된 모든 출력은 캡처된다. 테스트나 설정 메서드가 실패하면 해당 캡처된 출력이 실패 traceback과 함께 표시된다.

출력 캡처를 비활성화하고 `stdout`과 `stderr`를 정상적으로 받으려면 `-s` 또는 `--capture=no`를 사용한다:

```bash
pytest -s tests/utils/test_logging.py
```

테스트 결과를 JUnit 형식으로 출력하려면:

```bash
py.test tests --junitxml=result.xml
```

### 색상 제어

색상 없이(예: 흰 배경에 노란색이 읽기 어려운 경우):

```bash
pytest --color=no tests/utils/test_logging.py
```

### 온라인 붙여넣기 서비스로 테스트 보고서 전송

각 테스트 실패에 대한 URL 생성:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

이렇게 하면 테스트 실행 정보를 원격 붙여넣기 서비스에 제출하고 각 실패에 대한 URL이 제공된다. 평소처럼 테스트를 선택하거나 특정 실패만 전송하려면 -x 인수를 추가할 수 있다.

전체 테스트 세션 로그에 대한 URL 생성:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## 테스트 작성

대부분의 경우 동일한 테스트 스위트에서 `pytest`와 `unittest`를 함께 사용하는 것이 정상적으로 작동한다. 어떤 기능이 지원되는지는 여기(https://docs.pytest.org/en/stable/unittest.html)에서 읽을 수 있지만, 기억해야 할 중요한 점은 대부분의 `pytest` fixture가 작동하지 않는다는 것이다. 매개변수화도 마찬가지이지만 우리는 유사하게 작동하는 `parameterized` 모듈을 사용한다.

### 매개변수화

일반적으로 동일한 테스트를 여러 번 실행하되 다른 매개변수를 사용해야 하는 경우가 있다. 이는 테스트 내부에서 수행할 수 있지만 그러면 단일 매개변수 집합에 대해서만 테스트를 실행할 수 없다.

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

기본적으로 이 테스트는 3번 실행되며 `test_floor`의 마지막 3개 인수에 매개변수 목록의 해당 인수 값이 할당된다.

`negative`와 `integer` 두 매개변수 집합의 테스트만 실행하려면:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

또는 `negative`를 제외한 모든 서브 테스트 실행:

```bash
pytest -k "not negative" tests/test_mytest.py
```

방금 언급한 `-k` 필터를 사용하는 것 외에도 각 서브 테스트의 정확한 이름을 찾아 그 중 하나 또는 모두를 실행할 수 있다.

```bash
pytest test_this1.py --collect-only -q
```

목록이 표시된다:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

이제 2개의 특정 서브 테스트만 실행할 수 있다:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

parameterized 모듈(https://pypi.org/project/parameterized/)은 `unittest`와 `pytest` 테스트 모두에서 사용할 수 있다.

그러나 테스트가 `unittest`가 아닌 경우 `pytest.mark.parametrize`를 사용할 수 있다.

다음은 이번에는 `pytest`의 `parametrize` 마크를 사용하는 동일한 예시이다:

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

`parameterized`와 마찬가지로 `-k` 필터가 요구사항을 충족하지 못하는 경우 `pytest.mark.parametrize`를 사용하여 어떤 서브 테스트를 실행할지 정확하게 제어할 수 있다. 단, 이 매개변수화 함수는 서브 테스트에 대해 약간 다른 이름 집합을 생성한다:

```bash
pytest test_this2.py --collect-only -q
```

목록이 표시된다:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

이제 특정 테스트를 실행할 수 있다:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

이전 예시와 마찬가지로.

### 파일 및 디렉토리

테스트에서는 현재 테스트 파일의 위치에 상대적인 위치를 알아야 하는 경우가 종종 있는데, 이는 간단하지 않다. 테스트는 여러 디렉토리에서 호출되거나 서로 다른 깊이의 서브 디렉토리에 있을 수 있기 때문이다. 도움 클래스 `testing_utils.TestCasePlus`는 모든 기본 경로를 해결하고 이에 대한 간단한 접근자를 제공함으로써 이 문제를 해결한다:

- `pathlib` 객체(모두 완전히 해결됨):

  - `test_file_path` - 현재 테스트 파일 경로, 즉 `__file__`
  - `test_file_dir` - 현재 테스트 파일을 포함하는 디렉토리
  - `tests_dir` - `tests` 테스트 스위트의 디렉토리
  - `examples_dir` - `examples` 테스트 스위트의 디렉토리
  - `repo_root_dir` - 저장소의 디렉토리
  - `src_dir` - `src` 디렉토리(즉 `transformers` 서브 디렉토리가 있는 디렉토리)

- 문자열 경로 -- 위와 동일하지만 `pathlib` 객체 대신 문자열 경로를 반환한다:

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

이를 사용하려면 테스트가 `testing_utils.TestCasePlus`의 서브클래스에 있는지 확인하면 된다. 예를 들어:

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

`pathlib`를 통해 경로를 조작할 필요가 없거나 문자열 경로만 필요한 경우 `pathlib` 객체의 `str()` 메서드를 호출하거나 `_str`로 끝나는 접근자를 사용할 수 있다. 예를 들어:

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

#### 임시 파일 및 디렉토리

고유한 임시 파일과 디렉토리를 사용하는 것은 병렬 테스트 실행에 필수적이며, 테스트가 서로의 데이터를 덮어쓰지 않도록 한다. 또한 각각을 생성한 테스트가 끝나면 임시 파일과 디렉토리를 삭제하길 원한다. 따라서 이러한 요구사항을 충족하기 위해 `tempfile`과 같은 패키지를 사용하는 것이 필수적이다.

그러나 테스트를 디버깅할 때는 임시 파일이나 디렉토리에 들어가는 내용을 볼 수 있어야 하고 테스트를 다시 실행할 때마다 무작위화되지 않는 정확한 경로를 알아야 한다.

도움 클래스 `testing_utils.TestCasePlus`는 이러한 목적에 가장 적합하다. 이는 `unittest.TestCase`의 서브클래스이므로 테스트 모듈에서 쉽게 상속할 수 있다.

사용 예시:

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

이 코드는 고유한 임시 디렉토리를 생성하고 `tmp_dir`을 해당 위치로 설정한다.

- 고유한 임시 디렉토리 생성:

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir`은 생성된 임시 디렉토리의 경로를 포함한다. 테스트가 끝나면 자동으로 삭제된다.

- 선택한 임시 디렉토리를 생성하고, 테스트 시작 전에 비어 있는지 확인하며, 테스트 종료 후에도 지우지 않는다.

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

이는 특정 디렉토리를 모니터링하고 이전 테스트에서 데이터가 남아 있지 않은지 확인하려는 디버깅 시 유용하다.

- `before`와 `after` 매개변수를 직접 재정의하여 다음 동작 중 하나를 초래할 수 있다:

  - `before=True`: 임시 디렉토리는 테스트 시작 시 항상 비워진다.
  - `before=False`: 임시 디렉토리가 이미 있는 경우 기존 파일이 유지된다.
  - `after=True`: 임시 디렉토리는 테스트 종료 시 항상 삭제된다.
  - `after=False`: 임시 디렉토리는 테스트 종료 시 유지된다.


footnote: `rm -r`에 해당하는 작업을 안전하게 실행하기 위해 명시적인 `tmp_dir`을 사용할 때만 프로젝트 저장소 체크아웃의 서브 디렉토리가 허용되므로 항상 `./`로 시작하는 경로를 전달한다.

footnote: 각 테스트는 여러 임시 디렉토리를 등록할 수 있으며, 다른 동작이 요청되지 않는 한 모두 자동으로 삭제된다.


#### 임시 sys.path 재정의

다른 테스트에서 가져오기 위해 `sys.path`를 임시로 재정의해야 하는 경우 `ExtendSysPath` 컨텍스트 관리자를 사용할 수 있다. 예를 들어:


```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 테스트 건너뛰기

이는 버그를 발견하고 새 테스트를 작성했지만 버그가 아직 수정되지 않은 경우 유용하다. 메인 저장소에 커밋하려면 `make test` 중에 건너뛰도록 해야 한다.

방법:

- **skip**은 특정 조건이 충족되는 경우에만 테스트가 통과할 것으로 예상되며, 그렇지 않으면 pytest가 테스트 실행을 완전히 건너뛰어야 함을 의미한다. 일반적인 예는 Windows에서만 실행되는 테스트를 건너뛰거나 현재 이용할 수 없는 외부 리소스(예: 데이터베이스)에 의존하는 테스트를 건너뛰는 것이다.

- **xfail**은 어떤 이유로든 테스트가 실패할 것으로 예상됨을 의미한다. 일반적인 예는 아직 구현되지 않은 기능이나 아직 수정되지 않은 버그이다. 예상 실패 상태에서 테스트가 통과(`pytest.mark.xfail`로 표시됨)하면 xpass로 보고된다.

둘의 중요한 차이점은 `skip`은 테스트를 실행하지 않지만 `xfail`은 실행한다는 것이다. 따라서 테스트 실패를 유발하는 코드가 다른 테스트에 영향을 미치는 나쁜 상태를 초래하면 `xfail`을 사용하지 않아야 한다.

#### 구현

- 전체 테스트를 무조건 건너뛰는 방법:

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

또는 pytest를 통해:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

또는 pytest를 통해:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

테스트 내부의 검사를 기반으로 테스트를 건너뛰는 방법:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

또는 전체 모듈:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

또는 pytest를 통해:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 일부 import가 없을 때 모듈의 모든 테스트를 건너뛰는 방법:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 조건에 따라 테스트 건너뛰기:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

또는:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

또는 전체 모듈 건너뛰기:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

자세한 내용, 예시 및 방법은 (https://docs.pytest.org/en/latest/skipping.html)을 참조하라.



### 출력 캡처

#### stdout/stderr 출력 캡처

`stdout` 및/또는 `stderr`에 쓰는 함수를 테스트하기 위해 테스트는 `pytest`의 `capsys` 시스템(https://docs.pytest.org/en/latest/capture.html)을 사용하여 이 스트림에 액세스할 수 있다. 구현 방법:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # 캡처된 출력 스트림 소비
    # 선택사항: 소비된 스트림을 재생하려면:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # 테스트:
    assert msg in out
    assert msg in err
```

물론 대부분의 경우 `stderr`는 예외의 일부로 나타나므로 이 경우 try/except를 사용해야 한다:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

stdout를 캡처하는 또 다른 방법은 `contextlib.redirect_stdout`을 통해서이다:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # 선택사항: 소비된 스트림을 재생하려면:
    sys.stdout.write(out)
    # 테스트:
    assert msg in out
```

stdout 캡처의 잠재적 문제는 이미 출력된 모든 것을 재설정하는 `\r` 문자를 포함할 수 있다는 것이다. 이는 `pytest`에는 문제가 없지만 `pytest -s`에서는 이 문자가 버퍼에 포함되므로 `-s` 포함 및 미포함 모두에서 테스트 실행이 가능하게 하려면 캡처된 출력에 대한 추가 정리가 필요하다. `re.sub(r'~.*\r', '', buf, 0, re.M)` 사용.

그러나 `\r` 문자 포함 여부와 관계없이 모든 것을 자동으로 처리하는 도움 클래스 컨텍스트 관리자 래퍼가 있어 매우 간단하다:

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

완전한 테스트 예시:

```python
from testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

`stderr`를 캡처하려면 `CaptureStderr` 클래스를 사용한다:

```python
from testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

두 스트림을 동시에 캡처하려면 부모 클래스 `CaptureStd` 클래스를 사용한다:

```python
from testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

또한 테스트 문제 디버깅을 돕기 위해 기본적으로 이 컨텍스트 관리자는 컨텍스트를 종료할 때 캡처된 스트림을 자동으로 재생한다.


#### logger 스트림 캡처

logger의 출력을 확인하려면 `CaptureLogger`를 사용할 수 있다:

```python
from transformers import logging
from testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### 환경 변수로 테스트

특정 테스트에 대한 환경 변수 영향을 테스트하려면 도움 데코레이터 `transformers.testing_utils.mockenv`를 사용할 수 있다:

```python
from testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

외부 프로그램을 호출해야 하는 경우가 있는데, 이는 `os.environ`에서 `PYTHONPATH`가 여러 로컬 경로를 포함해야 함을 의미한다. 도움 클래스 `testing_utils.TestCasePlus`가 도움이 된다:

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # 이제 외부 프로그램을 호출하고 `env`를 전달한다
```

테스트 파일이 `tests` 테스트 스위트에 있는지 `examples` 테스트 스위트에 있는지에 따라 해당 디렉토리 중 하나를 포함하도록 `env[PYTHONPATH]`를 올바르게 설정하고 `src` 디렉토리를 설정하여 현재 저장소에 대해 테스트가 수행되도록 한다. 마지막으로 테스트가 호출되기 전에 `env[PYTHONPATH]` 설정이 있는 경우에도 적용된다.

이 도움 메서드는 `os.environ` 객체의 복사본을 생성하므로 원본은 변경되지 않는다.


### Getting reproducible results

일부 경우에 테스트의 무작위성을 제거하고 싶을 수 있다. 동일한 재현 가능한 결과를 얻으려면 시드를 고정해야 한다:

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
tf.random.set_seed(seed)
```

## 테스트 디버깅

경고 지점에서 디버거를 시작하려면:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```


## 여러 pytest 보고서를 생성하기 위한 대규모 해킹

다음은 CI 보고서를 더 잘 이해하는 데 도움이 되도록 몇 년 전에 만든 대규모 `pytest` 패치이다.

이를 활성화하려면 `tests/conftest.py`에 추가한다(또는 없는 경우 생성):

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

그런 다음 테스트 스위트를 실행할 때 다음과 같이 `--make-reports=mytests`를 추가한다:

```bash
pytest --make-reports=mytests tests
```

그러면 8개의 개별 보고서가 생성된다:

```bash
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

이제 `pytest`에서 하나의 출력만 오는 것이 아니라 각 유형의 보고서가 별도의 파일에 저장된다.

이 기능은 CI에서 가장 유용하며 문제를 동시에 더 쉽게 확인하고 개별 보고서를 보고 다운로드할 수 있게 한다.

`--make-reports=`에 다른 값을 사용하면 서로 덮어쓰지 않고 각기 다른 테스트 그룹에 대해 별도로 저장할 수 있다.

이 모든 기능은 이미 `pytest`에 있지만 쉽게 추출하는 방법이 없어서 `testing_utils.py`(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py)에 monkey patch 재작성을 추가했다. `pytest`의 기능으로 기여할 수 있는지 물어봤지만 내 제안은 환영받지 못했다.


## testing_utils.py 분석

```markdown
# I developed the bulk of this library while I worked at HF

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio  # noqa
import contextlib
import importlib.util
import inspect
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from io import StringIO
from pathlib import Path
from typing import Iterator, Union
from unittest import mock
from unittest.case import SkipTest

import numpy as np
from packaging import version
from parameterized import parameterized


# PyTorch 임포트 시도, 성공하면 _torch_available을 True로, 실패하면 False로 설정
try:
    import torch
    _torch_available = True
except Exception:
    _torch_available = False


# PyTorch 사용 가능 여부 플래그 반환
def is_torch_available():
    return _torch_available


# 환경 변수에서 불리언 플래그 파싱, 설정되지 않은 경우 기본값 사용
def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # 환경 변수가 설정되지 않은 경우 기본값 사용
        _value = default
    else:
        # 환경 변수가 설정된 경우 True 또는 False로 변환
        try:
            _value = strtobool(value)
        except ValueError:
            # 더 많은 값을 지원하지만 오류 메시지를 간단하게 유지
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


# 환경 변수에서 정수 값 파싱, 설정되지 않은 경우 기본값 사용
def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value


# 데코레이터: PyTorch가 필요한 테스트 표시
def require_torch(test_case):
    """
    PyTorch가 필요한 테스트를 표시하는 데코레이터.
    PyTorch가 설치되지 않은 경우 이 테스트들은 건너뛰어진다.
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


# 데코레이터: GPU가 없는 환경이 필요한 테스트 표시
def require_torch_no_gpus(test_case):
    """
    GPU 없는 설정이 필요한 테스트를 표시하는 데코레이터(PyTorch에서). GPU가 있는 머신에서는 이 테스트들이 건너뛰어진다.
    GPU 없는 테스트만 실행하려면 모든 테스트 이름에 no_gpu가 포함된다고 가정: $ pytest -sv ./tests -k "no_gpu"
    """
    import torch

    if is_torch_available() and torch.cuda.device_count() > 0:
        return unittest.skip("test requires an environment w/o GPUs")(test_case)
    else:
        return test_case


# 데코레이터: 다중 GPU 환경이 필요한 테스트 표시
def require_torch_multi_gpu(test_case):
    """
    다중 GPU 설정이 필요한 테스트를 표시하는 데코레이터(PyTorch에서). 여러 GPU가 없는 머신에서는 이 테스트들이 건너뛰어진다.
    다중 GPU 테스트만 실행하려면 모든 테스트 이름에 multi_gpu가 포함된다고 가정: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    else:
        return test_case


# 데코레이터: 0 또는 1개의 GPU 환경이 필요한 테스트 표시
def require_torch_non_multi_gpu(test_case):
    """
    0 또는 1개의 GPU 설정이 필요한 테스트를 표시하는 데코레이터(PyTorch에서).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 1:
        return unittest.skip("test requires 0 or 1 GPU")(test_case)
    else:
        return test_case


# 데코레이터: 0~2개의 GPU 환경이 필요한 테스트 표시
def require_torch_up_to_2_gpus(test_case):
    """
    0, 1 또는 2개의 GPU 설정이 필요한 테스트를 표시하는 데코레이터(PyTorch에서).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 2:
        return unittest.skip("test requires 0 or 1 or 2 GPUs")(test_case)
    else:
        return test_case


# PyTorch가 가능한 경우 장치를 cuda 또는 cpu로 설정
if is_torch_available():
    # CUDA_VISIBLE_DEVICES="" 환경 변수를 설정하여 cpu 모드 강제
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None


# 데코레이터: CUDA와 PyTorch가 필요한 테스트 표시
def require_torch_gpu(test_case):
    """CUDA와 PyTorch가 필요한 테스트를 표시하는 데코레이터."""
    if torch_device != "cuda":
        return unittest.skip("test requires CUDA")(test_case)
    else:
        return test_case


# deepspeed 사용 가능 여부 확인
def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


# 데코레이터: deepspeed가 필요한 테스트 표시
def require_deepspeed(test_case):
    """
    deepspeed가 필요한 테스트를 표시하는 데코레이터
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)
    else:
        return test_case


# bitsandbytes 사용 가능 여부 확인
def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


# 데코레이터: bitsandbytes가 필요한 테스트 표시
def require_bnb(test_case):
    """
    bitsandbytes가 필요한 테스트를 표시하는 데코레이터
    """
    if not is_bnb_available():
        return unittest.skip("test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")(
            test_case
        )
    else:
        return test_case


# 비데코레이터 함수: bitsandbytes가 없으면 테스트 건너뛰기
def require_bnb_non_decorator():
    """
    bitsandbytes가 없는 경우 테스트를 건너뛰는 비데코레이터 함수
    """
    if not is_bnb_available():
        raise SkipTest("Test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")


# 재현성을 위한 랜덤 시드 설정
def set_seed(seed: int = 42):
    """
    random, numpy, torch의 시드를 설정하여 재현 가능한 동작을 위한 도움 함수

    Args:
        seed (:obj:`int`): 설정할 시드.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ cuda를 사용할 수 없는 경우에도 안전하게 호출할 수 있다


# 사용 가능한 GPU 수 가져오기
def get_gpu_count():
    """
    사용 가능한 GPU 수를 반환한다(torch 또는 tf 사용 여부에 관계없이)
    """
    if is_torch_available():
        import torch
        return torch.cuda.device_count()
    else:
        return 0


# 두 텐서 또는 비텐서 숫자가 동일한지 비교
def torch_assert_equal(actual, expected, **kwargs):
    """
    두 텐서 또는 비텐서 숫자가 동일한지 비교한다
    """
    # assert_close는 pt-1.9 무렵에 추가되었으며 더 나은 검사를 수행한다 - 예를 들어 차원이 일치하는지 확인한다
    return torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)


# 두 텐서 또는 비텐서 숫자가 근접한지 비교
def torch_assert_close(actual, expected, **kwargs):
    """
    두 텐서 또는 비텐서 숫자가 근접한지 비교한다.
    """
    # assert_close는 pt-1.9 무렵에 추가되었으며 더 나은 검사를 수행한다 - 예를 들어 차원이 일치하는지 확인한다
    return torch.testing.assert_close(actual, expected, **kwargs)


# torch bf16 지원 여부 확인
def is_torch_bf16_available():
    # https://github.com/huggingface/transformers/blob/26eb566e43148c80d0ea098c76c3d128c0281c16/src/transformers/file_utils.py#L301 에서 가져옴
    if is_torch_available():
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            return False
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
            return False
        if int(torch.version.cuda.split(".")[0]) < 11:
            return False
        if not version.parse(torch.__version__) >= version.parse("1.09"):
            return False
        return True
    else:
        return False


# 데코레이터: bf16을 지원하는 CUDA 하드웨어와 PyTorch >= 1.9가 필요한 테스트 표시
def require_torch_bf16(test_case):
    """bf16을 지원하는 CUDA 하드웨어와 PyTorch >= 1.9가 필요한 테스트를 표시하는 데코레이터."""
    if not is_torch_bf16_available():
        return unittest.skip("test requires CUDA hardware supporting bf16 and PyTorch >= 1.9")(test_case)
    else:
        return test_case


def get_tests_dir(append_path=None):
    """
    테스트 디렉토리의 전체 경로를 가져온다.

    Args:
        append_path: 선택적 경로, tests 디렉토리 경로 뒤에 추가된다

    Return:
        테스트 디렉토리의 전체 경로를 반환하여 테스트가 어디에서든 호출될 수 있도록 한다.
        append_path가 제공되면 tests 디렉토리 뒤에 추가된다.

    참고:
        - inspect.stack()을 사용하여 호출자의 파일 경로를 가져온다
        - os.path.abspath를 사용하여 절대 경로를 가져온다
        - os.path.dirname을 사용하여 디렉토리 이름을 가져온다
        - append_path가 제공되면 os.path.join을 사용하여 경로를 연결한다
    """
    # 이 함수를 호출하는 파일의 경로 가져오기
    caller__file__ = inspect.stack()[1][1]
    # tests 디렉토리의 절대 경로 가져오기
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        # append_path가 제공되면 tests 디렉토리 뒤에 추가
        return os.path.join(tests_dir, append_path)
    else:
        # 그렇지 않으면 tests 디렉토리 경로 직접 반환
        return tests_dir


def parameterized_custom_name_func_join_params(func, param_num, param):
    """
    모든 매개변수가 서브 테스트 이름에 표시되도록 하는 사용자 정의 테스트 이름 생성기 함수.
    기본적으로 첫 번째 매개변수만 표시하거나 여러 매개변수의 경우 어떤 매개변수도 표시하지 않고 고유한 ID 시퀀스만 사용한다.

    사용법:

    @parameterized.expand(
        [
            (0, True),
            (0, False), 
            (1, True),
        ],
        name_func=parameterized_custom_name_func_join_params,
    )
    def test_determinism_wrt_rank(self, num_workers, pad_dataset):

    다음과 같이 생성된다:

    test_determinism_wrt_rank_0_true
    test_determinism_wrt_rank_0_false 
    test_determinism_wrt_rank_1_true

    """
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"



class CaptureStd:
    """
    표준 출력과 표준 오류를 캡처하기 위한 컨텍스트 관리자.

    캡처 가능:
    - stdout: 출력 재생, 정리 후 obj.out으로 가져오기
    - stderr: 출력 재생 후 obj.err로 가져오기
    - combined: 선택한 스트림을 합쳐 obj.combined으로 가져오기

    초기화 매개변수:
    - out - stdout 캡처 여부: True/False, 기본값 True
    - err - stderr 캡처 여부: True/False, 기본값 True 
    - replay - 재생 여부: True/False, 기본값 True. 기본적으로 각 캡처된 스트림은 컨텍스트 종료 시 재생되어
      테스트에서 무엇을 하는지 볼 수 있다. 이 동작이 필요하지 않으면 replay=False를 전달하여 비활성화할 수 있다.
    """

    def __init__(self, out=True, err=True, replay=True):
        # 캡처된 출력 재생 여부
        self.replay = replay

        # stdout 캡처 초기화
        if out:
            self.out_buf = StringIO()  # stdout 버퍼 생성
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        # stderr 캡처 초기화
        if err:
            self.err_buf = StringIO()  # stderr 버퍼 생성
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

            self.combined = "error: CaptureStd context is unfinished yet, called too early"

    def __enter__(self):
        # stdout 리다이렉트
        if self.out_buf is not None:
            self.out_old = sys.stdout  # 원래 stdout 저장
            sys.stdout = self.out_buf  # 버퍼로 리다이렉트

        # stderr 리다이렉트
        if self.err_buf is not None:
            self.err_old = sys.stderr  # 원래 stderr 저장
            sys.stderr = self.err_buf  # 버퍼로 리다이렉트

        self.combined = ""  # 합쳐진 출력 초기화

        return self

    def __exit__(self, *exc):
        # stdout 복원 및 캡처된 출력 가져오기
        if self.out_buf is not None:
            sys.stdout = self.out_old  # 원래 stdout 복원
            captured = self.out_buf.getvalue()  # 캡처된 출력 가져오기
            if self.replay:
                sys.stdout.write(captured)  # 출력 재생
            self.out = apply_print_resets(captured)  # 출력 정리
            self.combined += self.out  # 합쳐진 출력에 추가

        # stderr 복원 및 캡처된 출력 가져오기
        if self.err_buf is not None:
            sys.stderr = self.err_old  # 원래 stderr 복원
            captured = self.err_buf.getvalue()  # 캡처된 출력 가져오기
            if self.replay:
                sys.stderr.write(captured)  # 출력 재생
            self.err = captured  # 출력 저장
            self.combined += self.err  # 합쳐진 출력에 추가

    def __repr__(self):
        # 캡처된 내용의 문자열 표현 생성
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# 테스트에서는 필요한 스트림만 캡처하는 것이 좋다, 그렇지 않으면 내용을 놓치기 쉽다.
# 두 스트림을 동시에 캡처해야 하는 경우가 아니라면 아래의 서브클래스를 사용한다(코드가 더 간결하다).
# 또는 테스트가 필요하지 않은 스트림을 비활성화하도록 CaptureStd를 구성한다.


class CaptureStdout(CaptureStd):
    """CaptureStd와 동일하지만 stdout만 캡처한다"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """CaptureStd와 동일하지만 stderr만 캡처한다"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    `logging` 스트림 캡처용

    이 클래스는 테스트에서 logging 모듈의 로그 출력을 캡처하는 데 사용된다. 임시 StreamHandler를 추가하여 로그를 캡처하고
    컨텍스트 관리자가 종료될 때 해당 handler를 제거한다.

    Args:
        - logger: `logging` logger 객체, 캡처할 logger 인스턴스

    Results:
        캡처된 출력은 `self.out`을 통해 가져올 수 있다

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart") 
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """

    def __init__(self, logger):
        # logger 인스턴스 저장
        self.logger = logger
        # 캡처된 출력을 저장하기 위한 StringIO 객체 생성
        self.io = StringIO()
        # 출력을 StringIO로 리다이렉트하는 StreamHandler 생성
        self.sh = logging.StreamHandler(self.io)
        # 캡처된 출력 저장
        self.out = ""

    def __enter__(self):
        # 컨텍스트 진입 시 handler를 추가하여 캡처 시작
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        # 컨텍스트 종료 시 handler 제거
        self.logger.removeHandler(self.sh)
        # 캡처된 출력 가져오기
        self.out = self.io.getvalue()

    def __repr__(self):
        # 캡처된 내용의 문자열 표현 반환
        return f"captured: {self.out}\n"


@contextlib.contextmanager
# https://stackoverflow.com/a/64789046/9201239 에서 가져옴
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    주어진 경로를 `sys.path`에 임시로 추가한다.

    이 컨텍스트 관리자는 Python의 모듈 검색 경로를 임시로 수정하는 데 사용된다. 제공된 경로를 sys.path의 맨 앞에 추가하여
    Python이 이 새로 추가된 경로에서 모듈을 임포트할 수 있게 한다. 컨텍스트가 종료되면 해당 경로가 sys.path에서 자동으로 제거된다.

    매개변수:
        path: sys.path에 추가할 경로, 문자열 또는 os.PathLike 객체일 수 있다

    반환:
        Iterator[None]: 컨텍스트 관리자 구현을 위한 iterator

    사용 예시::

       with ExtendSysPath('/path/to/dir'):
           mymodule = importlib.import_module('mymodule')

    """

    # 경로를 문자열 형식으로 변환
    path = os.fspath(path)
    try:
        # sys.path의 맨 앞에 경로 삽입
        sys.path.insert(0, path)
        yield
    finally:
        # 컨텍스트 종료 시 추가된 경로 제거
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """
    이 클래스는 `unittest.TestCase`를 확장하고 추가 기능을 추가한다.

    기능1: 완전히 해결된 중요한 파일 및 디렉토리 경로 접근자 집합.

    테스트에서는 현재 테스트 파일의 위치에 상대적인 다양한 경로를 알아야 하는 경우가 종종 있는데,
    이는 간단하지 않다. 테스트는 여러 디렉토리에서 호출되거나 서로 다른 깊이의 서브 디렉토리에 있을 수 있기 때문이다.
    이 클래스는 모든 기본 경로를 정리하고 간단한 접근자를 제공함으로써 이 문제를 해결한다:

    * `pathlib` 객체(모두 완전히 해결됨):
       - `test_file_path` - 현재 테스트 파일 경로(=`__file__`)
       - `test_file_dir` - 현재 테스트 파일을 포함하는 디렉토리
       - `tests_dir` - `tests` 테스트 스위트의 디렉토리
       - `data_dir` - `tests/data` 테스트 스위트의 디렉토리
       - `repo_root_dir` - 저장소의 디렉토리
       - `src_dir` - `m4` 서브 디렉토리가 있는 디렉토리(이 경우 repo_root_dir과 동일)

    * 문자열 경로 - 위와 동일하지만 `pathlib` 객체 대신 문자열 형식의 경로 반환:
       - `test_file_path_str`
       - `test_file_dir_str` 
       - `tests_dir_str`
       - `data_dir_str`
       - `repo_root_dir_str`
       - `src_dir_str`

    기능2: 테스트 종료 시 삭제가 보장되는 유연한 자동 삭제 임시 디렉토리.

    1. 고유한 임시 디렉토리 생성:
    ::
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()

    `tmp_dir`은 생성된 임시 디렉토리의 pathlib 경로를 포함한다. 테스트 종료 시 자동으로 삭제된다.

    2. 사용자 정의 임시 디렉토리 생성, 테스트 시작 전 비어 있는지 확인, 테스트 후 지우지 않음:
    ::
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")

    이는 특정 디렉토리를 모니터링하고 이전 테스트에서 데이터가 남아 있지 않은지 확인하려는 디버깅 시 유용하다.

    3. `before`와 `after` 매개변수를 직접 재정의하여 앞의 두 옵션을 재정의하고 다음 동작을 초래할 수 있다:

    `before=True`: 임시 디렉토리는 테스트 시작 시 항상 비워진다.
    `before=False`: 임시 디렉토리가 이미 있는 경우 기존 파일이 유지된다.
    `after=True`: 임시 디렉토리는 테스트 종료 시 항상 삭제된다.
    `after=False`: 임시 디렉토리는 테스트 종료 시 항상 유지된다.

    pathlib가 아닌 버전을 반환하려면 `self.get_auto_remove_tmp_dir_str()`을 사용한다.

    참고1: `rm -r`에 해당하는 작업을 안전하게 실행하기 위해 명시적인 `tmp_dir`을 사용하는 경우
    프로젝트 저장소 체크아웃의 서브 디렉토리만 허용되므로 `/tmp` 또는 파일 시스템의 다른 중요한 부분을 실수로 삭제하지 않는다. 항상 `./`로 시작하는 경로를 전달한다.

    참고2: 각 테스트는 여러 임시 디렉토리를 등록할 수 있으며, 달리 요청되지 않는 한 모두 자동으로 삭제된다.

    기능3: 현재 테스트 스위트에 특화된 `PYTHONPATH`가 설정된 `os.environ` 객체의 복사본 가져오기.
    이는 테스트 스위트에서 외부 프로그램을 호출하는 데 유용하다 - 예를 들어 분산 훈련.

    ::
        def test_whatever(self):
            env = self.get_env()
    """

    def setUp(self):
        """
        setUp 메서드는 테스트 클래스의 경로 관련 속성을 초기화하는 데 사용된다. 주요 작업:

        1. 임시 디렉토리 목록 초기화:
           - self.teardown_tmp_dirs = [] 테스트 종료 시 정리해야 할 임시 디렉토리 저장에 사용

        2. 테스트 파일 경로 파싱:
           - inspect.getfile()을 사용하여 현재 테스트 클래스의 파일 경로 가져오기
           - Path().resolve()를 사용하여 경로를 절대 경로로 변환
           - self._test_file_dir에 테스트 파일이 있는 디렉토리 저장

        3. 저장소 루트 디렉토리 찾기:
           - 최대 3단계 부모 디렉토리까지 순서대로 탐색
           - 각 단계의 디렉토리가 "m4"와 "tests" 서브 디렉토리를 모두 포함하는지 확인
           - 발견되면 해당 디렉토리를 self._repo_root_dir으로 설정
           - 찾지 못하면 ValueError 발생

        4. 기타 관련 디렉토리 설정:
           - self._tests_dir: 테스트 디렉토리 (<repo_root>/tests)
           - self._data_dir: 테스트 데이터 디렉토리 (<repo_root>/tests/test_data) 
           - self._src_dir: 소스 디렉토리, 여기서는 저장소 루트 디렉토리와 동일
        """
        # get_auto_remove_tmp_dir 기능:
        self.teardown_tmp_dirs = []

        # repo_root, tests 등의 해결된 경로 찾기
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "m4").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._data_dir = self._repo_root_dir / "tests" / "test_data"
        self._src_dir = self._repo_root_dir  # m4는 저장소에서 "src/" 접두사를 사용하지 않음

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def test_file_path_str(self):
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        return self._tests_dir

    @property
    def tests_dir_str(self):
        return str(self._tests_dir)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def data_dir_str(self):
        return str(self._data_dir)

    @property
    def repo_root_dir(self):
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        return self._src_dir

    @property
    def src_dir_str(self):
        return str(self._src_dir)

    def get_env(self):
        """
        `PYTHONPATH`가 올바르게 설정된 `os.environ` 객체의 복사본을 반환한다. 이는 테스트 스위트에서 외부 프로그램을 호출하는 데 유용하다 - 예를 들어 분산 훈련.

        항상 먼저 `.`을 삽입하고, 그 다음 테스트 스위트 유형에 따라 `./tests`를 삽입하고, 마지막으로 미리 설정된 `PYTHONPATH`를 삽입한다(있는 경우)(모두 완전히 해결된 경로).
        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        매개변수:
            tmp_dir (`string`, 선택적):
                `None`인 경우:
                   - 고유한 임시 경로가 생성됨
                   - `before`가 `None`이면 `before=True`로 설정
                   - `after`가 `None`이면 `after=True`로 설정
                그렇지 않으면:
                   - `tmp_dir`이 생성됨
                   - `before`가 `None`이면 `before=True`로 설정
                   - `after`가 `None`이면 `after=False`로 설정
            before (`bool`, 선택적):
                `True`이고 `tmp_dir`이 이미 있으면 즉시 비움
                `False`이고 `tmp_dir`이 이미 있으면 기존 파일이 유지됨
            after (`bool`, 선택적):
                `True`이면 테스트 종료 시 `tmp_dir` 삭제
                `False`이면 테스트 종료 시 `tmp_dir` 및 내용 유지

        반환:
            tmp_dir(`string`): `tmp_dir`을 통해 전달되었거나 자동으로 선택된 임시 디렉토리의 경로
        """
        if tmp_dir is not None:
            # 사용자 정의 경로 제공 시 가장 가능성 있는 동작 정의.
            # 이는 디버그 모드를 의미할 가능성이 높으며 쉽게 찾을 수 있는 디렉토리를 원한다:
            # 1. 테스트 전에 비워짐(이미 있는 경우)
            # 2. 테스트 후에 유지됨
            if before is None:
                before = True
            if after is None:
                after = False

            # 파일 시스템의 일부를 삭제하지 않도록 상대 경로만 허용
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir`은 상대 경로만 허용됩니다, 즉 `./some/path`이지만 `{tmp_dir}`을 받았습니다"
                )

            # 제공된 경로 사용
            tmp_dir = Path(tmp_dir).resolve()

            # 시작 시 디렉토리가 비어 있는지 확인
            if before is True and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            tmp_dir.mkdir(parents=True, exist_ok=True)

        else:
            # 고유한 임시 경로를 자동으로 생성할 때 가장 가능성 있는 동작 정의
            # (비디버그 모드), 여기서 고유한 임시 디렉토리가 필요하다:
            # 1. 테스트 전에 비어 있음(이 경우 어차피 비어 있을 것이다)
            # 2. 테스트 후에 완전히 삭제됨
            if before is None:
                before = True
            if after is None:
                after = True

            # 고유한 임시 디렉토리 사용(`before`에 관계없이 항상 비어 있음)
            tmp_dir = Path(tempfile.mkdtemp())

        if after is True:
            # 삭제를 위해 등록
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def get_auto_remove_tmp_dir_str(self, *args, **kwargs):
        return str(self.get_auto_remove_tmp_dir(*args, **kwargs))

    def tearDown(self):
        # get_auto_remove_tmp_dir 기능: 등록된 임시 디렉토리 삭제
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """
    다음 사용을 허용하는 편리한 데코레이터 래퍼:

    @mockenv(RUN_SLOW=True, USE_TF=False) 
    def test_something():
        run_slow = os.getenv("RUN_SLOW", False)
        use_tf = os.getenv("USE_TF", False)

    또한 컨텍스트 관리자로 사용하려면 `mockenv_context`를 참조

    Args:
        **kwargs: 임시로 설정할 환경 변수 키-값 쌍

    Returns:
        os.environ을 임시로 수정하는 mock.patch.dict 데코레이터를 반환
    """
    # mock.patch.dict를 사용하여 환경 변수를 임시로 수정
    # os.environ - 수정할 딕셔너리
    # kwargs - 설정할 새 키-값 쌍
    return mock.patch.dict(os.environ, kwargs)


# https://stackoverflow.com/a/34333710/9201239 에서 가져온 코드
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    ``os.environ`` 딕셔너리를 임시로 인플레이스로 업데이트한다. mockenv와 유사.

    ``os.environ`` 딕셔너리는 모든 경우에 수정이 적용되도록 인플레이스로 업데이트된다.

    Args:
      remove: 삭제할 환경 변수.
      update: 추가/업데이트할 환경 변수와 값의 딕셔너리.

    Example:

    with mockenv_context(FOO="1"):
        execute_subprocess_async(cmd, env=self.get_env())
    """
    # 환경 변수 딕셔너리 참조 가져오기
    env = os.environ
    # 업데이트 딕셔너리가 제공되지 않으면 빈 딕셔너리 사용
    update = update or {}
    # 삭제할 변수가 제공되지 않으면 빈 목록 사용
    remove = remove or []

    # 업데이트되거나 삭제될 환경 변수 목록 가져오기(update와 remove의 합집합과 현재 환경 변수의 교집합)
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # 종료 시 복원해야 할 환경 변수와 값 저장
    update_after = {k: env[k] for k in stomped}
    # 종료 시 삭제해야 할 환경 변수 가져오기(update에 새로 추가되었지만 원래 환경에 없던 변수)
    remove_after = frozenset(k for k in update if k not in env)

    try:
        # 환경 변수 업데이트
        env.update(update)
        # 지정된 환경 변수 삭제
        [env.pop(k, None) for k in remove]
        yield
    finally:
        # 이전에 저장된 환경 변수 복원
        env.update(update_after)
        # 새로 추가된 환경 변수 삭제
        [env.pop(k) for k in remove_after]


# --- 테스트 네트워크 도움 함수 --- #


def get_xdist_worker_id():
    """
    pytest-xdist에서 실행 중일 때 worker id(정수)를 반환하고 그렇지 않으면 0을 반환한다
    
    pytest-xdist는 테스트를 병렬로 실행하는 pytest 플러그인이다. 각 병렬 프로세스는 고유한 worker id를 가진다.
    이 함수는 환경 변수에서 worker id를 가져오고, pytest-xdist에서 실행 중이 아니면 0을 반환한다.
    """
    worker_id_string = os.environ.get("PYTEST_XDIST_WORKER", "gw0")  # 환경 변수에서 worker id 가져오기, 기본값은 "gw0"
    return int(worker_id_string[2:])  # "gw" 접두사 제거 후 정수로 변환


DEFAULT_MASTER_PORT = 10999  # 기본 마스터 포트 번호


def get_unique_port_number():
    """
    테스트 스위트가 pytest-xdist에서 실행 중일 때 동시 테스트가 동일한 포트 번호를 사용하지 않도록 해야 한다.
    동일한 기본 포트 번호에 xdist worker id를 더하여 이를 달성할 수 있다.
    pytest-xdist에서 실행 중이 아닌 경우 0을 더한다.
    
    반환:
        int: 기본 포트 번호에 worker id를 더한 고유한 포트 번호
    """
    return DEFAULT_MASTER_PORT + get_xdist_worker_id()  # 기본 포트 번호에 worker id를 더하여 고유한 포트 번호 얻기


# --- test IO helper functions --- #


def write_file(file, content):
    """
    파일에 내용 쓰기
    
    Args:
        file: 쓸 파일 경로
        content: 쓸 내용
    """
    with open(file, "w") as f:
        f.write(content)


def read_json_file(file):
    """
    JSON 파일 읽기 및 파싱
    
    Args:
        file: JSON 파일 경로
        
    Returns:
        파싱된 JSON 객체
    """
    with open(file, "r") as fh:
        return json.load(fh)


def replace_str_in_file(file, text_to_search, replacement_text):
    """
    파일에서 지정된 텍스트 교체
    
    Args:
        file: 처리할 파일 경로
        text_to_search: 찾을 텍스트
        replacement_text: 교체할 텍스트
    """
    file = Path(file)
    text = file.read_text()
    text = text.replace(text_to_search, replacement_text)
    file.write_text(text)


#-- pytest conf functions --#

"""
이것은 `pytest`가 개별 보고서를 출력하도록 하는 트릭이다

이를 활성화하려면 `tests/conftest.py`에 추가한다:

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

그런 다음 실행한다:

```python
pytest --make-reports=mytests tests
```

그런 다음 `reports/mytests/` 디렉토리 아래의 각 보고서를 확인한다

```python
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

이제 모든 것을 포함하는 하나의 `pytest` 출력이 아니라 각 유형의 보고서를 각각의 파일에 별도로 저장할 수 있다.

"""python
# tests/conftest.py와 examples/conftest.py에서 여러 번 호출하지 않도록 함 - 한 번만 호출되도록 보장
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    이 함수는 `conftest.py`에서 정의된 `pytest_addoption` 래퍼를 통해 호출되어야 한다.

    두 개의 `conftest.py` 파일을 동시에 로드할 때 동일한 `pytest` 옵션을 추가하여 실패하지 않도록 한다.
    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store", 
            default=False,
            help="보고서 파일 생성. 이 옵션의 값은 보고서 이름의 접두사로 사용된다",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    테스트 스위트 실행 종료 시 여러 보고서를 생성한다 - 각 보고서는 현재 디렉토리의 전용 파일에 저장된다.
    보고서 파일에는 테스트 스위트 이름이 접두사로 붙는다.

    이 함수는 --duration과 -rA pytest 인수를 모방한다.

    이 함수는 `conftest.py`에서 정의된 `pytest_terminal_summary` 래퍼를 통해 호출되어야 한다.

    매개변수:
    - tr: `conftest.py`에서 전달된 `terminalreporter`
    - id: `tests` 또는 `examples`와 같은 고유 식별자로 최종 보고서 파일 이름에 포함된다 -
         일부 작업에는 여러 번의 pytest 실행이 있으므로 서로 덮어쓸 수 없기 때문에 이것이 필요하다.

    참고: 이 함수는 비공개 _pytest API를 사용하며 가능성은 낮지만 pytest가 내부 변경을 수행하면 중단될 수 있다 -
    또한 다양한 `pytest-` 플러그인에 의해 가로채져 방해를 받을 수 있는 terminalreporter의 기본 내부 메서드를 호출한다.
    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    # 보고서 디렉토리 및 파일 생성
    dir = f"reports/{id}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{k}.txt"
        for k in [
            "durations",
            "errors", 
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # 사용자 정의 지속 시간 보고서
    # 참고: 이 별도의 보고서를 얻기 위해 pytest --durations=XX를 호출할 필요가 없다
    # https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66 에서 가져옴
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # 초
            f.write("가장 느린 지속 시간\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} 지속 시간 < {durations_min} 초 생략")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # 보고서가 --tb=long(기본값)이라고 가정하므로 마지막 프레임으로 자름
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "실패 짧은 스택")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # 선택적 앞부분 추가 프레임 제거, 마지막만 유지
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # 참고: 보고서를 짧게 유지하기 위해 rep.sections는 출력하지 않음

    # 기존 보고서 함수 사용, 전용 파일에 기록하기 위해 파일 핸들을 가로챔
    # https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814 에서 가져옴
    # 참고: 일부 pytest 플러그인은 기본 `terminalreporter`를 가로채서 방해할 수 있다(예: pytest-instafail이 그렇게 한다)

    # line/short/long 스타일로 실패 보고
    config.option.tbstyle = "auto"  # 전체 traceback
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # 짧은 traceback
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # 오류당 한 줄
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # 일반 경고
        tr.summary_warnings()  # 최종 경고

    tr.reportchars = "wPpsxXEf"  # -rA 모방(summary_passes() 및 short_test_summary()용)

    # `passes` 보고서 건너뛰기, 5분 이상 걸리기 시작하여 CircleCI에서 > 10분 걸리면 타임아웃되기 때문에
    # (또한 이 보고서에 유용한 정보가 없는 것 같아 거의 읽을 필요가 없다)
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # 원래 설정 복원:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- 분산 테스트 함수 --- #


class _RunOutput:
    """
    서브 프로세스 실행 결과를 저장하는 클래스
    
    속성:
        returncode: 프로세스 반환 코드
        stdout: 표준 출력 내용
        stderr: 표준 오류 내용
    """
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    """
    스트림에서 데이터를 비동기적으로 읽고 콜백 함수로 처리한다
    
    Args:
        stream: 읽을 스트림 객체
        callback: 각 줄의 데이터를 처리하는 콜백 함수
    """
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    """
    서브 프로세스를 비동기적으로 실행하고 출력 스트림을 실시간으로 처리한다
    
    Args:
        cmd: 실행할 명령 목록
        env: 환경 변수 딕셔너리
        stdin: 표준 입력
        timeout: 타임아웃(초)
        quiet: 조용한 모드 여부(출력을 출력하지 않음)
        echo: 실행된 명령을 출력할지 여부
        
    Returns:
        반환 코드와 출력 내용을 포함하는 _RunOutput 객체
    """
    if echo:
        print("\nRunning: ", " ".join(cmd))

    # 서브 프로세스 생성
    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # 참고: `wait`를 사용하면 대량 데이터 처리 시 데드락이 발생할 수 있다
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # 중단되기 시작하면 다음 코드로 전환해야 한다. 문제는 완료 전에 데이터를 볼 수 없고, 중단 시 디버깅 정보가 없다.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        """출력을 sink와 pipe에 동시에 쓴다"""
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: timeout 매개변수가 효과가 없는 것 같다
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda line: tee(line, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda line: tee(line, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    """
    비동기 서브 프로세스를 실행하는 동기 래퍼 함수
    
    Args:
        cmd: 실행할 명령 목록
        env: 환경 변수 딕셔너리
        stdin: 표준 입력
        timeout: 타임아웃(초)
        quiet: 조용한 모드 여부
        echo: 실행된 명령을 출력할지 여부
        
    Returns:
        _RunOutput 객체
        
    Raises:
        RuntimeError: 프로세스가 0이 아닌 값을 반환하거나 출력이 없을 때
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # 서브 프로세스가 실제로 실행되어 출력을 생성했는지 확인
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result

```
