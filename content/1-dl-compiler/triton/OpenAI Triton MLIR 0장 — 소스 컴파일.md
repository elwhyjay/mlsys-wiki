# OpenAI/Triton MLIR 0장: 소스 컴파일

## 본 글은 GiantPandaCV에서 최초 게재되었으며, 저자의 허락 없이는 전재할 수 없습니다.

## 머리말

왜 또 새로운 시리즈를 시작하는가? 그 이유는, 최근 진행 중인 프로젝트가 모두 [MLIR](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=MLIR&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJNTElSIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjI3NjkyNDQ4LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.hUUhr6YWP5j7xDHrHWmDtrmOsljjZkoP7DFUflD30M4&zhida_source=entity>)와 관련되어 있고, 어느새 MLIR R&D 분야에서 점점 더 깊이 들어가고 있다는 것을 깨달았기 때문이다. 마침 얼마 전 모두가 GPT 열풍에 동참할 때, OpenAI가 현재 공개한 제품들을 살펴보다가 우연히 Triton이라는 스위스 군용칼 같은 도구를 발견했다. 사실 몇 년 전에도 Triton에 대해 들어본 적이 있는데, 그 당시 Triton 코드는 아직 MLIR로 리팩터링되기 전이었고, 코드 내부의 어떤 로직들도 그다지 명확하게 작성되어 있지 않았다. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" 논문과 함께 살펴보아도 새로운 점을 그렇게 많이 발견하지는 못했다. 이번에 다시 살펴보니 그 안에 좋은 최적화 기법들이 많이 있음을 발견했다. 사실 GPU에서 고성능 코드를 생성하기 위한 MLIR Dialect 설계 방법을 학습하려는 의도를 가지고 Triton에 대해 심층 분석을 진행해 보고자 한다.

* * *

## Triton이란 무엇인가?

Triton은 OpenAI가 GPU에서의 operator 최적화를 위해 제안한 programming language & compiler이다. NVIDIA GPU를 예로 들면, Triton을 사용하면 CUDA의 폐쇄형 생태계를 우회하여, 자신의 backend를 직접 [llvm IR](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=llvm+IR&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJsbHZtIElSIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjI3NjkyNDQ4LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.6xTWD-byLazYDxkIfOhMM9jCQxAjdGoL_f0jbFy6UWQ&zhida_source=entity>)에 연결하고, NVPTX를 거쳐 GPU에서 실행되는 코드를 생성할 수 있다. 이렇게 하면 전통적으로 손으로 CUDA 코드를 작성하는 방식과 비교했을 때, NVIDIA의 nvcc compiler에 의존하지 않고도 GPU에서 동작하는 machine code를 얻을 수 있다는 장점이 있다. 동시에, Triton의 일부 설계 이념은 딥러닝 분야뿐만 아니라 다른 데이터 과학 분야에서 고성능 컴퓨팅을 수행하는 데에도 풍부한 지침적 의미를 제공할 수 있다. 또한, Triton은 NVIDIA의 GPU만 지원하는 것이 아니라, AMD GPU, Intel GPU에 대한 후속 지원 방안도 내놓을 예정이다. 이는 바로 MLIR의 강점, 즉 Dialect를 설계함으로써 더 많은 backend를 지원할 수 있다는 점을 잘 보여준다. Triton에 대한 자세한 소개는 아래 공식 페이지를 통해 얻을 수 있으며, Zhihu에도 Triton에 대해 비교적 상세한 표면적 소개를 한 분들이 많이 있으니 여기서는 더 자세히 다루지 않겠다.

[Introducing Triton: Open-source GPU programming for neural networksopenai.com/research/triton](<https://link.zhihu.com/?target=https%3A//openai.com/research/triton>)

[https://github.com/openai/tritongithub.com/openai/triton](<https://link.zhihu.com/?target=https%3A//github.com/openai/triton>)

* * *

## Triton 소스 컴파일

이어서 여러분과 함께 소스로부터 Triton 코드를 컴파일해 보고자 한다. 앞으로 몇 개의 장으로 나누어, Triton의 설계와 구체적인 최적화 세부 사항을 분석하여, 보다 포괄적인 이해를 제공하고자 한다. 무엇보다도, Triton은 MLIR에서 보기 드문 성공적인 end-to-end 사례 중 하나로, compile 기술과 시스템 최적화를 연구하는 연구자나 엔지니어에게는 없어서는 안 될 좋은 자료이다.

### 0x0 먼저 공식 사이트에서 Triton의 공식 repo를 clone

    $ git clone https://github.com/openai/triton.git
    $ cd triton 
    $ git checkout 132fe1bb01e0a734d39c60835c76da257dbe7151

### 0x1 third-party dependencies 설치

Triton 전체 소스 컴파일 과정에서 가장 중요한 두 가지 dependencies를 사용해야 하는데, 하나는 llvm이고, 다른 하나는 [pybind11](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=pybind11&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJweWJpbmQxMSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIyNzY5MjQ0OCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.cvUHQvIE-LxwIxXG9eoqC9nrynZ6WGnlf-lyhzEl6sw&zhida_source=entity>)이다. 나는 Triton을 컴파일하고 빌드하는 과정에서, llvm과 pybind11을 수동으로 컴파일·설치한 뒤, Triton을 컴파일할 때 CMakeLists.txt를 통해 해당 경로를 지정하는 방식을 사용했다.

### 0x10 LLVM 다운로드 및 설정

왜 llvm을 사용해야 하는가? 사실 모두가 알고 있듯이, 이는 Triton의 가장 매력적인 점이다. 상위 수준의 Python 코드를 한 단계씩 llvm IR로 lowering하고, 이후 llvm 생태계를 거쳐 최종적으로 특정 디바이스에서 동작 가능한 machine code를 얻는다. llvm을 가장 중요한 backend로 삼고 있으며, Triton 내부 구현 또한 MLIR로 리팩터링되어 있다. MLIR은 마침 llvm의 매우 중요한 서브 프로젝트이기도 하다. 따라서 Triton 기반의 2차 개발을 하려는 엔지니어나 연구자에게 llvm 설치는 매우 중요하다.

    $ git clone https://github.com/llvm/llvm-project
    $ cd llvm-project 
    $ git checkout c5dede880d175f7229c9b2923f4753e12702305d
    $ mkdir build 
    $ cd build 
    $ cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=ON \
       -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;RISCV;AMDGPU" \
       -DMLIR_ENABLE_CUDA_RUNNER=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DCMAKE_C_COMPILER=clang \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DLLVM_ENABLE_RTTI=ON \
       -DLLVM_INSTALL_UTILS=ON \
       -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
    
    ninja -j8
    sudo ninja install

어느 정도 시간이 흐른 뒤 llvm을 설치할 수 있다. 왜 해당 commit id로 전환해야 하는지에 대해서는, Triton 공식 repo에 명시된 commit을 참고하여 컴파일한 것이다.

[](<https://link.zhihu.com/?target=https%3A//github.com/openai/triton/blob/319af1fb65e72b302a4ee38af64ba640f519c775/python/setup.py%23L71>)

### 0x11 pybind11 다운로드 및 설정

왜 pybind11을 사용해야 하는가? pybind11은 이미 현재 주류 AI 개발 도구에서 빠질 수 없는 컴포넌트이다. 대부분의 framework는 Python의 DSL을 사용자에게 노출하고, 사용자는 해당 Python 문법을 작성함으로써 이미 C++/CUDA 또는 assemble로 작성된 고성능 컴포넌트를 호출한다. 따라서 pybind11을 설치하는 목적은, import triton을 통해 해당 Python API를 매끄럽게 호출하여 고성능 operator 생성 작업을 완료할 수 있도록 하기 위함이다.

    $ pip install pytest
    $ git clone https://github.com/pybind/pybind11.git
    $ cd pybind11
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make check -j 8
    $ sudo make install

## 0x2 Triton 컴파일

    $ cd triton
    $ vim CMakeLists.txt (option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" ON))
    $ mkdir build 
    $ cd build 
    $ cmake ..
    $ make -j8

![이미지](images/img_01.png)

최종적으로 .so 파일, 즉 libtriton.so가 생성된 것을 볼 수 있다.

이어서 libtriton.so 파일을 triton/python/triton/_C 디렉토리로 옮기고, Triton의 Python 경로를 bashrc에 추가하기만 하면 된다.

    export TRITON_HOME=/home/Documents/compiler/triton
    export PYTHONPATH=$TRITON_HOME/python:${PYTHONPATH}

그런 다음 간단히 import triton을 실행하여 아무 오류도 없으면 Triton으로 개발을 진행할 수 있다.

![이미지](images/img_02.png)

이어서 triton/python/tutorials로 들어가, 아무 예시나 골라 검증해 본다. 여기서는 가장 흔하고 실용적인 03-matrix-multiplication.py를 선택하여, 바로 python 03-matrix-multiplication.py를 실행한다. 잠시 기다리면 최종 결과를 얻을 수 있다.

![이미지](images/img_03.png)

보다시피, Triton이 최종적으로 생성한 코드는 3090에서 single batch gemm의 일부 size에서 이미 cuBLAS를 능가했다.

또한, build 디렉토리에서 해당하는 세 가지 bin tool, 즉 **triton-opt, triton-reduce, triton-translate**를 확인할 수 있다.

그런 다음 본인 컴퓨터의 **ptxas**를 해당 build 디렉토리에 복사한다. 내 **ptxas**는 (/usr/local/cuda-11.6/bin) 아래에 있다. 이러한 도구들의 사용법은 후속 해설에서 서로 다른 layer의 dialect 간 conversion에 따라 자세히 소개할 예정이다. /triton/python/triton/common/backend.py 디렉토리를 열고, 다음 코드로 _path_to_binary를 교체한다. 이때 binary의 위치는 본인 컴퓨터의 ptxas 경로로 바꿔주면 된다.

    def _path_to_binary(binary: str):
        if binary == "ptxas":
            binary = "/usr/local/cuda-12.1/bin/ptxas"
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        paths = [
            os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
            os.path.join(base_dir, "third_party", "cuda", "bin", binary)
        ]
    
        for p in paths:
            bin = p.split(" ")[0]
            if os.path.exists(bin) and os.path.isfile(bin):
                result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
                if result is not None:
                    version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                    if version is not None:
                        return p, version.group(1)
        raise RuntimeError(f"Cannot find {binary}")

  


##  0x3 왜 이런 컴파일 방식을 사용하는가?

사실 어떤 분들은, Triton 튜토리얼대로 그냥 pip install -e . 만 하면 되지 않느냐고 할 것이다. 이렇게 하는 이유는, 이후에 Triton과 그에 대응하는 llvm을 개선해야 하기 때문이다. 매번 개선한 후, Triton과 llvm을 각각 컴파일해야 한다. 이런 분리된 방식은, 해당 llvm 코드나 Triton 소스 코드를 개선한 뒤, 단계별로 컴파일하고, 다시 새로운 shared library (libtriton.so)로 통합할 수 있도록 해준다.

후속 튜토리얼에서는 Triton의 frontend, optimizer, backend를 진입점으로 삼아, Triton이 사용자가 손으로 작성한 Python DSL을 어떻게 GPU에서 동작 가능한 machine code로 컴파일하는지 각각 설명할 예정이다.

* * *

## Triton의 현재 설계

Triton의 소스 코드를 보면, Triton은 현재 NVIDIA GPU에서 비교적 성숙한 자체 매핑 경로를 가지고 있다. 먼저 Python 언어 layer, 즉 Triton DSL을 추상화하여 AST를 얻고, 그런 다음 AST의 각 노드를 [Triton Dialect](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=Triton+Dialect&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJUcml0b24gRGlhbGVjdCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIyNzY5MjQ0OCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.8g6qAKyxNObrq8iR1uukdudYtydkQh9ihxeMNvE-moU&zhida_source=entity>)으로 lowering한다. Triton Dialect는 상위 언어 표현에 비교적 가까운 IR로, 그 주된 역할은 사용자가 해당 알고리즘을 작성할 때 정확성을 유지하기 위함이다. 이어서 [TritonGPU Dialect](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=TritonGPU+Dialect&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJUcml0b25HUFUgRGlhbGVjdCIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIyNzY5MjQ0OCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.0GDWaTpgu2RBCKXaVOYQ7O4KaXW9vvdolKeNP_lPhYY&zhida_source=entity>)으로 더 매핑된다. TritonGPU Dialect는 GPU 레벨에 더 가까운 IR로, 구체적인 성능 최적화를 위해 설계되었다. 그림에서 다른 파란색 모듈들, 예를 들어 [SCF](<https://zhida.zhihu.com/search?content_id=227692448&content_type=Article&match_order=1&q=SCF&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTY0OTgsInEiOiJTQ0YiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjc2OTI0NDgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.bMi3noK5w2k3jh0TCF662-eFi4lcX_dTqQIIcnThBbs&zhida_source=entity>), Arith, Tensor 등은 모두 MLIR 생태계에서 이미 구현되어 널리 사용되고 있는 Dialect들이다. 이러한 Dialect들은 TritonGPU Dialect와 함께 공존하며, 그런 다음 해당 LLVM Dialect로 lowering된다. LLVM Dialect는 LLVM IR에 가장 가까운 layer의 설계로, LLVM Dialect에서 LLVM IR로의 변환은 매우 용이하다. 최종적으로 코드는 LLVM의 NVPTX backend로 연결되어, 후속에 GPU에서 동작 가능한 고성능 machine code가 생성된다.

![이미지](images/img_04.png)

* * *

## Triton의 향후 지원

아래 그림에서 볼 수 있듯이, Triton의 향후 계획은 대부분의 compiler와 동일한 발전 청사진을 가지고 있다. 위로는 다양한 표현력을 가진 여러 frontend를 지원하고, 아래로는 다양한 벤더의 hardware와 연결하여, 최종적으로 application을 hardware에 효율적으로 매핑하는 것이다.

내 생각에는

![이미지](images/img_05.png)

* * *

위 PPT 자료 참고: [https://www.jokeren.tech/assets/Triton_next.pdf](<https://link.zhihu.com/?target=https%3A//www.jokeren.tech/assets/Triton_next.pdf>)
