> 블로그 출처: https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/ , Lei Mao의 글이며 저자의 전재 허가를 받았다. 앞으로 Lei Mao의 CUDA 관련 Blog도 일부 전재할 예정이며, 이는 하나의 완전한 칼럼이다. Blog는 조금 이른 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루고, 실용적인 엔지니어링 기법, 하위 수준 명령 분석, Cutlass 분석 등 여러 주제도 포함한다. 시간 흐름이 매우 명확한 칼럼이다.

# CUTLASS CUDA kernel 빌드와 개발

## 소개

CUTLASS(https://github.com/NVIDIA/cutlass)는 header-only library로, CUDA의 모든 계층과 규모에서 고성능 matrix-matrix multiplication(GEMM) 및 관련 계산을 구현하기 위한 일련의 CUDA C++ template abstraction으로 구성되어 있다.

이 블로그 글에서는 CUDA Docker container 안에서 CMake를 사용해 CUTLASS와 CuTe CUDA kernel을 빌드한다.

## CUDA Docker Container

CUTLASS kernel 개발용 CUDA Docker container를 만들 때 한 가지 선택지에 부딪힌다. Docker container 안에서 CUTLASS header-only library를 git clone할지, 아니면 CUTLASS header-only library를 CUDA kernel source code의 일부로 둘지이다.

처음에는 Docker container 안에서 CUTLASS header-only library를 clone했다. 하지만 Docker container에서 header-only library 구현을 살펴보려 하자 이것은 실행 가능하지 않았다. Docker container가 VS Code development container라면 Docker container에서 CUTLASS header-only library 구현을 살펴보는 시도를 여전히 할 수 있지만, CUTLASS header-only library를 수정하고 contribution하고 싶다면 이는 친화적이지 않다. 그래서 CUTLASS header-only library를 CUDA kernel source code의 일부로 두기로 했다.

### Docker image 빌드

다음 CUDA Dockerfile은 CUTLASS kernel 개발에 사용된다. 내 CUTLASS Examples GitHub repository에서도 찾을 수 있다.

```shell
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG CMAKE_VERSION=3.30.5
ARG GOOGLETEST_VERSION=1.15.2
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive

# package dependency 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        wget \
        git && \
    apt-get clean

# system locale
# UTF-8에 중요하다.
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# CMake 설치
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# GoogleTest 설치
RUN cd /tmp && \
    wget https://github.com/google/googletest/archive/refs/tags/v${GOOGLETEST_VERSION}.tar.gz && \
    tar -xzf v${GOOGLETEST_VERSION}.tar.gz && \
    cd googletest-${GOOGLETEST_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Nsight Compute GUI를 위해 QT6와 그 dependency 설치
# https://leimao.github.io/blog/Docker-Nsight-Compute/
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libasound2 \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libgl1-mesa-glx \
        libxkbfile-dev \
        openssh-client \
        xcb \
        xkb-data \
        libxcb-cursor0 \
        qt6-base-dev && \
    apt-get clean

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel
```

로컬에서 CUTLASS Docker image를 빌드하려면 다음 명령을 실행하라.

```shell
$ docker build -f docker/cuda.Dockerfile --no-cache --tag cuda:12.4.1 .
```

### Docker container 실행

custom Docker container를 실행하려면 다음 명령을 실행하라.

```shell
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt cuda:12.4.1
```

NVIDIA Nsight Compute가 포함된 custom Docker container를 실행하려면 다음 명령을 실행하라.

```shell
$ xhost +
$ docker run -it --rm --gpus device=0 -v $(pwd):/mnt -w /mnt -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --network host cuda:12.4.1
$ xhost -
```

## CUTLASS 예시

설치한 CUTLASS가 Docker container 안에서 동작한다는 것을 증명하기 위해, CUTLASS GitHub repository(https://github.com/NVIDIA/cutlass/tree/v3.5.0)에서 복사한 CUTLASS C++ 예시 두 개를 수정 없이 빌드하고 실행한다.

CUTLASS는 header-only이다. 각 CUTLASS build target은 cutlass/include와 `cutlass/tools/util/include`를 포함한 두 개의 핵심 header directory를 include해야 한다.

```shell
cmake_minimum_required(VERSION 3.28)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Toolkit 찾기
find_package(CUDAToolkit REQUIRED)

# CUTLASS include directory 설정
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

add_subdirectory(examples)
```

각 build target에 대해, NVCC compiler는 device code에서 host code의 일부 `constexpr`를 사용하려면 experimental flag `--expt-relaxed-constexpr`가 필요하다.

```shell
cmake_minimum_required(VERSION 3.28)

project(CUTLASS-GEMM-API-V3 VERSION 0.0.1 LANGUAGES CXX CUDA)

# compile code의 CUDA 아키텍처 설정
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
add_executable(${PROJECT_NAME} main.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${PROJECT_NAME} PRIVATE --expt-relaxed-constexpr)
```

### 예시 빌드

CMake로 CUTLASS 예시(https://github.com/leimao/CUTLASS-Examples/tree/f93e9d7bfa60ddc631b90d2f96be7bc036cb3e10/examples)를 빌드하려면 다음 명령을 실행하라.

```shell
$ cmake -B build
$ cmake --build build --config Release --parallel
```

### 예시 실행

CUTLASS 예시를 실행하려면 다음 명령을 실행하라.

```shell
$ ./build/examples/gemm_api_v2/CUTLASS-GEMM-API-V2
$ echo $?
0
```

```shell
$ ./build/examples/gemm_api_v3/CUTLASS-GEMM-API-V3
10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

Basic data-parallel GEMM
  Disposition: Passed
  Avg runtime: 0.175606 ms
  GFLOPs: 97831.9

StreamK GEMM with default load-balancing
  Disposition: Passed
  Avg runtime: 0.149729 ms
  GFLOPs: 114740
  Speedup vs Basic-DP: 1.173

StreamK emulating basic data-parallel GEMM
  Disposition: Passed
  Avg runtime: 0.177553 ms
  GFLOPs: 96759.2
  Speedup vs Basic-DP: 0.989

Basic split-K GEMM with tile-splitting factor 2
  Disposition: Passed
  Avg runtime: 0.183542 ms
  GFLOPs: 93601.7

StreamK emulating Split-K GEMM with tile-splitting factor 2
  Disposition: Passed
  Avg runtime: 0.173763 ms
  GFLOPs: 98869.8
  Speedup vs Basic-SplitK: 1.056
```

## 참고 자료

- CUTLASS(https://github.com/NVIDIA/cutlass)
- CUTLASS Examples - GitHub(https://github.com/leimao/CUTLASS-Examples)
- Nsight Compute In Docker(https://leimao.github.io/blog/Docker-Nsight-Compute/)
