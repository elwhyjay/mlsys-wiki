# Buddy-MLIR 프로젝트 상세 (MLIR 입문에 매우 좋은 선택)

며칠 전 훙빈(Hongbin)이 자신들 PLCT의 Buddy MLIR 프로젝트를 소개했는데, 꽤 흥미로운 프로젝트라는 느낌을 받았습니다. 마침 단오절을 맞이하여 이 프로젝트를 한번 살펴보면서 코드 구조를 대략적으로 정리하고 관련 예시들을 모두 실행해 보았습니다. 그 과정에서 몇 가지 작은 BUG도 발견하여 작은 수정 PR도 몇 개 제출했습니다. 그래서 이 글에서는 제 시각에서 Buddy-MLIR이 왜 우수한 프로젝트인지, 그리고 컴파일러 개발에 종사하는 동료나 입문자(소위 초보 사용자)에게 어떤 도움을 줄 수 있는지를 기록해 보려고 합니다. Let's Go!

> https://github.com/BBuf/tvm_mlir_learn 도 팔로우해서 MLIR 관련 지식을 더 알아보고 학습하시기 바랍니다. 필자는 최근 MLIR을 기반으로 양자화 학습 관련 프로젝트를 진행 중이며, 여러분과 더 많은 교류를 기대합니다.

# 0x0. 머리말

Buddy-MLIR 프로젝트 전체에서 제가 받은 가장 큰 느낌은, 결과가 어떻든 일단 먼저 run 시켜볼 수 있다는 점입니다. MLIR이 등장한 지 몇 년이 되었고 IREE 같은 스타 프로젝트도 성공을 거두었지만, TVM 의 사용 사례 풍부도와 비교하면 개인적으로 아직 약간의 차이가 있다고 느낍니다. 특히 중국어 커뮤니티에서는 더욱 그렇습니다. 이로 인해 한 가지 문제가 발생하는데, MLIR에 관심이 있거나 MLIR을 기반으로 어떤 개발 작업을 해야 하는 사람은 반드시 MLIR 공식 문서의 Toy Tutorial을 깊게 파고들어 속성으로 학습해야 한다는 것입니다. 공식 문서가 매우 상세하고 구조 구성도 잘 되어 있다는 것을 부정하지는 않지만, 완전 초보 사용자에게는 정말 그다지 친절하지 않습니다. **MLIR 관련 기초 개념을 어느 정도 이해한 후에 MLIR이 제공하는 컴포넌트로 실제 애플리케이션을 빠르게 구축하는 세계로 들어갈 방법은 없을까요?**

저는 개인적으로 최근에 출시된 Buddy-MLIR이 이러한 문제점을 완화해 준다고 생각합니다. 우리는 MLIR 기반으로 만들어진 애플리케이션을 매우 쉽게 실행해 본 다음, MLIR 관련 개념을 학습하면서 동시에 코드를 변형하여 자신만의 애플리케이션을 구축할 수 있습니다. **Buddy-MLIR의 또 다른 장점은 전체 엔지니어링의 조직 구조가 LLVM/MLIR 프로젝트 자체와 마찬가지로 매우 명확하다는 것이며, 덕분에 전체 엔지니어링을 파악하는 난이도와 관련 코드를 읽는 난이도가 많이 낮아졌다는 점입니다.** 이어서 Run 해보기와 엔지니어링 구조 분석 두 가지 측면에서 설명하겠습니다. 사실 이런 조직 구조는 OneFlow 저장소의 IR 부분에서도 완전히 동일합니다. 다만 OneFlow의 계산 그래프와 IR이 상호작용하기 때문에 현재 IR 부분을 별도의 저장소로 분리하지 않은 것일 뿐입니다. 그렇지 않았다면 Buddy-MLIR과 OneFlow-MLIR의 엔지니어링 구조도 완전히 동일하다는 것을 보실 수 있을 것입니다.

# 0x1. How to run?

어떻게 실행하는가? 이는 프로젝트를 받았을 때 가장 중요한 문제 중 하나입니다. 사실 Buddy-MLIR의 README를 따라가면 되지만, 실제 작업할 때 주의해야 할 몇 가지 세부사항이 있습니다. 초보 사용자를 위해 여기서는 Ubuntu 20.04 환경에서 Buddy-MLIR을 완전히 컴파일하고 Run하는 전체 과정을 기록해 두겠습니다.

Buddy-MLIR 프로젝트는 LLVM/MLIR 프로젝트를 기반으로 확장한 것입니다. 다시 말해 LLVM은 Buddy-MLIR의 의존성이므로, 먼저 이 의존성을 설치해야 합니다. 구체적인 작업 과정은 다음과 같습니다.
    
    
    $ git clone git@github.com:buddy-compiler/buddy-mlir.git  
    $ cd buddy-mlir  
    $ git submodule update --init  
      
    $ cd buddy-mlir  
    $ mkdir llvm/build  
    $ cd llvm/build  
    $ cmake -G Ninja ../llvm \  
        -DLLVM_ENABLE_PROJECTS="mlir" \  
        -DLLVM_TARGETS_TO_BUILD="host;RISCV" \  
        -DLLVM_ENABLE_ASSERTIONS=ON \  
        -DCMAKE_BUILD_TYPE=RELEASE  
    $ ninja  
    $ ninja check-mlir  
    

위 명령에 따라 작업하면 LLVM 프로젝트의 컴파일을 완료할 수 있으며, 컴파일 결과는 llvm/build 폴더에 저장됩니다. 다음으로 Buddy-MLIR 엔지니어링 디렉터리에서 LLVM 컴파일 결과가 제공하는 라이브러리를 기반으로 Buddy-MLIR 자체의 컴파일을 완료할 수 있습니다. Buddy-MLIR 엔지니어링 컴파일은 다음과 같습니다.
    
    
    $ cd buddy-mlir  
    $ mkdir build  
    $ cd build  
    $ cmake -G Ninja .. \  
        -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \  
        -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \  
        -DLLVM_ENABLE_ASSERTIONS=ON \  
        -DCMAKE_BUILD_TYPE=RELEASE  
    $ ninja check-buddy  
    

컴파일이 완료된 후 다음과 같은 출력이 나타나면, 즉 FileCheck가 성공하면 Buddy-MLIR의 빌드 흐름이 성공했음을 증명할 수 있습니다.
    
    
    Testing Time: 0.06s  
      Passed: 3  
    

Buddy-MLIR 오픈소스 엔지니어링에는 현재 세 가지 Dialect가 있는데, 즉 Bud Dialect, DIP Dialect 그리고 RVV Dialect입니다. 프로젝트 관련 소개로는 RVV Dialect는 아직 이해하지 못했으므로, 본 글에서는 Bud Dialect와 DIP Dialect만 다룰 것입니다. 그중 DIP Dialect는 디지털 이미지 처리(digital image processing)를 위해 추상화된 것입니다. Buddy-MLIR C/C++ frontend가 이미지 인코딩/디코딩에 OpenCV를 의존하기 때문에 Buddy-MLIR은 OpenCV 서드파티 라이브러리를 도입했습니다. 만약 OpenCV를 컴파일하지 않았다면 다음 명령을 사용해 컴파일할 수 있습니다.
    
    
    $ sudo apt-get install libgtk2.0-dev pkg-config  libcanberra-gtk-module  
    $ git clone https://github.com/opencv/opencv.git  
    $ cd opencv && mkdir build && cd build  
    $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..  
    $ make -j$(nproc)  
    $ sudo make install  
    

여기서 `/usr/local`은 임의의 사용자 정의 디렉터리로 바꿀 수 있습니다. 이후 DIP Dialect 관련 애플리케이션을 빌드할 때 `-DBUDDY_ENABLE_OPENCV=ON` 옵션을 명시하여 OpenCV를 활성화해야 합니다.

이제 Buddy-MLIR이 어떤 흥미로운 예시들을 제공하는지 살펴보겠습니다.

### 1\. IR 수준의 예시

IR 수준 예시는 상위(upstream) MLIR과 Buddy-MLIR에서 pass를 어떻게 사용하는지를 보여주며, 그중 일부 예시는 MLIR 통합 테스트에서 가져온 것입니다. 대부분의 경우 MLIR JIT 엔진인 mlir-cpu-runner를 직접 사용해 실행할 수 있습니다. lowering 파이프라인과 툴체인 설정은 makefile target에 지정되어 있습니다. 우리는 관심 있는 Dialect를 선택하여 해당 디렉터리로 이동해 실행할 target을 찾을 수 있습니다. Buddy-MLIR의 모든 예시는 `https://github.com/buddy-compiler/buddy-mlir/tree/main/examples` 이 디렉터리에 있습니다:

![이미지](images/img_01.png)Buddy-MLIR 예시 분류

임의의 Dialect 예시의 MakeFile을 열어보면, 그 안에 주로 세 종류의 테스트가 있다는 것을 발견할 수 있습니다.

  * `<Dialect Name>-<Operation Name>-lower`. 이 테스트는 lowering 파이프라인을 보여줍니다. `log.mlir` 파일을 생성합니다.
  * `<Dialect Name>-<Operation Name>-translate`. 이 테스트는 현재 Dialect 파일에서 생성된 LLVM IR을 보여줍니다. `log.ll` 파일을 생성합니다.
  * `<Dialect Name>-<Operation Name>-run`. 이 테스트는 MLIR JIT Engine을 사용해 LLVM IR을 실행하여 결과를 생성합니다.



MemRef Dialect 안의 `memref.dim` Op를 예로 들면, 컴파일 테스트 방법은 다음과 같습니다.
    
    
    $ cd buddy-mlir/examples/MLIRMemRef  
    $ make memref-dim-lower  
    $ make memref-dim-translate  
    $ make memref-dim-run  
    

원본 `memref.dim`은 다음과 같이 생겼습니다.
    
    
    func.func @main() {  
      %c0 = arith.constant 0 : index  
      %c1 = arith.constant 1 : index  
      %mem0 = memref.alloc() : memref<2x3xf32>  
      %mem1 = memref.cast %mem0 : memref<2x3xf32> to memref<?x?xf32>  
      %dim0 = memref.dim %mem0, %c0 : memref<2x3xf32>  
      %dim1 = memref.dim %mem0, %c1 : memref<2x3xf32>  
      %dim2 = memref.dim %mem1, %c0 : memref<?x?xf32>  
      %dim3 = memref.dim %mem1, %c1 : memref<?x?xf32>  
      vector.print %dim0 : index  
      vector.print %dim1 : index  
      vector.print %dim2 : index  
      vector.print %dim3 : index    
      memref.dealloc %mem0 : memref<2x3xf32>  
      func.return  
      
    }  
    

JIT Engine을 사용해 실행한 출력:
    
    
    2  
    3  
    2  
    3  
    

### 2\. Convolution Vectorization Examples

Buddy-MLIR은 2D 벡터화 컨볼루션 Pass인 `conv-vectorization`을 제공하는데, 이 Pass는 Coefficients Broadcasting algorithm with Strip Mining 알고리즘을 구현하고 있으며, strip mining size는 설정 가능합니다. 여기서는 256으로 설정하여 시연하겠습니다.
    
    
    $ cd buddy-mlir/build/bin  
    $ ./buddy-opt ../../examples/ConvOpt/conv2d.mlir -conv-vectorization="strip-mining=256"  
    

원본 `conv2d.mlir`은 다음과 같이 생겼습니다.
    
    
    func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {  
      linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)  
                     outs (%arg2: memref<?x?xf32>)  
      return  
    }  
    

위의 실행 명령을 거친 후 생성된 MLIR 파일 결과는 다음과 같습니다.
    
    
    #map0 = affine_map<(d0) -> (d0)>  
    #map1 = affine_map<(d0) -> (d0 ceildiv 256)>  
    module {  
      func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {  
        %c0 = arith.constant 0 : index  
        %c1 = arith.constant 1 : index  
        %c256 = arith.constant 256 : index  
        %cst = arith.constant 0.000000e+00 : f32  
        %0 = vector.splat %cst : vector<256xf32>  
        %1 = memref.dim %arg1, %c0 : memref<?x?xf32>  
        %2 = memref.dim %arg1, %c1 : memref<?x?xf32>  
        %3 = memref.dim %arg2, %c0 : memref<?x?xf32>  
        %4 = memref.dim %arg2, %c1 : memref<?x?xf32>  
        affine.for %arg3 = #map0(%c0) to #map0(%3) {  
          affine.for %arg4 = #map0(%c0) to #map0(%1) {  
            affine.for %arg5 = #map0(%c0) to #map0(%2) {  
              affine.for %arg6 = #map0(%c0) to #map1(%4) {  
                // 아래의 단계 1에 해당  
                %5 = affine.vector_load %arg1[%arg4, %arg5] : memref<?x?xf32>, vector<1xf32>  
                %6 = vector.broadcast %5 : vector<1xf32> to vector<256xf32>  
                %7 = arith.muli %arg6, %c256 : index  
                %8 = arith.subi %4, %7 : index  
                %9 = arith.cmpi sge, %8, %c256 : index  
                scf.if %9 {  
                  // 아래의 단계 2에 해당  
                  %10 = affine.vector_load %arg0[%arg3 + %arg4, %arg5 + %arg6 * 256] : memref<?x?xf32>, vector<256xf32>  
                  // 아래의 단계 3에 해당  
                  %11 = affine.vector_load %arg2[%arg3, %arg6 * 256] : memref<?x?xf32>, vector<256xf32>  
                  // 아래의 단계 4에 해당  
                  %12 = vector.fma %10, %6, %11 : vector<256xf32>  
                  // 아래의 단계 5에 해당  
                  affine.vector_store %12, %arg2[%arg3, %arg6 * 256] : memref<?x?xf32>, vector<256xf32>  
                } else {  
                  %10 = vector.create_mask %8 : vector<256xi1>  
                  %11 = arith.addi %arg3, %arg4 : index  
                  %12 = arith.muli %arg6, %c256 : index  
                  %13 = arith.addi %arg5, %12 : index  
                  %14 = vector.maskedload %arg0[%11, %13], %10, %0 : memref<?x?xf32>, vector<256xi1>, vector<256xf32> into vector<256xf32>  
                  %15 = vector.maskedload %arg2[%arg3, %12], %10, %0 : memref<?x?xf32>, vector<256xi1>, vector<256xf32> into vector<256xf32>  
                  %16 = vector.fma %14, %6, %15 : vector<256xf32>  
                  vector.maskedstore %arg2[%arg3, %12], %10, %16 : memref<?x?xf32>, vector<256xi1>, vector<256xf32>  
                }  
              }  
            }  
          }  
        }  
        return  
      }  
    }  
    

처음 이 변환을 보면 다소 혼란스러울 수 있는데, 이 알고리즘과 Pass 구현을 결합하여 이해해 봅시다.

Coefficients broadcasting(CB) 알고리즘은 2D 컨볼루션의 효율적인 구현 중 하나입니다. Buddy-MLIR은 MLIR 인프라스트럭처 기반으로 이 알고리즘의 구현을 완성했습니다. 이 알고리즘 구현에 관련된 MLIR Dialect와 Op를 여기에 나열합니다.

  * `affine.for`: 지정된 횟수만큼 루프 본문을 실행하는 operation.
  * `affine.vector_load`: 버퍼 슬라이스에서 하나의 벡터를 반환합니다 (MLIR MemRef 형식).
  * `affine.vector_store`: 하나의 벡터를 버퍼 슬라이스에 기록합니다 (MLIR MemRef 형식).
  * `vector.broadcast`: 스칼라 또는 벡터 값을 N-차원 결과 벡터로 broadcast합니다.
  * `vector.fma`: 벡터화된 타입의 곱셈-덧셈 혼합 명령.



그리고 CB 알고리즘의 과정은 아래 그림과 같습니다.

![이미지](images/img_02.png)CB 알고리즘 흐름

입력은 채널 수가 1인 이미지 또는 feature map이고, kernel의 채널 수도 1이라는 점에 주의하세요. 알고리즘의 실행 흐름은 대략 다음과 같습니다.

  * 먼저 kernel의 각 원소를 `vector_load`를 사용해 버퍼에 로드한 다음 `vector.broadcast`를 사용해 `vector1`로 broadcast 합니다.
  * 다음으로 feature map의 원소를 `vector_load`를 사용해 `vector2`로 로드합니다.
  * 세 번째 단계로 출력 feature map의 원소를 `vector_load`를 사용해 `vector3`으로 로드합니다.
  * 그 다음 `vector.fma`를 사용해 `vector1`과 `vector2`를 곱하고 `vector3`에 더합니다.
  * 마지막으로 `vector_store`를 사용해 위의 결과를 버퍼에 다시 기록합니다.



참고로, `conv-vectorization` Pass를 거친 후 생성된 MLIR 파일에는 두 부분이 있습니다. 다른 한 부분은 `vector.create_mask`와 `vector.maskedstore`를 사용했는데, 이는 위 그림에서 feature map의 각 행 마지막에 로드한 원소 바이트가 `fma` 명령이 필요로 하는 256Bit (이 256은 `-conv-vectorization="strip-mining=256"`을 통해 지정됨)에 부족한 경우에 해당하므로, Mask로 보충한 후 계산을 수행해야 합니다.

  * Edge detection example



Buddy-MLIR은 또한 최적화를 보여주기 위한 에지 검출(edge detection) 예시도 제공합니다. `conv-vectorization` pass는 우리의 알고리즘을 사용해 `linalg.conv_2d`를 lowering 하는 역할을 합니다. 그런 다음 `mlir-translate`와 `llc` 도구를 사용해 object 파일을 생성합니다. 마지막으로, 이 MLIR 컨볼루션 함수를 C++ 프로그램에서 호출합니다 (이 과정은 2절에서 자세히 소개합니다). 이 예시를 실행하기 전에 OpenCV가 설치되어 있는지 확인해야 하며, 설치 방법은 위에서 소개했습니다.

이 예시는 또한 AutoConfig 메커니즘의 "마법"도 보여주는데, 이는 `strip mining size`, `ISA SIMD/Vector extension`, `target triple`을 지정하지 않아도 도와줍니다. 단지 `BUDDY_EXAMPLES` 옵션만 활성화하면 되며, 툴체인 설정에 대해 걱정할 필요가 없습니다. 작업 명령은 다음과 같습니다.
    
    
    $ cd buddy-mlir/build  
    $ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON -DBUDDY_ENABLE_OPENCV=ON  
    $ ninja edge-detection  
    

물론, 우리는 자신만의 설정 값인 `-DBUDDY_CONV_OPT_STRIP_MINING` (예: 64)와 `-DBUDDY_OPT_ATTR` (예: avx2)를 사용할 수도 있습니다.

저장소는 `buddy-mlir/examples/ConvOpt/images/YuTu.png` 경로에 이미지 한 장을 제공하는데, 이는 중국 창어(Chang'e) 3호 임무의 일부를 구성하는 로봇 달 탐사차(YuTu)입니다. 그런 다음 아래 명령을 실행해 에지 검출을 수행합니다.
    
    
    $ cd bin  
    $ ./edge-detection ../../examples/ConvOpt/images/YuTu.png result.png  
    

![이미지](images/img_03.png)원본 이미지![이미지](images/img_04.png)에지 검출 후의 이미지

### 3\. Digital Image Processing Examples

Buddy-MLIR은 DIP Dialect 관련 시연 예시도 제공하는데, 구체적으로는 한 장의 이미지에 대해 Constant Padding 또는 Replicate Padding을 한 다음 컨볼루션을 수행하는 것입니다. 작업 단계는 위와 유사하므로 여기서는 다시 보여주지 않겠습니다. 관심 있는 독자는 직접 체험해 보실 수 있습니다. 링크: `https://github.com/buddy-compiler/buddy-mlir/tree/main/examples#digital-image-processing-examples`.

# 0x2. How to Understand?

위의 절에서는 주로 Buddy-MLIR에서 빌드한 애플리케이션을 어떻게 실행하는지 보여주었으며, 이번 절에서는 Buddy-MLIR의 구조에서 출발하여 이 엔지니어링을 이해하도록 안내하겠습니다. 엔지니어링의 전체 구조는 다음과 같이 요약할 수 있습니다.

![이미지](images/img_05.png)Buddy-MLIR 엔지니어링 구조

우리는 주로 `include`와 `lib` 두 폴더에 시선을 둘 것이며, 그 외의 문서, 테스트 그리고 도구류의 소스 코드는 독자가 선택적으로 살펴볼 수 있습니다.

## 2.1 Bud Dialect

위 그림에서 볼 수 있듯이, Buddy-MLIR에는 주로 세 가지 Dialect가 있는데, 즉: Bud Dialect, DIP Dialect 그리고 RVV Dialect입니다. Dialect의 정의는 LLVM 상위(upstream) Dialect와 동일한 파일 구조와 방법을 따르므로 여기서는 다시 설명하지 않겠습니다. 더 많은 세부사항을 알고 싶다면 https://github.com/BBuf/tvm_mlir_learn 저장소의 **MLIR: 무어의 법칙 종말 시대의 컴파일러 인프라스트럭처 (논문 해설)** 글을 참고하실 수 있습니다.

여기서는 주로 Bud Dialect에 어떤 operation들이 정의되어 있는지에 주목하겠습니다. `buddy-mlir/include/Dialect/Bud/BudOps.td`에서 볼 수 있듯이 Bud Dialect는 주로 4가지 유형의 operation을 정의하고 있습니다.

  * Bud_TestConstantOp. 이 Op는 상수(constant) Op를 테스트하는 데 사용됩니다.
  * Bud_TestPrintOp. 이 Op는 print Op를 테스트하는 데 사용됩니다.
  * Bud_TestEnumAttrOp. Op에서 열거(enum) attribute를 테스트합니다.
  * Bud_TestArrayAttrOp. Op에서 배열(array) attribute를 테스트합니다.



기본 operation을 구축한 후, 우리는 Bud Dialect를 위한 lowering Pipeline을 등록해야 하는데, 즉 `lib/Conversion/LowerBud/LowerBudPass.cpp`에 구현된 `LowerBudPass`입니다.

`bud::TestConstantOp`에 대한 구현은 다음과 같습니다.
    
    
    class BudTestConstantLowering : public OpRewritePattern<bud::TestConstantOp> {  
    public:  
      using OpRewritePattern<bud::TestConstantOp>::OpRewritePattern;  
      
      LogicalResult matchAndRewrite(bud::TestConstantOp op,  
                                    PatternRewriter &rewriter) const override {  
        auto loc = op.getLoc();  
        // Get type from the origin operation.  
        Type resultType = op.getResult().getType();  
        // Create constant operation.  
        Attribute zeroAttr = rewriter.getZeroAttr(resultType);  
        Value c0 = rewriter.create<mlir::arith::ConstantOp>(loc, resultType, zeroAttr);  
      
        rewriter.replaceOp(op, c0);  
        return success();  
      }  
    };  
    

`bud::TestConstantOp`에 매칭된 후 이를 `mlir::arith::ConstantOp`로 rewrite 하는 것을 볼 수 있습니다. 우리는 `buddy-mlir/examples/BudDialect`에서 `make bud-constant-lower`를 실행할 수 있습니다. 얻은 결과는 다음과 같습니다.
    
    
    module {  
      %i0 = bud.test_constant : i32  
    }  
      
    =>  
    module {  
      %c0_i32 = arith.constant 0 : i32  
    }  
    

다른 몇 가지 operation도 유사한데, 모두 Bud Dialect에 정의된 몇 가지 operation을 지정된 몇 가지 상위(upstream) Dialect로 Lower 합니다. 이 LowerBudPass의 구체적인 구현은 다음과 같습니다.
    
    
    namespace {  
    class LowerBudPass : public PassWrapper<LowerBudPass, OperationPass<ModuleOp>> {  
    public:  
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBudPass)  
      LowerBudPass() = default;  
      LowerBudPass(const LowerBudPass &) {}  
      
      StringRef getArgument() const final { return "lower-bud"; }  
      StringRef getDescription() const final { return "Lower Bud Dialect."; }  
      
      void runOnOperation() override;  
      
      void getDependentDialects(DialectRegistry &registry) const override {  
        // clang-format off  
        registry.insert<  
            buddy::bud::BudDialect,  
            func::FuncDialect,  
            vector::VectorDialect,  
            memref::MemRefDialect>();  
        // clang-format on  
      }  
    };  
    } // end anonymous namespace.  
      
    void LowerBudPass::runOnOperation() {  
      MLIRContext *context = &getContext();  
      ModuleOp module = getOperation();  
      
      ConversionTarget target(*context);  
      // clang-format off  
      target.addLegalDialect<  
          arith::ArithmeticDialect,  
          func::FuncDialect,  
          vector::VectorDialect,  
          memref::MemRefDialect>();  
      // clang-format on  
      target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();  
      
      RewritePatternSet patterns(context);  
      populateLowerBudConversionPatterns(patterns);  
      
      if (failed(applyPartialConversion(module, target, std::move(patterns))))  
        signalPassFailure();  
    }  
    

볼 수 있듯이 Bud Dialect의 operation은 주로 **arith::ArithmeticDialect, func::FuncDialect, vector::VectorDialect, memref::MemRefDialect** 로 Lower 됩니다.

위의 소개에서도 알 수 있듯이, Buddy Dialect는 사실 시연 역할만 할 뿐이며, 아마도 초보자에게 새로운 Dialect를 어떻게 빠르게 정의하고 MLIR 생태계에 접목시킬 수 있는지를 가르쳐 주는 역할일 것입니다.

## 2.2 DIP Dialect

DIP Dialect는 디지털 이미지 처리에 대한 추상화입니다. 여기서는 DIP Dialect가 현재 정의한 operation들을 보여드리겠습니다.
    
    
    def DIP_ConstantPadding : I32EnumAttrCase<"ConstantPadding", 0, "CONSTANT_PADDING">;  
    def DIP_ReplicatePadding : I32EnumAttrCase<"ReplicatePadding", 1, "REPLICATE_PADDING">;  
      
    def DIP_BoundaryOption : I32EnumAttr<"BoundaryOption",  
        "Specifies desired method of boundary extrapolation during image processing.",  
        [  
          DIP_ConstantPadding,  
          DIP_ReplicatePadding  
        ]>{  
      let genSpecializedAttr = 0;  
      let cppNamespace = "::buddy::dip";  
    }  
      
    def DIP_BoundaryOptionAttr : EnumAttr<DIP_Dialect, DIP_BoundaryOption, "boundary_option">;  
    

![이미지](images/img_06.png)DIP Dialect의 Corr2DOp

DIP Dialect는 유일한 operation인 DIP_Corr2DOp를 정의했는데, 이 Op는 2D 컨볼루션을 수행하기 전에 먼저 입력에 대해 Padding을 수행하여 컨볼루션 후 출력 feature map의 크기가 입력과 일치하도록 합니다. 여기에는 또한 많은 최적화 기법이 관련되어 있는데, 구체적으로는 https://github.com/buddy-compiler/buddy-mlir/blob/main/docs/dip-opt.md 이 문서와 https://github.com/buddy-compiler/buddy-mlir/blob/main/lib/Conversion/LowerDIP/LowerDIPPass.cpp 이 Pass 구현에 나타나 있습니다. 저는 이 알고리즘의 로직을 완전히 정리하지 못했으므로, 여기서는 이 부분을 설명하지 않겠습니다. 관심 있는 독자는 직접 연구해 보시기 바랍니다.

## 2.3 Interface

위에서 Buddy-MLIR 프로젝트에 정의된 두 가지 Dialect를 소개했는데, 이 절에서는 다음과 같은 질문에 답해야 합니다. 즉, 우리는 Buddy-MLIR을 기반으로 구축한 알고리즘을 어떻게 C/C++ frontend에서 호출하여 완전한 응용 프로그램을 구현할 수 있을까요?

이 목적을 달성하기 위해 Buddy-MLIR은 C/C++ frontend를 위한 데이터 구조 MemRef를 구현했습니다. https://github.com/buddy-compiler/buddy-mlir/blob/main/include/Interface/buddy/core/Container.h.
    
    
    // MemRef descriptor.  
    // - T represents the type of the elements.  
    // - N represents the number of dimensions.  
    // - The storage order is NCHW.  
    template <typename T, size_t N> class MemRef {  
    public:  
      // Constructor from shape.  
      MemRef(intptr_t sizes[N], T init = T(0));  
      // Constructor from data.  
      MemRef(const T *data, intptr_t sizes[N], intptr_t offset = 0);  
      // Copy constructor.  
      MemRef(const MemRef<T, N> &other);  
      // Copy assignment operator.  
      MemRef<T, N> &operator=(const MemRef<T, N> &other);  
      // Move constructor.  
      MemRef(MemRef<T, N> &&other) noexcept;  
      // Move assignment operator.  
      MemRef<T, N> &operator=(MemRef<T, N> &&other) noexcept;  
      // Desctrutor.  
      ~MemRef();  
      // Get the data pointer.  
      T *getData();  
      // Get the sizes (shape).  
      const intptr_t *getSizes() { return sizes; }  
      // Get the strides.  
      const intptr_t *getStrides() { return strides; }  
      // Get the rank of the memref.  
      size_t getRank() const { return N; }  
      // Get the size (number of elements).  
      size_t getSize() const { return size; }  
      // Get the element at index.  
      const T &operator[](size_t index) const;  
      T &operator[](size_t index);  
      
    protected:  
      // Default constructor.  
      // This constructor is desinged for derived domain-specific constructor.  
      MemRef() {};  
      // Set the strides.  
      // Computes the strides of the transposed tensor for transpose=true.  
      void setStrides();  
      // Compute the product of array elements.  
      size_t product(intptr_t sizes[N]) const;  
      
      // Data.  
      // The `aligned` and `allocated` members point to the same address, `aligned`  
      // member is responsible for handling data, and `allocated` member is  
      // resposible for handling the memory space.  
      T *allocated;  
      T *aligned;  
      // Offset.  
      intptr_t offset = 0;  
      // Shape.  
      intptr_t sizes[N];  
      // Strides.  
      intptr_t strides[N];  
      // Number of elements.  
      size_t size;  
    };  
    

구체적인 구현은: https://github.com/buddy-compiler/buddy-mlir/blob/main/lib/Interface/core/Container.cpp 에 있습니다. 여기서는 주로 이 사용자 정의 MemRef 클래스가 어떻게 C/C++ frontend에 서비스를 제공하는지를 정리해 보겠습니다. 여기서는 에지 검출을 예로 들겠습니다. 핵심 코드 구현은 다음과 같습니다.
    
    
    #include <iostream>  
    #include <opencv2/imgcodecs.hpp>  
    #include <time.h>  
      
    #include "Interface/buddy/core/ImageContainer.h"  
    #include "kernels.h"  
      
    using namespace cv;  
    using namespace std;  
      
    // Declare the conv2d C interface.  
    extern "C" {  
    void _mlir_ciface_conv_2d(Img<float, 2> *input, MemRef<float, 2> *kernel,  
                              MemRef<float, 2> *output);  
    }  
      
    int main(int argc, char *argv[]) {  
      printf("Start processing...\n");  
      
      // Read as grayscale image.  
      Mat image = imread(argv[1], IMREAD_GRAYSCALE);  
      if (image.empty()) {  
        cout << "Could not read the image: " << argv[1] << endl;  
        return 1;  
      }  
      Img<float, 2> input(image);  
      
      // Define the kernel.  
      float *kernelAlign = laplacianKernelAlign;  
      int kernelRows = laplacianKernelRows;  
      int kernelCols = laplacianKernelCols;  
      intptr_t sizesKernel[2] = {kernelRows, kernelCols};  
      MemRef<float, 2> kernel(kernelAlign, sizesKernel);  
      
      // Define the output.  
      int outputRows = image.rows - kernelRows + 1;  
      int outputCols = image.cols - kernelCols + 1;  
      intptr_t sizesOutput[2] = {outputRows, outputCols};  
      MemRef<float, 2> output(sizesOutput);  
      
      // Run the convolution and record the time.  
      clock_t start, end;  
      start = clock();  
      
      // Call the MLIR conv2d function.  
      _mlir_ciface_conv_2d(&input, &kernel, &output);  
      
      end = clock();  
      cout << "Execution time: " << (double)(end - start) / CLOCKS_PER_SEC << " s"  
           << endl;  
      
      // Define a cv::Mat with the output of the conv2d.  
      Mat outputImage(outputRows, outputCols, CV_32FC1, output.getData());  
      
      // Choose a PNG compression level  
      vector<int> compression_params;  
      compression_params.push_back(IMWRITE_PNG_COMPRESSION);  
      compression_params.push_back(9);  
      
      // Write output to PNG.  
      bool result = false;  
      try {  
        result = imwrite(argv[2], outputImage, compression_params);  
      } catch (const cv::Exception &ex) {  
        fprintf(stderr, "Exception converting image to PNG format: %s\n",  
                ex.what());  
      }  
      if (result)  
        cout << "Saved PNG file." << endl;  
      else  
        cout << "ERROR: Can't save PNG file." << endl;  
      
      return 0;  
    }  
    

여기서 Img 클래스의 base 클래스도 MemRef 클래스라는 점에 유의하세요.
    
    
    // Image container.  
    // - T represents the type of the elements.  
    // - N represents the number of dimensions.  
    template <typename T, size_t N> class Img : public MemRef<T, N> {  
    public:  
      Img(cv::Mat image);  
    };  
    

그리고 위의 응용 프로그램에서는 conv2d Op의 C frontend 함수를 정의했습니다.
    
    
    // Declare the conv2d C interface.  
    extern "C" {  
    void _mlir_ciface_conv_2d(Img<float, 2> *input, MemRef<float, 2> *kernel,  
                              MemRef<float, 2> *output);  
    }  
    

이 전역 C 함수는 buddy-opt를 실행하는 과정에서 `llvm.call` 명령으로 번역되는데, 즉 CMakeLists.txt에서 다음 부분입니다.
    
    
    add_custom_command(OUTPUT conv2d.o  
      COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${BUDDY_EXAMPLES_DIR}/ConvOpt/conv2d.mlir -conv-vectorization="strip-mining=${SPLITING_SIZE}" -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm='emit-c-wrappers=1' -reconcile-unrealized-casts |   
              ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |  
              ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=${BUDDY_TARGET_TRIPLE} -mattr=${BUDDY_OPT_ATTR} --filetype=obj -o ${BUDDY_BINARY_DIR}/../examples/ConvOpt/conv2d.o  
      DEPENDS buddy-opt)  
    

conv2d operation의 원본 MLIR 파일 내용은 다음과 같습니다.
    
    
    func.func @conv_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {  
      linalg.conv_2d ins (%arg0, %arg1: memref<?x?xf32>, memref<?x?xf32>)  
                     outs (%arg2: memref<?x?xf32>)  
      return  
    }  
    

`-convert-func-to-llvm='emit-c-wrappers=1'` 이 Pass를 실행할 때 위의 Func Dialect 아래의 conv2d operation을 LLVM IR로 번역하고 이를 `llvm.call` 명령으로 래핑합니다. 여기의 자세한 상호작용 과정은 buddy-mlir/llvm/mlir/docs/TargetLLVMIR.md 이 문서에서 볼 수 있습니다. 즉, MLIR이 C/C++의 frontend 인터페이스 기능을 제공하며, Buddy-MLIR이 이 frontend 인터페이스 기능을 활용해 end-to-end 애플리케이션 구축을 완성한 것입니다.

위에서 LLVM IR을 얻었고, 그 후 cmake 명령에서 볼 수 있듯이 LLVM `llc` 명령을 호출하여 LLVM 소스 파일을 지정된 아키텍처용 어셈블리 언어로 컴파일했습니다. 그런 다음 어셈블리 언어 출력은 native assembler와 linker를 통해 전달되어 native 실행 파일을 생성할 수 있습니다. 여기에서 실행 아키텍처와 일부 최적화 매개변수 등을 지정할 수 있습니다.

# 0x3. buddy-opt와 buddy-translate

위에서 소개한 구현된 Pass를 MLIR의 상위(upstream) Pass 관리 메커니즘에 추가하면 buddy-opt 도구가 구현됩니다.

그리고 buddy-translate는 단지 Buddy Dialect에서 LLVMIR로의 번역 기능 한 가지를 확장한 것입니다.

# 0x4. 정리

전체적으로 보면, Buddy-MLIR은 MLIR에 입문하거나 MLIR을 인프라스트럭처로 삼아 자신의 응용 프로그램을 구축하는 데 비교적 좋은 예시입니다. 필요한 독자나 개발자에게 학습하고 더 많은 가능성을 탐색해 볼 것을 추천합니다. 본 글에서는 RVV Dialect 관련 지식은 전혀 다루지 않았는데, 저도 현재 그다지 잘 알지 못하기 때문입니다. 후에 훙빈(Hongbin)이 이 Dialect의 동기와 세부사항에 대해 설명해 주기를 바랍니다.

# 0x5. 링크

https://github.com/buddy-compiler/buddy-mlir

