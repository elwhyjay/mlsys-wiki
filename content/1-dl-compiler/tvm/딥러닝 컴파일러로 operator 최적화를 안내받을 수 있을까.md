# 딥러닝 컴파일러로 operator 최적화를 안내받을 수 있을까

# 0x0. 머리말

예전에 Ansor 논문(https://zhuanlan.zhihu.com/p/390783734)을 읽으면서 다음과 같은 의문이 들었습니다. Ansor가 사람이 지정한 추론 규칙 아래에서 휴리스틱하게 고성능 Scheduler 템플릿을 생성한다면, 이 operator의 Scheduler 템플릿이 거꾸로 우리가 프로그램을 작성할 때 안내 역할을 해줄 수 있지 않을까? 라는 생각이었습니다. 그래서 이 실험을 시작하게 되었는데, 최근 업무 때문에 많이 미뤄지다가 이번 주말에 시간을 내어 실험 결과를 정리하고 이 글을 작성하게 되었습니다. 필자는 GEMM 최적화에만 익숙하기 때문에 여기서는 X86의 GEMM 최적화를 예시로 탐색해 보겠습니다. 이 글이 여러분에게 영감을 줄 수 있기를 바라며, 글의 모든 실험 코드는 https://github.com/BBuf/tvm_learn 에 올려두었습니다. 관심 있는 분들은 star 한 번 눌러 주시고 함께 공부해 주세요(TVM을 공부한 4개월 동안 이 저장소가 거의 100 star를 받았습니다. 정말 감사드립니다).

# 0x1. 부동소수점 피크 측정

사실 하드웨어의 부동소수점 피크를 어떻게 측정하는지는 1년 전에 작성한 글(https://zhuanlan.zhihu.com/p/268925243)에서 이미 다루었습니다. 새로 오신 분들을 위해 부동소수점 피크가 무엇인지 간단히 복습해 보겠습니다.

알고리즘의 부동소수점 피크 `gflops`는 계산량을 소요 시간으로 나누어 얻는 값입니다.

당연히 부동소수점 피크가 높을수록 알고리즘의 성능도 좋습니다.

따라서 최적화 전에 먼저 하드웨어의 부동소수점 피크를 측정해야 합니다. X86을 예로 들면, https://zhuanlan.zhihu.com/p/28226956 을 참고하여 `https://github.com/pigirons/cpufp`를 클론한 뒤 `sh build.sh`로 빌드하면 부동소수점 피크 측정용 실행 파일 `cpufp`를 얻을 수 있습니다.

그리고 `./cpu_fp -num_threads`를 실행하면 지정한 스레드 수에서 하드웨어의 최대 gflops를 측정할 수 있습니다.

여기서 실험에 사용한 CPU는 `64 Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz`입니다.

스레드를 1로 지정하여 측정한 부동소수점 피크는 다음과 같습니다.
    
    
    Thread(s): 1  
    avx512_vnni int8 perf: 262.2060 gops.  
    avx512f fp32 perf: 65.5496 gflops.  
    avx512f fp64 perf: 33.2332 gflops.  
    fma fp32 perf: 73.0070 gflops.  
    fma fp64 perf: 36.3787 gflops.  
    avx fp32 perf: 36.5239 gflops.  
    avx fp64 perf: 18.2485 gflops.  
    sse fp32 perf: 22.2130 gflops.  
    sse fp64 perf: 9.2662 gflops.  
    

이 글의 최적화는 모두 fma 명령어 기반이므로, 여기서는 fma fp32의 부동소수점 피크만 보면 됩니다. **「대략 73gflops 정도임을 알 수 있습니다」**.

# 0x2. GEMM 최적화 입문

https://github.com/flame/how-to-optimize-gemm/wiki 에서는 다양한 최적화 방법으로 GEMM을 최적화하는 방법을 소개하고 있습니다. 기본 방법은 출력을 여러 개의 4×4 sub-block으로 나누어 입력 데이터의 재사용을 높이는 것입니다. 동시에 register를 대량으로 사용하여 메모리 접근을 줄이고, 메모리 접근과 계산을 vectorize 하며, 포인터 계산을 제거하고, 메모리를 주소가 연속되도록 재구성합니다. 최종적으로 GEMM의 성능을 원본 버전 대비 8배 이상으로 끌어올립니다.

![이미지](images/img_01.png)how-to-optimize-gemm의 성능 그래프, 원본의 GFlops는 약 1.0 정도이고, 최적화 후에는 10GFlops에 가까워집니다![이미지](images/img_02.png)원본 버전의 GFlops, 여기서 사용한 하드웨어는 Core i5 CPU입니다

저는 how-to-optimize-gemm을 좀 더 간단히 수정한 버전을 만들어, 각 최적화 단계의 gflops를 더 직관적으로 확인할 수 있도록 했습니다. 주소는 다음과 같습니다. `https://github.com/BBuf/tvm_learn/tree/main/optimize_gemm/src`. 관심 있는 독자는 위에서 언급한 단계별 GEMM 최적화 블로그를 학습하면서 GEMM의 일반적인 최적화 기법들을 익혀 보시기 바랍니다. 제 테스트 기록에서 보면 **「block 분할과 register 대량 사용」** 그리고 **「메모리를 주소가 연속되도록 재구성」** 하는 것이 성능 향상의 핵심이었습니다(물론 시간 절약을 위해 이 부분을 학습하지 않아도 큰 문제는 없으며, 이후 설명에 영향을 주지 않습니다).

# 0x3. 더 우수한 GEMM 최적화

이전 절의 성능 최적화 그래프에서, 행렬이 비교적 작을 때 GEMM의 gflops가 그리 높지 않다는 것을 볼 수 있었습니다. 따라서 이 알고리즘에는 여전히 큰 최적화 여지가 있다고 추측할 수 있습니다.

이 절에서는 가오(Gao) 형님이 예전에 작성하신 GEMM을 소개하겠습니다. shape이 각각 (m,k)와 (k, 24)인 두 행렬이 주어졌을 때, 이 두 행렬의 곱을 구하는 것입니다. 행렬을 L1 Cache에 넣을 수 있도록, 여기서는 m과 k를 각각 적절한 값으로 설정합니다.

먼저 이 프로그램을 실행해 보고 gflops가 하드웨어 부동소수점 피크 대비 얼마나 도달하는지 확인해 보겠습니다. 테스트 결과는 다음과 같습니다.
    
    
    sgemm_kernel_x64_fma(24, 24, 64): time = 1.018230 us, perf = 72.407987 GFLOPS.  
    

첫 번째 절에서 측정한 하드웨어 부동소수점 피크는 다음과 같습니다.
    
    
    fma fp32 perf: 73.0070 gflops.  
    

이 GEMM의 gflops가 하드웨어 부동소수점 피크의 99%에 도달했음을 알 수 있습니다. 예전에 즈후(Zhihu)의 Lijiaoqiao Tiaoshui Guanjun이 이 알고리즘에 대해 자세히 설명한 적이 있습니다. https://zhuanlan.zhihu.com/p/383115932 를 참고하시기 바랍니다. 코드를 이해하기 어려운 독자는 한번 보시면 좋고, 여기서는 코드의 원리를 반복하지 않겠습니다.


당시에 저는 여러 최적화 방법을 찾아보고 스스로 고민도 해봤지만, L1 Cache 안에서 90% 이상의 하드웨어 활용률에 도달할 수 없었습니다. 이는 전적으로 제 프로그램에 큰 read/write 중복이 남아 있었기 때문이고, 이를 잘 해결할 방법을 떠올리지 못했기 때문입니다. 제 사고방식은 항상 block 크기를 고정한 뒤 k 차원을 순회하면서 한 번에 여러 행과 여러 열을 계산하는 것이었습니다. 매번 계산할 때마다 register를 거의 가득 사용했지만, **「당시에는 한 가지 문제, 즉 이 과정에서 다른 read/write 중복이 또 있는지, 그리고 현재 register 사용 방식이 합리적인지를 한 번도 진지하게 생각해 본 적이 없었습니다」**.

당시 제 방식을 복습해 보겠습니다. block 크기를 정한 다음, 매번 행렬 A의 8개 행에서 각각 8개 원소를 가져오고, 여기에 대응되는 행렬 B의 1개 열에서 8개 원소를 가져옵니다(여기서는 k에 대해 loop이기 때문). 여기서 8+1, 즉 9개의 register를 사용하고, 출력에 또 8개의 register가 필요하므로 총 17개의 ymm register를 사용합니다. 그런데 X86 아키텍처의 AVX는 16개의 256-bit register(YMM0~YMM15)만 제공합니다. 여기서 한 개를 더 사용하므로, 차선책으로 A의 데이터를 가져올 때 register 4개만 사용했고, 따라서 실제로 13개의 register만 사용했습니다. 코드는 https://github.com/BBuf/tvm_learn/blob/main/optimize_gemm/how_to_optimize_gemm/MMult_4x4_14.h 에서 볼 수 있습니다. 따라서 이 방식에서는 register를 다 채워 쓰지도 못하고, 동시에 read/write 중복도 많이 남아 있었기 때문에 성능이 높지 않다는 것은 충분히 이해되는 일이었습니다.

저는 가오(Gao) 형님의 코드가 제 계산 방식의 두 가지 핵심 문제를 정확히 해결했다고 봅니다. 이 코드는 16개의 ymm register를 완전히 사용하면서도 read/write 중복도 크게 줄였기 때문에, L1 Cache에서 99%의 하드웨어 활용률을 달성할 수 있었습니다. 이 코드는 매우 경험에 기반하고 tricky 한 코드로, 잘 짜 맞춰진 모습이 두드러집니다.

실험 편의를 위해 이 코드도 `tvm_learn` 저장소(https://github.com/BBuf/tvm_learn/tree/main/optimize_gemm/sgemm_kernel)에 복사해 두었습니다.

# 0x4. 컴파일러로 operator 최적화를 안내받을 수 있을까?

만약 여러분이 저처럼 효율적인 GEMM을 잘 짜 맞추는 데 익숙하지 않은데, GEMM operator를 좋은 성능까지 최적화해야 하는 상황이라면 어떻게 하시겠습니까?

그래서 제가 떠올린 것은, Ansor의 검색 결과를 기반으로 효율적인 GEMM 프로그램 작성을 안내받을 수 있지 않을까 하는 것이었습니다. 왜냐하면 Ansor는 AutoTVM처럼 사람이 직접 Scheduler를 지정하지 않아도 고성능 Scheduler를 생성할 수 있기 때문입니다.

먼저 GEMM operator를 검색하는 Ansor 프로그램을 작성했고, 코드는 다음 경로에 두었습니다. https://github.com/BBuf/tvm_learn/blob/main/optimize_gemm/auto_scheduler/gemm.py.

먼저 TVM Docs에서 제시한 전형적인 설정대로 검색을 돌려 보고, 현재 검색된 최적 프로그램의 GFlops를 확인해 보겠습니다.
    
    
    Execution time of this operator: 0.005 ms  
    GFlops:  14.80202901153263  
    

조금 낮네요. 음, 당황하지 말고 한 번 더 시도해 봅시다.

위의 Ansor 프로그램에서 조정 가능한 hyperparameter는 주로 다음 부분에 있습니다.
    
    
    log_file = "gemm.json"  
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)  
    tune_option = auto_scheduler.TuningOptions(  
        num_measure_trials=10,  # change this to 1000 to achieve the best performance  
        runner=measure_ctx.runner,  
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],  
        verbose=2,  
    )  
      
    

여기에 `num_measure_trials`를 1000으로 설정하면 최고 성능을 얻을 수 있다는 주석이 있는 것을 볼 수 있습니다. 이 parameter를 바꿔서 결과가 나아지는지 보겠습니다. 20분간 검색한 결과는 다음과 같습니다.
    
    
    Execution time of this operator: 0.004 ms  
    GFlops:  18.229696212591662  
    

결과가 약간 좋아지긴 했지만, gflops는 부동소수점 피크의 25% 정도밖에 되지 않습니다. 게다가 TVM은 여러 스레드를 사용했을 가능성이 있는데, 우리가 방금 측정한 피크는 단일 스레드 기준입니다.

그래서 Ansor 논문에서 X86에서의 단일 operator 최적화 능력 Benchmark 그래프를 가져왔습니다. 여기서 NRM이 2D GEMM을 의미합니다. 다만 아쉽게도 논문은 이 행렬의 크기를 언급하지 않습니다 QAQ. 이 그래프를 보면 Ansor는 GEMM 최적화에서 매우 강력한데, 그렇다면 여기서는 왜 기대한 결과가 나오지 않았을까요? 제 생각에는, 행렬이 매우 작은 경우에는 Ansor의 많은 scheduler들(예: cache_read, parallel, reorder)이 별다른 이득을 가져오지 않습니다. 왜냐하면 이때는 register를 가득 채워 쓰는지와 계산 중복을 제거하는 것이 관건이기 때문입니다. 그래서 행렬이 비교적 클 때 Ansor의 효과가 더 좋을 것이라고 추측합니다.

![이미지](images/img_04.png)Ansor 단일 operator의 튜닝 BenchMark

작은 행렬에서 성능이 평범하다면, Ansor가 큰 행렬에서는 더 나은 gflops를 얻을 수 있을까요? 계속 시도해 보겠습니다. 행렬의 m, n, k를 각각 2048, 24, 2048로 설정하고, `num_measure_trials`를 100으로, `target = tvm.target.Target("llvm -mcpu=skylake-avx512")`로 설정한 뒤 최종 gflops를 보겠습니다.
    
    
    Execution time of this operator: 0.319 ms  
    GFlops:  577.5124321897242  
    

577.5GFlops!!! 참고로, 여기서 llvm로 코드를 생성할 때 avx512 명령어 집합을 사용했습니다.

결과를 더 정확하게 비교하기 위해, CPU의 모든 스레드를 사용하여 부동소수점 피크를 다시 측정할 필요가 있습니다. 이 CPU의 코어 수는 16이므로, 앞서 gflops를 측정하던 방식대로 `./cpufp 16`을 실행하면 부동소수점 피크를 얻을 수 있습니다.
    
    
    Thread(s): 16  
    avx512_vnni int8 perf: 4283.9168 gops.  
    avx512f fp32 perf: 1070.8582 gflops.  
    avx512f fp64 perf: 535.5170 gflops.  
    fma fp32 perf: 1172.6536 gflops.  
    fma fp64 perf: 586.3434 gflops.  
    avx fp32 perf: 586.3019 gflops.  
    avx fp64 perf: 293.1215 gflops.  
    sse fp32 perf: 344.4352 gflops.  
    sse fp64 perf: 172.0953 gflops.  
    

여기서는 avx512f의 fp32 gflops를 기준으로 봐야 하는데, 꽤 괜찮은 결과로 약 54% 정도의 하드웨어 활용률이 나옵니다.

여기서 왜 16스레드의 부동소수점 피크와 비교했는가 하면, Ansor의 parallel scheduler 정책이 코어를 몇 개나 사용하는지 알아내지 못했기 때문에, 기본적으로 CPU의 스레드를 모두 채워 쓴다고 가정했습니다. 만약 잘 아시는 분이 계시다면 이 데이터를 업데이트할 수 있도록 알려주시면 좋겠습니다. 실제 하드웨어 활용률은 조금 더 좋을 수 있습니다.

# 0x5. 비교 실험 (Ansor vs 수동 block 분할)

여전히 위의 큰 행렬 크기인 m, n, k를 각각 2048, 24, 2048로 설정합니다. 0x3 절의 가오(Gao) 형님 GEMM 프로그램을 직접 실행하면서, m과 k를 2048로 설정합니다(그 프로그램에서 24는 고정되어 있습니다). 결과는 다음과 같습니다.
    
    
    sgemm_kernel_x64_fma(1024, 24, 1024): time = 741.568826 us, perf = 67.871850 GFLOPS.  
    

같은 행렬 크기에서, 수동으로 설계한 Kernel은 무려 90% 정도의 하드웨어 활용률을 보입니다.

같은 행렬 크기에 대해 Ansor는 50%+의 하드웨어 활용률을, 손으로 정밀하게 설계한 GEMM은 90%+의 하드웨어 활용률을 달성합니다. 이는 GEMM operator 최적화에서 경험 있는 사람의 수동 최적화가 Ansor의 성능보다 훨씬 좋다는 것을 보여줍니다.

# 0x6. 결론

위의 실험에서 보면, Ansor 기반의 GEMM operator 최적화는 여전히 손으로 정밀하게 설계한 Kernel만큼은 못합니다. 따라서 Ansor가 우리에게 높은 수준의 operator 최적화를 안내해 주기를 기대하는 것은 어렵습니다. Ansor 논문에서 알 수 있듯이, GEMM 같은 계산 집약형 operator를 최적화할 때는 고정된 규칙이 있습니다. 예를 들어 GEMM의 Scheduler로 "SSRSRS" tile 구조를 사용하는데, 여기서 "S"는 공간 loop의 한 tile 레벨을, "R"은 reduction loop의 한 tile 레벨을 의미합니다. 이 tile이 바로 위에서 언급한 block 분할입니다. 따라서 Ansor도 최적화 과정에서 사람의 경험을 많이 차용하고 있으며, 게다가 단지 operator의 Scheduler를 조정하는 측면에 한정되어 있어, register 사용 방식을 바꾸거나 pipeline을 조정하는 것은 할 수 없습니다.

다만 어떤 사람의 수준이 한정되어 있거나 Kernel 최적화에 그리 익숙하지 않고, 단지 호기심에 몇 가지 방법을 시도해 보고 싶은데, 직접 operator를 최적화하여 얻은 성능이 이런 자동 튜닝 도구만 못하다면, 이런 튜닝 결과가 보여주는 Scheduler를 바탕으로 자기 코드의 Scheduler가 도대체 어디에서 문제가 있는지 생각해 볼 수 있습니다. block 크기가 부적절한지, 아니면 지역성이 떨어지는지? 자동 검색된 코드의 Scheduler 목표를 보면 기계가 주는 영감을 얻을 수도 있고, 우리 코드를 개선하는 데 도움이 될 수 있습니다.

operator 최적화에서는 Scheduler 외에도 register 사용 시점, 하드웨어별 명령어 재배치, 명령어 집합 자체의 선택 등이 모두 최종 성능에 영향을 미치는 요인입니다. 그러나 이런 것들은 TVM 안에서 자동으로 처리하기 어렵고, LLVM 컴파일러에 맡기거나 사람이 직접 operator 최적화 코드를 작성할 수밖에 없습니다.

요컨대, 이 글은 필자가 몇 가지 작은 실험을 관찰하면서 얻은 내용으로, 제 견해가 모두 옳다고 보장할 수는 없습니다. 오류가 있다면 지적해 주시고 함께 교류해 주시면 감사하겠습니다.
