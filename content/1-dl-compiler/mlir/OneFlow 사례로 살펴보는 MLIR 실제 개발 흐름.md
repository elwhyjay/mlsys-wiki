# OneFlow 사례로 살펴보는 MLIR 실제 개발 흐름

### 머리말

최근 동료 Shenghang의 도움을 받아 OneFlow IR 관련 개발을 진행하면서 MLIR 실행 부분에 대한 새로운 통찰을 얻게 되어 이를 공유하고자 합니다. 저는 이전에 OneFlow IR의 전체 아키텍처를 이해하는 데 많은 시간을 투자했지만(제 Toy Tutorials 시리즈를 참고하세요), OneFlow IR의 JIT 실행 부분에 대해서는 늘 의문이 있었습니다. 최근 OneFlow와 MLIR의 통합 구현을 Job(OneFlow의 작업 함수로, 디바이스를 고려하지 않으면 컴퓨팅 그래프로 이해할 수 있습니다) 기반으로 재구성하면서 Shenghang의 도움을 받아 전체 과정을 이해하게 되었습니다. 따라서 이 글에서는 OneFlow와 MLIR의 결합 방식, OneFlow IR에 그래프 레벨 Pass를 추가하는 방법, OneFlow Operations가 어떻게 자동으로 MLIR Operations로 변환되는지, 그리고 OneFlow IR이 MLIR을 활용하여 연산 속도를 어떻게 끌어올리는지를 소개합니다. 저는 MLIR에 대한 지식이 아직 충분하지 않으며, 두 달 전부터 학습을 시작했습니다. 오류가 있으면 언제든지 지적해 주세요. 이 글은 https://github.com/Oneflow-Inc/oneflow 및 https://github.com/BBuf/tvm_mlir_learn 과 관련이 있습니다. 관심 있으시면 해당 프로젝트들을 팔로우하고 별을 눌러주세요.

> 이 글에서 언급된 'Op'와 'Operation'이라는 용어는 동일한 것을 의미하며 엄밀히 구분되지 않습니다.

### OneFlow는 MLIR과 어떻게 통합되나요?

OneFlow에 MLIR을 IR로 도입하면 여러 가지 이점이 있습니다. OneFlow에서 수동으로 작성하던 C++ operator 정의를 대체하여 개발 복잡도를 줄여줄 뿐만 아니라, operator 정의에서 발생하던 컨테이너 관련 오버헤드도 낮춰줍니다. 또한 MLIR이 제공하는 인프라(즉, 다양한 Dialect)를 통해 컴퓨팅 그래프 연산을 가속할 수 있습니다. 이 컴퓨팅 그래프는 Eager 또는 Lazy 컴퓨팅 그래프일 수 있습니다. Eager 컴퓨팅 그래프 기반 MLIR 가속과 관련된 `oneflow.jit.xxx` 작업은 아직 공식적으로 출시되지 않았기 때문에, OneFlow와 MLIR 통합 과정을 설명하기 위해 Lazy 컴퓨팅 그래프(Job)를 예시로 사용하겠습니다.

먼저 MLIR을 활성화하여 OneFlow를 컴파일해야 합니다. 컴파일 명령은 다음과 같습니다.
    
    
    git clone git@github.com:Oneflow-Inc/oneflow.git  
    cd oneflow && mkdir build && cd build  
    cmake-C ../cmake/caches/cn/fast/mlir-cuda-75.cmake -DBUILD_TESTING=ON .. && ninja   
    

그 다음 이를 테스트하기 위한 예제를 작성할 수 있습니다.
    
    
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = '1'  
    os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = '1'  
      
    @flow.unittest.skip_unless_1n1d()  
    class TestFuseBiasAddGeLUCPUMLIR(oneflow.unittest.TestCase):  
        def test_fused_bias_add_gelu_graph(test_case):  
            data = np.random.randn(1, 2, 3)  
            bias_data = np.random.randn(2)  
            x = flow.tensor(data, dtype=flow.float32)  
            bias = flow.tensor(bias_data, dtype=flow.float32)  
            y_eager = flow.gelu(flow._C.bias_add(x, bias, axis=1))  
      
            class FuseBiasAddGeLUGraph(flow.nn.Graph):  
                def __init__(self):  
                    super().__init__()  
      
                def build(self, x):  
                    return flow.gelu(flow._C.bias_add(x, bias, axis=1))  
      
            bias_add_gelu = FuseBiasAddGeLUGraph()  
            y_lazy = bias_add_gelu(x)  
            test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))  
    

이 예제를 실행하면 현재 실행 디렉터리에 `ir_pass` 로그 폴더가 생성됩니다. 로그 폴더 안에는 OneFlow MLIR 최적화 전후의 컴퓨팅 그래프(`.prototxt`)와 MLIR 표현식(`*.mlir`)이 기록되어 있습니다. 또한 MLIR 표현식의 컴퓨팅 그래프를 `graphviz`로 시각화할 수 있는 `*.mlir.dot` 파일도 있습니다. OneFlow가 학습 작업을 수행하는 경우, 이 로그 폴더에는 순방향 컴퓨팅 그래프와 MLIR 표현식뿐만 아니라 역방향 컴퓨팅 그래프와 MLIR 표현식도 함께 포함된다는 점에 유의해야 합니다. 따라서 MLIR은 신경망의 전체 연산 과정에서 사용할 수 있으며, 이는 순방향 추론 프레임워크와의 큰 차이점입니다. 또한 학습 속도 향상에도 기여할 수 있습니다.

`oneflow/api/python/ir.cpp` 코드에는 다음 두 줄이 포함되어 있습니다.
    
    
    REGISTER_JOB_PASS("IRRoundTripBeforeAD", IRRoundTrip<kBeforeAD>);  
    REGISTER_JOB_PASS("IRRoundTrip", IRRoundTrip<kAfterAD>);  
    

`RoundTrip`("왕복")이란 OneFlow Job과 MLIR 간의 변환 과정을 의미하며, 역방향 변환의 전후로 보아 `kBeforeAD`와 `kAfterAD`로 이해할 수 있습니다. 여기서는 변환 과정을 OneFlow Job의 Pass로 등록함으로써 OneFlow 컴퓨팅 그래프와 MLIR 사이의 연결을 구축합니다. OneFlow 스크립트를 실행할 때 MLIR이 OneFlow 컴퓨팅 그래프 위에서 동작하도록 하려면 `ONEFLOW_MLIR_ENABLE_ROUND_TRIP=1` 환경 변수를 켜기만 하면 됩니다.

다음으로, OneFlow 컴퓨팅 그래프와 MLIR 간의 연결을 구축한다는 것은 OneFlow 컴퓨팅 그래프의 Operations와 MLIR의 Operations 간의 일대일 변환을 수행한다는 의미입니다. MLIR Operation은 다양한 Dialect 레벨에서 정의됩니다. MLIR의 일반적인 접근 방식에 따라, 우리는 OneFlow Dialect를 구현하고 OneFlow Operations를 이 Dialect 레벨의 Operations로 일대일 매핑하는 기능을 구현했습니다. OneFlow Dialect 및 Operation의 정의는 여기서 자세히 다루지 않습니다. 자세한 내용은 공식 MLIR 문서(https://mlir.llvm.org/docs/OpDefinitions/)의 Dialect 및 ODS 섹션이나 TableGen 규칙을 기반으로 작성된 이전 글들을 참조하시기 바랍니다. 저는 이전에 `https://github.com/BBuf/tvm_mlir_learn`에서 OneFlow Dialect의 Op 정의와 MLIR Operation 정의를 결합하여 정리한 글(2부)을 작성한 바 있습니다. Dialect 및 Operation 정의 외에도 정의해야 할 사항들이 있는데, 예를 들어 OneFlow 데이터 type과 MLIR 데이터 type 간의 매핑 정의(`oneflow/ir/include/OneFlow/OneFlowEnums.td`), OneFlow Dialect Operation에 대한 일반적인 frontend 인터페이스 정의(`oneflow/ir/include/OneFlow/OneFlowEnums.td`) 등이 있습니다. 여기서는 Reshape Operation을 예로 들어, 이 Operation의 구성 요소를 간략히 살펴보겠습니다.
    
    
    def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {  
      let input = (ins  
        AnyType:$in  
      );  
      let output = (outs  
        AnyType:$out  
      );  
      let attrs = (ins  
        AnyI64ElementsAttr:$shape  
      );  
    }  
    

`OneFlow_ReshapeOp`에서 밑줄 앞부분은 Dialect의 이름이고, 밑줄 뒷부분은 해당 Dialect에 속하는 Operation의 이름입니다. 이 Operation은 `OneFlow_BaseOp` 베이스 클래스를 상속받아 제약 조건(Constraint)과 frontend 인터페이스를 선언합니다. 그 다음 Operation의 입력, 출력, 그리고 attribute를 정의하면 전체 과정이 완료됩니다. OneFlow Dialect Operation의 정의는 OneFlow User Op의 정의와 완전히 일치하므로, OneFlow와 MLIR 간 변환의 합법성을 보장합니다. OneFlow Reshape Op의 정의는 다음과 같습니다.
    
    
    REGISTER_USER_OP("reshape")  
        .Input("in")  
        .Output("out")  
        .Attr<Shape>("shape")  
        ...  
      
    

OneFlow Job과 MLIR 간 변환의 구체적인 구현은 `oneflow/ir/oneflow-translate`에 있으며, 주로 Job의 OpGraph를 순회하면서 노드와 엣지를 각각 처리하고 최종적으로 MLIR 표현식으로 변환하는 과정을 포함합니다. 또한, 연산 후 MLIR 표현식을 기반으로 Job을 다시 작성할 수도 있습니다. 전체 로직은 OneFlow Job OpGraph 안의 다양한 유형의 Operation과 엣지를 변환해야 하므로 다소 복잡합니다. 이 부분은 본 글의 핵심 주제가 아니므로 자세한 설명은 생략하겠습니다. 관심이 있으시면 코드를 직접 참조해 주세요.

### OneFlow IR은 어떻게 동작하나요?

위의 Operation 정의에서는 Reshape를 예로 들었습니다. 잠깐 살펴보면 `oneflow/ir/include/OneFlow/OneFlowOps.td`에 또 다른 사용자 정의 Operation인 `OneFlow_MlirJitOp`가 정의되어 있다는 것을 알 수 있습니다. 이 Operation은 MLIR 표현식을 실행하는 데 사용됩니다. `oneflow/ir/oneflow-extension/extension.cpp`에서는 MLIR이 제공하는 JIT 실행 엔진을 로드하고 결과로 나온 LLVM IR을 실행하기 위한 CPU 및 GPU kernel(소스 코드 제공)을 구현합니다. 그렇다면 LLVM IR은 어떻게 생성될까요? OneFlow MLIR 표현식을 단계적으로 lowering하여 얻어지며, 구체적인 lowering 과정은 다음과 같습니다.
    
    
    void AddLowerToLinalgMemRefPasses(PassManager& pm) {  
      pm.addPass(createLowerOneFlowToTosaPass());            // lower-oneflow-to-tosa  
      pm.addPass(createCSEPass());                           // cse  
      pm.addNestedPass<FuncOp>(tosa::createTosaToLinalg());  // tosa-to-linalg-on-tensors  
      auto p = createLinalgElementwiseOpFusionPass();  
      assert(p->initializeOptions("allow-folding-unit-dim-reshapes=true").succeeded());  
      pm.addNestedPass<FuncOp>(std::move(p));                     // linalg-fuse-elementwise-ops  
      pm.addNestedPass<FuncOp>(createLinalgBufferizePass());      // linalg-bufferize  
      pm.addNestedPass<FuncOp>(createTensorBufferizePass());      // tensor-bufferize  
      pm.addPass(createTensorConstantBufferizePass());            // tensor-constant-bufferize  
      pm.addPass(createFuncBufferizePass());                      // func-bufferize  
      pm.addPass(createBufferResultsToOutParamsPass());           // buffer-results-to-out-params  
      pm.addPass(createCanonicalizerPass());                      // canonicalize  
      pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());  // finalizing-bufferize  
    }  
      
    LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module) {  
      mlir::PassManager pm(context);  
      AddLowerToLinalgMemRefPasses(pm);  
      pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());  // convert-linalg-to-loops  
      pm.addNestedPass<FuncOp>(createLowerToCFGPass());            // convert-scf-to-std  
      pm.addPass(createConvertLinalgToLLVMPass());                 // convert-linalg-to-llvm  
      pm.addPass(createMemRefToLLVMPass());                        // convert-memref-to-llvm  
      pm.addPass(createLowerToLLVMPass());                         // convert-std-to-llvm  
      pm.addPass(createReconcileUnrealizedCastsPass());  
      return pm.run(module);  
    }  
    

OneFlow Dialect가 먼저 Tosa Dialect로, 그 다음 Linalg Dialect로, Loop Dialect로, 그리고 최종적으로 LLVM IR로 점진적으로 lowering되는 것을 볼 수 있습니다. 이러한 점진적인 lowering 과정에서 Linalg Dialect가 제공하는 중첩 루프 변환과 같은 최적화 기회를 활용하여 최종 IR의 성능을 향상시킬 수 있습니다. 여기서의 lowering 과정은 OneFlow가 `MlirJitOp` kernel을 호출할 때 트리거되며, 이는 `oneflow/ir/oneflow-extension/extension.cpp`에 있습니다. 이 호출 자체도 MLIR Pass로서 최적화 프로세스에 추가됩니다. JIT 호출 프로세스 Pass의 구현은 다음과 같이 단순화할 수 있습니다.
    
    
    class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {  
      void runOnOperation() override {  
        Operation* op = getOperation();  
        RewritePatternSet patterns(op->getContext());  
        oneflow::populateFuserPasses(patterns);  
        (void)applyPatternsAndFoldGreedily(op, std::move(patterns));  
      }  
    };  
      
    std::unique_ptr<Pass> createOutlineJitFunctionPass() {  
      return std::make_unique<OutlineJitFunctionPass>();  
    }  
      
    LogicalResult ApplyRoundTripPatterns(RoundTripOneFlowJobWrapperInterface& job_wrapper,  
                                         MLIRContext* context, OwningModuleRef& module) {  
      mlir::PassManager pm(context);  
      pm.addNestedPass<mlir::FuncOp>(::mlir::createCanonicalizerPass());  
      if (job_wrapper.IsLastIRPass() && std::getenv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS") != nullptr) {  
        pm.addPass(oneflow::createOutlineJitFunctionPass());  
      }  
      ...  
    }  
    

하지만 이 과정에는 여전히 해결해야 할 두 가지 문제가 남아 있습니다.

  * 첫 번째 문제는 Op fusion을 어떻게 수행할 것인가입니다. 위에서 보여준 JIT 실행 흐름은 단순히 연속적인 lowering만 고려합니다. 그렇다면 OneFlow Dialect 안에 fusion이 가능한 여러 Op가 있을 때는 어떻게 해야 할까요? 간단합니다. MLIR DRR 규칙을 사용하여 `oneflow/ir/include/OneFlow/OneFlowPatterns.td`에 TableGen 문법으로 일련의 fusion Pattern을 작성하면 됩니다. 예를 들어, `bias_add`와 `gelu` 두 Op를 OneFlow에서 하나의 `fused_bias_add_gelu` Op로 fusion할 수 있다면 다음과 같은 규칙을 작성할 수 있습니다.


    
    
    def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;  
    def FusedBiasAddGeluPattern : Pat<  
      (  
        OneFlow_GeluOp : $gelu_op  
        (  
          OneFlow_BiasAddOp  
            $a,  
            $b,  
            $bias_add_op_name,  
            $bias_add_device_tag,  
            $bias_add_device_name,  
            $bias_add_scope_symbol_id,  
            $bias_add_hierarchy,  
            $axis  
        ),  
        $gelu_op_name,  
        $gelu_device_tag,  
        $gelu_device_name,  
        $gelu_scope_symbol_id,  
        $gelu_hierarchy  
      ),  
      (OneFlow_FusedBiasAddGeluOp $a, $b,  
        $gelu_op_name,  
        $gelu_device_tag,  
        $gelu_device_name,  
        $gelu_scope_symbol_id,  
        $gelu_hierarchy,  
        $axis  
      ),  
      [  
        (IsGPU $bias_add_device_tag),  
        (IsGPU $gelu_device_tag)  
      ]  
    >;  
    

여기서는 MLIR의 DRR 규칙에 따라 표현식 매칭과 rewrite을 수행합니다. 현재 실행 중인 device가 GPU이고 두 Operation이 각각 `gelu`와 `bias_add`인 경우, 이들을 하나의 `fused_bias_add_gelu_op`로 합치는 것을 볼 수 있습니다. CUDA 환경에서는 이를 통해 메모리 read/write 연산 횟수를 줄이고 실행 효율을 높일 수 있습니다.

  * 두 번째 문제는 OneFlow Operation이 MLIR 인프라의 최적화를 더 많이 활용할 수 있도록 하는 방법입니다. 다단계 Dialect lowering 과정에서 OneFlow MLIR 표현식의 각 하위 함수는 하위 Dialect로 변환됩니다. 처음에는 Tosa Dialect로 변환되지만, 만약 이 하위 함수 안의 Operation이 Tosa Dialect로 변환하는 방법을 정의하지 않았다면 Tosa Dialect로 변환될 수 없습니다. 결과적으로 Linalg Dialect로 더 이상 진행될 수 없어, 반복적인 변경을 통해 얻을 수 있는 최적화 기회를 놓치게 됩니다(이는 TVM의 scheduler 최적화와 유사하다고 생각합니다). 이 문제를 해결하기 위해, 우리는 Tosa로 변환해야 하는 Operation 또는 Pattern을 함수로 추출하는 추가 Pass를 정의해야 합니다. 이 함수 안의 모든 OneFlow Operation은 Tosa로 변환될 수 있으며, 그 후 이 함수를 호출하는 OneFlow MLIR JIT Operation이 생성됩니다.


    
    
    def IsNotNestedInJit: Constraint<CPred<"(!$0.getDefiningOp()->getParentOfType<::mlir::FuncOp>()->hasAttr(\"llvm.emit_c_interface\"))">, "">;  
    def OutlineMulCast : NativeCodeCall<"::mlir::oneflow::OutlineMulCast($_builder, $0, $1)">;  
    // TODO: 가능하면 attr 바인딩 제거  
    def MulCastPattern : Pat<  
      (  
        OneFlow_ScalarMulByTensorOp : $mul_op  
        (  
          OneFlow_CastOp : $cast_op  
            $cast_x,  
            $cast_op_name,  
            $cast_device_tag,  
            $cast_device_name,  
            $cast_scope_symbol_id,  
            $cast_hierarchy,  
            $cast_dtype  
        ),  
        $scalar,  
        $mul_op_name,  
        $mul_device_tag,  
        $mul_device_name,  
        $mul_scope_symbol_id,  
        $mul_hierarchy  
      ),  
      (OutlineMulCast $mul_op, $cast_op),  
      [  
        (IsNotNestedInJit $mul_op)  
      ]  
    >;  
      
    ::llvm::SmallVector<::mlir::Value, 4> OutlineMulCast(::mlir::PatternRewriter& rewriter,  
                                                         mlir::OpResult mul_res,  
                                                         mlir::OpResult cast_res) {  
      if (auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_res.getDefiningOp())) {  
        if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {  
          // TODO: fusion되는 op들로부터 jit op의 op name을 생성하는 함수를 추출  
          SmallString<64> op_name_storage;  
          auto op_name =  
              (cast_op.op_name() + "__FUSE__" + mul_op.op_name()).toStringRef(op_name_storage);  
          SmallVector<::mlir::Value, 2> operands;  
          operands.push_back(cast_op.in());  
          operands.push_back(mul_op.scalar());  
          SmallVector<::mlir::Value, 1> results;  
          results.push_back(mul_op.y());  
          NamedAttrList attributes =  
              GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), mul_op);  
          SmallVector<Operation*, 4> ops = {cast_op, mul_op};  
          auto function =  
              GetOrInsertFuncOp(rewriter, mul_op->getLoc(), op_name, operands, results, ops);  
          auto created = rewriter.create<MlirJitOp>(mul_op.getLoc(), function, attributes, operands);  
          assert(DumpAssembly(rewriter, created).succeeded());  
          cast_op->dropAllUses();  
          cast_op.erase();  
          return created->getResults();  
        }  
      }  
      return {};  
    }  
      
    void populateFuserPasses(::mlir::RewritePatternSet& patterns) {  
      patterns.add<MulCastPattern>(patterns.getContext());  
    }  
    

이 과정은 MulCast Pattern을 사용하여 OneFlow Dialect를 Tosa Dialect로 수동 변환하는 작업을 포함합니다. 마지막으로, 이 변환 과정을 최적화 프로세스에 추가하면 전체 흐름이 마무리됩니다. MLIR 표현식의 Pattern은 Tosa 및 Linalg 레벨 모두에서 Dialect 변환을 거치게 되므로 최적화 기회를 얻을 수 있습니다.

### 정리

이 글에서는 OneFlow를 예시로 MLIR의 실제 동작 과정, 즉 딥러닝 프레임워크의 컴퓨팅 그래프를 어떻게 실행하고 MLIR을 통해 가속화하는지를 설명했습니다. 현재 제 이해가 부족한 부분이 있을 수 있으므로, 여러분의 비판과 수정을 환영합니다.
