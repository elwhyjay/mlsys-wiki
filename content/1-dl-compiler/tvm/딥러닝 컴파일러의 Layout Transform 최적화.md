# 딥러닝 컴파일러의 Layout Transform 최적화

> 더 많은 딥러닝 [컴파일러](<https://zhida.zhihu.com/search?content_id=228210363&content_type=Article&match_order=1&q=%E7%BC%96%E8%AF%91%E5%99%A8&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiLnvJbor5HlmagiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjgyMTAzNjMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.S87qW99rQzRHta9u5G62og25qa43x-bKwEydxdB5wmE&zhida_source=entity>) 관련 지식은

[https://github.com/BBuf/tvm_mlir_learngithub.com/BBuf/tvm_mlir_learn](<https://link.zhihu.com/?target=https%3A//github.com/BBuf/tvm_mlir_learn>)

에서 찾을 수 있습니다. 또한 cuda 학습 저장소도 함께 운영하고 있습니다.

[https://github.com/BBuf/how-to-optim-algorithm-in-cudagithub.com/BBuf/how-to-optim-algorithm-in-cuda](<https://link.zhihu.com/?target=https%3A//github.com/BBuf/how-to-optim-algorithm-in-cuda>)

그리고 딥러닝 프레임워크(PyTorch와 OneFlow)를 어떻게 학습할 것인가에 관한 학습 저장소도 있습니다.

[https://github.com/BBuf/how-to-learn-deep-learning-frameworkgithub.com/BBuf/how-to-learn-deep-learning-framework](<https://link.zhihu.com/?target=https%3A//github.com/BBuf/how-to-learn-deep-learning-framework>)

필요하신 분들은 **star를 눌러주세요**.

[![이미지](./深度学习编译器之Layerout Transform优化 - 知乎_files/v2-f0ed856bc8489c9380aca253034991f5_180x120.jpg)how-to-optim-algorithm-in-cuda/large-language-model-note at master · BBuf/how-to-optim-algorithm-in-cudagithub.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model-note](<https://link.zhihu.com/?target=https%3A//github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model-note>)

이 디렉터리에는 LLM 학습 및 추론과 관련된 일련의 글이 모여 있습니다.

본문 설명에서 일부 interface와 Interface가 혼용되어 사용되는데, 두 표현 모두 MLIR의 Interface를 의미하는 동일한 대상입니다.


## **0x0. 배경**

딥러닝 컴파일러 최적화 작업에 대한 해설을 이어갑니다. 이번 글에서 소개할 내용은 OneFlow 시스템에서 어떻게 MLIR을 기반으로 Layout Transform을 구현했는지에 관한 것입니다. 2D [convolutional neural network](<https://zhida.zhihu.com/search?content_id=228210363&content_type=Article&match_order=1&q=%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiLljbfnp6_npZ7nu4_nvZHnu5wiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjgyMTAzNjMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.MAZdGuEZgG1EjdSb6l1Z5yWuff8zBv5V4tTQxIckeNM&zhida_source=entity>)에서는 NCHW 데이터 포맷 외에 일반적으로 NHWC 데이터 포맷도 존재하며, conv 연산의 경우 NHWC 포맷으로 계산하는 것이 더 좋은 성능을 얻을 수 있습니다. 그러나 딥러닝 네트워크 학습은 일반적으로 NCHW로 진행되며, 보통은 추론 시에만 NCHW에서 NHWC로의 Layout Transform을 수행합니다. 여기에는 두 가지 문제가 있습니다. 먼저 Conv2D와 같은 op의 경우, NCHW 방식으로 학습되었을 때 저장된 weight 포맷은 [out_channels, in_channels, *kernel_size]이지만 NHWC 포맷으로 추론하려면 weight 포맷을 변환해야 합니다. 다음으로 weight가 없는 op의 경우에도, conv op 앞뒤에 삽입되는 Transpose 연산으로 인한 추가 오버헤드를 줄이기 위해 가능한 한 NHWC 연산을 지원하도록 만들어야 합니다. 예를 들어, x->conv->relu->conv->relu->out 형태의 작은 네트워크가 있다고 가정합시다. 이를 NHWC 포맷으로 실행하려면 두 conv의 weight를 변경하는 것 외에도, conv 앞뒤에 transpose를 삽입하여 conv op에 입력되는 데이터 포맷을 수정해야 합니다. 즉, **x- >transpose(0, 2, 3, 1)->conv->transpose(0, 3, 1, 2) -> relu -> transpose(0, 2, 3, 1)->conv->transpose(0, 3, 1, 2) -> relu->out** 가 됩니다. 그런 다음 주의 깊은 독자라면 알아챌 수 있듯이, 사실 여기에는 많은 중복된 Transpose가 있습니다. ReLU는 NHWC 포맷으로 연산하는 것을 지원하기 때문에 이 네트워크는 **x- >transpose(0, 2, 3, 1)->conv->relu->conv->relu->transpose(0, 3, 1, 2)->out** 으로 단순화할 수 있습니다. 이렇게 하면 Transpose op의 오버헤드를 절반으로 줄일 수 있습니다.

transpose 단순화를 수행하는 이유는 transpose op 자체에도 실행 및 스케줄링 오버헤드가 있기 때문입니다. transpose의 개수를 최대한 줄이지 않으면 NHWC로 변경하여 얻는 계산 가속이 Transpose의 오버헤드에 가려질 수 있습니다. 우리는 OneFlow를 기반으로 위와 같은 Layout Transform 최적화를 구현했으며, 아래에 테스트 결과를 제시합니다.

V100에서 이 최적화에 대해 테스트를 수행했고, 테스트 코드는

[](<https://link.zhihu.com/?target=https%3A//github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/test/OneFlow/auto_nhwc/test_resnet101_benchmark.py>)

에 있으며, 성능 결과는 다음과 같습니다.

  * nn.Graph의 AMP 옵션을 활성화.
  * 네트워크는 ResNet101을 선택하고 forward 추론을 수행.

batch_size| nchw| auto nhwc
---|---|---
16| 14s| 13s
32| 24s| 22s
64| 44s| 38s

BatchSize=64일 때 13.6%의 가속을 얻었으며, BatchSize가 작아질수록 가속비는 줄어들지만 항상 일정한 가속이 유지됩니다. 주의할 점은, 여기서 weight 파라미터 부분에 대해서는 미리 transpose를 수행했기 때문에 이 부분에는 추가 오버헤드가 없다는 것입니다. 실제로는 상수 폴딩 방식을 사용하여 처리했으며, 이에 대해서는 다음 글에서 설명하겠습니다.

## **0x1. 구현 해설**

구현에서는 주로 세 가지 문제를 해결해야 합니다. 첫 번째는 어떤 op가 NHWC 연산을 지원하는지를 어떻게 판단할 것인가, 두 번째는 Transpose op를 어떻게 삽입할 것인가, 세 번째는 중복된 Transpose 쌍을 어떻게 제거할 것인가입니다.

## **0x1.1 Interface를 기반으로 어떤 op가 NHWC 연산을 지원하는지 결정하기**

OneFlow에서 어떤 Op가 NHWC 계산을 지원하도록 하려면, Op 정의 시 NCHWCompatibleInterface를 선언하기만 하면 됩니다. conv를 예로 들면 다음과 같습니다.


    def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {}


여기서 DeclareOpInterfaceMethods는 이 Operator가 NCHWCompatibleInterface 인터페이스를 구현했음을 나타냅니다. 이 인터페이스는 NCHW 포맷과 호환되는 Operator가 구현해야 할 메서드를 정의합니다.

다른 임의의 Op가 NHWC 연산을 지원하도록 하려면, 이 인터페이스를 정의하고 인터페이스의 멤버 함수를 오버라이드하기만 하면 됩니다. 다음으로 NCHWCompatibleInterface의 정의를 살펴보겠습니다.


    def NCHWCompatibleInterface : OpInterface<"NCHWCompatible"> {
      let description = [{
        Interface of NCHW compatibility
      }];

      let methods = [
        InterfaceMethod<"",
            "bool", "IsNCHW", (ins)
        >,
        InterfaceMethod<"Create NHWC op and return the new op's results to be transposed",
            "llvm::SmallVector<mlir::Value, 4>", "NchwToNhwc", (ins "llvm::SmallVector<mlir::Value, 4>": $transposed_inputs, "PatternRewriter&": $rewriter)
        >,
        InterfaceMethod<"",
            "llvm::DenseSet<mlir::Value>", "OperandsToTranspose", (ins)
        >,
        InterfaceMethod<"",
            "llvm::DenseSet<mlir::Value>", "ResultsToTranspose", (ins)
        >,
      ];
      let cppNamespace = "::mlir::oneflow";
    }


이 인터페이스는 OpInterface 인터페이스를 상속하며, OpInterface는 MLIR 프레임워크에서 Operator Interface를 기술하는 베이스 클래스입니다. NCHWCompatibleInterface는 NCHW 포맷과 호환되는 Operator Interface를 나타냅니다. NCHWCompatibleInterface는 다음과 같은 메서드를 정의합니다.

  * IsNCHW: bool 값을 반환하며, 현재 Operator가 어떤 조건에서 NCHW 포맷의 데이터를 입력으로 처리하는지를 나타냅니다.
  * NchwToNhwc: Transpose 후의 입력과 rewriter를 받아 NCHW 포맷에서 NHWC 포맷으로의 변환에 사용됩니다.
  * OperandsToTranspose: Transpose가 필요한 입력 값들의 집합을 반환합니다.
  * ResultsToTranspose: Transpose가 필요한 출력 값들의 집합을 반환합니다.



다음으로 Conv2D Op에 해당하는 NCHWCompatibleInterface 인터페이스의 구현을 살펴봅시다.


    bool Conv2DOp::IsNCHW() { return this->getDataFormat().str() == "channels_first"; }

    llvm::DenseSet<Value> Conv2DOp::OperandsToTranspose() {
      if (this->get_addToOutput()) {
        return {this->getIn(), this->getWeight(), this->get_addToOutput()};
      } else {
        return {this->getIn(), this->getWeight()};
      }
    }

    llvm::DenseSet<Value> Conv2DOp::ResultsToTranspose() { return {this->getOut()}; }

    llvm::SmallVector<Value, 4> Conv2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                     PatternRewriter& rewriter) {
      auto conv_op = *this;
      SmallVector<Value, 4> operands;
      operands.push_back(value[0]);
      operands.push_back(value[1]);
      if (conv_op.getBias()) operands.push_back(conv_op.getBias());
      if (this->get_addToOutput()) { operands.push_back(value[2]); }
      NamedAttrList attributes = conv_op->getAttrs();
      attributes.set(conv_op.getDataFormatAttrName(), rewriter.getStringAttr("channels_last"));
      auto res = rewriter
                     .create<oneflow::Conv2DOp>(conv_op.getLoc(), getNHWCResultTypes(conv_op), operands,
                                                attributes)
                     ->getResults();
      llvm::SmallVector<Value, 4> results;
      results.push_back(res[0]);
      return results;
    }


여기서 IsNCHW 메서드는 bool 값을 반환하여 해당 Conv2DOp Operation이 NCHW 포맷을 사용하는지 여부를 나타냅니다. 이는 Operation의 data_format 속성을 검사하여 판단합니다. OperandsToTranspose 메서드는 Transpose가 필요한 입력 값들의 집합을 반환합니다. Conv2DOp의 경우 주요 입력에는 input, weight, bias(선택), addto_output(선택)이 포함되며, bias는 Transpose가 필요 없고 이 addto_output은 op fusion을 위한 OneFlow 특유의 출력이므로 독자는 무시해도 됩니다. ResultsToTranspose 메서드는 Transpose가 필요한 출력 값들의 집합을 반환합니다. Conv2DOp는 출력이 하나뿐이므로 출력 feature map의 값을 반환합니다. NchwToNhwc 메서드는 NCHW 포맷의 입력 값과 rewriter를 받아 NHWC 포맷의 결과 값을 반환합니다. 새로운 Conv2DOp Operation을 생성하고 data_format 속성을 channels_last로 설정함으로써 NCHW에서 NHWC로의 변환을 구현합니다.

## **0x1.2 Transpose op 삽입하기**

다음 단계는 네트워크 내의 op들에 탐욕적으로 Transpose op를 삽입하는 것입니다. 여기서의 아이디어는 네트워크 내의 모든 op에 대해 가능한 한 앞뒤에 Transpose를 각각 하나씩 삽입하는 것입니다. 이렇게 해야 Transpose 쌍을 제거할 때 최적의 해를 얻을 수 있습니다. 네트워크 내의 op에 Transpose를 삽입하는 로직은 다음 Pattern 코드에 설명되어 있습니다.


    struct AutoNhwcPattern : public OpInterfaceRewritePattern<NCHWCompatible> {
      explicit AutoNhwcPattern(mlir::MLIRContext* context)
          : OpInterfaceRewritePattern<NCHWCompatible>(context, /*benefit=*/1) {}

     public:
      LogicalResult matchAndRewrite(NCHWCompatible op, PatternRewriter& rewriter) const override {
        if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
          for (mlir::Value operand : op.OperandsToTranspose()) {
            if (operand.getType().cast<mlir::RankedTensorType>().getShape().size() != 4) {
              return failure();
            }
          }
          const auto device_name = OpTrait::IsOpConfCompatible<void>::getDeviceTag(op)
                                       .cast<mlir::StringAttr>()
                                       .getValue()
                                       .str();
          if (device_name == "cpu") { return failure(); }
        }
        llvm::SmallVector<int32_t> perm = getChannelLastTransposePerm();
        llvm::SmallVector<int32_t> result_perm = getChannelFirstTransposePerm();

        NamedAttrList transpose_attributes;
        if (InitTransposeAttributes(op, transpose_attributes, rewriter).succeeded()) {
          transpose_attributes.append(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, perm));
        } else {
          return failure();
        }
        // when op op has no sense of data_format and pre op is transpose, we greedily insert transpose
        // into this op, seeking more opportunities to eliminate transpose pattern.
        const bool greedily_transpose_flag = !op.IsNCHW() && IsInsertTransposeOpBefore(op, rewriter);

        if (op.IsNCHW() || greedily_transpose_flag) {
          // create transpose op for input operand
          SmallVector<Value, 4> tranposed_operands;
          llvm::DenseSet<Value> operand_transpose = op.OperandsToTranspose();
          int num_transposed_operand = 0;
          for (Value operand : op->getOperands()) {
            if (operand_transpose.find(operand) != operand_transpose.end()) {
              SmallVector<Value, 4> input_res = getInputOperandTransposeOp(
                  op, operand, transpose_attributes, num_transposed_operand, rewriter);
              tranposed_operands.push_back(input_res[0]);
              num_transposed_operand += 1;
            }
          }
          // create NHWC op
          SmallVector<Value, 4> created_results = op.NchwToNhwc(tranposed_operands, rewriter);
          // create transpose op for results
          int num_transposed_result = 0;
          transpose_attributes.set(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, result_perm));
          llvm::DenseSet<Value> transpose_result = op.ResultsToTranspose();

          for (Value result : op->getOpResults()) {
            if (transpose_result.find(result) != transpose_result.end()) {
              if (auto result_transpose_op =
                      getResultTransposeOp(op, created_results[num_transposed_result],
                                           transpose_attributes, num_transposed_result, rewriter)) {
                result.replaceAllUsesWith(result_transpose_op);
                num_transposed_result += 1;
              } else {
                return failure();
              }
            }
          }
        }
        return success();
      }
    };


먼저 AutoNhwcPattern 클래스는 OpInterfaceRewritePattern을 상속하며, OpInterfaceRewritePattern은 Operation을 다시 쓰는 데 사용되는 베이스 클래스입니다. AutoNhwcPattern은 NCHWCompatible Interface를 구현한 Operation을 대상으로 다시 쓰기를 수행하여 NCHW에서 NHWC로의 포맷 변환을 구현합니다. 그런 다음 AutoNhwcPattern은 matchAndRewrite 메서드를 오버라이드합니다. 이 메서드는 NCHWCompatible Interface를 가진 Operation을 만났을 때 호출되어 NCHW에서 NHWC로의 변환을 수행합니다. 다음으로 matchAndRewrite 메서드는 먼저 Operation이 변환 조건을 만족하는지, 즉 4차원인지, CPU 디바이스에서 실행되는지 등을 검사합니다. 만족하지 않으면 failure를 반환합니다. 만족하면 matchAndRewrite 메서드는 NCHW에서 NHWC로 그리고 NHWC에서 NCHW로의 변환 순서를 가져오고, Transpose Operation의 속성을 초기화합니다. 그런 다음 현재 Op가 NCHW 포맷이거나 이 Op의 이전 Op가 Transpose Op인 경우, 더 많은 최적화 기회를 얻기 위해 Transpose Op를 삽입하는 작업을 수행합니다.

여기에는 또한 몇 가지 관련 유틸리티 함수가 포함되어 있는데, 이에 대해서도 설명하겠습니다.


    llvm::SmallVector<int32_t> getChannelLastTransposePerm() { return {0, 2, 3, 1}; }

    llvm::SmallVector<int32_t> getChannelFirstTransposePerm() { return {0, 3, 1, 2}; }

    llvm::SmallVector<mlir::Value, 4> getInputOperandTransposeOp(NCHWCompatible op, Value val,
                                                                 NamedAttrList transpose_attributes,
                                                                 int num_transposed_operand,
                                                                 PatternRewriter& rewriter) {
      std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                                   + "_transpose_input_" + std::to_string(num_transposed_operand);
      transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                               rewriter.getStringAttr(transpose_name));
      SmallVector<Value, 4> input_operands;
      input_operands.push_back(val);
      auto res = rewriter
                     .create<oneflow::TransposeOp>(op.getLoc(), getNHWCType(val.getType()),
                                                   input_operands, transpose_attributes)
                     ->getResults();
      return res;
    }

    TransposeOp getResultTransposeOp(NCHWCompatible op, Value val, NamedAttrList transpose_attributes,
                                     int num_transposed_result, PatternRewriter& rewriter) {
      std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                                   + "_transpose_output_" + std::to_string(num_transposed_result);
      transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                               rewriter.getStringAttr(transpose_name));
      SmallVector<Value, 4> operands;
      operands.push_back(val);
      TransposeOp transpose_op = rewriter.create<oneflow::TransposeOp>(
          op.getLoc(), getNCHWType(val.getType()), operands, transpose_attributes);
      return transpose_op;
    }

    bool IsInsertTransposeOpBefore(NCHWCompatible op, PatternRewriter& rewriter) {
      bool insert_transpose_op_flag = false;
      for (mlir::Value operand : op->getOperands()) {
        TransposeOp transposeInputOp = operand.getDefiningOp<TransposeOp>();
        if (!transposeInputOp) continue;
        const auto perm = transposeInputOp.getPermAttr();
        if (perm.size() == 4 && perm[0] == rewriter.getSI32IntegerAttr(0)
            && perm[1] == rewriter.getSI32IntegerAttr(3) && perm[2] == rewriter.getSI32IntegerAttr(1)
            && perm[3] == rewriter.getSI32IntegerAttr(2)) {
          insert_transpose_op_flag = true;
          break;
        }
      }
      return insert_transpose_op_flag;
    }


여기서 getChannelLastTransposePerm과 getChannelFirstTransposePerm 메서드는 각각 NHWC에서 NCHW로 그리고 NCHW에서 NHWC로의 변환 순서를 반환합니다. getInputOperandTransposeOp 메서드는 Operation의 입력에 대한 Transpose Operation을 생성합니다. 입력 값, Transpose 속성, rewriter를 사용하여 TransposeOp를 생성하고 그 결과를 반환합니다. 마찬가지로 getResultTransposeOp 메서드는 Operation의 출력에 대한 Transpose Operation을 생성합니다. 출력 값, Transpose 속성, rewriter를 사용하여 TransposeOp를 생성하고 해당 Operation을 반환합니다. IsInsertTransposeOpBefore 메서드는 Operation의 입력에 이미 Transpose Operation이 있는지 확인합니다. 있다면, 그리고 그 Transpose Operation이 NHWC를 NCHW로 변환하는 것이라면 true를 반환하고, 그렇지 않으면 false를 반환합니다.

## **0x1.3 중복된 Transpose 쌍 제거하기**

다음으로, Transpose op가 삽입된 그래프에서 인접한 모든 Transpose 쌍을 가능한 한 제거해야 합니다. 코드 구현은 다음과 같습니다.


    bool IsRedundantTransposeMatch(ArrayAttr pre, ArrayAttr afe, mlir::PatternRewriter& rewriter) {
      const auto prePerm = pre.getValue().vec();
      const auto afePerm = afe.getValue().vec();
      if (prePerm.size() == 4 && afePerm.size() == 4) {
        // handle nchw->nhwc->nchw: (0, 2, 3, 1) -> (0, 3, 1, 2)
        if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[3] && prePerm[2] == afePerm[1]
            && prePerm[3] == afePerm[2] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
            && prePerm[1] == rewriter.getSI32IntegerAttr(2)
            && prePerm[2] == rewriter.getSI32IntegerAttr(3)
            && prePerm[3] == rewriter.getSI32IntegerAttr(1))
          return true;
        // handle nhwc->nchw->nhwc: (0, 3, 1, 2) -> (0, 2, 3, 1)
        if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[2] && prePerm[2] == afePerm[3]
            && prePerm[3] == afePerm[1] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
            && prePerm[1] == rewriter.getSI32IntegerAttr(3)
            && prePerm[2] == rewriter.getSI32IntegerAttr(1)
            && prePerm[3] == rewriter.getSI32IntegerAttr(2))
          return true;
      }
      return false;
    }

    struct AutoNhwcEliminateRedundantTransposePattern : public mlir::OpRewritePattern<TransposeOp> {
      explicit AutoNhwcEliminateRedundantTransposePattern(mlir::MLIRContext* context)
          : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
      mlir::LogicalResult matchAndRewrite(TransposeOp op,
                                          mlir::PatternRewriter& rewriter) const override {
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        if (!transposeInputOp
            || !IsRedundantTransposeMatch(op.getPermAttr(), transposeInputOp.getPermAttr(), rewriter)) {
          return failure();
        }
        rewriter.replaceOp(op, {transposeInputOp.getOperand()});
        return success();
      }
    };


IsRedundantTransposeMatch 메서드는 두 Transpose Operation의 순서가 중복을 일으키는지 확인합니다. 두 Transpose의 perm 속성을 비교하여 판단합니다. AutoNhwcPattern과 유사하게, AutoNhwcEliminateRedundantTransposePattern 클래스는 OpRewritePattern을 상속합니다. 이 클래스는 TransposeOp에 대해 다시 쓰기를 수행하여 Transpose 제거를 구현합니다. 만약 순서가 NHWC->NCHW->NHWC 또는 NCHW->NHWC->NCHW라면 중복된 Transpose로 판정됩니다. 입력 또한 TransposeOp에서 오고 두 Transpose의 순서가 중복을 일으키는 경우, matchAndRewrite 메서드는 TransposeOp를 그 입력의 TransposeOp로 대체하여 Transpose 제거를 구현합니다. matchAndRewrite 메서드는 먼저 TransposeOp의 입력을 가져오고, 그 입력 또한 TransposeOp에서 오는지 확인합니다. 그렇지 않거나 두 Transpose의 순서가 중복을 일으키지 않는다면 failure를 반환합니다. 마지막으로 success를 반환하여 중복된 Transpose가 성공적으로 제거되었음을 나타냅니다.

최종적으로, 위에서 소개한 두 Pass는 모두 AutoNhwcPass에 캡슐화되어 MLIR의 계산 그래프에 적용되어 전역 최적화를 완료합니다. 아래 코드에서 볼 수 있듯이, 이 최적화는 ONEFLOW_MLIR_PREFER_NHWC [환경 변수](<https://zhida.zhihu.com/search?content_id=228210363&content_type=Article&match_order=1&q=%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiLnjq_looPlj5jph48iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjgyMTAzNjMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.K-TzvG8Uz2TM3Na3hjJntc_Tzibse97ujcE0Wm0899g&zhida_source=entity>)가 켜져 있을 때만 정상적으로 동작합니다.


    void populateAutoNhwcPatterns(::mlir::RewritePatternSet& patterns) {
      bool enable_nhwc = ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PREFER_NHWC", false);
      if (enable_nhwc) {
        patterns.add<AutoNhwcPattern>(patterns.getContext());
        patterns.add<AutoNhwcEliminateRedundantTransposePattern>(patterns.getContext());
      }
    }

    class AutoNhwcPass : public AutoNhwcPassBase<AutoNhwcPass> {
      void runOnOperation() override {
        Operation* op = getOperation();
        RewritePatternSet patterns(op->getContext());
        oneflow::populateAutoNhwcPatterns(patterns);
        (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
      }
    };


## **보충: 0x1.4 weight의 transpose 제거**

여기에서는 weight에 대한 transpose가 어떻게 처리되는지 간략하게 설명할 필요가 있습니다. 0x1.2에서 우리는 weight(constant op)에 대해서도 Transpose Op를 삽입했는데, weight는 상수이므로 weight에 대한 Transpose Op는 [컴파일 타임](<https://zhida.zhihu.com/search?content_id=228210363&content_type=Article&match_order=1&q=%E7%BC%96%E8%AF%91%E6%9C%9F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzgzMTQ3NzMsInEiOiLnvJbor5HmnJ8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMjgyMTAzNjMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.oWAY2hYcs6GkLJWbT4dtH_7rYiGqzlK3pYxa4NmV45c&zhida_source=entity>)에 fold할 수 있습니다. 이 과정은

[](<https://link.zhihu.com/?target=https%3A//github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/oneflow-translate/lib/OneFlow/MLIROneFlowTranslation.cpp%23L808-L811>)

에서 완료되며, Constant Folding의 구현에 대해서는 나중에 별도로 소개하겠습니다.

## **0x2. 정리**

본 글에서는 OneFlow 컴파일러의 Layout Transform에 대해 소개했습니다. 이 기술은 이후 OneFlow 버전의 Stable Diffusion에서도 중요한 역할을 하여 추론 속도를 향상시켰습니다. TVM의 Ansor에도 유사한 최적화가 있는데, 서로 다른 Layout을 Op의 strategy로 설정하여 Op의 schedule에 영향을 주고, 탐색 시 Layout Transform을 고려하여 더 큰 탐색 공간과 더 좋은 결과를 얻습니다. Transpose의 추가 오버헤드를 처리하는 방법은 유일한 것이 아니며, 여기서는 개인적으로 비교적 간단하다고 생각하는 한 가지 방식을 채택했을 뿐입니다. 비슷한 필요성이 있는 독자라면 자유롭게 응용해도 좋습니다.
