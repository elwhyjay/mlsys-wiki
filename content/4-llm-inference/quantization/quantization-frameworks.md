# Quantization Workflows: Framework별 비교

> 이 문서는 [mul.md](../../10-Theoretical-stuff/mul.md) §11–§12의 quantization 이론을 전제로 한다.
> §12.1에서 소개한 두 패러다임(graph-level Q/DQ vs tensor-level subclass)을 각 구현체에서 실제로 어떻게 풀어내는지 정리한 reference document.

---

## 0. 두 축의 재정리

각 framework를 두 axis로 위치시키면 차이가 분명해진다.

**Representation axis**: quantization 의미를 어디에 인코딩하는가
- `graph entity`: graph topology에 Q/DQ node (또는 그에 준하는 op)가 명시적으로 존재
- `type/attribute`: graph는 fp인 채로 두고, tensor type 혹은 attribute에 metadata를 부착

**Fusion axis**: integer kernel을 어떻게 노출시키는가
- `DQ sliding`: DQ를 math-commutative op (Conv/MatMul/Eltwise)로 슬라이딩하여 integer GEMM 노출
- `subclass dispatch`: tensor subclass가 dispatcher hook으로 quantized kernel 직접 호출
- `direct op`: runtime에 Q/DQ가 아예 없고, 전용 integer op가 attribute를 읽음

| Framework | Representation | Fusion |
|---|---|---|
| ONNX Runtime | graph entity | DQ sliding |
| PyTorch PT2E | graph entity (ATen op) | DQ sliding (backend) |
| PyTorch torchao | type (tensor subclass) | subclass dispatch |
| TensorRT (explicit) | graph entity (ONNX QDQ) | DQ sliding |
| TensorRT-LLM | graph entity + format | DQ sliding |
| TFLite / LiteRT | attribute | direct op |
| OpenVINO | graph entity (`FakeQuantize`) | DQ sliding (LPT) |
| ExecuTorch | graph entity (PT2E) | DQ sliding (delegate) |
| vLLM | format (compressed-tensors 등) | subclass dispatch + 전용 kernel |
| MLIR quant dialect | type (`!quant.uniform`) | downstream lowering |

---

## 1. ONNX Runtime (ORT)

**Representation**: `QuantizeLinear` / `DequantizeLinear` graph node — ONNX 표준 QDQ format.

**Lowering**: `GraphTransformer` infrastructure 위의 selector + action 패턴.

```
DQ(A_q) -> Gemm(α, β, bias) -> Q(Y)
DQ(W_q) ----^
```
```
   ↓ (graph rewrite)

QGemm(A_q, s_A, z_A, W_q, s_W, z_W, C, s_Y, z_Y) -> Y_q
```

대표 fused op: `QGemm`, `QLinearMatMul`, `QLinearConv`, `QLinearAdd`.

**특징**:
- ONNX format의 표준 quantization 흐름
- selector가 안전성을 보수적으로 검사 (transpose flag, α/β, dtype, broadcast 등)
- 동일 graph를 CPU / CUDA / TensorRT / DML 등 여러 execution provider가 공유
- 이 selector 단계의 부정확함이 만든 실제 버그가 [qdq-gemm-alpha.md](qdq-gemm-alpha.md)의 사례

**Source**: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

---

## 2. PyTorch PT2E (Export-based Quantization)

**Representation**: pre-autograd ATen IR 위의 graph node.

```
torch.export(model)
  -> prepare_pt2e(model, Quantizer)   # observer 삽입
  -> calibrate
  -> convert_pt2e                      # observer → Q/DQ pair 교체
```

Q/DQ는 `quantized_decomposed` namespace의 ATen op:

```
torch.ops.quantized_decomposed.quantize_per_tensor
torch.ops.quantized_decomposed.dequantize_per_tensor
torch.ops.quantized_decomposed.quantize_per_channel
torch.ops.quantized_decomposed.dequantize_per_channel
```

각 op에 `scale, zero_point, quant_min, quant_max, dtype`이 명시적 인자로 박힌다.

**Quantizer interface**: backend 확장점. `Quantizer` subclass가 `QuantizationSpec` / `QuantizationAnnotation`으로 각 node의 input/output을 어떻게 quantize할지 annotate한다.

**Backend lowering**: `convert_pt2e` 이후 backend가 담당.
- **Inductor** (X86InductorQuantizer): `dq → op → q` 패턴을 `onednn.qconv2d_pointwise` 같은 native int kernel로 fuse
- **ExecuTorch delegate** (XNNPACK 등): 동일 패턴을 delegate-specific int kernel로 lowering

**Source**:
- https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quantizer.html
- `torch/ao/quantization/quantize_pt2e.py`

---

## 3. PyTorch torchao

**Representation**: **tensor subclass**. graph topology에는 Q/DQ가 등장하지 않는다.

```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
quantize_(model, Int4WeightOnlyConfig())
```

내부적으로 `nn.Linear.weight`가 `AffineQuantizedTensor` 같은 subclass instance로 swap된다. graph는 여전히 `F.linear(x, w)` 형태.

**Subclass 종류**:

| Tensor | 용도 | 저장 |
|---|---|---|
| `AffineQuantizedTensor` | int4 / int8 weight | int data + (scale, zero_point, block_size) |
| `Float8Tensor` | FP8 training / inference | FP8 data + per-tensor/row scale |
| `MX*Tensor` | MX format | low-precision data + `float8_e8m0fnu` scale |
| `NVFP4Tensor` | NVFP4 | packed FP4 + two-level scale |

**Lowering**: `__torch_dispatch__` hook이 dispatch 시점에 적절한 kernel 호출.
- tinygemm (bf16 × int4)
- GemLite
- CUTLASS FP8 / NVFP4
- `torch.compile`(Inductor) 결합 시 dequant + matmul fusion

**Available configs**:
- `Int4WeightOnlyConfig` (tinygemm + TensorCoreTiledLayout)
- `Int8WeightOnlyConfig`
- `Int8DynamicActivationInt8WeightConfig`
- `Float8DynamicActivationFloat8WeightConfig`
- `MXFP8` / `MXFP4` / `NVFP4` (Blackwell+)

**최근 변화 (2025)**:
- Blackwell MXFP8 / NVFP4 inference (diffusers에서 MXFP8 1.26x, NVFP4 1.68x)
- GemLite + SGLang 통합
- TorchAO paper at CodeML @ ICML 2025

**Source**:
- https://github.com/pytorch/ao
- https://arxiv.org/pdf/2507.16099
- https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/

---

## 4. TensorRT (Explicit Quantization)

**Representation**: ONNX의 `QuantizeLinear` / `DequantizeLinear` pair를 직접 소비.

**Explicit vs Implicit**:
- **Explicit** (현재 권장): Q/DQ가 graph에 명시되어 있고, builder는 이를 honor해야 함. 어디서 정확히 precision transition이 일어나는지가 모델에 박혀 있음
- **Implicit** (deprecated in 10.x): builder가 INT8 calibration을 자체 수행해서 어느 op를 양자화할지 결정

**Lowering**: DQ op를 math-commutative한 op로 슬라이딩하여 fused int kernel을 노출. ORT, OpenVINO LPT와 동일한 알고리즘적 아이디어.

**Source**: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html

---

## 5. TensorRT-LLM

LLM-specific stack. **TensorRT 위에서** 동작하나 quantization 진입점이 다르다.

**Quantization 경로**:
- 대부분의 PTQ는 **NVIDIA ModelOpt**가 담당 — FP8 (default), W4A8_AWQ, NVFP4, FP8 per-channel/per-token
- TRT-LLM 자체 구현: **SmoothQuant INT8**, INT8/FP8 KV cache, INT4/INT8 weight-only

**산출물**: TRT-LLM checkpoint → engine build.

**제약**: `int8_sq`와 `fp8`은 동시 사용 불가.

**Source**:
- https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md

---

## 6. TFLite / LiteRT

**가장 극단적인 "type/attribute" 진영.** runtime graph에 Q/DQ node가 **존재하지 않는다.**

**Representation**: tensor attribute.

```
tensor {
  scale: 0.0234375
  zero_point: -17
  quantization_dimension: 0   # per-axis인 경우
}
```

op는 dedicated int kernel을 가지며, attribute를 직접 read:

$$
\text{real} = (q - \text{zero\_point}) \cdot \text{scale}
$$

**Weight**: per-axis symmetric int8, `[-127, 127]`, `zero_point = 0` along output channel dim.
**Activation**: per-tensor asymmetric int8.

**변환 과정**: TF → TFLite converter가 변환 중 transient하게 Q/DQ node를 삽입하지만, 최종 flatbuffer에는 직접 quantized op만 남는다.

**Source**:
- https://www.tensorflow.org/lite/performance/quantization_spec
- https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization

---

## 7. OpenVINO (FakeQuantize)

**Representation**: 단일 `FakeQuantize` graph op. Q+DQ를 하나로 압축한 변형.

Parameters: `input_low, input_high, output_low, output_high, levels`.

$$
q = \text{round}\!\left(\frac{\text{clamp}(x, \text{in\_low}, \text{in\_high}) - \text{in\_low}}{\text{in\_high} - \text{in\_low}} \cdot (\text{levels} - 1)\right)
$$

$$
y = \frac{q}{\text{levels} - 1} \cdot (\text{out\_high} - \text{out\_low}) + \text{out\_low}
$$

출력 dtype은 fp — "fake"라는 이름은 값은 양자화되지만 저장은 fp로 한다는 의미.

**Lowering — LPT (Low Precision Transformations) pass**:
1. 각 `FakeQuantize`를 (Quantize + Dequantize)로 분해
2. Dequantize를 math-commutative op로 propagate
3. 인접 Conv / FullyConnected / Eltwise가 integer input을 받고 dequant scale과 fuse

**ONNX QDQ import 시**: `QuantizeLinear + DequantizeLinear` pair를 **다시 `FakeQuantize`로 fold**해서 LPT pipeline에 태운다. 다른 framework에서 변환된 모델도 OpenVINO 내부 표준 표현으로 통일.

**Source**:
- https://docs.openvino.ai/2023.3/openvino_docs_ops_quantization_FakeQuantize_1.html
- https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/advanced-guides/low-precision-transformations.html

---

## 8. ExecuTorch

**Representation**: PT2E의 `dq → op → q` reference representation을 그대로 소비.

**Lowering**: backend별 Quantizer + Partitioner.

| Backend | Quantizer | 지원 |
|---|---|---|
| XNNPACK | `XNNPACKQuantizer` | 8-bit symmetric weight, 8-bit asymmetric activation, static/dynamic, per-tensor/per-channel |
| CoreML | `CoreMLQuantizer` | iOS 17+ |
| QNN (Qualcomm) | QNN-specific | mobile NPU |
| MediaTek | vendor-specific | mobile NPU |
| Vulkan | vendor-specific | GPU on mobile |

각 backend의 Quantizer가 **자신이 lowering 가능한 패턴만 annotate**한다. `prepare_pt2e`가 이 annotation을 따라 observer를 삽입.

**최근 변화 (2025)**: RFC #13732 — multi-backend recipe. 하나의 export로 CoreML primary + XNNPACK CPU fallback 같은 multi-target 지원. **OPEN 상태 (2026-05 기준)**, 2025-08-29까지 활발한 토론 후 정체. `kimishpatel`은 별도 추상화 대신 기존 `Composable` quantizer 패턴을 확장하는 방안을 제안. `metascroy`는 CoreML이 보통 graph 전체를 소비하므로 XNNPACK fallback이 실효성이 적다고 지적. 결론 미합의.

**Source**:
- https://docs.pytorch.org/executorch/0.7/quantization-overview.html
- https://docs.pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html

---

## 9. vLLM

LLM serving에 특화. **on-disk weight format**을 인식해 적절한 kernel로 dispatch하는 plugin 구조.

**주요 quantization 경로**:

| 경로 | 내용 |
|---|---|
| **compressed-tensors** | vllm-project의 safetensors 확장. **LLM Compressor**가 산출. W4A16, W8A16, W8A8 (int8/fp8), W4A8, MXFP8/MXFP4/NVFP4 (A16 또는 activation 양자화). canonical format으로 자리잡는 중 |
| **GPTQ / GPTQModel** | group-wise 4-bit weight. ExLlamaV2 kernel default |
| **AWQ** | activation-aware weight quant. 공식 AWQ kernel |
| **FP8** | per-tensor 또는 per-channel/per-token. compressed-tensors 또는 torchao 경유 |
| **torchao** | `quantize_()` config를 직접 load (int4, int8, fp8, MX, NVFP4) |

**Kernel** (vLLM main, 2026-05 기준 source 확인):

| Kernel | Min compute capability | 비고 |
|---|---|---|
| `MarlinLinearKernel` | **75** (Turing+) | inline PTX, CUDA 전용 |
| `MacheteLinearKernel` | **90** (Hopper+) | CUTLASS 기반, shape 제약: `in % 64 == 0`, `out % 128 == 0` |
| `CutlassW4A8LinearKernel` | 90 (Hopper+) | priority 최상위 |
| `ExllamaLinearKernel`, `ConchLinearKernel`, `AllSparkLinearKernel` | varies | fallback |

**선택 메커니즘**: `vllm/model_executor/kernels/linear/__init__.py`의 `choose_mp_linear_kernel`.

```python
_POSSIBLE_KERNELS[CUDA] = [
    CutlassW4A8LinearKernel,    # priority 1
    MacheteLinearKernel,        # priority 2
    AllSparkLinearKernel,
    MarlinLinearKernel,
    ConchLinearKernel,
    ExllamaLinearKernel,
]
```

알고리즘:
1. priority list를 순회
2. 각 kernel에 대해 (a) `VLLM_DISABLED_KERNELS` env에 없는가, (b) `kernel.get_min_capability() <= compute_capability`, (c) `kernel.can_implement(config)` 통과를 검사
3. 처음으로 모두 통과한 kernel을 선택

즉 **batch size 기반 동적 선택이 아니다.** 정적 priority + compute capability + shape/quant compatibility 기반의 layer 단위 선택. 사용자는 `--linear-backend` 플래그나 `VLLM_DISABLED_KERNELS` 환경변수로 override 가능.

INT8 / FP8 / MXFP8 / MXFP4 / NVFP4 precision class마다 별도 priority list (`_POSSIBLE_*_KERNELS`)가 있고 동일한 selection 알고리즘이 적용된다.

**개발 주체**: Neural Magic / Red Hat이 vLLM project와 공동 개발.

**Source**:
- https://docs.vllm.ai/en/latest/features/quantization/
- https://github.com/vllm-project/compressed-tensors
- https://github.com/vllm-project/llm-compressor

---

## 10. MLIR `quant` Dialect

**Representation**: **quantized type 자체.** op가 아니다.

```
!quant.uniform<i8:f32, 0.02:5>
!quant.uniform<i8<-127:127>:f32:0, {0.01, 0.02, ...}>   # per-axis
```

Storage type + expressed type + scale + zero_point가 type signature에 인코딩된다. per-axis form은 quantized dimension까지 명시.

**Op**: `quant.qcast` / `quant.dcast` — expressed(fp)와 quantized 표현 사이 변환.

**Integer math는 dialect가 owner하지 않는다.** downstream project가 type conversion으로 lowering 책임을 짐.

**사용 프로젝트**:
- **TFLite / TOSA**: TF → TFLite 변환 path에서 quant dialect 사용
- **IREE**: cross-platform compiler
- **ONNX-MLIR**
- 다양한 hardware compiler

**2025 RFC**: `UniformQuantizedType`을 interface-based로 확장하여 FP8 (E5M2 / E4M3FN) 같은 비-integer storage 지원. float type이 `QuantizationInterface`를 구현하는 방식. **discourse.llvm.org에서 진행 중 (2026-05 기준 OPEN)**, 마지막 활동 2025-12-15 (Roman-Pevnyi의 gentle ping). PR #152966에 Float8E5M2 / Float8E4M3FN 구현체가 있으나 미머지. MX type 표현 방식, sub-byte storage container 분리 여부 등 architectural 질문이 미해결.

**Source**:
- https://mlir.llvm.org/docs/Dialects/QuantDialect/
- https://mlir.llvm.org/docs/Quantization/
- https://discourse.llvm.org/t/rfc-extending-uniformquantizedtype-with-interface-based-support-for-new-storage-types-in-quant-dialect/87803

---

## 11. Cross-cutting Observations

### 11.1 공통된 lowering trick: DQ Sliding

ORT, TensorRT, OpenVINO LPT, PyTorch Inductor 모두 핵심 알고리즘은 동일하다.

> Dequantize op를 math-commutative op (Conv / MatMul / Eltwise / Transpose / Reshape)로 슬라이딩 시켜, integer 도메인에서 GEMM이 가능하도록 노출한다.

표현이 graph node든, FakeQuantize든, ATen op든 — optimizer가 하는 일은 거의 같다. 차이는 IR이지 알고리즘이 아니다.

### 11.2 두 분파의 의의

- **Graph entity 진영** (ONNX, PT2E, TRT, OpenVINO, ExecuTorch): static graph backend, AOT compiler, interchange format에 유리
- **Type/attribute 진영** (torchao, TFLite, MLIR quant, vLLM compressed-tensors): LLM weight-only, FP8 training, runtime-specialized kernel에 유리

흥미로운 점은 **두 진영이 수렴하지 않는다**는 것. 각자의 사용 시나리오가 다르고, 같은 모델을 두 표현 사이에서 변환하는 과정에서 export/import 비용이 발생한다 (torchao → ONNX QDQ → TensorRT 같은 흐름).

### 11.3 표현이 만들어내는 버그

graph entity 진영은 **graph rewriter의 selector 정확성**이 정확성의 ground truth가 된다 — [qdq-gemm-alpha.md](qdq-gemm-alpha.md)의 α × bias 사례.

type/attribute 진영은 **dispatcher와 kernel의 일치성**이 ground truth가 된다 — tensor subclass가 어떤 kernel을 호출하는지가 type 자체에서 도출되므로, dispatch logic의 부정확함이 silent wrong-output을 만든다.

각 진영의 failure mode가 다르다.

---

## 12. 검증 기록

1차 소스를 확인하여 보강한 사항 (2026-05 기준):

- **vLLM kernel selection**: 정적 priority list + `min_capability` + `can_implement` 검사 방식임을 source 코드 (`vllm/model_executor/kernels/linear/__init__.py:choose_mp_linear_kernel`)로 확인. batch-size 기반 동적 선택은 사실이 아님 — 이전 secondary source의 부정확한 기술
- **ExecuTorch RFC #13732**: GitHub issue로 OPEN 상태 확인. 2025-08-29 이후 정체. 별도 추상화 vs `Composable` 패턴 확장 사이에서 합의 미도달
- **MLIR FP8 quant RFC**: discourse.llvm.org에서 OPEN 상태 확인. 2025-12-15 마지막 활동. PR #152966 미머지

§9, §8, §10에 결과를 반영했다. 본 문서의 다른 부분은 secondary source에 의존하므로, 사용 시점에 1차 소스로 재검증 권장.

---

## 참고

- 양자화 일반 이론: [mul.md](mul.md) §11–§13
- α × bias case study (ORT 사례): [qdq-gemm-alpha.md](qdq-gemm-alpha.md)
