# SGLang용 Profile Analysis SKILL - 세 표로 Kernel Fuse와 Overlap 기회 찾기

## 0x0. 서문

최근 이 skills를 다시 조금 업그레이드했으니 따로 공유합니다. 아래 링크로 이 SKILLS를 설치하면 바로 사용할 수 있습니다.

https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-torch-profiler-analysis

또는 여기에서도 가능합니다: https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-torch-profiler-analysis

주요 목적: Claude Code나 Codex로 SGLang model, 특히 새로운 model을 최적화하고 싶다면, 이 SKILL은 cold start profile token을 대량으로 절약해 주고, cold start analysis를 더 포괄적이고 깊게 만들어 줍니다.

## 0x1. feature

항목별로 나열하면 다음과 같습니다.

- workflow를 triage-only로 통일하고, 외부에는 entry script `analyze_sglang_torch_profile.py` 하나만 노출합니다.
- output은 `kernel table`, `overlap-opportunity table`, `fuse-pattern table` 세 table로 고정됩니다.
- 기본적으로 `>=1%` GPU time share 행만 render합니다.
- 기존 `trace.json(.gz)` 또는 profile directory 분석을 지원합니다.
- 실행 중인 SGLang server에 직접 profiling을 trigger하고 analysis result를 생성하는 것을 지원합니다.
- single-trace triage를 지원합니다.
- mapping+formal dual trace triage를 지원합니다.
- `profile_by_stage`를 지원해 prefill / decode 같은 stage별로 나누어 분석할 수 있습니다.
- diffusion model을 지원합니다.
- output에는 kernel에서 Python source path와 CPU op로 mapping한 result가 포함됩니다.
- `overlap-opportunity table`은 exclusive/hidden time ratio, dependency risk, kernel category 같은 rule로 candidate를 생성합니다.
- `fuse-pattern table`은 pattern registry와 catalog 기반 source-backed scan을 수행합니다.
- pattern scan scope는 SGLang, vLLM, FlashInfer, TensorRT-LLM에 이미 존재하는 fuse 또는 overlap 가능한 pattern을 포함합니다.
- pattern scan은 PR에 이미 등장했지만 아직 merge되지 않은 fuse 또는 overlap 가능한 pattern도 포함합니다.
- fuse pattern script matching logic은 deterministic하고 source-backed하게 유지하며, script layer에는 fuzzy string matching을 도입하지 않습니다.
- exact match가 충분하지 않을 때 table 뒤에 `AI similarity judgment`를 추가하고 `high / medium / low` 세 level로 표시할 수 있습니다.
- diffusion profile에 backend gate를 추가했습니다. 실행이 이미 diffusers backend로 fallback되었다면 analysis를 중단하고 backend selection problem으로 직접 표시합니다.
- reference directory에는 `source-map`, `heuristics`, `fuse-overlap-catalog`, `overlap-catalog`가 포함됩니다.
- output structure는 run 간 비교에 바로 사용할 수 있고, issue, PR, review comment에 바로 붙여 넣을 수도 있습니다.
- dominant kernel family를 찾고, fused path가 유효한지 확인하고, overlap headroom을 판단하며, profile evidence를 정리하는 데 사용할 수 있습니다.

## 0x2. Demo

Codex+GPT5.4

### 0x2.1 prompt

Based on the current skill, analyze `Qwen/Qwen3.5-4B` and `openai/gpt-oss-20b` on H100. Use the service launch commands from `sgl-cookbook`, and then give me the three result tables.

### 0x2.2 Qwen/Qwen3.5-4B result

#### Kernel Table
| Stage | Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| decode | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | gemm | 9.23 ms | 24.0% | 20 | python/sglang/srt/layers/logits_processor.py:878 _compute_lm_head | aten::mm |
| decode | nvjet_tst_256x8_64x6_4x1_v_bz_TNT | gemm | 6.32 ms | 16.4% | 175 | python/sglang/srt/layers/quantization/unquant.py:137 apply | aten::mm |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 5.83 ms | 15.1% | 365 | python/sglang/srt/layers/quantization/unquant.py:137 apply (site share 96%)<br>python/sglang/srt/models/qwen3_5_mtp.py:112 forward (site share 4%) | aten::mm<br>aten::mm |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 2.84 ms | 7.4% | 120 | python/sglang/srt/layers/quantization/unquant.py:137 apply | aten::mm |
| decode | nvjet_tst_64x8_64x16_1x1_h_bz_TNT | gemm | 2.17 ms | 5.6% | 120 | python/sglang/srt/layers/quantization/unquant.py:137 apply | aten::mm |
| decode | fused_sigmoid_gating_delta_rule_update_kernel | activation | 1.72 ms | 4.5% | 120 | python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py:243 fused_sigmoid_gating_delta_rule_update | cuLaunchKernelEx |
| decode | _fwd_kernel | other | 1.57 ms | 4.1% | 45 | python/sglang/srt/layers/attention/triton_ops/extend_attention.py:552 extend_attention_fwd | cuLaunchKernelEx |
| decode | void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue> | softmax | 1.24 ms | 3.2% | 20 | python/sglang/srt/speculative/eagle_worker_v2.py:399 draft_forward (site share 67%)<br>python/sglang/srt/speculative/eagle_worker_v2.py:548 _draft_extend_for_decode (site share 33%) | aten::_softmax<br>aten::_softmax |
| decode | nvjet_tst_128x8_64x12_4x1_v_bz_TNT | gemm | 1.14 ms | 3.0% | 55 | python/sglang/srt/layers/quantization/unquant.py:137 apply | aten::mm |
| decode | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.86 ms | 2.2% | 350 | python/sglang/srt/layers/layernorm.py:462 _forward_impl | cudaLaunchKernelExC |
| decode | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> | gemm | 0.67 ms | 1.7% | 365 | python/sglang/srt/layers/quantization/unquant.py:137 apply (site share 96%)<br>python/sglang/srt/models/qwen3_5_mtp.py:112 forward (site share 4%) | aten::mm<br>aten::mm |
| decode | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.48 ms | 1.2% | 200 | python/sglang/srt/models/qwen3_5.py:790 self_attention (site share 61%)<br>python/sglang/srt/layers/attention/triton_backend.py:849 forward_extend (site share 19%) | aten::copy_<br>aten::copy_ |
| decode | void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> | activation | 0.43 ms | 1.1% | 175 | python/sglang/srt/layers/activation.py:73 forward_cuda | sgl_kernel::silu_and_mul |

#### Overlap Opportunity Table
| Stage | Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| decode | P2 | headroom | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | python/sglang/srt/layers/logits_processor.py(878): _compute_lm_head | 9231.7 us, share 25.7%, excl 99.3% / hid 0.7% | high | check deps |
| decode | P1 | headroom | nvjet_tst_256x8_64x6_4x1_v_bz_TNT | python/sglang/srt/layers/quantization/unquant.py(137): apply | 6324.4 us, share 17.6%, excl 99.8% / hid 0.2% | low | try fusion |
| decode | P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | python/sglang/srt/layers/quantization/unquant.py(137): apply | 5833.2 us, share 16.2%, excl 99.7% / hid 0.3% | high | check deps |
| decode | P2 | headroom | fused_sigmoid_gating_delta_rule_update_kernel | python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py(243): fused_sigmoid_gating_delta_rule_update | 1720.4 us, share 4.8%, excl 99.9% / hid 0.1% | high | check deps |
| decode | P1 | headroom | _fwd_kernel | python/sglang/srt/layers/attention/triton_ops/extend_attention.py(552): extend_attention_fwd | 1566.6 us, share 4.4%, excl 100.0% / hid 0.0% | low | try fusion |

#### Fuse Opportunity Table
| Stage | Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| decode | PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 27.61 ms | 71.7% | nvjet_tst_512x8_64x3_2x1_v_bz_TNT (24.0%)<br>nvjet_tst_256x8_64x6_4x1_v_bz_TNT (16.4%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT (15.1%) | _compute_lm_head @ python/sglang/srt/layers/logits_processor.py:878<br>apply @ python/sglang/srt/layers/quantization/unquant.py:137<br>forward @ python/sglang/srt/models/qwen3_5_mtp.py:112<br>fast_topk @ python/sglang/srt/utils/common.py:2650 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 71.7% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| decode | Fused QK RoPE reshape + KV cache write | Likely | 0.96 ms | 2.5% | - | apply_rotary_emb @ python/sglang/srt/layers/rotary_embedding/utils.py:36<br>triton_mrope_fused @ python/sglang/srt/layers/rotary_embedding/triton_kernels.py:111<br>forward_native @ python/sglang/srt/layers/rotary_embedding/mrope.py:155<br>_draft_extend_for_decode @ python/sglang/srt/speculative/eagle_worker_v2.py:548 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 2.5% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| decode | vLLM-origin DSV3 router GEMM | Likely | 0.93 ms | 2.4% | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false> (1.7%) | layernorm_fn @ python/sglang/srt/layers/attention/fla/layernorm_gated.py:345<br>embedding @ python/sglang/srt/layers/quantization/unquant.py:104<br>apply @ python/sglang/srt/layers/quantization/unquant.py:137<br>forward @ python/sglang/srt/models/qwen3_5_mtp.py:112 | vllm/model_executor/layers/fused_moe/router/gate_linear.py<br>vllm/csrc/moe/dsv3_router_gemm_entry.cu | This trace matches a reusable upstream vLLM precedent at 2.4% related GPU time. vLLM already has a specialized DeepSeek router GEMM family for small decode batches. |
| decode | Fused residual add + RMSNorm | Likely | 0.86 ms | 2.2% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.2%) | _forward_impl @ python/sglang/srt/layers/layernorm.py:462 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 2.2% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |
| decode | Fused activation-and-mul (SwiGLU / GeGLU) | Likely | 0.43 ms | 1.1% | void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &(float silu<float>(float const&))> (1.1%) | forward_cuda @ python/sglang/srt/layers/activation.py:73 | python/sglang/srt/layers/activation.py | This trace already hits the `Fused activation-and-mul (SwiGLU / GeGLU)` family directly at 1.1% related GPU time. Packed MLP activation and multiply already has dedicated fused ops. |

### 0x2.3 openai/gpt-oss-20b result

#### Kernel Table
| Stage | Kernel | Category | GPU time | Share | Launches | Python location (site share) | CPU op |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| extend/prefill | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu | gemm | 2.03 ms | 35.6% | 24 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| extend/prefill | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1 | gemm | 1.31 ms | 23.0% | 24 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| extend/prefill | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | gemm | 0.39 ms | 6.8% | 1 | python/sglang/srt/layers/logits_processor.py:866 _compute_lm_head | aten::mm |
| extend/prefill | nvjet_tst_64x24_64x16_4x1_v_bz_bias_TNT | gemm | 0.31 ms | 5.5% | 24 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::addmm |
| extend/prefill | nvjet_tst_64x24_64x16_4x1_v_bz_splitK_bias_TNT | gemm | 0.29 ms | 5.0% | 24 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::addmm |
| extend/prefill | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<192>, cute::C<128>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<192>, cute::C<64>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 384, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<192, 128, 384, 128, false, true, true, true, true, true> > > > | gemm | 0.21 ms | 3.7% | 24 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | sgl_kernel::fwd |
| extend/prefill | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.12 ms | 2.2% | 50 | python/sglang/srt/layers/layernorm.py:177 forward_cuda | cudaLaunchKernelExC |
| extend/prefill | _reduce_grouped | reduce_topk | 0.09 ms | 1.6% | 24 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| extend/prefill | void tinygemm_kernel<16, 16, 8, 64, 16, 4, false> | gemm | 0.08 ms | 1.4% | 24 | python/sglang/srt/models/gpt_oss.py:148 forward | cudaLaunchKernel |
| extend/prefill | void flash::prepare_varlen_num_blocks_kernel<1, true> | gemm | 0.08 ms | 1.3% | 25 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | sgl_kernel::fwd |
| extend/prefill | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.07 ms | 1.3% | 25 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | aten::copy_ |
| extend/prefill | nvjet_tst_64x24_64x16_4x1_v_bz_splitK_TNT | gemm | 0.07 ms | 1.3% | 3 | python/sglang/srt/layers/quantization/unquant.py:134 apply (site share 71%)<br>python/sglang/srt/models/llama_eagle3.py:151 forward (site share 29%) | aten::mm<br>aten::mm |
| extend/prefill | nvjet_tst_256x24_64x6_2x1_v_bz_TNT | gemm | 0.07 ms | 1.2% | 1 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::mm |
| extend/prefill | nvjet_tst_256x8_64x6_2x1_v_bz_TNT | gemm | 0.07 ms | 1.2% | 1 | python/sglang/srt/layers/logits_processor.py:866 _compute_lm_head | aten::mm |
| extend/prefill | _combined_routing_compute | communication | 0.06 ms | 1.1% | 24 | python/sglang/srt/layers/moe/topk.py:304 forward_cuda | SortTokens |
| decode | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu | gemm | 4.84 ms | 21.3% | 120 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| decode | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1 | gemm | 2.98 ms | 13.1% | 120 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| decode | nvjet_tst_256x8_64x6_2x1_v_bz_TNT | gemm | 1.95 ms | 8.6% | 30 | python/sglang/srt/layers/logits_processor.py:866 _compute_lm_head (site share 50%)<br>python/sglang/srt/layers/quantization/unquant.py:134 apply (site share 50%) | aten::mm<br>aten::mm |
| decode | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | gemm | 1.93 ms | 8.5% | 5 | python/sglang/srt/layers/logits_processor.py:866 _compute_lm_head | aten::mm |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_bias_TNT | gemm | 1.55 ms | 6.8% | 120 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::addmm |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_bias_TNT | gemm | 1.34 ms | 5.9% | 120 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::addmm |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | gemm | 0.83 ms | 3.6% | 35 | python/sglang/srt/layers/quantization/unquant.py:134 apply (site share 87%)<br>python/sglang/srt/models/llama_eagle3.py:151 forward (site share 13%) | aten::mm<br>aten::mm |
| decode | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> | norm | 0.73 ms | 3.2% | 270 | python/sglang/srt/layers/layernorm.py:177 forward_cuda | cudaLaunchKernelExC |
| decode | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<192>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, true, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<192> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<64, 192, 128, 128, false, true, true, true, false, true> > > > | gemm | 0.68 ms | 3.0% | 60 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | sgl_kernel::fwd |
| decode | void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<192>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, true, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<192> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, true, false>, flash::VarlenDynamicPersistentTileScheduler<64, 192, 128, 128, true, true, true, true, true, true> > > > | gemm | 0.49 ms | 2.1% | 60 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | sgl_kernel::fwd |
| decode | nvjet_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 0.41 ms | 1.8% | 15 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::mm |
| decode | _reduce_grouped | reduce_topk | 0.40 ms | 1.8% | 120 | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220 triton_kernel_fused_experts_with_bias | cuLaunchKernelEx |
| decode | void tinygemm_kernel<16, 16, 8, 64, 16, 4, false> | gemm | 0.39 ms | 1.7% | 120 | python/sglang/srt/models/gpt_oss.py:148 forward | cudaLaunchKernel |
| decode | void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int)#1}> | memory | 0.29 ms | 1.3% | 125 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend | aten::copy_ |
| decode | _combined_routing_compute | communication | 0.28 ms | 1.2% | 120 | python/sglang/srt/layers/moe/topk.py:304 forward_cuda | SortTokens |
| decode | void flash::prepare_varlen_num_blocks_kernel<1, true> | gemm | 0.25 ms | 1.1% | 75 | python/sglang/srt/layers/attention/flashattention_backend.py:740 forward_extend (site share 92%)<br>python/sglang/srt/layers/attention/flashattention_backend.py:1128 forward_decode (site share 8%) | sgl_kernel::fwd<br>sgl_kernel::fwd |
| decode | void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, true, false, false> | gemm | 0.23 ms | 1.0% | 120 | python/sglang/srt/layers/quantization/unquant.py:134 apply | aten::addmm |

#### Overlap Opportunity Table
| Stage | Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extend/prefill | P1 | headroom | nvjet_tst_64x24_64x16_4x1_v_bz_bias_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 314.4 us, share 5.5%, excl 100.0% / hid 0.0% | low | try fusion |
| extend/prefill | P2 | headroom | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | python/sglang/srt/layers/logits_processor.py(866): _compute_lm_head | 387.5 us, share 6.8%, excl 100.0% / hid 0.0% | high | check deps |
| extend/prefill | P1 | headroom | nvjet_tst_64x24_64x16_4x1_v_bz_splitK_bias_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 286.0 us, share 5.0%, excl 100.0% / hid 0.0% | low | try fusion |
| extend/prefill | P1 | headroom | _combined_routing_compute | python/sglang/srt/layers/moe/topk.py(304): forward_cuda | 62.8 us, share 1.1%, excl 100.0% / hid 0.0% | low | try overlap |
| extend/prefill | P1 | headroom | _reduce_grouped | python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py(220): triton_kernel_fused_experts_with_bias | 90.8 us, share 1.6%, excl 100.0% / hid 0.0% | low | try fusion |
| decode | P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_bias_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 1548.8 us, share 6.9%, excl 100.0% / hid 0.0% | high | check deps |
| decode | P2 | headroom | nvjet_tst_512x8_64x3_2x1_v_bz_TNT | python/sglang/srt/layers/logits_processor.py(866): _compute_lm_head | 1932.8 us, share 8.6%, excl 99.0% / hid 1.0% | high | check deps |
| decode | P1 | headroom | nvjet_tst_256x8_64x6_2x1_v_bz_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 1948.8 us, share 8.7%, excl 94.0% / hid 6.0% | low | try fusion |
| decode | P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_bias_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 1343.8 us, share 6.0%, excl 99.9% / hid 0.1% | high | check deps |
| decode | P2 | headroom | nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT | python/sglang/srt/layers/quantization/unquant.py(134): apply | 827.6 us, share 3.7%, excl 99.0% / hid 1.0% | high | check deps |

#### Fuse Opportunity Table
| Stage | Pattern | Confidence | Related GPU time | Share | Evidence kernels | Current kernel Python location | Candidate fused Python path | Rationale |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| extend/prefill | PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 1.26 ms | 22.1% | nvjet_tst_512x8_64x3_2x1_v_bz_TNT (6.8%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_bias_TNT (5.5%)<br>nvjet_tst_64x24_64x16_4x1_v_bz_splitK_bias_TNT (5.0%) | _compute_lm_head @ python/sglang/srt/layers/logits_processor.py:866<br>apply @ python/sglang/srt/layers/quantization/unquant.py:134<br>forward @ python/sglang/srt/models/llama_eagle3.py:151<br>forward_cuda @ python/sglang/srt/layers/moe/topk.py:304 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 22.1% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| extend/prefill | Fused QK RoPE reshape + KV cache write | Conditional | 0.08 ms | 1.4% | - | apply_rope_with_cos_sin_cache_inplace @ python/sglang/jit_kernel/rope.py:179<br>_set_kv_buffer_impl @ python/sglang/srt/mem_cache/memory_pool.py:86 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 1.4% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| extend/prefill | vLLM-origin DSV3 router GEMM | Likely | 3.70 ms | 64.9% | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu (35.6%)<br>_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1 (23.0%)<br>void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<192>, cute::C<128>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, true, false, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<192>, cute::C<64>, cute::C<128> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 384, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<192, 128, 384, 128, false, true, true, true, true, true> > > > (3.7%) | embedding @ python/sglang/srt/layers/quantization/unquant.py:101<br>triton_kernel_fused_experts_with_bias @ python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220<br>forward_extend @ python/sglang/srt/layers/attention/flashattention_backend.py:740<br>forward @ python/sglang/srt/models/gpt_oss.py:148 | vllm/model_executor/layers/fused_moe/router/gate_linear.py<br>vllm/csrc/moe/dsv3_router_gemm_entry.cu | This trace matches a reusable upstream vLLM precedent at 64.9% related GPU time. vLLM already has a specialized DeepSeek router GEMM family for small decode batches. |
| extend/prefill | Fused residual add + RMSNorm | Likely | 0.12 ms | 2.2% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (2.2%) | forward_cuda @ python/sglang/srt/layers/layernorm.py:177 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 2.2% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |
| decode | PR #22392 CUTLASS FP8 scaled MM replacing nvjet | Likely | 8.20 ms | 36.1% | nvjet_tst_256x8_64x6_2x1_v_bz_TNT (8.6%)<br>nvjet_tst_512x8_64x3_2x1_v_bz_TNT (8.5%)<br>nvjet_tst_64x8_64x16_4x1_v_bz_bias_TNT (6.8%) | _compute_lm_head @ python/sglang/srt/layers/logits_processor.py:866<br>apply @ python/sglang/srt/layers/quantization/unquant.py:134<br>forward @ python/sglang/srt/models/llama_eagle3.py:151<br>forward_cuda @ python/sglang/srt/layers/moe/topk.py:304 | PR #22392<br>sgl-kernel/python/sgl_kernel/gemm.py<br>python/sglang/srt/layers/quantization/fp8_utils.py | This trace matches a PR-backed / in-flight pattern at 36.1% related GPU time. An open SGLang PR already replaces nvjet FP8 GEMM with CUTLASS to remove memset bubbles and extra copies. |
| decode | Fused QK RoPE reshape + KV cache write | Conditional | 0.40 ms | 1.7% | - | apply_rope_with_cos_sin_cache_inplace @ python/sglang/jit_kernel/rope.py:179<br>_set_kv_buffer_impl @ python/sglang/srt/mem_cache/memory_pool.py:86 | python/sglang/srt/layers/attention/utils.py | Related split kernels occupy 1.7% of cumulative GPU time, and the checked-out SGLang tree already exposes this fusion family. Attention prep already has a fused RoPE plus reshape plus cache write path. |
| decode | vLLM-origin DSV3 router GEMM | Likely | 9.92 ms | 43.6% | _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu (21.3%)<br>_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1 (13.1%)<br>void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cute::tuple<cute::C<64>, cute::C<192>, cute::C<64> >, 64, cutlass::bfloat16_t, float, cutlass::arch::Sm90, false, true, false, true, true, false, false, true, true, true, false, false, cutlass::bfloat16_t, 1>, flash::CollectiveEpilogueFwd<cute::tuple<cute::C<64>, cute::C<64>, cute::C<192> >, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::bfloat16_t, cutlass::arch::Sm90, 128, true, true, false, false>, flash::VarlenDynamicPersistentTileScheduler<64, 192, 128, 128, false, true, true, true, false, true> > > > (3.0%) | embedding @ python/sglang/srt/layers/quantization/unquant.py:101<br>triton_kernel_fused_experts_with_bias @ python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:220<br>forward @ python/sglang/srt/models/gpt_oss.py:148<br>apply @ python/sglang/srt/layers/quantization/unquant.py:134 | vllm/model_executor/layers/fused_moe/router/gate_linear.py<br>vllm/csrc/moe/dsv3_router_gemm_entry.cu | This trace matches a reusable upstream vLLM precedent at 43.6% related GPU time. vLLM already has a specialized DeepSeek router GEMM family for small decode batches. |
| decode | Fused residual add + RMSNorm | Likely | 0.73 ms | 3.2% | void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16> (3.2%) | forward_cuda @ python/sglang/srt/layers/layernorm.py:177 | python/sglang/srt/layers/layernorm.py<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py | This trace already hits the `Fused residual add + RMSNorm` family directly at 3.2% related GPU time. Residual add plus RMSNorm already has fused implementations across several backends. |

## 0x3. Why

위의 세 table을 보면, 이 SKILL은 우리가 새 model을 직접 분석할 때보다 kernel fuse와 overlap opportunity를 더 쉽게 파낼 수 있습니다. 결과를 얻은 뒤에는 Codex에게 PR을 만들어 성능을 향상시키라고 요청할 수 있으므로 profile과 analysis 기간을 크게 줄일 수 있습니다. 또한 이 SKILL이 있으면 token도 많이 절약됩니다.

또한 저는 개인적으로 Torch Compile을 매우 싫어합니다. 현재 Agent 능력과 이 SKILL 같은 도구를 결합하면 graph layer에서 Torch Compile이 수행하는 kernel fuse/overlap 능력을 상당 부분 가질 수 있다고 생각합니다. 따라서 Torch Compile 쪽을 다시 파고들 필요가 없습니다. Torch Compile을 정말 싫어하는 구체적인 예를 하나 들겠습니다.

SGLang Diffusion mainstream model이 H100, H200, B200에서 Torch Compile을 켰을 때와 끄었을 때의 performance를 테스트했습니다. 각 data group은 3번 측정해 변동이 없을 때만 기록해 data가 신뢰 가능하도록 했습니다.

| Model | eager denoise(s) | compile denoise(s) | compile impact |
| --- | ---: | ---: | ---: |
| flux | 7.0652 | 6.4804 | +8.28% |
| flux2 | 24.1751 | 22.0253 | +8.89% |
| qwen | 12.8252 | 12.5273 | +2.32% |
| qwen-edit | 29.4452 | 28.6798 | +2.60% |
| zimage | 0.6856 | 0.6701 | +2.26% |
| wan-t2v | 7.5818 | 7.3519 | +3.03% |
| wan-ti2v | 58.4174 | 52.6219 | +9.92% |
| ltx2 | 44.9372 | 43.2185 | +3.82% |
| wan-i2v | 8.5144 | 8.4211 | +1.10% |
| hunyuanvideo | 47.6284 | 42.6592 | +10.43% |
| mova-720p | 50.8231 | 50.2117 | +1.20% |
| helios | 94.8758 | 94.1037 | +0.81% |

#### H200

| Model | eager denoise(s) | compile denoise(s) | compile impact |
| --- | ---: | ---: | ---: |
| flux | 6.9427 | 6.3371 | +8.72% |
| flux2 | 24.0664 | 22.0491 | +8.38% |
| qwen | 12.4488 | 12.1663 | +2.27% |
| qwen-edit | 28.5379 | 27.7202 | +2.87% |
| zimage | 0.6719 | 0.6594 | +1.86% |
| wan-t2v | 7.6261 | 7.4161 | +2.75% |
| wan-ti2v | 56.4440 | 50.4035 | +10.70% |
| ltx2 | 42.6128 | 44.4436 | -4.30% |
| wan-i2v | 34.1123 | 34.8305 | -2.11% |
| hunyuanvideo | 46.0962 | 42.4175 | +7.98% |
| mova-720p | 78.1038 | 77.7102 | +0.50% |
| helios | 91.8249 | 92.0677 | -0.26% |

#### B200

| Model | eager denoise(s) | compile denoise(s) | compile impact |
| --- | ---: | ---: | ---: |
| flux | 4.0027 | 3.4524 | +13.75% |
| flux2 | 12.8983 | 11.0723 | +14.16% |
| qwen | 7.5173 | 8.2882 | -10.26% |
| qwen-edit | 16.1784 | 15.5610 | +3.82% |
| zimage | 0.3721 | 0.5699 | -53.15% |
| wan-t2v | 3.5343 | 3.3729 | +4.57% |
| wan-ti2v | 34.0196 | 27.8272 | +18.20% |
| ltx2 | 26.6880 | 33.7862 | -26.60% |
| hunyuanvideo | 26.0052 | 21.7892 | +16.21% |

Diffusion mainstream model에서는 hardware가 달라질 때 Torch Compile이 자주, 그리고 나쁘게 seesaw 현상을 보입니다. 또한 Torch Compile compile time은 video model에서 사람이 받아들이기 어려울 정도로 길어집니다.

이 SKILL로 결과를 얻은 뒤 이어지는 PR은 다음과 같을 수 있습니다. 여기의 analysis result를 바탕으로 fuse/overlap optimization을 시도하고, 효과가 확인되면 PR을 제출합니다.
