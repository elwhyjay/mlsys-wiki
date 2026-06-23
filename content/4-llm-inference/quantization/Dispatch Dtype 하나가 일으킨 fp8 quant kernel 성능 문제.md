# Dispatch Dtype 하나가 일으킨 fp8 quant kernel 성능 문제

최근 SGLang 커뮤니티가 sgl-kernel에서 해결한, `c10::Float8_e4m3fn` 데이터 타입을 잘못 사용해 발생한 일련의 fp8 quant kernel 성능 문제를 간단히 기록합니다.

관련 PR은 다음을 참고하세요: https://github.com/sgl-project/sglang/pull/8499 & https://github.com/sgl-project/sglang/pull/8290 & https://github.com/sgl-project/sglang/pull/8449

이 문제는 커뮤니티의 @strgrb가 발견했습니다. 그는 cute로 `per_token_group_quant_8bit` kernel(PR 8290)을 작성했는데, 이 kernel의 기능은 https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu 안의 kernel과 같습니다. 그런데 cute로 구현한 뒤 보니 sgl-kernel의 kernel보다 성능이 10%-20% 더 빨랐습니다. 이후 토론과 분석을 거쳐, 문제가 kernel이 quantization을 마친 뒤 float->fp8 변환을 할 때 `__nv_fp8_e4m3` dtype을 쓰지 않고 `c10::Float8_e4m3fn`을 쓴 데 있다는 것을 발견했습니다. 그 결과 kernel은 하드웨어 가속 `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` convert 명령을 실제로 사용하지 못했고, 많은 조건 분기와 bit operation을 포함한 software function call로 구현되어 최종적으로 성능이 떨어졌습니다.

처음 quant 관련 kernel을 구현할 때 우리는 이 점을 고려하지 않았고, 지금은 모두 수정했습니다. 처음 이 몇몇 kernel을 구현할 때 Dispatch dtype은 vLLM 쪽 코드를 참고해 `c10::Float8_e4m3fn` 타입을 바로 사용했습니다. 이 타입이 CUDA 내장 `__nv_fp8_e4m3` 타입이 아니라는 점을 고려하지 못했던 것입니다. 우리가 최근 이 수정을 마친 뒤 vLLM도 이 수정사항을 apply한 것을 보았습니다.

![](img/dispatch-dtype-fp8-quant-kernel-performance-97ce85bf/001.png)

![](img/dispatch-dtype-fp8-quant-kernel-performance-97ce85bf/002.png)

세상은 거대한 임시 무대 같습니다. 아래는 H20에서 `per_token_group_quant_8bit`를 수정한 뒤와 수정 전을 NCU로 비교한 결과입니다. 대역폭이 명확히 올라간 것을 볼 수 있습니다.

![](img/dispatch-dtype-fp8-quant-kernel-performance-97ce85bf/003.png)

다음은 sgl-kernel benchmark 도구로 측정한 `per_tensor_quant`와 `per_token_quant`의 성능 비교입니다.

## Benchmark In B200

10%-20% 성능 향상.

### main branch

```shell

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_tensor_quant_fp8.py
INFO 07-28 20:40:56 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-tensor-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    26.272001    25.792001
1         16.0    128.0    36.704000    32.671999
2         16.0    256.0    51.328000    45.024000
3         16.0    512.0    92.160001    77.760004
4         16.0   1024.0   171.744004   143.296003
5         16.0   2048.0   317.344010   264.463991
6         32.0     64.0    36.736000    32.607999
7         32.0    128.0    51.360000    45.024000
8         32.0    256.0    92.160001    77.823997
9         32.0    512.0   171.072006   143.424004
10        32.0   1024.0   317.312002   264.335990
11        32.0   2048.0   604.319990   501.488000
12        64.0     64.0    51.295999    44.992000
13        64.0    128.0    92.160001    77.760004
14        64.0    256.0   170.975998   143.552005
15        64.0    512.0   317.279994   264.351994
16        64.0   1024.0   603.327990   501.568019
17        64.0   2048.0  1178.463995   957.280010
18       128.0     64.0    92.128001    77.791996
19       128.0    128.0   170.816004   143.391997
20       128.0    256.0   317.247987   264.272004
21       128.0    512.0   603.456020   501.215994
22       128.0   1024.0  1178.591967   957.583994
23       128.0   2048.0  2327.423930  1868.095994

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_token_quant_fp8.py 
INFO 07-28 20:41:36 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-token-dynamic-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    21.183999    21.472000
1         16.0    128.0    28.511999    30.751999
2         16.0    256.0    43.903999    47.263999
3         16.0    512.0    78.879997    73.760003
4         16.0   1024.0   140.159994   131.040007
5         16.0   2048.0   265.632004   228.128001
6         16.0   4096.0   518.335998   424.495995
7         32.0     64.0    28.511999    30.688001
8         32.0    128.0    43.712001    47.375999
9         32.0    256.0    78.911997    73.951997
10        32.0    512.0   140.320003   131.072000
11        32.0   1024.0   265.760005   228.095993
12        32.0   2048.0   518.527985   424.416006
13        32.0   4096.0  1014.143944   799.488008
14        64.0     64.0    43.552000    47.263999
15        64.0    128.0    78.752004    73.696002
16        64.0    256.0   140.223995   131.007999
17        64.0    512.0   265.695989   228.192002
18        64.0   1024.0   518.416017   424.383998
19        64.0   2048.0  1014.783978   799.647987
20        64.0   4096.0  2010.143995  1548.256040
21       128.0     64.0    78.815997    73.792003
22       128.0    128.0   140.223995   130.943999
23       128.0    256.0   265.632004   227.904007
24       128.0    512.0   518.335998   424.224004
25       128.0   1024.0  1014.719963   799.647987
26       128.0   2048.0  2009.183884  1548.287988
27       128.0   4096.0  3996.495962  3045.552015
```

### pr

```shell

  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_tensor_quant_fp8.py 
INFO 07-28 21:53:16 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-tensor-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    25.984000    23.968000
1         16.0    128.0    36.160000    28.096000
2         16.0    256.0    51.231999    36.800001
3         16.0    512.0    92.096001    61.087999
4         16.0   1024.0   171.552002   110.656001
5         16.0   2048.0   317.088008   206.367999
6         32.0     64.0    36.095999    27.968001
7         32.0    128.0    51.199999    36.575999
8         32.0    256.0    92.096001    60.927998
9         32.0    512.0   171.072006   110.912003
10        32.0   1024.0   316.832006   206.432000
11        32.0   2048.0   604.031980   399.648011
12        64.0     64.0    51.199999    36.640000
13        64.0    128.0    92.096001    60.736001
14        64.0    256.0   170.992002   110.944003
15        64.0    512.0   316.927999   206.367999
16        64.0   1024.0   603.200018   399.951994
17        64.0   2048.0  1177.872002   772.351980
18       128.0     64.0    93.152002    60.800001
19       128.0    128.0   170.975998   111.231998
20       128.0    256.0   316.895992   206.432000
21       128.0    512.0   603.424013   399.744004
22       128.0   1024.0  1178.319991   773.343980
23       128.0   2048.0  2327.344060  1525.455952

➜  sgl-kernel git:(main) ✗ python3 /home/yineng/bbuf/sglang/sgl-kernel/benchmark/bench_per_token_quant_fp8.py
INFO 07-28 21:52:31 [__init__.py:235] Automatically detected platform cuda.
✅ All implementations match
per-token-dynamic-quant-fp8-performance:
    batch_size  seq_len         VLLM   SGL Kernel
0         16.0     64.0    21.600001    19.936001
1         16.0    128.0    28.352000    28.543999
2         16.0    256.0    43.903999    36.991999
3         16.0    512.0    78.879997    56.768000
4         16.0   1024.0   140.592001   105.952002
5         16.0   2048.0   266.240001   183.487996
6         16.0   4096.0   519.136012   337.280005
7         32.0     64.0    29.088000    28.960001
8         32.0    128.0    44.415999    37.856001
9         32.0    256.0    79.328001    56.832001
10        32.0    512.0   140.415996   105.407998
11        32.0   1024.0   265.695989   182.559997
12        32.0   2048.0   518.735975   336.511999
13        32.0   4096.0  1014.271975   642.080009
14        64.0     64.0    45.952000    37.184000
15        64.0    128.0    79.200000    56.864001
16        64.0    256.0   140.640005   106.016003
17        64.0    512.0   266.207993   183.359995
18        64.0   1024.0   519.263983   337.568015
19        64.0   2048.0  1015.504003   643.136024
20        64.0   4096.0  2060.096025  1255.615950
21       128.0     64.0    81.568003    56.832001
22       128.0    128.0   140.799999   105.696000
23       128.0    256.0   266.463995   182.720006
24       128.0    512.0   519.263983   336.928010
25       128.0   1024.0  1015.679955   643.215984
26       128.0   2048.0  2011.104107  1271.391988
27       128.0   4096.0  4044.928074  2506.623983
```

Nsight Compute에서 kernel의 SASS 코드를 얻을 수 있습니다. 명령을 비교하면 이 성능 향상의 이유를 찾을 수 있습니다.

아래는 수정 후의 `per_token_group_quant_8bit` kernel(https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu)에 대응하는 SASS 코드입니다(NCU Source 부분에서 복사). `group_output[i * vec_size + j] = DST_DTYPE(q_val);` 이 코드 줄에 대응하는 SASS를 분석해 볼 수 있습니다.      
```sass
LDC R1, c[0x0][0x28]
      S2R R3, SR_CTAID.Y
      ULDC UR4, c[0x0][0x0]
      S2R R24, SR_TID.X
      IMAD R3, R3, UR4, RZ
      ULDC UR4, c[0x0][0x22c]
      SHF.R.U32.HI R18, RZ, 0x5, R24
      LEA.HI R18, R3, R18, RZ, 0x1b
      IMAD.SHL.U32 R19, R18, 0x80, RZ
      ISETP.GE.AND P0, PT, R19, UR4, PT
@P0   EXIT
      S2UR UR38, SR_CTAID.X
      IMAD.SHL.U32 R0, R24, 0x4, RZ
      ULDC.64 UR4, c[0x0][0x220]
      ULDC.64 UR36, c[0x0][0x208]
      LOP3.LUT R0, R0, 0x7c, RZ, 0xc0, !PT
      LDC R2, c[0x0][0x230]
      IMAD.IADD R19, R19, 0x1, R0
      SHF.R.S32.HI R22, RZ, 0x1f, R19
      LDC R9, c[0x0][0x24c]
      IMAD R2, R2, UR38, RZ
      IADD3 R0, P0, R2, R19, RZ
      IMAD.X R3, RZ, RZ, R22, P0
      LEA R16, P0, R0, UR4, 0x1
      ULDC UR4, c[0x0][0x244]
      LEA.HI.X R17, R0, UR5, R3, 0x1, P0
      LDG.E.64 R16, desc[UR36][R16.64]
      F2FP.BF16.F32.PACK_AB R0, RZ, UR4
      BSSY B6, 0x7fce7d7d9150
      SHF.R.U64 R23, R16, 0x10, R17
      HFMA2.BF16_V2 R4, -RZ.H0_H0, RZ.H0_H0, |R17|.H0_H0
      SHF.R.U32.HI R2, RZ, 0x10, R17
      PRMT R3, R16, 0x5410, R23
      HFMA2.BF16_V2 R5, -RZ.H0_H0, RZ.H0_H0, |R2|.H0_H0
      HFMA2.BF16_V2 R3, -RZ.H0_H0, RZ.H0_H0, |R3|
      VHMNMX.BF16_V2 R4, R0.H0_H0, R3.H0_H0, R4.H0_H0, !PT
      VHMNMX.BF16_V2 R5, R0.H0_H0, R3.H1_H1, R5.H0_H0, !PT
      HMNMX2.BF16_V2 R4, R4.H0_H0, R5.H0_H0, !PT
      PRMT R4, R4, 0x5410, R4
      SHFL.BFLY PT, R3, R4, 0x10, 0x1f
      HMNMX2.BF16_V2 R3, R4.H0_H0, R3.H0_H0, !PT
      MUFU.RCP R4, R9
      PRMT R3, R3, 0x5410, R3
      SHFL.BFLY PT, R0, R3, 0x8, 0x1f
      HMNMX2.BF16_V2 R0, R3.H0_H0, R0.H0_H0, !PT
      PRMT R0, R0, 0x5410, R0
      SHFL.BFLY PT, R5, R0, 0x4, 0x1f
      HMNMX2.BF16_V2 R5, R0.H0_H0, R5.H0_H0, !PT
      PRMT R5, R5, 0x5410, R5
      SHFL.BFLY PT, R6, R5, 0x2, 0x1f
      HMNMX2.BF16_V2 R6, R5.H0_H0, R6.H0_H0, !PT
      PRMT R6, R6, 0x5410, R6
      SHFL.BFLY PT, R7, R6, 0x1, 0x1f
      HMNMX2.BF16_V2 R3, R6.H0_H0, R7.H0_H0, !PT
      FFMA R7, R4, -R9, 1
      IMAD.U32 R0, R3, 0x10000, RZ
      FFMA R7, R4, R7, R4
      FCHK P0, R0, R9
      FFMA R4, R0, R7, RZ
      FFMA R3, R4, -R9, R0
      FFMA R4, R7, R3, R4
@!P0  BRA 0x7fce7d7d9140
      ULDC UR4, c[0x0][0x24c]
      IMAD.MOV.U32 R4, RZ, RZ, R0
      MOV R20, 0x0
      IMAD.U32 R5, RZ, RZ, UR4
      MOV R21, 0x0
      CALL.ABS.NOINC 0x67c07d7d9f0100
      BSYNC B6
      F2F.BF16.F32 R3, R4
      IMAD.U32 R0, R16, 0x10000, RZ
      ULDC.64 UR4, c[0x0][0x248]
      IMAD.U32 R6, R23, 0x10000, RZ
      IMAD.U32 R8, R17, 0x10000, RZ
      IMAD.U32 R10, R2, 0x10000, RZ
      IMAD.U32 R7, RZ, RZ, UR4
      ULDC UR4, c[0x0][0x234]
      UIMAD UR4, UR38, UR4, URZ
      IMAD.U32 R3, R3, 0x10000, RZ
      FSETP.GE.AND P1, PT, |R3|, 8.50705917302346158658e+37, PT
@P1   FMUL R3, R3, 0.25
      FSETP.GEU.AND P0, PT, |R3|, 1.175494350822287508e-38, PT
@!P0  FMUL R3, R3, 16777216
@!P0  FMUL R0, R0, 16777216
@!P0  FMUL R6, R6, 16777216
@!P0  FMUL R8, R8, 16777216
      MUFU.RCP R5, R3
@!P0  FMUL R10, R10, 16777216
      LOP3.LUT P0, RZ, R24, 0x1f, RZ, 0xc0, !PT
      FMUL R0, R5, R0
      FMUL R2, R5, R6
      FMUL R3, R5, R8
      FMUL R5, R5, R10
@P1   FFMA R0, R0, 0.25, -RZ
@P1   FFMA R2, R2, 0.25, -RZ
@P1   FFMA R3, R3, 0.25, -RZ
@P1   FFMA R5, R5, 0.25, -RZ
      F2FP.BF16.F32.PACK_AB R8, RZ, UR5
      F2FP.BF16.F32.PACK_AB R0, R7, R0
      F2FP.BF16.F32.PACK_AB R2, RZ, R2
      F2FP.BF16.F32.PACK_AB R3, RZ, R3
      F2FP.BF16.F32.PACK_AB R9, RZ, R5
      HMNMX2.BF16_V2 R6, R2.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R5, R0.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R7, R3.H0_H0, R0.H1_H1, !PT
@!P0  LDC.64 R2, c[0x0][0x238]
      HMNMX2.BF16_V2 R9, R9.H0_H0, R0.H1_H1, !PT
      HMNMX2.BF16_V2 R0, R5.H0_H0, R8.H0_H0, PT
      HMNMX2.BF16_V2 R5, R6.H0_H0, R8.H0_H0, PT
      HMNMX2.BF16_V2 R10, R7.H0_H0, R8.H0_H0, PT
      IMAD.U32 R0, R0, 0x10000, RZ
      LDC.64 R6, c[0x0][0x210]
      HMNMX2.BF16_V2 R11, R9.H0_H0, R8.H0_H0, PT
      IMAD.U32 R5, R5, 0x10000, RZ
      IMAD.U32 R10, R10, 0x10000, RZ
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
      IMAD.U32 R11, R11, 0x10000, RZ
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R5, RZ, R5, RZ
@!P0  LDC.64 R8, c[0x0][0x218]
      LOP3.LUT R0, R0, 0xff, RZ, 0xc0, !PT
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R11, RZ, R11, RZ
      LOP3.LUT R5, R5, 0xffff, RZ, 0xc0, !PT
      F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R10, RZ, R10, RZ
@!P0  IMAD R2, R2, UR38, RZ
      LOP3.LUT R11, R11, 0xffff, RZ, 0xc0, !PT
      PRMT R5, R5, 0x7604, R0
@!P0  IMAD R3, R18, R3, R2
      LOP3.LUT R10, R10, 0xff, RZ, 0xc0, !PT
      IMAD.SHL.U32 R11, R11, 0x1000000, RZ
      IADD3 R6, P1, P2, R19, UR4, R6
      PRMT R10, R5, 0x5410, R10
      IADD3.X R7, R22, R7, RZ, P1, P2
      LOP3.LUT R11, R10, R11, RZ, 0xfc, !PT
@!P0  IMAD.WIDE.U32 R8, R3, 0x4, R8
@!P0  STG.E desc[UR36][R8.64], R4
      STG.E desc[UR36][R6.64], R11
      EXIT
      BRA 0x7fce7d7d9590
```

우리는 `group_output[i * vec_size + j] = DST_DTYPE(q_val);` 이 코드 줄에 대응하는 SASS 코드 부분을 찾아야 했지만 SASS 코드에 익숙하지 않았습니다. 이럴 때 Claude 4를 활용하면 꽤 좋습니다.

제가 이 CUDA kernel의 SASS 코드를 분석해 `group_output[i * vec_size + j] = DST_DTYPE(q_val);` 이 코드 줄에 대응하는 SASS 부분을 찾아보겠습니다.

--------------------------------**Claude 4.0 생성 구분선**--------------------------------

Read file: sglang/sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu
이제 SASS 코드를 분석하겠습니다. CUDA 코드의 로직을 기준으로 보면, `group_output[i * vec_size + j] = DST_DTYPE(q_val);` 이 줄은 두 번째 loop 안에서 quantization된 값을 output buffer에 저장하는 데 사용됩니다.

SASS 코드에서 다음 핵심 부분을 식별할 수 있습니다.

1. **데이터 load 부분**(첫 번째 loop):
   - `LDG.E.64 R16, desc[UR36][R16.64]` - 입력 데이터 load
   - 이후의 `SHF.R.U64`, `PRMT` 등 명령은 vector 데이터를 처리합니다.

2. **quantization 계산 부분**:
   - `MUFU.RCP R4, R9` - 역수 계산(나눗셈에 사용)
   - `FMUL R0, R5, R0`, `FMUL R2, R5, R6` 등 - quantization 계산 수행(val / y_s)
   - `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` - float32를 FP8 형식으로 변환

3. **데이터 저장 부분**(`group_output[i * vec_size + j] = DST_DTYPE(q_val);`에 대응):

```sass
STG.E desc[UR36][R6.64], R11
```

이 SASS 코드 줄이 바로 `group_output[i * vec_size + j] = DST_DTYPE(q_val);`에 대응하는 저장 작업입니다.

구체적으로는:

- `STG.E` - global memory store 명령
- `desc[UR36][R6.64]` - target address이며, `R6.64`에는 계산된 store address `(i * vec_size + j)`가 들어 있습니다.
- `R11` - 저장할 데이터이며, quantization 변환을 거친 값입니다.

SASS 코드에서 주소 계산은 다음 명령으로 이루어집니다.
```sass
IADD3 R6, P1, P2, R19, UR4, R6
IADD3.X R7, R22, R7, RZ, P1, P2
```

데이터 준비는 다음을 통해 이루어집니다.
```sass
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R5, RZ, R5, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R11, RZ, R11, RZ
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R10, RZ, R10, RZ
```

이 명령들이 float32 값을 FP8 형식으로 변환한 뒤, bit operation으로 최종 저장값 `R11`을 조합합니다.

따라서 **`group_output[i * vec_size + j] = DST_DTYPE(q_val);` 이 CUDA 코드 줄에 대응하는 SASS 코드는 `STG.E desc[UR36][R6.64], R11`**입니다.

--------------------------------**Claude 4.0 생성 구분선**--------------------------------

SASS 코드 분석에 따르면 `DST_DTYPE(q_val)` 타입 변환에 대응하는 명령은 다음입니다.

```sass
F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R0, RZ, R0, RZ
```

이 명령은 float32를 FP8(E4M3) 형식으로 변환합니다.
- `F2FP` - Float to Float Packed(부동소수점 packed 변환)
- `SATFINITE` - saturation과 finite 검사
- `E4M3` - exponent 4비트, mantissa 3비트의 FP8 형식
- `F32` - source 형식은 float32
- `PACK_AB_MERGE_C` - pack 및 merge operation

그런데 `DST_DTYPE`을 `c10::Float8_e4m3fn`으로 설정하면, 즉 수정 전 상황에서는 SASS 코드가 매우 복잡해지고 많은 function call이 추가됩니다.

```sass
// 1st call
MOV R8, 0xc50
CALL.REL.NOINC 0x7fce7d7975c0

// 2nd call  
MOV R8, 0xe50
CALL.REL.NOINC 0x7fce7d7975c0

// 3rd call
MOV R8, 0x1080
CALL.REL.NOINC 0x7fce7d7975c0

// 4th call
MOV R8, 0x1290
CALL.REL.NOINC 0x7fce7d7975c0

// 5th call
MOV R8, 0x14b0
CALL.REL.NOINC 0x7fce7d7975c0

// 6th call
MOV R8, 0x16c0
CALL.REL.NOINC 0x7fce7d7975c0

// 7th call
MOV R8, 0x18e0
CALL.REL.NOINC 0x7fce7d7975c0

// 8th call
MOV R8, 0x1af0
CALL.REL.NOINC 0x7fce7d7975c0

// 9th call
MOV R8, 0x1d10
CALL.REL.NOINC 0x7fce7d7975c0
```

이 함수가 하는 일은 float32를 `c10::Float8_e4m3fn` 타입으로 변환하는 것입니다. 이 타입은 CUDA 내장 `__nv_fp8_e4m3` 타입이 아니므로 software function call로 구현되어야 하고, 많은 조건 분기와 bit operation을 포함합니다. 그 결과 성능이 떨어졌습니다.

![](img/dispatch-dtype-fp8-quant-kernel-performance-97ce85bf/004.png)

아래에서는 Claude 4.0으로 이 변환 함수를 분석해 보겠습니다.

--------------------------------**Claude 4.0 생성 구분선**--------------------------------

제가 이 SASS 코드에 주석 설명을 붙여 보겠습니다.

```sass
@!P0  BRA 0x7fce7d796440                    // 조건 P0가 false이면 주소 0x7fce7d796440으로 분기
EXIT                                        // 현재 실행 block 종료
SHF.R.U32.HI R9, RZ, 0x17, R3              // R3를 오른쪽으로 23비트 shift하고 결과를 R9에 저장(float32의 exponent 부분 추출)
BSSY B1, 0x7fce7d797c10                     // 동기화 block B1 시작, target address는 0x7fce7d797c10
SHF.R.U32.HI R11, RZ, 0x17, R0             // R0를 오른쪽으로 23비트 shift하고 결과를 R11에 저장(다른 float32의 exponent 부분 추출)
IMAD.MOV.U32 R17, RZ, RZ, R3               // R3 값을 R17로 이동
LOP3.LUT R9, R9, 0xff, RZ, 0xc0, !PT       // R9와 0xff를 AND해 하위 8비트 추출(exponent 값의 유효 비트)
LOP3.LUT R11, R11, 0xff, RZ, 0xc0, !PT     // R11과 0xff를 AND해 하위 8비트 추출(exponent 값의 유효 비트)
VIADD R12, R9, 0xffffffff                  // R12 = R9 - 1(exponent 값에서 1 감소)
VIADD R13, R11, 0xffffffff                 // R13 = R11 - 1(exponent 값에서 1 감소)
ISETP.GT.U32.AND P0, PT, R12, 0xfd, PT     // 조건 P0 설정: R12 > 253(0xfd)이면 true
ISETP.GT.U32.OR P0, PT, R13, 0xfd, P0      // 조건 P0 설정: R13 > 253(0xfd)이거나 이전 P0가 true이면 true
@!P0  IMAD.MOV.U32 R10, RZ, RZ, RZ         // P0가 false이면 R10을 0으로 설정
```

## 기능 설명

이 SASS 코드의 주요 기능은 **FP8 변환의 전처리와 경계 검사**입니다.

1. **exponent 추출**:
   - `SHF.R.U32.HI R9, RZ, 0x17, R3` 및 `SHF.R.U32.HI R11, RZ, 0x17, R0`
   - float32 값에서 exponent 부분을 추출합니다(IEEE 754 형식에서 exponent는 23-30비트에 위치).

2. **exponent 처리**:
   - `LOP3.LUT` 명령으로 exponent의 유효 비트를 추출합니다(부호 비트 제거).
   - `VIADD` 명령으로 exponent 값을 1 줄입니다(뒤의 변환 계산에 사용).

3. **경계 검사**:
   - `ISETP.GT.U32.AND/OR` 명령으로 exponent 값이 253을 초과하는지 확인합니다.
   - 이는 FP8 형식의 exponent 범위 검사이며, 변환된 값이 유효 범위 안에 있도록 보장합니다.

4. **조건 설정**:
   - 경계 검사 결과에 따라 condition flag P0를 설정합니다.
   - 범위를 넘으면 P0가 true이고, 그렇지 않으면 false입니다.

이 코드는 FP8 변환 함수의 일부로, float32를 FP8 형식으로 변환하기 전에 필요한 전처리와 유효성 검사를 담당합니다.

--------------------------------**Claude 4.0 생성 구분선**--------------------------------


최종적으로 문제의 원인은 분명해졌습니다. `c10::Float8_e4m3fn`은 CUDA 내장 `__nv_fp8_e4m3` 타입이 아니므로 software function call로 구현해야 하고, 여기에는 많은 조건 분기와 bit operation이 포함되어 성능이 떨어졌습니다. quant 관련 kernel을 처음 구현할 때 우리가 이 점을 고려하지 않았던 것이며, 지금은 모두 수정했습니다.
