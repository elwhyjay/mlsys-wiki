# [LLM 추론 최적화][PTX 어셈블리] CUDA 12 PTX 어셈블리: PRMT 명령어 상세 해설 - 범용 모드

> 원문: https://zhuanlan.zhihu.com/p/660630414

### 0x00 서문

키워드: PRMT.B32 어셈블리 명령어

이전에 NV FasterTransformer의 Weight Only Int8/Int4에서 사용되는 고속 역양자화 기술을 정리할 때 이 두 명령어를 이미 언급했습니다. 그 중 PRMT.B32 어셈블리 명령어는 INT8->FP16/BF16의 고속 역양자화 연산에 사용되며, 연산 단위는 바이트(byte)입니다. LOP3.B32 어셈블리 명령어는 INT4->FP16/BF16의 고속 역양자화 연산에 사용되며, 연산 단위는 비트(bit)입니다. 자세한 내용은 다음을 참조하세요:

이전에 이 두 명령어에 대해서는 NV FasterTransformer에서 사용되는 용법만 다뤘습니다. 예를 들어 PRMT 명령어는 범용 모드만 설명하고 { .f4e, .b4e, .rc8, .ecl, .ecr, .rc16 } 등의 모드의 구체적인 사용법은 소개하지 않았습니다. 따라서 본 글에서는 PRMT와 LOP3 두 명령어를 정리하여 누락된 부분을 보충합니다. 본 내용은 개인적인 CUDA/PTX ISA 어셈블리 노트로만 사용됩니다.

더 많은 기술 노트와 CUDA 학습 노트는 CUDA-Learn-Notes(CUDA Learn Notes with PyTorch)를 참고해 주세요. CUDA-Learn-Notes에는 **LLM/VLM** 문서 정리와 **FlashAttention/SGEMM/HGEMM/GEMV** 등 주요 **CUDA Kernel**의 **예제 구현**이 포함되어 있으며, 현재 **3k+ stars**를 달성했습니다. 링크: https://github.com/xlite-dev/CUDA-Learn-Notes

![](images/A30_llm_ptx_prmt_instruction/v2-cae076e970b2cec6399017ceed59e24a_1440w.png)
CUDA Learn Notes with PyTorch

### 0x01 PRMT 명령어 상세 해설 - 범용 모드

NV PTX ISA 8.1 문서의 9.7.8.7 Data Movement and Conversion Instructions: prmt 섹션을 참고합니다.
- **prmt:** permute bytes from register pair
```
prmt.b32{.mode} d, a, b, c; 
.mode = { .f4e, .b4e, .rc8, .ecl, .ecr, .rc16 }
```

PRMT 명령어는 두 개의 32비트 레지스터 a, b에서 임의의 4개 바이트를 선택하여 32비트 값으로 재구성하고 목적 레지스터에 저장합니다. 범용 형태(모드 미지정)에서 최종 선택되는 4개 바이트는 4개의 4bit 셀렉터로 구성됩니다. PRMT 명령어는 두 소스 레지스터 a, b의 바이트를 0~7로 번호를 매기며, 구체적인 형태는:
```
{b,a} = {{b7,b6,b5,b4}, {b3,b2,b1,b0}}
```

목적 레지스터의 각 바이트에 대해 4비트 셀렉터가 정의됩니다. 셀렉터 값의 하위 3비트(lsb)는 8개 소스 바이트 중 어떤 것을 목적 위치로 이동할지를 지정합니다. msb는 원본 바이트 값을 직접 복사할지, 부호 확장을 할지를 정의합니다. msb=0이면 원본 bit 값을 직접 복사하고, msb=1이면 부호 확장을 수행합니다. 간단히 하기 위해 여기서는 PRMT 명령어의 범용 형태만 다룹니다. (실제로 이 명령어에는 f4e, b4e, rc8 등의 특수 모드도 있습니다)

![](images/A30_llm_ptx_prmt_instruction/v2-f03e55f3f89799aee0072a3cc30aa3f1_1440w.jpg)

다음 명령어에 대해:
```
prmt.b32{.mode} d, a, b, c; 
```

d는 목적 피연산자, a, b는 각각 두 개의 32bit 소스 피연산자, c는 셀렉터입니다. c에서 하위 16비트만 유효합니다(32bit 레지스터를 입력해도). 왜냐하면 d.b_i에는 4개의 결정할 바이트가 있고, 각 바이트는 4bit의 셀렉터가 필요하여 {b,a} = {{b7,b6,b5,b4}, {b3,b2,b1,b0}} 에서 해당 바이트를 선택합니다. 4bit인 이유는 3bit로 0~7을 인덱싱하여 8개 바이트를 커버할 수 있고, 1bit가 부호 확장 여부를 나타내기 때문입니다. 예시:
```
c[3:0] -> 0001 -> msb 0 lsb 001 -> 부호 확장 안 함, index 1의 바이트를 선택, 즉 b1 -> d.b0 = b1
```

### 0x02 총결

본 글은 CUDA 12 PTX ISA 8.1 어셈블리 명령어 세트의 PRMT.B32 명령어 범용 모드의 구체적인 사용법을 정리했습니다. PRMT 명령어의 연산 단위는 바이트(byte)이며, 정수 바이트의 permute 연산에 적합합니다.

지속 업데이트 중, 오탈자는 발견 후 수정합니다...
