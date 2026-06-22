# CUDA Register Bank Conflict 문제 탐구

## 문제 설명

CUDA 프로그램을 작성하다가 다음을 발견했습니다. FFMA 명령의 레지스터가 모두 같은 bank에 있을 때, 이것이 성능 문제를 일으킬까요?

### CUDA-C 소스 코드

```c
// global memory에서 float4 벡터 두 개를 로드한다.
float4 vec_A = *ptr_A;
float4 vec_B = *ptr_B;

// 결과 벡터를 0으로 초기화한다.
float4 vec_C = float4{ 0.0f, 0.0f, 0.0f, 0.0f };

// loop unrolling으로 각 component에 FMA (Fused Multiply-Add) 연산을 수행한다.
#pragma unroll(64)
for(int i = 0 ; i < loop ; i ++){
    // vec_C.x = vec_A.x * vec_B.x + vec_C.x
    vec_C.x = fmaf(vec_A.x, vec_B.x, vec_C.x);
    vec_C.y = fmaf(vec_A.y, vec_B.y, vec_C.y);
    vec_C.z = fmaf(vec_A.z, vec_B.z, vec_C.z);
    vec_C.w = fmaf(vec_A.w, vec_B.w, vec_C.w);
}

// 결과를 global memory에 다시 쓴다.
*ptr_C = vec_C;
```

### 역어셈블한 SASS 코드

```sass
......

// FFMA 명령 형식: FFMA 대상 레지스터, 소스 레지스터1, 소스 레지스터2, 소스 레지스터3
// 실행: 대상 = 소스1 * 소스2 + 소스3

/*0190*/ FFMA R17, R4, R8, R12 ;   // R17 = R4 * R8 + R12
                                    // R4, R8, R12가 모두 4의 배수 -> 가능한 bank conflict

/*01a0*/ FFMA R12, R5, R9, R13 ;   // R12 = R5 * R9 + R13

/*01b0*/ FFMA R13, R6, R10, R14 ;  // R13 = R6 * R10 + R14

/*01c0*/ FFMA R14, R7, R11, R15 ;  // R14 = R7 * R11 + R15

......
```

### 문제 분석

공개 자료에 따르면 CUDA의 register bank 할당 규칙은 `Rx % 4`입니다.

첫 번째 명령을 예로 들면 다음과 같습니다.

```
FFMA R17, R4, R8, R12
```

- R4 % 4 = 0 (bank 0)
- R8 % 4 = 0 (bank 0)
- R12 % 4 = 0 (bank 0)

세 소스 operand가 모두 같은 bank에 있으므로 이론적으로는 bank conflict가 발생합니다.

**테스트 환경:**

- 컴파일러: nvcc 11.8
- 아키텍처: compute_86 (Ampere)

## 답변 1: GPU 아키텍처 발전이 가져온 변화

### 역사적 배경: 초기 아키텍처의 Register Bank Conflict

**Kepler/Maxwell/Pascal** 시대에는 다음과 같았습니다.

- **레지스터 구조**: 4-bank 구조
- **각 bank**: 한 주기에 32-bit operand 하나만 서비스 가능
- **Bank 할당 규칙**: `bank = reg_id % 4`
- **문제**: 한 명령의 3개 소스 operand가 모두 같은 bank에 떨어지면 bank conflict가 생겨 latency가 증가함

예: `FFMA R17, R4, R8, R12`

- R4 -> bank 0
- R8 -> bank 0
- R12 -> bank 0
- **결과**: 세 소스 레지스터가 모두 같은 bank에 있어 conflict 발생

---

### Volta (SM70) 아키텍처의 큰 개선

**Volta**부터 NVIDIA는 register file을 크게 개선했습니다.

#### 1. Dual-Ported Register File

- 각 bank가 **한 주기 안에 operand 두 개**를 제공할 수 있음
- **2 read 1 write** (2R1W) 지원

#### 2. 스케줄링/리네이밍 메커니즘

- FFMA/FADD/FMAD 같은 3-source operand 명령은 스케줄링/리네이밍 메커니즘을 사용함
- 3개 소스 레지스터가 모두 같은 bank에서 오더라도 하드웨어가 한 주기 안에 read 작업을 완료할 수 있음
- Pascal처럼 엄격한 conflict가 더 이상 발생하지 않음

#### 3. 컴파일러 동작 변화

- Volta+ 아키텍처에서 NVCC/ptxas는 **"레지스터 번호를 흩뜨리는" 최적화를 거의 적극적으로 수행하지 않음**
- 하드웨어가 이미 뒷받침할 수 있기 때문

---

### Ampere (compute_86) 아키텍처의 동작

사용한 **arch=86 (Ampere)** 아키텍처는 다음과 같습니다.

- register file은 여전히 **dual-ported**
- 한 bank가 주기마다 **operand 두 개**를 읽을 수 있음
- **3-source 명령 (FFMA)**에 대해:
  - 세 소스가 모두 같은 bank에 매핑되더라도 예전 아키텍처처럼 뚜렷한 penalty가 생기지는 않음
  - 모든 operand access pattern이 매우 겹치는 일부 corner case에서는 아주 작은 구조적 conflict가 있을 수 있음
  - 하지만 대부분의 경우 하드웨어가 스케줄링/latency hiding으로 가려 줌

---

### NVIDIA 공식 설명

NVIDIA는 Volta 백서에서 다음을 언급했습니다.

> "The register file has been redesigned to greatly reduce bank conflicts."

---

### 요약: 아키텍처별 최적화 전략

| 아키텍처 | Register Bank Conflict | 최적화 제안 |
|------|----------------------|---------|
| **Pascal 및 이전** | 신경 써야 함 | 수동 padding / 레지스터 shuffle |
| **Volta / Turing / Ampere / Ada** | 기본적으로 크게 신경 쓰지 않아도 됨 | 컴파일러가 더 이상 의도적으로 레지스터를 흩뜨리지 않으며, 하드웨어가 이미 최적화됨 |

**결론**: 그렇습니다. 현재 (Volta+)는 기본적으로 register bank conflict를 크게 신경 쓰지 않아도 됩니다. 하드웨어가 이미 충분한 최적화를 수행했기 때문입니다.

---

### 검증 방법

정말 bank conflict가 존재하는지 확인하고 싶다면 **Nsight Compute**로 성능 분석을 할 수 있습니다.

**핵심 성능 카운터:**

- `sm__inst_executed_pipe_*.sum`
- `sm__sass_reg_bank_conflicts`

Ampere에서 검증해 보면 bank conflict stall이 거의 없다는 것을 확인할 수 있을 것입니다.

**권장 실험:**

작은 실험 kernel과 Nsight Compute 카운터 비교를 작성해, bank conflict 유무에 따른 실제 성능 차이를 검증합니다.

## 답변 2: Volta 이후의 레지스터 아키텍처 변화

### 핵심 변화

Volta 이후 register file 아키텍처에는 큰 변화가 있었습니다.

**Register file 구조:**

- **Bank 수**: 기존 4 bank에서 2 bank로 변경
- **Access pattern**: 2R1W
- **Conflict 규칙**: 3개 소스 operand가 "모두 홀수 또는 모두 짝수"가 아니면 bank conflict가 발생하지 않음

**예:**

```
R4 (짝수), R8 (짝수), R12 (짝수) -> conflict 가능
R4 (짝수), R5 (홀수), R8 (짝수) -> conflict 없음
```

---

### 추가 최적화 메커니즘

**Reuse Cache:**

- 컴파일러가 bank conflict 없는 SASS 코드를 더 쉽게 생성할 수 있음
- 레지스터 값 재사용이 access conflict를 추가로 줄임

---

### 현대 GPU의 관심점 이동

GPU 아키텍처가 발전하면서 최적화 초점도 바뀌었습니다.

#### Tensor Core 시대의 변화

| 항목 | 전통 CUDA Core | 현대 Tensor Core |
|------|---------------|-----------------|
| **주요 연산 성능 원천** | FFMA/FMAD 명령 | MMA (Matrix Multiply-Accumulate) |
| **데이터 출처** | Register | Shared Memory |
| **병목** | Register bank conflict | Shared Memory bandwidth |

**H100 이후의 흐름:**

- MMA 명령의 소스 operand는 주로 **Shared Memory**에서 오며 Register가 아님
- Tensor Core 연산 성능 증가는 WMMA (Warp Matrix Multiply-Accumulate)가 Register에서 읽는 능력을 크게 앞섬
- **결론**: Register bank conflict의 중요성은 더 낮아짐

## 답변 3: Conflict는 여전히 존재하지만, 구체적 상황을 분석해야 함

### Conflict가 생기는 근본 원인

하드웨어가 이미 최적화되었더라도, 일부 상황에서는 register bank conflict가 여전히 존재합니다.

#### 컴파일러의 레지스터 할당 전략

**문제 상황:**

```c
// shared memory에서 float4 세 개를 로드한다.
float4 vec_A = *ptr_A;  // R4-R7에 할당
float4 vec_B = *ptr_B;  // R8-R11에 할당
float4 vec_C = ...;     // R12-R15에 할당
```

**컴파일러 동작:**

1. `float4` 세 개를 shared memory에서 로드하고, 이는 세 개의 **연속 주소**에 대응함
2. 컴파일러가 레지스터를 할당할 때 자연스럽게 **연속 레지스터**를 할당함
3. 각 `float4`의 **같은 위치** 데이터에 FMA 연산을 수행하면 자연스럽게 register bank conflict가 나타남

**예:**

```
vec_C.x = fmaf(vec_A.x, vec_B.x, vec_C.x);
// 대응: FFMA R12, R4, R8, R12
// R4, R8, R12가 모두 4의 배수 -> 같은 bank
```

---

### 해결책

Register bank conflict를 해결하는 사고방식은 **shared memory bank conflict**를 해결하는 방식과 같습니다.

#### 방법 1: Layout 변환

- 미리 데이터 layout을 변환함
- 연속 레지스터 할당 패턴을 깨뜨림

#### 방법 2: Swizzle 변환

- **Shared memory 단계**에서 swizzle 변환을 수행함
- 이후 bank conflict를 덜 고려해도 되게 만들 수 있음
- **권장**: 성능 요구가 높은 상황에서는 shared memory 단계에서 최적화

---

### 실제 고려 사항

이 문제의 단순한 예제에 대해서는 다음과 같습니다.

**현실적 상황:**

- 데이터 양이 작음
- Kernel launch 자체의 오버헤드가 작지 않음
- bank conflict가 있는지 여부가 결과상 **그렇게 뚜렷해 보이지 않음**

**결론**:

이 정도 데이터에 transpose나 swizzle을 하는 것은 **조금 과한 조치**입니다.

---

### 최적화 제안 요약

| 상황 | 최적화 필요 여부 | 최적화 방법 |
|------|------------|---------|
| **소규모 계산** | 필요 없음 | Kernel launch 오버헤드가 더 큼 |
| **성능 핵심 경로** | 필요함 | Shared Memory 단계에서 swizzle |
| **현대 아키텍처 (Volta+)** | 일반적으로 필요 없음 | 하드웨어가 이미 최적화됨 |
| **구형 아키텍처 (Pascal-)** | 필요함 | 수동 padding / 레지스터 shuffle |

---

## 정리

### 세 가지 관점 종합

1. **답변 1**: 현대 아키텍처(Volta+)는 하드웨어가 크게 최적화되어 기본적으로 신경 쓰지 않아도 됨
2. **답변 2**: 아키텍처 변화(2 bank + 2R1W)와 Tensor Core 시대의 관심점 이동
3. **답변 3**: 특정 상황에서는 여전히 고려해야 하지만, 최적화 비용을 저울질해야 함

### 최종 제안

- **Ampere (sm_86) 및 최신 아키텍처**: 일반적으로 register bank conflict를 걱정하지 않아도 됨
- **성능 민감 코드**: Nsight Compute로 검증하고 필요할 때만 최적화
- **최적화 우선순위**: Shared Memory > Register > 기타
