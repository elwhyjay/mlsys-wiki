# TVM schedule 상세 예시

schedule과 계산 로직의 분리는 자동 코드 생성 기술의 핵심 개념으로, MIT CASIL 그룹의 Jonathan Ragan-Kelley가 2012년 SIGGRAPH에서 발표한 논문에서 처음 제시되었고, 이후 2013년 PLDI 논문에서 schedule에 대한 정확한 정의가 주어졌다[[1]](<https://zhuanlan.zhihu.com/p/94846767#ref_1>).

  1. When and where should be the value at each coordinate in each function be computed?
  2. Where should they be stored?
  3. How long are values cached and communicated across multiple consumers, and when are they independently recomputed by each?



실제로 schedule이란 일련의 최적화 선택들의 집합이다. 이러한 선택은 계산 결과에는 영향을 주지 않지만, architecture에 대한 이해를 담고 있기 때문에 성능에는 결정적이다. 흔히 하나 또는 여러 선택의 조합을 통틀어 schedule이라고 부른다. Halide와 TVM 공식 사이트에는 다양한 schedule이 자세히 소개되어 있지만, 각 schedule이 알고리즘에 어떤 영향을 주는지 직관적으로 비교할 수 있는 예시는 부족하다(다만 Halide 공식 사이트의 몇몇 애니메이션은 매우 잘 만들어져 있다). 그래서 이 글에서는 구체적인 예시를 통해 각 schedule을 보여주고자 한다.

이 글에서는 `tvm.schedule`에 포함된 API들을 정리하고, 구체적인 프로그램 예시가 생성하는 중간 코드(IR)를 통해 사용 방법과 schedule 효과를 자세히 설명한다.

TVM guide 주소: [https://docs.tvm.ai/api/python/schedule.html](<https://link.zhihu.com/?target=https%3A//docs.tvm.ai/api/python/schedule.html>)

예시 코드 링크: [https://github.com/StrongSpoon/tvm.schedule](<https://link.zhihu.com/?target=https%3A//github.com/StrongSpoon/tvm.schedule>)

(구분선 위쪽이 schedule 적용 전의 중간 코드이고, 아래쪽이 schedule 적용 후의 결과이다. 예시 코드와 함께 보면 더 이해하기 쉽다.)

schedule을 이해하는 일은 본질적으로 컴퓨터 시스템 구조, loop optimization, 그리고 멀티스레드 프로그래밍을 함께 이해하는 과정이므로, 이 글에서도 순서대로 다룬다.

## **1. 저장 계층(storage hierarchy) 관련 schedule**

**cache_read(tensor, scope, readers)**
    
    
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    
    ---------cutting line---------
    // attr [A.shared] storage_scope = "shared"
    allocate A.shared[float32 * 1048576]
    produce A.shared {
      for (ax0, 0, 1024) {
        for (ax1, 0, 1024) {
          A.shared[((ax0*1024) + ax1)] = A[((ax0*1024) + ax1)]
        }
      }
    }
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A.shared[((i*1024) + k)])
        }
      }
    }
    

> `cache_read`는 tensor를 지정된 저장 계층 scope의 cache로 읽어 들인다. 이 설계의 의의는 현재 계산 장치의 on-chip memory hierarchy를 명시적으로 활용하는 데 있다. 이 예시에서는 먼저 `A`의 데이터를 shared memory에 load한 다음 `B`를 계산한다. 여기서 stage라는 개념을 도입해야 하는데, 하나의 op는 하나의 stage에 대응하며, 따라서 `cache_read`를 호출하면 stage가 하나 늘어난다.

**cache_write(tensor, scope)**
    
    
     produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    
    ---------cutting line---------
    // attr [B.local] storage_scope = "local"
    allocate B.local[float32 * 1024]
    produce B.local {
      for (i.c, 0, 1024) {
        B.local[i.c] = 0f
        for (k, 0, 1024) {
          B.local[i.c] = (B.local[i.c] + A[((i.c*1024) + k)])
        }
      }
    }
    produce B {
      for (i, 0, 1024) {
        B[i] = B.local[i]
      }
    }
    

> `cache_write`는 `cache_read`에 대응하는 schedule로, 먼저 shared memory에 계산 결과를 저장한 뒤 마지막에 결과를 global memory로 write back한다. 물론 실제 시나리오에서는 보통 결과를 먼저 register에 두었다가 마지막에 write back하는 경우가 많다.

**set_scope**
    
    
     // attr [B] storage_scope = "global"
    allocate B[float32 * 1024]
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    produce C {
      for (i, 0, 1024) {
        C[i] = (B[i] + 10f)
      }
    }
    
    ---------cutting line---------
    // attr [B] storage_scope = "shared"
    allocate B[float32 * 1024]
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    produce C {
      for (i, 0, 1024) {
        C[i] = (B[i] + 10f)
      }
    }
    

> `set_scope`는 stage의 계산 결과가 위치하는 저장 계층을 지정하여 tensor에 가장 적절한 저장 위치를 선택한다. 스레드 간에 공유되는 메모리를 설정할 때 유용하다. 사실 `set_scope`는 `cache_read`의 부분 연산이라 할 수 있다.

**storage_align**
    
    
     // attr [A.shared] storage_scope = "shared"
    allocate A.shared[float32 * 1048576]
    produce A.shared {
      for (ax0, 0, 1024) {
        for (ax1, 0, 1024) {
          A.shared[((ax0*1024) + ax1)] = A[((ax0*1024) + ax1)]
        }
      }
    }
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A.shared[((i*1024) + k)])
        }
      }
    }
    
    ---------cutting line---------
    // attr [A.shared] storage_scope = "shared"
    allocate A.shared[float32 * 1134592]
    produce A.shared {
      for (ax0, 0, 1024) {
        for (ax1, 0, 1024) {
          A.shared[((ax0*1108) + ax1)] = A[((ax0*1024) + ax1)]
        }
      }
    }
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A.shared[((i*1108) + k)])
        }
      }
    }
    

> `storage_align`은 stage에 해당하는 저장 공간을 factor 단위로, offset만큼의 편이를 두고 다시 정렬한다. 이를 통해 GPU shared memory 접근 시의 bank conflict를 피할 수 있다. bank conflict에 대해서는 [[2]](<https://zhuanlan.zhihu.com/p/94846767#ref_2>)를 참고하면 된다.

**compute_at**
    
    
     // attr [B.rf] storage_scope = "global"
    allocate B.rf[float32 * 32]
    produce B.rf {
      for (k.inner, 0, 32) {
        B.rf[k.inner] = 0f
        for (k.outer, 0, 32) {
          B.rf[k.inner] = (B.rf[k.inner] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    produce B {
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
      // attr [reduce_temp0] storage_scope = "local"
      allocate reduce_temp0[float32 * 1]
      // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
      tvm_thread_allreduce((uint32)1, B.rf[threadIdx.x], (bool)1, reduce_temp0, threadIdx.x)
      B[0] = reduce_temp0[0]
    }
    
    ---------cutting line---------
    produce B {
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
      // attr [B.rf] storage_scope = "local"
      allocate B.rf[float32 * 1]
      // attr [reduce_temp0] storage_scope = "local"
      allocate reduce_temp0[float32 * 1]
      produce B.rf {
        B.rf[0] = 0f
        for (k.outer, 0, 32) {
          B.rf[0] = (B.rf[0] + A[((k.outer*32) + threadIdx.x)])
        }
      }
      // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
      tvm_thread_allreduce((uint32)1, B.rf[0], (bool)1, reduce_temp0, threadIdx.x)
      B[0] = reduce_temp0[0]
    }
    

> `compute_at`은 현재 stage를 대상 stage의 지정된 iter 방향에 부착(attach)하여, 대상 stage와 동일한 병렬 방식을 따르면서 그 내부에서 현재 stage의 계산을 수행하도록 만든다. `compute_at`은 보통 `cache_read`, `cache_write`와 함께 사용된다.

**compute_inline**
    
    
     // attr [Apad] storage_scope = "global"
    allocate Apad[float32 * 1056784]
    produce Apad {
      for (yy, 0, 1028) {
        for (xx, 0, 1028) {
          Apad[((yy*1028) + xx)] = tvm_if_then_else(((((2 <= yy) && (yy < 1026)) && (2 <= xx)) && (xx < 1026)), A[(((yy*1024) + xx) - 2050)], 0f)
        }
      }
    }
    produce B {
      for (yy, 0, 1026) {
        for (xx, 0, 1026) {
          B[((yy*1026) + xx)] = 0f
          for (ry, 0, 3) {
            for (rx, 0, 3) {
              B[((yy*1026) + xx)] = (B[((yy*1026) + xx)] + (Apad[((((yy*1028) + (ry*1028)) + xx) + rx)]*W[((ry*3) + rx)]))
            }
          }
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      for (yy, 0, 1026) {
        for (xx, 0, 1026) {
          B[((yy*1026) + xx)] = 0f
          for (ry, 0, 3) {
            for (rx, 0, 3) {
              B[((yy*1026) + xx)] = (B[((yy*1026) + xx)] + (tvm_if_then_else(((((2 <= (yy + ry)) && ((yy + ry) < 1026)) && (2 <= (xx + rx))) && ((xx + rx) < 1026)), A[(((((yy*1024) + (ry*1024)) + xx) + rx) - 2050)], 0f)*W[((ry*3) + rx)]))
            }
          }
        }
      }
    }
    

> `compute_inline`은 독립적인 계산 연산을 inline 함수 형태로 바꾸어, 원래 계산 결과가 사용되는 시점에 inline 함수를 호출해 연산을 수행하게 한다. 이를 통해 stage를 하나 줄일 수 있다.

**compute_root**
    
    
     produce B {
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
      // attr [B.rf] storage_scope = "local"
      allocate B.rf[float32 * 1]
      // attr [reduce_temp0] storage_scope = "local"
      allocate reduce_temp0[float32 * 1]
      produce B.rf {
        B.rf[0] = 0f
        for (k.outer, 0, 32) {
          B.rf[0] = (B.rf[0] + A[((k.outer*32) + threadIdx.x)])
        }
      }
      // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
      tvm_thread_allreduce((uint32)1, B.rf[0], (bool)1, reduce_temp0, threadIdx.x)
      B[0] = reduce_temp0[0]
    }
    
    ---------cutting line---------
    // attr [B.rf] storage_scope = "global"
    allocate B.rf[float32 * 32]
    produce B.rf {
      for (k.inner, 0, 32) {
        B.rf[k.inner] = 0f
        for (k.outer, 0, 32) {
          B.rf[k.inner] = (B.rf[k.inner] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    produce B {
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
      // attr [reduce_temp0] storage_scope = "local"
      allocate reduce_temp0[float32 * 1]
      // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
      tvm_thread_allreduce((uint32)1, B.rf[threadIdx.x], (bool)1, reduce_temp0, threadIdx.x)
      B[0] = reduce_temp0[0]
    }
    

> `compute_root`는 `compute_at`의 반대 연산이다. 아무 schedule도 적용하지 않을 경우 각 stage는 기본적으로 `compute_root` 상태이므로, 이 schedule은 이전에 어떤 stage에 적용해 두었던 `compute_at` 연산을 취소하는 것과 같다.

## 2. 일반적인 loop 최적화

**fuse**
    
    
     produce B {
      B[0] = 0f
      for (k.outer, 0, 32) {
        for (k.inner, 0, 32) {
          B[0] = (B[0] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      B[0] = 0f
      for (k.outer.k.inner.fused, 0, 1024) {
        B[0] = (B[0] + A[k.outer.k.inner.fused])
      }
    }
    

> `fuse`는 두 iter를 융합(fuse)하여 두 단계의 loop를 하나로 합치는 데 사용된다. 반환값은 iter 타입이며, 여러 번에 걸쳐 융합할 수도 있다.

**split**
    
    
     produce B {
      B[0] = 0f
      for (k, 0, 1024) {
        B[0] = (B[0] + A[k])
      }
    }
    
    ---------cutting line---------
    produce B {
      B[0] = 0f
      for (k.outer, 0, 32) {
        for (k.inner, 0, 32) {
          B[0] = (B[0] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    

> `split`은 `fuse`의 반대 연산으로, iter를 factor 단위로 outer/inner 두 개의 iteration으로 분리하여 loop 깊이를 한 단계 늘린다. loop 작업을 더 작은 하위 task로 나누는 데 쓰인다. 예를 들어 CUDA에서는 `gridDim`과 `blockDim`이 최대 3차원까지 가능하므로, `split`을 통해 grid나 block에 bind할 새로운 차원을 만들어낼 수 있다[[3]](<https://zhuanlan.zhihu.com/p/94846767#ref_3>).

**reorder**
    
    
     produce C {
      for (i.outer, 0, 32) {
        for (i.inner, 0, 32) {
          for (j.outer, 0, 32) {
            for (j.inner, 0, 32) {
              C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = (A[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + B[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)])
            }
          }
        }
      }
    }
    
    ---------cutting line---------
    produce C {
      for (i.outer, 0, 32) {
        for (j.outer, 0, 32) {
          for (j.inner, 0, 32) {
            for (i.inner, 0, 32) {
              C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = (A[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + B[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)])
            }
          }
        }
      }
    }
    

> `reorder`는 loop iter의 내외 순서를 재배치하는 데 사용된다. locality 원리에 따라 cache에 이미 올라와 있는 데이터를 최대한 활용함으로써, 데이터를 반복적으로 load/store하는 상황을 줄인다. 어떤 순서가 최적인지는 매우 흥미로운 문제이다. 행렬 곱셈을 예로 들면 M, N, K 세 차원 중 K를 가장 바깥쪽에 두는 것이 locality를 가장 잘 살리는 경우가 많다. 구체적인 예시는 그때그때 따져봐야 한다.

**tile**
    
    
     produce C {
      for (i, 0, 1024) {
        for (j, 0, 1024) {
          C[((i*1024) + j)] = 0f
          for (K, 0, 1024) {
            C[((i*1024) + j)] = (C[((i*1024) + j)] + (A[((i*1024) + K)]*B[((K*1024) + j)]))
          }
        }
      }
    }
    
    ---------cutting line---------
    produce C {
      for (i.outer, 0, 32) {
        for (j.outer, 0, 32) {
          for (i.inner, 0, 32) {
            for (j.inner, 0, 32) {
              C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = 0f
              for (K, 0, 1024) {
                C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = (C[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + (A[(((i.outer*32768) + (i.inner*1024)) + K)]*B[(((K*1024) + (j.outer*32)) + j.inner)]))
              }
            }
          }
        }
      }
    }
    

> `tile`은 stage의 두 차원을 각각의 factor에 따라 분할하고, 정해진 순서대로 outer 두 개와 inner 두 개의 iter를 반환한다. 이를 통해 loop 깊이를 늘리고 더 작은 단위의 계산 task로 만든다. 사실 `tile`은 `split`과 `reorder`로 구현 가능하며, 행렬 곱셈이나 convolution 계산에서 매우 중요한 schedule이다.

**unroll**
    
    
     produce C {
      for (i.outer, 0, 256) {
        for (i.inner, 0, 4) {
          for (j, 0, 1024) {
            C[(((i.outer*4096) + (i.inner*1024)) + j)] = (A[(((i.outer*4096) + (i.inner*1024)) + j)] + B[(((i.outer*4096) + (i.inner*1024)) + j)])
          }
        }
      }
    }
    
    ---------cutting line---------
    produce C {
      for (i.outer, 0, 256) {
        for (j, 0, 1024) {
          C[((i.outer*4096) + j)] = (A[((i.outer*4096) + j)] + B[((i.outer*4096) + j)])
        }
        for (j, 0, 1024) {
          C[(((i.outer*4096) + j) + 1024)] = (A[(((i.outer*4096) + j) + 1024)] + B[(((i.outer*4096) + j) + 1024)])
        }
        for (j, 0, 1024) {
          C[(((i.outer*4096) + j) + 2048)] = (A[(((i.outer*4096) + j) + 2048)] + B[(((i.outer*4096) + j) + 2048)])
        }
        for (j, 0, 1024) {
          C[(((i.outer*4096) + j) + 3072)] = (A[(((i.outer*4096) + j) + 3072)] + B[(((i.outer*4096) + j) + 3072)])
        }
      }
    }
    

> `unroll`은 흔히 사용되는 loop 최적화 기법이다. 분기 예측 실패를 줄이고, loop body 안에 데이터 의존성이 없는 경우 동시 실행 기회를 늘리며, 명령어 파이프라인 스케줄링에도 유리하다[[4]](<https://zhuanlan.zhihu.com/p/94846767#ref_4>).

## **3. 멀티스레드 병렬 최적화**

**vectorize**
    
    
     produce C {
      for (x.outer, 0, 32) {
        for (y.outer, 0, 32) {
          for (x.inner, 0, 32) {
            for (y.inner, 0, 32) {
              C[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = (A[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] + B[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)])
            }
          }
        }
      }
    }
    
    ---------cutting line---------
    produce C {
      for (x.outer, 0, 32) {
        for (y.outer, 0, 32) {
          for (x.inner, 0, 32) {
            C[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] = (A[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] + B[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)])
          }
        }
      }
    }
    

> `vectorize`는 iter 방향의 loop iteration을 ramp로 치환하여, SIMD 명령으로 데이터를 일괄 계산하도록 한다. 데이터 size가 상수이고 분할된 iter가 2의 거듭제곱(SIMD가 한 번에 처리 가능한 수량)일 때에만 치환이 일어나며, 그렇지 않으면 `vectorize`는 효과가 없다. SIMD 계산 장치에서 자주 쓰이는 schedule이다.

**bind**
    
    
     produce B {
      B[0] = 0f
      for (k.outer, 0, 32) {
        for (k.inner, 0, 32) {
          B[0] = (B[0] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 32
      // attr [reduce_temp0] storage_scope = "local"
      allocate reduce_temp0[float32 * 1]
      // attr [iter_var(threadIdx.x, , threadIdx.x)] thread_extent = 32
      // attr [comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f])] reduce_scope = reinterpret((uint64)0)
      tvm_thread_allreduce((uint32)1, A[((blockIdx.x*32) + threadIdx.x)], (bool)1, reduce_temp0, blockIdx.x, threadIdx.x)
      B[0] = reduce_temp0[0]
    }
    

> `bind`는 iter를 block 또는 thread의 index에 바인딩하여, loop의 작업을 스레드에 분배함으로써 병렬 계산을 구현한다. CUDA 백엔드에서 가장 핵심적인 부분이다.

**parallel**
    
    
     produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (l, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + l)])
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        parallel (l, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + l)])
        }
      }
    }
    

> `parallel`은 지정된 iter의 for loop를 parallel 연산으로 치환하여, GPU 이외의 CPU 등 장치에서 병렬 실행을 가능하게 한다.

## 4. 그 외 schedule

**pragma**
    
    
     produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (l.outer, 0, 256) {
          for (l.inner, 0, 4) {
            B[i] = (B[i] + A[(((i*1024) + (l.outer*4)) + l.inner)])
          }
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (l.outer, 0, 256) {
          B[i] = (B[i] + A[((i*1024) + (l.outer*4))])
          B[i] = (B[i] + A[(((i*1024) + (l.outer*4)) + 1)])
          B[i] = (B[i] + A[(((i*1024) + (l.outer*4)) + 2)])
          B[i] = (B[i] + A[(((i*1024) + (l.outer*4)) + 3)])
        }
      }
    }
    

> `pragma`는 컴파일러 주석을 추가하는 데 사용되며, 컴파일러가 pragma의 요구에 따라 unroll, vectorize 등 schedule 기능을 수행하도록 한다. 사실 새로운 최적화 규칙은 모두 일종의 pragma로 볼 수 있으며, directive라고도 불린다[[5]](<https://zhuanlan.zhihu.com/p/94846767#ref_5>).

**prefetch**
    
    
     produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    
    ---------cutting line---------
    produce B {
      for (i, 0, 1024) {
        B[i] = 0f
        for (k, 0, 1024) {
          for (prefetch.A.1, 0, 1) {
            for (prefetch.A.0, 0, 1) {
              prefetch(tvm_address_of(A[(((k*1024) + i) + 1024)]), 0, 3, 1)
            }
          }
          B[i] = (B[i] + A[((i*1024) + k)])
        }
      }
    }
    

> `prefetch`는 데이터의 공간 locality를 활용하여, 직전 iter의 계산과 이후 iter의 메모리 접근을 overlap시킴으로써 메모리 접근과 계산의 병렬도를 높이고 실행 시간을 줄인다. 본질적으로는 소프트웨어 파이프라이닝 개념이며, 하드웨어 prefetch가 아니다.

**tensorize**
    
    
     produce C {
      for (i, 0, 1024) {
        for (j.outer, 0, 32) {
          for (j.inner, 0, 16) {
            C[(((i*512) + (j.outer*16)) + j.inner)] = 0f
            for (k, 0, 64) {
              C[(((i*512) + (j.outer*16)) + j.inner)] = (C[(((i*512) + (j.outer*16)) + j.inner)] + (A[((i*64) + k)]*B[(((j.outer*1024) + (j.inner*64)) + k)]))
            }
          }
        }
      }
    }
    
    ---------cutting line---------
    produce C {
      for (i, 0, 1024) {
        for (j.outer, 0, 32) {
          gemv_update(tvm_access_ptr(type_annotation(), C, ((i*512) + (j.outer*16)), 16, 2), tvm_access_ptr(type_annotation(), A, (i*64), 64, 1), tvm_access_ptr(type_annotation(), B, (j.outer*1024), 1024, 1), 16, 64, 64)
        }
      }
    }
    

> `tensorize`는 계산을 하나의 덩어리로 묶어 `tensor_intrin` 함수로 컴파일한다. 자주 쓰이는 계산들은 이미 잘 만들어진 built-in schedule이 마련되어 있는 경우가 많은데, `tensorize`를 통해 이러한 내장 intrinsic을 직접 호출할 수 있다. 이는 컴퓨터 과학에서 intrinsic이 본래 의미하던 바이기도 하다[[6]](<https://zhuanlan.zhihu.com/p/94846767#ref_6>).

**rfactor(tensor, axis, factor_axis=0)**
    
    
     produce B {
      B[0] = 0f
      for (k.outer, 0, 32) {
        for (k.inner, 0, 32) {
          B[0] = (B[0] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    
    ---------cutting line---------
    // attr [B.rf] storage_scope = "global"
    allocate B.rf[float32 * 32]
    produce B.rf {
      for (k.inner, 0, 32) {
        B.rf[k.inner] = 0f
        for (k.outer, 0, 32) {
          B.rf[k.inner] = (B.rf[k.inner] + A[((k.outer*32) + k.inner)])
        }
      }
    }
    produce B {
      B[0] = 0f
      for (k.inner.v, 0, 32) {
        B[0] = (B[0] + B.rf[k.inner.v])
      }
    }
    

> `rfactor`는 원래의 tensor에 대해 axis 방향으로, factor_axis 간격을 두고 reduction 연산을 수행한다.

**set_store_predicate**
    
    
     // attr [B.rf] storage_scope = "global"
    allocate B.rf[float32 * 1]
    produce B {
      B[0] = 0f
      for (k.inner.v, 0, 16) {
        produce B.rf {
          B.rf[0] = 0f
          for (k.outer, 0, 64) {
            B.rf[0] = (B.rf[0] + A[((k.outer*16) + k.inner.v)])
          }
        }
        B[0] = (B[0] + B.rf[0])
      }
    }
    
    ---------cutting line---------
    // attr [B.rf] storage_scope = "global"
    allocate B.rf[float32 * 1]
    produce B {
      B[0] = 0f
      for (k.inner.v, 0, 16) {
        produce B.rf {
          B.rf[0] = 0f
          for (k.outer, 0, 64) {
            B.rf[0] = (B.rf[0] + A[((k.outer*16) + k.inner.v)])
          }
        }
        if ((threadIdx.x == 0)) {
          B[0] = (B[0] + B.rf[0])
        }
      }
    }
    

> `set_store_predicate`는 store의 조건을 설정한다. 멀티스레드 schedule에서 write 연산 간 충돌을 예방하는 데 적합하다.

**create_group(outputs, inputs, include_inputs=False)**
    
    
     // attr [D] storage_scope = "global"
    allocate D[float32 * 1048576]
    // attr [F] storage_scope = "global"
    allocate F[float32 * 1024]
    produce D {
      for (i, 0, 1024) {
        for (j, 0, 1024) {
          D[((i*1024) + j)] = (A[((i*1024) + j)] + B[((i*1024) + j)])
        }
      }
    }
    produce E {
      for (i, 0, 1024) {
        for (j, 0, 1024) {
          E[((i*1024) + j)] = (D[((i*1024) + j)] + B[((i*1024) + j)])
        }
      }
    }
    produce F {
      for (i, 0, 1024) {
        F[i] = 0f
        for (k, 0, 1024) {
          F[i] = (F[i] + E[((i*1024) + k)])
        }
      }
    }
    
    ---------cutting line---------
    // attr [F] storage_scope = "global"
    allocate F[float32 * 1024]
    // attr [D] storage_scope = "global"
    allocate D[float32 * 1]
    produce F {
      for (i, 0, 1024) {
        F[i] = 0f
        for (k, 0, 1024) {
          produce D {
            D[0] = (A[((i*1024) + k)] + B[((i*1024) + k)])
          }
          produce E {
            E[((i*1024) + k)] = (D[0] + B[((i*1024) + k)])
          }
          F[i] = (F[i] + E[((i*1024) + k)])
        }
      }
    }
    

> `create_group`은 inputs에서 outputs까지의 모든 stage에 대해 group을 만든다. group은 본질적으로 하나의 가상 stage이며, 이 가상 stage를 조작함으로써 group 안의 모든 stage를 한꺼번에 다룰 수 있다. 이 예시에서는 `compute_at`을 사용해 group 안의 D와 E를 함께 지정된 연산에 부착했다.

  


감사의 말: 베이징대학 고에너지 효율 컴퓨팅 응용 센터의 리즈신 학생이 예시 코드 작성과 글 정리를 도와주었다.

## 참고문헌

  1. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_1_0>)<http://people.csail.mit.edu/jrk/halide-pldi13.pdf>
  2. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_2_0>)<https://devblogs.nvidia.com/using-shared-memory-cuda-cc/>
  3. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_3_0>)<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model>
  4. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_4_0>)<https://en.wikipedia.org/wiki/Loop_unrolling>
  5. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_5_0>)<https://en.wikipedia.org/wiki/Directive_(programming)>
  6. [^](<https://zhuanlan.zhihu.com/p/94846767#ref_6_0>)<https://en.wikipedia.org/wiki/Intrinsic_function>

