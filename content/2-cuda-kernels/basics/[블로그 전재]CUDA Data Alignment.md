> 블로그 출처: https://leimao.github.io/blog/CUDA-Data-Alignment/ , Lei Mao의 글이며 저자의 전재 허가를 받았다.

# CUDA Data Alignment

## 소개

최고의 성능을 얻기 위해서는 C++의 data alignment 요구 사항(https://leimao.github.io/blog/CPP-Data-Alignment/)과 비슷하게 CUDA에서도 data alignment가 필요하다.

이 블로그 글에서는 CUDA의 data alignment 요구 사항을 빠르게 논의한다.

## 전역 메모리의 coalesced access

전역 메모리는 device memory에 존재하며, device memory는 32, 64 또는 128 byte memory transaction으로 접근된다. 이러한 memory transaction은 반드시 naturally aligned되어야 한다. 즉, 크기에 맞춰 정렬된 32, 64 또는 128 byte device memory segment, 다시 말해 시작 주소가 해당 크기의 배수인 segment만 memory transaction으로 read 또는 write될 수 있다.

하나의 warp가 전역 메모리에 접근하는 instruction을 실행할 때, 각 thread가 접근하는 word size와 memory address가 thread 사이에 어떻게 분포하는지에 따라 warp 안 thread들의 memory access를 하나 이상의 memory transaction으로 coalesce한다. 일반적으로 필요한 transaction이 많을수록 thread가 접근하는 word 이외에 전송되는 unused word도 많아지고, 이에 따라 instruction throughput이 낮아진다.

compute capability 6.0 이상인 device의 경우 요구 사항은 쉽게 요약할 수 있다. warp 안 thread들의 concurrent access는 여러 transaction으로 coalesce되며, transaction 수는 warp 안 모든 thread를 처리하는 데 필요한 32 byte transaction 수와 같다.

전역 메모리에 존재하는 variable address나 driver 또는 runtime API의 memory allocation routine, 예를 들어 `cudaMalloc` 또는 `cudaMallocPitch`가 반환하는 모든 address는 항상 최소 256 byte에 align되어 있다.

## 예시

예를 들어 32개 thread로 구성된 warp에서 각 thread가 4 byte data를 read하려 하고, warp 안 모든 thread의 4 byte data(128 byte data)가 서로 인접해 있으며 32 byte aligned, 즉 첫 번째 4 byte data의 address가 32의 배수라면 memory access는 coalesced되고 GPU는 $\frac{4\times 32}{32}=4$번의 32 byte memory transaction을 수행한다. GPU가 가능한 한 적은 transaction을 수행하므로 최대 memory transaction throughput이 달성된다.

128 byte data가 memory에서 32 byte aligned가 아니라 예를 들어 4 byte aligned라면, 추가 32 byte memory transaction이 하나 더 필요하다. 따라서 memory access throughput은 최대 이론 throughput의 $\frac{4}{5}=80%$가 된다. 데이터 segment 5개를 가로지르기 때문이다.

또한 모든 thread의 4 byte data가 서로 인접하지 않고 memory에 sparse하게 흩어져 있다면, 최대 32번의 32 byte memory transaction이 필요할 수 있고, throughput은 최대 이론 throughput의 $\frac{4}{32}=12.5%$에 불과하다.

## 크기와 alignment 요구 사항

전역 메모리 instruction은 크기가 1, 2, 4, 8 또는 16 byte인 word의 read 또는 write를 지원한다. data type의 크기가 1, 2, 4, 8 또는 16 byte이고 data가 naturally aligned, 즉 address가 해당 크기의 배수일 때만, 전역 메모리에 있는 data에 대한 모든 access(variable 또는 pointer를 통한 access)는 단일 전역 메모리 instruction으로 compile된다.

이 크기와 alignment 요구 사항을 만족하지 않으면 access는 interleaved access pattern을 가진 여러 instruction으로 compile되며, 이러한 instruction은 완전한 coalescing을 방해한다. 따라서 전역 메모리에 존재하는 data에는 이 요구 사항을 만족하는 type을 사용하는 것이 좋다.

naturally aligned되지 않은 8 byte 또는 16 byte word를 read하면 잘못된 결과, 몇 word만큼 어긋난 결과가 생성된다. 따라서 이러한 type의 어떤 값이나 값 배열의 시작 주소 alignment를 유지하는 데 특별히 주의해야 한다.

그러므로 크기가 1, 2, 4, 8 또는 16 byte인 word를 사용하는 것은 때때로 간단하다. 위에서 말했듯 memory allocation CUDA API가 반환하는 시작 memory address는 항상 최소 256 byte aligned이고, 이는 이미 1, 2, 4, 8 또는 16 byte aligned이기 때문이다. 따라서 word sequence, 예를 들어 numerical array, matrix, tensor를 할당된 memory에 안전하게 저장할 수 있으며, 8 byte 또는 16 byte 크기 word를 read해 잘못된 결과가 생길 것을 걱정할 필요가 없다. 최적의 memory access throughput을 달성하려면 kernel 구현에서 coalesced memory access도 naturally aligned되도록 특별히 주의해야 한다.

하지만 word size가 1, 2, 4, 8 또는 16 byte가 아니라면 어떻게 해야 할까? memory allocation CUDA API가 반환하는 시작 memory address는 naturally aligned임을 보장하지 않으므로 memory access throughput이 크게 손상된다. 보통 이 문제를 처리하는 방법은 두 가지이다.

- alignment 요구 사항이 이미 지정되고 충족되는 built-in vector type(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#built-in-vector-types)을 사용한다.
- GCC에서 structure data alignment를 강제하기 위해 사용하는 compiler specifier `alignas`와 비슷하게, NVCC에서는 compiler specifier `__align__`을 사용해 structure data alignment를 강제한다.

```c++
struct __align__(4) int8_3_4_t
{
    int8_t x;
    int8_t y;
    int8_t z;
};

struct __align__(16) float3_16_t
{
    float x;
    float y;
    float z;
};
```

## 결론

항상 word size를 1, 2, 4, 8 또는 16 byte로 만들고, data를 naturally aligned되게 하라.

할당된 memory가 동일한 type word의 sequence에만 사용된다면, word를 read해 잘못된 결과가 생성되는 경우는 거의 발생하지 않는다. memory allocation CUDA API가 반환하는 시작 memory address가 항상 최소 256 byte aligned이기 때문이다. 하지만 여러 type의 word sequence를 위한 큰 memory block을, padding이 있든 없든, 할당한다면 어떤 word나 word sequence의 시작 address alignment를 유지하는 데 특별히 주의해야 한다. 그렇지 않으면 잘못된 결과가 생성될 수 있기 때문이다. 특히 naturally aligned되지 않은 8 byte 또는 16 byte word에서 그렇다.

## 참고 문헌

- C++ Data Alignment(https://leimao.github.io/blog/CPP-Data-Alignment/)
- CUDA Device Memory Access(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#device-memory-accesses)
- Coalesced Access to Global Memory(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
