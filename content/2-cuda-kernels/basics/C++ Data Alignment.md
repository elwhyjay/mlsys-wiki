> 블로그 출처: https://leimao.github.io/blog/CPP-Data-Alignment/ , Lei Mao의 글이며 저자의 전재 허가를 받았다.

# C++ Data Alignment

## 소개

Data alignment는 현대 컴퓨터 hardware 계산에서 중요한 특성이다. data가 naturally aligned되어 있을 때 CPU는 memory를 가장 효율적으로 read하고 write한다. 이는 보통 data의 memory address가 data size의 배수라는 뜻이다. 예를 들어 32 bit 아키텍처에서 data가 연속된 4 byte에 저장되어 있고 첫 번째 byte가 4 byte boundary에 위치한다면, 해당 data는 aligned되어 있을 수 있다.

성능 외에도 data alignment는 많은 programming language의 가정 조건이다. programming language가 가능한 한 data alignment 문제를 처리해 주지만, 일부 low-level programming language에서는 unaligned data access가 발생할 수 있고, 이러한 동작은 undefined behavior이다.

이 블로그 글에서는 aligned memory address와 aligned memory access를 포함한 data alignment, 그리고 C++에서 최대한 data alignment를 보장하는 방법을 빠르게 논의하고자 한다.

## Data Alignment

memory address $a$가 $n$의 배수일 때, 여기서 $n$은 2의 거듭제곱이다, 우리는 memory address $a$가 $n$ byte aligned라고 말한다. $m$ byte data와 $n$ byte aligned address가 있다고 가정하자. $m$이 $n$으로 나누어떨어지지 않으면, $m$ byte data는 $\lceil\frac{m+n-1}{n}\rceil \times n$ byte data로 padding된다.

$kn + 1, kn + 2, \cdots, (k + 1)n$ byte data에 대한 access는 모두 같은 latency를 갖는다. CPU는 매번 memory에서 $n$ byte data를 read하고, 이 data는 보통 CPU에 cache되기 때문이다. 즉, data가 $n$ byte aligned address에 저장되어 있고 그 저장 크기 $m$이 $n$의 배수가 아니라면, 일부 memory access bandwidth가 낭비된다.

access되는 data length가 $n$ byte이고 data address가 $n$ byte aligned일 때, 우리는 이러한 memory access가 aligned되었다고 말한다. memory access가 aligned되지 않았을 때는 unaligned되었다고 말한다. 정의상 single byte memory access는 항상 aligned되어 있다는 점에 유의하라. 이론적으로 $n$의 배수가 아닌 memory address에서 $n$ byte data에 access할 수 있지만, 이는 더 많은 memory access bandwidth를 낭비한다. 하지만 C와 C++ standard는 memory access가 aligned되어 있다고 가정하므로, unaligned address에 access하면 undefined behavior가 발생할 수 있다.

## Data Alignment 요구 사항

`alignof`는 특정 data type의 alignment 요구 사항을 확인하는 데 사용할 수 있다.

```c++
#include <cassert>

struct float4_4_t
{
    float data[4];
};

// float4_32_t type의 각 object는 32 byte boundary에 align된다.
// SIMD instruction에 유용할 수 있다.
struct alignas(32) float4_32_t
{
    float data[4];
};

// 같은 declaration의 다른 alignas보다 약한 유효한 non-zero alignment는 무시된다.
struct alignas(1) float4_1_t
{
    float data[4];
};

// object에 access하면 undefined behavior가 발생한다.
// 1 byte structure member alignment.
// size = 32, alignment = 1 byte이며, 이 structure member에는 padding이 없다.
// float에는 4 byte alignment가 필요하므로 이는 비정상적이다.
#pragma pack(push, 1)
struct alignas(1) float4_1_ub_t
{
    float data[4];
};
#pragma pack(pop)

int main()
{
    assert(alignof(float4_4_t) == 4);
    assert(alignof(float4_32_t) == 32);
    assert(alignof(float4_1_t) == 4);
    assert(alignof(float4_1_ub_t) == 1);

    assert(sizeof(float4_4_t) == 16);
    assert(sizeof(float4_32_t) == 32);
    assert(sizeof(float4_1_t) == 16);
    assert(sizeof(float4_1_ub_t) == 16);
}
```

## Memory Allocation

GNU 문서(https://www.gnu.org/software/libc/manual/html_node/Aligned-Memory-Blocks.html)에 따르면 GNU system에서 `malloc` 또는 `realloc`이 반환하는 block address는 항상 8의 배수이다. 64 bit system에서는 16의 배수이다. array의 기본 memory address alignment는 element의 alignment 요구 사항에 의해 결정된다.

할당된 static memory와 dynamic memory에는 custom data alignment를 사용할 수 있다. `alignas(T)`는 static array의 byte alignment를 지정하는 데 사용할 수 있고, `aligned_alloc`은 dynamic memory의 buffer byte alignment를 지정하는 데 사용할 수 있다.

```c++
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main()
{
    unsigned char buf1[sizeof(int) / sizeof(char)];
    std::cout << "默认 "
              << alignof(unsigned char[sizeof(int) / sizeof(char)]) << "字节"
              << " 对齐地址: " << static_cast<void*>(buf1) << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) % alignof(int) << std::endl;

    alignas(int) unsigned char buf2[sizeof(int) / sizeof(char)];
    std::cout << alignof(int)
              << "字节对齐地址: " << static_cast<void*>(buf2)
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) % alignof(int) << std::endl;

    void* p1 = malloc(sizeof(int));
    std::cout << "默认 "
              << "16字节"
              << " 对齐地址: " << p1 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 1024 << std::endl;
    free(p1);

    void* p2 = aligned_alloc(1024, sizeof(int));
    std::cout << "1024字节对齐地址: " << p2 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 1024 << std::endl;
    free(p2);
}
```

```shell
$ g++ alloc.cpp -o alloc -std=c++11
$ ./alloc
Default 1-byte aligned addr: 0x7ffd46d76304
0
0
4-byte aligned addr: 0x7ffd46d76300
0
0
Default 16-byte aligned addr: 0x559a6e1c42c0
0
704
1024-byte aligned addr: 0x559a6e1c4400
0
0
```

## Undefined Behavior

data alignment이 올바르지 않으면 static array 또는 dynamic buffer에 data를 write하는 일이 undefined behavior를 일으킬 수 있다. 예를 들어 `unsigned char buf[sizeof(T) / sizeof(char)]` 위에 type T의 object를 만들면, 특히 `reinterpret_cast`와 unaligned memory address increment를 사용할 때 read/write의 undefined behavior가 발생할 수 있다. `malloc`으로 할당한 dynamic buffer 위에 object T를 만드는 경우도 마찬가지이다. 하지만 `malloc`이 반환하는 address는 32 bit 아키텍처에서는 8 byte aligned이고 64 bit 아키텍처에서는 16 byte aligned이므로, 8 byte와 16 byte alignment는 거의 모든 data, 특히 primitive type에 의해 충족될 수 있다. 따라서 undefined behavior가 발생할 가능성은 크지 않다.

예를 들어 다음 data structure `Bar`는 `sizeof(Bar) == 6`과 `alignof(Bar) == 2`를 갖는다.

```c++
struct Bar
{
    char arr[3];    // 3 byte + padding byte 1개
    short s;        // 2 byte
};
```

alignment 요구 사항은 항상 data structure 안 각 member의 최대 alignment 요구 사항이다. 현대 컴퓨터에서는 이것이 반드시 2의 거듭제곱이어야 한다. data structure `Bar`의 경우 `alignof(char) == 1`이고 `alignof(short) == 2`이다. 따라서 `sizeof(Bar) == max(alignof(char), alignof(short)) == 2`이다.

memory 안의 `Bar` object가 2 byte aligned라면, 1 byte 요구 사항이 필요한 `char` type property에 대한 access는 자동으로 만족된다. padding byte가 존재하므로, 2 byte 요구 사항이 필요한 `short` type property에 대한 access도 자동으로 만족된다.

내 x86-64 아키텍처 컴퓨터에서 `malloc(sizeof(Bar))`를 사용할 수 있고, 반환되는 pointer는 16 byte aligned address가 될 것이다. 이는 `Bar` data structure의 2 byte alignment 요구 사항을 만족한다.

data alignment로 인한 undefined behavior가 발생하지 않도록 완전히 보장하려면, `T buf[N]`, `alignas(T) unsigned char buf[N * sizeof(T) / sizeof(char)]`, `aligned_alloc(alignas(T), N * sizeof(T))`를 사용해 memory를 할당한 다음 그 위에 type `T`의 object를 만들어야 한다. 이는 compiler가 data structure를 생성할 때 `sizeof(T)`가 반드시 `alignas(T)`의 배수여야 함을 의미하기도 한다. 그렇지 않으면 array의 두 번째 element부터 unaligned되기 시작할 수 있다.

## 결론

data alignment를 수동으로 보장하는 것은 오류가 생기기 쉽다. 따라서 대부분의 사용 사례에서는 dynamic memory allocation에 `new`, `delete`, `std::vector` 같은 high-level interface function과 STL container를 사용하고, 위험한 pointer type conversion을 줄여 type safety를 보장하려 해야 한다.

## 참고 문헌

- Data Alignment(https://www.songho.ca/misc/alignment/dataalign.html)
- alignas specifier(en.cppreference.com/w/cpp/language/alignas)
- std::aligned_alloc(https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)
