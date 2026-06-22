> 블로그 출처: https://leimao.github.io/blog/CPU-Cache-False-Sharing/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다.

# CPU Cache False Sharing(CPU Cache False Sharing)

## 소개

CPU clock frequency와 core 수 외에도 CPU cache는 CPU performance의 또 다른 핵심 속성이다. 예를 들어 Intel server급 Xeon CPU는 보통 flagship Intel desktop급 Core i9 CPU보다 최대 clock frequency가 낮지만, Intel Xeon CPU는 보통 더 많은 core와 더 큰 cache를 가진다. 따라서 같은 세대의 Intel Xeon CPU는 multithreaded application에서 Intel Core i9 CPU보다 항상 더 좋은 performance을 보이며, 물론 Intel Xeon CPU의 가격도 훨씬 높다.

일반적으로 CPU가 memory의 data를 어떻게 cache하는지 제어할 수는 없지만, CPU는 특정 heuristic rule에 따라 memory를 cache한다. 사용자는 프로그램이 CPU cache-friendly한 방식으로 만들어지도록 해야 한다. CPU cache 동작이 프로그램의 예상 동작과 일치하면 프로그램은 좋은 performance을 낼 수 있다.

이 블로그 글에서는，Scott Meyers의 CPU cache(https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf) false sharing 예제를 빌려 implementation 세부 사항이 CPU cache와 프로그램 performance에 얼마나 중요한지 보여준다.

## CPU Cache

CPU specification은 Linux의 `lscpu` 명령으로 얻을 수 있다. 이 경우 내 Intel i9-9900K CPU는 CPU core마다 256 KB L1 data cache, 256 KB L1 instruction cache, 2 MB L2 cache를 가지며, 모든 CPU core가 공유하는 16 MB L3 cache를 가진다.

```shell
$ lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  16
  On-line CPU(s) list:   0-15
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
    CPU family:          6
    Model:               158
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
    Stepping:            12
    CPU max MHz:         5000.0000
    CPU min MHz:         800.0000
    BogoMIPS:            7200.00
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mc
                         a cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss 
                         ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art
                          arch_perfmon pebs bts rep_good nopl xtopology nonstop_
                         tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cp
                         l vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid ss
                         e4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes 
                         xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_f
                         ault epb invpcid_single ssbd ibrs ibpb stibp tpr_shadow
                          vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust 
                         bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap cl
                         flushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm
                          ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp
                          md_clear flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   256 KiB (8 instances)
  L1i:                   256 KiB (8 instances)
  L2:                    2 MiB (8 instances)
  L3:                    16 MiB (1 instance)
NUMA:                    
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-15
Vulnerabilities:         
  Itlb multihit:         KVM: Mitigation: VMX disabled
  L1tf:                  Not affected
  Mds:                   Mitigation; Clear CPU buffers; SMT vulnerable
  Meltdown:              Not affected
  Mmio stale data:       Mitigation; Clear CPU buffers; SMT vulnerable
  Retbleed:              Mitigation; IBRS
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl
                          and seccomp
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer
                          sanitization
  Spectre v2:            Mitigation; IBRS, IBPB conditional, RSB filling
  Srbds:                 Mitigation; Microcode
  Tsx async abort:       Mitigation; TSX disabled
```

L1 cache는 L2 cache보다 훨씬 빠르고, L2 cache는 L3 cache보다 훨씬 빠르며, L3 cache는 main memory보다 훨씬 빠르다. Cache access performance은 CPU processor까지의 물리적 거리와 관련이 있다.

## CPU Cache 규칙

CPU cache는 몇 가지 규칙을 따른다. 이 규칙을 이해하면 더 나은 코드를 작성하는 데 도움이 된다.

- cache는 cache line으로 구성되며, 각 line은 main memory에서 온 여러 인접 word를 저장한다.
- 어떤 word가 이미 cache에 있으면 그 word는 main memory가 아니라 cache에서 읽힌다. 따라서 read 속도가 훨씬 빠르다. Cache 안의 word를 overwrite하면 결국 main memory의 해당 word도 overwrite된다.
- 어떤 word가 cache에 없으면 그 word를 포함하는 전체 cache line을 main memory에서 읽는다. 따라서 read 속도가 느리다.
- hardware는 cache line을 speculative prefetch한다. 즉 CPU가 아직 현재 cache line의 iterative read instruction을 읽고 있을 때 다음 cache line이 이미 cache에 준비되어 있을 수 있다.
- 같은 cache line이 여러 cache(서로 다른 CPU core에 속함)에 cache되어 있을 때, 어떤 cache line 하나가 한 thread에 의해 overwrite되면 모든 cache line이 invalidate되고, overwrite가 main memory에 반영된 뒤 다시 main memory에서 읽어야 한다.

## False Sharing

다음 예제는 Scott Meyers의 CppCon 2014 강연 "CPU Caches and Why You Care"(https://youtu.be/WDIkqP4JbkE)에 나오는 false sharing pseudocode 예제(https://www.aristeia.com/TalkNotes/codedive-CPUCachesHandouts.pdf)의 C++ implementation이다. 같은 cache line이 거의 매 iteration마다 여러 cache에서 invalidate되어 multithreaded application performance이 심각하게 영향을 받는 방식을 보여준다. 이 문제를 해결하는 방법도 간단하고 우아하다.

```c++
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

// generic matrix class template，fordemonstrate CPU cache behavior
template <typename T>
class Matrix
{
public:
    // constructor：create m-row n-column matrix，data is stored contiguously in memory
    Matrix(size_t m, size_t n) : m_num_rows{m}, m_num_cols{n}, m_data(m * n){};
    
    // overload()operator，supportmat(i,j)accessstyle
    T& operator()(size_t i, size_t j) noexcept
    {
        return m_data[i * m_num_cols + j];  // row-major storage：row*cols + col
    }
    T operator()(size_t i, size_t j) const noexcept
    {
        return m_data[i * m_num_cols + j];
    }
    
    // overload[]operator，supportmat[i][j]accessstyle
    T* operator[](size_t i) noexcept { return &m_data[i * m_num_cols]; }
    T const* operator[](size_t i) const noexcept
    {
        return &m_data[i * m_num_cols];
    }
    
    // get matrix dimensions
    size_t get_num_rows() const noexcept { return m_num_rows; };
    size_t get_num_cols() const noexcept { return m_num_cols; };
    
    // get underlying data pointer，fordirect memory access
    T* data() noexcept { return m_data.data(); }
    T const* data() const noexcept { return m_data.data(); }

private:
    size_t m_num_rows;
    size_t m_num_cols;
    std::vector<T> m_data;  // data is stored contiguously in memory，this is important for cache friendliness
};

// performancemeasurefunctiontemplate：measurefunctionexecutetime
template <class T>
float measure_performance(std::function<T(void)> bound_function,
                          int num_repeats = 100, int num_warmups = 100)
{
    // warmup stage：warm up CPU cache，avoid first-run overhead affecting measurements
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function();
    }

    // formal measurement stage
    std::chrono::steady_clock::time_point time_start{
        std::chrono::steady_clock::now()};
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function();
    }
    std::chrono::steady_clock::time_point time_end{
        std::chrono::steady_clock::now()};

    // calculateaverage latency
    auto const time_elapsed{
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end -
                                                              time_start)
            .count()};
    float const latency{time_elapsed / static_cast<float>(num_repeats)};

    return latency;
}

// randomly initialize matrix：fill random integer values
void random_initialize_matrix(Matrix<int>& mat, unsigned int seed)
{
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    std::default_random_engine e(seed);
    std::uniform_int_distribution<int> uniform_dist(-1024, 1024);
    for (size_t i{0}; i < num_rows; ++i)
    {
        for (size_t j{0}; j < num_cols; ++j)
        {
            mat[i][j] = uniform_dist(e);
        }
    }
}

// row-major traversal：count odd values in matrix
// this access pattern is CPU-cache friendly，because data is contiguous in memory
size_t count_odd_values_row_major(Matrix<int> const& mat)
{
    size_t num_odd_values{0};
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    // outer loop traverses rows，inner loop traverses columns - access contiguous memory
    for (size_t i{0}; i < num_rows; ++i)
    {
        for (size_t j{0}; j < num_cols; ++j)
        {
            if (mat[i][j] % 2 != 0)
            {
                ++num_odd_values;
            }
        }
    }
    return num_odd_values;
}

// column-major traversal：count odd values in matrix
// this access pattern is not CPU-cache friendly，because cross-row access causes cache misses
size_t count_odd_values_column_major(Matrix<int> const& mat)
{
    size_t num_odd_values{0};
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    // outer loop traverses columns，inner loop traverses rows - access non-contiguous memory
    for (size_t j{0}; j < num_cols; ++j)
    {
        for (size_t i{0}; i < num_rows; ++i)
        {
            if (mat[i][j] % 2 != 0)
            {
                ++num_odd_values;
            }
        }
    }
    return num_odd_values;
}

// multi-thread version（non-scalable）：show false sharing issue
// multiple threads simultaneously write shared results array，causing cache line invalidation
size_t
multi_thread_count_odd_values_row_major_non_scalable(Matrix<int> const& mat,
                                                     size_t num_threads)
{
    std::vector<std::thread> workers{};
    workers.reserve(num_threads);
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    size_t const num_elements{num_rows * num_cols};
    size_t const trunk_size{(num_elements + num_threads - 1) / num_threads};

    // shared result array - false sharing occurs here！
    std::vector<size_t> results(num_threads, 0);
    for (size_t i{0}; i < num_threads; ++i)
    {
        workers.emplace_back(
            [&, i]()
            {
                size_t const start_pos{i * trunk_size};
                size_t const end_pos{
                    std::min((i + 1) * trunk_size, num_elements)};
                for (size_t j{start_pos}; j < end_pos; ++j)
                {
                    if (mat.data()[j] % 2 != 0)
                    {
                        // false sharing issue：
                        // resultsarrayaccessed by multiple threads。contiguous memory block containing this entry
                        // is cached in the CPU for thread reads。multiple threads cache the same content in multiple caches。
                        // however，writing the array in main memory invalidates all caches with the same content。
                        // CPU must read the updated entry from main memory and recache the array。
                        // this significantly reduces performance。
                        ++results[i];  // each write may invalidate cache of other threads
                    }
                }
            });
    }
    for (int i{0}; i < num_threads; ++i)
    {
        workers[i].join();
    }

    // sum results from all threads
    size_t num_odd_values{0};
    for (int i{0}; i < num_threads; ++i)
    {
        num_odd_values += results[i];
    }

    return num_odd_values;
}

// multi-thread version（scalable）：solve false sharing issue
// use local variables to avoid frequent writes to shared memory，write result once at the end
size_t multi_thread_count_odd_values_row_major_scalable(Matrix<int> const& mat,
                                                        size_t num_threads)
{
    std::vector<std::thread> workers{};
    workers.reserve(num_threads);
    size_t const num_rows{mat.get_num_rows()};
    size_t const num_cols{mat.get_num_cols()};
    size_t const num_elements{num_rows * num_cols};
    size_t const trunk_size{(num_elements + num_threads - 1) / num_threads};

    std::vector<size_t> results(num_threads, 0);
    for (size_t i{0}; i < num_threads; ++i)
    {
        workers.emplace_back(
            [&, i]()
            {
                // key optimization：use local variable to accumulate result
                size_t count = 0;  // local variable，does not cause false sharing
                size_t const start_pos{i * trunk_size};
                size_t const end_pos{
                    std::min((i + 1) * trunk_size, num_elements)};
                for (size_t j{start_pos}; j < end_pos; ++j)
                {
                    if (mat.data()[j] % 2 != 0)
                    {
                        // modify only local variable，avoid cache invalidation
                        ++count;
                    }
                }
                // write shared array once at the end，greatly reduce false sharing
                results[i] = count;
            });
    }
    for (int i{0}; i < num_threads; ++i)
    {
        workers[i].join();
    }

    // sum results from all threads
    size_t num_odd_values{0};
    for (int i{0}; i < num_threads; ++i)
    {
        num_odd_values += results[i];
    }

    return num_odd_values;
}

int main()
{
    unsigned int const seed{0U};
    int const num_repeats{100};    // test repeat count
    int const num_warmups{100};    // warmup count

    size_t const num_threads{8};   // thread count

    float latency{0};

    // createtestmatrix：1000x2000 = 2,000,000 integers
    Matrix<int> mat(1000, 2000);
    random_initialize_matrix(mat, seed);

    // verify correctness of all implementations
    assert(count_odd_values_row_major(mat) ==
           count_odd_values_column_major(mat));

    assert(
        count_odd_values_row_major(mat) ==
        multi_thread_count_odd_values_row_major_non_scalable(mat, num_threads));

    assert(count_odd_values_row_major(mat) ==
           multi_thread_count_odd_values_row_major_scalable(mat, num_threads));

    // create function object for performance test
    std::function<size_t(void)> const function_1{
        std::bind(count_odd_values_row_major, mat)};
    std::function<size_t(void)> const function_2{
        std::bind(count_odd_values_column_major, mat)};
    std::function<size_t(void)> const function_3{
        std::bind(multi_thread_count_odd_values_row_major_non_scalable, mat,
                  num_threads)};
    std::function<size_t(void)> const function_4{std::bind(
        multi_thread_count_odd_values_row_major_scalable, mat, num_threads)};

    // performancetest1：single-threadrow-major traversal（cache friendly）
    latency = measure_performance(function_1, num_repeats, num_warmups);
    std::cout << "Single-Thread Row-Major Traversal" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    // performancetest2：single-threadcolumn-major traversal（cache unfriendly）
    latency = measure_performance(function_2, num_repeats, num_warmups);
    std::cout << "Single-Thread Column-Major Traversal" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    // performancetest3：multi-thread version（with false sharing issue）
    latency = measure_performance(function_3, num_repeats, num_warmups);
    std::cout << num_threads << "-Thread Row-Major Non-Scalable Traversal"
              << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;

    // performancetest4：multi-thread version（solve false sharing issue）
    latency = measure_performance(function_4, num_repeats, num_warmups);
    std::cout << num_threads << "-Thread Row-Major Scalable Traversal"
              << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
}
```

latency 측정 결과에서 false sharing 때문에 8-thread traversal performance이 매우 나쁘고 single-thread traversal보다 거의 낫지 않다는 것을 볼 수 있다. 거의 모든 false sharing(마지막 output array write 제외)을 제거하면 8-thread traversal performance이 theoretical value까지 크게 향상된다.

```shell
$ g++ false_sharing.cpp -o false_sharing -lpthread -std=c++14
$ ./false_sharing 
Single-Thread Row-Major Traversal
Latency: 16.840 ms
Single-Thread Column-Major Traversal
Latency: 20.210 ms
8-Thread Row-Major Non-Scalable Traversal
Latency: 16.520 ms
8-Thread Row-Major Scalable Traversal
Latency: 2.740 ms
```

## CPU Cache VS GPUcache

CPU cache와 GPU cache 사이에는 많은 유사점이 있다.

예를 들어 GPU의 global memory memory IO를 개선하려면 memory access가 coalesced이기를 원한다. 그래야 cache line의 모든 entry를 GPU thread가 사용할 수 있다.

또한 많은 thread가 iteration 중 global memory를 읽고 쓰면 GPU에서도 false sharing이 나타날 수 있다. GPU의 false sharing 문제를 해결하려면 CPU의 해결책과 비슷하게 local variable 또는 shared memory를 사용해 intermediate value를 저장하고, algorithm 끝에서 local variable 또는 shared memory에서 global memory로 한 번만 write한다.

## 참고 자료

- CPU Caches and Why You Care - Scott Meyers(https://www.aristeia.
com/TalkNotes/codedive-CPUCachesHandouts.pdf)
