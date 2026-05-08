// Multi-locus allele-frequency estimation on CPU and GPU (CUDA)
//
// Genotype encoding per sample per locus:
//   0   -> aa  (0 copies of allele A)
//   1   -> Aa  (1 copy  of allele A)
//   2   -> AA  (2 copies of allele A)
//   255 -> missing
//
// Memory layout:
//   genotypes[locus * n_samples + sample]
//   i.e. locus-major, contiguous samples within each locus.
//
// Build:
//   nvcc -O3 -std=c++17 allele_frequency_multilocus_cuda.cu -o allele_freq
//
// Jetson build example (native on Jetson):
//   nvcc -O3 -std=c++17 -arch=sm_72 allele_frequency_multilocus_cuda.cu -o allele_freq
//
// Run examples:
//   ./allele_freq
//   ./allele_freq 10000 100000
//
// Notes:
// - CPU code is the correctness reference.
// - GPU code launches one block per locus and reduces across samples.
// - Benchmark includes H2D, kernel, D2H timing, and CPU timing.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(err__));                                      \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

static constexpr uint8_t MISSING = 255;
static constexpr int THREADS_PER_BLOCK = 256;

struct BenchmarkResult {
    double cpu_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_gpu_ms = 0.0;
};

struct ValidationResult {
    double max_abs_diff = 0.0;
    size_t mismatches = 0;
};

bool checked_genotype_count(int n_loci, int n_samples, size_t& total_genotypes) {
    if (n_loci <= 0 || n_samples <= 0) {
        return false;
    }

    const size_t loci = static_cast<size_t>(n_loci);
    const size_t samples = static_cast<size_t>(n_samples);
    if (loci > std::numeric_limits<size_t>::max() / samples) {
        return false;
    }

    total_genotypes = loci * samples;
    return true;
}

int parse_positive_int_arg(const char* value, const char* name) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive 32-bit integer");
    }
    return static_cast<int>(parsed);
}

// -----------------------------
// Data generation
// -----------------------------

void fill_random_genotypes(std::vector<uint8_t>& genotypes,
                           int n_loci,
                           int n_samples,
                           double missing_rate = 0.02,
                           uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> p_dist(0.01, 0.99);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    for (int locus = 0; locus < n_loci; ++locus) {
        double p = p_dist(rng);                 // allele A frequency for simulation
        double p0 = (1.0 - p) * (1.0 - p);      // aa
        double p1 = 2.0 * p * (1.0 - p);        // Aa
        // p2 = p*p for AA

        size_t base = static_cast<size_t>(locus) * static_cast<size_t>(n_samples);
        for (int sample = 0; sample < n_samples; ++sample) {
            double u = u01(rng);
            if (u < missing_rate) {
                genotypes[base + sample] = MISSING;
                continue;
            }

            double g = u01(rng);
            if (g < p0) {
                genotypes[base + sample] = 0;
            } else if (g < p0 + p1) {
                genotypes[base + sample] = 1;
            } else {
                genotypes[base + sample] = 2;
            }
        }
    }
}

// -----------------------------
// CPU reference implementation
// -----------------------------

void allele_frequency_cpu(const uint8_t* genotypes,
                          int n_loci,
                          int n_samples,
                          std::vector<uint64_t>& allele_counts,
                          std::vector<uint64_t>& called_counts,
                          std::vector<double>& freqs) {
    allele_counts.assign(n_loci, 0);
    called_counts.assign(n_loci, 0);
    freqs.assign(n_loci, -1.0);

    for (int locus = 0; locus < n_loci; ++locus) {
        uint64_t allele_sum = 0;
        uint64_t called = 0;

        size_t base = static_cast<size_t>(locus) * static_cast<size_t>(n_samples);
        for (int sample = 0; sample < n_samples; ++sample) {
            uint8_t g = genotypes[base + sample];
            if (g != MISSING) {
                allele_sum += static_cast<uint64_t>(g);
                called += 1;
            }
        }

        allele_counts[locus] = allele_sum;
        called_counts[locus] = called;
        freqs[locus] = (called > 0) ? static_cast<double>(allele_sum) / (2.0 * static_cast<double>(called))
                                    : -1.0;
    }
}

// -----------------------------
// CUDA kernel
// One block computes one locus.
// Threads stride over samples, accumulate local partial sums,
// then reduce in shared memory.
// -----------------------------

__global__ void allele_frequency_kernel_per_locus(const uint8_t* __restrict__ genotypes,
                                                  int n_samples,
                                                  uint64_t* __restrict__ allele_counts,
                                                  uint64_t* __restrict__ called_counts) {
    int locus = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ uint64_t s_alleles[THREADS_PER_BLOCK];
    __shared__ uint64_t s_called[THREADS_PER_BLOCK];

    uint64_t local_alleles = 0;
    uint64_t local_called = 0;

    size_t base = static_cast<size_t>(locus) * static_cast<size_t>(n_samples);

    for (int sample = tid; sample < n_samples; sample += blockDim.x) {
        uint8_t g = genotypes[base + sample];
        if (g != MISSING) {
            local_alleles += static_cast<uint64_t>(g);
            local_called += 1;
        }
    }

    s_alleles[tid] = local_alleles;
    s_called[tid] = local_called;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_alleles[tid] += s_alleles[tid + stride];
            s_called[tid] += s_called[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        allele_counts[locus] = s_alleles[0];
        called_counts[locus] = s_called[0];
    }
}

void counts_to_freqs(const std::vector<uint64_t>& allele_counts,
                     const std::vector<uint64_t>& called_counts,
                     std::vector<double>& freqs) {
    size_t n = allele_counts.size();
    freqs.resize(n);
    for (size_t i = 0; i < n; ++i) {
        freqs[i] = (called_counts[i] > 0)
                       ? static_cast<double>(allele_counts[i]) / (2.0 * static_cast<double>(called_counts[i]))
                       : -1.0;
    }
}

void allele_frequency_gpu(const std::vector<uint8_t>& h_genotypes,
                          int n_loci,
                          int n_samples,
                          std::vector<uint64_t>& h_allele_counts,
                          std::vector<uint64_t>& h_called_counts,
                          std::vector<double>& h_freqs,
                          BenchmarkResult& bench) {
    uint8_t* d_genotypes = nullptr;
    uint64_t* d_allele_counts = nullptr;
    uint64_t* d_called_counts = nullptr;

    size_t total_genotypes = 0;
    if (!checked_genotype_count(n_loci, n_samples, total_genotypes) || total_genotypes != h_genotypes.size()) {
        throw std::invalid_argument("genotype dimensions do not match the host genotype vector size");
    }

    const size_t genotype_bytes = total_genotypes * sizeof(uint8_t);
    const size_t count_bytes = static_cast<size_t>(n_loci) * sizeof(uint64_t);

    h_allele_counts.resize(n_loci);
    h_called_counts.resize(n_loci);

    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_genotypes), genotype_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_allele_counts), count_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_called_counts), count_bytes));

    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_genotypes, h_genotypes.data(), genotype_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_allele_counts, 0, count_bytes));
    CUDA_CHECK(cudaMemset(d_called_counts, 0, count_bytes));
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    allele_frequency_kernel_per_locus<<<n_loci, THREADS_PER_BLOCK>>>(
        d_genotypes, n_samples, d_allele_counts, d_called_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(h_allele_counts.data(), d_allele_counts, count_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_called_counts.data(), d_called_counts, count_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));

    float h2d_ms_f = 0.0f;
    float kernel_ms_f = 0.0f;
    float d2h_ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms_f, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_f, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms_f, start_d2h, stop_d2h));

    bench.h2d_ms = static_cast<double>(h2d_ms_f);
    bench.kernel_ms = static_cast<double>(kernel_ms_f);
    bench.d2h_ms = static_cast<double>(d2h_ms_f);
    bench.total_gpu_ms = bench.h2d_ms + bench.kernel_ms + bench.d2h_ms;

    counts_to_freqs(h_allele_counts, h_called_counts, h_freqs);

    CUDA_CHECK(cudaFree(d_genotypes));
    CUDA_CHECK(cudaFree(d_allele_counts));
    CUDA_CHECK(cudaFree(d_called_counts));

    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));
}

// -----------------------------
// Validation and reporting
// -----------------------------

ValidationResult compare_freqs(const std::vector<double>& a,
                               const std::vector<double>& b,
                               double tol = 1e-12) {
    ValidationResult out;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs(a[i] - b[i]);
        out.max_abs_diff = std::max(out.max_abs_diff, diff);
        if (diff > tol) {
            out.mismatches += 1;
        }
    }
    out.mismatches += std::max(a.size(), b.size()) - n;
    return out;
}

void print_preview(const std::vector<uint64_t>& allele_counts,
                   const std::vector<uint64_t>& called_counts,
                   const std::vector<double>& freqs,
                   int max_rows = 8) {
    int rows = std::min<int>(max_rows, static_cast<int>(freqs.size()));
    std::cout << "\nPreview (first " << rows << " loci):\n";
    std::cout << "locus\tallele_count\tcalled_count\tfreq\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << i << '\t' << allele_counts[i] << '\t' << called_counts[i] << '\t' << freqs[i] << '\n';
    }
}

void print_device_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No CUDA-capable device was found. Run this program on a CUDA system such as NVIDIA Jetson.\n";
        std::exit(EXIT_FAILURE);
    }

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "CUDA device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
}

// -----------------------------
// Main benchmark harness
// -----------------------------

int main(int argc, char** argv) {
    int n_loci = 4096;
    int n_samples = 65536;

    try {
        if (argc >= 2) {
            n_loci = parse_positive_int_arg(argv[1], "n_loci");
        }
        if (argc >= 3) {
            n_samples = parse_positive_int_arg(argv[2], "n_samples");
        }
        if (argc > 3) {
            throw std::invalid_argument("usage: ./allele_freq [n_loci] [n_samples]");
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    size_t total_genotypes = 0;
    if (!checked_genotype_count(n_loci, n_samples, total_genotypes)) {
        std::cerr << "n_loci * n_samples is not representable on this platform\n";
        return EXIT_FAILURE;
    }

    const double genotype_mb = static_cast<double>(total_genotypes) / (1024.0 * 1024.0);

    std::cout << "Multi-locus allele-frequency estimation\n";
    std::cout << "n_loci    = " << n_loci << "\n";
    std::cout << "n_samples = " << n_samples << "\n";
    std::cout << "genotypes = " << total_genotypes << " entries (~" << genotype_mb << " MiB as uint8)\n\n";

    print_device_info();

    std::vector<uint8_t> genotypes(total_genotypes);
    fill_random_genotypes(genotypes, n_loci, n_samples, 0.02, 42);

    std::vector<uint64_t> cpu_allele_counts, cpu_called_counts;
    std::vector<double> cpu_freqs;

    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    allele_frequency_cpu(genotypes.data(), n_loci, n_samples,
                         cpu_allele_counts, cpu_called_counts, cpu_freqs);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();

    BenchmarkResult bench;
    bench.cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    std::vector<uint64_t> gpu_allele_counts, gpu_called_counts;
    std::vector<double> gpu_freqs;
    allele_frequency_gpu(genotypes, n_loci, n_samples,
                         gpu_allele_counts, gpu_called_counts, gpu_freqs, bench);

    ValidationResult val = compare_freqs(cpu_freqs, gpu_freqs);

    print_preview(cpu_allele_counts, cpu_called_counts, cpu_freqs);

    std::cout << "\nValidation:\n";
    std::cout << "max_abs_diff = " << val.max_abs_diff << "\n";
    std::cout << "mismatches   = " << val.mismatches << "\n";

    std::cout << "\nTiming (ms):\n";
    std::cout << "CPU           = " << bench.cpu_ms << "\n";
    std::cout << "GPU H2D       = " << bench.h2d_ms << "\n";
    std::cout << "GPU kernel    = " << bench.kernel_ms << "\n";
    std::cout << "GPU D2H       = " << bench.d2h_ms << "\n";
    std::cout << "GPU total     = " << bench.total_gpu_ms << "\n";

    if (bench.total_gpu_ms > 0.0) {
        std::cout << "Speedup (CPU / GPU total) = " << (bench.cpu_ms / bench.total_gpu_ms) << "x\n";
    }

    return (val.mismatches == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
