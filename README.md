# DMHTD
1st International Workshop on Novel Data Mining Methods for the Analysis of High-Throughput Biological Data (DM-HT-D)

## CUDA allele-frequency benchmark

This repository includes a CUDA/C++ benchmark for multi-locus allele-frequency estimation in `allele_frequency_multilocus_cuda.cu`. It compares a CPU reference implementation with a CUDA implementation that launches one block per locus and reduces over samples inside the block.

Build on a CUDA-capable system:

```bash
nvcc -O3 -std=c++17 allele_frequency_multilocus_cuda.cu -o allele_freq
```

On NVIDIA Jetson devices, build natively with the architecture that matches the board. For example, Jetson AGX Xavier/NX can use:

```bash
nvcc -O3 -std=c++17 -arch=sm_72 allele_frequency_multilocus_cuda.cu -o allele_freq
```

Run a small smoke test first, then scale up according to available unified memory:

```bash
./allele_freq 128 4096
./allele_freq 4096 65536
```
