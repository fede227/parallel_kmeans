# Parallel K-Means
 <p align="left">
  <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA RTX" />
  <img src="https://img.shields.io/badge/Algorithm-K--Means-orange?style=for-the-badge" alt="K-Means" />
</p>
CUDA parallelization of **Lloyd's algorithm**, built as a fast clustering step for **approximate nearest-neighbour (ANN) search**.
 
> **Status:** early-stage research code. The implementation is not yet refined, but it is sufficient for a first round of benchmarks showing that the parallel approach is worth pursuing.
 
## Overview
 
The goal is a time-efficient clustering of a dataset that can later be used to run ANN searches over the clusters. For this purpose the clustering does not need to be as precise as possible, especially when extra precision costs significant computation time.
 
Tests currently run on two datasets:
 
- **Fashion-MNIST** (60,000 samples, 784 features)
- **SIFT** (1,000,000 samples, 128 features)
## Repository structure
 
| File | Description |
|---|---|
| `parallel_kmeans.cu` | CUDA C++ implementation of the parallel algorithm. Block and grid sizes are currently tuned for an NVIDIA RTX 5060 GPU. |
| `sequential_kmeans.cpp` | Sequential SINGLE THREADED C++ implementation, used as the benchmark baseline and as ground truth for checking results. |
 
## Benchmarks
 
Each run performs **100 iterations** of assignment and centroid update. Inertia is not computed, so the timings measure only the algorithm itself.
 
**Setup:** AMD Ryzen 7 260 (CPU) vs. NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB (GPU). Block size is 128 for all three kernels (assignment, partial sums, averaging).
 
| Dataset | Clusters (K) | CPU time (s) | GPU time (s) | Speedup |
|---|---:|---:|---:|---:|
| Fashion-MNIST | 10 | 42.343 | 0.126 | ~336× |
| SIFT | 256 | 2368.88 | 8.537 | ~277.5× |
 
*Times are averages in seconds.*
 
The speedup grows with the number of clusters on SIFT, reaching about **277×** at K = 256.
 
### Profiling
 
Originally profiling the parallel version with `nsys` showed that the **assignment kernel takes about 96% of GPU time**. It is the least optimized kernel: each thread (one per data element) computes the distance to *all* centroids. This scales poorly with the number of clusters. It works well for small K (the Fashion-MNIST runs and the first SIFT run) and poorly for large K (the last SIFT run). To solve this issue a transposed version of the dataset matrix was used in this kernel to allow for coalescent reads from memory (since matrixes in CUDA are stored in the row-major format) this resulted in an important speedup as can be seen from the comparison below. 
| Dataset | Clusters (K) | GPU time Not Transposed (s) | GPU time Transposed (s) | Speedup |
|---|---:|---:|---:|---:|
| Fashion-MNIST | 10 | 0.433 | **0.126** | ~3.4× |
| SIFT | 256 | 51.956 | **8.537** | ~6.1× |
 
More benchmarks will follow, with different block sizes, to find the best value for each GPU architecture.
 
### Centroids on Fashion-MNIST
 
| Initial centroids (random) | Centroids found by the algorithm |
|:---:|:---:|
| <img width="400" alt="Initial centroids on Fashion-MNIST" src="https://github.com/user-attachments/assets/bbb1a13d-5abb-496c-bd84-04eb24fc2bc0" /> | <img width="400" alt="Final centroids on Fashion-MNIST" src="https://github.com/user-attachments/assets/bc4271b2-24ec-416c-83f1-22802a9b3212" /> |
 
## Roadmap
 
- [ ] Computing inertia and running more tests
- [ ] First ANN-search implementation based on the clustered dataset
- [ ] Block-size benchmarks across different GPU architectures
- [ ] Develop parallel multi-threaded approach to test against CUDA-based parallelization using OpenMP
