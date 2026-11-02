# Parallel K-Means
 
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
| Fashion-MNIST | 10 | 42.343 | 0.43354 | ~97.7× |
| SIFT | 10 | - | 2.38144 | - |
| SIFT | 256 | 2368.88 | 51.956 | ~45.6× |
 
*Times are averages in seconds.*
 
The speedup grows with the number of clusters on SIFT, reaching about **46×** at K = 256.
 
### Profiling
 
Profiling the parallel version with `nsys` shows that the **assignment kernel takes about 96% of GPU time**. It is the least optimized kernel: each thread (one per data element) computes the distance to *all* centroids. This scales poorly with the number of clusters. It works well for small K (the Fashion-MNIST runs and the first SIFT run) and poorly for large K (the last SIFT run).
 
More benchmarks will follow, with different block sizes, to find the best value for each GPU architecture.
 
### Centroids on Fashion-MNIST
 
| Initial centroids (random) | Centroids found by the algorithm |
|:---:|:---:|
| <img width="400" alt="Initial centroids on Fashion-MNIST" src="https://github.com/user-attachments/assets/bbb1a13d-5abb-496c-bd84-04eb24fc2bc0" /> | <img width="400" alt="Final centroids on Fashion-MNIST" src="https://github.com/user-attachments/assets/bc4271b2-24ec-416c-83f1-22802a9b3212" /> |
 
## Roadmap
 
- [ ] New assignment kernel that makes better use of GPU resources and scales better with the number of clusters
- [ ] First ANN-search implementation based on the clustered dataset
- [ ] Block-size benchmarks across different GPU architectures
- [ ] Develop parallel multi-threaded approach to test against CUDA-based parallelization using OpenMP
