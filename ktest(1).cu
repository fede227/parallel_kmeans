#define NOMINMAX            // [WIN] evita che le macro min/max di windows.h rompano std::numeric_limits<>::max()
#include <iostream>
#include <vector>
#include <H5Cpp.h>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <limits>
#include <cstdint>          // [WIN] uint8_t non è garantito transitivamente con MSVC
#include <opencv2/opencv.hpp>
#include <chrono>           // [WIN] sostituisce <sys/time.h>, che non esiste su Windows
#include <cuda_runtime.h>
using namespace std;

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                          \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
        exit(1);                                                               \
    }                                                                          \
}

constexpr int KC=256;

__global__ void assignment(uint8_t* data, float* centroids, uint8_t*d_labels, size_t num_samples, size_t num_features, size_t num_centroids, float* d_inertia){
    int idx=blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx<num_samples){
    float min_dist=INFINITY;
    uint8_t min_index=0;
    for(int i=0; i<num_centroids; i++){
        float dist=0.0f;
        for(int j=0; j<num_features; j++){
            float diff = data[idx*num_features+j] - centroids[i*num_features + j];
            dist+=diff*diff;
        }
        if(dist<min_dist){
            min_dist=dist;
            min_index=i;
        }
    }
    d_labels[idx]=min_index;
    d_inertia[idx]=min_dist;
    }
}

__global__ void partial_sums(uint8_t * data, int * partials, uint8_t*d_labels, size_t num_samples, size_t num_features, size_t num_centroids, size_t chunk_size, size_t num_chunks){
    int idx=(blockIdx.x * blockDim.x + threadIdx.x) ;
    int chunk_idx=blockIdx.y; 
    if(idx>=num_features || chunk_idx>=num_chunks){
        return;
    }
    int i0=chunk_idx*chunk_size; 
    int imax=min(i0 + chunk_size, num_samples);
    int sums[KC];
      #pragma unroll
    for(int i=0; i<KC; i++){
        sums[i]=0;
    }
    for(int i=i0; i<imax; i++){
        int   lab = d_labels[i];
        int   v   = data[i * num_features + idx];
        #pragma unroll
        for (int k = 0; k < KC; ++k)
            sums[k] += (lab == k) ? v : 0;
    }
    #pragma unroll
    for(int i=0; i<KC; i++){
        partials[(idx*num_centroids + i)*num_chunks + chunk_idx]=sums[i];
    }

}

__global__ void averaging(float* centroids, int* partials, int* counts, size_t num_samples, size_t num_features, size_t num_centroids, size_t chunk_size, size_t num_chunks){
    int idx=(blockIdx.x * blockDim.x + threadIdx.x) ;
    int k=blockIdx.y;
    if(k>=num_centroids || idx>=num_features || counts[k]==0){
        return;
    }
    int sum=0;
    int i0=idx*num_centroids*num_chunks+k*num_chunks;
    int imax=i0+num_chunks;
    for(int i=i0; i<imax;i++){
        sum+=partials[i];
    }

    centroids[k*num_features + idx]=(float) sum/counts[k];
}

struct ClusteringResult {
    std::vector<int> assignments;
    std::vector<float> centroids;
    double inertia;
    int iterations_performed;
    bool converged;
};

// [WIN] gettimeofday non esiste su Windows: stessa semantica (secondi come double) con std::chrono
double cpuSecond() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

cv::Mat vectorToMat(const std::vector<float>& data, size_t offset) {
    cv::Mat img(28, 28, CV_8UC1);
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float val = data[offset + r * 28 + c] ;
            img.at<uchar>(r, c) = static_cast<uchar>(std::clamp(val, 0.0f, 255.0f));
        }
    }
    return img;
}

void displayCentroids(const std::vector<float>& centroids, int K) {
    int tile_size = 28;
    cv::Mat grid(tile_size * 2, tile_size * 5, CV_8UC1, cv::Scalar(0));

    for (int k = 0; k < K; ++k) {
        int row = k / 5;
        int col = k % 5;
        cv::Mat centroid_img = vectorToMat(centroids, k * 784);
        centroid_img.copyTo(grid(cv::Rect(col * tile_size, row * tile_size, tile_size, tile_size)));
    }

    cv::Mat resized_grid;
    cv::resize(grid, resized_grid, cv::Size(140 * 5, 56 * 5), 0, 0, cv::INTER_NEAREST);

    cv::imshow("I 10 Centroidi Appresi dal K-Means (Fashion-MNIST)", resized_grid);
    cv::waitKey(0);
}

std::vector<float> init_centroids(const std::vector<uint8_t>& data, size_t num_samples, size_t num_features, int K) {
    std::vector<float> centroids(K * num_features);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, num_samples - 1);

    std::unordered_set<size_t> chosen_indices;

    for (int k = 0; k < K; ++k) {
        size_t rand_idx;
        do {
            rand_idx = dist(rng);
        } while (chosen_indices.count(rand_idx) > 0);

        chosen_indices.insert(rand_idx);
        size_t sample_offset = rand_idx * num_features;
        size_t centroid_offset = k * num_features;

        for (size_t j = 0; j < num_features; ++j) {
            centroids[centroid_offset + j] = data[sample_offset + j];
        }
    }

    return centroids;
}

int main() {
    try {
        H5::H5File file("sift-128-euclidean.hdf5", H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("train");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        size_t num_samples = dims[0];
        size_t num_features = dims[1];
        int K = 256;

        std::cout << "Immagini caricate: " << num_samples
        << " | Dimensione vettore: " << num_features << std::endl;

        std::vector<float> float_data(num_samples * num_features);
        dataset.read(float_data.data(), H5::PredType::NATIVE_FLOAT);

        std::vector<uint8_t> data(num_samples*num_features);
        for(size_t i=0; i<float_data.size(); i++){
            float v=float_data[i];
            if(v<0.0f || v>255.0f || v!=std::floor(v)){
                std::cerr<<"Valore fuori range"<<endl;
                return -1;
            }
            data[i]=static_cast<uint8_t>(v);
        }

        uint8_t * d_data;
        size_t nBytes=sizeof(uint8_t)*num_samples*num_features;
        CHECK(cudaMalloc((void **)&d_data, nBytes));
        CHECK(cudaMemcpy(d_data, data.data(), nBytes, cudaMemcpyHostToDevice));

        std::vector<float> centroids = init_centroids(data, num_samples, num_features, K);
        std::vector<float> dists(K, -1.0f);

        ClusteringResult result;
        result.assignments = std::vector<int>(num_samples, -1);
        result.centroids = centroids;
        result.converged = false;
        result.inertia = 0.0;
        result.iterations_performed = 0;

        float*  d_centroids; uint8_t*    d_labels;
        int*  d_partials;
         int*  d_counts;
        float* d_inertia;

        CHECK(cudaMalloc(&d_centroids, K * num_features * sizeof(float)));
        CHECK(cudaMalloc(&d_labels,    num_samples * sizeof(uint8_t)));
        CHECK(cudaMalloc(&d_counts,   K * sizeof(int)));
        CHECK(cudaMalloc(&d_inertia,   sizeof(float)*num_samples));
        int chunk_size=1024; // in origine 256
        int chunks=(num_samples + chunk_size - 1) / chunk_size;


        CHECK(cudaMalloc(&d_partials,   sizeof(int)*num_features*K*chunks));

        CHECK(cudaMemcpy(d_centroids, centroids.data(),
                         centroids.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_labels, 0xFF, num_samples * sizeof(uint8_t)));

        int max_iter = 100;
        double epsilon = 1.0;
        float prev_inertia = std::numeric_limits<float>::max();
        double iStart, iElapse;
        int block_dimx = 256; //in origine 32
        dim3 block(block_dimx);
        dim3 grid((num_samples + block.x - 1) / block.x);
        iStart = cpuSecond();
        int iter=0;
        while(iter<max_iter){


        assignment<<<grid, block>>>(d_data, d_centroids, d_labels, num_samples, num_features, K, d_inertia);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> h_inertia(num_samples);
        std::vector<int> counts(K, 0);
        std::vector<uint8_t> h_labels(num_samples);
        CHECK(cudaMemcpy(h_inertia.data(), d_inertia, num_samples * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_labels.data(), d_labels, num_samples * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        double inertia = 0.0;
        int pp=0;
        for (float v : h_inertia){
            inertia += v;
            counts[h_labels[pp]]++;
            pp++;

        }

        dim3 block2(128);
        dim3 grid2((num_features+block2.x-1)/block2.x, chunks);

        partial_sums<<<grid2, block2>>>(d_data, d_partials, d_labels, num_samples, num_features, K, chunk_size, chunks);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(d_counts, counts.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        dim3 block3(128);
        dim3 grid3((num_features+block3.x-1)/block3.x, K);
        averaging<<<grid3, block3>>>(d_centroids, d_partials, d_counts, num_samples, num_features, K, chunk_size, chunks);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iter++;
        }
        iElapse = cpuSecond() - iStart;
        cout<<"Tempo di esecuzione GPU: "<<iElapse<<endl;
        printf("fatto!\n");
         CHECK(cudaMemcpy(result.centroids.data(), d_centroids, K * num_features * sizeof(float), cudaMemcpyDeviceToHost));


        if (num_features == 784) displayCentroids(result.centroids, K);
        CHECK(cudaFree(d_data));
    } catch (H5::Exception& e) {
        e.printErrorStack();
        return -1;
    }

    return 0;
}
