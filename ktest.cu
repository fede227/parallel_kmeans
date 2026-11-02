#include <iostream>
#include <vector>
#include <H5Cpp.h>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
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

__global__ void assignment(uint8_t* data, float* centroids, uint8_t*d_labels, size_t num_samples, size_t num_features, size_t num_centroids, float* d_inertia){
    int idx=blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx<num_samples){
    float min_dist=INFINITY;
    uint8_t min_index;
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

__global__ void partial_sums(uin8_t* data, uint8_t * sums, int*d_labels, size_t num_samples, size_t num_features, size_t num_centroids, size_t chunks){
    int idx=(blockIdx.x * blockDim.x + threadIdx.x) ;
    int
    int i0=c*chunks;
    int imax=min(num_features, chunks+i0);


}

struct ClusteringResult {
    std::vector<int> assignments;
    std::vector<float> centroids;
    double inertia;
    int iterations_performed;
    bool converged;
};

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
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

std::vector<float> init_centroids(const std::vector<uint8_t>& data, size_t num_samples, size_t num_features, int K, double * d_inertia) {
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
        H5::H5File file("fashion-mnist-784-euclidean.hdf5", H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("train");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        size_t num_samples = dims[0];
        size_t num_features = dims[1];
        int K = 10;

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
        float*  d_partial;   int*    d_counts;
        float* d_inertia;

        CHECK(cudaMalloc(&d_centroids, K * num_features * sizeof(float)));
        CHECK(cudaMalloc(&d_labels,    num_samples * sizeof(uint8_t)));
        CHECK(cudaMalloc(&d_counts,    K * sizeof(int)));
        CHECK(cudaMalloc(&d_inertia,   sizeof(double)*num_features));


        CHECK(cudaMemcpy(d_centroids, centroids.data(),
                         centroids.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_labels, 0xFF, num_samples * sizeof(int)));

        float* gpuRef = (float *)malloc(nBytes);
        int max_iter = 100;
        double epsilon = 1.0;
        float prev_inertia = std::numeric_limits<float>::max();
        double iStart, iElapse;
        int block_dimx = 32;
        dim3 block(block_dimx);
        dim3 grid((num_samples + block.x - 1) / block.x);
        iStart = cpuSecond();
        float inertia=0.0f;



        assignment<<<grid, block>>>(d_data, d_centroids, d_labels, num_samples, num_features, K, d_inertia);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(gpuRef, d_inertia, sizeof(double)*num_features, cudaMemcpyDeviceToHost));

        for(int i=0; i<num_samples; i++){
            inertia+=gpuRef[i];
        }



        iElapse = cpuSecond() - iStart;
        cout<<"Tempo di esecuzione GPU: "<<iElapse<<endl;
        printf("fatto!\n");



        displayCentroids(result.centroids, K);
        CHECK(cudaFree(d_data));
    } catch (H5::Exception& e) {
        e.printErrorStack();
        return -1;
    }

    return 0;
}

