#define NOMINMAX            
#include <iostream>
#include <vector>
#include <H5Cpp.h>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <limits>
#include <chrono>          
#include <opencv2/opencv.hpp>

struct ClusteringResult {
    std::vector<int> assignments;
    std::vector<float> centroids;
    double inertia;
    int iterations_performed;
    bool converged;
};

std::vector<float> init_centroids(const std::vector<float>& data, size_t num_samples, size_t num_features, int K) {
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

float squared_euclidean_distance(const std::vector<float>& data, const std::vector<float>& centroids, size_t sample_idx, size_t centroid_idx, size_t num_features) {
    size_t sample_offset = sample_idx * num_features;
    size_t centroid_offset = centroid_idx * num_features;
    float sum_sq = 0.0f;
    for (size_t j = 0; j < num_features; ++j) {
        float diff = data[sample_offset + j] - centroids[centroid_offset + j];
        sum_sq += diff * diff;
    }
    return sum_sq;
}

void recalculate_centroids(const std::vector<float>& data, const std::vector<int>& assignments, std::vector<float>& centroids, size_t num_samples, size_t num_features, int K) {
    std::vector<double> sums(K * num_features, 0.0f);
    std::vector<int> counts(K, 0);

    for (size_t i = 0; i < num_samples; ++i) {
        int k = assignments[i];
        counts[k]++;

        size_t sample_offset = i * num_features;
        size_t cluster_offset = k * num_features;

        for (size_t j = 0; j < num_features; ++j) {
            sums[cluster_offset + j] += data[sample_offset + j];
        }
    }

    for (int k = 0; k < K; ++k) {
        size_t cluster_offset = k * num_features;

        if (counts[k] > 0) {
            for (size_t j = 0; j < num_features; ++j) {
                centroids[cluster_offset + j] = sums[cluster_offset + j] / counts[k];
            }
        }
    }
}

double calculate_inertia(const std::vector<float>& data, const std::vector<float>& centroids, const std::vector<int>& assignments, size_t num_samples, size_t num_features) {
    double inertia = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        int k = assignments[i];
        inertia += squared_euclidean_distance(data, centroids, i, k, num_features);
    }
    return inertia;
}

cv::Mat vectorToMat(const std::vector<float>& data, size_t offset) {
    cv::Mat img(28, 28, CV_8UC1);
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float val = data[offset + r * 28 + c] * 255.0f;
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

        std::vector<float> data(num_samples * num_features);
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

        for (size_t i = 0; i < data.size(); ++i) {
            data[i] /= 255.0f;
        }

        auto t_start = std::chrono::steady_clock::now();

        std::vector<float> centroids = init_centroids(data, num_samples, num_features, K);
        std::vector<float> dists(K, -1.0f);

        ClusteringResult result;
        result.assignments = std::vector<int>(num_samples, -1);
        result.centroids = centroids;
        result.converged = false;
        result.inertia = 0.0;
        result.iterations_performed = 0;

        int max_iter = 100;
        double epsilon = 1.0;
        double prev_inertia = std::numeric_limits<double>::max();

        while (!result.converged && result.iterations_performed < max_iter) {
            for (size_t i = 0; i < num_samples; ++i) {
                for (int j = 0; j < K; ++j) {
                    dists[j] = squared_euclidean_distance(data, result.centroids, i, j, num_features);
                }
                auto min_it = std::min_element(dists.begin(), dists.end());
                int min_index = std::distance(dists.begin(), min_it);
                result.assignments[i] = min_index;
                result.inertia+= squared_euclidean_distance(data, centroids, i, min_index, num_features);
            }

            recalculate_centroids(data, result.assignments, result.centroids, num_samples, num_features, K);

            double delta_inertia = std::abs(prev_inertia - result.inertia);
            if (delta_inertia < epsilon) {
                result.converged = true;
            }

            prev_inertia = result.inertia;
            result.iterations_performed++;

            std::cout << "Iterazione " << result.iterations_performed << " - Inerzia: " << result.inertia << std::endl;
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "Finito dopo " << result.iterations_performed << " iterazioni con inerzia finale di " << result.inertia << std::endl;
        std::cout << "Tempo di esecuzione CPU: " << elapsed << " s ("
                  << elapsed / result.iterations_performed << " s per iterazione)" << std::endl;

        if (num_features == 784) displayCentroids(result.centroids, K);   // [SIFT] la griglia 28x28 vale solo per immagini a 784 feature

    } catch (H5::Exception& e) {
        e.printErrorStack();
        return -1;
    }

    return 0;
}
