#pragma once

#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "mesh.h"
#include "utils.h"
#include "gpu_traverse.cuh"

#define CUDA_CHECK_ERRORS() do {                                          \
    cudaError_t err = cudaGetLastError();                                 \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA kernel failed : " << cudaGetErrorString(err);  \
        exit(-1);                                                         \
    }                                                                     \
} while (0)

CUDA_GLOBAL void mesh_sample_surface_uniform_entry(
    const glm::vec3 *i_vertices,
    const Face *i_faces,
    const float *i_face_weights_prefix_sum,
    int n_faces,
    curandState *io_rand_states,
    glm::vec3 *o_out_points,
    glm::vec3 *o_out_barycentrics,
    uint32_t *out_face_idxs,    
    int n_points
);

struct GPUMeshSampler {
    thrust::device_vector<glm::vec3> vertices_;
    thrust::device_vector<Face> faces_;
    thrust::device_vector<float> face_weights_prefix_sum_;

    int max_points_;
    thrust::device_vector<curandState> rand_states_;

    GPUMeshSampler(const Mesh &mesh, int max_points) {
        vertices_ = mesh.vertices;
        faces_ = mesh.faces;

        max_points_ = max_points;
        rand_states_.resize(max_points_);
        init_rand_state_entry<<<(max_points + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(rand_states_.data().get(), max_points);

        fill_face_weights_cpu(mesh);
    }

    void fill_face_weights_cpu(const Mesh &mesh) {
        std::vector<float> h_face_weights;

        h_face_weights.resize(mesh.faces.size());

        for (int i = 0; i < mesh.faces.size(); i++) {
            Face face = mesh.faces[i];
            glm::vec3 v0 = mesh.vertices[face.v1];
            glm::vec3 v1 = mesh.vertices[face.v2];
            glm::vec3 v2 = mesh.vertices[face.v3];

            float area = glm::length(glm::cross(v1 - v0, v2 - v0)) * 0.5f;
            h_face_weights[i] = area;
        }

        float total_area = 0.0f;
        for (int i = 0; i < h_face_weights.size(); i++) {
            total_area += h_face_weights[i];
        }

        for (int i = 0; i < h_face_weights.size(); i++) {
            h_face_weights[i] /= total_area;
        }

        for (int i = 1; i < h_face_weights.size(); i++) {
            h_face_weights[i] += h_face_weights[i - 1];
        }

        face_weights_prefix_sum_ = h_face_weights;
    }

    void sample(glm::vec3 *out_points, glm::vec3 *out_barycentrics, uint32_t *out_face_idxs, int n_points, cudaStream_t stream = 0) {
        if (n_points > max_points_) {
            std::cerr << "Error: n_points exceeds max_points_" << std::endl;
            exit(1);
        }

        mesh_sample_surface_uniform_entry<<<(n_points + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            vertices_.data().get(),
            faces_.data().get(),
            face_weights_prefix_sum_.data().get(),
            faces_.size(),
            rand_states_.data().get(),
            out_points,
            out_barycentrics,
            out_face_idxs,
            n_points
        );
        CUDA_CHECK_ERRORS();
    }
};