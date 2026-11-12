#pragma once

#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "cpu_traverse.h"
#include "texture_sample.h"

#define BLOCK_SIZE 256

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states);

CUDA_GLOBAL void ray_query_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    const TextureDeviceView *texture_views,
    const MaterialDeviceView *materials,
    int n_textures,
    HitResults o_hits,
    int n_rays,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
);

CUDA_GLOBAL void point_query_entry(
    const glm::vec3 *i_points,
    const BVHDataPointers i_dp,
    SDFHitResults o_out,
    int n_points
);

CUDA_GLOBAL void ray_query_all_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    HitResults o_hits,
    uint32_t *o_n_hits,
    int max_hits_per_ray,
    int n_rays,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
);

struct GPUTraverser {
    const Mesh &mesh;

    thrust::device_vector<glm::vec3> vertices;
    thrust::device_vector<glm::vec3> vertex_normals;
    thrust::device_vector<glm::vec2> uvs;
    thrust::device_vector<Face> faces;
    thrust::device_vector<BVHNode> nodes;
    thrust::device_vector<MaterialDeviceView> materials;
    thrust::device_vector<TextureDeviceView> texture_views;
    std::vector<thrust::device_vector<uint8_t>> texture_pixels;  // Owned pixel data

    GPUTraverser(const Mesh &mesh) : mesh(mesh) {
        if (!mesh.bvh) {
            std::cerr << "Error: Mesh does not have BVH built. Use build_bvh=True when loading." << std::endl;
            exit(1);
        }

        vertices = mesh.vertices;
        if (!mesh.vertex_normals.empty()) {
            vertex_normals = mesh.vertex_normals;
        }
        faces = mesh.faces;
        nodes = mesh.bvh->nodes;
        if (!mesh.uvs.empty()) {
            uvs = mesh.uvs;
        }

        // Upload materials
        if (!mesh.materials.empty()) {
            std::vector<MaterialDeviceView> mat_views(mesh.materials.size());
            for (size_t i = 0; i < mesh.materials.size(); i++) {
                mat_views[i].base_color = mesh.materials[i].base_color;
                mat_views[i].texture_id = mesh.materials[i].texture_id;
            }
            materials = mat_views;
        }

        // Upload textures
        if (!mesh.textures.empty()) {
            texture_pixels.resize(mesh.textures.size());

            // First, upload ALL pixel data
            for (size_t i = 0; i < mesh.textures.size(); i++) {
                const Texture& tex = mesh.textures[i];
                texture_pixels[i] = tex.pixels;
            }

            // Then, create views with the stable device pointers
            std::vector<TextureDeviceView> tex_views(mesh.textures.size());
            for (size_t i = 0; i < mesh.textures.size(); i++) {
                const Texture& tex = mesh.textures[i];
                tex_views[i].pixels = thrust::raw_pointer_cast(texture_pixels[i].data());
                tex_views[i].width = tex.width;
                tex_views[i].height = tex.height;
                tex_views[i].channels = tex.channels;
            }

            texture_views = tex_views;
        }
    }

    BVHDataPointers get_data_pointers() const {
        return {
            vertices.data().get(),
            vertex_normals.empty() ? nullptr : vertex_normals.data().get(),
            faces.data().get(),
            nodes.data().get(),
            uvs.empty() ? nullptr : uvs.data().get(),
            materials.empty() ? nullptr : reinterpret_cast<const Material*>(materials.data().get()),
            nullptr  // textures pointer will be passed separately via texture_views
        };
    }

    void ray_query(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_dists,
        uint32_t *o_prim_idxs,
        glm::vec3 *o_normals,
        glm::vec2 *o_uvs,
        glm::vec3 *o_colors,
        glm::vec3 *o_barycentrics,
        int n_rays,
        bool allow_negative = false,
        bool allow_backward = true,
        bool allow_forward = true,
        cudaStream_t stream = 0
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_dists, o_prim_idxs, o_normals, o_uvs, o_colors, o_barycentrics};

        ray_query_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            rays,
            get_data_pointers(),
            texture_views.empty() ? nullptr : texture_views.data().get(),
            materials.empty() ? nullptr : materials.data().get(),
            (int)texture_views.size(),
            hits,
            n_rays,
            allow_negative,
            allow_backward,
            allow_forward
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in ray_query: %s\n", cudaGetErrorString(err));
        }
    }

    void ray_query_all(
        glm::vec3 *i_ray_origs,
        glm::vec3 *i_ray_vecs,
        bool *o_masks,
        float *o_dists,
        uint32_t *o_prim_idxs,
        uint32_t *o_n_hits,
        int max_hits_per_ray,
        int n_rays,
        bool allow_negative = false,
        bool allow_backward = true,
        bool allow_forward = true
    ) {
        Rays rays = {i_ray_origs, i_ray_vecs};
        HitResults hits = {o_masks, o_dists, o_prim_idxs, nullptr, nullptr};

        ray_query_all_entry<<<(n_rays + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            rays,
            get_data_pointers(),
            hits,
            o_n_hits,
            max_hits_per_ray,
            n_rays,
            allow_negative,
            allow_backward,
            allow_forward
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in ray_query_all: %s\n", cudaGetErrorString(err));
        }
    }    

    void point_query(
        const glm::vec3 *i_points,
        float *o_t,
        glm::vec3 *o_closests,
        glm::vec3 *o_barycentricses,
        uint32_t *o_face_idxs,
        int n_points
    ) {
        SDFHitResults out = {o_t, o_closests, o_barycentricses, o_face_idxs};

        point_query_entry<<<(n_points + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            i_points,
            get_data_pointers(),
            out,
            n_points
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in point_query: %s\n", cudaGetErrorString(err));
        }
    }
};

// floats per training-segment record written by trace_shell_segments:
// enter(3) exit(3) dir(3) outer_normal(3) gt_normal(3) gt_color(3) gt_t(1) gt_mask(1)
#define SEG_REC_FLOATS 20

void trace_shell_segments(
    const GPUTraverser &outer,
    const GPUTraverser &inner,
    const GPUTraverser &fine,
    const glm::vec3 *i_origs,
    const glm::vec3 *i_dirs,
    float *o_records,
    bool *o_valid,
    int n_rays,
    int max_segments,
    float shift_eps,
    float no_exit_t,
    cudaStream_t stream = 0
);
