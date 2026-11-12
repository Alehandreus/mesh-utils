#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "cpu_traverse.h"
#include "utils.h"
#include "gpu_traverse.cuh"

#define EPS 1e-8

// GPU texture sampling implementation
CUDA_DEVICE glm::vec3 sampleTextureGPU(
    const TextureDeviceView& texture,
    float u, float v
) {
    // Handle empty texture
    if (!texture.pixels || texture.width == 0 || texture.height == 0) {
        return glm::vec3(1.0f);  // Default white
    }

    // Wrap UV coordinates to [0, 1)
    u = u - floorf(u);
    v = v - floorf(v);

    // Flip V axis (GLTF/OpenGL convention)
    v = 1.0f - v;

    // Calculate pixel coordinates
    float x = u * (texture.width - 1);
    float y = v * (texture.height - 1);

    // Get integer coordinates and interpolation factors
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, texture.width - 1);
    int y1 = min(y0 + 1, texture.height - 1);

    float tx = x - x0;
    float ty = y - y0;

    // Fetch pixel helper
    auto fetchPixel = [&](int xi, int yi) -> glm::vec3 {
        size_t idx = (yi * texture.width + xi) * 4;
        return glm::vec3(
            texture.pixels[idx + 0] / 255.0f,  // R
            texture.pixels[idx + 1] / 255.0f,  // G
            texture.pixels[idx + 2] / 255.0f   // B
        );
    };

    // Fetch 4 corner pixels
    glm::vec3 c00 = fetchPixel(x0, y0);
    glm::vec3 c10 = fetchPixel(x1, y0);
    glm::vec3 c01 = fetchPixel(x0, y1);
    glm::vec3 c11 = fetchPixel(x1, y1);

    // Bilinear interpolation
    glm::vec3 c0 = {
        c00.x * (1 - tx) + c10.x * tx,
        c00.y * (1 - tx) + c10.y * tx,
        c00.z * (1 - tx) + c10.z * tx
    };
    glm::vec3 c1 = {
        c01.x * (1 - tx) + c11.x * tx,
        c01.y * (1 - tx) + c11.y * tx,
        c01.z * (1 - tx) + c11.z * tx
    };
    glm::vec3 result = {
        c0.x * (1 - ty) + c1.x * ty,
        c0.y * (1 - ty) + c1.y * ty,
        c0.z * (1 - ty) + c1.z * ty
    };

    return result;
}

CUDA_GLOBAL void init_rand_state_entry(curandState *states, int n_states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_states) {
        return;
    }
    curand_init(1234, i, 0, states + i);
}

// ============================================================================
// Fast GPU closest-hit traversal.
//
// Same results as the generic ray_query() in cpu_traverse.cpp, but:
//  * everything is inlined in this TU (the generic path calls into utils.cpp,
//    which under separable compilation can't be inlined),
//  * per-ray reciprocal directions are precomputed once,
//  * nodes are culled against the current closest hit when pushed and popped
//    (the generic path only culls leaves), using the box entry t kept on the stack,
//  * the facing test (cross product + normalize per triangle) is only evaluated
//    for triangles that would actually become the new closest hit,
//  * nodes / vertices are fetched through the read-only cache.
// ============================================================================

static_assert(sizeof(BVHNode) == 32, "fastrt expects 32-byte BVH nodes");
static_assert(sizeof(Face) == 16, "fastrt expects 16-byte faces");

namespace fastrt {

struct RayPre {
    float ox, oy, oz;
    float dx, dy, dz;
    float ix, iy, iz;      // 1/d, valid only where the axis is not parallel
    bool px, py, pz;       // |d| < eps on this axis (parallel)
};

__device__ __forceinline__ glm::vec3 ld3(const glm::vec3 *p) {
    const float *f = reinterpret_cast<const float *>(p);
    return glm::vec3(__ldg(f), __ldg(f + 1), __ldg(f + 2));
}

__device__ __forceinline__ bool slab(float o, float inv, bool par, float mn, float mx, float &tmin, float &tmax) {
    if (par) {
        return !(o < mn || o > mx);
    }
    float t1 = (mn - o) * inv;
    float t2 = (mx - o) * inv;
    if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    return tmin <= tmax;
}

// identical semantics to ray_box_intersection() in utils.cpp
__device__ __forceinline__ bool box_hit(const RayPre &r, const float4 &a, const float4 &b, bool allow_negative, float &t_enter) {
    // a = (min.x, min.y, min.z, max.x), b = (max.y, max.z, left_first, n_prims)
    float tmin = -FLT_MAX, tmax = FLT_MAX;
    if (!slab(r.ox, r.ix, r.px, a.x, a.w, tmin, tmax)) return false;
    if (!slab(r.oy, r.iy, r.py, a.y, b.x, tmin, tmax)) return false;
    if (!slab(r.oz, r.iz, r.pz, a.z, b.y, tmin, tmax)) return false;
    if (!allow_negative) {
        if (tmax < 0.0f) return false;
        if (tmin < 0.0f) tmin = 0.0f;
    }
    t_enter = tmin;
    return true;
}

__device__ __forceinline__ void load_node(const BVHNode *nodes, uint32_t idx, float4 &a, float4 &b) {
    const float4 *p = reinterpret_cast<const float4 *>(nodes + idx);
    a = __ldg(p);
    b = __ldg(p + 1);
}

__device__ __forceinline__ HitResult ray_query_fast(
    const Ray &ray,
    const BVHDataPointers &dp,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
) {
    const float dir_eps = 1e-8f;
    RayPre r;
    r.ox = ray.origin.x; r.oy = ray.origin.y; r.oz = ray.origin.z;
    r.dx = ray.vector.x; r.dy = ray.vector.y; r.dz = ray.vector.z;
    r.px = fabsf(r.dx) < dir_eps; r.ix = r.px ? 0.0f : 1.0f / r.dx;
    r.py = fabsf(r.dy) < dir_eps; r.iy = r.py ? 0.0f : 1.0f / r.dy;
    r.pz = fabsf(r.dz) < dir_eps; r.iz = r.pz ? 0.0f : 1.0f / r.dz;

    const bool facing_cull = !(allow_backward && allow_forward);

    uint32_t stack_idx[TRAVERSE_STACK_SIZE];
    float    stack_t[TRAVERSE_STACK_SIZE];
    int sp = 0;

    float best_t = FLT_MAX;
    bool  best_hit = false;
    uint32_t best_prim = 0;
    float best_u = 0.0f, best_v = 0.0f;

    float4 na, nb;
    load_node(dp.nodes, 0, na, nb);
    {
        // the generic path tests the root box only when the root is a leaf
        float t0;
        uint32_t n0 = __float_as_uint(nb.w);
        if (n0 > 0) {
            if (!box_hit(r, na, nb, allow_negative, t0)) t0 = 0.0f;
        } else {
            t0 = -FLT_MAX;
        }
        stack_idx[sp] = 0; stack_t[sp] = t0; sp++;
    }

    while (sp > 0) {
        --sp;
        uint32_t node_idx = stack_idx[sp];
        if (stack_t[sp] > best_t + (float)TRAVERSE_EPS) {
            continue;
        }
        load_node(dp.nodes, node_idx, na, nb);
        uint32_t first = __float_as_uint(nb.z);
        uint32_t count = __float_as_uint(nb.w);

        if (count > 0) {
            for (uint32_t prim_i = first; prim_i < first + count; prim_i++) {
                const int4 f = __ldg(reinterpret_cast<const int4 *>(dp.faces + prim_i));
                const glm::vec3 a = ld3(dp.vertices + (uint32_t)f.x);
                const glm::vec3 b = ld3(dp.vertices + (uint32_t)f.y);
                const glm::vec3 c = ld3(dp.vertices + (uint32_t)f.z);

                // Moller-Trumbore, same arithmetic as ray_triangle_intersection()
                const float eps = 1e-8f;
                glm::vec3 e1 = vsub(b, a);
                glm::vec3 e2 = vsub(c, a);
                glm::vec3 pvec = glm::vec3(r.dy * e2.z - r.dz * e2.y, r.dz * e2.x - r.dx * e2.z, r.dx * e2.y - r.dy * e2.x);
                float det = vdot(e1, pvec);
                if (det > -eps && det < eps) continue;
                float inv_det = 1.0f / det;
                glm::vec3 tvec = glm::vec3(r.ox - a.x, r.oy - a.y, r.oz - a.z);
                float u = vdot(tvec, pvec) * inv_det;
                if (u < -eps || u > 1.0f + eps) continue;
                glm::vec3 qvec = glm::vec3(tvec.y * e1.z - tvec.z * e1.y, tvec.z * e1.x - tvec.x * e1.z, tvec.x * e1.y - tvec.y * e1.x);
                float v = (r.dx * qvec.x + r.dy * qvec.y + r.dz * qvec.z) * inv_det;
                if (v < -eps || (u + v) > 1.0f + eps) continue;
                float t = vdot(e2, qvec) * inv_det;
                if (!allow_negative && t < eps) continue;
                if (!(t < best_t)) continue;

                if (facing_cull) {
                    glm::vec3 n = glm::vec3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
                    float len2 = vdot(n, n);
                    if (len2 >= 1e-12f) {
                        float facing = vdot(vscale(n, 1.0f / sqrtf(len2)), ray.vector);
                        if (facing > 0.0f && !allow_backward) continue;
                        if (facing < 0.0f && !allow_forward) continue;
                    }
                }

                best_t = t; best_hit = true; best_prim = prim_i; best_u = u; best_v = v;
            }
        } else {
            uint32_t left = first, right = first + 1;
            float4 la, lb, ra, rb;
            load_node(dp.nodes, left, la, lb);
            load_node(dp.nodes, right, ra, rb);
            float lt, rt;
            bool lh = box_hit(r, la, lb, allow_negative, lt);
            bool rh = box_hit(r, ra, rb, allow_negative, rt);
            const float cull = best_t + (float)TRAVERSE_EPS;
            lh = lh && !(lt > cull);
            rh = rh && !(rt > cull);
            if (lh && rh) {
                // push the farther child first so the nearer one is popped next
                if (lt < rt) {
                    stack_idx[sp] = right; stack_t[sp] = rt; sp++;
                    stack_idx[sp] = left;  stack_t[sp] = lt; sp++;
                } else {
                    stack_idx[sp] = left;  stack_t[sp] = lt; sp++;
                    stack_idx[sp] = right; stack_t[sp] = rt; sp++;
                }
            } else if (lh) {
                stack_idx[sp] = left;  stack_t[sp] = lt; sp++;
            } else if (rh) {
                stack_idx[sp] = right; stack_t[sp] = rt; sp++;
            }
        }
    }

    HitResult hit;
    if (!best_hit) {
        hit.hit = false;
        hit.t = 0;
        return hit;
    }
    hit = HitResult(true, best_t, best_u, best_v);
    hit.prim_idx = best_prim;

    const Face &face = dp.faces[best_prim];
    glm::vec3 a = ld3(dp.vertices + face.v1);
    glm::vec3 b = ld3(dp.vertices + face.v2);
    glm::vec3 c = ld3(dp.vertices + face.v3);

    bool have_normal = false;
    if (dp.vertex_normals) {
        glm::vec3 n1 = ld3(dp.vertex_normals + face.v1);
        glm::vec3 n2 = ld3(dp.vertex_normals + face.v2);
        glm::vec3 n3 = ld3(dp.vertex_normals + face.v3);
        glm::vec3 n = {
            hit.barycentrics.x * n1.x + hit.barycentrics.y * n2.x + hit.barycentrics.z * n3.x,
            hit.barycentrics.x * n1.y + hit.barycentrics.y * n2.y + hit.barycentrics.z * n3.y,
            hit.barycentrics.x * n1.z + hit.barycentrics.y * n2.z + hit.barycentrics.z * n3.z
        };
        float len2 = vdot(n, n);
        if (len2 > 1e-20f) {
            hit.normal = vscale(n, 1.0f / sqrtf(len2));
            have_normal = true;
        }
    }
    if (!have_normal) {
        glm::vec3 e1 = vsub(b, a), e2 = vsub(c, a);
        glm::vec3 n = glm::vec3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
        float len2 = vdot(n, n);
        hit.normal = len2 < 1e-12f ? glm::vec3(0.0f) : vscale(n, 1.0f / sqrtf(len2));
    }

    if (dp.uvs) {
        hit.uv = hit.barycentrics.x * dp.uvs[face.v1]
               + hit.barycentrics.y * dp.uvs[face.v2]
               + hit.barycentrics.z * dp.uvs[face.v3];
    }
    hit.color = glm::vec3(1.0f);
    return hit;
}

} // namespace fastrt

// texture / base-color lookup shared by ray_query_entry and the segment walker
__device__ __forceinline__ void apply_material_color(
    HitResult &hit,
    const BVHDataPointers &i_dp,
    const TextureDeviceView *texture_views,
    const MaterialDeviceView *materials,
    int n_textures
) {
    if (hit.hit && materials && texture_views && i_dp.uvs) {
        const Face &face = i_dp.faces[hit.prim_idx];
        if (face.material_idx >= 0) {
            const MaterialDeviceView& mat = materials[face.material_idx];

            // Use texture color if available, otherwise use base color
            if (mat.texture_id >= 0 && mat.texture_id < n_textures) {
                hit.color = sampleTextureGPU(
                    texture_views[mat.texture_id],
                    hit.uv.x,
                    hit.uv.y
                );
            } else {
                hit.color = mat.base_color;
            }
        }
    }
}

CUDA_GLOBAL void ray_query_entry(
    const Rays i_rays,
    const BVHDataPointers i_dp,
    const TextureDeviceView *texture_views,
    const MaterialDeviceView *materials,
    int n_textures,
    HitResults o_out,
    int n_rays,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    HitResult hit = fastrt::ray_query_fast(ray, i_dp, allow_negative, allow_backward, allow_forward);
    apply_material_color(hit, i_dp, texture_views, materials, n_textures);

    o_out.fill(i, hit);
}

// ============================================================================
// Double-shell segment walk (training data generation).
//
// For every ray this reproduces, step by step, the per-iteration torch loop in
// trainer/train.py: enter outer shell -> find outer exit, inner shell hit and
// fine-mesh hit -> emit one training segment -> re-enter the outer shell.
// Arithmetic is written with explicit __fmul_rn/__fadd_rn so that it rounds
// exactly like the separate torch mul/add kernels (no FMA contraction).
//
// Output record k of ray i lives at o_records[(k * n_rays + i) * SEG_REC_FLOATS];
// o_valid[k * n_rays + i] says whether ray i reached segment k.
// ============================================================================

namespace seg {

__device__ __noinline__ HitResult trace(
    const glm::vec3 &o, const glm::vec3 &d, const BVHDataPointers &dp,
    bool allow_negative, bool allow_backward, bool allow_forward
) {
    return fastrt::ray_query_fast(Ray(o, d), dp, allow_negative, allow_backward, allow_forward);
}

// a + b * s, rounded like torch's two separate kernels
__device__ __forceinline__ glm::vec3 axpy(const glm::vec3 &a, const glm::vec3 &b, float s) {
    return glm::vec3(
        __fadd_rn(a.x, __fmul_rn(b.x, s)),
        __fadd_rn(a.y, __fmul_rn(b.y, s)),
        __fadd_rn(a.z, __fmul_rn(b.z, s))
    );
}

__device__ __forceinline__ void put3(float *r, int off, const glm::vec3 &v) {
    r[off] = v.x; r[off + 1] = v.y; r[off + 2] = v.z;
}

} // namespace seg

CUDA_GLOBAL void shell_segments_entry(
    const glm::vec3 *i_origs,
    const glm::vec3 *i_dirs,
    const BVHDataPointers outer,
    const BVHDataPointers inner,
    const BVHDataPointers fine,
    const TextureDeviceView *fine_texture_views,
    const MaterialDeviceView *fine_materials,
    int fine_n_textures,
    float *o_records,
    bool *o_valid,
    int n_rays,
    int max_segments,
    float shift_eps,
    float no_exit_t
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    const glm::vec3 d = i_dirs[i];
    const glm::vec3 x = i_origs[i];

    int k = 0;

    // first entry into the outer shell (front faces only)
    HitResult h = seg::trace(x, d, outer, false, false, true);
    bool alive = h.hit;
    glm::vec3 enter = seg::axpy(x, d, h.t);
    glm::vec3 enter_normal = h.normal;

    for (; k < max_segments && alive; k++) {
        enter = seg::axpy(enter, d, shift_eps);

        // exit from the outer shell (back faces only)
        HitResult he = seg::trace(enter, d, outer, false, true, false);
        float exit_t = he.hit ? he.t : no_exit_t;
        glm::vec3 exit_p = seg::axpy(enter, d, exit_t);

        HitResult hi = seg::trace(enter, d, inner, false, true, true);
        glm::vec3 inner_p = seg::axpy(enter, d, hi.t);

        HitResult hf = seg::trace(enter, d, fine, false, true, true);
        apply_material_color(hf, fine, fine_texture_views, fine_materials, fine_n_textures);

        bool gt = hf.hit && (hf.t < exit_t);
        bool inner_apply = hi.hit && (hi.t < exit_t);

        float *r = o_records + ((size_t)k * n_rays + i) * SEG_REC_FLOATS;
        seg::put3(r, 0, enter);
        seg::put3(r, 3, inner_apply ? inner_p : exit_p);
        seg::put3(r, 6, d);
        seg::put3(r, 9, enter_normal);
        seg::put3(r, 12, hf.hit ? hf.normal : glm::vec3(0.0f));
        seg::put3(r, 15, hf.color);
        r[18] = hf.t;
        r[19] = gt ? 1.0f : 0.0f;
        o_valid[(size_t)k * n_rays + i] = true;

        // re-enter the outer shell further along the ray (front faces only)
        glm::vec3 from = seg::axpy(exit_p, d, shift_eps);
        HitResult hn = seg::trace(from, d, outer, false, false, true);

        alive = !gt && hf.hit && he.hit && hn.hit;
        enter = seg::axpy(from, d, hn.t);
        enter_normal = hn.normal;
    }

    for (; k < max_segments; k++) {
        o_valid[(size_t)k * n_rays + i] = false;
    }
}

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
    cudaStream_t stream
) {
    shell_segments_entry<<<(n_rays + 127) / 128, 128, 0, stream>>>(
        i_origs, i_dirs,
        outer.get_data_pointers(), inner.get_data_pointers(), fine.get_data_pointers(),
        fine.texture_views.empty() ? nullptr : fine.texture_views.data().get(),
        fine.materials.empty() ? nullptr : fine.materials.data().get(),
        (int)fine.texture_views.size(),
        o_records, o_valid, n_rays, max_segments, shift_eps, no_exit_t
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in trace_shell_segments: %s\n", cudaGetErrorString(err));
    }
}

CUDA_GLOBAL void point_query_entry(
    const glm::vec3 *i_points,
    const BVHDataPointers i_dp,
    SDFHitResults o_out,
    int n_points
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) {
        return;
    }

    glm::vec3 point = i_points[i];

    SDFHitResult hit = point_query(point, i_dp);
    o_out.fill(i, hit);
}

CUDA_DEVICE void ray_query_all_gpu(
    const Ray &i_ray,
    const BVHDataPointers &i_dp,
    HitResults o_hits,
    uint32_t &o_n_hits,
    int max_hits,
    bool allow_negative,
    bool allow_backward,
    bool allow_forward
) {
    uint32_t node_stack[TRAVERSE_STACK_SIZE];
    int cur_stack_size = 0;
    node_stack[cur_stack_size++] = 0; // root node

    int n_hits_registered = 0;

    while (cur_stack_size > 0 && n_hits_registered < max_hits) {
        uint32_t node_idx = node_stack[--cur_stack_size];
        const BVHNode &node = i_dp.nodes[node_idx];

        bool is_leaf = node.is_leaf();

        if (is_leaf) {
            HitResult node_hit = {false, FLT_MAX};
            for (int prim_i = node.left_first_prim; prim_i < node.left_first_prim + node.n_prims; prim_i++) {
                const Face &face = i_dp.faces[prim_i];

                HitResult prim_hit = ray_triangle_intersection(i_ray, face, i_dp.vertices, allow_negative);
                prim_hit.prim_idx = prim_i;

                glm::vec3 normal = ray_triangle_norm(face, i_dp.vertices);
                float facing = vdot(normal, i_ray.vector);
                if (facing > 0.0f && !allow_backward) {
                    continue;
                }
                if (facing < 0.0f && !allow_forward) {
                    continue;
                }

                if (prim_hit.hit) {
                    o_hits.fill(n_hits_registered++, prim_hit);
                    if (n_hits_registered >= max_hits) {
                        break;
                    }
                }
            }
        } else {
            uint32_t left = node.left();
            HitResult left_hit = ray_box_intersection(i_ray, i_dp.nodes[left].bbox, allow_negative);
            if (left_hit.hit) {
                node_stack[cur_stack_size++] = left;
            }

            uint32_t right = node.right();            
            HitResult right_hit = ray_box_intersection(i_ray, i_dp.nodes[right].bbox, allow_negative);
            if (right_hit.hit) {
                node_stack[cur_stack_size++] = right;
            }
        }
    }

    o_n_hits = n_hits_registered;

    // Manual bubble/insertion sort - no temporary buffers needed
    // For small n_hits_registered (< 100), this is often faster than thrust
    for (int i = 1; i < n_hits_registered; i++) {
        float key_t = o_hits.t[i];
        bool key_mask = o_hits.masks[i];
        uint32_t key_prim = o_hits.prim_idxs ? o_hits.prim_idxs[i] : 0;
        
        int j = i - 1;
        while (j >= 0 && o_hits.t[j] > key_t) {
            o_hits.t[j + 1] = o_hits.t[j];
            o_hits.masks[j + 1] = o_hits.masks[j];
            if (o_hits.prim_idxs) {
                o_hits.prim_idxs[j + 1] = o_hits.prim_idxs[j];
            }
            j--;
        }
        o_hits.t[j + 1] = key_t;
        o_hits.masks[j + 1] = key_mask;
        if (o_hits.prim_idxs) {
            o_hits.prim_idxs[j + 1] = key_prim;
        }
    }
}

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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) {
        return;
    }

    Ray ray = i_rays[i];

    HitResults o_hits_moved = HitResults{
        o_hits.masks + i * max_hits_per_ray,
        o_hits.t + i * max_hits_per_ray,
        o_hits.prim_idxs + i * max_hits_per_ray,
        // o_hits.normals + i * max_hits_per_ray
    };

    ray_query_all_gpu(
        ray,
        i_dp,
        o_hits_moved,
        o_n_hits[i],
        max_hits_per_ray,
        allow_negative,
        allow_backward,
        allow_forward
    );
}
