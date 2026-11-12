#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cpu_traverse.h"
#include "material.h"

#include <cuda_runtime.h>
#include "gpu_traverse.cuh"
#include "mesh_sampler.cuh"

namespace nb = nanobind;

using h_float3 = nb::ndarray<float, nb::shape<3>, nb::numpy>;

using h_float2_batch = nb::ndarray<float, nb::shape<-1, 2>, nb::device::cpu, nb::c_contig>;
using h_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uint3_batch = nb::ndarray<uint32_t, nb::shape<-1, 3>, nb::device::cpu, nb::c_contig>;
using h_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cpu, nb::c_contig>;
using h_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>;
using h_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cpu, nb::c_contig>;

using d_float2_batch = nb::ndarray<float, nb::shape<-1, 2>, nb::device::cuda, nb::c_contig>;
using d_float3_batch = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cuda, nb::c_contig>;
using d_bool_batch = nb::ndarray<bool, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_boolN_batch = nb::ndarray<bool, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_float_batch = nb::ndarray<float, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_floatN_batch = nb::ndarray<float, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_uint_batch = nb::ndarray<uint32_t, nb::shape<-1>, nb::device::cuda, nb::c_contig>;
using d_uintN_batch = nb::ndarray<uint32_t, nb::shape<-1, -1>, nb::device::cuda, nb::c_contig>;
using d_int_batch = nb::ndarray<int, nb::shape<-1>, nb::device::cuda, nb::c_contig>;

NB_MODULE(mesh_utils_impl, m) {
    nb::class_<Mesh>(m, "Mesh")
        .def_static("from_file", [](const char *scene_path,
                                    const char *up_axis,
                                    const char *forward_axis,
                                    float scale,
                                    bool build_bvh,
                                    int max_leaf_size,
                                    bool smooth_normals) {
            return Mesh(scene_path, up_axis, forward_axis, scale, build_bvh, max_leaf_size, smooth_normals);
        }, nb::arg("scene_path"),
           nb::kw_only(),
           nb::arg("up_axis") = "y",
           nb::arg("forward_axis") = "-z",
           nb::arg("scale") = 1.0f,
           nb::arg("build_bvh") = false,
           nb::arg("max_leaf_size") = 25,
           nb::arg("smooth_normals") = false)
        .def("get_num_vertices", [](Mesh& self) {
            return self.vertices.size();
        })
        .def("get_num_faces", [](Mesh& self) {
            return self.faces.size();
        })
        .def("get_vertices", [](Mesh& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.vertices.data(),
                {self.vertices.size(), 3}
            );
        })
        .def("get_faces", [](Mesh& self) {
            return nb::ndarray<uint32_t, nb::numpy>(
                (uint32_t *) self.faces.data(),
                {self.faces.size(), 4}
            );
        })
        .def("has_uvs", [](Mesh& self) {
            return !self.uvs.empty();
        })
        .def("get_uvs", [](Mesh& self) {
            return nb::ndarray<float, nb::numpy>(
                (float *) self.uvs.data(),
                {self.uvs.size(), 2}
            );
        })
        .def("save_preview", &Mesh::save_preview,
             nb::arg("filename"),
             nb::arg("width") = 512,
             nb::arg("height") = 512)
        .def("save_to_obj", &Mesh::save_to_obj)
        .def("get_bounds", [](Mesh& self) {
            auto [min, max] = self.bounds();
            return nb::make_tuple(
                h_float3(&min).cast(),
                h_float3(&max).cast()
            );
        })
        .def("vertices_memory_bytes", &Mesh::vertices_memory_bytes)
        .def("faces_memory_bytes", &Mesh::faces_memory_bytes)
        .def("bvh_memory_bytes", &Mesh::bvh_memory_bytes)
        .def("vertices_faces_bvh_memory_bytes", &Mesh::vertices_faces_bvh_memory_bytes)
    ;

    nb::class_<CPUTraverser>(m, "CPUTraverser")
        .def(nb::init<const Mesh&>())
        .def("ray_query", [](
            CPUTraverser& self,
            h_float3_batch& i_ray_origs,
            h_float3_batch& i_ray_vecs,
            h_bool_batch& o_mask,
            h_float_batch& o_t,
            h_uint_batch& o_prim_idx,
            h_float3_batch& o_normals,
            h_float2_batch& o_uvs,
            h_float3_batch& o_colors,
            h_float3_batch& o_barycentrics,
            bool allow_negative,
            bool allow_backward,
            bool allow_forward
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),
                (glm::vec2 *) o_uvs.data(),
                (glm::vec3 *) o_colors.data(),
                (glm::vec3 *) o_barycentrics.data(),
                n_rays,
                allow_negative,
                allow_backward,
                allow_forward
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_t"),
           nb::arg("o_prim_idx"), nb::arg("o_normals"), nb::arg("o_uvs"), nb::arg("o_colors"),
           nb::arg("o_barycentrics"),
           nb::arg("allow_negative") = false, nb::arg("allow_backward") = true, nb::arg("allow_forward") = true)
        .def("point_query", [](
            CPUTraverser& self,
            h_float3_batch& i_points,
            h_float_batch& o_t,
            h_float3_batch& o_closests
        ) {
            uint32_t n_points = i_points.shape(0);

            self.point_query(
                (glm::vec3 *) i_points.data(),
                o_t.data(),
                (glm::vec3 *) o_closests.data(),
                n_points
            );
        })
    ;

    nb::class_<GPUMeshSampler>(m, "GPUMeshSampler")
        .def(nb::init<const Mesh&, int>(), nb::arg("mesh"), nb::arg("max_points"))
        .def("sample", [](
            GPUMeshSampler& self,
            d_float3_batch& o_points,
            d_float3_batch& o_coords,
            d_uint_batch& o_face_idxs,
            int n_points,
            uintptr_t stream
        ) {
            if (n_points > self.max_points_) {
                throw std::runtime_error("n_points exceeds max_points set in constructor");
            }
            if (o_points.shape(0) < n_points || o_points.shape(1) != 3) {
                throw std::runtime_error("o_points has incorrect shape");
            }

            self.sample(
                (glm::vec3 *) o_points.data(),
                (glm::vec3 *) o_coords.data(),
                o_face_idxs.data(),
                n_points,
                (cudaStream_t) stream
            );
        }, nb::arg("o_points"), nb::arg("o_coords"), nb::arg("o_face_idxs"), nb::arg("n_points"), nb::arg("stream") = 0)
    ;

    m.def("trace_shell_segments", [](
            const GPUTraverser &outer,
            const GPUTraverser &inner,
            const GPUTraverser &fine,
            d_float3_batch &i_origs,
            d_float3_batch &i_dirs,
            nb::ndarray<float, nb::shape<-1, -1, SEG_REC_FLOATS>, nb::device::cuda, nb::c_contig> &o_records,
            d_boolN_batch &o_valid,
            float shift_eps,
            float no_exit_t,
            uintptr_t stream
        ) {
            int n_rays = i_origs.shape(0);
            int max_segments = o_records.shape(0);
            if (i_dirs.shape(0) != (size_t)n_rays || o_records.shape(1) != (size_t)n_rays ||
                o_valid.shape(0) != (size_t)max_segments || o_valid.shape(1) != (size_t)n_rays) {
                throw std::runtime_error("trace_shell_segments: inconsistent shapes");
            }
            trace_shell_segments(outer, inner, fine,
                (const glm::vec3 *) i_origs.data(), (const glm::vec3 *) i_dirs.data(),
                o_records.data(), o_valid.data(), n_rays, max_segments, shift_eps, no_exit_t, (cudaStream_t) stream);
        }, nb::arg("outer"), nb::arg("inner"), nb::arg("fine"), nb::arg("i_origs"), nb::arg("i_dirs"),
           nb::arg("o_records"), nb::arg("o_valid"), nb::arg("shift_eps"), nb::arg("no_exit_t"), nb::arg("stream") = 0);

    nb::class_<GPUTraverser>(m, "GPUTraverser")
        .def(nb::init<const Mesh&>())
        .def("ray_query", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_bool_batch& o_mask,
            d_float_batch& o_t,
            d_uint_batch& o_prim_idx,
            d_float3_batch& o_normals,
            d_float2_batch& o_uvs,
            d_float3_batch& o_colors,
            d_float3_batch& o_barycentrics,
            bool allow_negative,
            bool allow_backward,
            bool allow_forward,
            uintptr_t stream
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_t.data(),
                o_prim_idx.data(),
                (glm::vec3 *) o_normals.data(),
                (glm::vec2 *) o_uvs.data(),
                (glm::vec3 *) o_colors.data(),
                (glm::vec3 *) o_barycentrics.data(),
                n_rays,
                allow_negative,
                allow_backward,
                allow_forward,
                (cudaStream_t) stream
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_t"),
           nb::arg("o_prim_idx"), nb::arg("o_normals"), nb::arg("o_uvs"), nb::arg("o_colors"),
           nb::arg("o_barycentrics"),
           nb::arg("allow_negative") = false, nb::arg("allow_backward") = true, nb::arg("allow_forward") = true,
           nb::arg("stream") = 0)
        .def("ray_query_all", [](
            GPUTraverser& self,
            d_float3_batch& i_ray_origs,
            d_float3_batch& i_ray_vecs,
            d_boolN_batch& o_mask,
            d_floatN_batch& o_dist,
            d_uintN_batch& o_prim_idx,
            d_uint_batch& o_n_hits,
            int max_hits_per_ray,
            bool allow_negative,
            bool allow_backward,
            bool allow_forward
        ) {
            uint32_t n_rays = i_ray_origs.shape(0);

            self.ray_query_all(
                (glm::vec3 *) i_ray_origs.data(),
                (glm::vec3 *) i_ray_vecs.data(),
                o_mask.data(),
                o_dist.data(),
                o_prim_idx.data(),
                o_n_hits.data(),
                max_hits_per_ray,
                n_rays,
                allow_negative,
                allow_backward,
                allow_forward
            );
        }, nb::arg("i_ray_origs"), nb::arg("i_ray_vecs"), nb::arg("o_mask"), nb::arg("o_dist"),
           nb::arg("o_prim_idx"), nb::arg("o_n_hits"), nb::arg("max_hits_per_ray"), nb::arg("allow_negative") = false, nb::arg("allow_backward") = true, nb::arg("allow_forward") = true)        
        .def("point_query", [](
            GPUTraverser& self,
            d_float3_batch& i_points,
            d_float_batch& o_t,
            d_float3_batch& o_closests,
            d_float3_batch& o_barycentricses,
            d_uint_batch& o_face_idxs
        ) {
            uint32_t n_points = i_points.shape(0);

            self.point_query(
                (glm::vec3 *) i_points.data(),
                o_t.data(),
                (glm::vec3 *) o_closests.data(),
                (glm::vec3 *) o_barycentricses.data(),
                o_face_idxs.data(),
                n_points
            );
        })
    ;
}
