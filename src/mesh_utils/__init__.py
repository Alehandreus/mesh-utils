import torch
import numpy as np
from collections import namedtuple
from .mesh_utils_impl import Mesh, GPUTraverser, GPUMeshSampler
from .mesh_utils_impl import trace_shell_segments as _trace_shell_segments

RayQueryResult = namedtuple('RayQueryResult', ['mask', 'distance', 'normals', 'uv', 'color', 'face_idx', 'barycentrics'])
SampleResult = namedtuple('SampleResult', ['points', 'barycentrics', 'face_indices'])


def generate_camera_rays(mesh, img_size=512, distance=1.0, device=None):
    """Generate camera rays for rendering a mesh.

    Automatically positions the camera to view the entire mesh from
    an upper-right viewpoint using PyTorch so the outputs can be pushed
    straight to a GPU.

    Args:
        mesh: Mesh object to generate rays for
        img_size: Image resolution (default 512x512)
        distance: Distance of the camera from the mesh
        device: Torch device (string or torch.device). Defaults to CUDA if
            available, otherwise CPU.

    Returns:
        cam_poses: (img_size*img_size, 3) tensor of ray origins
        dirs: (img_size*img_size, 3) tensor of ray directions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n_pixels = img_size * img_size

    mesh_min, mesh_max = mesh.get_bounds()
    mesh_min = torch.as_tensor(mesh_min, dtype=torch.float32, device=device)
    mesh_max = torch.as_tensor(mesh_max, dtype=torch.float32, device=device)

    max_extent = torch.max(mesh_max - mesh_min)
    center = (mesh_max + mesh_min) * 0.5

    cam_pos = torch.stack([
        center[0] + max_extent * 1.0,
        center[1] + max_extent * 0.5,
        center[2] + max_extent * 1.5,
    ])
    cam_dir = (center - cam_pos) / distance

    camera_up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    x_dir = torch.linalg.cross(cam_dir, camera_up)
    x_dir = x_dir / torch.linalg.norm(x_dir) * (max_extent / 2)

    y_dir = -torch.linalg.cross(x_dir, cam_dir)
    y_dir = y_dir / torch.linalg.norm(y_dir) * (max_extent / 2)

    x_coords = torch.linspace(-1, 1, img_size, device=device)
    y_coords = torch.linspace(-1, 1, img_size, device=device)
    x_flat = x_coords.repeat(img_size)
    y_flat = y_coords.repeat_interleave(img_size)

    cam_poses = cam_pos.unsqueeze(0).repeat(n_pixels, 1)
    dirs = cam_dir.unsqueeze(0) + x_dir.unsqueeze(0) * x_flat.unsqueeze(1) + y_dir.unsqueeze(0) * y_flat.unsqueeze(1)

    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True)

    return cam_poses, dirs


class GPURayTracer:
    def __init__(self, mesh):
        """Initialize GPU ray tracer.

        Args:
            mesh: Mesh object to trace rays against (must have BVH built)
        """
        self.bvh_traverser = GPUTraverser(mesh)

    def trace(self, cam_poses, dirs, allow_negative=True, allow_backward=True, allow_forward=True):
        """Trace rays through the mesh.

        Args:
            cam_poses: (n_rays, 3) tensor of ray origins
            dirs: (n_rays, 3) tensor of ray directions
            allow_negative: Allow negative ray distances (behind ray origin)
            allow_backward: Allow hits facing away from ray direction
            allow_forward: Allow hits facing towards ray direction

        Returns:
            RayQueryResult with:
                mask: (n_rays,) bool tensor of hits
                distance: (n_rays,) float tensor of hit distances
                normals: (n_rays, 3) float tensor of surface normals
                uv: (n_rays, 2) float tensor of UV coordinates
                color: (n_rays, 3) float tensor of surface colors
                face_idx: (n_rays,) uint32 tensor of the hit face index (undefined for misses)
        """
        n_rays = cam_poses.shape[0]

        face_idxs     = torch.zeros((n_rays,),    dtype=torch.uint32,  device="cuda")
        mask          = torch.zeros((n_rays,),    dtype=torch.bool,    device="cuda")
        t             = torch.zeros((n_rays,),    dtype=torch.float32, device="cuda") + 1e9
        normals       = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")
        uvs           = torch.zeros((n_rays, 2),  dtype=torch.float32, device="cuda")
        colors        = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")
        barycentrics  = torch.zeros((n_rays, 3),  dtype=torch.float32, device="cuda")

        self.bvh_traverser.ray_query(
            cam_poses,
            dirs,
            mask,
            t,
            face_idxs,
            normals,
            uvs,
            colors,
            barycentrics,
            allow_negative,
            allow_backward,
            allow_forward,
            torch.cuda.current_stream().cuda_stream,
        )

        return RayQueryResult(mask=mask, distance=t, normals=normals, uv=uvs, color=colors, face_idx=face_idxs.long(), barycentrics=barycentrics)


class GPUSampler:
    def __init__(self, mesh, max_points=100000):
        """Initialize mesh sampler.

        Args:
            mesh: Mesh object to sample from
            max_points: Maximum number of points that can be sampled
        """
        self.sampler = GPUMeshSampler(mesh, max_points)
        self.max_points = max_points

    def sample(self, n_points):
        """Sample points on the mesh surface.

        Args:
            n_points: Number of points to sample

        Returns:
            SampleResult with:
                points: (n_points, 3) tensor of sampled positions
                barycentrics: (n_points, 3) tensor of barycentric coordinates
                face_indices: (n_points,) tensor of face indices
        """
        if n_points > self.max_points:
            raise ValueError(f"n_points ({n_points}) exceeds max_points ({self.max_points})")

        # Allocate output tensors
        points = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
        barycentrics = torch.zeros((n_points, 3), dtype=torch.float32, device="cuda")
        face_indices = torch.zeros((n_points,), dtype=torch.uint32, device="cuda")

        # Sample
        self.sampler.sample(points, barycentrics, face_indices, n_points, torch.cuda.current_stream().cuda_stream)

        return SampleResult(points=points, barycentrics=barycentrics, face_indices=face_indices.long())


ShellSegments = namedtuple('ShellSegments', [
    'enter_points', 'exit_points', 'directions', 'outer_normals',
    'gt_normals', 'gt_colors', 'gt_distance', 'gt_mask', 'valid', 'records',
])


def trace_shell_segments(outer, inner, fine, origins, directions, max_segments=11, shift_eps=1e-8, no_exit_t=1e-3):
    """Walk rays through a double shell in one kernel launch.

    For each ray: enter the outer shell, then repeatedly emit a segment
    (outer entry point -> inner-shell hit or outer exit, plus the fine-mesh
    hit along it) and re-enter the outer shell, until the fine mesh is hit
    inside a segment, the ray escapes, or max_segments is reached.

    Args:
        outer, inner, fine: GPURayTracer for the outer shell, inner shell and fine mesh
        origins, directions: (n_rays, 3) float32 CUDA tensors
        max_segments: maximum number of segments per ray
        shift_eps: distance rays are pushed forward before each re-trace
        no_exit_t: segment length used when a ray finds no outer-shell exit

    Returns:
        ShellSegments whose fields have a leading (max_segments, n_rays) shape;
        entry [k, i] is meaningful only where valid[k, i] is True.
        gt_distance is the fine-mesh hit distance from the segment entry point,
        gt_mask whether that hit lies inside the segment. `records` is the
        packed (max_segments, n_rays, 20) buffer all other fields view into.
    """
    n = origins.shape[0]
    origins = origins.contiguous().float()
    directions = directions.contiguous().float()
    records = torch.empty((max_segments, n, 20), dtype=torch.float32, device=origins.device)
    valid = torch.empty((max_segments, n), dtype=torch.bool, device=origins.device)
    _trace_shell_segments(
        outer.bvh_traverser, inner.bvh_traverser, fine.bvh_traverser,
        origins, directions, records, valid, shift_eps, no_exit_t,
        torch.cuda.current_stream().cuda_stream,
    )
    return ShellSegments(
        enter_points=records[..., 0:3], exit_points=records[..., 3:6],
        directions=records[..., 6:9], outer_normals=records[..., 9:12],
        gt_normals=records[..., 12:15], gt_colors=records[..., 15:18],
        gt_distance=records[..., 18], gt_mask=records[..., 19] > 0.5,
        valid=valid, records=records,
    )
