from trimesh.scene.transforms import SceneGraph
import trimesh.transformations as tra
import io
import os
import itertools
import uuid
import numpy as np
from numpy.typing import NDArray
from shapely import vectorized
from typing import Optional, Tuple, Hashable
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial.transform import Rotation as R
from scene_synthesizer.scene import SupportSurface
from scene_synthesizer.constants import EDGE_KEY_METADATA

def batch_rotation_matrix(angle: NDArray, direction: NDArray) -> NDArray:
    """
    Creates a batch of rotation matrices from a batch of angles and directions.
    Args:
        angle (NDArray): (N,) float, the angle of rotation.
        direction (NDArray): (N, 3) float, the direction of rotation.

    Returns:
        NDArray: (N, 4, 4) float, the rotation matrices.
    """
    N = len(angle)
    
    # Normalize direction
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    
    # Create identity matrices
    I = np.eye(3)[None, :, :].repeat(N, axis=0)
    
    # Create skew-symmetric matrices [k]×
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -direction[:, 2]
    K[:, 0, 2] = direction[:, 1]
    K[:, 1, 0] = direction[:, 2]
    K[:, 1, 2] = -direction[:, 0]
    K[:, 2, 0] = -direction[:, 1]
    K[:, 2, 1] = direction[:, 0]
    
    # Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
    sin_theta = np.sin(angle)[:, None, None]
    cos_theta = np.cos(angle)[:, None, None]
    
    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    matrices = np.zeros((N, 4, 4))
    matrices[:, :3, :3] = R
    matrices[:, 3, 3] = 1
    return matrices


def batch_translation_matrix(direction: NDArray) -> NDArray:
    """
    Creates a batch of translation matrices from a batch of 3D translations.

    Args:
        direction (NDArray): (N, 3) float, the direction of translation.

    Returns:
        NDArray: (N, 4, 4) float, the translation matrices.
    """
    N = direction.shape[0]
    matrices = np.tile(np.eye(4), (N, 1, 1))
    matrices[:, :3, 3] = direction
    return matrices


def sample_random_z_rotations(
    lower: float = 0.0, 
    upper: float = 2.0 * np.pi, 
    num_samples: int = 1, 
    seed = None
) -> NDArray:
    """
    Generates a batch of random rotation matrices around the Z-axis.
    This method is optimized for generating multiple matrices efficiently.

    Args:
        lower (float): The lower bound for the uniform distribution of radians.
        upper (float): The upper bound for the uniform distribution of radians.
        num_samples (int): The number of rotation matrices to generate.
                            Defaults to 1.
        seed (int, numpy.random._generator.Generator, optional):
            A seed or random number generator. Defaults to None which
            creates a new default random number generator.

    Returns:
        NDArray: If num_samples is 1, returns a single (4,4) rotation matrix.
                    Otherwise, returns an array of shape (num_samples, 4, 4)
                    containing the rotation matrices.
    """
    rng = np.random.default_rng(seed)

    rad_batch = rng.uniform(lower, upper, size=num_samples)
    
    cos_angles = np.cos(rad_batch)
    sin_angles = np.sin(rad_batch)
    
    # Initialize batch of 4x4 matrices
    rotation_matrices = np.zeros((num_samples, 4, 4))
    
    # Top-left 2x2 for Z-rotation
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 1] = -sin_angles
    rotation_matrices[:, 1, 0] = sin_angles
    rotation_matrices[:, 1, 1] = cos_angles
    
    # Z-axis rotation keeps Z' = Z
    rotation_matrices[:, 2, 2] = 1.0
    
    # Homogeneous coordinate
    rotation_matrices[:, 3, 3] = 1.0
    
    return rotation_matrices

def point_to_translation_matrix(translations):
    """
    Creates a batch of 4x4 translation matrices from a batch of 3D coordinates.

    Args:
        translations: A NumPy array of shape (N, 3) where N is the batch size,
                    and each row represents the (x, y, z) translation vector.

    Returns:
        A NumPy array of shape (N, 4, 4) containing the translation matrices.
    """
    N = translations.shape[0]
    matrices = np.tile(np.eye(4, dtype=translations.dtype), (N, 1, 1))
    matrices[:, :3, 3] = translations
    return matrices


def check_within_polygon(polygon, points: NDArray):
    """Check if points are within a polygon."""
    mask = vectorized.contains(polygon, *points.T)
    return mask


def get_support_transforms(supports: list[SupportSurface]):
    """Get the transform of a support polygon."""
    return np.concatenate([s.transform[np.newaxis, :, :] for s in supports])


def get_support_node_transforms(scene, supports: list[SupportSurface]):
    """Get the transform of a support polygon."""
    return np.concatenate([
        scene.graph.get(s.node_name)[0][np.newaxis, :, :] for s in supports])

def get_support_node_names(supports: list[SupportSurface]):
    """Get the node name of a support polygon."""
    return np.array([s.node_name for s in supports])


def get_transform_batch(scene, frame_tos: list[str], frame_from: str | None = None):
    """Get the transform of a batch of objects."""
    res = np.zeros((len(frame_tos), 4, 4))
    unique_frame_tos = np.unique(frame_tos)
    for frame_to in unique_frame_tos:
        mask = np.where(frame_tos == frame_to)[0]
        transform = scene_graph_transform_get(scene._scene.graph, frame_to, frame_from, scene.edge_batch, scene.cache)[0]
        if len(transform.shape) == 2:
            transform = np.tile(transform, (len(mask), 1, 1))
            res[mask] = transform
        else:
            res[mask] = transform[mask]
    return res


def homogeneous_inv_batch(matrices):
    """Inverse homogeneous matrix.
    Slightly faster than np.linalg.inv or trimesh.transformations.inverse_matrix.
    """
    # return np.linalg.inv(matrices)
    if len(matrices.shape) == 2:
        res = np.eye(4)
        res[:3, :3] = matrices[:3, :3].swapaxes(0, 1) # RT
        res[:3, 3:] = (-res[:3, :3]) @ (matrices[:3, 3:]) # -RT * t
    else:
        res = np.tile(np.eye(4), (matrices.shape[0], 1, 1))
        res[:, :3, :3] = matrices[:, :3, :3].swapaxes(1, 2) # RT
        res[:, :3, 3:] = (-res[:, :3, :3]) @ (matrices[:, :3, 3:]) # -RT * t
    return res

_identity = np.eye(4)
_identity.flags["WRITEABLE"] = False
# modified from trimesh.scene.transforms.SceneGraph.get() to support batched scenes
def scene_graph_transform_get(
    scene_graph: SceneGraph, 
    frame_to: Hashable, 
    frame_from: Optional[Hashable] = None,
    edge_batch: dict = None,
    cache: dict = None,
) -> Tuple[NDArray[np.float64], Optional[Hashable]]:
    """
    Get the transform from one frame to another.

    Parameters
    ------------
    frame_to : hashable
        Node name, usually a string (eg 'mesh_0')
    frame_from : hashable
        Node name, usually a string (eg 'world').
        If None it will be set to scene_graph.base_frame

    Returns
    ----------
    transform : (4, 4) float
        Homogeneous transformation matrix
    geometry
        The name of the geometry if it exists

    Raises
    -----------
    ValueError
        If the frames aren't connected.
    """

    # use base frame if not specified
    if frame_from is None:
        frame_from = scene_graph.base_frame

    # look up transform to see if we have it already
    key = (frame_from, frame_to)
    # print ("key", key)
    if key in cache:
        # print ("cache hit", cache[key][0].shape)
        return cache[key]

    # get the geometry at the final node if any
    geometry = scene_graph.transforms.node_data[frame_to].get("geometry")

    # get a local reference to edge data
    data = scene_graph.transforms.edge_data
    # print ("data", list(data.keys()))

    if frame_from == frame_to:
        # if we're going from ourscene_graph return identity
        matrix = _identity
    elif key in data:
        # if the path is just an edge return early
        if key in edge_batch:
            matrix = edge_batch[key]
        else:
            matrix = data[key]["matrix"]
        # print ("key", key)
        # print (matrix.shape)
    else:
        # we have a 3+ node path
        # get the path from the forest always going from
        # parent -> child -> child
        path = scene_graph.transforms.shortest_path(frame_from, frame_to)
        # the path should always start with `frame_from`
        assert path[0] == frame_from
        # and end with the `frame_to` node
        assert path[-1] == frame_to

        # loop through pairs of the path
        matrices = []
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) in edge_batch:
                forward = {"matrix": edge_batch[(u, v)]}
            else:
                forward = data.get((u, v))
            # print ("in path", (u, v), forward["matrix"].shape)
            if forward is not None:
                if "matrix" in forward:
                    # append the matrix from u to v
                    matrices.append(forward["matrix"])
                continue
            # since forwards didn't exist backward must
            # exist otherwise this is a disconnected path
            # and we should raise an error anyway
            backward = data[(v, u)]
            if "matrix" in backward:
                # append the inverted backwards matrix
                matrices.append(np.linalg.inv(backward["matrix"]))
        # filter out any identity matrices
        matrices = [m for m in matrices if np.abs(m - _identity).max() > 1e-8]
        # print ("batch transform get matrices", [m.shape for m in matrices])
        if len(matrices) == 0:
            matrix = _identity
        elif len(matrices) == 1:
            matrix = matrices[0]
        else:
            # multiply matrices into single transform
            matrix = batched_multi_dot(matrices)

    # if instructed to repair rigid transforms do it here
    # if scene_graph.repair_rigid is not None:
        # matrix = trimesh.transformations.fix_rigid(matrix, max_deviance=scene_graph.repair_rigid)

    # matrix being edited in-place leads to subtle bugs
    matrix.flags["WRITEABLE"] = False

    # store the result
    cache[key] = (matrix, geometry)

    return matrix, geometry


def batched_multi_dot(matrices):
    """
    Batched matrix multiplication.
    """
    if len(matrices) == 1:
        return matrices[0]
    else:
        res = matrices[0]
        for i in range(1, len(matrices)):
            res = res @ matrices[i]
        return res


def make_mesh_buffer(obj_id: str, obj_asset, io_buf=False, default_folder="/tmp/sgv2"):
    # asset_mesh = obj_asset.mesh(use_collision_geometry=True)
    node_name_T_mesh_list = obj_asset.node_named_geometries(use_collision_geometry=True)
    paths = []
    for node_name, T, mesh in node_name_T_mesh_list:
        if io_buf:
            mesh_buffer = io.BytesIO()
            mesh.export(mesh_buffer, file_type='stl')
            return mesh_buffer
        else:
            if not os.path.exists(default_folder):
                os.makedirs(default_folder)
            fn = node_name.replace("object/", f"{obj_id}___") # triple _ to seperate obj_id and node_name
            path = os.path.join(default_folder, f"{fn}.stl")
            mesh.export(path)
            paths.append(path)
    return paths


def transform_matrix_to_vectors(matrix):
    """
    Converts a 4x4 transformation matrix to a translation vector and a quaternion.

    Args:
        matrix: A 4x4 numpy array representing the transformation.

    Returns:
        A tuple containing:
        - translation (numpy.ndarray): A 3-element numpy array for the translation (x, y, z).
        - quaternion (numpy.ndarray): A 4-element numpy array for the quaternion (x, y, z, w).
    """
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 numpy array.")

    # Extract the translation vector from the last column
    translation = matrix[:3, 3]

    # Extract the 3x3 rotation submatrix
    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to a quaternion
    # Note: SciPy returns quaternions in (x, y, z, w) order.
    try:
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()
    except ValueError as e:
        # Handle cases where the rotation matrix is not valid (e.g., contains shear)
        print(f"Error converting rotation matrix to quaternion: {e}")
        return None, None

    return translation, quaternion

def batch_transform_matrix_to_vectors(matrices, wxyz=True):
    pos = matrices[..., :3, 3].copy()
    pos.flags["WRITEABLE"] = True
    r = matrices[..., :3, :3].copy()
    r.flags["WRITEABLE"] = True
    quat = R.from_matrix(r).as_quat() # xyzw
    if wxyz:
        quat = quat[..., [3, 0, 1, 2]]
    return pos, quat

def batch_forward_kinematics(scene, joint_names=None, configuration=None, edge_batch=None, env_ids=None):
    """Helper function to update the scene graph by running forward kinematics on all edges specified through joint_names.

    Args:
        scene (trimesh.Scene): Scene.
        joint_names (list[str], optional): Names of joints to update. If None, use all joints. Defaults to None.
        configuration (list[NDArray], optional): Configurations. Needs to be same size as joint_names or None. If None will use configuration stored in graph. Defaults to None.
    """
    if configuration is not None and joint_names is not None:
        if len(configuration.shape) == 1:
            assert len(joint_names) == len(configuration), f"Length of {configuration} != {joint_names}"
        else:
            assert configuration.shape[1] == len(joint_names), f"Length of {configuration} != {joint_names}"

    # Create joint map
    scene_edge_data = scene.graph.transforms.edge_data
    joint_map = {}
    for k in scene.graph.transforms.edge_data:
        edge_data = scene_edge_data[k]
        if (
            EDGE_KEY_METADATA in edge_data
            and edge_data[EDGE_KEY_METADATA] is not None
            and "joint" in edge_data[EDGE_KEY_METADATA]
        ):
            joint_data = edge_data[EDGE_KEY_METADATA]["joint"]
            joint_map[joint_data["name"]] = k

    if joint_names is None:
        joint_names = joint_map.keys()

    for i, joint_name in enumerate(joint_names):
        matrix = None
        edge_data = scene_edge_data[joint_map[joint_name]]
        joint_data = edge_data[EDGE_KEY_METADATA]["joint"]

        if joint_data["type"] == "fixed" or joint_data["type"] == "floating":
            matrix = None
            if 'origin' in joint_data:
                matrix = np.array(joint_data.get("origin"))
        elif joint_data["type"] == "revolute" or joint_data["type"] == "continuous":
            if configuration is None:
                matrix = (
                    joint_data["origin"]
                    @ tra.rotation_matrix(joint_data["q"], joint_data["axis"])
                )
            else:
                matrix = (
                    joint_data["origin"]
                    @ batch_rotation_matrix(configuration[:, i], joint_data["axis"])
                )
                joint_data["q"] = configuration[:, i]
        elif joint_data["type"] == "prismatic":
            if configuration is None:
                matrix = (
                    joint_data["origin"]
                    @ tra.translation_matrix(joint_data["q"] * np.asarray(joint_data["axis"]))
                )
            else:
                direction = np.tile(
                    np.asarray(joint_data["axis"])[np.newaxis, :], (configuration.shape[0], 1))
                matrix = (
                    joint_data["origin"]
                    @ batch_translation_matrix(configuration[:, i:i+1] * direction)
                )
                joint_data["q"] = configuration[:, i]

        if matrix is not None:
            if env_ids is None or env_ids[0] == 0:
                if len(matrix.shape) == 2:
                    edge_data["matrix"] = matrix
                else:
                    edge_data["matrix"] = matrix[0]

            if env_ids is None:
                edge_batch[joint_map[joint_name]] = matrix
            else:
                if joint_map[joint_name] not in edge_batch:
                    edge_batch[joint_map[joint_name]] = np.zeros((len(env_ids), 4, 4))
                edge_batch[joint_map[joint_name]][env_ids] = matrix

    invalidate_scenegraph_cache(scene)


def invalidate_scenegraph_cache(scene):
    """Invalidate cache of scene's graph.

    Args:
        scene (trimesh.Scene): The scene.
    """
    scene.graph._modified = str(uuid.uuid4())
    scene.graph.transforms._modified = str(uuid.uuid4())
    scene.graph._cache.clear()
    scene.graph.transforms._cache.clear()
    scene.cache = {}


def create_ring_polygon(
    center: NDArray, 
    inner_r: float, 
    outer_r: float,
    dir_start: NDArray | None = None,
    dir_end: NDArray | None = None, 
    split: int = 36
) -> Polygon:
    """Create a polygon that approximates a ring.

    Args:
        center (NDArray): shape (2,), the center of the ring.
        inner_r (float): the inner radius of the ring.
        outer_r (float): the outer radius of the ring.
        dir_start (NDArray | None, optional): shape (2,), the start direction of the ring. Defaults to None.
        dir_end (NDArray | None, optional): shape (2,),the end direction of the ring. Defaults to None.
        split (int, optional): controls how many points along the circle to approximate the ring. Defaults to 36.

    Returns:
        Polygon: ring polygon
    """
    shell, hole = [], []
    if dir_start is None and dir_end is None:
        rad = np.linspace(0, np.pi*2, split)
        shell = center + outer_r * np.stack([np.cos(rad), np.sin(rad)], axis=1)
        hole = center + inner_r * np.stack([np.cos(rad), np.sin(rad)], axis=1)
        shell = np.concatenate([shell, shell[0:1]], axis=0)
        hole = np.concatenate([hole, hole[0:1]], axis=0)
        polygon = Polygon(shell=shell, holes=[hole])
    else:
        dir_start = normalize_direction_vector(dir_start)
        dir_end = normalize_direction_vector(dir_end)
        rad_start = np.arctan2(dir_start[1], dir_start[0])
        rad_end = np.arctan2(dir_end[1], dir_end[0])
        rad = np.linspace(rad_start, rad_end, split)
        outer_shell = center + outer_r * np.stack([np.cos(rad), np.sin(rad)], axis=1)
        inner_shell = center + inner_r * np.stack([np.cos(rad), np.sin(rad)], axis=1)
        shell = np.concatenate([outer_shell, inner_shell[::-1]], axis=0)
        polygon = Polygon(shell=shell)
    return polygon


def normalize_direction_vector(direction: NDArray) -> NDArray:
    """Normalize a direction vector

    Args:
        direction (NDArray): shape (2,), the direction vector to normalize.

    Returns:
        NDArray: shape (2,), the normalized direction vector.
    """
    if not np.isclose(np.linalg.norm(direction), 1.0):
        direction = direction / (np.linalg.norm(direction) + 1e-5)
    return direction


def create_region_polygon(
    center, 
    dir_start: NDArray, 
    dir_end: NDArray, 
    distance_start: float | None = None,
    distance_end: float | None = None,
    size: float = 100.0,
) -> Polygon:
    """Create a polygon that approximates a region.

    Args:
        center (NDArray): shape (2,), the center of the region.
        dir_start (NDArray): shape (2,), the start direction of the region.
        dir_end (NDArray): shape (2,), the end direction of the region.
        distance_start (float | None, optional): the start distance of the region. Defaults to None.
        distance_end (float | None, optional): the end distance of the region. Defaults to None.
        size (float, optional): the size of the region. Defaults to 100.0.

    Returns:
        Polygon: region polygon
    """
    dir_start = normalize_direction_vector(dir_start)
    dir_end = normalize_direction_vector(dir_end)
    if distance_start is None and distance_end is None:
        if size is None:
            raise ValueError("Either size or distance_start and distance_end must be provided")
    if distance_start is not None and distance_end is not None:
        size = distance_end - distance_start
    
    if distance_start is None:
        distance_start = 0.0
    if distance_end is None:
        distance_end = distance_start + size
    
    if np.isclose(np.dot(dir_start, dir_end), -1.0):
        mid_dir = ROT_2D_90 @ dir_end
    else:
        mid_dir = normalize_direction_vector((dir_start + dir_end) / 2)
    
    if np.isclose(np.dot(mid_dir, dir_start), -1.0):
        vertices = np.stack([
            center + mid_dir * distance_start,
            center + mid_dir * distance_start + dir_start * size,
            center + mid_dir * distance_start + (dir_start + mid_dir) * size,
            center + mid_dir * distance_start + (dir_end + mid_dir) * size,
            center + mid_dir * distance_start + dir_end * size,
            center + mid_dir * distance_start
        ])
    else:
        vertices = np.stack([
            center + mid_dir * distance_start, 
            center + mid_dir * distance_start + dir_start * size, 
            center + mid_dir * distance_start + dir_end * size, 
            center], axis=0)
    return Polygon(vertices)


def create_polygon_from_unordered_points(points: NDArray) -> Polygon:
    """Create a polygon from unordered points.

    Args:
        points (NDArray): shape (N, 2), the unordered points.

    Returns:
        Polygon: polygon
    """
    mean = np.mean(points, axis=0)
    vec = points - mean
    rad = np.arctan2(vec[:, 1], vec[:, 0])
    indices = np.argsort(rad)
    points = points[:][indices]
    points = np.concatenate([points, points[0:1]], axis=0)
    return Polygon(points)

def signed_angle(v1: NDArray, v2: NDArray):
    """
    Calculates the signed angle between two 2D vectors or batches of 2D vectors.

    Args:
        v1: The first vector(s) as a NumPy array. Shape can be either (2,) for a single vector
            or (N, 2) for a batch of N vectors.
        v2: The second vector(s) as a NumPy array. Shape can be either (2,) for a single vector
            or (N, 2) for a batch of N vectors.

    Returns:
        The oriented angle(s) in radians. Returns a single float for single vectors,
        or an array of shape (N,) for batched vectors.
    """
    # Ensure inputs are numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # Handle both single vectors and batched vectors
    if v1.ndim == 1:
        x1, y1 = v1
        x2, y2 = v2
        angle = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    else:
        # For batched vectors, v1 and v2 should have shape (N, 2)
        x1, y1 = v1[:, 0], v1[:, 1]
        x2, y2 = v2[:, 0], v2[:, 1]
        angle = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    
    return angle


ROT_2D_45 = np.array(
    [[np.cos(np.pi/4), -np.sin(np.pi/4)],
    [np.sin(np.pi/4), np.cos(np.pi/4)]]
)

ROT_2D_90 = np.array(
    [[0, -1],
    [1, 0]]
)

ROT_2D_270 = np.array(
    [[0, 1],
    [-1, 0]]
)

ROT_2D_315 = np.array(
    [[np.cos(7*np.pi/4), -np.sin(7*np.pi/4)],
    [np.sin(7*np.pi/4), np.cos(7*np.pi/4)]]
)

def angle_to_2d_rotation_matrix(angle: float) -> NDArray:
    """Convert an angle to a 2D rotation matrix.

    Args:
        angle (float): the angle to convert, in degrees

    Returns:
        NDArray: shape (2, 2), the rotation matrix.
    """
    angle = np.deg2rad(angle)
    return np.array(
        [[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]]
    )


def resolve_multi_polygon(polygon: MultiPolygon) -> list[Polygon]:
    """Reduce a multi-polygon to a list of polygons.

    Args:
        polygon (MultiPolygon): the multi-polygon to reduce.

    Returns:
        list[Polygon]: the list of polygons.
    """
    return list(
            itertools.chain(
                [resolve_multi_polygon(p) if isinstance(p, MultiPolygon) else p for p in polygon.geoms]
            )
        )


def get_mesh_bbox_corners(mesh_vertices: NDArray) -> NDArray:
    """Get the corners of the bounding box of a mesh.

    Args:
        mesh_vertices (NDArray): shape (N, 3), the mesh vertices.

    Returns:
        NDArray: shape (8, 3), the corners of the bounding box.
    """
    min_x = np.min(mesh_vertices[:, 0])
    max_x = np.max(mesh_vertices[:, 0])
    min_y = np.min(mesh_vertices[:, 1])
    max_y = np.max(mesh_vertices[:, 1])
    min_z = np.min(mesh_vertices[:, 2])
    max_z = np.max(mesh_vertices[:, 2])
    return np.array([
        [min_x, min_y, min_z], 
        [max_x, min_y, min_z], 
        [min_x, max_y, min_z], 
        [max_x, max_y, min_z], 
        [min_x, min_y, max_z], 
        [max_x, min_y, max_z], 
        [min_x, max_y, max_z], 
        [max_x, max_y, max_z]]
    )


def get_min_xy_edge(bbox_corners: NDArray) -> float:
    """Get the minimum edge length of the bounding box.

    Args:
        bbox_corners (NDArray): shape (8, 3), the corners of the bounding box.

    Returns:
        float: the minimum edge length of the bounding box.
    """
    return min([
        abs(bbox_corners[1, 0] - bbox_corners[0, 0]),
        abs(bbox_corners[2, 1] - bbox_corners[0, 1]),
    ])


def get_edge_point_from_bbox_corners(bbox_corners: NDArray, axis: int, loc: str) -> float:
    """Get the point on the edge of the bounding box.

    Args:
        bbox_corners (NDArray): shape (8, 3), the corners of the bounding box.
        axis (int): the axis to get the point on. 0: x, 1: y, 2: z.
        loc (str): the location of the point on the edge.

    Returns:
        float: the point on the edge of the bounding box.
    """
    if loc == "min":
        return np.min(bbox_corners[:, axis])
    elif loc == "center":
        return np.mean(bbox_corners[:, axis])
    elif loc == "max":
        return np.max(bbox_corners[:, axis])
    else:
        raise ValueError(f"Invalid location: {loc}")


def visualize_mesh(mesh: NDArray, mesh2: NDArray = None):
    """Visualize a mesh.

    Args:
        mesh (NDArray): shape (N, 3), the mesh vertices.
    """
    import trimesh
    if mesh2 is not None:
        mesh[..., 2] += 0.01
        # offset mesh on z so points are not overlapping with mesh2
    pc1 = trimesh.PointCloud(mesh, colors=np.array([[255, 0, 0, 255]]*len(mesh), dtype=np.uint8))
    pc2 = trimesh.PointCloud(mesh2, colors=np.array([[0, 0, 255, 255]]*len(mesh2), dtype=np.uint8))
    trimesh.Scene([pc1, pc2]).show()


def visualize_polygon(polygon: Polygon):
    """Visualize a polygon.

    Args:
        polygon (Polygon): the polygon to visualize.
    """
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon
    fig, ax = plt.subplots()
    plot_polygon(polygon, ax=ax)
    plt.show()

def visualize_polygons(polygons: list[Polygon]):
    """Visualize a list of polygons.

    Args:
        polygons (list[Polygon]): the list of polygons to visualize.
    """
    import matplotlib.pyplot as plt
    from shapely.plotting import plot_polygon
    fig, ax = plt.subplots()
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray", "black"]
    for i, polygon in enumerate(polygons):
        plot_polygon(polygon, ax=ax, color=colors[i])
    plt.show()


def batch_face_to_ori(obj_to_parent_pos2d: NDArray) -> NDArray:
    """Batch face to orientation. By default it aligns (0, -1) to the direction `obj_to_parent_pos2d`

    Args:
        obj_to_parent_pos2d (NDArray): shape (N, 2), the object to parent position.

    Returns:
        NDArray: shape (N, 4, 4), the orientation.
    """
    direction = normalize_direction_vector(obj_to_parent_pos2d)

    rotation_matrices = np.eye(4)[np.newaxis, :, :].repeat(direction.shape[0], axis=0)

    # Fill in the top-left 2x2 part with the rotation values
    rotation_matrices[:, 0, 0] = -direction[:, 1]
    rotation_matrices[:, 0, 1] = -direction[:, 0]
    rotation_matrices[:, 1, 0] = direction[:, 0]
    rotation_matrices[:, 1, 1] = -direction[:, 1]

    return rotation_matrices
