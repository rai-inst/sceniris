import numpy as np
from numpy.typing import NDArray
import trimesh
import trimesh.transformations as tra

from scene_synthesizer.assets import Asset as _Asset
from scene_synthesizer.assets import (
    URDFAsset as _URDFAsset, 
    USDAsset as _USDAsset, 
    MJCFAsset as _MJCFAsset, 
    OpenSCADAsset as _OpenSCADAsset, 
    MeshAsset as _MeshAsset,
    TrimeshAsset as _TrimeshAsset
)

from utils import sample_random_z_rotations
from scene_synthesizer.utils import log

class Asset(_Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str) or "fname" in kwargs:
            fname = kwargs["fname"] if "fname" in kwargs else args[0]
            fname = fname.lower()
            if fname.endswith(".urdf"):
                return super(Asset, cls).__new__(URDFAsset)
            elif (
                fname.endswith(".usd")
                or fname.endswith(".usda")
                or fname.endswith(".usdc")
                or fname.endswith(".usdz")
            ):
                return super(Asset, cls).__new__(USDAsset)
            elif fname.endswith(".xml"):
                return super(Asset, cls).__new__(MJCFAsset)
            elif fname.endswith(".scad"):
                return super(Asset, cls).__new__(OpenSCADAsset)
            else:
                return super(Asset, cls).__new__(MeshAsset)
        return super(Asset, cls).__new__(cls)

    def sample_stable_pose(
        self, 
        seed = None, 
        num_samples: int =1, 
        z_rotation: bool=True,
        **kwargs
    ) -> NDArray:
        """Return a stable pose according to their likelihood.

        seed (int, numpy.random._generator.Generator, optional): A seed or random number generator. Defaults to None which creates a new default random number generator.
        num_samples (int): The number of samples to return.
        z_rotation (bool): Whether to rotate the poses around the z-axis.
        **kwargs: Additional arguments to pass to the `compute_stable_poses` method.

        Returns:
            NDArray: batch of homogeneous 4x4 matrix (N, 4, 4)

        """
        rng = np.random.default_rng(seed)

        if not hasattr(self, "_stable_poses") or self._stable_poses is None:
            self.compute_stable_poses(**kwargs)

        poses, probabilities = self._stable_poses
        if len(poses) == 0:
            raise RuntimeError(f"Unable to detect any stable poses for {self}.")
        probabilities = np.array(probabilities) / sum(
            probabilities
        )

        indices = rng.choice(len(poses), num_samples, p=probabilities)
        if z_rotation:
            inplane_rot = sample_random_z_rotations(num_samples=num_samples, seed=seed)
            return np.einsum('bmn,bnk->bmk', inplane_rot, poses[indices])
        else:
            return poses[indices]
        
    def _filter_collision_geometry(self, node_names):
        filtered_node_names = []
        for name in node_names:
            if "visuals" in name:
                if name.replace("visuals", "collisions") in node_names:
                    continue
            elif "visual" in name:
                if name.replace("visual", "collision") in node_names:
                    continue
            filtered_node_names.append(name)
        return filtered_node_names
        
    @property
    def collision_node_names(self):
        scene = self.as_trimesh_scene(use_collision_geometry=True)
        node_names = scene.graph.nodes_geometry
        return self._filter_collision_geometry(node_names)

    def node_named_geometries(self, use_collision_geometry=True):
        # don't know why this use_collision_geometry is not working in scene_synthesizer
        # so we need to filter the collision geometry manually
        scene = self.as_trimesh_scene(use_collision_geometry=use_collision_geometry)
        node_name_T_mesh_list = []
        node_names = scene.graph.nodes_geometry
        if use_collision_geometry:
            node_names = self._filter_collision_geometry(node_names)
        for node_name in node_names:
            T, geom_name = scene.graph.get(node_name)
            mesh = scene.geometry[geom_name]
            node_name_T_mesh_list.append((node_name, T, mesh))
        return node_name_T_mesh_list


class URDFAsset(_URDFAsset, Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class USDAsset(_USDAsset, Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MJCFAsset(_MJCFAsset, Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class OpenSCADAsset(_OpenSCADAsset, Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class MeshAsset(_MeshAsset, Asset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlaneAsset(_TrimeshAsset):
    def __init__(
        self,
        width=1,
        depth=1,
        normal=(0, 0, 1),
        center=(0, 0, 0),
    ):
        """An asset representing a plane. Can be used to model a ground plane, often used in simulators.

        Note: During USD export, only a square plane will be considered with the length of min(width, depth). Also normals are restricted to
              'X', 'Y', 'Z', and their negative counterparts.

        Args:
            width (float, optional): Width of the plane. Defaults to 1.
            depth (float, optional): Depth of the plane. Defaults to 1.
            normal (tuple[float]): A 3-dimensional unit vector of the normal of the plane. Defaults to (0, 0, 1), ie., pointing in the "z" direction.
            center (tuple[float]): A 3-dimensional vector of the origin of the plane. Defaults to (0, 0, 0).
        """
        if np.linalg.norm(normal) != 1.0:
            raise ValueError(f"Normal vector of plane {normal} needs to have unit length.")

        corners = [
            (-width / 2.0, -depth / 2.0, 0),
            (width / 2.0, -depth / 2.0, 0),
            (-width / 2.0, depth / 2.0, 0),
            (width / 2.0, depth / 2.0, 0),
        ]
        transform = trimesh.geometry.align_vectors((0, 0, 1), normal)
        transform[:3, 3] = center

        vertices = tra.transform_points(corners, transform) 
        faces = [[0, 1, 3, 2]]
        plane = trimesh.Trimesh(vertices=vertices, faces=faces)

        super().__init__(plane)