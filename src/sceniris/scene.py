from typing import Any

import itertools
import os
import re
import time
import logging
import networkx as nx
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

import trimesh
import trimesh.transformations as tra
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.geom.types import Mesh, WorldConfig
from curobo.geom.sdf.world import (
    WorldCollisionConfig, 
    CollisionQueryBuffer
)
from curobo.util.logger import setup_curobo_logger
setup_curobo_logger("error")
from curobo.geom.sdf.world_mesh import WorldMeshCollision

from scene_synthesizer import Scene as _Scene
from scene_synthesizer.scene import SupportSurface, Container
from scene_synthesizer.assets import PlaneAsset
from scene_synthesizer import utils
try:
    from pyglet.app import run as _pyglet_app_run
except BaseException as E:
    _pyglet_app_run = utils.late_bind_exception(E)

from sceniris.pose_generators import (
    OrientationGeneratorConst,
    OrientationGeneratorStablePoses,
    OrientationGeneratorUniformAroundZ,
    PositionIteratorNone,
    PositionIteratorUniform, 
    PositionIterator2DCollection,
)
from sceniris.utils import (
    point_to_translation_matrix,
    get_support_transforms,
    get_support_node_names,
    get_transform_batch,
    scene_graph_transform_get,
    make_mesh_buffer,
    batch_forward_kinematics,
    batch_transform_matrix_to_vectors,
    invalidate_scenegraph_cache,
    homogeneous_inv_batch,
    visualize_mesh,
)

from sceniris.constraints import SurfaceRelation, TrackingTransform
from sceniris.asset import Asset

logger = logging.getLogger("sceniris")

class Scene(_Scene):
    def __init__(
        self, 
        num_envs: int = 128, 
        env_size: float = 1.0, 
        robot_centric_frame_transforms = None, 
        collision_checker_backend: str = "curobo",
        tmp_mesh_file_folder: str = "/tmp/sceniris",
        workspace_limits: list[list[float]] | None = None,
        cfg: dict[str, Any] = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._cfg = cfg
        if cfg is not None:
            if isinstance(cfg["objects"], list):
                cfg["objects"] = {obj["id"]: obj for obj in cfg["objects"]}
        self.usd_scene = None
        self.obj_available_cache = {}
        self.obj_pose_scenes = {}
        self.num_envs = num_envs
        self.env_size = env_size
        self.workspace_limits = workspace_limits
        self.assets = {}
        self.edge_batch = {}
        self.joint_states = defaultdict(dict) # dict: obj_id -> dict: joint_id (f"{obj_id}/{joint_name}") -> value
        self.cache = {}
        self._asset_mesh_cache = {}
        self.isaac_env = None
        self.robot_centric_frame_transforms = robot_centric_frame_transforms
        self.collision_checker_backend = collision_checker_backend
        self.tmp_mesh_file_folder = tmp_mesh_file_folder
        self._object_enabled = {}
        self._all_mesh_names = defaultdict(list) # dict: obj_id -> mesh_names <list[str]>
        self.valid_env_mask = np.ones((self.num_envs,), dtype=np.bool_)
        self._no_relationship_scene = trimesh.Scene()
        self._plane_asset = None
        self._curobo_mesh_sphere_cache = {}
        self.CUROBO_SPHERE_APPROX_N = 200
        self._reachability_checker = None

    def _init_reachability_checker(self):
        """Initialize the reachability checker if not already done."""
        if self._reachability_checker is None:
            try:
                from sceniris.reachability_checker import ReachabilityChecker
                self._reachability_checker = ReachabilityChecker()
                print("Reachability checker initialized")
            except Exception as e:
                print(f"Warning: Could not initialize reachability checker: {e}")
                self._reachability_checker = None

    def collision_check(
        self, 
        query_obj_id: str, 
        query_obj_asset, 
        query_obj_T: NDArray, 
        env_ids: torch.Tensor | None = None,
        query_obj_joint_configs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        Collision check execution.

        Args:
            query_obj_id (str): The id of the object, "query object", to check if there is any collision with the rest of the scene.
            query_obj_asset (trimesh.Scene): The asset of the object.
            query_obj_T (NDArray): the obj_to_world transform of the object.
            env_ids (torch.Tensor | None, optional): The env ids to check collision with. Defaults to None (all envs).
            query_obj_joint_configs (dict[str, Any] | None, optional): 
                The joint configs of the query object.

        Returns:
            torch.Tensor: (N,) bool tensor, where N is the number of envs. True means there is collision.
        """
        tensor_args = TensorDeviceType()
        world_configs = []

        # if curobo world ccheck is available, use it (used by generation from cfg)
        if hasattr(self, "_curobo_world_ccheck") and self._curobo_world_ccheck is not None:
            # if no object is enabled, return all False (no collision)
            if sum(self._object_enabled.values()) == 0:
                return torch.zeros(len(env_ids), dtype=torch.bool)
            self._update_curobo_world_ccheck(update_transforms=True)
            world_ccheck = self._curobo_world_ccheck
        # otherwise, create it every time from scratch (used by step by step scene generation))
        else:
            if len(self._asset_mesh_cache) < 1:
                return torch.zeros(len(env_ids), dtype=torch.bool)

            # calculate mesh_transform for all envs to save a bit time
            mesh_transforms = {}
            # TODO: maybe replace self._asset_mesh_cache with object ids being considered for current collision check
            for obj_id, mesh_paths in self._asset_mesh_cache.items():
                for mesh_path in mesh_paths:
                    mesh_node_name = os.path.basename(mesh_path).replace(".stl", "").replace("___", "/")
                    mesh_transform = scene_graph_transform_get(
                        self._scene.graph, mesh_node_name, edge_batch=self.edge_batch, cache=self.cache)[0]
                    mesh_transforms[mesh_node_name] = mesh_transform
            
            # will be replaced by curobo world from USD
            for env_id in env_ids:
                meshes = []
                # TODO replace it with all [collision] geometry in the scene
                for obj_id, mesh_paths in self._asset_mesh_cache.items():
                    for mesh_path in mesh_paths:
                        # recover node_name from mesh_path
                        mesh_node_name = os.path.basename(mesh_path).replace(".stl", "").replace("___", "/")
                        mesh_transform = mesh_transforms[mesh_node_name]
                        if len(mesh_transform.shape) == 3:
                            mesh_transform = mesh_transform[env_id]
                        pose = Pose.from_matrix(mesh_transform.copy())
                        pose = torch.cat([pose.position, pose.quaternion], dim=1)[0].cpu()
                        # create from Mesh, use mesh's pose (not the inverse)
                        mesh = Mesh(
                            name=mesh_node_name,
                            pose=pose,
                            file_path=mesh_path,
                        )
                        meshes.append(mesh)
                world_configs.append(WorldConfig(mesh=meshes))
            world_coll_config = WorldCollisionConfig(
                tensor_args, world_model=world_configs
            )
            world_ccheck = WorldMeshCollision(world_coll_config)

        # prepare query object
        if len(query_obj_T.shape) == 2:
            query_obj_T = np.tile(query_obj_T, (len(env_ids), 1, 1))
        query_obj_pose = torch.from_numpy(query_obj_T).to(torch.float32)

        query_obj_scene = query_obj_asset.as_trimesh_scene(use_collision_geometry=True)
        query_obj_edge_batch = {}
        # update joint states to get local transforms
        if query_obj_joint_configs is not None:
            batch_forward_kinematics(
                query_obj_scene, 
                joint_names=query_obj_joint_configs["joint_names"], 
                configuration=query_obj_joint_configs["configuration"],
                edge_batch=query_obj_edge_batch
            )
        query_node_name_T_mesh_list = query_obj_asset.node_named_geometries(use_collision_geometry=True)
        default_mesh_folder = "/tmp/sgv2"
        make_mesh_buffer(query_obj_id, query_obj_asset, default_folder=default_mesh_folder)
        N_SPH_BASE = 200
        mesh_spheres = []
        for node_name, T, mesh in query_node_name_T_mesh_list:
            fn = node_name.replace("object/", f"{query_obj_id}/") # triple _ to seperate obj_id and node_name
            # mesh_file_path = os.path.join(default_mesh_folder, f"{fn}.stl")
            # query_mesh = Mesh(
            #     name=node_name,
            #     pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            #     file_path=mesh_file_path,
            # )
            query_sph_approx = self._curobo_mesh_sphere_cache[query_obj_id][fn]

            # mesh_height = mesh.bounds[1, 2] - mesh.bounds[0, 2]
            # mesh is in original frame
            # the origin is reflected in transform in the scene graph (local_T below)
            local_T = scene_graph_transform_get(
                query_obj_scene.graph, node_name, edge_batch=query_obj_edge_batch, cache={}
            )[0].copy() # (N, 4, 4) or (4, 4)

            local_T.flags["WRITEABLE"] = True
            local_T = torch.from_numpy(local_T).to(torch.float32)
            local_T = query_obj_pose @ local_T # (N, 4, 4) because query_obj_pose is (N, 4, 4)

            n_sph = N_SPH_BASE
            # query_sph_approx = query_mesh.get_bounding_spheres(n_spheres=n_sph)
            n_sph = len(query_sph_approx)
            sph_pos = torch.stack([torch.tensor(s.position, dtype=torch.float32) for s in query_sph_approx], dim=0) # (n_sph, 3)
            # print ("sph_pos z bound", sph_pos[:, 2].min(), sph_pos[:, 2].max())

            # -------------
            # TODO: <jshang> important note:
            # for some reason the spheres are using the center of the mesh as the origin, not the same as specified
            # in mesh (center, center, bottom). So an additional offset on z is needed to raise the spheres up
            # otherwise it always collides with the support surface
            # local_T is supposed to do this but it doesn't work for some reason
            # sph_pos[..., 2] += mesh_height / 2
            # -------------
            sph_pos_homo = torch.cat([sph_pos, torch.ones((n_sph, 1), dtype=torch.float32)], dim=1) # (n_sph, 4)
            sph_pos_homo = sph_pos_homo.unsqueeze(0).unsqueeze(-1) # (1, n_sph, 4, 1)
            sph_pos_homo = local_T.unsqueeze(1) @ sph_pos_homo # (N, n_sph, 4, 1)
            sph_pos = sph_pos_homo[..., :3, 0] # (N, n_sph, 3)
            # print ("sph_pos after local_T z bound", sph_pos[..., 2].min(), sph_pos[..., 2].max())
            # sph_pos = sph_pos.unsqueeze(0) + local_T[:, :3, 3].unsqueeze(1)      # (N, n_sph, 3)
            sph_radius = torch.stack([torch.tensor([s.radius], dtype=torch.float32) for s in query_sph_approx], dim=0)\
                        .unsqueeze(0).repeat(len(env_ids), 1, 1)                 # (N, n_sph, 1)
            sph = tensor_args.to_device(torch.cat([sph_pos, sph_radius], dim=2)) # (N, n_sph, 4)
            sph = sph.unsqueeze(1) # (N, horizon(1), n_sph, 4)
            mesh_spheres.append(sph)
        
        # stack all parts' spheres
        mesh_spheres = torch.cat(mesh_spheres, dim=2) # (N, horizon, n_sph * n_parts, 4)

        # query curobo
        collision_query_buffer = CollisionQueryBuffer.initialize_from_shape(
            mesh_spheres.shape, tensor_args, world_ccheck.collision_types
        )
        act_distance = tensor_args.to_device([0.0])
        weight = tensor_args.to_device([1])
        if hasattr(self, "_curobo_world_ccheck") and self._curobo_world_ccheck is not None:
            env_query_idx = env_ids.to(device=tensor_args.device, dtype=torch.int32)
        else:
            env_query_idx = torch.arange(mesh_spheres.shape[0], device=tensor_args.device, dtype=torch.int32)
        d = world_ccheck.get_sphere_distance(
            mesh_spheres, collision_query_buffer, weight, act_distance, env_query_idx=env_query_idx
        )
        # for i in range(d.size(0)):
        #     colliding_mesh = mesh_spheres[i, 0, torch.nonzero(d[i, 0] > 0).flatten(), :3].cpu().numpy() #(n_colliding, 3)
        #     visualize_mesh(colliding_mesh, mesh_spheres[i, 0, :, :3].cpu().numpy())
        # gather results by collapsing horizon (dim1) and n_sph*n_parts (dim2)
        return torch.any(d > 0, dim=(1,2)).cpu() # (N,)

    def place_objects(
        self,
        obj_id_iterator,
        obj_asset_iterator,
        obj_position_iterator,
        env_obj_position_iterator,
        obj_orientation_iterator,
        parent_id=None,
        max_iter=10,
        max_iter_per_support=10,
        distance_above_support=0.005,
        joint_type="floating",
        valid_placement_fn=lambda obj_asset, support, placement_T: True,
        debug=False,
        constraints = [],
        joint_states = None,
        check_reachability=False,
        **kwargs,
    ):
        """
        This is modified from scene_synthesizer's `place_objects`.
        Add objects and place them in a non-colliding pose on top of a support surface or inside a container.

        Args:
            obj_id_iterator (iterator): Iterator for sampling name of the object to place.
            obj_asset_iterator (iterator): Iterator for sampling asset to be placed.
            obj_position_iterator (iterator, optional): Iterator for sampling object positions in the support frame.
            env_obj_position_iterator (iterator, optional): per-env object position iterator. This is used to accomodate
                complex object placement constraints where per-env object position iterators are created.
            obj_orientation_iterator (iterator, optional): Iterator for sampling object orientation in the object asset frame.
            parent_id (str): Name of the node in the scene graph at which to attach the object. 
                Or None if same as support node. Defaults to None.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            max_iter_per_support (int, optional): Not used. All supports are jointly sampled in the underlying 2D position sampler.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.002.
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). 
                None has a similar effect as "fixed". Defaults to "floating".
            valid_placement_fn (function, optional): Not used.
            debug (bool, optional): Not used.
            check_reachability (bool, optional): Whether to check reachability of the object. Defaults to False.

            **use_collision_geometry (bool, optional): Defaults to default_use_collision_geometry.
            **kwargs: Keyword arguments that will be delegated to add_object.

        Returns:
            ndarray: env ids wher objects are **not** successfully placed.
        """
        use_collision_geometry = kwargs.pop('use_collision_geometry', self._default_use_collision_geometry)
        for obj_id, obj_asset, in zip(obj_id_iterator, obj_asset_iterator):
            if isinstance(obj_position_iterator, dict):
                position_iterator = obj_position_iterator[obj_id]
            else:
                position_iterator = obj_position_iterator

            if isinstance(obj_orientation_iterator, dict):
                orientation_iterator = obj_orientation_iterator[obj_id]
            else:
                orientation_iterator = obj_orientation_iterator

            if env_obj_position_iterator is not None:
                if isinstance(env_obj_position_iterator, dict):
                    env_position_iterator = env_obj_position_iterator[obj_id]
                else:
                    env_position_iterator = env_obj_position_iterator
            else:
                env_position_iterator = None

            if parent_id is None:
                parent_id = "world"

            joint_ids = []
            joint_value_ranges = []
            # is joint_states is None, the object will use default joint states
            if joint_states is not None and len(joint_states) > 0:
                for js in joint_states:
                    jids = js.get("joint_ids", [])
                    if len(jids) == 0:
                        continue
                    real_joint_ids = [f"{obj_id}/{jid}" for jid in jids]
                    limits = self.get_joint_limits(joint_ids=real_joint_ids) # (num_joints, 2)
                    max_val = js.get("max", 1.0) * (limits[:, 1]-limits[:, 0])+limits[:, 0]
                    min_val = js.get("min", 0.0) * (limits[:, 1]-limits[:, 0])+limits[:, 0]
                    val_range = np.stack([min_val, max_val], axis=1)
                    joint_ids.extend(real_joint_ids)
                    joint_value_ranges.append(val_range)
                self.joint_states[obj_id] = {
                    jid: np.zeros((self.num_envs,), np.float32)
                    for jid in joint_ids
                }
            # joint_value_ranges = np.concatenate(joint_value_ranges, axis=0) # (num_joints, 2)

            iter = 0
            env_ids: torch.Tensor = torch.arange(self.num_envs, dtype=torch.int32)
            edge_key = (parent_id, obj_id)
            while iter < max_iter:
                n_working_envs = len(env_ids)
                if "fixed_world_positions" in kwargs:
                    #TODO: <jshang> handle fixed_world_positions
                    pos = kwargs["fixed_world_positions"]
                else:
                    if env_position_iterator is not None and isinstance(env_position_iterator, list) and len(env_position_iterator) > 0:
                        # st = time.time()
                        pos_raw = {"samples": [], "support_refs": []}
                        working_env_ids = []
                        failed_env_ids = [] # env ids that failed to sample
                        for env_idx in env_ids:
                            if isinstance(env_position_iterator[env_idx], PositionIteratorNone):
                                failed_env_ids.append(env_idx)
                                continue
                            env_pos_raw = env_position_iterator[env_idx].sample(1)
                            if isinstance(env_pos_raw, dict): # PositionIteratorCollection
                                # skip invalid samples
                                if np.isnan(env_pos_raw["samples"]).any():
                                    failed_env_ids.append(env_idx)
                                    continue
                                pos_raw["samples"].append(env_pos_raw["samples"])
                                pos_raw["support_refs"].append(env_pos_raw["support_refs"])
                                logger.debug(f"env_pos_raw, {env_pos_raw}")
                            else: # PositionIterator2D or PositionIteratorNone
                                # skip invalid samples
                                if np.isnan(env_pos_raw).any():
                                    failed_env_ids.append(env_idx)
                                    continue
                                pos_raw["samples"].append(env_pos_raw)
                                pos_raw["support_refs"].append(np.array([env_position_iterator[env_idx].support])) # could be None
                            working_env_ids.append(env_idx)
                            # filter out nan pos from PositionIteratorNone (means no valid positions)
                        working_env_ids = torch.from_numpy(np.array(working_env_ids)).to(torch.int32)
                        n_working_envs = len(working_env_ids)
                        try:
                            pos_raw["samples"] = np.concatenate(pos_raw["samples"], axis=0)
                            pos_raw["support_refs"] = np.concatenate(pos_raw["support_refs"], axis=0)
                        except ValueError as e:
                            return env_ids
                    else:
                        pos_raw = position_iterator.sample(n_working_envs)
                        working_env_ids = env_ids[:]
                        failed_env_ids = []
                    
                # orientation (on normalized surface)
                ori = orientation_iterator.sample(n_working_envs)
                if len(ori.shape) == 2:
                    ori = np.tile(ori, (n_working_envs, 1, 1))
                
                if isinstance(pos_raw, dict):
                    pos = pos_raw["samples"]
                    support = pos_raw["support_refs"]
                else:
                    pos = pos_raw
                    support = position_iterator.support

                logger.debug(f"support, {support}")

                # To avoid collisions with the support surface
                pos3d = np.concatenate([pos, np.full((pos.shape[0], 1), distance_above_support)], axis=1) \
                    if pos.shape[-1] == 2 else pos  # normalized surface

                logger.debug(f"{obj_id} sampled pos, {pos3d}")

                # Transform plane coordinates into scene coordinates
                if isinstance(support, np.ndarray):
                    # transform 3D coordinate with respect to the normalized surface to mesh frame
                    support_transforms = get_support_transforms(support)
                    placement_T = support_transforms @ point_to_translation_matrix(pos3d) @ ori
                    support_node_names = get_support_node_names(support)
                    parent_to_support_node = get_transform_batch(
                        self, support_node_names, frame_from=parent_id) # parent -> mesh
                    # support_node_names already has the same length as working_env_ids, so no need to index
                elif isinstance(support, SupportSurface):
                    placement_T = support.transform @ point_to_translation_matrix(pos3d) @ ori
                    parent_to_support_node = scene_graph_transform_get(
                        self._scene.graph, 
                        support.node_name,
                        frame_from=parent_id, 
                        edge_batch=self.edge_batch, 
                        cache=self.cache)[0] # parent -> mesh
                    # only one support node, so need to index after getting transforms from all envs
                    if len(parent_to_support_node.shape) == 3:
                        parent_to_support_node = parent_to_support_node[working_env_ids]
                    support_node_names = [support.node_name]
                else:
                    raise ValueError(f"Invalid supports: {support}")

                logger.debug(f"placement_T before parent to support node, {placement_T[..., :3, 3]}")
                logger.debug(f"parent_to_support_node, {parent_to_support_node[..., :3, 3]}")
                placement_T = parent_to_support_node @ placement_T # parent -> mesh @ mesh -> obj
                logger.debug(f"placement T, {placement_T[..., :3, 3]}")
                
                if (parent_id is not None) and (parent_id in self._scene.graph.transforms.node_data):
                    world_to_parent = scene_graph_transform_get(
                        self._scene.graph, parent_id, edge_batch=self.edge_batch, cache=self.cache)[0]
                    if len(world_to_parent.shape) == 3:
                        world_to_parent = world_to_parent[working_env_ids]
                else:
                    world_to_parent = np.eye(4)
            
                world_T = world_to_parent @ placement_T # T_w_obj
                logger.debug(f"world T, {world_T[..., :3, 3]}")

                joint_values = []
                if joint_states is not None and len(joint_states) > 0:
                    for joint_group_idx, js in enumerate(joint_states):
                        distribution = js.get("distribution", "uniform")
                        if distribution == "uniform":
                            val = np.random.random( (n_working_envs, len(js["joint_ids"])) ) * \
                                (joint_value_ranges[joint_group_idx][:, 1] - joint_value_ranges[joint_group_idx][:, 0]) + \
                                 joint_value_ranges[joint_group_idx][:, 0] # (num_envs, num_joints)
                            joint_values.append(val)
                        else:
                            raise ValueError(f"Invalid distribution for joint states: {distribution}")
                
                if len(joint_values) > 0:
                    joint_values = np.concatenate(joint_values, axis=1) # (num_envs, num_joints)
                    self.update_configuration(
                        configuration=joint_values,
                        joint_ids=joint_ids,
                        env_ids=working_env_ids,
                    )
                    for joint_idx, jid in enumerate(joint_ids):
                        self.joint_states[obj_id][jid][working_env_ids] = joint_values[:, joint_idx]

                # Check custom validity function
                # disable for now

                # Check collisions
                has_collision = self.collision_check(obj_id, obj_asset, world_T, env_ids=working_env_ids)
                logger.debug(f"placing {obj_id} has collision, {has_collision.sum().item()}")

                has_reachability_issue = torch.zeros_like(has_collision, dtype=torch.bool)
                if check_reachability:
                    # Initialize reachability checker if needed
                    if self._reachability_checker is None:
                        self._init_reachability_checker()
                    
                    if self._reachability_checker is not None:
                        # Convert world_T to numpy for reachability checking
                        world_T_np = world_T.cpu().numpy() if torch.is_tensor(world_T) else world_T
                        
                        # Use optimal RM4D discretization: test exactly 36 angles (one per theta bin)
                        # This ensures COMPLETE coverage of all RM4D discrete orientation categories
                        
                        # Using new orientation threshold method - no need for manual theta generation
                        
                        # Use batch reachability checking with new orientation threshold method
                        if len(world_T_np.shape) == 3 and len(world_T_np) > 1:
                            # Batch check all poses using the new orientation threshold method
                            reachable_mask = np.zeros(len(world_T_np), dtype=bool)
                            positions = world_T_np[:, :3, 3]
                            distances = np.sqrt(np.sum(positions**2, axis=1))
                            
                            for i, pose in enumerate(world_T_np):
                                position = pose[:3, 3]
                                # print('position', position)
                                # exit()
                                try:
                                    # Use the new orientation threshold method (10% of orientations must be reachable)
                                    reachable_with_any_orientation = self._reachability_checker.reachability_map.is_position_reachable_with_orientation_threshold(
                                        position, threshold=0.1)
                                except (IndexError, ValueError):
                                    # Position is outside the map bounds
                                    reachable_with_any_orientation = False
                                
                                reachable_mask[i] = reachable_with_any_orientation
                                
                               
                        
                        # Convert to tensor and set unreachable poses
                        has_reachability_issue = torch.from_numpy(~reachable_mask).to(has_collision.device)

                retry_env_ids = working_env_ids[(has_collision==True).nonzero().flatten().cpu()]
                # Combine collision and reachability issues
                retry_env_ids_reachability = working_env_ids[(has_reachability_issue==True).nonzero().flatten().cpu()]
                retry_env_ids = torch.cat([retry_env_ids, retry_env_ids_reachability])
                retry_env_ids = torch.unique(retry_env_ids) 
                if len(failed_env_ids) > 0:
                    retry_env_ids = torch.cat([retry_env_ids, torch.from_numpy(np.array(failed_env_ids)).to(torch.int32)])
                
                if edge_key not in self.edge_batch:
                    self.edge_batch[edge_key] = np.eye(4)
                self.edge_batch[edge_key].flags["WRITEABLE"] = True
                if len(placement_T.shape) == 3:
                    if len(self.edge_batch[edge_key].shape) == 2:
                        self.edge_batch[edge_key] = np.tile(self.edge_batch[edge_key], (self.num_envs, 1, 1))
                    self.edge_batch[edge_key][working_env_ids] = placement_T
                else:
                    self.edge_batch[edge_key] = placement_T
                self.edge_batch[edge_key].flags["WRITEABLE"] = False

                env_ids = retry_env_ids
                if len(env_ids) == 0:
                    logger.debug(f"{obj_id} succesfully placed")
                    break
                logger.debug(f"{len(env_ids)} envs have collision, retrying")
                iter += 1
            
            if len(env_ids) > 0:
                logger.debug(f"after {iter} iterations, {len(env_ids)} envs are not valid")
                return env_ids.numpy() # return invalid env_ids

            logger.debug(f"Adding {obj_id} to {parent_id}")
            
            # use env0 transform to update trimesh scene
            trimesh_transform = self.edge_batch[edge_key][0] if len(self.edge_batch[edge_key].shape) == 3 else self.edge_batch[edge_key]
            if obj_id in self._scene.metadata["object_nodes"]:
                pass # skip modifing trimesh scene
                # self._scene.graph.transforms.edge_data[edge_key].update(matrix=trimesh_transform)
            else:
                self.add_object(
                    obj_id=obj_id,
                    asset=obj_asset,
                    parent_id=parent_id,
                    use_collision_geometry=use_collision_geometry,
                    transform=trimesh_transform,
                    joint_type=joint_type,
                    **kwargs,
                )

            invalidate_scenegraph_cache(self)

            if obj_id not in self.assets:
                self.assets[obj_id] = obj_asset
                self._asset_mesh_cache[obj_id] = make_mesh_buffer(obj_id, obj_asset)
            
            # update collision check things
            self._object_enabled[obj_id] = True
            # st = time.time()
            self._update_curobo_world_ccheck(update_transforms=True, update_enabled=True, update_obj_ids=[obj_id])
            # print (f"update curobo world ccheck taken: {time.time() - st:.4f}s")

        return env_ids.numpy()
    
    def place_object(
        self,
        obj_id,
        obj_asset,
        support_id=None,
        parent_id=None,
        obj_position_iterator=None,
        obj_orientation_iterator=None,
        max_iter=10,
        distance_above_support=0.005,
        joint_type="floating",
        valid_placement_fn=lambda obj_asset, support, placement_T: True,
        constraint = None,
        joint_states = None,
        obj_position_iterator_xy_limit = None,
        erosion_distance: float = 0.02,
        **kwargs,
    ):
        """Add object by placing it in a non-colliding pose on top of a support surface or inside a container.

        Args:
            obj_id (str): Name of the object to place.
            obj_asset (scene.Asset): The asset that represents the object to be placed.
            support_id (str, optional): Defines the support that will be used for placing. Defaults to None. Will be ignored if obj_position_iterator is provided.
            parent_id (str): Name of the object in the scene on which to place the object. Or None if any support surface works. Defaults to None.
            obj_position_iterator (iterator, optional): Iterator for sampling object positions in the support frame. Defaults to PositionIteratorUniform.
            obj_orientation_iterator (iterator, optional): Iterator for sampling object orientation in the object asset frame. Defaults to utils.orientation_generator_uniform_around_z.
            max_iter (int, optional): Maximum number of attempts to find a placement pose. Defaults to 100.
            distance_above_support (float, optional): Distance the object mesh will be placed above the support surface. Defaults to 0.0.
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "floating".
            valid_placement_fn (function, optional): Function for testing valid placements. Defaults to returning True.
            **use_collision_geometry (bool, optional): Defaults to default_use_collision_geometry.
            **kwargs: Keyword arguments that will be delegated to add_object.

        Raises:
            RuntimeError: In case the support_id does not exist.

        Returns:
            bool: Success.
        """
        if obj_orientation_iterator is None:
            obj_orientation_iterator = OrientationGeneratorUniformAroundZ(
                seed=self._rng, replenish_size=self.num_envs*2)

        env_obj_position_iterator = None
        if obj_position_iterator is None:
            if support_id is not None:
                if support_id not in self._scene.metadata["support_polygons"]:
                    raise RuntimeError(f"Support id '{support_id}' does not exist.")
                if constraint is not None:
                    env_obj_position_iterator = []
                    scene_support = constraint.scene_supports
                    for env_idx in range(self.num_envs):
                        logger.debug(f"number of supports for env {env_idx}, {len(scene_support[env_idx])}")
                        if len(scene_support[env_idx]) == 0:
                            env_obj_position_iterator.append(
                                PositionIteratorNone(seed=self._rng, replenish_size=4, xy_limit=obj_position_iterator_xy_limit)
                            )
                        elif len(scene_support[env_idx]) == 1:
                            env_obj_position_iterator.append(
                                PositionIteratorUniform(
                                    seed=self._rng, replenish_size=4, xy_limit=obj_position_iterator_xy_limit,
                                    erosion_distance=erosion_distance
                                )(scene_support[env_idx][0])
                            )
                        else:
                            env_obj_position_iterator.append(
                                PositionIterator2DCollection(
                                    position_iterators=[
                                        PositionIteratorUniform(
                                            seed=self._rng, replenish_size=4, xy_limit=obj_position_iterator_xy_limit,
                                            erosion_distance=erosion_distance
                                        )(s) \
                                            for s in scene_support[env_idx]
                                    ],
                                    replenish_size=4,
                                )
                            )
                    obj_position_iterators = [env_obj_position_iterator[0]]
                else:
                    logger.debug(f'support polygon counts {support_id}, {len(self._scene.metadata["support_polygons"][support_id])}')
                    obj_position_iterators = [
                        PositionIteratorUniform(
                            seed=self._rng, replenish_size=self.num_envs*2, 
                            xy_limit=obj_position_iterator_xy_limit,
                            erosion_distance=erosion_distance
                            )(s) \
                            for s in self._scene.metadata["support_polygons"][support_id]
                    ]
                
                if len(obj_position_iterators) == 1:
                    obj_position_iterator = obj_position_iterators[0]
                else:
                    obj_position_iterator = PositionIterator2DCollection(
                        position_iterators=obj_position_iterators,
                        replenish_size=self.num_envs*2,
                    )
            else:
                raise ValueError("Please pass in either support_id or obj_position_iterator")

        return self.place_objects(
            obj_id_iterator=itertools.repeat(obj_id, 1),
            obj_asset_iterator=itertools.repeat(obj_asset, 1),
            obj_position_iterator=obj_position_iterator,
            env_obj_position_iterator=env_obj_position_iterator,
            obj_orientation_iterator=obj_orientation_iterator,
            parent_id=parent_id,
            max_iter=max_iter,
            distance_above_support=distance_above_support,
            joint_type=joint_type,
            valid_placement_fn=valid_placement_fn,
            constraints=[constraint],
            joint_states=joint_states,
            **kwargs,
        )

    def update_configuration(self, configuration: NDArray, obj_id=None, joint_ids=None, env_ids=None):
        """Set configuration of articulated objects, indiviual joints, or for the entire scene at once.
        If obj_id and joint_ids are specified the joint names will be a concatenation of obj_id and joint_ids.

        Args:
            configuration (NDArray): New configuration value(s).
            obj_id (str, optional): Object identifier to configure. If None and joint_ids=None, all joints in the scene 
                are expected to be updated. Defaults to None.
            joint_ids (list[str], optional): List of joint names to update. If None, all joints of the object are 
                expected to be updated. Defaults to None.
            env_ids (NDArray, optional): The env ids to update. Defaults to None, updating all envs.
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs, dtype=np.int32)
        joint_names = []
        scene_joint_names = self.get_joint_names()
        if obj_id is None and joint_ids is None:
            # set configuration for entire scene
            joint_names = scene_joint_names
        elif joint_ids is None:
            # set configuration for entire object
            if obj_id not in self.metadata["object_nodes"].keys():
                raise ValueError(f"Unknown object_id: {obj_id}")

            joint_names = self.get_joint_names(obj_id=obj_id)
        elif obj_id is None:
            # set configuration for single joint(s)
            for joint_id in joint_ids:
                if joint_id not in scene_joint_names:
                    raise ValueError(f"Unknown joint_id: {joint_id}")
            joint_names = joint_ids
        else:
            # set configuration for single joint(s) but name is split into (obj_id, joint_id)
            if obj_id not in self.metadata["object_nodes"].keys():
                raise ValueError(f"Unknown object_id: {obj_id}")

            joint_names = [obj_id + "/" + joint_id for joint_id in joint_ids]

        # check configuration vector length
        if len(configuration.shape) == 1:
            assert len(configuration) == len(joint_names), f"Length of {configuration} != {joint_names}"
        else:
            assert configuration.shape[1] == len(joint_names), f"Length of {configuration} != {joint_names}"

        # update scene graph
        batch_forward_kinematics(self._scene, joint_names, configuration, edge_batch=self.edge_batch, env_ids=env_ids)

    
    @classmethod
    def gen_from_cfg(cls, cfg: dict[str, Any], **kwargs):
        """Generate scene from cfg.

        Args:
            cfg (dict[str, Any]): The scene gen cfg.
            **kwargs: Keyword arguments that will overwrite cfg.
        """
        # merge override kwargs
        for k, v in kwargs.items():
            cfg[k] = v
        
        # create scene instance from cfg
        scene = cls.init_from_env_cfg(cfg)

        # initialize assets, gen tree, trimesh scene, and collision checker
        scene._load_assets()
        scene._build_gen_tree()
        scene._init_trimesh_scene()
        scene._init_collision_checker()

        return scene

    def _load_assets(self) -> None:
        """load assets from cfg."""
        cfg = self._cfg
        self.assets_extents: dict[str, NDArray] = {}
        for obj_id, obj_cfg in cfg["objects"].items():
            asset_path = obj_cfg["asset_path"]
            self.assets[obj_id] = Asset(asset_path, origin=("center", "center", "bottom"))
            asset_mesh = self.assets[obj_id].as_trimesh_scene(use_collision_geometry=True).dump(concatenate=True)
            self.assets_extents[obj_id] = asset_mesh.bounding_box_oriented.extents # width, depth, height (x,y,z)
        

    def _build_gen_tree(self) -> None:
        """Build a generation tree from cfg by topo sort. Objects will follow the order of the topo sort result, 
        in this case, generating from root node to leaf nodes."""
        # this graph is not the same as the scene graph stored in trimesh
        # this graph has extra edges that reflect the constraints to determine the order of placement
        cfg = self._cfg
        simple_graph = nx.DiGraph()
        simple_graph.add_node("_plane")

        # add node
        for obj_id, obj_cfg in cfg["objects"].items():
            simple_graph.add_node(obj_id)
        
        # add edges, this should be done after adding all nodes
        for obj_id, obj_cfg in cfg["objects"].items():
            parent_id = obj_cfg.get("parent_id", None)
            if parent_id is None:
                parent_id = "_plane"
            if "/" in parent_id:
                parent_id = parent_id.split("/")[0]
            simple_graph.add_edge(parent_id, obj_id) # edge: parent -> obj
            for constraint in obj_cfg.get("constraints", []):
                if constraint["anchor_object_ids"] is not None:
                    for anchor_obj_id in constraint["anchor_object_ids"]:
                        simple_graph.add_edge(anchor_obj_id, obj_id) # edge: anchor_obj -> obj

        assert nx.is_directed_acyclic_graph(simple_graph), "The scene graph is not a DAG"

        nodes = list(nx.topological_sort(simple_graph))
        nodes.remove("_plane")
        self._gen_node_order = nodes


    def _init_trimesh_scene(self) -> None:
        """
        Initialize the trimesh scene for the case where the scene is generated from cfg.
        This automatically adds a plane at z=0.
        If there are workspace limits (specified by `workspace_limits` in env_cfg, [[minx, miny], [maxx, maxy]]), 
        the plane will be initialized by the workspace limits.
        Otherwise, the plane will be initialized by the env_size, centered at (0, 0, 0) by default.
        """
        if self.workspace_limits is not None:
            width = self.workspace_limits[1][0] - self.workspace_limits[0][0]
            depth = self.workspace_limits[1][1] - self.workspace_limits[0][1]
            center = (
                (self.workspace_limits[1][0] + self.workspace_limits[0][0]) / 2,
                (self.workspace_limits[1][1] + self.workspace_limits[0][1]) / 2,
                0
            )
        else:
            width = depth = self.env_size
            center = (0, 0, 0)
            
        self._plane_asset = PlaneAsset(width=width, depth=depth, center=center)
        self.add_object(
            asset=self._plane_asset,
            obj_id="_plane",
        )
        for i, obj_id in enumerate(self._gen_node_order):
            if obj_id in self.assets:
                # give assets enough space so that the lable_support can work properly
                # init_pos = np.array([100*i, 100*i, 100*i])
                init_transform = np.eye(4)
                # init_transform[:3, 3] = init_pos
                self.add_object(
                    obj_id=obj_id,
                    asset=self.assets[obj_id],
                    parent_id=self._cfg["objects"][obj_id].get("parent_id", "_plane"),
                    use_collision_geometry=True,
                    transform=init_transform,
                    joint_type="floating"
                )

    def _init_collision_checker(self) -> None:
        """
        Initialize collision checker. Currently only curobo is supported.
        """
        if self.collision_checker_backend == "curobo":
            self._curobo_world_configs = None
            self._curobo_world_coll_config = None
            self._curobo_world_ccheck = None
            self._init_collision_checker_curobo()
        else:
            raise ValueError(f"Unsupported collision checker backend: {self.collision_checker_backend}")

    def _init_collision_checker_curobo(self) -> None:
        """
        Initialize curobo collision check. Assemble world configs and world mesh collision config. 
        The object meshes are provided, with all poses set to identity, and all objects are disabled.
        """
        tensor_args = TensorDeviceType()
        env_meshes = [[] for _ in range(self.num_envs)]
        for obj_id, asset in self.assets.items():
            asset_mesh_paths = make_mesh_buffer(obj_id, asset)
            self._curobo_mesh_sphere_cache[obj_id] = {}
            for mesh_path in asset_mesh_paths:
                mesh_node_name = os.path.basename(mesh_path).replace(".stl", "").replace("___", "/").replace("object/", f"{obj_id}/")
                mesh = Mesh(
                    name=mesh_node_name,
                    file_path=mesh_path,
                    pose=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32) # identity transform
                )
                self._curobo_mesh_sphere_cache[obj_id][mesh_node_name] = mesh.get_bounding_spheres(n_spheres=self.CUROBO_SPHERE_APPROX_N)
                for env_id in range(self.num_envs):
                    env_meshes[env_id].append(mesh)
                self._all_mesh_names[obj_id].append(mesh_node_name)
            self._object_enabled[obj_id] = False # disable all objects
        
        self._curobo_world_configs = [WorldConfig(mesh=meshes) for meshes in env_meshes]
        self._curobo_world_coll_config = WorldCollisionConfig(
            tensor_args, world_model=self._curobo_world_configs
        )
        self._curobo_world_ccheck = WorldMeshCollision(self._curobo_world_coll_config)
        self._update_curobo_world_ccheck(update_transforms=False)

    def _update_curobo_world_ccheck(
        self, 
        update_enabled: bool = True, 
        update_transforms: bool = True, 
        update_obj_ids: list[str] = [],
        env_ids = None,
    ) -> None:
        """Update the curobo world ccheck. This is only used by generation from cfg and collision check backend=curobo, 
        where `self._curobo_world_ccheck` is available.

        Args:
            update_enabled (bool): Whether to update the enabled state of the objects.
            update_transforms (bool): Whether to update the transforms of the objects.
            update_obj_ids (list[str]): The object ids to update. Defaults to [], will use all enabled objects.
            env_ids (NDArray, optional): The env ids to update. Defaults to None, updating all envs.
        """
        if not hasattr(self, "_curobo_world_ccheck") or self._curobo_world_ccheck is None:
            return
        
        obj_ids = update_obj_ids if len(update_obj_ids) > 0 else list(self._object_enabled.keys())
        for obj_id in obj_ids:
            enabled = self._object_enabled[obj_id]
            for mesh_node_name in self._all_mesh_names[obj_id]:
                # all envs have the same mesh indices
                mesh_idx = self._curobo_world_ccheck.get_mesh_idx(mesh_node_name, env_idx=0)
                if update_enabled:
                    # all env_ids enabled
                    if enabled:
                        self._curobo_world_ccheck._mesh_tensor_list[2][:, mesh_idx] = 1
                    else:
                        self._curobo_world_ccheck._mesh_tensor_list[2][:, mesh_idx] = 0
                if update_transforms:
                    if not enabled:
                        continue
                    mesh_transform = scene_graph_transform_get(
                        self._scene.graph, mesh_node_name, edge_batch=self.edge_batch, cache=self.cache)[0]
                    # to directly modify _mesh_tensor_list[1] (mesh's world ccheck transform), use inverse transform
                    # quat is wxyz
                    pos, quat = batch_transform_matrix_to_vectors(homogeneous_inv_batch(mesh_transform), wxyz=True) 
                    t = np.concatenate([pos, quat], axis=-1)
                    t = torch.from_numpy(t).to(self._curobo_world_ccheck.tensor_args.device, dtype=torch.float32)
                    if env_ids is None:
                        eids = torch.arange(self.num_envs, dtype=torch.long, device=self._curobo_world_ccheck.tensor_args.device)
                    else:
                        eids = torch.from_numpy(env_ids).to(self._curobo_world_ccheck.tensor_args.device, dtype=torch.long)
                    if len(t.shape) > 1:
                        self._curobo_world_ccheck._mesh_tensor_list[1][eids, mesh_idx, :7] = t[eids]
                    else:
                        self._curobo_world_ccheck._mesh_tensor_list[1][eids, mesh_idx, :7] = t

    def gen(self) -> None:
        """
        Execute generation tree. The generated env instances will be stored in the attributes of this class instance.
        Use `export_scene_to_poses_and_joint_states` to get the poses, joint states, and the valid env mask.
        """
        # reset valid env mask
        self.valid_env_mask[:] = True
        # reset enabled objects in collision checker
        for obj_id in self._object_enabled.keys():
            self._object_enabled[obj_id] = False
        self._update_curobo_world_ccheck(update_transforms=False, update_enabled=True)

        # traverse the generation tree
        for obj_id in self._gen_node_order:
            # TODO: handle sub traversal maybe
            object_cfg = self._cfg["objects"][obj_id]
            placement_args = self._parse_obj_placement_cfg(object_cfg) 
            # placement_args: obj_position_iterator, obj_orientation_iterator, constraint, support_id, erosion_distance

            invalid_env_ids = self.place_object(
                obj_id=obj_id,
                obj_asset=self.assets[obj_id],
                parent_id=self._cfg["objects"][obj_id].get("parent_id", "_plane"),
                **placement_args,
            )
            self.valid_env_mask[invalid_env_ids] = False

    def export_scene_to_poses_and_joint_states(
        self, 
        wxyz: bool = True
    ) -> tuple[dict[str, NDArray], dict[str, dict[str, NDArray]], NDArray]:
        """Export the generated scene instances to pure poses and joint states.

        Args:
            wxyz (bool): if True, quat is represented in scalar-first format (wxyz). Otherwise, xyzw.

        Returns:
            poses, joint_states, and a mask for valid envs
            poses is a dict[object_id<str>, pose vector<np.ndarray>(shape N, 7)]
            joint_states is a dict[object_id<str>, dict[joint_id<str>, state<np.ndarray>(shape N)]]
            valid_env_mask is an np.ndarray shape (N), where True means the env at corresponding index is valid
        """
        poses = {}
        for obj_id in self.assets.keys():
            obj_world_T = scene_graph_transform_get(
                self._scene.graph, obj_id, edge_batch=self.edge_batch, cache=self.cache)[0]
            poses[obj_id] = np.concatenate(batch_transform_matrix_to_vectors(obj_world_T, wxyz=wxyz), axis=-1) # (num_envs, 7)

        return poses, self.joint_states, self.valid_env_mask

    @classmethod
    def init_from_env_cfg(cls, cfg: dict[str, Any]):
        """Definition of env_cfg:
        {
            "num_envs": int,
            "env_size": float, # by default, there will be a plane at z=0, with size (env_size, env_size) created.
            "robot_centric_axis_transforms": dict[robot_name<str>, transform<NDArray>]}, # Rotation only, 
                4x4 transform the definition of the left/right/front/back/top/bottom of the env.
                If no transform (or identity), use the definition of https://scene-synthesizer.github.io/concepts/assets.html,
                where left (-x) front (-y) bottom (-z), right (x) back (y) top (z)
                This is useful to transform coordinate system to any robot centric view.
            "workspace_limits": list<list<float>>: # [[minx, miny], [maxx, maxy]], will ignore env_size
            "collision_checker_backend": str, # "curobo" or "trimesh"
        }

        Args:
            env_cfg (dict[str, Any]): _description_
        """
        env_cfg = cfg["env"]
        scene = cls(
            cfg=cfg,
            num_envs=env_cfg["num_envs"],
            env_size=env_cfg.get("env_size", 1.0),
            robot_centric_frame_transforms=env_cfg.get("robot_centric_frame_transforms", None),
            collision_checker_backend=env_cfg.get("collision_checker_backend", "curobo"),
            workspace_limits=env_cfg.get("workspace_limits", None)
        )
        return scene

    def _parse_obj_placement_cfg(self, obj_cfg: dict[str, Any]) -> dict[str, Any]:
        """definition of obj_cfg:
        {
            "id": str,
            "asset_path": str,
            "joint_states": list[dict[str, Any]], # list of joint states, each joint state is a dict with the following keys:
                "joint_ids": list[str], # the joint ids to be controlled
                "max": float, # the maximum value of the joint, relative, range (0-1)
                "min": float, # the minimum value of the joint, relative, range (0-1)
                "distribution": Literal["uniform"] # the distribution of the joint state
            "position": dict[str, Any] | None, # the position of the object, only used for the object that is not a part
                of the parent object
                "xy_limit": list[list[float]] # the limit of the position on xy plane, relatively to the parent support surface.
                    the value is [[minx, miny], [maxx, maxy]] and each element is in [0, 1], meaning from the min to max
                    of that axis.
            "rotation": dict[str, Any] | None, # the rotation of the object, only used for the object that is not a part of the parent object
                if None, uniformly sample a rotation around z.
                "type": Literal["uniform_z", "stable", "constant"], # the type of the rotation. `stable` is randomly sample a stable
                    pose of the object. `uniform_z` is uniformly sample a rotation around z. 
                    `constant` is a constant rotation
                "orientation": float | NDArray, # the orientation of the object, only used for type `constant`.
                    float value means constant rotation around z, in rad, otherwise, it should be a 4x4 transform matrix.
                "lower": float, # the lower bound of the rotation, only used for "uniform_z"
                "upper": float, # the upper bound of the rotation, only used for "uniform_z"
            "parent_id": str, # must be the object that supports or contains the object to be added. 
                If None, the object will be placed on the plane.
            "relation_to_parent": Literal["top", "inside"] | None, If None, the object will be placed on the top of the plane.
                `top` means place the object on the top of the object surface that there is no roof.
                `inside` means place the object inside the object where a part can hold that object.
                For placing inside something, there are two ways: 
                    use the id of theparent object part that holds the object as `parent_id`, and set `relation_to_parent` to `on`,
                    or use the id of the object as `parent_id`, and set `relation_to_parent` to `inside`.
                    By default, we will use "top".
            "ratio_on_support": float | None,
                a value in [0.0, 1.0] indicating how much of the projection of the asset should be included in the support surface, i.e. 
                Area(polygon(asset_project) & polygon(support_surface)) / Area(polygon(asset_project)).
                This is implemented in an estimated way where since we don't know the final pose of the object.
                Defaults to 1.0, meaning that the entire asset (projected on the support surface) should be 
                included in the support surface. If None, means 0.0.
            "constraints": [{ # other constraints
                "anchor_object_ids": list[str],
                "base_support_id": str, # the object that provides the support surface, if not provided, use the support surface of the parent_id
                "direction": Literal["x", "y", "-x", "-y", "front", "back", "left", "right", "middle"] | list[float] | None 
                    # this only handles relation on a surface (x-y plane), z-axis is the normal of the surface
                    # for placing on top of another object, use `parent_id` and `relation_to_parent`
                    # middle is only for multiple anchor object ids (place the object in the region surrounded by other objects)
                    # if None, the object will be placed according to the distance specified.
                "direction_tolerance_angle": float ((0, 180), degrees), # not valid for `direction=middle`
                "distance": float, # distance to the anchor object, only work for single anchor object. 
                "distance_start_bbox": bool, # if True, the distance will be considered as the distance to the bbox of the anchor object, 
                    instead of the origin of the object. This may reduce a lot of potential collisions. Default to True.
                "distance_type": Literal["greater", "less", "equal"] | None = None, # if it will be placed with within or beyond the distance
                "distance_relax": float, # relax the distance constraint by this amount, if the distance type is equal
                "relation_axis_transform": Literal["world"] or object_id<str> or robot_name<str> or NDArray (4x4). # Rotation only. 
                    When direction is specified, this is used to align the direction with world, an object, or the base support.
                    If None, the axis is aligned based on the support surface. If str, it should be any object id or "world".
                    If using robot_name, it should be one of the robot name in the env_cfg.
                "max_mesh_projection_z": float, # the maximum z when consider the mesh project on the support surface.
                    Higher z means more object part will be considered, and potentially more region will be include in the clearance.
            }]
            "fixed_world_positions": list[NDArray] | None, # if not None, the object will be placed at one of the provided positions (in world frame),
              ignoring all other conditions. Defaults to None.
        }

        Args:
            obj_cfg (dict[str, Any]): _description_

        Returns:
            dict[str, Any]: _description_
        """
        # return obj_position_iterator, obj_orientation_iterator, joint_states, constraint, support_id, obj_position_iterator_limit
        placement_args = {}

        obj_id = obj_cfg["id"]
        parent_id = obj_cfg.get("parent_id", None)
        if parent_id is None:
            parent_id = "_plane"
        
        relation_to_parent = obj_cfg.get("relation_to_parent", "top")
        constraints = obj_cfg.get("constraints", [])

        # support surface: composing support_id and label it if it does not exist
        support_id = parent_id
        asset_minxy_length = min(self.assets_extents[obj_id][:2])
        logger.debug(obj_id, "asset_minxy_length:", asset_minxy_length)
        if support_id is not None:
            # assign support id. each object placed on the support will have its unique support id.
            support_id = f"_support_{support_id}_{relation_to_parent}"
            ratio_on_support = obj_cfg.get("ratio_on_support", 0.0)
            if ratio_on_support is not None:
                ratio_on_support = min(1.0, ratio_on_support)
                ratio_on_support = max(0.0, ratio_on_support)
                erosion_distance = asset_minxy_length / 2 * ratio_on_support
            else:
                erosion_distance = 0.0
            placement_args["erosion_distance"] = erosion_distance
            if support_id not in self._scene.metadata["support_polygons"]:
                if relation_to_parent == "top":
                    self.label_support(
                        support_id, 
                        geom_ids=parent_id, 
                        exclude_support_polyhedra=True, 
                        min_area=0.005, 
                    )
                elif relation_to_parent == "inside":
                    self.label_support(
                        support_id, 
                        geom_ids=parent_id, 
                        min_area=0.005,
                        consider_support_polyhedra=True,
                    )
            logger.debug(f'existing supports, {list(self._scene.metadata["support_polygons"].keys())}')
        placement_args["support_id"] = support_id

        # position and orientation
        if obj_cfg.get("position", None) is not None:
            position_cfg = obj_cfg["position"]
            placement_args["obj_position_iterator_xy_limit"] = position_cfg["xy_limit"]
        else:
            placement_args["obj_position_iterator_xy_limit"] = None
        placement_args["obj_position_iterator"] = None

        if obj_cfg.get("rotation", None) is not None:
            rotation_cfg = obj_cfg["rotation"]
            if rotation_cfg["type"] == "uniform_z":
                placement_args["obj_orientation_iterator"] = OrientationGeneratorUniformAroundZ(
                    lower=rotation_cfg.get("lower", -180.0),
                    upper=rotation_cfg.get("upper", 180.0),
                    replenish_size=self.num_envs*2,
                    degrees=True
                )
            elif rotation_cfg["type"] == "stable":
                placement_args["obj_orientation_iterator"] = OrientationGeneratorStablePoses(
                    asset=self.assets[obj_id],
                    z_rotation=rotation_cfg.get("z_rotation", True),
                    replenish_size=self.num_envs*2,
                )
            elif rotation_cfg["type"] == "constant":
                placement_args["obj_orientation_iterator"] = OrientationGeneratorConst(
                    orientation=rotation_cfg.get("orientation", 0.0),
                    replenish_size=self.num_envs*2,
                    degrees=True
                )
            else:
                placement_args["obj_orientation_iterator"] = None
        
        # joint states
        placement_args["joint_states"] = obj_cfg.get("joint_states", None)

        # constraints
        if len(constraints) > 0:
            placement_args["constraint"] = []
            for c in constraints:
                anchor_object_ids = c.get("anchor_object_ids", None)
                if anchor_object_ids is None:
                    continue
                if c.get("base_support_id", None) is None:
                    base_support_id = support_id
                constraint = SurfaceRelation(
                    scene=self,
                    asset_to_add = self.assets[obj_id],
                    anchor_transforms = [
                        TrackingTransform(
                            parent_id=aoi,
                        ) for aoi in anchor_object_ids
                    ],
                    base_support = self._scene.metadata["support_polygons"][base_support_id],
                    direction = c.get("direction", None),
                    direction_tolerance_angle = c.get("direction_tolerance_angle", 90.0),
                    distance = c.get("distance", None),
                    distance_type = c.get("distance_type", None),
                    distance_relax = c.get("distance_relax", 0.01),
                    distance_start_bbox = c.get("distance_start_bbox", False),
                    max_mesh_projection_z = c.get("max_mesh_projection_z", 1.0),
                    relation_axis_transform = c.get("relation_axis_transform", np.eye(4)),
                )
                placement_args["constraint"].append(constraint)
            if len(placement_args["constraint"]) == 1:
                placement_args["constraint"] = placement_args["constraint"][0]
            else:
                c = placement_args["constraint"][0]
                for i in range(1, len(placement_args["constraint"])):
                    # TODO: handle more cases
                    c = c & placement_args["constraint"][i]
                placement_args["constraint"] = c
        else:
            placement_args["constraint"] = None
        
        placement_args["check_reachability"] = obj_cfg.get("reachable", False)
        return placement_args

    def show(self, layers=None, other_scene=None, env_ids: NDArray[np.int32] | None = None, enable_viewer=True):
        """Show scene using the trimesh viewer.

        Args:
            layers (list[str], optional): Filter to show only certain layers, e.g. 'visual' or 'collision'. 
                Defaults to None, showing everything.
            other_scene (trimesh.Scene, optional): Another trimesh scene that will be appended to the scene itself. 
                Defaults to None.
            env_ids (NDArray[np.int32], optional): The env ids to show. Defaults to None, showing env 0.
            enable_viewer (bool, optional): Whether to use the viewer. Defaults to True. 
                If False, export the scene to html.

        Returns:
            trimesh.viewer.windowed.SceneViewer: The viewer.
        """
        
        def get_scene_by_env_id(env_id: int, scene_transform: NDArray):
            scene_copy = self._scene.copy()
            for (u, v) in self._scene.graph.transforms.edge_data:
                transform = scene_graph_transform_get(self._scene.graph, v, u, self.edge_batch, self.cache)[0]
                if len(transform.shape) == 3:
                    transform = transform[env_id]
                scene_copy.graph.update(v, frame_from=u, matrix=transform)
            scene_copy.apply_transform(scene_transform)
            return scene_copy
        
        def combine_scenes(scene_list: list[trimesh.Scene]) -> trimesh.Scene:
            combined_scene = trimesh.Scene(base_frame="world")
            for env_index, s in enumerate(scene_list):
                combined_scene.add_geometry(
                    s.to_mesh(), node_name=f"env_{env_index}", transform=np.eye(4), parent_node_name="world")
            return combined_scene

        if env_ids is None:
            scene_to_show = self._scene if other_scene is None else self._scene + other_scene
        else:
            num_vis_envs = len(env_ids)
            n_cols = int(np.floor(np.sqrt(num_vis_envs)))

            scene_to_show = combine_scenes([
                get_scene_by_env_id(env_id, np.array([
                    [1, 0, 0, self.env_size * 1.2 * (env_id // n_cols)],
                    [0, 1, 0, self.env_size * 1.2 * (env_id % n_cols)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])) for env_id in env_ids])
            # scene_to_show.bounds = get_scene_bounds(scene_to_show)
        
        if enable_viewer:
            viewer = trimesh.viewer.SceneViewer(
                scene=scene_to_show,
                smooth=False,
                start_loop=False,
            )

            if layers is not None and len(layers) > 0:
                for k, v in scene_to_show.geometry.items():
                    if ("layer" in v.metadata and not v.metadata["layer"] in layers) or (
                        "layer" not in v.metadata and None not in layers
                    ):
                        viewer.hide_geometry(node=k)

            _pyglet_app_run()

            return viewer
        else:
            scene_to_show.to_mesh().export(f"tmp/gen_scene.stl")
            html = trimesh.viewer.scene_to_html(scene_to_show)
            with open("tmp/gen_scene.html", "w") as f:
                f.write(html)
    
    # only one-line change of the super class
    def add_object(
        self,
        asset,
        obj_id=None,
        transform=None,
        translation=None,
        parent_id=None,
        connect_obj_id=None,
        connect_obj_anchor=None,
        connect_parent_id=None,
        connect_parent_anchor=None,
        joint_type="fixed",
        **kwargs,
    ):
        """Add a named object mesh to the scene.

        Args:
            asset (scene.Asset): Asset to be added.
            obj_id (str): Name of the object. If None, automatically generates a string.
            transform (np.ndarray): Homogenous 4x4 matrix describing the objects pose in scene coordinates. If None, is identity. Defaults to None.
            translation (list[float], tuple[float]): 3-vector describing the translation of the object. Cannot be set together with transform. Defaults to None.
            parent_id (str): Name of the parent object/frame in the scene graph. Defaults to base frame of scene.
            connect_obj_id (str): Name of a geometry in the asset to which to which the connect_obj_anchor refers. If this is None, the entire object is considered.
            connect_obj_anchor (tuple(str)): (["center", "com", "centroid", "bottom", "top"])*3 defining the coordinate origin of the object in all three dimensions (x, y, z).
            connect_parent_id (str): Name of an existing object in the scene next to which the new one will be added. If this is base_frame or None, all objects are considered.
            connect_parent_anchor (tuple(str)): (["center", "com", "centroid", "bottom", "top"])*3 defining the coordinate origin of the parent subscene/object in all three dimensions (x, y, z).
            joint_type (str, optional): The type of joint that will be used to connect this object to the scene ("floating" or "fixed"). None has a similar effect as "fixed". Defaults to "fixed".
            **use_collision_geometry (bool, optional): Whether to use collision or visual geometry, or both (if None). Defaults to default_use_collision_geometry.

        Returns:
            str: obj_id of added object.
        """
        if obj_id is not None and obj_id in self._scene.metadata["object_nodes"]:
            # if the object is already in the scene, return the obj_id
            return obj_id

        return super().add_object(
            asset, 
            obj_id, 
            transform, 
            translation, 
            parent_id, 
            connect_obj_id, 
            connect_obj_anchor, 
            connect_parent_id, 
            connect_parent_anchor, 
            joint_type, 
            **kwargs
        )

    # modify original label_support to support "top surfaces only" 
    # the original function only support "excluding top surfaces"
    def label_support(
        self,
        label,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        layer="collision",
        **kwargs,
    ):
        """Gives one or multiple support areas in the scene a string identifier which can be used for e.g. placement.

        Args:
            label (str): String identifier.
            gravity ([np.ndarray], optional): Gravity vector in scene coordinates. Defaults to np.array([0, 0, -1.0]).
            gravity_tolerance (float, optional): Tolerance for comparsion between surface normals and gravity vector (dot product). Defaults to 0.5.
            erosion_distance (float, optional): Clearance from support surface edges. Defaults to 0.02.
            layer (str, optional): Layer name to search for support geometries. Defaults to 'collision'.
            **obj_ids (str): Regular expression of object identifiers to use for finding supports.
            **geom_ids (str): Regular expression of geometry identifiers to use for finding supports.
            **min_area (float): Minimum area of support facets [m^2]. Defaults to 0.01.
            **consider_support_polyhedra (bool): If set to True, will sample raycasts to ensure support surface has a "roof". Can be used to exclude top surfaces in shelves.
            **exclude_support_polyhedra (bool): If set to True, will exclude support surfaces that has a "roof". Default to False.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            list[trimesh.path.polygons.Polygon]: List of support polygons.
            list[np.ndarray]: List of homogenous 4x4 matrices describing the polygon poses in scene coordinates.
            list[str]: List of node names that represent the reference frames for the transformations.
            list[int]: List of facet indices of the mesh that form the support polygon.
        """
        support_data = self._get_support_polygons(
            gravity=gravity,
            gravity_tolerance=gravity_tolerance,
            erosion_distance=erosion_distance,
            layer=layer,
            **kwargs,
        )

        if kwargs.get("consider_support_polyhedra", False):
            (
                is_support_polyhedra,
                _,
            ) = self._get_support_polyhedra(support_surfaces=support_data)
            logger.info(
                f"Only {np.sum(is_support_polyhedra)}/{len(is_support_polyhedra)} support surfaces"
                " used for placing objects"
            )
            support_data = [s for (s, b) in zip(support_data, is_support_polyhedra) if b]
        elif kwargs.get("exclude_support_polyhedra", False):
            (
                is_support_polyhedra,
                _,
            ) = self._get_support_polyhedra(support_surfaces=support_data)
            support_data = [s for (s, b) in zip(support_data, is_support_polyhedra) if not b]

        if len(support_data) == 0:
            logger.warning(f"No supports found for label '{label}'.")
        else:
            self._scene.metadata["support_polygons"][label] = support_data

        return support_data

    # modify original _get_support_polyhedra that does not consider the scene instance
    # when doing ray casting (i.e., check if there is a roof above the support surface)
    def _get_support_polyhedra(
        self,
        support_surfaces=None,
        min_volume=0.000001,
        distance_above_support=0.001,
        min_area=0.01,
        gravity=np.array([0, 0, -1.0]),
        gravity_tolerance=0.1,
        erosion_distance=0.02,
        ray_cast_count=10,
        max_height=10.0,
        layer="collision",
        **kwargs,
    ):
        """Creates support polyhedra which are volumes created by extruding support polygons until collision.

        Args:
            min_volume (float, optional): Only return polyhedra with volume greater than this minimum. Defaults to 0.000001.
            distance_above_support (float, optional): Support polyhedra are above the support polygon by this amount. Defaults to 0.001.
            min_area (float, optional): See _get_support_polyhedra. Defaults to 0.01.
            gravity (np.ndarray, optional): See _get_support_polyhedra. Defaults to np.array([0, 0, -1.0]).
            erosion_distance (float, optional): See _get_support_polyhedra. Defaults to 0.02.
            ray_cast_count (int, optional): For testing collisions to extrude support polygons. Defaults to 10.
            max_height (float, optional): Maximum height for container volume (in extrusion direction). Defaults to 10.0.
            layer (str, optional): Name of the layer of the support geometry. Defaults to 'collision'.
            **obj_ids (str): Regular expression of object ids to consider.
            **geom_ids (str): Regular expression of geometry identifiers to use for finding supports.
            **min_x (float): Minimum x coordinate in scene.
            **min_y (float): Minimum y coordinate in scene.
            **min_z (float): Minimum z coordinate in scene.
            **max_x (float): Maximum x coordinate in scene.
            **max_y (float): Maximum y coordinate in scene.
            **max_z (float): Maximum z coordinate in scene.

        Returns:
            list[trimesh.Trimesh]: Support polyhedra in the scene that satisfy the filter criteria.
        """
        if support_surfaces is None:
            support_surfaces = self._get_support_polygons(
                min_area=min_area,
                gravity=gravity,
                gravity_tolerance=gravity_tolerance,
                erosion_distance=erosion_distance,
                layer=layer,
                **kwargs,
            )

        if len(support_surfaces) == 0:
            logger.warning("Warning! No support polygons selected.")

        support_polyhedra = []
        support_polyhedra_mask = []

        # Modified by <jshang>: do not use the entire scene, use the mesh it self
        # (see below in the for loop)
        # original version `scene_mesh = self._scene.dump(concatenate=True)`
        # use the no relationship version so that all other objects are eliminated 
        # this is to ensure that the ray casting only consider the support object itself
        asset_mesh_cache = {}

        for support_surface in support_surfaces:
            support_obj_name = os.path.dirname(support_surface.node_name)
            if support_obj_name not in asset_mesh_cache:
                asset = self.assets.get(support_obj_name, None)
                if asset is None:
                    asset = self._plane_asset
                
                support_obj_transform = self._scene.graph.get(support_obj_name)[0]
                scene_mesh = asset.mesh().copy().apply_transform(support_obj_transform)
                asset_mesh_cache[support_obj_name] = scene_mesh
            else:
                scene_mesh = asset_mesh_cache[support_obj_name]
            
            (is_support_polyhedra, inscribing_polyhedra,) = self._compute_support_polyhedra(
                support_surface=support_surface,
                mesh=scene_mesh,
                gravity=gravity,
                ray_cast_count=ray_cast_count,
                min_volume=min_volume,
                distance_above_support=distance_above_support,
                max_height=max_height,
                erosion_distance=erosion_distance,
                # debug=True
            )
            # print (f"{support_surface.node_name} is_support_polyhedra {is_support_polyhedra}")
            logger.debug(f"{support_surface.node_name} is_support_polyhedra {is_support_polyhedra}")
            if is_support_polyhedra:
                support_polyhedra.append(
                    Container(
                        geometry=inscribing_polyhedra,
                        node_name=support_surface.node_name,
                        transform=support_surface.transform,
                        support_surface=support_surface,
                    )
                )

            support_polyhedra_mask.append(is_support_polyhedra)
        
        del asset_mesh_cache

        return (
            support_polyhedra_mask,
            support_polyhedra,
        )

    def _compute_support_polyhedra(
        self,
        support_surface,
        mesh,
        gravity,
        ray_cast_count,
        min_volume,
        distance_above_support,
        max_height,
        erosion_distance,
        **kwargs,
    ):
        """
        Modify original _compute_support_polyhedra to fix bugs.
        """
        origins, intersections = self._raycast_surface(
            support_surface=support_surface,
            ray_cast_count=ray_cast_count,
            mesh=mesh,
            gravity=gravity,
            distance_above_support=distance_above_support,
            **kwargs,
        )

        # if no intersection occurs we don't deem this a support polyhedra (e.g. top of shelf or table)
        if len(intersections) > 0:
            distances = np.linalg.norm((intersections - origins), axis=1)
            min_distance = np.min(distances)
            assert min_distance >= 0

            # <jshang> modifiation: as long as there is hit, believes it is a support polyhedra
            if min_distance >= trimesh.constants.tol.merge and min_distance <= max_height:
                if support_surface.polygon.geom_type == "MultiPolygon":
                    # This is probably due to the erosion operation when creating supports
                    return False, None

                inscribing_polyhedra = trimesh.creation.extrude_polygon(
                    support_surface.polygon, min_distance - erosion_distance, engine="triangle"
                )
                return True, inscribing_polyhedra

        return False, None
    
    # void this function in the super class to speed up
    def sync_collision_manager(self):
        pass

    # for debug temp modifying:
    def _raycast_surface(
        self,
        support_surface,
        ray_cast_count,
        mesh=None,
        gravity=None,
        distance_above_support=1e-3,
        debug=False,
    ):
        """
        Extrudes a support polygon until collision.

        Args:
            support_surface (SupportSurface): The support surface.
            ray_cast_count (int, optional): For testing collisions to extrude support polygons.
            mesh (trimesh.Trimesh, optional): Defaults to the scene's mesh.
            gravity (np.ndarray, optional): Defaults to np.array([0, 0, -1]) in the surface's coordinate frame.
            distance_above_support (float, optional): Support polyhedra are above the support polygon by this amount.
            debug (bool, optional): Whether to visualize the raycasting results.

        Returns:
            list[np.ndarray]: List of ray origins on the surface.
            list[np.ndarray]: List of ray intersections on the mesh.
        """
        # for each support polygon, sample raycasts to determine maximum height of extrusion in direction of gravity
        pts = utils.sample_polygon(support_surface.polygon, count=ray_cast_count, seed=self._rng)
        # pts = np.array(support_surface.polygon.exterior.coords)

        if len(pts) == 0:
            return [], []

        pts3d_local = np.column_stack([pts, distance_above_support * np.ones(len(pts))])
        T = self._scene.graph.get(support_surface.node_name)[0] @ support_surface.transform
        pts3d = trimesh.transform_points(points=pts3d_local, matrix=T)

        if mesh is None:
            mesh = self._scene.dump(concatenate=True)
        if gravity is None:
            gravity = T[:3, :3] @ np.array([0, 0, -1])

        import trimesh.transformations as tra
        intersections, ray_ids, _ = self._raycasts(
            origins=pts3d,
            directions=np.array(len(pts) * [list(-tra.unit_vector(gravity))]),
            mesh=mesh,
        )

        if len(intersections) == 0:
            return [], []

        origins = pts3d[ray_ids]
        if debug:
            surface_color = utils.random_color(seed=self._rng)
            surface_path = trimesh.load_path(support_surface.polygon).to_3D().apply_transform(T)
            surface_path.colors = len(surface_path.entities) * [surface_color]
            ray_path = trimesh.load_path(
                np.swapaxes(np.stack([origins, intersections], axis=2), 1, 2)
            )
            # print ("=====Raycasting Results=====")
            # print ("origins", origins)
            # print ("intersections", intersections)
            # print ("ray_ids", ray_ids)
            trimesh.Scene([mesh, surface_path, ray_path]).show()

        return origins, intersections