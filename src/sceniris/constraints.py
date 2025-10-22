# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import numpy as np
import time
from numpy.typing import NDArray
from typing import Literal
from scipy.spatial import ConvexHull
from dataclasses import dataclass as _dataclass
import shapely
from copy import deepcopy
from sceniris.asset import Asset
from sceniris.utils import (
    angle_to_2d_rotation_matrix,
    create_polygon_from_unordered_points,
    create_ring_polygon, 
    get_mesh_bbox_corners,
    get_min_xy_edge,
    resolve_multi_polygon,
    scene_graph_transform_get,
)
from trimesh.path.polygons import Polygon
from scene_synthesizer.scene import SupportSurface

@_dataclass
class TrackingTransform:
    parent_id: str

class SurfaceRelation:
    """Single surface relation"""
    def __init__(
        self,
        scene,
        asset_to_add: Asset | None,
        anchor_transforms: list[TrackingTransform],
        base_support: list[SupportSurface],
        direction: str | NDArray | None = None,
        relation_axis_transform: str | NDArray = np.eye(4),
        direction_tolerance_angle: float = 90.0,
        distance: float | None = None,
        distance_type: Literal["greater", "less", "equal"] | None = None,
        distance_relax: float = 0.01,
        distance_start_bbox: bool = True,
        max_mesh_projection_z: float = 1.0,
        next_to_connect_point_anchor: tuple[str | list[str] | None, str | list[str] | None] = (None, None),
        next_to_connect_point_asset: tuple[str | list[str] | None, str | list[str] | None] = (None, None)
    ):
        """Define a surface relation. By calling `.scene_supports`, the scene supports will be computed.

        Args:
            scene: scene object.
            asset_to_add (Asset): asset to add. Used to compute the bounding box / mesh projection to help some
                constraints.
            anchor_transforms (list[TrackingTransform]): anchor transforms.
                It is used to define the anchor points of the surface relation. Usually their parent_id is the node
                    that involves the relation.
            base_support (list[SupportSurface]): base support surface to consider surface relation.
                Note: all polygons in SupportSurface are using 2D coordinates (local to its support node (mesh)).
            direction (str | NDArray | None, optional): direction wrt the anchor point(s). Defaults to None.
            relation_axis_transform (str | NDArray, optional): if apply the transform on top of the base support 
                transform. Defaults to np.eye(4), i.e., aligned with the surface. When it is a string, it can be a node 
                in the scene graph.
            distance (float | None, optional): distance to the anchor point(s). Defaults to None.
            distance_type (Literal["greater", "less", "equal"] | None, optional): if it represents the area further than
                the distance (`greater`), closer than (`less`), or equal (`equal`) to the distance. Defaults to None.
                It is only used when distance is provided.
            distance_relax (float, optional): if `distance_type=equal`, give it a relax so that points can be sampled. 
                Usually the value is small. Defaults to 0.01.
            distance_start_bbox (bool, optional): Use the bbox of the mesh to compute the distance.
                Expensive to compute, but reduces potential collisions or give better candidate areas for large objects.
            max_mesh_projection_z (float, optional): max z value in a mesh to consider bbox when 
                `distance_start_bbox=True`. Points with z value greater than this value will be ignored. 
                Defaults to 1.0. This is useful for placing objects under a part of another object, e.g., 
                under a table or under an opened drawer.
            next_to_connect_point_anchor (str | list[str] | None, str | list[str] | None, optional): 
                the point of the anchor object that the object to add is connected. The value should be a two-element
                tuple for specifiying the location on x (first element) and y (second element). 
                Each element is a string from ["min", "center", "max"] of a list of string from the list or None 
                (None equals to using all min, center, max).
                (center, center) value is now allowed since objects will collide.
                Defaults to (None, None).
            next_to_connect_point_asset (str | list[str] | None, str | list[str] | None, optional): 
                the point of the asset to add that is used to connected to the anchor point. The value should be in the 
                same format as `next_to_connect_point_anchor`. Defaults to (None, None).
        """
        self.scene = scene
        self.asset_to_add = asset_to_add
        self.anchor_transforms = anchor_transforms
        self.base_support = base_support

        self.num_envs = scene.num_envs
        self.world_to_surface_transforms = None
        self._true_anchor_transforms = []

        self.direction = direction
        if isinstance(self.direction, str):
            self.direction = self.direction.lower()
        if isinstance(self.direction, list):
            self.direction = np.array(self.direction)
        self.relation_axis_transform = relation_axis_transform
        self.direction_tolerance_angle = direction_tolerance_angle
        self.distance = distance
        self.distance_type = distance_type
        self.distance_relax = distance_relax

        self.distance_start_bbox = distance_start_bbox
        self.max_z = max_mesh_projection_z # max z to consider bbox

        self.next_to_connect_point_anchor = next_to_connect_point_anchor
        self.next_to_connect_point_asset = next_to_connect_point_asset
        self._polygon = None
        self._true_base_support = None
        self._scene_supports = None
        self._from_raw = True
        self.init_world_to_surface_transforms()

    def init_world_to_surface_transforms(self):
        """Initialize the transform converting support surface frame to world world frame.
        `self.world_to_surface_transforms` is a (len(base_support), num_envs, 4, 4) numpy array.
        """
        world_to_surface_transforms = []
        for i in range(len(self.base_support)):
            support_node = self.base_support[i].node_name
            # T_w_mesh (coordinate, not frame, frame is T_mesh_w)
            mesh_transform = scene_graph_transform_get(self.scene.graph, support_node, edge_batch=self.scene.edge_batch, cache=self.scene.cache)[0]
            # T_w_normalized = T_w_mesh @ T_mesh_normalized (coordinate, not frame)
            w_s = mesh_transform @ self.base_support[i].transform
            if len(w_s.shape) == 2:
                w_s = np.tile(w_s, (self.num_envs, 1, 1))
            world_to_surface_transforms.append(w_s)
        self.world_to_surface_transforms = np.stack(world_to_surface_transforms, axis=0) # (len(base_support), num_envs, 4, 4))

    @staticmethod
    def from_calculated_scene_supports(scene, scene_supports: list[list[SupportSurface]]):
        """copy a SurfaceRelation from calculated scene supports.

        Args:
            scene (Scene): scene object
            scene_supports (list[list[SupportSurface]]): scene supports. The first level is env, 
                the second level is surface.

        Returns:
            SurfaceRelation: a new SurfaceRelation object
        """
        surface_relation = SurfaceRelation(scene, None, [], [])
        surface_relation._scene_supports = scene_supports
        surface_relation._from_raw = False
        return surface_relation

    def add_new_polygon_to_scene_support(
        self, 
        scene_support: list[SupportSurface], 
        polygon: Polygon, 
        ori_support: SupportSurface
    ) -> None:
        """Add a new polygon to generated scene support that satisfies the surface relation. The condition is checked by
        if the new polygon intersects with the original support surface. If so, the intersection will be computed and 
        added to the scene support.

        Args:
            scene_support (list[SupportSurface]): list of scene supports that accepts new SupportSurface.
            polygon (Polygon): the new polygon to add.
            ori_support (SupportSurface): the original support surface.
        """
        if shapely.intersects(ori_support.polygon, polygon):
            new_polygon = shapely.intersection(ori_support.polygon, polygon)
            if isinstance(new_polygon, shapely.MultiPolygon):
                new_polygon = resolve_multi_polygon(new_polygon)
            else:
                new_polygon = [new_polygon]
            for p in new_polygon:
                new_support = deepcopy(ori_support)
                new_support.polygon = p
                scene_support.append(new_support)

    def _mesh_projection_convex_hull(
        self, 
        points: NDArray, 
        max_z: float = 1.0, 
        backend: Literal["scipy", "shapely"] = "scipy"
    ) -> Polygon:
        """Compute the convex hull of the projected mesh vertices on the surface, with a max z value filtering out 
            vertices higher than `max_z` (with respect to the surface).

        Args:
            points (NDArray): mesh vertices transformed to surface frame. (N, 3)
            max_z (float, optional): max z value in a mesh to consider bbox when 
                `distance_start_bbox=True`. Points with z value greater than this value will be ignored. 
                Defaults to 1.0. This is useful for placing objects under a part of another object, e.g., 
                under a table or under an opened drawer.
            backend (Literal["scipy", "shapely"], optional): backend implementation of convex hull computation. 
                Defaults to `scipy`, which is 200x faster than `shapely`.

        Returns:
            Polygon: convex hull of projected mesh vertices on the surface.
        """
        if backend == "shapely":
            points = points[points[:, 2] < max_z]
            return shapely.convex_hull(shapely.MultiPoint(points))
        elif backend == "scipy":
            # scipy impl is faster than shapely
            hull = ConvexHull(points)
            vertices = points[hull.vertices]
            polygon = Polygon(vertices)
            # fix shapely.errors.GEOSException: TopologyException: side location conflict
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            return polygon
        else:
            raise ValueError(f"Invalid backend: {backend}")

    @property
    def scene_supports(self) -> list[list[SupportSurface]]:
        """Compute the scene supports that satisfy the surface relation.

        Returns:
            list[list[SupportSurface]]: scene supports that satisfy the surface relation. The first level is env, 
                the second level is surface.
        """
        # cache the scene supports, since the computation is expensive
        if self._scene_supports is not None:
            return self._scene_supports

        st = time.time()
        if not self._from_raw:
            raise ValueError("Only from raw surface relation can do the computation")

        anchors_world = []
        for anchor in self.anchor_transforms:
            # directly get the transform from the scene graph
            t = scene_graph_transform_get(
                self.scene.graph, anchor.parent_id, edge_batch=self.scene.edge_batch, cache=self.scene.cache)[0]
            if len(t.shape) == 2:
                t = np.tile(t, (self.num_envs, 1, 1))
            t = np.tile(t, (len(self.base_support), 1, 1, 1))
            anchors_world.append(t)
        anchors_world = np.stack(anchors_world, axis=0) # (len(anchor_transforms), len(base_support), num_envs, 4, 4)
        anchors_T = np.linalg.inv(self.world_to_surface_transforms) @ anchors_world # mesh
        anchors_mesh_vertices_surface = []
        anchors_raw_mesh_vertices = []

        if isinstance(self.direction, str) and "next_to" in self.direction:
            self.distance_start_bbox = True
        
        if self.distance_start_bbox:
            # this implementation for time efficiency
            # for RAM efficiency, please compute the vertices env by env and make it a convex hull immediately
            # stt = time.time()
            for anchor_idx, anchor in enumerate(self.anchor_transforms):
                anchor_id = anchor.parent_id
                asset = self.scene.assets[anchor_id]
                # TODO: make get mesh vertices surface a function
                node_name_T_mesh_list = asset.node_named_geometries()
                mesh_vertices = []
                raw_mesh_vertices = []
                for node_name, ori_T, mesh in node_name_T_mesh_list:
                    scene_node_name = node_name.replace("object/", f"{anchor_id}/")
                    T = scene_graph_transform_get(
                        self.scene.graph, scene_node_name, edge_batch=self.scene.edge_batch, cache=self.scene.cache)[0] # (N, 4, 4) or (4, 4)
                    mesh_T = np.linalg.inv(self.world_to_surface_transforms) @ T
                    v = np.array(mesh.vertices) # (N, 3)
                    # (len(base_support), num_envs, 1, 4, 4)) @ (N, 4, 1) -> (len(base_support), num_envs, N, 4, 1))
                    v = mesh_T[:, :, np.newaxis, ...] @ (np.concatenate([v, np.ones((v.shape[0], 1), dtype=v.dtype)], axis=1)[:, :, np.newaxis])
                    mesh_vertices.append(v[..., :3, 0])
                    raw_v = ori_T @ (np.concatenate([mesh.vertices, np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)], axis=1)[:, :, np.newaxis])
                    raw_mesh_vertices.append(raw_v[..., :3, 0])
                mesh_vertices = np.concatenate(mesh_vertices, axis=2) # (len(base_support), num_envs, N*n_meshes, 3)
                raw_mesh_vertices = np.concatenate(raw_mesh_vertices, axis=0) # (N*n_meshes, 3)
                anchors_mesh_vertices_surface.append(mesh_vertices) # list of mesh vertices
                anchors_raw_mesh_vertices.append(raw_mesh_vertices) # list of raw mesh vertices
            
            # compute the mesh vertices of the asset to add
            node_name_T_mesh_list = self.asset_to_add.node_named_geometries()
            mesh_vertices = []
            for node_name, mesh_T, mesh in node_name_T_mesh_list:
                v = np.array(mesh.vertices) # (N, 3)
                # (4, 4) @ (N, 4, 1) -> (N, 4, 1)
                v = mesh_T @ (np.concatenate([v, np.ones((v.shape[0], 1), dtype=v.dtype)], axis=1)[:, :, np.newaxis])
                mesh_vertices.append(v[..., :3, 0])
            mesh_vertices = np.concatenate(mesh_vertices, axis=0) # (N*n_meshes, 3)
            asset_to_add_mesh_vertices = mesh_vertices # list of mesh vertices
            asset_to_add_bbox_corners = get_mesh_bbox_corners(asset_to_add_mesh_vertices)

        scene_support_surfaces = []
        if self.direction is None:
            # case that there is no direction, only distance is used
            # this is for only one anchor point
            assert len(self.anchor_transforms) == 1, "Only one anchor point is allowed when direction is None"
            if self.distance is None:
                raise ValueError("Distance is required when direction is None")
            if self.distance_type == None:
                self.distance_type = "equal"
            
            anchor_pos = anchors_T[0, :, :, :2, 3]
            for env_idx in range(self.num_envs):
                scene_support_surfaces.append([])
                for surface_idx in range(len(self.base_support)):
                    if self.distance_type == "equal":
                        inner_r = self.distance - self.distance_relax
                        outer_r = self.distance + self.distance_relax
                    elif self.distance_type == "greater":
                        inner_r = self.distance
                        outer_r = self.distance + 100
                    elif self.distance_type == "less":
                        inner_r = 0.01
                        outer_r = self.distance
                    else:
                        raise ValueError(f"Invalid distance type: {self.distance_type}")

                    if self.distance_start_bbox:
                        asset_to_add_min_edge = get_min_xy_edge(asset_to_add_bbox_corners)
                        mesh_vertices = anchors_mesh_vertices_surface[0][surface_idx, env_idx] # (N*n_meshes, 3)
                        convex_hull: Polygon = self._mesh_projection_convex_hull(mesh_vertices, max_z = self.max_z) # Polygon
                        polygon = shapely.difference(convex_hull.buffer(outer_r), convex_hull.buffer(inner_r + asset_to_add_min_edge)) # TODO: faster impl / batch impl
                    else:
                        center = anchor_pos[surface_idx, env_idx]
                        polygon = create_ring_polygon(
                            center = center, 
                            inner_r = inner_r, 
                            outer_r = outer_r,
                            split = 64
                        )
                    self.add_new_polygon_to_scene_support(
                        scene_support_surfaces[env_idx],
                        polygon,
                        self.base_support[surface_idx],
                    )
        else:
            # case that there is direction
            if (isinstance(self.direction, str) and self.direction in ["x", "y", "-x", "-y", "front", "back", "left", "right"]) or \
                isinstance(self.direction, np.ndarray):
                if isinstance(self.direction, str) and self.direction in ["x", "y", "-x", "-y", "front", "back", "left", "right"]:
                    if self.direction == "x" or self.direction == "right":
                        direction = np.array([1.0, 0.0])
                    elif self.direction == "-x" or self.direction == "left":
                        direction = np.array([-1.0, 0.0])
                    elif self.direction == "y" or self.direction == "back":
                        direction = np.array([0.0, 1.0])
                    elif self.direction == "-y" or self.direction == "front":
                        direction = np.array([0.0, -1.0])
                elif isinstance(self.direction, np.ndarray):
                    assert len(self.direction.shape) == 1 and len(self.direction) == 2
                    direction = self.direction.copy()
                    if not np.isclose(np.linalg.norm(direction), 1.0):
                        direction = direction / (np.linalg.norm(direction) + 1e-5)
                    
                if len(self.anchor_transforms) > 1: # a group of objects
                    if self.direction_align_anchor:
                        raise ValueError("Direction align anchor is not supported for multiple objects")
                    if not all([anchor.parent_id == self.anchor_transforms[0].parent_id for anchor in self.anchor_transforms]):
                        raise ValueError("If anchoring on multiple objects, they must share the same parent")
                    # assume all anchor_transforms share the same parent
                    anchor_pos = np.mean(anchors_T[:, :, :, :2, 3], axis=0)
                else:
                    anchor_pos = anchors_T[0, :, :, :2, 3]

                if isinstance(self.relation_axis_transform, str):
                    if self.relation_axis_transform == "world":
                        relation_axis_transform = self.world_to_surface_transforms[..., :3, :3] # (surface, num_envs, 3, 3)
                    else:
                        found = False
                        for anchor_idx, anchor in enumerate(self.anchor_transforms):
                            if anchor.parent_id == self.relation_axis_transform:
                                relation_axis_transform = anchors_T[anchor_idx, :, :, :3, :3] # T_mesh_anchor
                                found = True
                                break
                        if not found:
                            relation_axis_transform = np.eye(4)
                else:
                    relation_axis_transform = self.relation_axis_transform
                
                if len(relation_axis_transform.shape) == 2:
                    relation_axis_transform = np.tile(relation_axis_transform, (len(self.base_support), self.num_envs, 1, 1))
                elif len(relation_axis_transform.shape) == 3:
                    relation_axis_transform = np.tile(relation_axis_transform, (len(self.base_support), 1, 1, 1))

                # (surface, num_envs, 3, 1)
                direction = relation_axis_transform[..., :3, :3] @ (np.concatenate([direction, np.zeros((1,), dtype=direction.dtype)])[..., np.newaxis])
                if len(direction.shape) == 2:
                    direction = np.tile(direction, (self.num_envs, 1, 1))
                direction = direction[..., :2, :]
                
                start_dir = (angle_to_2d_rotation_matrix(abs(self.direction_tolerance_angle)) @ direction)[..., 0] # (surface, num_envs, 2)
                end_dir = (angle_to_2d_rotation_matrix(-abs(self.direction_tolerance_angle)) @ direction)[..., 0] # (surface, num_envs, 2)
                if self.distance is None:
                    # if no distance is provided
                    inner_r = 0.01
                    outer_r = 100.0
                else:
                    if self.distance_type == "equal":
                        inner_r = self.distance - self.distance_relax
                        outer_r = self.distance + self.distance_relax
                    elif self.distance_type == "greater":
                        inner_r = self.distance
                        outer_r = self.distance + 100
                    elif self.distance_type == "less":
                        inner_r = 0.01
                        outer_r = self.distance
                    else:
                        raise ValueError(f"Invalid distance type: {self.distance_type}")
                for env_idx in range(self.num_envs):
                    scene_support_surfaces.append([])
                    for surface_idx in range(len(self.base_support)):
                        # need to get 2D coordinate of the anchor transform on the support surface
                        center = anchor_pos[surface_idx, env_idx]
                        if self.distance_start_bbox:
                            outer_polygons = []
                            inner_polygons = []
                            true_inner_r = inner_r + get_min_xy_edge(asset_to_add_bbox_corners)
                            for anchor_idx in range(len(self.anchor_transforms)):
                                mesh_vertices = anchors_mesh_vertices_surface[0][surface_idx, env_idx] # (N*n_meshes, 3)
                                convex_hull: Polygon = self._mesh_projection_convex_hull(mesh_vertices, max_z = self.max_z) # Polygon
                                outer_polygons.append(convex_hull.buffer(outer_r))
                                inner_polygons.append(convex_hull.buffer(true_inner_r))
                            polygon = shapely.difference(shapely.union_all(outer_polygons), shapely.union_all(inner_polygons))
                            direction_polygon = create_ring_polygon(
                                center, true_inner_r, 100, 
                                dir_start=start_dir[surface_idx, env_idx], dir_end=end_dir[surface_idx, env_idx])
                            polygon = shapely.intersection(polygon, direction_polygon)
                        else:
                            polygon = create_ring_polygon(
                                center, inner_r, outer_r, 
                                dir_start=start_dir[surface_idx, env_idx], dir_end=end_dir[surface_idx, env_idx])
                        self.add_new_polygon_to_scene_support(
                            scene_support_surfaces[env_idx],
                            polygon,
                            self.base_support[surface_idx],
                        )
            elif isinstance(self.direction, str) and self.direction == "middle":
                # ignore self.direction
                if len(self.anchor_transforms) == 1:
                    raise ValueError("Direction middle is not supported for single object")
                elif len(self.anchor_transforms) == 2:
                    for env_idx in range(self.num_envs):
                        scene_support_surfaces.append([])
                        for surface_idx in range(len(self.base_support)):
                            env_suf_anchor_pos = anchors_T[:, surface_idx, env_idx, :2, 3]
                            minx = np.min(env_suf_anchor_pos[:, 0])
                            miny = np.min(env_suf_anchor_pos[:, 1])
                            maxx = np.max(env_suf_anchor_pos[:, 0])
                            maxy = np.max(env_suf_anchor_pos[:, 1])
                            polygon = Polygon([
                                (minx, miny),
                                (maxx, miny),
                                (maxx, maxy),
                                (minx, maxy),
                                (minx, miny)
                            ])
                            if self.distance_start_bbox:
                                for anchor_idx in range(len(self.anchor_transforms)):
                                    mesh_vertices = anchors_mesh_vertices_surface[anchor_idx][surface_idx, env_idx] # (N*n_meshes, 3)
                                    convex_hull: Polygon = self._mesh_projection_convex_hull(mesh_vertices, max_z = self.max_z) # Polygon
                                    polygon = shapely.difference(polygon, convex_hull)
                                    
                            self.add_new_polygon_to_scene_support(
                                scene_support_surfaces[env_idx],
                                polygon,
                                self.base_support[surface_idx],
                            )
                elif len(self.anchor_transforms) > 2:
                    for env_idx in range(self.num_envs):
                        scene_support_surfaces.append([])
                        for surface_idx in range(len(self.base_support)):
                            env_suf_anchor_pos = anchors_T[:, surface_idx, env_idx, :2, 3]
                            polygon = create_polygon_from_unordered_points(env_suf_anchor_pos)
                            if self.distance_start_bbox:
                                for anchor_idx in range(len(self.anchor_transforms)):
                                    mesh_vertices = anchors_mesh_vertices_surface[anchor_idx][surface_idx, env_idx] # (N*n_meshes, 3)
                                    convex_hull: Polygon = self._mesh_projection_convex_hull(mesh_vertices, max_z = self.max_z) # Polygon
                                    asset_to_add_min_edge = get_min_xy_edge(asset_to_add_bbox_corners)
                                    polygon = shapely.difference(polygon, convex_hull.buffer(asset_to_add_min_edge))
                            self.add_new_polygon_to_scene_support(
                                scene_support_surfaces[env_idx],
                                polygon,
                                self.base_support[surface_idx],
                            )
            
            # TODO: finish next_to case after making the object to be 
            # TODO: process with connect point
            # elif isinstance(self.direction, str) and self.direction == "next_to":
            #     # case that the direction is next to the anchor
            #     assert len(self.anchor_transforms) == 1, "Only one anchor is allowed when direction is next_to"
            #     anchor_pos = anchors_T[0, :, :, :2, 3]
            #     relation_axis_transform = anchors_T[0, :, :, :3, :3]
            #     if self.next_to_connect_point_anchor[0] is None:
            #         self.next_to_connect_point_anchor[0] = ["min", "center", "max"]
            #     if isinstance(self.next_to_connect_point_anchor[0], str):
            #         self.next_to_connect_point_anchor[0] = [self.next_to_connect_point_anchor[0]]
            #     if self.next_to_connect_point_anchor[1] is None:
            #         self.next_to_connect_point_anchor[1] = ["min", "center", "max"]
            #     if isinstance(self.next_to_connect_point_anchor[1], str):
            #         self.next_to_connect_point_anchor[1] = [self.next_to_connect_point_anchor[1]]

            #     anchor_bbox_corners = get_mesh_bbox_corners(anchors_raw_mesh_vertices[0])
            #     anchor_offsets = []
            #     for x_dir in self.next_to_connect_point_anchor[0]:
            #         for y_dir in self.next_to_connect_point_anchor[1]:
            #             anchor_offset = np.array([
            #                 get_edge_point_from_bbox_corners(anchor_bbox_corners, axis=0, loc=x_dir), 
            #                 get_edge_point_from_bbox_corners(anchor_bbox_corners, axis=1, loc=y_dir)
            #             ])
            #             anchor_offsets.append(anchor_offset)
            #     target_offsets = []
            #     for x_dir in self.next_to_connect_point_asset[0]:
            #         for y_dir in self.next_to_connect_point_asset[1]:
            #             target_offset = np.array([
            #                 get_edge_point_from_bbox_corners(asset_to_add_bbox_corners, axis=0, loc=x_dir), 
            #                 get_edge_point_from_bbox_corners(asset_to_add_bbox_corners, axis=1, loc=y_dir)
            #             ])
            #             target_offsets.append(target_offset)
                    
            #     direction = relation_axis_transform[..., :3, :3] @ (np.concatenate([direction, np.zeros((1,), dtype=direction.dtype)])[..., np.newaxis])
            #     if len(direction.shape) == 2:
            #         direction = np.tile(direction, (self.num_envs, 1, 1))
            #     direction = direction[..., :2, :]

        self._scene_supports = scene_support_surfaces
        return scene_support_surfaces
    
    def _group_supports_by_source(
        self, 
        s1: list[SupportSurface], 
        s2: list[SupportSurface]
    ) -> list[list[SupportSurface]]:
        """Group the supports form two sources according to their surfaces. If any of two supports are from
        the same source (identified by their surface transforms are identical (close for numerical purpose)), they will 
        be grouped together.

        Args:
            s1 (list[SupportSurface]): supports from the first source.
            s2 (list[SupportSurface]): supports from the second source.

        Returns:
            list[list[SupportSurface]]: grouped supports. The first level is same surface, the second level is supports.
        """
        def find_close(all_t, t) -> int:
            """Find the index of the closest transform in `all_t` to `t`.

            Args:
                all_t (list[NDArray]): list of transforms.
                t (NDArray): the target transform.

            Returns:
                int: the index of the closest transform.
            """
            for i, t_p in enumerate(all_t):
                if np.allclose(t_p, t):
                    return i
            return -1
        
        all_transforms = []
        grouped_supports = [] # len(grouped_supports) == len(all_transforms)
        for s in s1+s2:
            t = s.transform
            idx = find_close(all_transforms, t)
            if idx == -1:
                all_transforms.append(t)
                grouped_supports.append([])
                grouped_supports[-1].append(s)
            else:
                grouped_supports[idx].append(s)
        return grouped_supports

    def __or__(self, other):
        """Union the scene supports of two SurfaceRelation objects."""
        assert self.base_support[0].node_name == other.base_support[0].node_name, "Base support must be the same"
        scene_support_surfaces = [[] for _ in range(self.num_envs)]
        for env_idx in range(len(self._scene_supports)):
            grouped_supports = self._group_supports_by_source(self._scene_supports[env_idx], other._scene_supports[env_idx])
            for g in grouped_supports:
                if len(g) == 1:
                    scene_support_surfaces[env_idx].append(g[0])
                else:
                    new_support = deepcopy(g[0])
                    union_polygon = shapely.union_all([s.polygon for s in g])
                    new_support.polygon = union_polygon
                    scene_support_surfaces[env_idx].append(new_support)
        return SurfaceRelation.from_calculated_scene_supports(self.scene, scene_support_surfaces)

    def __and__(self, other):
        """Intersect the scene supports of two SurfaceRelation objects."""
        assert self.base_support[0].node_name == other.base_support[0].node_name, "Base support must be the same"
        scene_support_surfaces = [[] for _ in range(self.num_envs)]
        for env_idx in range(len(self._scene_supports)):
            grouped_supports = self._group_supports_by_source(self._scene_supports[env_idx], other._scene_supports[env_idx])
            for g in grouped_supports:
                if len(g) == 1:
                    scene_support_surfaces[env_idx].append(g[0])
                else:
                    new_support = deepcopy(g[0])
                    union_polygon = shapely.intersection_all([s.polygon for s in g])
                    new_support.polygon = union_polygon
                    scene_support_surfaces[env_idx].append(new_support)
        return SurfaceRelation.from_calculated_scene_supports(self.scene, scene_support_surfaces)
