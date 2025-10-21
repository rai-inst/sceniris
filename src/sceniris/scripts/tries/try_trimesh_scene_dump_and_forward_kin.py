# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from argparse import ArgumentParser
import os
import numpy as np

from utils import scene_graph_transform_get, batch_forward_kinematics

parser = ArgumentParser()
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--no_viewer", action="store_true")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_size", type=float, default=1.0)
args = parser.parse_args()

use_original_ss = args.use_original_ss
NUM_ENVS = args.num_envs
VISUALIZE = args.visualize

from sceniris.asset import Asset


drawer = Asset(
    os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
    origin=("center", "center", "bottom"),
)
query_obj_scene = drawer.as_trimesh_scene(use_collision_geometry=True)
before_node_name_T_mesh_list = drawer.node_named_geometries()
for node_name, T, mesh in before_node_name_T_mesh_list:
    print (node_name, T[:3, 3])


print("============================")

query_obj_scene = drawer.as_trimesh_scene(use_collision_geometry=True)
query_obj_edge_batch = {}
query_obj_joint_configs = {
    "joint_names": ["object/drawer_joint_lower", "object/drawer_joint_upper"],
    "configuration": np.array([[0.0, 0.0], [0.01, 0.1], [0.1, 0.1]]),
}
query_obj_id = "object"
if query_obj_joint_configs is not None:
    batch_forward_kinematics(
        query_obj_scene, 
        joint_names=query_obj_joint_configs["joint_names"], 
        configuration=query_obj_joint_configs["configuration"],
        edge_batch=query_obj_edge_batch
    )
query_node_name_T_mesh_list = drawer.node_named_geometries(use_collision_geometry=True)
for node_name, T, mesh in query_node_name_T_mesh_list:
    local_T = scene_graph_transform_get(
        query_obj_scene.graph, node_name, edge_batch=query_obj_edge_batch, cache={}
    )[0]
    if len(local_T.shape) == 3:
        print (node_name, local_T[:, :3, 3], local_T.shape)
    else:
        print (node_name, local_T[:3, 3], local_T.shape)

    print (type(mesh))
    print (mesh.vertices, type(mesh.vertices), type(np.array(mesh.vertices)))

os.system(f"rm -rf /tmp/sgv2")
