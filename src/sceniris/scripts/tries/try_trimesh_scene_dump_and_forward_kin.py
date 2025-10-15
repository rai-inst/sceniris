# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from argparse import ArgumentParser
import os
import numpy as np

from utils import scene_graph_transform_get, batch_forward_kinematics

parser = ArgumentParser()
parser.add_argument("--use_original_ss", action="store_true")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--no_viewer", action="store_true")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_size", type=float, default=1.0)
args = parser.parse_args()

use_original_ss = args.use_original_ss
NUM_ENVS = args.num_envs
VISUALIZE = args.visualize

if use_original_ss:
    from scene_synthesizer.assets import Asset
    from scene_synthesizer.scene import Scene
else:
    from sceniris.scene import Scene
    from sceniris.asset import Asset


drawer = Asset(
    os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
    origin=("center", "center", "bottom"),
)
query_obj_scene = drawer.as_trimesh_scene(use_collision_geometry=True)
# joint_map = {}
# scene_edge_data = query_obj_scene.graph.transforms.edge_data
# EDGE_KEY_METADATA = "metadata"
# for k in query_obj_scene.graph.transforms.edge_data:
#     edge_data = scene_edge_data[k]
#     if (
#         EDGE_KEY_METADATA in edge_data
#         and edge_data[EDGE_KEY_METADATA] is not None
#         and "joint" in edge_data[EDGE_KEY_METADATA]
#     ):
#         joint_data = edge_data[EDGE_KEY_METADATA]["joint"]
#         joint_map[joint_data["name"]] = k
# print (joint_map)
# joint map: {
# 'object/drawer_joint_lower': ('object/unit_1_cabinet', 'object/drawer_joint_lower_frame'), 
# 'object/drawer_joint_upper': ('object/unit_1_cabinet', 'object/drawer_joint_upper_frame')
# }
# exit()
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
# default_mesh_folder = "/tmp/sgv2"
# make_mesh_buffer(query_obj_id, drawer, default_folder=default_mesh_folder)
# N_SPH_BASE = 100
# collision_check_results = []
for node_name, T, mesh in query_node_name_T_mesh_list:
    # fn = node_name.replace("object/", f"{query_obj_id}__") # double __ to seperate obj_id and node_name
    # path = os.path.join(default_mesh_folder, f"{fn}.stl")
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

