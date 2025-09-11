from sceniris.scene import Scene
import os
import numpy as np
import logging
from argparse import ArgumentParser
# logger = logging.getLogger("scene_synthesizer") # Get a logger instance
# logger.setLevel(logging.DEBUG)

parser = ArgumentParser()
parser.add_argument("--use_original_ss", action="store_true")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--no_viewer", action="store_true")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_size", type=float, default=1.0)
parser.add_argument("--benchmark_dir", type=str, default="benchmark_reachability")
parser.add_argument("--distance_start_bbox", action="store_true")
parser.add_argument("--cfg_level", type=str, default="mid", choices=["mid", "hard", "hard+"])
args = parser.parse_args()

"""
{
    "id": str,
    "asset_path": str,
    "parent_id": str, # must be the object that supports or contains the object to be added
    "relation_to_parent": Literal["in", "top"], # will have bottom
    "constraints": [{ # other constraints
        "anchor_object_ids": list[str],
        "base_support_id": str, # the object that provides the support surface, if not provided, use the support surface of the parent_id
        "direction": Literal["x", "y", "-x", "-y", "front", "back", "left", "right", "middle"] | list[float] | None 
            # this only handles relation on a surface (x-y plane), z-axis is the normal of the surface
            # for placing on top of another object, use `parent_id` and `relation_to_parent`
            # middle is only for multiple anchor object ids (place the object in the region surrounded by other objects)
            # if None, the object will be placed in the region surrounded by other objects
        "direction_tolerance_angle": float ((0, 180), degrees), # not valid for `direction=middle`
        "distance": float | Literal["next_to"], # distance to the anchor object, only work for single anchor object. 
        "distance_start_bbox": bool, # if True, the distance will be considered as the distance to the bbox of the anchor object, 
            instead of the origin of the object. This may reduce a lot of potential collisions.
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
        ignoring all other conditions, otherwise use the constraints to place the object
}
"""

from table_top_configs import config_4, table_top_config_2
from kitchen_configs import kitchen_config_1, kitchen_config_2

cfg = table_top_config_2

# cfg = kitchen_config_2
cfg["env"].update({
    "num_envs": args.num_envs,
    # "env_size": args.env_size,
})
if "env_size" not in cfg["env"]:
    cfg["env"]["env_size"] = args.env_size

# args.cfg_level = "kitchen_config_2"
# args.distance_start_bbox = True
REPEAT = 5 # 5

benchmark_dir = os.path.join(args.benchmark_dir, args.cfg_level)
if args.distance_start_bbox:
    benchmark_dir = benchmark_dir + "_dsb"

import time
st = time.time()
scene = Scene.gen_from_cfg(cfg)
scene.gen()
time_taken = time.time() - st
print (f"first round gen taken: {time_taken:.4f}s, invalid envs: {args.num_envs - int(scene.valid_env_mask.sum())}")

valid_env_ids = np.where(scene.valid_env_mask)[0]

# Export the first valid environment to USD
if len(valid_env_ids) > 0:
    for export_env_id in valid_env_ids:
        output_usd_path = os.path.join(benchmark_dir, f"scene_env_{export_env_id:03d}.usd")
        print(f"Exporting environment {export_env_id} to USD for Isaac Sim: {output_usd_path}")
        scene.export_scene_to_usd_isaac_sim(env_id=export_env_id, output_path=output_usd_path)

if args.visualize:
    # scene.show_supports()
    # scene.show_graph()
    scene.show(env_ids=np.arange(4))
