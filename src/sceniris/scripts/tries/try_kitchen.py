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
parser.add_argument("--benchmark_dir", type=str, default="benchmark")
parser.add_argument("--distance_start_bbox", action="store_true")
parser.add_argument("--cfg_level", type=str, default="mid", choices=["mid", "hard"])
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

from kitchen_configs import kitchen_config_1, kitchen_config_2
base_asset_path = os.path.expanduser("~/fm_storage/fm_assets/sceniris_benchmark_assets/Assets")

kitchen_config_1_env_size = 3.866
cfg = kitchen_config_2
cfg["env"].update({
    "num_envs": args.num_envs,
    # "env_size": args.env_size,
})
if "env_size" not in cfg["env"]:
    cfg["env"]["env_size"] = args.env_size


args.cfg_level = "kitchen_config_1"
args.distance_start_bbox = True
REPEAT = 1 # 5

benchmark_dir = os.path.join(args.benchmark_dir, args.cfg_level)
if args.distance_start_bbox:
    benchmark_dir = benchmark_dir + "_dsb"

if not os.path.exists(benchmark_dir):
    os.makedirs(benchmark_dir)
# f = open(os.path.join(benchmark_dir, f"benchmark_{args.num_envs:05d}.csv"), "w")

import time
st = time.time()
scene = Scene.gen_from_cfg(cfg)
scene.gen()
scene.label_support(label="base_cabinat_support", geom_ids="base_cabinat", exclude_support_polyhedra=True)
time_taken = time.time() - st
print (f"first round gen taken: {time_taken:.4f}s, invalid envs: {args.num_envs - int(scene.valid_env_mask.sum())}")
# f.write(f"0\t{time_taken:.4f}\t{args.num_envs - int(scene.valid_env_mask.sum())}\n")

for i in range(0):
    st = time.time()
    scene.gen()
    time_taken = time.time() - st
    print (f"{i+2} round gen taken: {time_taken:.4f}s, invalid envs: {args.num_envs - int(scene.valid_env_mask.sum())}")
    # f.write(f"{i+1}\t{time_taken:.4f}\t{args.num_envs - int(scene.valid_env_mask.sum())}\n")
# f.close()

if args.visualize:
    print (list(scene._scene.metadata["support_polygons"].keys()))
    # scene.show_supports()
    # scene.show_graph()
    scene.show(env_ids=np.arange(1))
