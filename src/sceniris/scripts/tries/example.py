# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from sceniris.scene import Scene
import os
import numpy as np
from argparse import ArgumentParser

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

mid_cfg = {
    "env": {
        "num_envs": args.num_envs,
        "env_size": args.env_size,
    },
    "objects": [
        {
            "id": "apple",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/013_apple/textured/textured.usd"),
            'reachable': True,
            # "position": {
            #     "xy_limit": [[0.2, 0.2], [0.8, 0.8]],  # Use fractions: 10%-90% of support surface
            #     "z_limit": [0.0, 0.5],
            # },
            # "constraints": [
            #     {
            #         "anchor_object_ids": ["drawer", "banana"],
            #         "direction": "middle",
            #         "distance_start_bbox": args.distance_start_bbox,
            #     }
            # ]
        },
        {
            "id": "drawer",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
            "joint_states": [
                {
                    "joint_ids": ["drawer_joint_lower"],
                    "max": 0.3,
                    "min": 0.1,
                    "distribution": "uniform",
                },
                {
                    "joint_ids": ["drawer_joint_upper"],
                    "max": 1.0,
                    "min": 0.7,
                    "distribution": "uniform",
                },
            ]
        },
        {
            "id": "banana",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
            "parent_id": "drawer/unit_1_upper_drawer", # might need to optimize the part name in trimesh scene
            "relation_to_parent": "top",
            "reachable": True,
        },
        # {
        #     "id": "banana",
        #     "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
        #     "constraints": [
        #         {
        #             "anchor_object_ids": ["drawer"],
        #             "distance": 0.1,
        #             "distance_type": "greater",
        #             "relation_axis_transform": "drawer",
        #             "max_mesh_projection_z": 0.1,
        #             "distance_start_bbox": args.distance_start_bbox,
        #         }
        #     ]
        # },
    ]
}




if args.cfg_level == "hard":
    cfg = hard_cfg
elif args.cfg_level == "mid":
    cfg = mid_cfg
else:
    raise ValueError(f"Invalid cfg level: {args.cfg_level}")


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

if not os.path.exists(benchmark_dir):
    os.makedirs(benchmark_dir)

print ("bench mark result will be saved to ", os.path.join(benchmark_dir, f"benchmark_{args.num_envs:05d}.csv"))

f = open(os.path.join(benchmark_dir, f"benchmark_{args.num_envs:05d}.csv"), "w")

import time
st = time.time()
scene = Scene.gen_from_cfg(cfg)
scene.gen()
time_taken = time.time() - st
print (f"first round gen taken: {time_taken:.4f}s, invalid envs: {args.num_envs - int(scene.valid_env_mask.sum())}")
f.write(f"0\t{time_taken:.4f}\t{args.num_envs - int(scene.valid_env_mask.sum())}\n")

for i in range(REPEAT):
    st = time.time()
    scene.gen()
    time_taken = time.time() - st
    print (f"{i+2} round gen taken: {time_taken:.4f}s, invalid envs: {args.num_envs - int(scene.valid_env_mask.sum())}")
    f.write(f"{i+1}\t{time_taken:.4f}\t{args.num_envs - int(scene.valid_env_mask.sum())}\n")
f.close()

# find valid env_ids
valid_env_ids = np.where(scene.valid_env_mask)[0]

if args.visualize:
    # scene.show_supports()
    # scene.show_graph()
    scene.show(env_ids=valid_env_ids[:4] )