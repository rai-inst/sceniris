# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# step-by-step creation; not supported now 
from argparse import ArgumentParser
import os
import numpy as np

import time
from multiprocessing import Pool
import psutil

from sceniris.constraints import SurfaceRelation, TrackingTransform

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

from sceniris.asset import PlaneAsset

if use_original_ss:
    from scene_synthesizer.assets import Asset
    from scene_synthesizer.scene import Scene
else:
    from sceniris.scene import Scene
    from sceniris.asset import Asset

def generate_scene(i):

    my_scene = Scene(num_envs=NUM_ENVS, env_size=args.env_size)

    my_scene.add_object(
        asset=PlaneAsset(width=args.env_size, depth=args.env_size),
        obj_id="plane",
    )

    my_scene.label_support(
        "plane_support",
        obj_ids="plane"
    )

    drawer = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
        origin=("center", "center", "bottom"),
    )

    banana = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    apple = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/013_apple/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    my_scene.place_object(
        obj_asset=drawer,
        obj_id="drawer",
        support_id="plane_support",
        parent_id="plane",
        # joint_type="fixed",
    )
    for j in my_scene.get_joint_names():
        limits = my_scene.get_joint_limits(joint_ids=[j])[0]

        # my_scene.update_configuration(
        #     joint_ids=[j],
        #     configuration=np.random.random((my_scene.num_envs, 1))*(limits[1]-limits[0])+limits[0]
        # )
        if "upper" in j:
            my_scene.update_configuration(
                joint_ids=[j],
                configuration=np.ones((my_scene.num_envs, 1))*(limits[1]-limits[0])+limits[0]
            )
        else:
            my_scene.update_configuration(
                joint_ids=[j],
                configuration=np.random.random((my_scene.num_envs, 1))*0.2*(limits[1]-limits[0])+limits[0]
            )

    # my_scene.label_containment(
    #     "drawer_upper_drawer",
    #     min_area=0.001
    #     # geom_ids="upper_drawer",
    # )
    # my_scene.show_containers()

    # my_scene.show_graph()

    my_scene.label_support(
        "drawer_support",
        geom_ids="upper_drawer",
    )
    # print (type(my_scene._scene.metadata["support_polygons"]["drawer_support"][0].polygon.exterior.coords[0])) # tuple
    # print (my_scene._scene.metadata["support_polygons"]["drawer_support"]) # tuple
    # my_scene.show_supports()
    # exit()

    build_constraint_st = time.time()
    constraint = SurfaceRelation(
        scene=my_scene,
        anchor_transforms=[TrackingTransform(parent_id="drawer")],
        base_support=my_scene._scene.metadata[
            "support_polygons"]["plane_support"],
        distance_type="greater",
        distance=0.1,
        # direction = np.array([1.0, 1.0]),
        max_mesh_projection_z=0.1,
        distance_start_bbox=False,
    )
    print(f"build constraint {time.time() - build_constraint_st}")

    my_scene.place_object(
        obj_asset=banana,
        obj_id="banana",
        support_id="plane_support",
        parent_id="plane",
        constraint=constraint
        # joint_type="fixed",
    )



    constraint_2 = SurfaceRelation(
        scene = my_scene,
        anchor_transforms = [
            TrackingTransform(parent_id="drawer"), 
            TrackingTransform(parent_id="banana")
        ],
        base_support = my_scene._scene.metadata["support_polygons"]["plane_support"],
        direction = "middle",
        distance_start_bbox = False,
    )
    my_scene.place_object(
        obj_asset=apple,
        obj_id="apple",
        support_id="plane_support",
        parent_id="plane",
        constraint = constraint_2
    )

    if i % 10 == 0:
        print(f"RAM after execution: {psutil.virtual_memory().percent}% used")
        print(f"Available RAM after execution: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    if VISUALIZE:
        # my_scene.show()
        # my_scene.show_graph()
        # print (my_scene._scene.graph.geometry_nodes)
        # print (my_scene._scene.graph.nodes)
        for i in range(0, 1):
            my_scene._scene.metadata["support_polygons"][f"constraint_support_{i}"] = constraint.scene_supports[i]
        # for i in range(0, 1):
            # my_scene._scene.metadata["support_polygons"][f"constraint_2_support_{i}"] = constraint_2.scene_supports[i]
        del my_scene._scene.metadata["support_polygons"]["plane_support"]
        my_scene.show_supports()
        my_scene.show(env_ids=np.arange(4), enable_viewer=not args.no_viewer)
    
    # mesh = my_scene._scene.to_mesh()
    # print (mesh)

    # del my_scene

    # my_scene.show_graph() # show scene graph
    # my_scene.show()


    # for j in my_scene.get_joint_names():
    #     print (j)
    #     limits = my_scene.get_joint_limits(joint_ids=[j])[0]

    #     my_scene.update_configuration(
    #         joint_ids=[j],
    #         configuration=[np.random.uniform(limits[0], limits[1]-0.2)]
    #     )

    # my_scene.show()
    # my_scene.show_supports()

    del my_scene

if __name__ == "__main__":
    # RAM stats before execution
    print(f"RAM before execution: {psutil.virtual_memory().percent}% used")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    start = time.time()
    if args.use_original_ss:
        pool = Pool(processes=10)
        pool.map(generate_scene, range(NUM_ENVS))
    else:
        generate_scene(0)
    print (time.time() - start)
