from argparse import ArgumentParser
import os
import numpy as np

import time
from multiprocessing import Pool
import psutil

from scene_gen_asset import PlaneAsset
from utils import scene_graph_transform_get
from constraints import SurfaceRelation, TrackingTransform

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
    from scene_gen_scene import Scene
    from scene_gen_asset import Asset

from pose_generators import PositionIteratorList


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

    # print(my_scene._scene.metadata["support_polygons"]["plane_support"])

    drawer = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
        origin=("center", "center", "bottom"),
    )

    banana = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    mug = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/025_mug/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    orange = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/017_orange/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    apple = Asset(
        os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/013_apple/textured/textured.usd"),
        origin=("center", "center", "bottom"),
    )

    drawer_position_iterator = PositionIteratorList(
        positions=np.array([[0.5, 0.8]])
    )
    drawer_position_iterator(my_scene._scene.metadata["support_polygons"]["plane_support"][0])
    my_scene.place_object(
        obj_asset=drawer,
        obj_id="drawer",
        parent_id="plane",
        obj_position_iterator=drawer_position_iterator,
        # joint_type="fixed",
    )
    # for j in my_scene.get_joint_names():
    #     limits = my_scene.get_joint_limits(joint_ids=[j])[0]

    #     my_scene.update_configuration(
    #         joint_ids=[j],
    #         configuration=np.random.random((my_scene.num_envs, 1))*(limits[1]-limits[0])+limits[0]
    #     )
    
    banana_position_iterator = PositionIteratorList(
        positions=np.array([[0.2, 0.3]])
    )
    banana_position_iterator(my_scene._scene.metadata["support_polygons"]["plane_support"][0])
    mug_position_iterator = PositionIteratorList(
        positions=np.array([[0.8, 0.3]])
    )
    mug_position_iterator(my_scene._scene.metadata["support_polygons"]["plane_support"][0])
    orange_position_iterator = PositionIteratorList(
        positions=np.array([[0.5, 0.2]])
    )
    orange_position_iterator(my_scene._scene.metadata["support_polygons"]["plane_support"][0])
    my_scene.place_object(
        obj_asset=banana,
        obj_id="banana",
        parent_id="plane",
        obj_position_iterator=banana_position_iterator,
    )
    my_scene.place_object(
        obj_asset=mug,
        obj_id="mug",
        parent_id="plane",
        obj_position_iterator=mug_position_iterator,
    )
    my_scene.place_object(
        obj_asset=orange,
        obj_id="orange",
        parent_id="plane",
        obj_position_iterator=orange_position_iterator,
    )

    my_scene.label_support(
        "drawer_support",
        geom_ids="upper_drawer",
    )

    drawer_transform = scene_graph_transform_get(my_scene._scene.graph, "drawer", edge_batch=my_scene.edge_batch, cache=my_scene.cache)[0]
    banana_transform = scene_graph_transform_get(my_scene._scene.graph, "banana", edge_batch=my_scene.edge_batch, cache=my_scene.cache)[0]
    mug_transform = scene_graph_transform_get(my_scene._scene.graph, "mug", edge_batch=my_scene.edge_batch, cache=my_scene.cache)[0]
    orange_transform = scene_graph_transform_get(my_scene._scene.graph, "orange", edge_batch=my_scene.edge_batch, cache=my_scene.cache)[0]
    constraint = SurfaceRelation(
        scene = my_scene,
        anchor_transforms = [
            TrackingTransform(parent_id="drawer", transform=drawer_transform), 
            TrackingTransform(parent_id="banana", transform=banana_transform),
            TrackingTransform(parent_id="mug", transform=mug_transform),
            TrackingTransform(parent_id="orange", transform=orange_transform),
        ],
        base_support = my_scene._scene.metadata["support_polygons"]["plane_support"],
        direction = "middle"
    )
    my_scene.place_object(
        obj_asset=apple,
        obj_id="apple",
        support_id="plane_support",
        parent_id="plane",
        constraint = constraint
    )

    if i % 10 == 0:
        print(f"RAM after execution: {psutil.virtual_memory().percent}% used")
        print(f"Available RAM after execution: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    if VISUALIZE:
        for i in range(0, 1):
            my_scene._scene.metadata["support_polygons"][f"constraint_support_{i}"] = constraint.scene_supports[i]
        del my_scene._scene.metadata["support_polygons"]["plane_support"]
        my_scene.show_supports()
        my_scene.show(env_ids=np.arange(4), enable_viewer=not args.no_viewer)

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
