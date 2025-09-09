import os
import numpy as np
distance_start_bbox = False

base_asset_path = os.path.expanduser("~/fm_storage/fm_assets/sceniris_benchmark_assets/Assets")
test_asset_path = os.path.expanduser("~/fm_storage/fm_assets/scene_gen_test_asset/Assets")
robocasa_asset_path = os.path.expanduser("~/fm_storage/fm_assets/robocasa_objects_usd_v2/robocasa_objects/objaverse")
robocasa_fixture_path = os.path.expanduser("~/fm_storage/fm_assets/robocasa_fixtures_usd_v2/robocasa_fixtures")

config_4 = {
    "objects": [
        {
            "id": "orange",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/017_orange/textured/textured.usd"),
            "constraints": [
                {
                    "anchor_object_ids": ["drawer", "mug"],
                    "direction": "middle",
                    "distance_start_bbox": distance_start_bbox,
                }
            ]
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
            ],
            "position":{
                "xy_limit": [[0.0, 0.5], [1.0, 1.0]], # back (+y) half of the table 
            },
            "rotation":{
                "type": "uniform_z",
                "lower": -15.0,
                "upper": 15.0
            }
        },
        {
            "id": "banana",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
            # "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/017_orange/textured/textured.usd"),
            "parent_id": "drawer/unit_1_upper_drawer",
            "relation_to_parent": "top",
        },
        {
            "id": "mug",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/025_mug/textured/textured.usd"),
            "constraints": [
                {
                    "anchor_object_ids": ["drawer"],
                    "direction": "front",
                    "distance": 0.4,
                    "distance_type": "greater",
                    "relation_axis_transform": "drawer",
                    "distance_start_bbox": distance_start_bbox,
                }
            ]
        },
    ]
}

table_top_config_2 = {
   "env": {
        "env_size": 5.0
    },
    "objects": [
        {
            "id": "table",
            "asset_path": os.path.join(base_asset_path, "Table", "table2.usd"),
            "rotation":{
                "type": "uniform_z",
                "lower": -5.0,
                "upper": 5.0
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "drawer",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/usd/unit_1/unit_1.usd"),
            "parent_id": "table",
            "relation_to_parent": "top",
            "joint_states": [
                {
                    "joint_ids": ["drawer_joint_lower"],
                    "max": 1.0,
                    "min": 0.7,
                    "distribution": "uniform",
                },
                {
                    "joint_ids": ["drawer_joint_upper"],
                    "max": 0.3,
                    "min": 0.1,
                    "distribution": "uniform",
                },
            ],
            "ratio_on_support": 1.0
        },
        {
            "id": "banana",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/011_banana/textured/textured.usd"),
            "parent_id": "drawer",
            "relation_to_parent": "top",
        },
        {
            "id": "mug",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/025_mug/textured/textured.usd"),
            "parent_id": "table",
            "constraints": [
                {
                    "anchor_object_ids": ["drawer"],
                    "direction": "right",
                    "distance": 0.5,
                    "distance_type": "less",
                    "relation_axis_transform": "drawer",
                }
            ]
        },
        {
            "id": "lemon",
            "asset_path": os.path.expanduser("~/fm_storage/fm_assets/ycb_fixed_v2/014_lemon/textured/textured.usd"),
            "parent_id": "drawer",
            "relation_to_parent": "inside",
        },
        {
            "id": "chair",
            "asset_path": os.path.join(base_asset_path, "Chair", "chair5.usd"),
            "constraints":[
                {
                    "anchor_object_ids": ["table"],
                    "distance": 0.8,
                    "direction_tolerance_angle": 10.0,
                    "direction": "front",
                    "distance_type": "equal"
                }
            ],
            "rotation":{
                "type": "face_to",
                "face_to_object_id": "table"
            },
        },
    ] 
}