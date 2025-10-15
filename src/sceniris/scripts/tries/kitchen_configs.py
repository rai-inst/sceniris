# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import numpy as np

base_asset_path = os.path.expanduser("~/fm_storage/fm_assets/sceniris_benchmark_assets/Assets")
robocasa_asset_path = os.path.expanduser("~/fm_storage/fm_assets/robocasa_objects_usd_v2/robocasa_objects/objaverse")
robocasa_fixture_path = os.path.expanduser("~/fm_storage/fm_assets/robocasa_fixtures_usd_v2/robocasa_fixtures")

kitchen_config_1_env_size = 3.866
kitchen_config_1 = {
    "env": {
        "env_size": kitchen_config_1_env_size
    },
    "objects": [
        {
            "id": "dining_table",
            "asset_path": os.path.join(base_asset_path, "Table049", "Table049.usd"),
            "position":{
                "xy_limit": [[0.0, 0.0], [2.5/kitchen_config_1_env_size, 1.4/kitchen_config_1_env_size]], 
            },
            "rotation":{
                "type": "uniform_z",
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "base_cabinat",
            "asset_path": os.path.join(base_asset_path, "Kitchen_Cabinet002", "Kitchen_Cabinet002.usd"),
            "position": {
                "xy_limit": [[1.933/kitchen_config_1_env_size, 0.85/kitchen_config_1_env_size], [1.943/kitchen_config_1_env_size, 0.851/kitchen_config_1_env_size]], 
            },
            "rotation": {
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            # "fixed_world_positions": [np.array([1.933, 0.85, 0.0])]
        },
        {
            "id": "toaster",
            "asset_path": os.path.join(base_asset_path, "Toaster003", "Toaster003.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.5,
                "upper": 0.5
            },
            "position": {
                "xy_limit": [[0.0, 0.6], [0.3, 1.0]]
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "coffee_machine",
            "asset_path": os.path.join(base_asset_path, "CoffeeMachine006", "CoffeeMachine006.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            "ratio_on_support": 1.0
            # "constraints":[
            #     {
            #         "anchor_object_ids": ["toaster"],
            #         "direction": "right",
            #         "direction_tolerance_angle": 5.0,
            #         "distance": 0.6,
            #         "distance_type": "less"
            #     }
            # ]
        },
        {
            "id": "microwave",
            "asset_path": os.path.join(base_asset_path, "Microwave017", "Microwave017.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            "position": {
                "xy_limit": [[0.0, 0.0], [0.8, 1.0]]
            },
            "ratio_on_support": 1.0
            # "constraints":[
            #     {
            #         "anchor_object_ids": ["coffee_machine"],
            #         "direction": "right",
            #         "direction_tolerance_angle": 5.0,
            #         "distance": 0.7,
            #         "distance_type": "less"
            #     }
            # ]
        },
    ]
}

kitchen_config_2 = {
    "env": {
        "env_size": kitchen_config_1_env_size
    },
    "objects": [
        {
            "id": "dining_table",
            "asset_path": os.path.join(base_asset_path, "Table049", "Table049.usd"),
            "position":{
                "xy_limit": [[0.9, 0.3], [2.5/kitchen_config_1_env_size, 1.4/kitchen_config_1_env_size]], 
            },
            "rotation":{
                "type": "uniform_z",
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "teapot",
            "asset_path": os.path.join(robocasa_asset_path, "teapot", "teapot_5", "model", "model.usd"),
            "parent_id": "dining_table",
            "relation_to_parent": "top",
            "rotation": {
                "type": "uniform_z"
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "aloe",
            "asset_path": os.path.join(robocasa_fixture_path, "accessories", "plants", "aloe", "model", "model.usd"),
            "parent_id": "dining_table",
            "relation_to_parent": "top",
            "rotation": {
                "type": "uniform_z",
                "lower": -90.0,
                "upper": 90.0
            },
            "ratio_on_support": 1.0
        },
        # {
        #     "id": "chair_1",
        #     "asset_path": os.path.join(base_asset_path, "Chair", "chair3.usd"),
        #     "constraints":[
        #         {
        #             "anchor_object_ids": ["dining_table"],
        #             "distance": 0.6,
        #             "direction_tolerance_angle": 22.5,
        #             "direction": "left",
        #             "distance_type": "equal"
        #         }
        #     ],
        #     "rotation":{
        #         "type": "face_to",
        #         "face_to_object_id": "dining_table"
        #     },
        # },
        {
            "id": "chair_2",
            "asset_path": os.path.join(base_asset_path, "Chair", "chair3.usd"),
            "constraints":[
                {
                    "anchor_object_ids": ["dining_table"],
                    "distance": 0.8,
                    "direction_tolerance_angle": 22.5,
                    "direction": "right",
                    "distance_type": "equal"
                }
            ],
            "rotation":{
                "type": "face_to",
                "face_to_object_id": "dining_table"
            },
        },
        {
            "id": "chair_3",
            "asset_path": os.path.join(base_asset_path, "Chair", "chair3.usd"),
            "constraints":[
                {
                    "anchor_object_ids": ["dining_table"],
                    "distance": 0.8,
                    "direction": "front",
                    "direction_tolerance_angle": 22.5,
                    "distance_type": "equal"
                }
            ],
            "rotation":{
                "type": "face_to",
                "face_to_object_id": "dining_table"
            },
        },
        {
            "id": "base_cabinat",
            "asset_path": os.path.join(base_asset_path, "Kitchen_Cabinet002", "Kitchen_Cabinet002.usd"),
            "rotation": {
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            "fixed_world_positions": [np.array([1.933, 0.85, 0.0])]
        },
        # {
        #     "id": "top_cabinat",
        #     "asset_path": os.path.join(base_asset_path, "Kitchen_TopCabinet", "Kitchen_TopCabinet.usd"),
        #     "rotation": {
        #         "type": "uniform_z",
        #         "lower": -0.00001,
        #         "upper": 0.00001
        #     },
        #     "fixed_world_positions": [np.array([1.933*2, 0.85*2, 1.8])]
        # },
        {
            "id": "cutting_board",
            "asset_path": os.path.join(robocasa_asset_path, "cutting_board", "cutting_board_12", "model", "model.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "position": {
                "xy_limit": [[0.5, 0.0], [0.9, 0.4]]
            },
            "rotation": {
                "type": "uniform_z",
                "lower": 90.0,
                "upper": 90.001
            }
        },
        {
            "id": "knife",
            "asset_path": os.path.join(robocasa_asset_path, "knife", "knife_1", "model", "model.usd"),
            "parent_id": "base_cabinat",
            "constraints": [
                {
                    "anchor_object_ids": ["cutting_board"],
                    "direction": "right",
                    "direction_tolerance_angle": 10,
                    "distance": 0.4,
                    "distance_type": "less"
                }
            ]
        },
        {
            "id": "lemon",
            "asset_path": os.path.join(robocasa_asset_path, "lemon", "lemon_1", "model", "model.usd"),
            "parent_id": "cutting_board",
            "relation_to_parent": "top",
        },
        {
            "id": "lime",
            "asset_path": os.path.join(robocasa_asset_path, "lime", "lime_1", "model", "model.usd"),
            "parent_id": "cutting_board",
            "relation_to_parent": "top",
        },
        {
            "id": "bottled_drink_1",
            "asset_path": os.path.join(robocasa_asset_path, "bottled_drink", "bottled_drink_0", "model", "model.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "position": {
                "xy_limit": [[0.8, 0.8], [1.0, 1.0]]
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "bottled_drink_2",
            "asset_path": os.path.join(robocasa_asset_path, "bottled_drink", "bottled_drink_2", "model", "model.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "position": {
                "xy_limit": [[0.8, 0.8], [1.0, 1.0]]
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "bottled_drink_3",
            "asset_path": os.path.join(robocasa_asset_path, "bottled_drink", "bottled_drink_4", "model", "model.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "position": {
                "xy_limit": [[0.8, 0.8], [1.0, 1.0]]
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "toaster",
            "asset_path": os.path.join(base_asset_path, "Toaster003", "Toaster003.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.5,
                "upper": 0.5
            },
            "position": {
                "xy_limit": [[0.0, 0.3], [0.2, 1.0]]
            },
            "ratio_on_support": 1.0
        },
        {
            "id": "coffee_machine",
            "asset_path": os.path.join(base_asset_path, "CoffeeMachine006", "CoffeeMachine006.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            "ratio_on_support": 1.0,
            "constraints":[
                {
                    "anchor_object_ids": ["toaster"],
                    "direction": "right",
                    "direction_tolerance_angle": 30.0,
                    "distance": 0.6,
                    "distance_type": "less"
                }
            ]
        },
        {
            "id": "microwave",
            "asset_path": os.path.join(base_asset_path, "Microwave017", "Microwave017.usd"),
            "parent_id": "base_cabinat",
            "relation_to_parent": "top",
            "rotation":{
                "type": "uniform_z",
                "lower": -0.00001,
                "upper": 0.00001
            },
            "position": {
                "xy_limit": [[0.0, 0.0], [0.8, 1.0]]
            },
            "ratio_on_support": 1.0,
            # "constraints":[
            #     {
            #         "anchor_object_ids": ["coffee_machine"],
            #         "direction": "right",
            #         "direction_tolerance_angle": 30.0,
            #         "distance": 0.5,
            #         "distance_type": "greater"
            #     }
            # ]
        },
    ]
}