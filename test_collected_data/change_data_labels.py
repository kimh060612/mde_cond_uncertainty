from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob

META_DATA_PATH = "/datasets/ATI/MDE/orbbec_realworld_dataset"
CSV_DATA_PATH = "/home/kimh060612/ati_workspace/mde_cond_uncertainty/orbbec_canonical_parameter_frame_matches_by_scene"
SCENE_RENAMES = {
    "comlab_scene_dark_normal_topology2": "comlab_scene_dark_normal_topology3",
    "comlab_scene_dim_normal_topology2": "comlab_scene_dim_normal_topology3",
    "comlab_scene_normal_normal_topology2": "comlab_scene_normal_normal_topology3",
}

ROW_PREFIX = [
    "scene",
    "source_rgb_path",
    "source_depth_path",
    "matched_rgb_path",
    "matched_depth_path"
]

if __name__ == "__main__":
    
    csv_path_list = glob(f"{CSV_DATA_PATH}/comlab_scene_*_normal_topology3_canonical_frame_matches.csv")
    meta_data_path_list = glob(f"{META_DATA_PATH}/comlab_scene_*_normal_topology3/*/*/metadata/*.json")
    
    for csv_path in csv_path_list:
        csv_path = Path(csv_path)
        text = csv_path.read_text()
        updated_text = text
        for scene, rename_target in SCENE_RENAMES.items():
            updated_text = updated_text.replace(scene, rename_target)
        if updated_text != text:
            csv_path.write_text(updated_text)
    
    for meta_data_path in meta_data_path_list:
        meta_data_path = Path(meta_data_path)
        text = meta_data_path.read_text()
        updated_text = text
        for scene, rename_target in SCENE_RENAMES.items():
            updated_text = updated_text.replace(scene, rename_target)
        if updated_text != text:
            meta_data_path.write_text(updated_text)
