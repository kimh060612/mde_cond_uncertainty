from PIL import Image
import numpy as np
import json
import csv
import os
from tqdm import tqdm

DATA_PATH = "/issac-sim/dataset/realworld_dataset/evaluation_auto_exposure"
SCENE_TYPES = [ 
    f"ae_scene2_{x}_{y}" 
    for x in ["normal", "dark", "dim"] 
    for y in ["fast", "slow"]
]
# EXPOSURE_SETS = [10, 20, 40, 80, 160, 320]
# GAIN_SETS = [16, 32, 64, 128]
SUB_DIR_NAMES = [""]


if __name__ == "__main__":
    for scene in SCENE_TYPES:
        print(f"Scene: {scene}")
        capture_config = json.load(open(os.path.join(DATA_PATH, scene, "capture_config.json"), "r"))
        N_frames = capture_config["num_frames"]
        reward_metric_list = []
        valid_metric_frames = 0
        skipped_uninformative = 0
        skipped_examples = []
        for sub_dir in tqdm(SUB_DIR_NAMES, desc=f"Processing {scene}"):
            if not os.path.exists(os.path.join(DATA_PATH, scene, sub_dir, "metrics.json")):
                raise ValueError(f"Metrics not found for {scene} {sub_dir}, please run the evaluation first.")
            metric_rows = json.load(open(os.path.join(DATA_PATH, scene, sub_dir, "metrics.json"), "r"))
            rgb_path_dir = os.path.join(DATA_PATH, scene, sub_dir, "rgb")
            pred_depth_path_dir = os.path.join(DATA_PATH, scene, sub_dir, "pred_depth")
            
            a1_list = [row["a1"] for row in metric_rows if row["a1"] >= 0. and row["abs_rel"] >= 0.]
            abs_rel_list = [row["abs_rel"] for row in metric_rows if row["a1"] >= 0. and row["abs_rel"] >= 0.]
            if len(a1_list) == 0 or len(abs_rel_list) == 0:
                print(f"  No valid metric frames found in {scene} {sub_dir}, skipping.")
                continue
            
            mean_a1 = np.mean(a1_list)
            mean_abs_rel = np.mean(abs_rel_list)
            print(f"[scene] {scene}  Mean a1: {mean_a1:.4f}, Mean abs_rel: {mean_abs_rel:.4f}")

    
