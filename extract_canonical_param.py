import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoImageProcessor
from evaluation_utils.eval_metrics import compute_comprehensive_depth_metrics
from evaluation_utils.eval_utils import align_relative_prediction_to_depth_space
from model.dav2_ati_model import MODEL_IDS
from utils.train_utils import seed_everything




@hydra.main(config_path="configs", config_name="extract_canonical_param")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_IDS[cfg.model.model_id]
    print(f"Using model: {model_id}")
    print(f"dataset root: {cfg.dataset.dataset_root}")
    seed = cfg.training.seed
    seed_everything(seed)
    
    
    
    
if __name__ == "__main__":
    main()