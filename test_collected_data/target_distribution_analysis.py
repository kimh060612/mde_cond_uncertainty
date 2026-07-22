from pathlib import Path
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import pandas as pd
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
from model.loss_target import ssi_independent_meter_space_depth_loss

### Unseen/Seen Degrade Distribution 차이 plotting
import sys

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "base_caminduce.yaml"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "loss_performance_correlation.csv"
DEFAULT_DATASET_ROOT = Path("/datasets/ATI/MDE/orbbec_realworld_dataset")
DEFAULT_REPLACEABLE_DATASET_PREFIXES = (
    "/datasets/ATI/MDE/orbbec_realworld_dataset",
)

CSV_DATA_PATH = "/home/kimh060612/ati_workspace/mde_cond_uncertainty/orbbec_canonical_parameter_frame_matches_by_scene"
csv_path_list = glob(f"{CSV_DATA_PATH}/*.csv")

DF_KEYS = {
    "s_abs_rel": "source_metric_abs_rel",
    "s_a1": "source_metric_a1",
    "c_abs_rel": "canonical_metric_abs_rel",
    "c_a1": "canonical_metric_a1",
    "s_exp_t": "source_exposure",
    "s_gain": "source_gain",
    "c_exp_t": "canonical_exposure",
    "c_gain": "canonical_gain",
    "r_depth_abs_diff": "raw_depth_mean_abs_diff",
    "depth_abs_diff": "depth_mean_abs_diff",
    "r_overlap": "registration_overlap_ratio",
    "r_ecc_score": "registration_ecc_score",
    "dtw_cost": "matched_lap_dtw_cost",
}

MOTION_STATE_SPACE = ["fast", "spin", "rotate"] # "stop", "slow", 
LIGHT_STATE_SPACE = ["normal", "dark", "dim"]
CONTEXT_STATE = [
    (s, l)
    for s in MOTION_STATE_SPACE
    for l in LIGHT_STATE_SPACE
]
SEEN_TOPOLOGY = ["topology1", "topology3"] # "topology4"
UNSEEN_TOPOLOGY = ["topology2", "topology5"]
TOPOLOGY_LIST = ["seen", "unseen"] # UNSEEN_TOPOLOGY + SEEN_TOPOLOGY

print(CONTEXT_STATE)
distribution_source_ABSREL = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}
distribution_source_Delta1 = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}
distribution_optimal_ABSREL = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}
distribution_optimal_Delta1 = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}

distribution_ssi_target_loss = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}
distribution_degradation_percent_abs_rel = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}
distribution_degradation_percent_a1 = {
    f"{s}_{l}": { t: [] for t in TOPOLOGY_LIST } 
    for s, l in CONTEXT_STATE
}

df_ssi_absrel_degrade = pd.DataFrame(columns=[
    "scene",
    "lap_dir",
    "frame_index",
    "motion", 
    "lighting", 
    "topology", 
    "source_exposure", 
    "source_gain",
    "optimal_exposure",
    "optimal_gain",
    "source_abs_rel",
    "optimal_abs_rel",
    "source_a1",
    "optimal_a1",
    "ssi_target_loss", 
    "abs_rel_degrade", 
    "a1_degrade"
])

def processor_stats(
    processor: AutoImageProcessor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = getattr(processor, "image_mean", None) or [0.485, 0.456, 0.406]
    std = getattr(processor, "image_std", None) or [0.229, 0.224, 0.225]
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return mean_tensor, std_tensor

def prepare_pixel_values(
    images: torch.Tensor,
    processor: AutoImageProcessor,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    if not normalize:
        return images
    mean, std = processor_stats(processor, device=images.device, dtype=images.dtype)
    return (images - mean) / std.clamp_min(1e-12)


def finite_stats(values: torch.Tensor) -> tuple[float, float, float, float]:
    values = values.detach().float().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.min().item()),
        float(values.max().item()),
    )

def _load_depth(path: Path, depth_scale: float=1.0) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Depth array not found: {path}")

    depth = np.load(path) / depth_scale
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return torch.from_numpy(np.asarray(depth, dtype=np.float32))

@torch.inference_mode()
def predict_depth(
    model: AutoModelForDepthEstimation,
    pixel_values: torch.Tensor,
    *,
    target_size: tuple[int, int],
    inference_batch_size: int,
    amp: bool,
    softplus: bool,
) -> torch.Tensor:
    depths: list[torch.Tensor] = []
    batch_size = max(1, int(inference_batch_size))

    for start in range(0, pixel_values.shape[0], batch_size):
        chunk = pixel_values[start : start + batch_size]
        with torch.autocast(device_type=chunk.device.type, enabled=amp):
            outputs = model(pixel_values=chunk)
            depth = outputs.predicted_depth

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(
            depth.float(),
            size=target_size,
            mode="bicubic",
            align_corners=False,
        )
        if softplus:
            depth = F.softplus(depth)
        depths.append(depth)

    return torch.cat(depths, dim=0)

def main():    
    model_id = MODEL_IDS.get("small", None)
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        cache_dir=None,
        local_files_only=False,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id,
        cache_dir=None,
        local_files_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    count_self_optimal = 0
    count_total = 0
    ### First, plotting distribution of 
    for csv_path in csv_path_list:
        csv_file_name = csv_path.split("/")[-1].split(".")[0]
        if csv_file_name.split("_")[3] == "normal": continue
        df = pd.read_csv(csv_path)
        print("Processing CSV file:", csv_file_name)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if not row["match_status"] == "matched": continue
            scene_name = row["scene"]
            lighting_cond = scene_name.split("_")[2]
            topology_number = scene_name.split("_")[-1]
            motion_cond = row["source_motion_label"]
            if not (motion_cond, lighting_cond) in CONTEXT_STATE: continue
            if not (topology_number in SEEN_TOPOLOGY) and not (topology_number in UNSEEN_TOPOLOGY): continue
            topology_type = "seen" if topology_number in SEEN_TOPOLOGY else "unseen"
            
            source_lap_idx = row["source_lap_dir"]
            source_frame_idx = row["source_frame_index"]
            
            distribution_optimal_ABSREL[f"{motion_cond}_{lighting_cond}"][topology_type].append(row["canonical_metric_abs_rel"])
            distribution_optimal_Delta1[f"{motion_cond}_{lighting_cond}"][topology_type].append(row["canonical_metric_a1"])
            
            distribution_source_ABSREL[f"{motion_cond}_{lighting_cond}"][topology_type].append(row["source_metric_abs_rel"])
            distribution_source_Delta1[f"{motion_cond}_{lighting_cond}"][topology_type].append(row["source_metric_a1"])
            source_exp_t = row["source_exposure"]
            source_gain = row["source_gain"]
            optimal_exp_t = row["canonical_exposure"]
            optimal_gain = row["canonical_gain"]
            
            canonical_abs_rel = row["canonical_metric_abs_rel"]
            if np.isfinite(canonical_abs_rel) and canonical_abs_rel > 1e-6 and \
                row["source_metric_abs_rel"] > 1e-6 and row["source_metric_a1"] > 1e-6 and \
                not (source_exp_t == optimal_exp_t and source_gain == optimal_gain):
                
                source_rgb_path = row["source_rgb_path"]
                canonical_rgb_path = row["matched_rgb_path"]
                source_depth_path = row["source_depth_path"]
                canonical_depth_path = row["matched_depth_path"]
                source_rgb = np.asarray(Image.open(source_rgb_path).convert("RGB").copy())
                canonical_rgb = np.asarray(Image.open(canonical_rgb_path).convert("RGB").copy())
                candidate_gt_depth = _load_depth(Path(source_depth_path), depth_scale=1000.0).unsqueeze(0).to(device)
                canonical_gt_depth = _load_depth(Path(canonical_depth_path), depth_scale=1000.0).unsqueeze(0).to(device)
                
                candidate_pixel_values = prepare_pixel_values(
                    torch.tensor(source_rgb).permute(2, 0, 1).unsqueeze(0),
                    processor=processor,
                    device=device,
                    normalize=True,
                )
                canonical_pixel_values = prepare_pixel_values(
                    torch.tensor(canonical_rgb).permute(2, 0, 1).unsqueeze(0),
                    processor=processor,
                    device=device,
                    normalize=True,
                )
                candidate_depth = predict_depth(
                    model,
                    candidate_pixel_values,
                    target_size=candidate_pixel_values.shape[-2:],
                    inference_batch_size=1,
                    amp=False,
                    softplus=True,
                )
                canonical_depth = predict_depth(
                    model,
                    canonical_pixel_values,
                    target_size=canonical_pixel_values.shape[-2:],
                    inference_batch_size=1,
                    amp=False,
                    softplus=True,
                )
                
                print(candidate_depth.shape, canonical_depth.shape, candidate_gt_depth.shape, canonical_gt_depth.shape)
                ssi_independent_meter_space_loss = ssi_independent_meter_space_depth_loss(
                    candidate_depth,
                    canonical_depth,
                    candidate_gt_depth.unsqueeze(1),
                    canonical_gt_depth.unsqueeze(1),
                )
                distribution_ssi_target_loss[f"{motion_cond}_{lighting_cond}"][topology_type].append(ssi_independent_meter_space_loss.item())
                ratio_Abs_rel_degrade = (row["source_metric_abs_rel"] - canonical_abs_rel) / canonical_abs_rel * 100.
                distribution_degradation_percent_abs_rel[f"{motion_cond}_{lighting_cond}"][topology_type].append(ratio_Abs_rel_degrade)
                ratio_A1_degrade = (row["canonical_metric_a1"] - row["source_metric_a1"]) / row["canonical_metric_a1"] * 100.
                distribution_degradation_percent_a1[f"{motion_cond}_{lighting_cond}"][topology_type].append(ratio_A1_degrade)
                df_ssi_absrel_degrade.loc[len(df_ssi_absrel_degrade)] = [
                    scene_name,
                    source_lap_idx,
                    source_frame_idx,
                    motion_cond,
                    lighting_cond,
                    topology_type,
                    source_exp_t,
                    source_gain,
                    optimal_exp_t,
                    optimal_gain,
                    row["source_metric_abs_rel"],
                    canonical_abs_rel,
                    row["source_metric_a1"],
                    row["canonical_metric_a1"],
                    ssi_independent_meter_space_loss.item(),
                    ratio_Abs_rel_degrade,
                    ratio_A1_degrade
                ]
            else:
                df_ssi_absrel_degrade.loc[len(df_ssi_absrel_degrade)] = [
                    scene_name,
                    source_lap_idx,
                    source_frame_idx,
                    motion_cond,
                    lighting_cond,
                    topology_type,
                    source_exp_t,
                    source_gain,
                    optimal_exp_t,
                    optimal_gain,
                    row["source_metric_abs_rel"],
                    canonical_abs_rel,
                    row["source_metric_a1"],
                    row["canonical_metric_a1"],
                    float("0.0"),
                    float("0.0"),
                    float("0.0")
                ]
            
            if source_exp_t == optimal_exp_t and source_gain == optimal_gain:
                count_self_optimal += 1
            count_total += 1

    print(f"Number of self canonical frames: {count_self_optimal} / {count_total}")
    df_ssi_absrel_degrade.to_csv("./output/ssi_absrel_degrade_distribution.csv", index=False)
    
    for s, l in CONTEXT_STATE:
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 5))
        print(f"Distribution of {s}, {l}")
        pooled_degrade_absrel = np.concatenate([
            np.asarray(
                distribution_degradation_percent_abs_rel[f"{s}_{l}"][t],
                dtype=float,
            )
            for t in TOPOLOGY_LIST
        ])
        pooled_degrade_absrel = pooled_degrade_absrel[np.isfinite(pooled_degrade_absrel)]
        absrel_upper = np.quantile(pooled_degrade_absrel, 0.99)
        pooled_ssi_target_loss = np.concatenate([
            np.asarray(
                distribution_ssi_target_loss[f"{s}_{l}"][t],
                dtype=float,
            )
            for t in TOPOLOGY_LIST
        ])
        pooled_ssi_target_loss = pooled_ssi_target_loss[np.isfinite(pooled_ssi_target_loss)]
        ssi_target_loss_upper = np.quantile(pooled_ssi_target_loss, 0.99)
        common_absrel_metric_bins = np.linspace(0.0, 0.6, 41)
        common_degrade_absrel_bins = np.linspace(0.0, absrel_upper, 101)
        common_ssi_target_loss_bins = np.linspace(0.0, ssi_target_loss_upper, 101)
        topology_colors = {"seen": "royalblue", "unseen": "darkorange"}
        topology_colors_degrade = {"seen": "mediumseagreen", "unseen": "goldenrod"}
        
        for t in TOPOLOGY_LIST:
            np_optimal_absrel = np.array(distribution_optimal_ABSREL[f"{s}_{l}"][t])
            np_optimal_a1 = np.array(distribution_optimal_Delta1[f"{s}_{l}"][t])
            np_source_absrel = np.array(distribution_source_ABSREL[f"{s}_{l}"][t])
            np_source_a1 = np.array(distribution_source_Delta1[f"{s}_{l}"][t])
            np_degrade_absrel = np.array(distribution_degradation_percent_abs_rel[f"{s}_{l}"][t])
            np_degrade_a1 = np.array(distribution_degradation_percent_a1[f"{s}_{l}"][t])
            np_ssi_target_loss = np.array(distribution_ssi_target_loss[f"{s}_{l}"][t])
            
            df_whole_dataset = pd.DataFrame({
                "optimal_abs_rel": np_optimal_absrel,
                "optimal_a1": np_optimal_a1,
                "source_abs_rel": np_source_absrel,
                "source_a1": np_source_a1,
            })
            df_degrade = pd.DataFrame({
                "degrade_abs_rel": np_degrade_absrel,
                "degrade_a1": np_degrade_a1,
                "ssi_target_loss": np_ssi_target_loss
            })
            df_degrade.loc[
                ~df_degrade["degrade_abs_rel"].between(0.0, absrel_upper),
                "degrade_abs_rel",
            ] = np.nan
            
            print(np_optimal_absrel.min(), np_optimal_absrel.max())
            print(np_source_absrel.min(), np_source_absrel.max())
            print(np_optimal_a1.min(), np_optimal_a1.max())
            print(np_source_a1.min(), np_source_a1.max())
            print(np_degrade_absrel.shape, np_degrade_a1.shape)
            
            sns.histplot(
                data=df_whole_dataset, 
                x='optimal_abs_rel', 
                stat='probability',  # Standardizes the y-axis so total area equals 1
                kde=True,        # Automatically superimposes a smooth density curve
                color=topology_colors[t], 
                bins=common_absrel_metric_bins, 
                label=t,
                ax=axs[0]        # Hooks directly into the first Matplotlib sub-axis
            )
            axs[0].set_title('Distribution of Optimal Abs_rel')
            axs[0].set_xlabel('Abs_rel Range')
            axs[0].set_xlim(0.0, 0.6)    
            
            sns.histplot(
                data=df_whole_dataset, 
                x='source_abs_rel', 
                stat='probability',  # Standardizes the y-axis so total area equals 1
                kde=True,        # Automatically superimposes a smooth density curve
                color=topology_colors[t], 
                bins=common_absrel_metric_bins, 
                label=t,
                ax=axs[1]        # Hooks directly into the first Matplotlib sub-axis
            )
            axs[1].set_title('Distribution of Source Abs_rel')
            axs[1].set_xlabel('Abs_rel Range')
            axs[1].set_xlim(0.0, 0.6)
            
            sns.histplot(
                data=df_degrade, 
                x='degrade_abs_rel', 
                stat='probability',  # Standardizes the y-axis so total area equals 1
                kde=True,        # Automatically superimposes a smooth density curve
                color=topology_colors[t], 
                bins=common_degrade_absrel_bins, 
                label=t,
                ax=axs[2]        # Hooks directly into the first Matplotlib sub-axis
            )
            axs[2].set_title('Distribution of degrade Abs_rel')
            axs[2].set_xlabel('Abs_rel Range')
            axs[2].set_xlim(0.0, absrel_upper)
            
            sns.histplot(
                data=df_degrade, 
                x='ssi_target_loss',
                stat='probability',  # Standardizes the y-axis so total area equals 1
                kde=True,        # Automatically superimposes a smooth density curve
                color=topology_colors_degrade[t], 
                bins=common_ssi_target_loss_bins, 
                label=t,
                ax=axs[3]        # Hooks directly into the first Matplotlib sub-axis
            )
            axs[3].set_title('Distribution of SSI Target Loss')
            axs[3].set_xlabel('SSI Target Loss Range')
            axs[3].set_xlim(0.0, ssi_target_loss_upper)
            
        
        # 5. Clean up spacing and display
        axs[0].legend(title="Topology")
        axs[1].legend(title="Topology")
        axs[2].legend(title="Topology")
        axs[3].legend(title="Topology")
        plt.tight_layout()
        plt.savefig(f"./output/distribution/distribution_analysis_{s}_{l}.png", dpi=300)


if __name__ == "__main__":
    main()