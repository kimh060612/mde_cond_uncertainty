# infer_da2_gaussian_nll.py

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from model.dav2_model import GaussianDepthAnythingV2, MODEL_IDS


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="metric-indoor-small", choices=list(MODEL_IDS.keys()))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_depth", type=str, default="pred_depth.npy")
    parser.add_argument("--out_uncertainty", type=str, default="pred_uncertainty.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = MODEL_IDS[args.model]

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = GaussianDepthAnythingV2(model_id=model_id).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    target_size = image.size[::-1]  # PIL: W,H -> H,W

    out = model(pixel_values, target_size=target_size)

    depth = out["mu"][0, 0].cpu().numpy()
    uncertainty_std = out["std"][0, 0].cpu().numpy()
    uncertainty_var = out["var"][0, 0].cpu().numpy()

    np.save(args.out_depth, depth)
    np.save(args.out_uncertainty, uncertainty_std)

    print("depth:", depth.shape, depth.min(), depth.max())
    print("std uncertainty:", uncertainty_std.shape, uncertainty_std.min(), uncertainty_std.max())
    print("var uncertainty:", uncertainty_var.shape, uncertainty_var.min(), uncertainty_var.max())


if __name__ == "__main__":
    main()