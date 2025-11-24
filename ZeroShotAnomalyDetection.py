"""
zero_shot_dinov3.py

Zero-shot anomaly detection and embedding visualization with a local DINOv3 model.
No CLIP required.

Usage example:
python zero_shot_dinov3.py --dataset mvtec --category bottle --repo_dir ./dinov3 \
    --weights ./dinov3_vits16_pretrain_lvd1689m-08c60483.pth --result_path ./Result/zs --batch_size 8
"""

import os
import math
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Optional: reuse your dataset registry if available
try:
    from Datasets import DATASET_REGISTRY
except Exception:
    DATASET_REGISTRY = None
    print("Warning: Datasets registry not found. You can still run with image folder input if needed.")

# ----------------------
# Utilities
# ----------------------
def load_dinov3_local(repo_dir, model_name, weights, device):
    """
    Load a local DINOv3 model via torch.hub (repo_dir points to the dinov3 repo).
    model_name: e.g. 'dinov3_vits16' or 'dinov3_vitl16'
    weights: path to local checkpoint
    """
    model = torch.hub.load(repo_dir, model_name, source='local', weights=weights)
    model.to(device)
    model.eval()
    return model

def extract_patch_embeddings(dino_model, images, layers=None, device='cpu'):
    """
    Run images through dino_model and collect patch embeddings from requested layers.
    Returns:
        concat: (B, N, C_total)
        patch_grid: (H_patches, W_patches)
    """
    # Prepare hooks
    features = {}
    num_blocks = len(dino_model.blocks)
    if layers is None:
        layers = sorted(list({max(0, num_blocks//4 - 1), max(0, num_blocks//2 - 1), max(0, num_blocks - 1)}))
    layers = [min(max(0, int(l)), num_blocks-1) for l in layers]

    def make_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, (list, tuple)):
                out = out[0]
            features[idx] = out.detach()
        return hook

    handles = [dino_model.blocks[l].register_forward_hook(make_hook(l)) for l in layers]

    with torch.no_grad():
        _ = dino_model(images)

    for h in handles:
        h.remove()

    # Collect features and remove CLS token
    layer_feats = []
    for l in layers:
        t = features[l]
        if t.dim() == 3:
            t = t[:, 1:, :]  # remove cls token
        else:
            raise RuntimeError(f"Unexpected token dims: {t.shape}")
        layer_feats.append(t)

    concat = torch.cat(layer_feats, dim=2)  # (B, N, C_total)
    B, N, C_total = concat.shape

    # Get actual patch grid size from model
    patch_size = dino_model.patch_embed.patch_size  # e.g., (16,16)
    H_img, W_img = images.shape[2], images.shape[3]
    H_patches = math.ceil(H_img / patch_size[0])
    W_patches = math.ceil(W_img / patch_size[1])

    # Safety: trim extra tokens if N > H_patches*W_patches
    expected_N = H_patches * W_patches
    if N > expected_N:
        concat = concat[:, :expected_N, :]
    elif N < expected_N:
        # optional: pad with zeros
        pad = torch.zeros(B, expected_N - N, C_total, device=concat.device)
        concat = torch.cat([concat, pad], dim=1)

    return concat, (H_patches, W_patches)


def per_image_patch_anomaly_scores(patch_feats, method='cosine_to_mean'):
    """
    Compute anomaly score per patch for each image.
    patch_feats: (B, N, C)
    method: 'cosine_to_mean' (default) computes 1 - cosine(similarity to mean patch embedding)
    Returns: (B, N) anomaly scores in [0, 2]
    """
    B, N, C = patch_feats.shape
    if method == 'cosine_to_mean':
        # L2 normalize
        pf = F.normalize(patch_feats, dim=2)  # (B,N,C)
        mean = F.normalize(pf.mean(dim=1, keepdim=True), dim=2)  # (B,1,C)
        sim = (pf * mean).sum(dim=2)  # (B,N) cosine similarity in [-1,1]
        score = 1.0 - sim  # higher -> more anomalous (range 0..2)
        return score
    elif method == 'mahalanobis':
        # Mahalanobis across patches: build cov of patch-feats per image
        scores = torch.zeros((B, N), device=patch_feats.device)
        for i in range(B):
            X = patch_feats[i]  # (N,C)
            mu = X.mean(dim=0, keepdim=True)  # (1,C)
            Xc = X - mu  # (N,C)
            cov = (Xc.t() @ Xc) / (N - 1 + 1e-6)  # (C,C)
            # regularize cov
            cov += 1e-6 * torch.eye(C, device=X.device)
            inv = torch.linalg.inv(cov)
            # Mahalanobis distance per patch
            d = torch.einsum('nc,cd,nd->n', Xc, inv, Xc)  # (N,)
            scores[i] = d
        # normalize per image
        scores = (scores - scores.min(dim=1, keepdim=True)[0]) / (scores.ptp(dim=1, keepdim=True) + 1e-8)
        return scores
    else:
        raise ValueError("Unknown method")

def upsample_score_map(scores, patch_grid, out_size):
    """
    Reshape patch scores to correct grid and upsample.
    patch_grid: (H_patches, W_patches)
    out_size: (H_img, W_img)
    """
    single = False
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        single = True
    B, N = scores.shape
    H_patches, W_patches = patch_grid

    # Safety: trim or pad to match expected N
    expected_N = H_patches * W_patches
    if N > expected_N:
        scores = scores[:, :expected_N]
    elif N < expected_N:
        pad = torch.zeros(B, expected_N - N, device=scores.device)
        scores = torch.cat([scores, pad], dim=1)

    scores_2d = scores.view(B, 1, H_patches, W_patches)
    up = F.interpolate(scores_2d, size=out_size, mode="bilinear", align_corners=False)
    return up.squeeze(1)


def visualize_and_save(image_tensors, heatmaps, image_paths, result_dir):
    """
    image_tensors: (B,3,H,W) normalized in ImageNet stats
    heatmaps: (B,H,W) values in range roughly [0,1] (we'll normalize)
    image_paths: list of original file paths (for naming)
    Saves overlay visualizations to result_dir.
    """
    os.makedirs(result_dir, exist_ok=True)
    # # inverse normalize (ImageNet)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])


    B = image_tensors.shape[0]
    for i in range(B):
        img = image_tensors[i].cpu().numpy().transpose(1,2,0)  # H W C (float, normalized)
        print(img)
        # img = (img * std) + mean
        # img = image_tensors[i].cpu().numpy().transpose(1,2,0)
        # img = (img * 255).clip(0,255).astype(np.uint8)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        hm = heatmaps[i].cpu().numpy()
        # normalize heatmap 0..1
        hm = (hm - hm.min()) / (hm.ptp() + 1e-8)
        hm_uint8 = (hm * 255).astype(np.uint8)
        # print(hm_uint8)
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img[:, :, ::-1], 0.6, hm_color, 0.4, 0)  # convert RGB->BGR for OpenCV draw

        # write side-by-side: original | overlay | heatmap
        hm_map_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        side = np.hstack([img[:, :, ::-1], overlay, hm_map_rgb])  # BGR order consistent
        fname = os.path.basename(image_paths[i])
        outp = os.path.join(result_dir, f"zs_{fname}")
        cv2.imwrite(outp, side)
    return

# ----------------------
# Main pipeline using DATASET_REGISTRY if available (like your test script)
# ----------------------
def prepare_data_from_registry(dataset_name, category, batch_size, device, resize=512, imagesize=512, **kwargs):
    dataset_name = dataset_name.lower()
    if DATASET_REGISTRY is None:
        raise RuntimeError("DATASET_REGISTRY not available. Provide images differently.")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
    test_dataset = dataset_cls(source=root_path, split=split_cls.TEST, classname=category, resize=resize, imagesize=imagesize)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return loader

# ----------------------
# CLI Entrypoint
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default="./dinov3", help="local dinov3 repo dir for torch.hub")
    parser.add_argument("--weights", type=str, required=True, help="local dinov3 weights file")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="dinov3_vits16 or dinov3_vitl16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--result_path", type=str, default="./Result/zs")
    parser.add_argument("--method", type=str, choices=['cosine_to_mean','mahalanobis'], default='cosine_to_mean')
    parser.add_argument("--layers", type=str, default=None, help="comma separated layer indices (optional)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_path, exist_ok=True)

    # load dinov3
    print("Loading DINOv3 model...")
    Dino = load_dinov3_local(args.repo_dir, args.model_name, args.weights, device)
    print("Model loaded. Number of transformer blocks:", len(Dino.blocks))

    # prepare dataset loader
    try:
        dataloader = prepare_data_from_registry(args.dataset, args.category, args.batch_size, device, resize=512, imagesize=512)
    except Exception as e:
        raise RuntimeError("Failed to prepare dataset from registry. Ensure Datasets package is available.") from e

    # parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None

    # iterate
    for i, batch in enumerate(tqdm(dataloader, desc="Batches")):
        if i > 1:
            break
        images = batch["image"].to(device)  # (B,3,H,W), normalized
        image_paths = batch["image_path"]
        # get patch embeddings
        patch_feats, (Hp, Wp) = extract_patch_embeddings(Dino, images, layers=layers, device=device)
        # compute anomaly scores per patch
        scores = per_image_patch_anomaly_scores(patch_feats, method=args.method)
        # normalize scores per image into 0..1
        scores = (scores - scores.min(dim=1, keepdim=True)[0]) / (scores.max(dim=1, keepdim=True)[0] - scores.min(dim=1, keepdim=True)[0] + 1e-8)
        H, W = images.shape[2], images.shape[3]
        up = upsample_score_map(scores, (Hp, Wp), (H, W))  # (B,H,W)
        # visualize & save
        visualize_and_save(images, up, image_paths, args.result_path)

    print("Done. Visualizations saved to", args.result_path)

if __name__ == "__main__":
    main()
