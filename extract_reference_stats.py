#!/usr/bin/env python3
"""
Extract Inception-v3 features from real Defactify images
and save reference statistics for FID/KID computation.

Usage:
    python extract_reference_stats.py \
        --output-dir _oupipeline_v5tputs \
        --n-samples 5000 \
        --batch-size 32

This script generates reference_stats.npz with:
  - mu: (2048,) mean features
  - sigma: (2048, 2048) covariance matrix
  - n_images: number of images processed
  - feature_dim: 2048 (Inception-v3 avg pool size)
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# Handle HF dataset loading
try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("extract_reference_stats")


def get_inception_model(device):
    """Load pre-trained Inception-v3 model, remove classification head."""
    log.info("Loading Inception-v3 model...")
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        aux_logits=False
    )
    inception.eval()
    # Remove final classification layer to get 2048-D features
    inception = torch.nn.Sequential(*list(inception.children())[:-1])
    return inception.to(device)


def get_image_transform():
    """Standard ImageNet preprocessing for Inception-v3."""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features_batch(images, inception_model, device, transform):
    """Extract Inception features from image batch.
    
    Args:
        images: List of PIL Images
        inception_model: Inception-v3 without head
        device: torch device
        transform: image preprocessing function
        
    Returns:
        features: (N, 2048) numpy array
    """
    # Convert images and apply transform
    tensors = torch.stack([
        transform(img.convert('RGB')) for img in images
    ])
    
    with torch.no_grad():
        features = inception_model(tensors.to(device))
    
    return features.cpu().numpy().reshape(len(images), -1)


def load_defactify_dataset():
    """Load Defactify dataset from HuggingFace or local cache."""
    if not HAS_HF_DATASETS:
        log.error("datasets package not found. Install with: pip install datasets")
        return None
    
    try:
        log.info("Loading Defactify dataset (train split with real images)...")
        dataset = load_dataset(
            "Rajarshi-Roy-research/Defactify_Image_Dataset",
            split="train"
        )
        log.info(f"✓ Loaded dataset with {len(dataset)} images")
        return dataset
    except Exception as e:
        log.error(f"Failed to load from HuggingFace: {e}")
        log.warning("Attempting to load from local cache...")
        return None


def extract_features_from_dataset(dataset, inception_model, device, transform, 
                                   n_samples=5000, batch_size=32):
    """Extract features from HF dataset."""
    all_features = []
    image_count = 0
    
    log.info(f"Extracting features from up to {n_samples} images (batch_size={batch_size})...")
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataset.iter(batch_size=batch_size))):
            if image_count >= n_samples:
                break
            
            # Get images from batch
            try:
                if "image" in batch:
                    images = batch["image"]
                elif "img" in batch:
                    images = batch["img"]
                else:
                    # Try first image-like column
                    image_cols = [k for k in batch.keys() 
                                 if isinstance(batch[k][0], Image.Image)]
                    if image_cols:
                        images = batch[image_cols[0]]
                    else:
                        log.warning(f"No image column found. Available: {list(batch.keys())}")
                        continue
                
                # Convert to PIL if needed
                images = [
                    img if isinstance(img, Image.Image) else Image.fromarray(img)
                    for img in images
                ]
                
                # Extract features
                batch_features = extract_features_batch(
                    images, inception_model, device, transform
                )
                all_features.extend(batch_features)
                image_count += len(images)
                
            except Exception as e:
                log.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    except Exception as e:
        log.error(f"Error during feature extraction: {e}")
    
    if not all_features:
        return None
        
    return np.array(all_features, dtype=np.float32)


def extract_features_from_directory(image_dir, inception_model, device, transform,
                                     n_samples=5000):
    """Extract features from local directory of images (fallback)."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        log.error(f"Directory not found: {image_dir}")
        return None
    
    # Find image files
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
        image_paths.extend(image_dir.glob(f"**/{ext}"))
        image_paths.extend(image_dir.glob(f"**/{ext.upper()}"))
    
    image_paths = list(set(image_paths))[:n_samples]
    
    if not image_paths:
        log.error(f"No images found in {image_dir}")
        return None
    
    log.info(f"Found {len(image_paths)} images in {image_dir}")
    all_features = []
    
    for img_path in tqdm(image_paths, desc="Extracting from directory"):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = inception_model(tensor.to(device))
            
            all_features.append(features.cpu().numpy().reshape(1, -1))
        except Exception as e:
            log.warning(f"Failed to process {img_path}: {e}")
            continue
    
    if not all_features:
        return None
    
    return np.vstack(all_features).astype(np.float32)


def main(output_dir="_oupipeline_v5tputs", n_samples=5000, batch_size=32):
    """Main extraction pipeline."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Inception model
    inception_model = get_inception_model(device)
    transform = get_image_transform()
    
    # Try to load dataset
    dataset = load_defactify_dataset()
    all_features = None
    
    if dataset is not None:
        all_features = extract_features_from_dataset(
            dataset, inception_model, device, transform, n_samples, batch_size
        )
    
    # Fallback: try to load from local cache or directory
    if all_features is None:
        log.info("Attempting fallback: checking for local image directory...")
        # Try common locations
        fallback_dirs = [
            output_path.parent / "defactify_images",
            Path.home() / "datasets" / "defactify",
            Path.home() / "data" / "defactify",
        ]
        
        for fallback_dir in fallback_dirs:
            if fallback_dir.exists():
                log.info(f"Found directory: {fallback_dir}")
                all_features = extract_features_from_directory(
                    str(fallback_dir), inception_model, device, transform, n_samples
                )
                if all_features is not None:
                    break
    
    if all_features is None:
        log.error("No dataset found and no images extracted.")
        log.error("Please ensure Defactify dataset is available or place images in a local directory.")
        return False
    
    # Compute statistics
    log.info(f"✓ Extracted {all_features.shape[0]} images with {all_features.shape[1]} features")
    log.info("Computing reference statistics...")
    
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features.T)
    
    log.info(f"Reference mean shape: {mu.shape}")
    log.info(f"Reference cov shape: {sigma.shape}")
    
    # Save statistics
    stats_file = output_path / "reference_stats.npz"
    np.savez(
        stats_file,
        mu=mu,
        sigma=sigma,
        n_images=all_features.shape[0],
        feature_dim=all_features.shape[1]
    )
    log.info(f"✓ Saved reference stats to {stats_file}")
    log.info(f"  File size: {stats_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Also save summary
    summary_file = output_path / "reference_stats_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Reference Statistics Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Images used: {all_features.shape[0]}\n")
        f.write(f"Feature dimension: {all_features.shape[1]}\n")
        f.write(f"\nMean features:\n")
        f.write(f"  [{mu[0]:.6f}, {mu[1]:.6f}, ..., {mu[-1]:.6f}]\n")
        f.write(f"\nCovariance matrix shape: {sigma.shape}\n")
        f.write(f"Covariance diagonal (variance):\n")
        std_devs = np.sqrt(np.diag(sigma))
        f.write(f"  [{std_devs[0]:.6f}, {std_devs[1]:.6f}, ..., {std_devs[-1]:.6f}]\n")
        f.write(f"\nUsage:\n")
        f.write(f"  This reference can be used for:\n")
        f.write(f"  • FID (Fréchet Inception Distance) computation\n")
        f.write(f"  • KID (Kernel Inception Distance) computation\n")
        f.write(f"  • Per-image quality assessment against real distribution\n")
    
    log.info(f"✓ Saved summary to {summary_file}")
    
    log.info("\n" + "=" * 60)
    log.info("✓ SUCCESS: Reference statistics ready!")
    log.info("=" * 60)
    log.info(f"Location: {stats_file}")
    log.info(f"Size: {stats_file.stat().st_size / 1024 / 1024:.1f} MB")
    log.info(f"Images: {all_features.shape[0]}")
    log.info(f"Features: {all_features.shape[1]}")
    log.info("")
    log.info("The dashboard will now automatically use these stats for FID/KID evaluation.")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract Inception features from Defactify dataset for FID/KID/IS evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="_oupipeline_v5tputs",
        help="Output directory to save reference stats"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of images to sample (default: 5000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)"
    )
    
    args = parser.parse_args()
    
    success = main(args.output_dir, args.n_samples, args.batch_size)
    sys.exit(0 if success else 1)
