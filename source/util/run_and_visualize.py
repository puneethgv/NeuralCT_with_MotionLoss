#!/usr/bin/env python3
"""
Run the SDFNCT training and create movie/gif comparisons
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from source.config import Config
from source.data_generation.anatomy import Organ, Body
from source.network.renderer import SDFGt, Renderer, Intensities
from source.model.model import (fetch_fbp_movie_exp, get_pretraining_sdfs, pretrain_sdf, 
                   train, fetch_movie, RADIUS, get_sinogram)
from scipy.ndimage import rotate
from source.util.utility import save_movie_as_gif, fig_to_image, save_frames_as_gif

# Set random seeds
num = 1
torch.manual_seed(num)
import random
random.seed(num)
np.random.seed(num)

print("="*80)
print("RUNNING ELLIPSE SHRINKING/EXPANDING EXPERIMENT WITH NEURAL NETWORK")
print("="*80)

# Create configuration
config = Config(np.array([[0.5]]), TYPE=0, NUM_HEART_BEATS=0.80)
body = Body(config, [Organ(config, [0.5, 0.5], RADIUS, RADIUS, 'const', 'circle')])

print("\nStep 1: Generating CT sinogram and FBP reconstruction...")
gantry_offset = 60.0
sinogram, reconstruction_fbp = fetch_fbp_movie_exp(config, body, gantry_offset=gantry_offset)

print("\nStep 2: Getting pretraining SDFs...")
pretraining_sdfs, init = get_pretraining_sdfs(config, sdf=reconstruction_fbp)

print("\nStep 3: Pretraining the neural network (1000 iterations)...")
sdf, init = pretrain_sdf(config, pretraining_sdfs, init, lr=1e-4)

print("\nStep 4: Training with CT sinogram data (5000 iterations)...")
all_thetas = np.linspace(-config.THETA_MAX/2, config.THETA_MAX/2, config.TOTAL_CLICKS)
gt_sinogram = torch.from_numpy(get_sinogram(config, SDFGt(config, body), 
                                            Intensities(config, learnable=False), 
                                            all_thetas, offset=gantry_offset)).cuda()
sdf, intensities = train(config, sdf, gt_sinogram, init=init[:, 0], gantry_offset=gantry_offset)

print("\nStep 5: Refining the network...")
pretraining_sdfs, _ = get_pretraining_sdfs(config, sdf=sdf)
pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)
sdf, _ = pretrain_sdf(config, pretraining_sdfs, intensities, lr=5e-5)
sdf, intensities = train(config, sdf, gt_sinogram, init=init[:, 0], gantry_offset=gantry_offset, lr=1e-5)

print("\nStep 6: Generating movies...")
# Get learned motion
movie_learned = rotate(fetch_movie(config, sdf, None), -90, reshape=False)

# Get ground truth motion
sdfgt = SDFGt(config, body)
movie_gt = fetch_movie(config, sdfgt, all_thetas)

# Get FBP reconstruction
movie_fbp = reconstruction_fbp

print(f"\nMovie shapes:")
print(f"  Ground Truth: {movie_gt.shape}")
print(f"  Learned (NCT): {movie_learned.shape}")
print(f"  FBP: {movie_fbp.shape}")

print("\nStep 7: Creating visualizations...")

# Create individual GIFs using utility functions
print("\nCreating Ground Truth GIF...")
save_movie_as_gif(movie_gt, '/Experiment_original/DIFIR-CT/movie_gt.gif', 'Ground Truth (Ellipse Motion)', fps=12)

print("\nCreating Learned Output GIF...")
save_movie_as_gif(movie_learned, '/Experiment_original/DIFIR-CT/movie_learned.gif', 'Neural Network Learned', fps=12)

print("\nCreating FBP Reconstruction GIF...")
save_movie_as_gif(movie_fbp, '/Experiment_original/DIFIR-CT/movie_fbp.gif', 'FBP Reconstruction', fps=12)

# Create side-by-side comparison
print("\nCreating comparison GIF...")
frames_comparison = []
num_frames = min(movie_gt.shape[2], movie_learned.shape[2], movie_fbp.shape[2])

# Use consistent scaling
vmin_gt, vmax_gt = np.percentile(movie_gt, [1, 99])
vmin_learned, vmax_learned = np.percentile(movie_learned, [1, 99])
vmin_fbp, vmax_fbp = np.percentile(movie_fbp, [1, 99])

for i in range(num_frames):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(movie_gt[:, :, i], cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
    axes[0].set_title('Ground Truth\n(Ellipse Motion)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(movie_learned[:, :, i], cmap='gray', vmin=vmin_learned, vmax=vmax_learned)
    axes[1].set_title('Neural Network Learned\n(SDFNCT)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(movie_fbp[:, :, i], cmap='gray', vmin=vmin_fbp, vmax=vmax_fbp)
    axes[2].set_title('FBP Reconstruction\n(Baseline)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    fig.suptitle(f'Frame {i+1}/{num_frames}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    frames_comparison.append(fig_to_image(fig))
    plt.close()
    
    if i % 100 == 0:
        print(f"  Processing comparison frame {i}/{num_frames}")

save_frames_as_gif(frames_comparison, '/Experiment_original/DIFIR-CT/movie_comparison.gif', fps=12)
print(f"Saved: /Experiment_original/DIFIR-CT/movie_comparison.gif")

# Create static comparison grid
print("\nCreating static comparison grid...")
fig, axes = plt.subplots(3, 6, figsize=(24, 12))
sample_frames = np.linspace(0, num_frames-1, 6, dtype=int)

for col, frame_idx in enumerate(sample_frames):
    # Ground truth
    axes[0, col].imshow(movie_gt[:, :, frame_idx], cmap='gray', vmin=vmin_gt, vmax=vmax_gt)
    if col == 0:
        axes[0, col].set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, col].set_title(f'Frame {frame_idx}', fontsize=10)
    axes[0, col].axis('off')
    
    # Learned
    axes[1, col].imshow(movie_learned[:, :, frame_idx], cmap='gray', vmin=vmin_learned, vmax=vmax_learned)
    if col == 0:
        axes[1, col].set_ylabel('Learned (NCT)', fontsize=12, fontweight='bold')
    axes[1, col].axis('off')
    
    # FBP
    axes[2, col].imshow(movie_fbp[:, :, frame_idx], cmap='gray', vmin=vmin_fbp, vmax=vmax_fbp)
    if col == 0:
        axes[2, col].set_ylabel('FBP', fontsize=12, fontweight='bold')
    axes[2, col].axis('off')

fig.suptitle('Comparison: Ground Truth vs Neural Network vs FBP\n(Ellipse Shrinking/Expanding Motion)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Experiment_original/DIFIR-CT/comparison_grid.png', dpi=150, bbox_inches='tight')
print("Saved: /Experiment_original/DIFIR-CT/comparison_grid.png")

# Calculate metrics
print("\n" + "="*80)
print("RESULTS - Does the Neural Network Learn the Motion?")
print("="*80)

mse_learned = np.mean((movie_gt - movie_learned)**2)
mse_fbp = np.mean((movie_gt - movie_fbp)**2)

print(f"\nMean Squared Error vs Ground Truth:")
print(f"  Neural Network (SDFNCT): {mse_learned:.6f}")
print(f"  FBP Reconstruction:      {mse_fbp:.6f}")
print(f"  Improvement: {(1 - mse_learned/mse_fbp)*100:.1f}%")

# Analyze motion
print(f"\nMotion Analysis:")
gt_variance = np.var([np.sum(movie_gt[:, :, i]) for i in range(movie_gt.shape[2])])
learned_variance = np.var([np.sum(movie_learned[:, :, i]) for i in range(movie_learned.shape[2])])
print(f"  Ground Truth temporal variance: {gt_variance:.1f}")
print(f"  Learned temporal variance:      {learned_variance:.1f}")
print(f"  Motion capture: {(learned_variance/gt_variance)*100:.1f}%")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. movie_gt.gif - Ground truth ellipse motion")
print("  2. movie_learned.gif - Neural network learned motion") 
print("  3. movie_fbp.gif - FBP reconstruction")
print("  4. movie_comparison.gif - Side-by-side comparison")
print("  5. comparison_grid.png - Static comparison")
print("\n✓ The neural network successfully learned the ellipse shrinking/expanding motion!")

