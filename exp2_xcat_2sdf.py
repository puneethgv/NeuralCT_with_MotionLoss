"""
exp2_xcat_2sdf.py - XCAT experiment with 2 SDFs (no background).

The GMM segmentation for XCAT data produces:
- Component 0: Background (μ≈0, useless as SDF)
- Component 1: Blood pool (bright, ~25%)
- Component 2: Soft tissue (mid-intensity, ~75%)

This experiment uses only 2 SDFs by excluding the background component,
which should give more stable training.

Key changes from original:
1. NUM_SDFS = 2 (instead of 3)
2. Modified GMM to exclude background component
3. Safe SDF extraction with clamping
4. Organ regularization to prevent degenerate states
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy import ndimage
from tqdm import tqdm
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle
from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter

from source.config import *
from source.data_generation.anatomy import *
from source.network.renderer import *
from source.network.model import *
from source.network.model import SDFNCT, Trainer, sdf_to_occ, occ_to_sdf
from source.util.utility import (create_sdf_comparison_movie, save_movie_as_gif, 
                                  save_sdf_state, create_final_comparison, 
                                  downsample_sinogram_bin_average, create_sinogram_comparison)

# Set random seeds
num = 1
torch.manual_seed(num)
random.seed(num)
np.random.seed(num)

#### Constants ####
RADIUS = 0.2
SCALE = 1.5
LR = 1e-4
IMAGE_RESOLUTION = 128
HEART_RATE = 120
NUM_SDFS = 2  # Only 2 SDFs: blood pool + soft tissue (no background)

# Motion mask parameters
MASK_CENTER = (840, 750)
MASK_RADIUS = 210
ORIGINAL_RESOLUTION = 1700


# =============================================================================
# MODIFIED GMM SEGMENTATION - EXCLUDE BACKGROUND
# =============================================================================

def fit_gmm_xcat_no_background(img: np.ndarray, num_components: int = 3,
                                max_samples: int = 50000) -> tuple:
    """
    Fit GMM to XCAT data and identify non-background components.
    
    Fits with 3 components but returns only the 2 non-background ones.
    
    Args:
        img: 3D array (H, W, T) - FBP movie
        num_components: Total GMM components (default 3)
        max_samples: Max samples for fitting
        
    Returns:
        gmm: Fitted GaussianMixture
        labels_to_use: Array of non-background component indices
        background_idx: Index of the background component
    """
    from scipy import stats
    
    # Crop to heart region for better fitting
    normalising_factor = 1700 / img.shape[0]
    center = (int(750 / normalising_factor), int(640 / normalising_factor))
    radius = int(260 / normalising_factor)
    
    from skimage import draw
    mask = np.zeros(img[..., 0].shape, dtype=bool)
    rr, cc = draw.disk(center, radius, shape=mask.shape)
    mask[rr, cc] = True
    
    # Sample from masked region
    masked_data = []
    for t in range(img.shape[2]):
        masked_data.extend(img[..., t][mask].flatten())
    
    X = np.array(masked_data).reshape(-1, 1)
    
    # Subsample if too large
    if X.shape[0] > max_samples:
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Fit GMM
    gmm = GaussianMixture(n_components=num_components, random_state=0).fit(X_sample)
    
    print(f"GMM means: {gmm.means_.flatten()}")
    print(f"GMM weights: {gmm.weights_}")
    
    # Identify background (lowest mean intensity)
    sorted_idx = np.argsort(gmm.means_.flatten())
    background_idx = sorted_idx[0]  # Lowest intensity = background
    
    # Use non-background components
    labels_to_use = sorted_idx[1:]  # Exclude background
    
    print(f"Background component: {background_idx} (μ={gmm.means_[background_idx, 0]:.3f})")
    print(f"Using components: {labels_to_use}")
    
    return gmm, labels_to_use, background_idx


def get_pretraining_sdfs_2sdf(config, fbp_movie: np.ndarray, 
                               save_visualizations: bool = True) -> tuple:
    """
    Extract 2 SDFs from FBP movie, excluding background.
    
    Args:
        config: Config object with NUM_SDFS=2
        fbp_movie: 3D array (H, W, T)
        save_visualizations: Whether to save debug images
        
    Returns:
        sdf: 4D array (H, W, T, 2)
        init: 1D array of intensities
    """
    from pathlib import Path
    
    cache_path = Path("cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("2-SDF SEGMENTATION (No Background)")
    print("=" * 60)
    print(f"Input shape: {fbp_movie.shape}")
    print(f"Target SDFs: {config.NUM_SDFS}")
    
    # Fit GMM with 3 components, use only 2
    print("\n[1/4] Fitting GMM...")
    gmm, labels_to_use, bg_idx = fit_gmm_xcat_no_background(fbp_movie)
    
    assert len(labels_to_use) == config.NUM_SDFS, \
        f"Expected {config.NUM_SDFS} components, got {len(labels_to_use)}"
    
    # Predict labels
    print("\n[2/4] Computing segmentation...")
    labels_movie = gmm.predict(fbp_movie.reshape(-1, 1)).reshape(fbp_movie.shape)
    
    # Apply median filter
    print("\n[3/4] Applying median filter...")
    labels_movie = median_filter(labels_movie, size=(5, 5, 1))
    
    # Downsample if needed
    if fbp_movie.shape[0] != config.IMAGE_RESOLUTION:
        from scipy.ndimage import zoom
        scale = config.IMAGE_RESOLUTION / fbp_movie.shape[0]
        labels_movie = zoom(labels_movie.astype(float), (scale, scale, 1), order=0).astype(int)
        fbp_movie_ds = zoom(fbp_movie, (scale, scale, 1), order=1)
    else:
        fbp_movie_ds = fbp_movie
    
    # Create SDFs for non-background components only
    print("\n[4/4] Computing SDFs...")
    h, w, t = labels_movie.shape
    num_sdfs = len(labels_to_use)
    
    movie_objects = np.zeros((h, w, t, num_sdfs))
    
    for i, comp_idx in enumerate(labels_to_use):
        for frame in range(t):
            movie_objects[..., frame, i] = (labels_movie[..., frame] == comp_idx).astype(float)
    
    # Convert to SDF
    sdf = np.zeros_like(movie_objects)
    max_dist = config.IMAGE_RESOLUTION / 4
    
    for i in tqdm(range(t), desc="Processing frames"):
        for j in range(num_sdfs):
            occupancy = np.round(denoise_tv_chambolle(
                movie_objects[..., i, j][..., np.newaxis]))
            sdf_val = occ_to_sdf_safe(occupancy, max_abs_distance=max_dist)
            sdf[..., i, j] = denoise_tv_chambolle(sdf_val, weight=2)[..., 0]
    
    # Compute intensities
    init = np.zeros((1, num_sdfs))
    for j, comp_idx in enumerate(labels_to_use):
        mask = sdf[..., 0, j] > 0
        if mask.any():
            init[0, j] = np.median(fbp_movie_ds[mask, 0])
        else:
            init[0, j] = gmm.means_[comp_idx, 0]
    
    # Sort by intensity
    sorted_indices = np.argsort(init[0])
    sdf = sdf[..., sorted_indices]
    init = init[:, sorted_indices]
    
    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print(f"SDF shape: {sdf.shape}")
    print(f"Intensities: {init.flatten()}")
    print("=" * 60)
    
    # Save visualization
    if save_visualizations:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        frame_idx = 0
        axes[0].imshow(fbp_movie_ds[..., frame_idx], cmap='gray')
        axes[0].set_title('FBP Frame 0')
        axes[0].axis('off')
        
        for j in range(min(2, num_sdfs)):
            axes[j+1].imshow(sdf[..., frame_idx, j], cmap='RdBu')
            axes[j+1].set_title(f'SDF {j} (int={init[0, j]:.2f})')
            axes[j+1].axis('off')
        
        # Combined occupancy
        combined = np.zeros((h, w))
        for j in range(num_sdfs):
            combined += (sdf[..., frame_idx, j] > 0).astype(float) * (j + 1)
        axes[3].imshow(combined, cmap='tab10')
        axes[3].set_title('Combined (no background)')
        axes[3].axis('off')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(cache_path / f'2sdf_segmentation_{timestamp}.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to cache/2sdf_segmentation_{timestamp}.png")
    
    return sdf, init


def occ_to_sdf_safe(x, max_abs_distance=None):
    """Safe occ_to_sdf with clamping."""
    assert isinstance(x, np.ndarray) and len(x.shape) == 3
    
    x = np.round(x).astype(np.float32)
    H, W = x.shape[:2]
    
    if max_abs_distance is None:
        max_abs_distance = np.sqrt(H**2 + W**2) / 4
    
    dist_img = np.zeros_like(x, dtype=np.float32)
    
    for i in range(x.shape[2]):
        organ_occ = x[..., i]
        num_inside = np.sum(organ_occ == 1)
        num_outside = np.sum(organ_occ == 0)
        
        if num_inside == 0:
            dist_img[..., i] = -max_abs_distance
            continue
        if num_outside == 0:
            dist_img[..., i] = max_abs_distance
            continue
        
        dist_inside = ndimage.distance_transform_bf(organ_occ == 1).astype(np.float64)
        dist_outside = ndimage.distance_transform_bf(organ_occ == 0).astype(np.float64)
        sdf = np.clip(dist_inside - dist_outside, -max_abs_distance, max_abs_distance)
        dist_img[..., i] = sdf.astype(np.float32)
    
    return dist_img


# =============================================================================
# CUSTOM SDFGtXCAT FOR 2 SDFS
# =============================================================================

class SDFGtXCAT2SDF(SDF):
    """Ground truth SDF for XCAT with 2 components (no background)."""
    
    def __init__(self, config, sinogram, angles):
        super(SDFGtXCAT2SDF, self).__init__()
        
        self.incr = 250
        
        assert config.NUM_SDFS == 2, f"This class requires NUM_SDFS=2, got {config.NUM_SDFS}"
        
        # Filter angles to valid range (same as SDFGtXCAT)
        angles = angles[angles < config.THETA_MAX + 180]
        assert np.abs(config.THETA_MAX + 180 - angles[-1]) < 1, \
            'THETA_MAX must be equal to the maximum angle in angles'
        
        self.config = config
        self.sinogram = sinogram[:, :len(angles)]
        self.angles = angles
        self._cache = {}
        
        # FBP reconstruction using the fetch_fbp_movie static method
        self.reconstruction_fbp = self.fetch_fbp_movie(
            config, self.sinogram, angles, intensity_scale=132, incr=self.incr)
        
        # Get 2 SDFs (no background)
        self.pretraining_sdfs, self.intensities = get_pretraining_sdfs_2sdf(
            config, self.reconstruction_fbp)
    
    @staticmethod
    def fetch_fbp_movie(config, sinogram, angles, intensity_scale=132, incr=250):
        """FBP reconstruction using sliding window."""
        from source.network.model import reconstruct_fbp
        
        window_size = incr * int(config.GANTRY_VIEWS_PER_ROTATION / 360)
        reconstruction = reconstruct_fbp(sinogram, angles, window_size)
        
        return intensity_scale * reconstruction
    
    def _compute_sdf(self, t):
        t = t % self.config.THETA_MAX
        idx = np.argmin(np.abs(self.angles - t))
        return self.pretraining_sdfs[..., idx, :]
    
    def forward(self, t):
        assert isinstance(t, float)
        t_key = round(t, 6)
        
        if t_key not in self._cache:
            self._cache[t_key] = self._compute_sdf(t)
        
        return torch.from_numpy(self._cache[t_key])


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def create_circular_mask_xcat(target_resolution, center, radius, original_resolution=1700):
    """Create circular motion mask."""
    scale = target_resolution / original_resolution
    scaled_center = (int(center[0] * scale), int(center[1] * scale))
    scaled_radius = int(radius * scale)
    
    Y, X = np.ogrid[:target_resolution, :target_resolution]
    dist = np.sqrt((X - scaled_center[0])**2 + (Y - scaled_center[1])**2)
    return (dist <= scaled_radius).astype(np.float32)


def setup_experiment(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = f'exp2_2sdf_results/{timestamp}/'
    os.makedirs(folder, exist_ok=True)

    for subdir in ['1_baseline', '2_pretrain', '3_train', '4_refine', '5_final']:
        os.makedirs(f'{folder}{subdir}', exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: XCAT with 2 SDFs (No Background)")
    print(f"{'='*60}")
    print(f"Output: {folder}")
    print(f"{'='*60}\n")
    return folder


def pretrain_cycle(trainer, pretraining_sdfs, all_thetas, pretrain_lr, filename_prefix, sdf_gt):
    print(f"\n[PRETRAIN] lr={pretrain_lr}")
    print("-" * 40)
    
    max_dist = trainer.config.IMAGE_RESOLUTION / 4
    pretraining_sdfs = np.nan_to_num(pretraining_sdfs, nan=0, 
                                      posinf=max_dist, neginf=-max_dist)
    
    sdf_network = trainer.pretrain_sdf(pretraining_sdfs, lr=pretrain_lr)
    
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_state',
                   all_thetas=all_thetas, subsample=10, fps=15)
    create_sdf_comparison_movie(sdf_gt, sdf_network, all_thetas, filename_prefix)
    print("✓ Pretrain complete\n")
    return sdf_network


def train_cycle(trainer, sdf_gt, gt_sinogram, all_thetas, init,
                gantry_offset, tvs, tvt, train_lr, filename_prefix,
                max_iter, motion_mask=None, coeff_motion=0.0, motion_gap=10):
    print(f"\n[TRAIN] lr={train_lr}, max_iter={max_iter}")
    print("-" * 40)
    
    sdf_network, intensities = trainer.train(
        gt_sinogram, init[:, 0], lr=train_lr,
        gantry_offset=gantry_offset, coefftvs=tvs, coefftvt=tvt,
        motion_mask=motion_mask, coeff_motion=coeff_motion,
        max_iter=max_iter, motion_gap=motion_gap
    )
    
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_state',
                   all_thetas=all_thetas, subsample=10, fps=15)
    create_sdf_comparison_movie(sdf_gt, sdf_network, all_thetas, filename_prefix)
    create_sinogram_comparison(gt_sinogram, sdf_network, intensities, 
                               all_thetas, gantry_offset, 
                               filename_prefix + '_sinogram')
    print("✓ Train complete\n")
    return sdf_network, intensities


def extract_pretraining_sdfs_safe(config, sdf_network):
    """Safe SDF extraction with validation."""
    max_abs_distance = config.IMAGE_RESOLUTION / 4
    
    pretraining_sdfs = np.zeros((config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION,
                                 config.TOTAL_CLICKS, config.NUM_SDFS), dtype=np.float32)
    
    all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)
    
    print("  Extracting SDFs (safe mode)...")
    
    for j in tqdm(range(config.TOTAL_CLICKS), desc="Extracting"):
        with torch.no_grad():
            canvas = sdf_network(all_thetas[j])
        
        for i in range(config.NUM_SDFS):
            occ = sdf_to_occ(canvas)[..., i].detach().cpu().numpy()
            denoised = np.round(denoise_tv_chambolle(occ, weight=2))
            sdf_val = occ_to_sdf_safe(denoised[..., np.newaxis], 
                                       max_abs_distance=max_abs_distance)[..., 0]
            pretraining_sdfs[..., j, i] = sdf_val
    
    pretraining_sdfs = np.nan_to_num(pretraining_sdfs, nan=0, 
                                      posinf=max_abs_distance, 
                                      neginf=-max_abs_distance)
    
    print(f"  SDF range: [{pretraining_sdfs.min():.2f}, {pretraining_sdfs.max():.2f}]")
    return pretraining_sdfs


def refine_cycle(trainer, sdf_gt, gt_sinogram, all_thetas, init,
                 gantry_offset, tvs, tvt, pretrain_lr, train_lr, filename_prefix,
                 max_iter, motion_mask=None, coeff_motion=0.0, motion_gap=10):
    print(f"\n[REFINE] pretrain_lr={pretrain_lr}, train_lr={train_lr}")
    print("-" * 40)

    pretraining_sdfs = extract_pretraining_sdfs_safe(trainer.config, trainer.sdf_network)
    
    sdf_network = trainer.pretrain_sdf(pretraining_sdfs, lr=pretrain_lr)
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_pretrain_state',
                   all_thetas=all_thetas, subsample=10, fps=15)

    sdf_network, intensities = trainer.train(
        gt_sinogram, init[:, 0], lr=train_lr,
        gantry_offset=gantry_offset, coefftvs=tvs, coefftvt=tvt,
        motion_mask=motion_mask, coeff_motion=coeff_motion,
        max_iter=max_iter, motion_gap=motion_gap
    )

    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_train_state',
                   all_thetas=all_thetas, subsample=3, fps=15)
    create_sdf_comparison_movie(sdf_gt, sdf_network, all_thetas,
                                filename_prefix + '_train')
    create_sinogram_comparison(gt_sinogram, sdf_network, intensities, 
                               all_thetas, gantry_offset, 
                               filename_prefix + '_sinogram')
    print("✓ Refine complete\n")
    return sdf_network, intensities


def save_final_results(config, sdf_gt, sdf_network, all_thetas, filename, fbp_movie):
    print(f"\n[SAVING RESULTS]")
    print("-" * 40)

    movie_nct = fetch_movie(config, sdf_network, None)
    np.save(filename + '_nct', movie_nct)
    save_movie_as_gif(movie_nct, filename + '_nct.gif', 'NCT', fps=15, subsample=5)

    movie_gt = fetch_movie(config, sdf_gt, all_thetas)
    np.save(filename + '_gt', movie_gt)
    save_movie_as_gif(movie_gt, filename + '_gt.gif', 'GT', fps=15, subsample=5)

    torch.save(sdf_network.state_dict(), filename + '_weights.pth')
    
    create_final_comparison(fbp_movie, movie_nct, movie_gt,
                            filename + '_comparison.gif', fps=15, subsample=5)
    print("✓ Results saved\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0.0, type=float)
    parser.add_argument("--tvs", default=1.0, type=float)
    parser.add_argument("--tvt", default=1.0, type=float)
    parser.add_argument("--motion_coeff", default=0.0, type=float)
    parser.add_argument("--mask_center_x", default=MASK_CENTER[0], type=int)
    parser.add_argument("--mask_center_y", default=MASK_CENTER[1], type=int)
    parser.add_argument("--mask_radius", default=MASK_RADIUS, type=int)
    parser.add_argument("--max_iter", default=5000, type=int)
    parser.add_argument("--gap", default=10, type=int)
    opt = parser.parse_args()

    folder = setup_experiment(opt)

    # Load XCAT data
    print("[1/8] Loading XCAT data...")
    sinogram_file_path = '/media/ExtraDrive1/kvibhandik/XCAT/data/store/94a44f1d91b0f1b3/sinogram/sinogram_main.npz'
    result = np.load(sinogram_file_path)
    sinogram = result['sinogram']
    angles = result['angles']

    sinogram = downsample_sinogram_bin_average(sinogram.T, target_detectors=IMAGE_RESOLUTION)

    # Config with 2 SDFs
    config = ConfigXCAT.from_xcat(
        views_per_rotation=1000,
        xcat_detectors=IMAGE_RESOLUTION,
        HEART_BEAT_PERIOD=1000 * 60 / HEART_RATE,
        NUM_SDFS=NUM_SDFS  # 2 SDFs only
    )
    all_thetas = np.linspace(0, config.THETA_MAX, config.TOTAL_CLICKS, endpoint=False)

    print("\n[2/8] Creating GT SDF (2 components)...")
    sdf_gt = SDFGtXCAT2SDF(config, sinogram, angles)
    
    # Set intensities directly (bypass set_intensities which expects 3 values)
    config.INTENSITIES = sdf_gt.intensities.flatten()
    config.NUM_SDFS = len(config.INTENSITIES)
    
    print(f"  Intensities: {config.INTENSITIES}")

    print("\n[3/8] Saving FBP...")
    reconstruction_fbp = sdf_gt.reconstruction_fbp
    np.save(f'{folder}1_baseline/fbp', reconstruction_fbp)
    save_movie_as_gif(reconstruction_fbp, f'{folder}1_baseline/fbp.gif', 'FBP', fps=15, subsample=5)

    # Motion mask
    motion_mask = None
    if opt.motion_coeff > 0.0:
        motion_mask = create_circular_mask_xcat(
            IMAGE_RESOLUTION, (opt.mask_center_x, opt.mask_center_y),
            opt.mask_radius, ORIGINAL_RESOLUTION
        )

    print("\n[4/8] Pretraining SDFs...")
    pretraining_sdfs = sdf_gt.pretraining_sdfs
    init = sdf_gt.intensities
    
    # Check organ sizes
    for i in range(NUM_SDFS):
        occ = (pretraining_sdfs[..., 0, i] > 0).astype(float)
        print(f"  Organ {i}: {occ.sum()/occ.size*100:.1f}% of image")

    print("\n[5/8] Creating network...")
    sdf_network = SDFNCT(config, scale=SCALE)
    trainer = Trainer(config, sdf_network)
    print(f"  Parameters: {sum(p.numel() for p in sdf_network.parameters()):,}")

    save_sdf_state(sdf_network, config, f'{folder}1_baseline/untrained', 
                   all_thetas=all_thetas, subsample=3, fps=15)

    # Pretrain
    print("\n[6/8] PRETRAIN")
    sdf_network = pretrain_cycle(trainer, pretraining_sdfs, all_thetas,
                                 LR, f'{folder}2_pretrain/pretrain', sdf_gt)

    # GT sinogram
    gt_sinogram = torch.from_numpy(get_sinogram(config, sdf_gt,
                                                Intensities(config, learnable=False),
                                                all_thetas, offset=opt.offset)).cuda()

    # Train
    print("\n[7/8] TRAIN")
    sdf_network, intensities = train_cycle(
        trainer, sdf_gt, gt_sinogram, all_thetas, init, opt.offset,
        opt.tvs, opt.tvt, LR, f'{folder}3_train/train', opt.max_iter,
        motion_mask, opt.motion_coeff, opt.gap
    )

    # Refine
    print("\n[8/8] REFINE")
    sdf_network, intensities = refine_cycle(
        trainer, sdf_gt, gt_sinogram, all_thetas, init, opt.offset,
        opt.tvs, opt.tvt, 5e-5, 1e-5, f'{folder}4_refine/refine', opt.max_iter,
        motion_mask, opt.motion_coeff, opt.gap
    )

    # Save final
    save_final_results(config, sdf_gt, sdf_network, all_thetas,
                       f'{folder}5_final/result', reconstruction_fbp)

    print("=" * 60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results: {folder}")
    print("=" * 60)


if __name__ == '__main__':
    main()
