import os
import random
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm
from datetime import datetime

from source.config import *
from source.data_generation.anatomy import *
from source.network.renderer import *
from source.network.model import *
from source.network.model import SDFNCT, Trainer
from source.util.utility import create_sdf_comparison_movie, save_movie_as_gif, save_sdf_state, create_final_comparison, downsample_sinogram_bin_average, create_sinogram_comparison
from scipy.fft import fft, fftfreq

# Set random seeds
num = 1
torch.manual_seed(num)
random.seed(num)
np.random.seed(num)

#### Organise these constants ####
RADIUS = 0.2  # Radius of organ
SCALE = 1.5   # Scale of SDFNCT
LR = 1e-4     # Learning rate
##################################
IMAGE_RESOLUTION = 128  # Image resolution
HEART_RATE = 120     # Heart rate in bpm

# Motion mask parameters (at original 1700x1700 resolution)
MASK_CENTER = (840, 750)  # (x, y) center of circular ROI
MASK_RADIUS = 210  # radius of circular ROI
ORIGINAL_RESOLUTION = 1700  # Original FBP resolution


def create_circular_mask_xcat(target_resolution, center, radius, original_resolution=1700):
    """Create a circular motion mask for XCAT data.
    
    Args:
        target_resolution: Target image resolution (e.g., 128)
        center: (x, y) center of circle at original resolution
        radius: Radius of circle at original resolution
        original_resolution: Original image resolution (default 1700)
        
    Returns:
        mask: 2D array (target_resolution, target_resolution), 1=motion allowed, 0=forbidden
    """
    # Scale coordinates to target resolution
    scale = target_resolution / original_resolution
    scaled_center = (int(center[0] * scale), int(center[1] * scale))
    scaled_radius = int(radius * scale)
    
    # Create mask at target resolution
    Y, X = np.ogrid[:target_resolution, :target_resolution]
    dist_from_center = np.sqrt((X - scaled_center[0])**2 + (Y - scaled_center[1])**2)
    mask = (dist_from_center <= scaled_radius).astype(np.float32)
    
    print(f"Created circular motion mask:")
    print(f"  Original: center={center}, radius={radius} at {original_resolution}x{original_resolution}")
    print(f"  Scaled: center={scaled_center}, radius={scaled_radius} at {target_resolution}x{target_resolution}")
    print(f"  Mask coverage: {mask.sum() / mask.size * 100:.1f}% of image")
    
    return mask


def setup_experiment(args):
    """Setup experiment directory and return filename prefix."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = f'exp2_results/{timestamp}_tvs{int(args.tvs*1000)}_tvt{int(args.tvt*1000)}/'
    os.makedirs(folder, exist_ok=True)

    # Create organized subdirectories
    for subdir in ['1_baseline', '2_pretrain', '3_train', '4_refine', '5_final']:
        os.makedirs(f'{folder}{subdir}', exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Dynamic CT Reconstruction")
    print(f"{'='*60}")
    print(f"Output folder: {folder}")
    print(
        f"Parameters: tvs={args.tvs}, tvt={args.tvt}, offset={args.offset}, rate={args.rate}")
    print(f"{'='*60}\n")
    return folder


def pretrain_cycle(trainer: Trainer, sdf_gt, pretraining_sdfs, all_thetas, pretrain_lr, filename_prefix):
    """Pretrain and visualize."""
    print(f"\n[PRETRAIN CYCLE] lr={pretrain_lr}")
    print("-" * 40)
    sdf_network = trainer.pretrain_sdf(pretraining_sdfs, lr=pretrain_lr)
    print("Saving pretrain state visualization...")
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_state',
                   all_thetas=all_thetas, subsample=10, fps=15)
    print("Saving pretrain comparison movie...")
    create_sdf_comparison_movie(
        sdf_gt, sdf_network, all_thetas, filename_prefix)
    print("✓ Pretrain cycle complete\n")
    return sdf_network


def train_cycle(trainer: Trainer, sdf_gt, gt_sinogram, all_thetas, init,
                gantry_offset, tvs, tvt, train_lr, filename_prefix,
                motion_mask=None, coeff_motion=1.0, max_iter=5000, motion_gap=10):
    """Train and visualize."""
    print(f"\n[TRAIN CYCLE] lr={train_lr}, tvs={tvs}, tvt={tvt}, max_iter={max_iter}, gap={motion_gap}")
    if motion_mask is not None:
        print(f"  Motion regularization: coeff={coeff_motion}")
    print("-" * 40)
    sdf_network, intensities = trainer.train(gt_sinogram, init[:, 0],
                                             lr=train_lr,
                                             gantry_offset=gantry_offset,
                                             coefftvs=tvs, coefftvt=tvt,
                                             motion_mask=motion_mask,
                                             coeff_motion=coeff_motion,
                                             max_iter=max_iter,
                                             motion_gap=motion_gap)
    print("Saving train state visualization...")
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_state',
                   all_thetas=all_thetas, subsample=10, fps=15)
    print("Saving train comparison movie...")
    create_sdf_comparison_movie(
        sdf_gt, sdf_network, all_thetas, filename_prefix)
    print("Saving sinogram comparison...")
    create_sinogram_comparison(
        gt_sinogram, sdf_network, intensities, all_thetas, gantry_offset, filename_prefix + '_sinogram_comparison')

    print("✓ Train cycle complete\n")
    return sdf_network, intensities


def refine_cycle(trainer: Trainer, sdf_gt, gt_sinogram, all_thetas, init,
                 gantry_offset, tvs, tvt, pretrain_lr, train_lr, filename_prefix,
                 motion_mask=None, coeff_motion=1.0, max_iter=5000, motion_gap=10):
    """Pretrain (refined) -> train (refined) -> visualize cycle."""
    print(f"\n[REFINE CYCLE] pretrain_lr={pretrain_lr}, train_lr={train_lr}, max_iter={max_iter}, gap={motion_gap}")
    if motion_mask is not None:
        print(f"  Motion regularization: coeff={coeff_motion}")
    print("-" * 40)

    # Pretrain with rotation
    print("Extracting SDFs from trained network...")
    pretraining_sdfs, _ = get_pretraining_sdfs(
        trainer.config, source=trainer.sdf_network)
    # pretraining_sdfs = rotate(pretraining_sdfs, -90, reshape=False)

    print("Refined pretraining...")
    sdf_network = trainer.pretrain_sdf(pretraining_sdfs, lr=pretrain_lr)
    print("Saving refined pretrain state...")
    # Use temporary intensities for comparison before training
    # init is 2D (1, num_sdfs), need to flatten to 1D
    temp_intensities = Intensities(trainer.config, learnable=False, init=init.flatten())
    create_sinogram_comparison(
        gt_sinogram, sdf_network, temp_intensities, all_thetas, gantry_offset, filename_prefix + '_pretrain_refine_sinogram_comparison')

    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_pretrain_refine_state',
                   all_thetas=all_thetas, subsample=10, fps=15)
    print("Saving refined pretrain comparison...")
    create_sdf_comparison_movie(sdf_gt, sdf_network, all_thetas,
                                filename_prefix + '_pretrain_refine')

    # Train with motion regularization
    print("\nRefined training...")
    sdf_network, intensities = trainer.train(gt_sinogram, init[:, 0],
                                             lr=train_lr,
                                             gantry_offset=gantry_offset,
                                             coefftvs=tvs, coefftvt=tvt,
                                             motion_mask=motion_mask,
                                             coeff_motion=coeff_motion,
                                             max_iter=max_iter,
                                             motion_gap=motion_gap)

    print("Saving sinogram comparison...")
    create_sinogram_comparison(
        gt_sinogram, sdf_network, intensities, all_thetas, gantry_offset, filename_prefix + '_train_refine_sinogram_comparison')

    print("Saving refined train state...")
    save_sdf_state(sdf_network, trainer.config, filename_prefix + '_train_refine_state',
                   all_thetas=all_thetas, subsample=3, fps=15)
    print("Saving refined train comparison...")
    create_sdf_comparison_movie(sdf_gt, sdf_network, all_thetas,
                                filename_prefix + '_train_refine')
    print("✓ Refine cycle complete\n")
    return sdf_network, intensities


def analyze_frequency_spectrum(config, sdf_gt, sdf_network, all_thetas, filename):
    """Analyze and visualize frequency spectrum of GT SDF vs learned SDF.

    Performs Fourier analysis on:
    1. Ground truth SDF temporal dynamics
    2. Learned network SDF temporal dynamics
    3. Compares frequency components
    """
    print(f"\n[FREQUENCY SPECTRUM ANALYSIS]")
    print("-" * 40)

    # Sample points across the image space
    grid_size = 64
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    points = np.array([[p[0], p[1]] for p in points])

    # Extract temporal signals for GT and learned SDFs
    print("Extracting temporal SDF signals (this may take a moment)...")
    num_angles = len(all_thetas)

    # GT temporal signal - use the body.is_inside method
    gt_signal = np.zeros((points.shape[0], num_angles))
    for i, theta in enumerate(tqdm(all_thetas, desc="GT SDF")):
        # Get occupancy from body (this is what GT uses internally)
        inside = sdf_gt.body.is_inside(points, theta)
        gt_signal[:, i] = np.sum(inside, axis=1)  # Sum across organs

    # Learned temporal signal
    learned_signal = np.zeros((points.shape[0], num_angles))
    with torch.no_grad():
        for i, theta in enumerate(tqdm(all_thetas, desc="Learned SDF")):
            # Create time coordinate for the network
            t_coord = np.full((points.shape[0], 1), theta / config.THETA_MAX)
            coords = np.hstack([points, t_coord])
            coords_tensor = torch.from_numpy(coords).float().cuda()

            sdf_vals = sdf_network(coords_tensor).detach().cpu().numpy()
            learned_signal[:, i] = sdf_vals.reshape(-1)

    # Compute FFT for both signals (averaging across all points)
    print("\nComputing FFT spectra...")
    gt_fft = np.zeros(num_angles)
    learned_fft = np.zeros(num_angles)

    for i in range(points.shape[0]):
        gt_spectrum = np.abs(fft(gt_signal[i, :]))
        learned_spectrum = np.abs(fft(learned_signal[i, :]))
        gt_fft += gt_spectrum
        learned_fft += learned_spectrum

    # Normalize
    gt_fft /= points.shape[0]
    learned_fft /= points.shape[0]

    # Frequency bins
    freqs = fftfreq(num_angles, 1.0)
    positive_freqs_idx = freqs >= 0
    freqs = freqs[positive_freqs_idx]
    gt_fft = gt_fft[positive_freqs_idx]
    learned_fft = learned_fft[positive_freqs_idx]

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Frequency Spectrum Analysis: GT vs Learned SDF',
                 fontsize=16, fontweight='bold')

    # Plot 1: Full spectrum comparison
    ax = axes[0, 0]
    ax.semilogy(freqs, gt_fft, 'b-', label='GT SDF', linewidth=2, alpha=0.7)
    ax.semilogy(freqs, learned_fft, 'r-',
                label='Learned SDF', linewidth=2, alpha=0.7)
    ax.set_xlabel('Frequency (cycles per gantry rotation)')
    ax.set_ylabel('Magnitude (log scale)')
    ax.set_title('Full Frequency Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoomed spectrum (first 20 frequencies)
    ax = axes[0, 1]
    zoom_idx = min(20, len(freqs))
    ax.bar(np.arange(zoom_idx) - 0.2,
           gt_fft[:zoom_idx], width=0.4, label='GT SDF', alpha=0.7)
    ax.bar(np.arange(zoom_idx) + 0.2,
           learned_fft[:zoom_idx], width=0.4, label='Learned SDF', alpha=0.7)
    ax.set_xlabel('Frequency bin')
    ax.set_ylabel('Magnitude')
    ax.set_title('Low Frequency Components (0-20)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Difference spectrum
    ax = axes[1, 0]
    diff_fft = np.abs(gt_fft - learned_fft)
    ax.semilogy(freqs, diff_fft, 'g-', linewidth=2)
    ax.set_xlabel('Frequency (cycles per gantry rotation)')
    ax.set_ylabel('Magnitude Difference (log scale)')
    ax.set_title('Spectral Difference |GT - Learned|')
    ax.grid(True, alpha=0.3)

    # Plot 4: Energy distribution
    ax = axes[1, 1]
    gt_energy = np.cumsum(gt_fft**2)
    learned_energy = np.cumsum(learned_fft**2)
    gt_energy /= gt_energy[-1]
    learned_energy /= learned_energy[-1]

    ax.plot(freqs, gt_energy, 'b-', label='GT SDF', linewidth=2)
    ax.plot(freqs, learned_energy, 'r-', label='Learned SDF', linewidth=2)
    ax.axhline(y=0.95, color='k', linestyle='--',
               alpha=0.5, label='95% energy')
    ax.set_xlabel('Frequency (cycles per gantry rotation)')
    ax.set_ylabel('Cumulative Energy Fraction')
    ax.set_title('Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([freqs[0], freqs[min(100, len(freqs)-1)]])

    plt.tight_layout()
    plt.savefig(filename + '_spectrum_analysis.png',
                dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}_spectrum_analysis.png")

    # Save statistics
    gt_peak_freq = freqs[np.argmax(gt_fft[1:])] if len(gt_fft) > 1 else 0
    learned_peak_freq = freqs[np.argmax(learned_fft[1:])] if len(
        learned_fft) > 1 else 0

    stats = {
        'gt_peak_frequency': float(gt_peak_freq),
        'learned_peak_frequency': float(learned_peak_freq),
        'gt_energy_95': float(freqs[np.argmax(gt_energy >= 0.95)]) if np.any(gt_energy >= 0.95) else freqs[-1],
        'learned_energy_95': float(freqs[np.argmax(learned_energy >= 0.95)]) if np.any(learned_energy >= 0.95) else freqs[-1],
        'mean_spectral_error': float(np.mean(diff_fft)),
        'max_spectral_error': float(np.max(diff_fft))
    }

    with open(filename + '_spectrum_stats.txt', 'w') as f:
        f.write("FREQUENCY SPECTRUM ANALYSIS STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(
            f"GT Peak Frequency: {stats['gt_peak_frequency']:.4f} cycles/rotation\n")
        f.write(
            f"Learned Peak Frequency: {stats['learned_peak_frequency']:.4f} cycles/rotation\n")
        f.write(
            f"GT 95% Energy Cutoff: {stats['gt_energy_95']:.4f} cycles/rotation\n")
        f.write(
            f"Learned 95% Energy Cutoff: {stats['learned_energy_95']:.4f} cycles/rotation\n")
        f.write(f"Mean Spectral Error: {stats['mean_spectral_error']:.6f}\n")
        f.write(f"Max Spectral Error: {stats['max_spectral_error']:.6f}\n")

    print(f"  Saved: {filename}_spectrum_stats.txt")
    print(f"\n  GT Peak Frequency: {stats['gt_peak_frequency']:.4f}")
    print(f"  Learned Peak Frequency: {stats['learned_peak_frequency']:.4f}")
    print(f"  Mean Spectral Error: {stats['mean_spectral_error']:.6f}")
    print("✓ Frequency analysis complete\n")

    return stats


def save_final_results(config: Config, sdf_gt, sdf_network, all_thetas, filename, fbp_movie):
    """Save final reconstruction and ground truth movies as both numpy arrays and GIFs."""
    print(f"\n[SAVING FINAL RESULTS]")
    print("-" * 40)

    # Save neural network reconstruction
    print("Generating neural network reconstruction movie...")
    movie_nct = fetch_movie(config, sdf_network, None)
    np.save(filename + '_nct', movie_nct)
    print(f"  Saved: {filename}_nct.npy")
    save_movie_as_gif(movie_nct, filename + '_nct.gif',
                      'Neural Network Reconstruction', fps=15, subsample=5)

    # Save ground truth
    print("Generating ground truth movie...")
    movie_gt = fetch_movie(config, sdf_gt, all_thetas)
    np.save(filename + '_gt', movie_gt)
    print(f"  Saved: {filename}_gt.npy")
    save_movie_as_gif(movie_gt, filename + '_gt.gif',
                      'Ground Truth', fps=15, subsample=5)

    # Save trained model weights
    print("Saving trained model weights...")
    torch.save(sdf_network.state_dict(), filename + '_weights.pth')
    print(f"  Saved: {filename}_weights.pth")

    # Frequency spectrum analysis - skip for XCAT data (no body attribute)
    # TODO: Implement XCAT-compatible frequency analysis if needed
    print("\nSkipping frequency spectrum analysis (not compatible with XCAT data)...")

    # Final comparison: GT vs NCT vs FBP
    print("Creating final 3-way comparison (GT vs NCT vs FBP)...")
    create_final_comparison(fbp_movie, movie_nct, movie_gt,
                            filename + '_comparison.gif', fps=15, subsample=5)
    print("✓ All results saved\n")
    print("\n FILE ORGANIZATION:")
    print("  1_baseline/    - FBP reconstruction, untrained network")
    print("  2_pretrain/    - Initial pretraining phase")
    print("  3_train/       - Main training phase")
    print("  4_refine/      - Refinement cycles")
    print("  5_final/       - Final results and comparisons")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", default=0.0,
                        type=float, help="Gantry offset")
    parser.add_argument("--rate", default=1.0, type=float, help="Heartrate")
    parser.add_argument("--tvs", default=1.0, type=float,
                        help="Coeff. of spatial TV")
    parser.add_argument("--tvt", default=1.0, type=float,
                        help="Coeff. of temporal TV")
    parser.add_argument("--motion_coeff", default=0.0, type=float,
                        help="Coeff. of motion regularization (0.0 = disabled)")
    parser.add_argument("--mask_center_x", default=MASK_CENTER[0], type=int,
                        help="Mask center X at original resolution")
    parser.add_argument("--mask_center_y", default=MASK_CENTER[1], type=int,
                        help="Mask center Y at original resolution")
    parser.add_argument("--mask_radius", default=MASK_RADIUS, type=int,
                        help="Mask radius at original resolution")
    parser.add_argument("--max_iter", default=5000, type=int,
                        help="Max training iterations (default 5000)")
    parser.add_argument("--gap", default=10, type=int,
                        help="Temporal gap between frames for motion detection (default 10)")
    opt = parser.parse_args()

    folder = setup_experiment(opt)

    # Setup
    print("[1/8] Setting up configuration...")

    # Create GT SDFs
    print("\n[2/8] Creating ground truth SDF...")
    # Load sinogram directly from file
    sinogram_file_path = '/media/ExtraDrive1/kvibhandik/XCAT/data/store/94a44f1d91b0f1b3/sinogram/sinogram_main.npz'
    result = np.load(sinogram_file_path)
    sinogram = result['sinogram']
    angles = result['angles']

    sinogram = downsample_sinogram_bin_average(
        sinogram.T,
        target_detectors=IMAGE_RESOLUTION
    )

    config = ConfigXCAT.from_xcat(
        views_per_rotation=1000,
        xcat_detectors=IMAGE_RESOLUTION,
        HEART_BEAT_PERIOD=1000 * 60 / HEART_RATE,
        NUM_SDFS=3
    )
    all_thetas = np.linspace(
        0, config.THETA_MAX, config.TOTAL_CLICKS, endpoint=False)
    print(
        f"  Total angles: {len(all_thetas)}, Theta range: [{all_thetas[0]:.1f}, {all_thetas[-1]:.1f}]")

    sdf_gt = SDFGtXCAT(config, sinogram, angles)

    config.set_intensities(sdf_gt.intensities)

    # Generate sinogram and FBP
    print("\n[3/8] Generating sinogram and FBP reconstruction...")
    sinogram, reconstruction_fbp = sdf_gt.sinogram, sdf_gt.reconstruction_fbp
    np.save(f'{folder}1_baseline/fbp', reconstruction_fbp)
    print(f"  Saved: {folder}1_baseline/fbp.npy")
    print("  Saving FBP GIF...")
    save_movie_as_gif(
        reconstruction_fbp, f'{folder}1_baseline/fbp.gif', 'FBP Reconstruction', fps=15, subsample=5)

    # Create motion mask if motion regularization is enabled
    motion_mask = None
    if opt.motion_coeff > 0.0:
        print(f"\n[3.5/8] Creating motion mask for regularization...")
        motion_mask = create_circular_mask_xcat(
            target_resolution=IMAGE_RESOLUTION,
            center=(opt.mask_center_x, opt.mask_center_y),
            radius=opt.mask_radius,
            original_resolution=ORIGINAL_RESOLUTION
        )
        # Save mask visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(motion_mask, cmap='Blues')
        plt.title(f'Motion Mask (coeff={opt.motion_coeff})')
        plt.colorbar(label='1=motion allowed, 0=forbidden')
        plt.savefig(f'{folder}1_baseline/motion_mask.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved mask visualization: {folder}1_baseline/motion_mask.png")

    # Extract pretraining SDFs
    print("\n[4/8] Extracting pretraining SDFs from FBP...")
    pretraining_sdfs, init = sdf_gt.pretraining_sdfs, sdf_gt.intensities
    print(f"  Pretraining SDFs shape: {pretraining_sdfs.shape}")
    print(f"  Initial intensities: {init}")

    # Create network and trainer
    print("\n[5/8] Creating neural network...")
    sdf_network = SDFNCT(config, scale=SCALE)
    trainer = Trainer(config, sdf_network)
    print(
        f"  Network parameters: {sum(p.numel() for p in sdf_network.parameters()):,}")

    # Save untrained state
    print("\n  Saving untrained network state...")
    save_sdf_state(sdf_network, config,
                   f'{folder}1_baseline/untrained_network', all_thetas=all_thetas, subsample=3, fps=15)

    # Pretrain cycle
    print("\n[6/8] PRETRAIN CYCLE")
    sdf_network = pretrain_cycle(trainer, sdf_gt, pretraining_sdfs, all_thetas,
                                 pretrain_lr=LR, filename_prefix=f'{folder}2_pretrain/pretrain')

    # Get ground truth sinogram
    print("[6.5/8] Generating ground truth sinogram...")
    gt_sinogram = torch.from_numpy(get_sinogram(config, sdf_gt,
                                                Intensities(
                                                    config, learnable=False),
                                                all_thetas, offset=opt.offset)).cuda()
    print(f"  GT sinogram shape: {gt_sinogram.shape}")

    # Initial training
    print("\n[7/8] TRAIN CYCLE")
    sdf_network, intensities = train_cycle(trainer, sdf_gt, gt_sinogram,
                                           all_thetas, init, opt.offset, opt.tvs, opt.tvt,
                                           train_lr=LR, filename_prefix=f'{folder}3_train/train',
                                           motion_mask=motion_mask, coeff_motion=opt.motion_coeff,
                                           max_iter=opt.max_iter, motion_gap=opt.gap)

    # Refinement cycle
    print("\n[8/8] REFINE CYCLE")
    sdf_network, intensities = refine_cycle(trainer, sdf_gt, gt_sinogram,
                                            all_thetas, init, opt.offset, opt.tvs, opt.tvt,
                                            pretrain_lr=5e-5, train_lr=1e-5,
                                            filename_prefix=f'{folder}4_refine/refine',
                                            motion_mask=motion_mask, coeff_motion=opt.motion_coeff,
                                            max_iter=opt.max_iter, motion_gap=opt.gap)

    # Save final results
    save_final_results(config, sdf_gt, sdf_network, all_thetas,
                       f'{folder}5_final/result', reconstruction_fbp)

    print("=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {folder}")
    print("=" * 60)


if __name__ == '__main__':
    main()

