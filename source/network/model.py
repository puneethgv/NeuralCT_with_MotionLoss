import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import warnings
from torch import optim
from torch.optim.lr_scheduler import StepLR
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from source.config import *
from source.data_generation.anatomy import *
from source.network.renderer import *
from source.network.siren import *
from tqdm import tqdm
from typing import Optional, Union
from source.util.utility import cuda_tensors, disk_cache
from scipy.ndimage import median_filter, zoom
from scipy import stats

RADIUS = 0.15
BAND = 180


def fetch_fbp_movie(config, body):

    intensities = Intensities(config, learnable=False)
    config = deepcopy(config)
    body = deepcopy(body)
    incr = 180

    config.THETA_MAX = config.THETA_MAX+2*incr
    if isinstance(body, SDF):
        sdf = body
    else:
        body.config.THETA_MAX = config.THETA_MAX
        sdf = SDFGt(config, body)
    THETA_MAX = config.THETA_MAX+incr

    with torch.no_grad():
        from skimage.transform import iradon
        all_thetas = np.linspace(-incr, config.THETA_MAX,
                                 config.TOTAL_CLICKS + 2*incr*2)
        gtrenderer = Renderer(config, sdf, intensities)
        sinogt = gtrenderer(all_thetas).detach().cpu().numpy()

    sinogram = sinogt.reshape(config.IMAGE_RESOLUTION, config.TOTAL_CLICKS +
                              2*incr*int(config.GANTRY_VIEWS_PER_ROTATION/360))
    reconstruction_fbp = np.zeros(
        (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION, config.TOTAL_CLICKS))
    count = 0
    for i in tqdm(range(0, config.TOTAL_CLICKS)):
        reconstruction_fbp[..., count] = iradon(
            sinogram[..., i:i+2*incr], theta=all_thetas[i:i+2*incr], circle=True).T
        count += 1

    return sinogram, 131*reconstruction_fbp


def generate_sinogram(config, sdf, intensities, all_thetas, gantry_offset=0.0):
    """Generate sinogram from SDF. (Single responsibility: sinogram generation)"""
    print(f"Generating sinogram ({len(all_thetas)} projections)...")
    with torch.no_grad():
        renderer = Renderer(config, sdf, intensities, offset=gantry_offset)
        sinogram = renderer(all_thetas).detach().cpu().numpy()
    print("✓ Sinogram complete")
    return sinogram.reshape(config.IMAGE_RESOLUTION, len(all_thetas))


@disk_cache("cache")
def reconstruct_fbp(sinogram, all_thetas, window_size, gantry_offset=0.0):
    """Perform FBP reconstruction. (Single responsibility: reconstruction)

    Results are cached to disk based on input arguments. If called with the same
    sinogram, angles, window_size, and offset, the cached result is returned.
    """
    from skimage.transform import iradon

    num_frames = sinogram.shape[1] - window_size + 1
    resolution = sinogram.shape[0]
    reconstruction = np.zeros((resolution, resolution, num_frames))

    print(f"FBP reconstruction ({num_frames} frames)...")
    for i in tqdm(range(num_frames), desc="FBP"):
        theta_slice = all_thetas[i:i + window_size]
        sino_slice = sinogram[:, i:i + window_size]
        frame = iradon(sino_slice, theta=theta_slice, circle=True)
        reconstruction[..., i] = frame
    print("✓ FBP reconstruction complete")
    return reconstruction


def fetch_fbp_movie_exp(config, sdf_gt, gantry_offset=0.0, intensity_scale=132):
    """Orchestrates sinogram generation and FBP reconstruction.

    Args:
        config: Config object
        sdf_gt: Ground truth SDF (injected, not created internally)
        gantry_offset: Gantry rotation offset
        intensity_scale: Scaling factor for reconstruction
    """
    incr = 180
    window_size = incr * int(config.GANTRY_VIEWS_PER_ROTATION / 360)

    all_thetas = np.linspace(
        -config.THETA_MAX/2 - incr/2,
        config.THETA_MAX/2 + incr/2,
        config.TOTAL_CLICKS + window_size
    )

    intensities = Intensities(config, learnable=False)
    sinogram = generate_sinogram(
        config, sdf_gt, intensities, all_thetas, gantry_offset)
    reconstruction = reconstruct_fbp(
        sinogram, all_thetas, window_size, gantry_offset)

    return sinogram, intensity_scale * reconstruction


def find_background_channel(image):

    assert isinstance(image, np.ndarray) and len(image.shape) == 3

    total = []
    for i in range(image.shape[2]):
        total.append(np.sum(image[..., i]))

    return total.index(max(total))


def get_n_objects(img, num_components=2):

    assert isinstance(img, np.ndarray) and len(
        img.shape) == 3, 'img must be a 3D numpy array'
    assert isinstance(num_components, int), 'num_components must be a integer'

    num_components += 3
    proceed = True
    count = 0
    while proceed and count < 3:
        mask = np.random.randint(0, img.shape[2], int(0.02*img.shape[2]))
        X = img[..., mask].reshape(-1, 1)
        gm = GaussianMixture(n_components=num_components,
                             random_state=0).fit(X)
        labels_to_use = np.where(gm.means_[:, 0] > 0.15)[0]

        if labels_to_use.shape[0] >= num_components-3:
            print('Found labels : {} needed {}. Tried {} times'.format(
                labels_to_use, num_components-3, count+1))
            proceed = False
        else:
            print('Segmentation failed, found labels : {} needed {}. Tried {} times'.format(
                labels_to_use, num_components-3, count+1))
            count += 1

    labels = np.zeros((img.shape))
    for i in range(img.shape[2]):
        label_image = np.zeros(
            (img.shape[0], img.shape[1], labels_to_use.shape[0]))
        count = 1
        sizes = []
        for idx, k in enumerate(labels_to_use):
            lbi = (gm.predict(img[..., i].reshape(-1, 1)
                              ).reshape(img.shape[0], img.shape[1]) == k)
            label_image[..., idx] = lbi*count
            sizes.append(np.sum(lbi))
            count += 1

        sizes2 = sizes
        sizes2.sort()

        if sizes2[num_components-3:] != []:
            labels_to_remove = sizes.index(sizes2[num_components-3:])
            label_image = np.delete(label_image, labels_to_remove, axis=2)

        label_image = np.sum(label_image, axis=2)

        labels[..., i] = label_image
    return labels.reshape(img.shape)

# @disk_cache("cache")


def get_n_objects_for_movie(fbp_movie, num_components=2):

    print("Computing Segmentations...")
    movie = get_n_objects(fbp_movie.copy(), num_components=num_components)
    movie_objects = np.zeros(
        (movie.shape[0], movie.shape[1], movie.shape[2], num_components))
    labels = np.arange(0, np.max(movie[..., 0])).astype(int)

    for i in range(movie.shape[2]):
        for l in labels:
            movie_objects[..., i, l] = (movie[..., i] == l+1)

    print("Computing SDFs...")
    init = np.zeros((1, num_components))
    for j in tqdm(range(num_components)):
        for i in range(movie.shape[2]):
            occupancy = np.round(denoise_tv_chambolle(
                movie_objects[..., i, j][..., np.newaxis]))
            movie_objects[..., i, j] = denoise_tv_chambolle(
                occ_to_sdf(occupancy), weight=2)[..., 0]
        img = fbp_movie[..., 0]
        test = np.where(movie_objects[..., 0, 0] > 0)  # [...,0]
        init[0, j] = np.median(img[test[0], test[1]])

    return movie_objects, init


def create_synthetic_pretraining_sdfs(config):
    """Create synthetic SDFs from random organs for pretraining."""
    pretraining_sdfs = np.zeros((config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION,
                                 config.TOTAL_CLICKS, config.NUM_SDFS))
    all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)

    for i in range(config.NUM_SDFS):
        organ = _create_default_organ(config, i)
        sdf_gt = SDFGt(organ.config, Body(organ.config, [organ]))

        for j in range(config.TOTAL_CLICKS):
            pretraining_sdfs[..., j, i] = denoise_tv_chambolle(
                sdf_gt(all_thetas[j])[..., 0].detach().cpu().numpy())

    return pretraining_sdfs, config.INTENSITIES


def extract_pretraining_sdfs_from_fbp(config, fbp_movie):
    """Extract SDFs by segmenting FBP reconstruction."""
    return get_n_objects_for_movie(fbp_movie, num_components=config.NUM_SDFS)


def extract_pretraining_sdfs_from_sdf(config, sdf):
    """Extract SDFs from an existing SDF model."""
    pretraining_sdfs = np.zeros((config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION,
                                 config.TOTAL_CLICKS, config.NUM_SDFS))
    all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)

    for i in range(config.NUM_SDFS):
        for j in range(config.TOTAL_CLICKS):
            occ = sdf_to_occ(sdf(all_thetas[j]))[..., i].detach().cpu().numpy()
            denoised = np.round(denoise_tv_chambolle(occ, weight=2))
            pretraining_sdfs[..., j, i] = occ_to_sdf(
                denoised[..., np.newaxis])[..., 0]

    return pretraining_sdfs, None


def get_pretraining_sdfs(config, source=None, xcat=False):
    """Dispatcher - delegates to appropriate function based on source type."""
    if source is None:
        return create_synthetic_pretraining_sdfs(config)
    elif isinstance(source, np.ndarray):
        if xcat:
            return extract_pretraining_sdfs_from_fbp_xcat(config, source)
        else:
            return extract_pretraining_sdfs_from_fbp(config, source)
    elif isinstance(source, SDF):
        return extract_pretraining_sdfs_from_sdf(config, source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")


# ============================================================================
# XCAT-Specific Segmentation Functions
# ============================================================================
# These functions are optimized for XCAT cardiac data with:
# - Heart-region cropping before GMM fitting
# - Better sampling (50k samples max)
# - Median filter post-processing
# - Spatial denoising
# - Intensity-based component sorting


def _crop_heart_region_xcat(movie: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """
    Crop the central heart region from XCAT FBP movie using a circular mask.

    Uses same logic as Experiment/source crop_heart_from_movie - creates a
    circular mask centered on the heart to exclude background pixels that
    would otherwise dominate the GMM fitting.

    Args:
        movie: 3D array (H, W, T)
        margin: Not used - kept for API compatibility

    Returns:
        Cropped movie focusing on heart region (circular mask applied)
    """
    from skimage import draw

    # Use same normalization as Experiment/source for XCAT images
    # Reference: 1700x1700 XCAT images have heart centered at (750, 640) with radius 260
    normalising_factor = 1700 / movie.shape[0]
    center = (750 / normalising_factor, 640 / normalising_factor)
    radius = 260 / normalising_factor

    # Create circular mask
    mask = np.zeros(movie[..., 0].shape, dtype=bool)
    rr, cc = draw.disk(center, radius, shape=mask.shape)
    mask[rr, cc] = True

    # Apply mask - set pixels outside circle to 0
    masked_movie = np.zeros_like(movie)
    for t in range(movie.shape[2]):
        masked_movie[..., t][mask] = movie[..., t][mask]

    # Crop to bounding box of circle
    y_min = int(max(center[0] - radius, 0))
    y_max = int(min(center[0] + radius, movie.shape[0]))
    x_min = int(max(center[1] - radius, 0))
    x_max = int(min(center[1] + radius, movie.shape[1]))

    cropped = masked_movie[y_min:y_max, x_min:x_max, :]

    return cropped


def _downsample_movie_xcat(movie: np.ndarray, target_size: int = 128) -> np.ndarray:
    """
    Downsample movie spatially for faster processing.

    Args:
        movie: 3D array (H, W, T)
        target_size: Target spatial resolution

    Returns:
        Downsampled movie (target_size, target_size, T)
    """
    h, w, t = movie.shape
    if h == target_size and w == target_size:
        return movie

    zoom_factors = (target_size / h, target_size / w, 1)
    downsampled = np.zeros((target_size, target_size, t))

    for i in range(t):
        downsampled[:, :, i] = zoom(
            movie[:, :, i],
            zoom_factors[:2],
            order=3,  # Cubic interpolation
            mode='reflect'
        )

    return downsampled


def visualize_gmm_fit_xcat(img, gm, labels_to_use, frame_idx=0, save_path=None):
    """
    Visualize GMM fit for XCAT data - histogram and spatial domains.

    Args:
        img: 3D image array (H, W, T)
        gm: Fitted GaussianMixture model
        labels_to_use: Component indices being used
        frame_idx: Frame to visualize
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get frame for visualization
    frame = img[..., frame_idx]

    # 1. Histogram with GMM components
    frame_data = frame.flatten()
    x_range = np.linspace(np.min(frame_data), np.max(frame_data), 1000)

    axes[0].hist(frame_data, bins=100, density=True, alpha=0.6, color='gray')
    axes[0].set_title('Intensity Histogram with GMM Components')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Density')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(gm.means_)))
    for i, (mean, covar, weight) in enumerate(zip(gm.means_, gm.covariances_, gm.weights_)):
        std = np.sqrt(covar[0]) if covar.ndim == 1 else np.sqrt(covar[0, 0])
        pdf = weight * stats.norm.pdf(x_range, mean[0], std)
        is_selected = i in labels_to_use
        linestyle = '-' if is_selected else '--'
        label = f'Comp {i} (μ={mean[0]:.3f}, σ={std:.3f})'
        if is_selected:
            label += ' ✓'
        axes[0].plot(x_range, pdf, linestyle=linestyle,
                     linewidth=2, color=colors[i], label=label)
    axes[0].legend(fontsize=8)

    # 2. Original image
    axes[1].imshow(img[..., frame_idx], cmap='gray')
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    # 3. GMM segmentation
    label_predicts = gm.predict(
        img[..., frame_idx].reshape(-1, 1)).reshape(img.shape[0], img.shape[1])
    seg_image = np.zeros((img.shape[0], img.shape[1], 3))
    for idx, component_idx in enumerate(labels_to_use):
        mask = label_predicts == component_idx
        seg_image[mask] = colors[component_idx][:3]

    axes[2].imshow(seg_image)
    axes[2].set_title('GMM Segmentation')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved XCAT GMM visualization to {save_path}")

    return fig


def fit_gmm_xcat(img: np.ndarray, num_components: int = 3,
                 max_samples: int = 50000, max_retries: int = 3) -> tuple:
    """
    Fit GMM to XCAT cardiac data with improved robustness.

    Key improvements over standard GMM fitting:
    - Crops heart region to exclude background
    - Limits sample size for stability (50k max)
    - Retry logic if segmentation fails

    Args:
        img: 3D array (H, W, T) - FBP movie
        num_components: Number of GMM components (typically 3 for XCAT)
        max_samples: Maximum samples for GMM fitting
        max_retries: Number of retry attempts

    Returns:
        gmm: Fitted GaussianMixture model
        labels_to_use: Array of component indices
    """
    # Crop to heart region for better GMM fitting
    cropped_img = _crop_heart_region_xcat(img)

    proceed = True
    count = 0
    gmm = None
    labels_to_use = None

    while proceed and count < max_retries:
        # Random temporal sampling
        mask = np.random.randint(
            0, cropped_img.shape[2], int(cropped_img.shape[2]))
        X = cropped_img[..., mask].reshape(-1, 1)

        # Subsample if too large (for stability)
        if X.shape[0] > max_samples:
            sample_indices = np.random.choice(
                X.shape[0], max_samples, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X

        gmm = GaussianMixture(n_components=num_components,
                              random_state=0).fit(X_sample)

        print(f"XCAT GMM means: {gmm.means_.flatten()}")

        # Use all components for XCAT (no thresholding)
        labels_to_use = np.arange(num_components)

        if labels_to_use.shape[0] >= num_components - 1:
            print(
                f'Found {len(labels_to_use)} labels. Tried {count + 1} times')
            proceed = False
        else:
            print(f'Segmentation attempt {count + 1} incomplete')
            count += 1

    return gmm, labels_to_use


@disk_cache("cache")
def get_n_objects_for_movie_xcat(fbp_movie: np.ndarray, num_components: int = 3,
                                 target_resolution: int = 128,
                                 apply_median_filter: bool = True,
                                 save_visualizations: bool = True) -> tuple:
    """
    XCAT-specific object segmentation with improved GMM fitting.

    Key improvements over standard get_n_objects_for_movie:
    - Crops heart region before GMM (excludes background)
    - Subsamples to 50k pixels for stable fitting
    - Applies median filter to clean segmentation
    - Downsamples for efficient processing
    - Sorts components by intensity

    Args:
        fbp_movie: 3D array (H, W, T) - FBP reconstruction
        num_components: Number of object classes (default 3 for XCAT)
        target_resolution: Spatial resolution for processing
        apply_median_filter: Whether to apply spatial median filter
        save_visualizations: Whether to save debug visualizations

    Returns:
        sdf: 4D array (H, W, T, num_components) - SDF for each component
        init: 1D array (1, num_components) - Median intensities
    """
    from pathlib import Path
    from datetime import datetime

    cache_path = Path("cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("XCAT GMM SEGMENTATION")
    print("=" * 60)
    print(f"Input shape: {fbp_movie.shape}")
    print(f"Target resolution: {target_resolution}")
    print(f"Num components: {num_components}")

    # Step 1: Fit GMM with XCAT-specific improvements
    print("\n[1/5] Fitting GMM...")
    gmm, labels_to_use = fit_gmm_xcat(fbp_movie, num_components=num_components)

    # Save visualization
    if save_visualizations:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = cache_path / f"xcat_gmm_fit_{timestamp}.png"
        fig = visualize_gmm_fit_xcat(
            fbp_movie, gmm, labels_to_use, save_path=str(viz_path))
        plt.close(fig)

    # Step 2: Predict labels for full movie
    print("\n[2/5] Computing segmentation labels...")
    labels_movie = gmm.predict(
        fbp_movie.reshape(-1, 1)).reshape(fbp_movie.shape)

    # Step 3: Apply median filter to clean up noise
    if apply_median_filter:
        print("\n[3/5] Applying median filter...")
        labels_movie = median_filter(labels_movie, size=(5, 5, 1))

    # Step 4: Downsample if needed
    if fbp_movie.shape[0] != target_resolution:
        print(
            f"\n[4/5] Downsampling from {fbp_movie.shape[0]} to {target_resolution}...")
        labels_movie = _downsample_movie_xcat(
            labels_movie.astype(float), target_resolution).astype(int)
        fbp_movie_ds = _downsample_movie_xcat(fbp_movie, target_resolution)
    else:
        print("\n[4/5] Skipping downsampling (already at target resolution)")
        fbp_movie_ds = fbp_movie

    # Step 5: Create binary masks and convert to SDFs
    print("\n[5/5] Computing SDFs...")
    h, w, t = labels_movie.shape
    movie_objects = np.zeros((h, w, t, num_components))

    # Create binary mask for each component
    for i in range(t):
        for l in range(num_components):
            movie_objects[..., i, l] = (labels_movie[..., i] == l)

    # Convert occupancy to SDF with denoising
    sdf = np.zeros_like(movie_objects)
    for i in tqdm(range(t), desc="Processing frames"):
        for j in range(num_components):
            occupancy = np.round(denoise_tv_chambolle(
                movie_objects[..., i, j][..., np.newaxis]))
            sdf[..., i, j] = denoise_tv_chambolle(
                occ_to_sdf(occupancy), weight=2)[..., 0]

    # Compute median intensities for each component
    init = np.zeros((1, num_components))
    for j in range(num_components):
        object_coords = np.where(sdf[..., j] > 0)
        if len(object_coords[0]) > 0:
            init[0, j] = np.median(
                fbp_movie_ds[object_coords[0], object_coords[1], object_coords[2]])

    # Sort components by intensity
    sorted_indices = np.argsort(init[0])
    sdf = sdf[..., sorted_indices]
    init = init[:, sorted_indices]

    print("\n" + "=" * 60)
    print("XCAT SEGMENTATION COMPLETE")
    print(f"SDF shape: {sdf.shape}")
    print(f"Component intensities (sorted): {init.flatten()}")
    print("=" * 60)

    return sdf, init


def extract_pretraining_sdfs_from_fbp_xcat(config, fbp_movie: np.ndarray) -> tuple:
    """
    Extract SDFs from FBP reconstruction for XCAT data.

    Uses XCAT-specific segmentation with:
    - Heart-region cropping
    - Improved GMM fitting
    - Median filter cleanup

    Args:
        config: Config object with NUM_SDFS
        fbp_movie: 3D FBP reconstruction (H, W, T)

    Returns:
        pretraining_sdfs: 4D array of SDFs
        init: Median intensity per component
    """
    return get_n_objects_for_movie_xcat(
        fbp_movie,
        num_components=config.NUM_SDFS,
        target_resolution=config.IMAGE_RESOLUTION
    )

# pretraining_sdfs, _ = get_pretraining_sdfs(config)
# print(np.mean(np.sqrt(np.gradient(pretraining_sdfs,axis=0)**2 + np.gradient(pretraining_sdfs,axis=1)**2)))
# print(np.mean(np.abs(np.gradient(pretraining_sdfs,axis=2))))


class FourierFeatures(nn.Module):
    '''
    Learning a function as a fourier series
    Refer: https://colab.research.google.com/github/ndahlquist/pytorch-fourier-feature-networks/blob/master/demo.ipynb#scrollTo=QDs4Im9WTQoy
    '''

    def __init__(self, input_channels, output_channels, mapping_size=128, scale=1.5, testing=False):
        super(FourierFeatures, self).__init__()

        assert isinstance(
            input_channels, int), 'input_channels must be an integer'
        assert isinstance(
            output_channels, int), 'output_channels must be an integer'
        assert isinstance(mapping_size, int), 'maping_size must be an integer'
        assert isinstance(scale, float), 'scale must be an float'
        assert isinstance(testing, bool), 'testing should be a bool'

        self.mapping_size = mapping_size
        self.output_channels = output_channels
        self.testing = testing

        if self.testing:
            self.B = torch.ones((1, self.mapping_size, self.output_channels))
        else:
            self.B = torch.randn(
                (1, self.mapping_size, self.output_channels))*scale

        self.B = self.B.cuda()
        self.net = Siren(input_channels, 128, 3,
                         (2*self.mapping_size+1)*self.output_channels)

    def forward(self, x, t):

        assert isinstance(x, torch.Tensor) and len(
            x.shape) == 2, 'x must be a 2D tensor'
        assert isinstance(t, torch.Tensor) or isinstance(
            t, float) and t >= -1 and t <= 1, 't must be a float between -1 and 1'

        if self.testing:
            fourier_coeffs = torch.ones(
                (x.shape[0], self.mapping_size*2+1, self.output_channels)).type_as(x)
        else:
            fourier_coeffs = self.net(
                x).view(-1, self.mapping_size*2+1, self.output_channels)

        fourier_coeffs_dc = fourier_coeffs[:, -1:, :]
        fourier_coeffs_ac = fourier_coeffs[:, :-1, :]

        assert fourier_coeffs_dc.shape == (
            x.shape[0], 1, self.output_channels), 'Inavild size for fourier_coeffs_dc : {}'.format(fourier_coeffs_dc.shape)
        assert fourier_coeffs_ac.shape == (
            x.shape[0], self.mapping_size*2, self.output_channels),  'Inavild size for fourier_coeffs_ac : {}'.format(fourier_coeffs_ac.shape)

        t = (2*np.pi*t*self.B).repeat(x.shape[0], 1, 1)

        tsins = torch.cat([torch.sin(t), torch.cos(t)], dim=1).type_as(x)

        assert tsins.shape == (
            x.shape[0], 2*self.mapping_size, self.output_channels)
        series = torch.mul(fourier_coeffs_ac, tsins)
        assert series.shape == (
            x.shape[0], 2*self.mapping_size, self.output_channels)
        val_t = torch.mean(series, dim=1, keepdim=True)
        assert val_t.shape == (x.shape[0], 1, self.output_channels)
        val_t = val_t + fourier_coeffs_dc
        assert val_t.shape == (x.shape[0], 1, self.output_channels)

        return val_t.squeeze(1)

# ff =  FourierFeatures(2, 2, testing=True).cuda()
# x = torch.Tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]).cuda()
# t = 1.0
# val_t = ff(x,t)
# print(torch.norm(val_t - 1.5*torch.ones(x.shape).type_as(x)))


class SDFNCT(SDF):
    def __init__(self, config, scale=1.5):
        super(SDFNCT, self).__init__()

        assert isinstance(
            config, (Config, ConfigXCAT)), 'config must be an instance of class Config'

        self.config = config
        x, y = np.meshgrid(np.linspace(0, 1, self.config.IMAGE_RESOLUTION), np.linspace(
            0, 1, self.config.IMAGE_RESOLUTION))
        self.pts = torch.autograd.Variable(2*(torch.from_numpy(np.hstack(
            (x.reshape(-1, 1), y.reshape(-1, 1)))).cuda().float()-0.5), requires_grad=True)

        self.encoder = Siren(2, 256, 3, config.NUM_SDFS).cuda()
        self.velocity = FourierFeatures(2, config.NUM_SDFS, scale=scale).cuda()

    def compute_sdf_t(self, t):
        assert isinstance(t, torch.Tensor) or isinstance(
            t, float), 't = {} must be a float or a tensor here'.format(t)
        assert t >= -1 and t <= 1, 't = {} is out of range'.format(t)

        displacement = self.velocity(self.pts, t)
        init_sdf = self.encoder(self.pts)
        assert init_sdf.shape == displacement.shape

        canvas = (init_sdf + displacement)*self.config.SDF_SCALING
        if not (torch.min(canvas) < -1 and torch.max(canvas) > 1):
            warnings.warn('SDF values are in a narrow range between (-1,1)')

        canvas = canvas.view(self.config.IMAGE_RESOLUTION,
                             self.config.IMAGE_RESOLUTION, self.config.NUM_SDFS)

        return canvas

    def forward(self, t):

        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= - \
            self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(
                t)

        t = 2*get_phase(self.config, t) - 1

        canvas = self.compute_sdf_t(t)
        assert len(canvas.shape) == 3, 'Canvas must be a 3D tensor, instead is of shape: {}'.format(
            canvas.shape)

        return canvas

    def grad(self, t):

        assert isinstance(t, float), 't = {} must be a float here'.format(t)
        assert t >= - \
            self.config.THETA_MAX and t <= self.config.THETA_MAX, 't = {} is out of range'.format(
                t)

        t = torch.autograd.Variable(torch.Tensor(
            [2*get_phase(self.config, t) - 1]).cuda().float(), requires_grad=True)

        canvas = self.compute_sdf_t(t)/self.config.SDF_SCALING

        dc_dxy = gradient(canvas, self.pts)
        assert len(dc_dxy.shape) == 2, 'Must be a 2D tensor, instead is {}'.format(
            dc_dxy.shape)

        occupancy = sdf_to_occ(canvas)
        assert len(occupancy.shape) == 3

        do_dxy = gradient(occupancy, self.pts)
        assert len(do_dxy.shape) == 2, 'Must be a 2D tensor, instead is {}'.format(
            do_dxy.shape)

        dc_dt = gradient(occupancy, t)/(np.prod(canvas.shape))
#         dc_dt = gradient(canvas, t)/(np.prod(canvas.shape))
        assert len(dc_dt.shape) == 1, 'Must be a 1D tensor, instead is {}'.format(
            dc_dt.shape)

        eikonal = torch.abs(torch.norm(dc_dxy, dim=1) - 1).mean()
        total_variation_space = torch.norm(do_dxy, dim=1).mean()
        total_variation_time = torch.abs(dc_dt)

        return eikonal, total_variation_space, total_variation_time


class Trainer:

    def __init__(self, config, model: SDFNCT):
        self.config = config
        self.sdf_network = model.cuda()

    def _create_optimizer(self, lr):
        """Factory method - override in subclass to use different optimizer."""
        return optim.Adam(list(self.sdf_network.parameters()), lr=lr)

    def _create_scheduler(self, optimizer):
        """Factory method - override in subclass to use different scheduler."""
        return StepLR(optimizer, step_size=1, gamma=0.95)

    @disk_cache("cache")
    def pretrain_sdf(self, pretraining_sdfs, lr=1e-4):
        """Pretrain SDF network on ground truth SDFs."""
        if isinstance(pretraining_sdfs, np.ndarray):
            pretraining_sdfs = torch.from_numpy(pretraining_sdfs).cuda()

        assert len(
            pretraining_sdfs.shape) == 4, f'Invalid shape: {pretraining_sdfs.shape}'

        optimizer = self._create_optimizer(lr)
        scheduler = self._create_scheduler(optimizer)
        gt = pretraining_sdfs

        for itr in range(1000):
            optimizer.zero_grad()
            t = np.random.randint(0, self.config.TOTAL_CLICKS, 1)[0]
            theta = t*(self.config.THETA_MAX /
                       self.config.TOTAL_CLICKS)  # [0, 360]

            pred = self.sdf_network(theta)
            target = gt[..., t, :]
            assert target.shape == pred.shape, f'target: {target.shape}, pred: {pred.shape}'

            eikonal, _, _ = self.sdf_network.grad(theta)
            loss1 = torch.abs(pred - target).mean()
            loss = loss1 + 0.1*eikonal
            loss.backward()
            optimizer.step()

            if itr % 200 == 0:
                print(f'itr: {itr}, loss: {loss.item():.4f}, lossP: {loss1.item():.4f}, '
                      f'lossE: {eikonal.item():.4f}, lr: {scheduler.get_last_lr()[0]*1e4:.4f}')

                # Save target and prediction pairs as images
                from pathlib import Path
                pretrain_dir = Path("cache/pretrain_pairs")
                pretrain_dir.mkdir(parents=True, exist_ok=True)

                target_np = target.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy()

                # Save each SDF channel
                for ch in range(target_np.shape[2]):
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(target_np[..., ch], cmap='gray')
                    axes[0].set_title(f'Target SDF {ch}')
                    axes[0].axis('off')
                    axes[1].imshow(pred_np[..., ch], cmap='gray')
                    axes[1].set_title(f'Pred SDF {ch}')
                    axes[1].axis('off')
                    plt.tight_layout()
                    plt.savefig(
                        pretrain_dir / f"itr{itr}_sdf{ch}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)

                scheduler.step()

        return self.sdf_network

    def train(self, gt_sinogram, init, lr=1e-4,
              gantry_offset=0.0, coefftvs=0.5, coefftvt=0.5,
              motion_mask=None, coeff_motion=1.0, max_iter=5000, motion_gap=10):
        """Train on sinogram data with optional motion regularization.
        
        Args:
            gt_sinogram: Ground truth sinogram tensor
            init: Initial intensity values
            lr: Learning rate (default 1e-4)
            gantry_offset: Gantry rotation offset (default 0.0)
            coefftvs: Spatial total variation coefficient (default 0.5)
            coefftvt: Temporal total variation coefficient (default 0.5)
            motion_mask: Optional motion mask (H, W), 1=motion allowed, 0=forbidden
            coeff_motion: Motion regularization coefficient (default 1.0)
            max_iter: Maximum training iterations (default 5000)
            motion_gap: Temporal gap between frames for motion detection (default 10)
            max_iter: Maximum training iterations (default 5000)
            
        Returns:
            sdf_network: Trained SDFNCT model
            intensities: Learned intensity values
        """
        optimizer = self._create_optimizer(lr)
        scheduler = self._create_scheduler(optimizer)

        intensities = Intensities(self.config, learnable=False, init=init)
        renderer = Renderer(self.config, self.sdf_network, intensities,
                            offset=gantry_offset)

        # Print motion regularization info if enabled
        if motion_mask is not None:
            print(f"  Motion regularization enabled: coeff={coeff_motion}")
            print(f"  Mask coverage: {np.sum(motion_mask) / motion_mask.size * 100:.1f}% of image")

        for itr in range(max_iter):
            optimizer.zero_grad()
            t = np.random.randint(
                0, self.config.TOTAL_CLICKS, self.config.BATCH_SIZE)
            theta = t*(self.config.THETA_MAX/self.config.TOTAL_CLICKS)
            pred = renderer(theta)

            target = gt_sinogram[:, t]
            loss1 = torch.abs(pred - target).mean()*100

            eikonal, total_variation_space, total_variation_time = self.sdf_network.grad(
                theta[0])
            assert target.shape == pred.shape, f'target: {target.shape}, pred: {pred.shape}'

            loss = loss1 + 0.1*eikonal + coefftvs * \
                total_variation_space + coefftvt*total_variation_time

            # Motion regularization loss
            loss_motion = torch.tensor(0.0, device=gt_sinogram.device)
            if motion_mask is not None:
                # Use first few samples from batch with a temporal gap
                gap = motion_gap  # Use the parameter instead of hardcoded value
                max_motion_samples = min(16, len(t))  # Limit for memory
                valid_pairs = 0
                for i in range(max_motion_samples):
                    t_idx = t[i]
                    t_idx_next = t_idx + gap
                    # Only use if the paired frame is within bounds
                    if t_idx_next < self.config.TOTAL_CLICKS:
                        theta1 = t_idx * (self.config.THETA_MAX / self.config.TOTAL_CLICKS)
                        theta2 = t_idx_next * (self.config.THETA_MAX / self.config.TOTAL_CLICKS)
                        # Enable debug on first iteration
                        debug_motion = (itr == 0 and i == 0)
                        loss_motion = loss_motion + compute_motion_regularization(
                            self.sdf_network, motion_mask, theta1, theta2, self.config, debug=debug_motion)
                        valid_pairs += 1
                if valid_pairs > 0:
                    loss_motion = loss_motion / valid_pairs
                    loss = loss + coeff_motion * loss_motion

            loss.backward()
            optimizer.step()

            if itr % 200 == 0:
                if motion_mask is not None:
                    print(f'itr: {itr}, loss: {loss.item():.4f}, lossP: {loss1.item():.4f}, '
                          f'lossE: {eikonal.item():.4f}, lossTVs: {total_variation_space.item():.4f}, '
                          f'lossTVt: {total_variation_time.item():.4f}, lossMot: {loss_motion.item():.6f}, '
                          f'lr: {scheduler.get_last_lr()[0]*1e4:.4f}')
                else:
                    print(f'itr: {itr}, loss: {loss.item():.4f}, lossP: {loss1.item():.4f}, '
                          f'lossE: {eikonal.item():.4f}, lossTVs: {total_variation_space.item():.4f}, '
                          f'lossTVt: {total_variation_time.item():.4f}, lr: {scheduler.get_last_lr()[0]*1e4:.4f}')
                scheduler.step()

            if loss1.item() < 0.08:
                break

        return self.sdf_network, intensities


def get_sinogram(config, sdf, intensities, all_thetas=None, offset=0.0):
    renderer = Renderer(config, sdf, intensities, offset=offset)
    if all_thetas is None:
        all_thetas = np.linspace(0, config.THETA_MAX, config.TOTAL_CLICKS)
    sinogram = renderer.forward(all_thetas).detach().cpu().numpy()

    return sinogram


# ============================================================================
# Motion Regularization Functions
# ============================================================================
# These functions implement motion regularization loss that penalizes motion
# detected OUTSIDE a precomputed mask. This is useful for constraining neural
# reconstructions to have motion only in expected regions (e.g., heart).

def compute_motion_regularization(sdf, mask, theta1, theta2, config, organ_idx=0, debug=False):
    """
    Compute motion regularization loss using temporal SDF product.
    
    The key insight: if SDF(x,t1) * SDF(x,t2) < 0, the boundary crossed point x,
    indicating motion. We penalize motion detected OUTSIDE the precomputed mask.
    
    Args:
        sdf: SDFNCT model
        mask: Precomputed motion mask (H, W), 1=motion allowed, 0=motion forbidden
        theta1, theta2: Adjacent time points (gantry angles)
        config: Configuration object
        organ_idx: Which organ's SDF to use (default 0 for cardiac)
        debug: If True, print debug information
        
    Returns:
        loss_motion: Scalar tensor penalizing motion outside the mask
    """
    # Get SDF at two adjacent times
    sdf_t1 = sdf(theta1)[:, :, organ_idx]  # (H, W)
    sdf_t2 = sdf(theta2)[:, :, organ_idx]  # (H, W)
    
    # Motion indicator: negative product means boundary crossed (motion detected)
    product = sdf_t1 * sdf_t2
    motion_magnitude = torch.relu(-product)  # Positive where motion detected, 0 otherwise
    
    # Convert mask to tensor if needed
    if isinstance(mask, np.ndarray):
        mask_tensor = torch.from_numpy(mask).to(product.device).float()
    else:
        mask_tensor = mask.to(product.device).float()
    
    # outside_mask = 1 where motion should NOT happen
    outside_mask = 1.0 - mask_tensor
    
    # Penalize motion detected outside the mask
    loss_motion = (motion_magnitude * outside_mask).sum()
    
    if debug:
        print(f"  [DEBUG Motion] theta1={theta1:.2f}, theta2={theta2:.2f}")
        print(f"    SDF t1: min={sdf_t1.min().item():.4f}, max={sdf_t1.max().item():.4f}")
        print(f"    SDF t2: min={sdf_t2.min().item():.4f}, max={sdf_t2.max().item():.4f}")
        print(f"    Product: min={product.min().item():.4f}, max={product.max().item():.4f}")
        print(f"    Negative products (motion detected): {(product < 0).sum().item()} pixels")
        print(f"    Motion outside mask: {(motion_magnitude * outside_mask).sum().item():.6f}")
    
    return loss_motion


def save_sdf_products_visualization(sdf, mask, config, save_path, n_samples=10, organ_idx=0):
    """
    Save a grid of 10 random SDF product visualizations to see motion detection.
    
    Each subplot shows:
    - Red/blue: SDF product (red = negative = motion detected)
    - Green contour: mask boundary
    
    Args:
        sdf: SDFNCT model
        mask: Precomputed motion mask (H, W)
        config: Configuration object
        save_path: Path to save the image
        n_samples: Number of random samples to visualize (default 10)
        organ_idx: Which organ's SDF to use (default 0)
    """
    # Create 2x5 grid for 10 samples
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(n_samples):
            # Sample random adjacent pair
            t_idx = np.random.randint(0, config.TOTAL_CLICKS - 1)
            theta1 = t_idx * (config.THETA_MAX / config.TOTAL_CLICKS)
            theta2 = (t_idx + 1) * (config.THETA_MAX / config.TOTAL_CLICKS)
            
            # Get SDF at two adjacent times
            sdf_t1 = sdf(theta1)[:, :, organ_idx].cpu().numpy()
            sdf_t2 = sdf(theta2)[:, :, organ_idx].cpu().numpy()
            
            # Compute product
            product = sdf_t1 * sdf_t2
            
            # Threshold: motion detected where product < 0
            motion_detected = (product < 0).astype(float)
            
            # Plot
            ax = axes[i]
            
            # Show product values (red = negative = motion)
            im = ax.imshow(product, cmap='RdBu', vmin=-np.abs(product).max(), vmax=np.abs(product).max())
            
            # Overlay motion detected regions
            ax.contour(motion_detected, levels=[0.5], colors='yellow', linewidths=1)
            
            # Overlay mask boundary
            if mask is not None:
                ax.contour(mask, levels=[0.5], colors='green', linewidths=1, linestyles='--')
            
            ax.set_title(f't={t_idx}->{t_idx+1}', fontsize=9)
            ax.axis('off')
    
    plt.suptitle('SDF Products: Blue=same sign (no motion), Red=opposite sign (motion)\nYellow=motion boundary, Green=mask', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved SDF product visualization to: {save_path}')


def create_motion_mask_from_fbp(fbp_movie: np.ndarray, threshold_percentile: float = 75,
                                 dilation_size: int = 5, heart_region: bool = True) -> np.ndarray:
    """
    Create a motion mask from FBP movie for XCAT data.
    
    This helper function makes it easy to create motion masks for XCAT experiments.
    The mask indicates regions where motion IS ALLOWED (1) vs forbidden (0).
    
    Args:
        fbp_movie: 3D array (H, W, T) - FBP reconstruction
        threshold_percentile: Percentile threshold for motion detection (default 75)
        dilation_size: Kernel size for mask dilation (default 5)
        heart_region: If True, automatically crops to heart region for XCAT data
        
    Returns:
        mask: 2D array (H, W) where 1=motion allowed, 0=motion forbidden
        
    Example:
        >>> fbp_movie = np.load('fbp_reconstruction.npy')
        >>> motion_mask = create_motion_mask_from_fbp(fbp_movie)
        >>> # Pass to Trainer.train(..., motion_mask=motion_mask)
    """
    # Compute temporal variance to detect motion
    temporal_variance = np.var(fbp_movie, axis=2)
    
    # Optionally focus on heart region for XCAT
    if heart_region and fbp_movie.shape[0] > 200:
        # Use same cropping logic as XCAT segmentation
        normalising_factor = 1700 / fbp_movie.shape[0]
        center = (int(750 / normalising_factor), int(640 / normalising_factor))
        radius = int(260 / normalising_factor)
        
        # Create circular mask for heart region
        from skimage import draw
        heart_mask = np.zeros(temporal_variance.shape, dtype=bool)
        rr, cc = draw.disk(center, radius, shape=heart_mask.shape)
        heart_mask[rr, cc] = True
        
        # Only consider variance within heart region
        temporal_variance = temporal_variance * heart_mask
    
    # Threshold to create binary mask
    threshold = np.percentile(temporal_variance[temporal_variance > 0], threshold_percentile)
    mask = (temporal_variance > threshold).astype(np.float32)
    
    # Dilate to include nearby regions
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def fetch_movie(config, sdf, all_thetas=None):
    frames = np.zeros((config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION,
                       config.TOTAL_CLICKS, config.NUM_SDFS))
    for i in range(config.NUM_SDFS):
        if all_thetas is None:
            all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)
        for j in range(config.TOTAL_CLICKS):
            frames[..., j, i] = sdf_to_occ(
                sdf(all_thetas[j]))[..., i].detach().cpu().numpy()

    intensities = config.INTENSITIES.reshape(1, 1, 1, -1)
    movie = np.sum(frames*intensities, axis=3)

    return movie


def save_movie(movie, file_name, subsample=5):
    """Save movie frames as PNG files and create a ZIP archive.

    Args:
        movie: 3D numpy array (H, W, T)
        file_name: Output directory name
        subsample: Save every Nth frame (default=5 to reduce file count)
    """
    assert isinstance(movie, np.ndarray) and len(
        movie.shape) == 3, 'movie must be a 3D numpy array'

    os.system('rm -r {}/ && mkdir {}'.format(file_name, file_name))
    frame_count = 0
    for i in range(0, movie.shape[2], subsample):
        plt.imsave('{}/{}.png'.format(file_name, i),
                   movie[..., i], cmap='gray')
        frame_count += 1
    if os.path.exists('{}.zip'.format(file_name)):
        os.system('rm {}.zip'.format(file_name))

    os.system('zip -r {}.zip {}/'.format(file_name, file_name))
    print(f'Saved {frame_count} frames to {file_name}.zip')
