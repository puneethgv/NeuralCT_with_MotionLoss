from pathlib import Path
import json
import hashlib
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import wraps
import torch
import multiprocessing as mp
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# Disk Caching Utilities
# =============================================================================

def disk_cache(cache_dir: str = "cache"):
    """
    Decorator for caching function results to disk.

    Works with numpy arrays and tuples of numpy arrays. If the function is called
    with the same arguments, returns cached result from disk instead of recomputing.

    Supports:
        - Single numpy array: saved as .npy
        - Tuple of numpy arrays: saved as .npz with keys 'arr_0', 'arr_1', etc.

    Args:
        cache_dir: Directory to store cached results (default: "cache")

    Usage:
        @disk_cache("cache")
        def expensive_function(arr, param1, param2):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Generate a unique hash from function name and arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            cache_file_npy = cache_path / f"{func.__name__}_{cache_key}.npy"
            cache_file_npz = cache_path / f"{func.__name__}_{cache_key}.npz"
            metadata_file = cache_path / f"{func.__name__}_{cache_key}.json"

            # Check if cached result exists (either .npy or .npz)
            if metadata_file.exists():
                if cache_file_npy.exists():
                    print(
                        f"[CACHE HIT] Loading cached result for {func.__name__}")
                    return np.load(cache_file_npy)
                elif cache_file_npz.exists():
                    print(
                        f"[CACHE HIT] Loading cached result for {func.__name__}")
                    data = np.load(cache_file_npz)
                    # Reconstruct tuple
                    return tuple(data[f'arr_{i}'] for i in range(len(data.files)))

            # Compute result
            print(f"[CACHE MISS] Computing {func.__name__}...")
            result = func(*args, **kwargs)

            # Save result based on type
            if isinstance(result, tuple):
                # Save as .npz for tuple of arrays
                save_dict = {f'arr_{i}': arr for i, arr in enumerate(result)}
                np.savez(cache_file_npz, **save_dict)
                cache_file_used = cache_file_npz
            else:
                # Save as .npy for single array
                np.save(cache_file_npy, result)
                cache_file_used = cache_file_npy

            # Save metadata
            metadata = {
                "function": func.__name__,
                "cache_key": cache_key,
                "return_type": "tuple" if isinstance(result, tuple) else "array",
                "num_arrays": len(result) if isinstance(result, tuple) else 1
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[CACHED] Saved to {cache_file_used}")
            return result

        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a unique hash key from function arguments."""
    hasher = hashlib.sha256()

    # Add function name
    hasher.update(func_name.encode())

    # Hash positional arguments
    for arg in args:
        hasher.update(_hash_arg(arg))

    # Hash keyword arguments (sorted for consistency)
    for key in sorted(kwargs.keys()):
        hasher.update(key.encode())
        hasher.update(_hash_arg(kwargs[key]))

    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity


def _hash_arg(arg) -> bytes:
    """Convert an argument to bytes for hashing.

    Uses stable hashing methods that produce the same result across Python sessions.
    """
    if isinstance(arg, np.ndarray):
        # Use hashlib for stable hashing of numpy arrays (Python's hash() is not stable across runs)
        arr_hash = hashlib.md5(arg.tobytes()).hexdigest()
        return f"ndarray:{arg.shape}:{arg.dtype}:{arr_hash}".encode()
    elif isinstance(arg, (list, tuple)):
        # Recursively hash sequences
        result = b""
        for item in arg:
            result += _hash_arg(item)
        return result
    elif isinstance(arg, dict):
        result = b""
        for k in sorted(arg.keys()):
            result += str(k).encode() + _hash_arg(arg[k])
        return result
    elif isinstance(arg, (int, float, str, bool)):
        return str(arg).encode()
    elif arg is None:
        return b"None"
    else:
        # For other objects, use string representation
        return str(arg).encode()


# =============================================================================


def _render_frame_worker(args):
    """Worker for parallel rendering. Must be top-level for pickling."""
    idx, theta, panels, panel_configs, color_ranges, figsize = args

    n_panels = len(panel_configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(
        figsize[0] * n_panels, figsize[1]))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, title, cmap, show_cb) in zip(axes, panel_configs):
        data = panels.get(name)
        if data is None:
            ax.axis('off')
            continue
        vmin, vmax = color_ranges.get(name, (None, None))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Customize title and labels based on panel type
        if name in ['gt', 'pred', 'diff']:
            # For sinograms, show frame index instead of angle
            if theta == int(theta):
                title_text = f'{title}\nFrame: {int(theta)}'
            else:
                title_text = f'{title}\nAngle: {theta:.1f}°'
            ax.set_title(title_text, fontsize=14, fontweight='bold')
            ax.set_xlabel('Projection Angle')
            ax.set_ylabel('Detector Position')
        else:
            ax.set_title(f'{title}\nAngle: {theta:.1f}°',
                         fontsize=14, fontweight='bold')
            ax.axis('off')

        if show_cb:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return idx, image


@dataclass
class ColorRange:
    """Color range for visualization."""
    vmin: float
    vmax: float
    cmap: str = 'gray'


@dataclass
class PanelConfig:
    """Configuration for a single panel in the layout."""
    name: str
    title: str
    color_range: ColorRange
    show_colorbar: bool = True


# -----------------------------------------------------------------------------
# Data Providers (Strategy Pattern - computes frame data)
# -----------------------------------------------------------------------------

class DataProvider(ABC):
    """Abstract base class for providing frame data."""

    @abstractmethod
    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        """Return dict of panel_name -> 2D numpy array for given theta."""
        pass

    @abstractmethod
    def get_panel_configs(self) -> List[PanelConfig]:
        """Return list of panel configurations."""
        pass

    def compute_color_ranges(self, thetas: np.ndarray,
                             sample_count: int = 10) -> Dict[str, ColorRange]:
        """Pre-compute color ranges from sample frames."""
        sample_thetas = np.linspace(thetas[0], thetas[-1],
                                    min(sample_count, len(thetas)))

        panel_values = {}
        for theta in sample_thetas:
            frame_data = self.get_frame(theta)
            for name, data in frame_data.items():
                if name not in panel_values:
                    panel_values[name] = []
                panel_values[name].append(data)

        ranges = {}
        for name, values in panel_values.items():
            all_data = np.concatenate([v.flatten() for v in values])
            vmin, vmax = np.percentile(all_data, [1, 99])
            # Find matching panel config for cmap
            cmap = 'gray'
            for pc in self.get_panel_configs():
                if pc.name == name:
                    cmap = pc.color_range.cmap
                    break
            ranges[name] = ColorRange(vmin, vmax, cmap)

        return ranges


class SDFDataProvider(DataProvider):
    """Provides SDF data from a single SDF model."""

    def __init__(self, sdf, config, include_occupancy: bool = True):
        self.sdf = sdf
        self.config = config
        self.include_occupancy = include_occupancy
        # Import here to avoid circular imports
        from source.network.renderer import sdf_to_occ
        self.sdf_to_occ = sdf_to_occ

    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            sdf_val = self.sdf(theta)
            sdf_np = sdf_val.detach().cpu().numpy()

            # Save each SDF channel separately
            result = {}
            num_sdfs = sdf_np.shape[-1]
            for i in range(num_sdfs):
                result[f'sdf_{i}'] = sdf_np[..., i]

            if self.include_occupancy:
                # Convert SDF to occupancy (matching renderer.py:272)
                occ_val = self.sdf_to_occ(sdf_val)

                # Get intensities from config (matching renderer.py:274)
                # INTENSITIES can be 1D or 2D, flatten to ensure 1D then reshape to (1, 1, NUM_SDFS)
                intensities_array = np.array(self.config.INTENSITIES).flatten()
                intensities = torch.from_numpy(
                    intensities_array).view(1, 1, -1).type_as(occ_val)

                # Multiply occupancy by intensities and sum (matching renderer.py:277, 281)
                canvas = occ_val * intensities
                occ_combined = canvas.sum(dim=-1).detach().cpu().numpy()
                result['occupancy'] = occ_combined

            return result

    def get_panel_configs(self) -> List[PanelConfig]:
        configs = []
        # Add separate panel for each SDF channel
        num_sdfs = self.config.NUM_SDFS
        for i in range(num_sdfs):
            configs.append(
                PanelConfig(f'sdf_{i}', f'SDF Channel {i}',
                            ColorRange(-1, 1, 'RdBu_r'))
            )
        if self.include_occupancy:
            configs.append(
                PanelConfig('occupancy', 'Object Estimate',
                            ColorRange(0, 1, 'gray'))
            )
        return configs


class ComparisonDataProvider(DataProvider):
    """Provides comparison data between GT and predicted SDF."""

    def __init__(self, sdf_gt, sdf_pred, include_error: bool = True):
        self.sdf_gt = sdf_gt
        self.sdf_pred = sdf_pred
        self.include_error = include_error

    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        gt = self.sdf_gt(theta)[..., 0].detach().cpu().numpy()
        pred = self.sdf_pred(theta)[..., 0].detach().cpu().numpy()

        result = {'ground_truth': gt, 'predicted': pred}

        if self.include_error:
            result['error'] = np.abs(gt - pred)

        return result

    def get_panel_configs(self) -> List[PanelConfig]:
        configs = [
            PanelConfig('ground_truth', 'Ground Truth SDF',
                        ColorRange(-1, 1, 'RdBu_r')),
            PanelConfig('predicted', 'Predicted SDF',
                        ColorRange(-1, 1, 'RdBu_r')),
        ]
        if self.include_error:
            configs.append(
                PanelConfig('error', 'Absolute Error',
                            ColorRange(0, 0.5, 'hot'))
            )
        return configs


class MovieArrayDataProvider(DataProvider):
    """Provides data from a pre-computed 3D numpy array (H, W, T)."""

    def __init__(self, movie: np.ndarray, title: str = 'Movie',
                 cmap: str = 'gray'):
        self.movie = movie
        self.title = title
        self.cmap = cmap
        self.num_frames = movie.shape[2] if len(movie.shape) == 3 else 1

    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        # theta is used as frame index for array-based movies
        idx = int(theta) if isinstance(theta, (int, float)) else 0
        idx = min(idx, self.num_frames - 1)

        if len(self.movie.shape) == 3:
            return {'frame': self.movie[:, :, idx]}
        return {'frame': self.movie}

    def get_panel_configs(self) -> List[PanelConfig]:
        vmin, vmax = np.percentile(self.movie, [1, 99])
        return [
            PanelConfig('frame', self.title,
                        ColorRange(vmin, vmax, self.cmap))
        ]


class MultiMovieComparisonProvider(DataProvider):
    """Provides data from multiple pre-computed movie arrays for comparison."""

    def __init__(self, movies: Dict[str, np.ndarray],
                 titles: Dict[str, str] = None,
                 cmaps: Dict[str, str] = None):
        """
        Args:
            movies: Dict of name -> 3D numpy array (H, W, T)
            titles: Dict of name -> display title
            cmaps: Dict of name -> colormap name
        """
        self.movies = movies
        self.titles = titles or {k: k for k in movies}
        self.cmaps = cmaps or {k: 'gray' for k in movies}

        # All movies must have same number of frames
        self.num_frames = list(movies.values())[0].shape[2]

    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        idx = int(theta)
        idx = min(idx, self.num_frames - 1)
        return {name: movie[:, :, idx] for name, movie in self.movies.items()}

    def get_panel_configs(self) -> List[PanelConfig]:
        configs = []
        for name in self.movies.keys():
            movie = self.movies[name]
            vmin, vmax = np.percentile(movie, [1, 99])
            configs.append(PanelConfig(
                name=name,
                title=self.titles[name],
                color_range=ColorRange(vmin, vmax, self.cmaps[name]),
                show_colorbar=False
            ))
        return configs


class SinogramComparisonProvider(DataProvider):
    """Provides 2D sinogram comparison: GT, Predicted, and Difference."""

    def __init__(self, gt_sinogram, sdf_network, intensities, all_thetas, gantry_offset):
        from source.network.model import get_sinogram

        # Generate predicted sinogram
        with torch.no_grad():
            config = sdf_network.config if hasattr(
                sdf_network, 'config') else intensities.config
            pred_sinogram = get_sinogram(
                config, sdf_network, intensities, all_thetas, gantry_offset)

        # Convert to numpy
        if torch.is_tensor(gt_sinogram):
            gt_sinogram = gt_sinogram.detach().cpu().numpy()
        if torch.is_tensor(pred_sinogram):
            pred_sinogram = pred_sinogram.detach().cpu().numpy()

        if gt_sinogram.shape != pred_sinogram.shape:
            raise ValueError(
                f"Shape mismatch: GT {gt_sinogram.shape} vs Pred {pred_sinogram.shape}")

        self.gt_sinogram = gt_sinogram
        self.pred_sinogram = pred_sinogram
        self.difference = np.abs(gt_sinogram - pred_sinogram)

    def get_frame(self, theta: float) -> Dict[str, np.ndarray]:
        return {
            'gt': self.gt_sinogram,
            'pred': self.pred_sinogram,
            'diff': self.difference
        }

    def get_panel_configs(self) -> List[PanelConfig]:
        # Compute color ranges
        vmin = min(np.percentile(self.gt_sinogram, 1),
                   np.percentile(self.pred_sinogram, 1))
        vmax = max(np.percentile(self.gt_sinogram, 99),
                   np.percentile(self.pred_sinogram, 99))
        diff_vmax = np.percentile(self.difference, 99)

        return [
            PanelConfig(
                name='gt',
                title='Ground Truth\nSinogram',
                color_range=ColorRange(vmin, vmax, 'gray'),
                show_colorbar=True
            ),
            PanelConfig(
                name='pred',
                title='Predicted\nSinogram',
                color_range=ColorRange(vmin, vmax, 'gray'),
                show_colorbar=True
            ),
            PanelConfig(
                name='diff',
                title='Absolute\nDifference',
                color_range=ColorRange(0, diff_vmax, 'hot'),
                show_colorbar=True
            )
        ]


class MovieBuilder:
    """Orchestrates movie creation with parallel rendering."""

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.panel_configs = data_provider.get_panel_configs()
        self.figsize = (8, 8)
        self.subsample = 1
        self.fps = 15

    def with_subsample(self, subsample: int) -> 'MovieBuilder':
        self.subsample = subsample
        return self

    def with_fps(self, fps: int) -> 'MovieBuilder':
        self.fps = fps
        return self

    def with_figsize(self, width: float, height: float) -> 'MovieBuilder':
        self.figsize = (width, height)
        return self

    def build(self, thetas: np.ndarray, output_path: str):
        print(f"Creating movie: {output_path}")

        # Compute color ranges
        color_ranges = self.data_provider.compute_color_ranges(thetas)
        color_ranges_simple = {name: (cr.vmin, cr.vmax)
                               for name, cr in color_ranges.items()}

        # Simplify panel configs for pickling: (name, title, cmap, show_colorbar)
        panel_configs_simple = [
            (pc.name, pc.title, pc.color_range.cmap, pc.show_colorbar)
            for pc in self.panel_configs
        ]

        # Compute frame data on main process (GPU)
        frame_indices = list(range(0, len(thetas), self.subsample))
        print("Computing frame data...")
        args_list = []
        for idx in tqdm(frame_indices, desc="Computing"):
            theta = thetas[idx]
            panels = self.data_provider.get_frame(theta)
            args_list.append((idx, theta, panels, panel_configs_simple,
                              color_ranges_simple, self.figsize))

        # Render in parallel
        num_workers = max(1, int(mp.cpu_count() * 0.9))
        print(f"Rendering with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(_render_frame_worker, args_list),
                                total=len(args_list), desc="Rendering"))

        results.sort(key=lambda x: x[0])
        frames = [img for _, img in results]

        imageio.mimsave(output_path, frames, fps=self.fps)
        print(f"✓ Saved {len(frames)} frames to: {output_path}")


# =============================================================================
# Convenience Functions (Facade Pattern - simple API for common use cases)
# =============================================================================

def create_sdf_comparison_movie(sdf_gt, sdf_pred, all_thetas, filename,
                                subsample=3, fps=20):
    """
    Create side-by-side comparison movie of GT SDF vs Predicted SDF vs Error.

    This is a convenience function that uses the SOLID-compliant MovieBuilder
    internally.
    """
    provider = ComparisonDataProvider(sdf_gt, sdf_pred, include_error=True)

    MovieBuilder(provider)\
        .with_subsample(subsample)\
        .with_fps(fps)\
        .build(all_thetas, f"{filename}.gif")


def save_movie_as_gif(movie, output_file, title, fps=15, subsample=5):
    """
    Convert a 3D numpy array movie to a GIF file.

    This is a convenience function that uses the SOLID-compliant MovieBuilder
    internally.

    Args:
        subsample: Save every Nth frame (default=5 to reduce file size)
    """
    provider = MovieArrayDataProvider(movie, title=title)
    num_frames = movie.shape[2] if len(movie.shape) == 3 else 1

    # Use frame indices as "thetas" for array-based movies
    thetas = np.arange(num_frames, dtype=float)

    MovieBuilder(provider)\
        .with_subsample(subsample)\
        .with_fps(fps)\
        .build(thetas, output_file)


def save_sdf_state(sdf, config, filename, all_thetas=None, subsample=5, fps=15):
    """
    Save visualization of an SDF model's current state (SDF + Occupancy side-by-side).

    This is a convenience function that uses the SOLID-compliant MovieBuilder
    internally.
    """
    if all_thetas is None:
        all_thetas = np.linspace(0., config.THETA_MAX, config.TOTAL_CLICKS)

    provider = SDFDataProvider(sdf, config, include_occupancy=True)

    MovieBuilder(provider)\
        .with_subsample(subsample)\
        .with_fps(fps)\
        .build(all_thetas, f"{filename}.gif")


def create_final_comparison(fbp: np.ndarray, nct: np.ndarray, gt: np.ndarray,
                            output_file: str, fps: int = 15, subsample: int = 5):
    """
    Create side-by-side comparison GIF of GT vs NCT vs FBP.

    Args:
        fbp: FBP reconstruction array (H, W, T)
        nct: Neural network reconstruction array (H, W, T)
        gt: Ground truth array (H, W, T)
        output_file: Output GIF path
        fps: Frames per second
        subsample: Save every Nth frame (default=5 to reduce file size)
    """
    provider = MultiMovieComparisonProvider(
        movies={'gt': gt, 'nct': nct, 'fbp': fbp},
        titles={
            'gt': 'Ground Truth\n(Actual Motion)',
            'nct': 'Neural Network\n(SDFNCT)',
            'fbp': 'FBP Reconstruction\n(Baseline)'
        }
    )

    num_frames = gt.shape[2]
    thetas = np.arange(num_frames, dtype=float)

    MovieBuilder(provider)\
        .with_subsample(subsample)\
        .with_fps(fps)\
        .with_figsize(7, 7)\
        .build(thetas, output_file)


def create_sinogram_comparison(gt_sinogram, sdf_network, intensities, all_thetas,
                               gantry_offset, filename):
    """
    Create side-by-side comparison visualization of GT sinogram vs Predicted sinogram.

    Args:
        gt_sinogram: Ground truth sinogram (tensor or array)
        sdf_network: Trained SDF network
        intensities: Intensities object
        all_thetas: Array of projection angles
        gantry_offset: Gantry rotation offset
        filename: Output filename prefix
    """
    provider = SinogramComparisonProvider(
        gt_sinogram, sdf_network, intensities, all_thetas, gantry_offset)

    MovieBuilder(provider)\
        .with_figsize(6, 6)\
        .build(np.array([0.0]), f"{filename}_sinogram_comparison.png")


# =============================================================================
# Legacy Utilities (kept for compatibility)
# =============================================================================

def cuda_tensors(*param_names):
    """Convert specified numpy arrays to CUDA tensors before function call."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'

            args = list(args)
            for name in param_names:
                if name in kwargs and isinstance(kwargs[name], np.ndarray):
                    kwargs[name] = torch.from_numpy(kwargs[name]).cuda()
                elif name in params:
                    idx = params.index(name)
                    if idx < len(args) and isinstance(args[idx], np.ndarray):
                        args[idx] = torch.from_numpy(args[idx]).cuda()

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def fig_to_image(fig):
    """Convert a matplotlib figure to a numpy image array."""
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def save_frames_as_gif(frames, output_file, fps=15, loop=0):
    """Save a list of image frames as a GIF file."""
    imageio.mimsave(output_file, frames, fps=fps, loop=loop)
    print(f"✓ Saved {len(frames)} frames to: {output_file}")


def downsample_sinogram_bin_average(sinogram, target_detectors=128):
    """
    Downsample sinogram using bin averaging - most physically accurate for detector binning.
    Averages adjacent detector bins to simulate larger detector pixels.

    Args:
        sinogram: NumPy array with shape [detectors, angles]
        target_detectors: Desired number of detectors after downsampling

    Returns:
        Downsampled sinogram with shape [target_detectors, angles]
    """
    n_detectors, n_angles = sinogram.shape

    # Calculate bin size for detector dimension
    bin_size = n_detectors / target_detectors

    # Create output array
    downsampled = np.zeros((target_detectors, n_angles))

    for i in range(target_detectors):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)

        # Average all detector values in this bin
        downsampled[i, :] = np.mean(sinogram[start_idx:end_idx, :], axis=0)

    return downsampled
