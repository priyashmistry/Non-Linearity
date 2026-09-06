# =========================
# Standard library
# =========================
import os
from collections import defaultdict
from pathlib import Path
import argparse
import traceback


# =========================
# Third-party libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
from scipy.optimize import least_squares, curve_fit
from scipy.signal import medfilt
from scipy.stats import gaussian_kde
from scipy.ndimage import label
from astropy.io import fits
import skimage.transform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=AstropyWarning)


# ============================================================
# FITS loading utility
# ============================================================

"""def load_fits_images(directory):
    Load FITS images and headers from a directory.
    images, headers = [], []

    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".fits"):
            path = os.path.join(directory, fname)

            with fits.open(path) as hdul:
                data = hdul[0].data.astype(float)
                data[~np.isfinite(data)] = np.nan  # sanitize early
                images.append(data)
                headers.append(hdul[0].header)

    return np.array(images), headers"""

def load_fits_images(directory):
    """Load FITS and FITS.GZ images and headers from a directory."""
    images, headers = [], []

    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith((".fits", ".fits.gz")):
            path = os.path.join(directory, fname)

            with fits.open(path) as hdul:
                data = hdul[0].data.astype(float)
                data[~np.isfinite(data)] = np.nan  # sanitize early
                images.append(data)
                headers.append(hdul[0].header)

    return np.array(images), headers

# ============================================================
# AMP mode detection
# ============================================================

def detect_amp_mode(headers_dict):
    """
    Detect AMP_MODE from FITS headers.

    Rules:
    - If DETA1NM, DETA2NM, DETA3NM, DETA4NM all exist -> AMP_MODE = 4
    - If only DETA1NM and DETA2NM exist:
        - values Q1,Q2 -> AMP_MODE = 2.1
        - values Q3,Q4 -> AMP_MODE = 2.2
    """

    found = {}

    # Read any one header from any exposure directory
    for exp_dir, headers in headers_dict.items():

        if len(headers) == 0:
            continue

        hdr = headers[0]

        for i in range(1, 5):
            key = f"DETA{i}NM"

            if key in hdr:
                found[key] = str(hdr[key]).strip()

        break  # only need one file

    keys_found = sorted(found.keys())

    # --------------------------------------------------------
    # Case 1: all four amps present
    # --------------------------------------------------------
    required_all = [f"DETA{i}NM" for i in range(1, 5)]

    if all(k in found for k in required_all):
        return 4

    # --------------------------------------------------------
    # Case 2: only DETA1NM + DETA2NM
    # --------------------------------------------------------
    if ("DETA1NM" in found) and ("DETA2NM" in found):

        vals = {
            found["DETA1NM"],
            found["DETA2NM"]
        }

        if vals == {"Q1", "Q2"}:
            return 2.1

        if vals == {"Q3", "Q4"}:
            return 2.2

    return np.nan

# ============================================================
# Order / robust mask utilities
# ============================================================

def _mask_1d(array, jump_factor, smooth_window):
    n = len(array)
    mask_1d = np.zeros(n, dtype=int)

    array_smooth = np.convolve(
        array,
        np.ones(smooth_window) / smooth_window,
        mode="same",
    )

    state = 0
    for i in range(1, n):
        prev = array_smooth[i - 1]
        curr = array_smooth[i]

        if prev != 0:
            rel_diff = curr / (prev + 1e-8)
        else:
            rel_diff = curr / (np.median(array_smooth[array_smooth > 0]) + 1e-8)

        if rel_diff > jump_factor:
            state = 1
        elif rel_diff < 1 / jump_factor:
            state = 0

        mask_1d[i] = state

    return mask_1d


def make_order_mask(image, jump_factor=1.5, smooth_window=5, direction="row"):
    ny, nx = image.shape
    mask = np.zeros_like(image, dtype=int)

    if direction == "row":
        for y in range(ny):
            mask[y, :] = _mask_1d(image[y, :], jump_factor, smooth_window)

    elif direction == "col":
        for x in range(nx):
            mask[:, x] = _mask_1d(image[:, x], jump_factor, smooth_window)

    else:
        raise ValueError("direction must be 'row' or 'col'")

    return mask


def shrink_mask(mask):
    ny, nx = mask.shape
    new_mask = mask.copy()

    left_remove = [1, 2, 3, 7, 8, 9]
    right_remove = [1, 2, 3, 8, 9, 10]

    for y in range(ny):
        row = mask[y, :]
        in_band = False

        for x in range(nx + 1):
            if x < nx and row[x] == 1 and not in_band:
                in_band = True
                start = x

            elif (x == nx or row[x] == 0) and in_band:
                end = x - 1

                for k in left_remove:
                    idx = start + (k - 1)
                    if start <= idx <= end:
                        new_mask[y, idx] = 0

                for k in right_remove:
                    idx = end - (k - 1)
                    if start <= idx <= end:
                        new_mask[y, idx] = 0

                in_band = False

    return new_mask


def make_robust_mask(image, jump_factor=1.5, smooth_window=5):
    mask_row = make_order_mask(
        image,
        jump_factor=jump_factor,
        smooth_window=smooth_window,
        direction="row",
    )

    shrunk_mask = shrink_mask(mask_row)
    return shrunk_mask


def straighten_fibres(mask, width=5, smooth_kernel=11, min_rows=30):
    """
    Straighten thin vertical fibre-like components by:
    - extracting x-centre per row
    - smoothing x(y)
    - redrawing with constant width

    width: final fibre thickness in pixels (odd recommended: 3,5,7)
    smooth_kernel: median filter kernel for x(y), must be odd
    min_rows: ignore tiny components
    """
    mask = mask.astype(bool)
    ny, nx = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    # Label connected components (each fibre becomes one label)
    lab, n = label(mask)

    half = width // 2

    for comp_id in range(1, n + 1):
        comp = (lab == comp_id)

        ys, xs = np.where(comp)
        if ys.size == 0:
            continue

        # Reject tiny components
        if (ys.max() - ys.min() + 1) < min_rows:
            continue

        # For each row, find centre x of this component
        y_min, y_max = ys.min(), ys.max()
        x_center = np.full(y_max - y_min + 1, np.nan)

        for y in range(y_min, y_max + 1):
            xrow = np.where(comp[y])[0]
            if xrow.size > 0:
                x_center[y - y_min] = np.mean(xrow)

        valid = np.isfinite(x_center)
        if valid.sum() < 5:
            continue

        # Fill gaps in centreline
        x_center[~valid] = np.interp(
            np.where(~valid)[0],
            np.where(valid)[0],
            x_center[valid],
        )

        # Smooth the centreline
        x_smooth = medfilt(x_center, kernel_size=smooth_kernel)

        # Redraw constant width fibre
        for i, y in enumerate(range(y_min, y_max + 1)):
            xc = int(np.round(x_smooth[i]))
            L = max(0, xc - half)
            R = min(nx - 1, xc + half)
            out[y, L:R + 1] = True

    return out


# ============================================================
# Feature-length mask creation
# ============================================================

def create_masks(image, feature_ranges):
    """
    Create boolean masks by classifying contiguous segments
    according to their pixel length.
    """
    base_mask = (image > 0)
    height, width = base_mask.shape

    masks = {
        key: np.zeros_like(base_mask, dtype=bool)
        for key in feature_ranges
    }

    for i in range(height):
        row = base_mask[i].astype(int)

        starts = np.where(np.diff(np.concatenate([[0], row])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([row, [0]])) == -1)[0]
        lengths = ends - starts

        for s, e, l in zip(starts, ends, lengths):
            for key, allowed_lengths in feature_ranges.items():
                if l in allowed_lengths:
                    masks[key][i, s:e + 1] = True

    return masks


# ============================================================
# FITS saving utility
# ============================================================

def save_masks_as_fits(masks, SAVE_DIR, prefix=""):
    # Ensure the directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    for key, m in masks.items():
        fname = f"{prefix}{key}_mask.fits"
        path = os.path.join(SAVE_DIR, fname)

        fits.PrimaryHDU(m.astype(np.uint8)).writeto(
            path,
            overwrite=True,
        )

        print(f"Saved mask '{key}': {path}")


# ============================================================
# Plotting utility
# ============================================================

# Example for plot_masks
def plot_masks(masks, output_dir, name, title_prefix="", downsample_factor=2):
    
    fig, axes = plt.subplots(1, len(masks), figsize=(15, 5))

    if len(masks) == 1:
        axes = [axes]

    for ax, (key, m) in zip(axes, masks.items()):
        if downsample_factor > 1:
            m_plot = skimage.transform.resize(
                m.astype(float),
                (m.shape[0] // downsample_factor, m.shape[1] // downsample_factor),
                anti_aliasing=True
            )
        else:
            m_plot = m

        im = ax.imshow(m_plot, cmap="gray", origin="lower")
        im.set_rasterized(False)
        ax.set_title(f"{title_prefix}{key}")
        ax.axis("off")
        ax.set_rasterized(False)

    plt.savefig(os.path.join(output_dir, f"mask_{name}.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Linear model
# ============================================================

def linear(x, m, c):
    return m * x + c


# ============================================================
# Poly model
# ============================================================

def model_ratio_x6(x, a):
    """Ratio model anchored at one, using a scaled sixth-order term."""
    u = np.asarray(x, dtype=float) / 10000.0
    return 1.0 + a * u**6


# ============================================================
# Physical / parametric models (even-order only)
# ============================================================

def G_poly_even(M, params):
    g0, g6 = params
    m = M / 10000.0
    poly = 1 + g6 * m**6
    poly = np.maximum(poly, 1e-6)  # prevents negative gain
    return g0 * poly


def y_model_even(M, params, s):
    G = G_poly_even(M, params)
    return np.sqrt(M / G + (s / G)**2)


def residuals_even(params, M, y, yerr, s):
    return (y - y_model_even(M, params, s)) / yerr


def fit_quadrant_even(M, y, yerr, s):
    p0 = np.array([1.0, 0.0])
    lower = np.array([1e-3, -1.0])
    upper = np.array([1e3, 1.0])

    res = least_squares(
        residuals_even,
        p0,
        bounds=(lower, upper),
        args=(M, y, yerr, s),
        jac="2-point",
        max_nfev=20000,
    )

    params = res.x
    dof = max(1, M.size - params.size)
    s_sq = 2 * res.cost / dof

    try:
        JTJ_inv = np.linalg.inv(res.jac.T @ res.jac)
        cov = s_sq * JTJ_inv
        errs = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except np.linalg.LinAlgError:
        cov = None
        errs = np.full_like(params, np.nan)

    return params, errs, cov


def robust_equal_width_bins(x, y, n_bins=25, min_count=30,
                            statistic="mode", mode_method="histogram"):
    """Return the binned density ridge, scatter and fitting uncertainty."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    rows = []
    for i in range(n_bins):
        use = ((x >= edges[i]) &
               (x <= edges[i + 1] if i == n_bins - 1 else x < edges[i + 1]))
        vals = y[use]
        xs = x[use]
        if vals.size < min_count:
            continue
        if statistic == "mode":
            # Use the peak of the same one-dimensional distribution represented
            # by the hexbin, rather than its median (which is biased for the
            # strongly skewed CCD distributions).
            if mode_method == "histogram":
                hist, hist_edges = np.histogram(vals, bins=30)
                j = np.argmax(hist)
                centre = 0.5 * (hist_edges[j] + hist_edges[j + 1])
            elif mode_method == "kde":
                kde = gaussian_kde(vals)
                grid = np.linspace(vals.min(), vals.max(), 1000)
                centre = grid[np.argmax(kde(grid))]
            else:
                raise ValueError("mode_method must be 'histogram' or 'kde'")
        elif statistic == "median":
            centre = np.median(vals)
        else:
            raise ValueError("statistic must be 'mode' or 'median'")

        scatter = np.std(vals, ddof=1)

        # This is used only as a relative fit weight.  Cap the effective sample
        # size because neighbouring CCD pixels are not independent.
        n_effective = min(vals.size, 1000)
        error = scatter / np.sqrt(n_effective)
        x_centre = 0.5 * (edges[i] + edges[i + 1])
        rows.append((x_centre, centre, scatter, error, vals.size))
    if not rows:
        return tuple(np.array([]) for _ in range(5))
    return tuple(np.asarray(v) for v in zip(*rows))


# ============================================================
# Gaussian clip
# ============================================================

def gaussian_clip(x, y, nsigma=5.0, return_mask=False):
    """Robust MAD clipping without imposing artificial distribution symmetry."""
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    centre = np.median(y[finite])
    scatter = 1.4826 * np.median(np.abs(y[finite] - centre))
    if np.isfinite(scatter) and scatter > 0:
        keep = finite & (np.abs(y - centre) <= nsigma * scatter)
    else:
        keep = finite
    if return_mask:
        return x[keep], y[keep], keep
    return x[keep], y[keep]



# ============================================================
# Detector geometry / plotting layout
# ============================================================

quadrant_slices = {
    "Q1": (slice(2056, 4112), slice(0, 2101)),
    "Q2": (slice(0, 2056), slice(0, 2101)),
    "Q3": (slice(0, 2056), slice(2101, 4202)),
    "Q4": (slice(2056, 4112), slice(2101, 4202)),
}

plot_order = [["Q1", "Q4"], ["Q2", "Q3"]]
quadrants = ["Q1", "Q2", "Q3", "Q4"]

# ============================================================
# Runtime data containers
# ============================================================

images_dict = {}              # Input images
headers_dict = {}             # Headers
utmjd_dict = {}               # UTMJD

master_flats = {}             # Master flat-fields for normalisation
norm_factors_dict = {}        # Image normalisation factors
normalized_images_dict = {}   # Normalised images

results_nf = {}      # Normalisation factors summary
results_vm = {}      # Variance vs mean fit results
results_er = {}      # Exposure ratio fit results

results_nf_summary = {}   # exposure -> quadrant -> scalar median
results_nf_perfile = {}   # exposure -> quadrant -> array of 15
results_utmjd = {}        # exposure -> array of 15



def get_masks(SAVE_MASK, output_dir, default_mask, JUMP_FACTOR=1.5, SMOOTH_WINDOW=5, SAVE_DIR="Masks"):
    
    if default_mask:
        
        mask_file = "/Users/z5517274/Priyash/Non-Linearity Correction/Masks/region_mask_mask.fits"
        with fits.open(mask_file) as hdul:
            region_mask = hdul[0].data
            
        region_mask = region_mask.astype(bool)

        return region_mask, region_mask
    
    # ============================================================
    # Build normalised median images (short / long exposure)
    # ============================================================

    imgs_10 = np.array(images_dict["1_0"])
    median_10 = np.nanmedian(imgs_10, axis=0)

    imgs_01 = np.array(images_dict["0_1"])
    median_01 = np.nanmedian(imgs_01, axis=0)

    img_0_1 = np.nan_to_num(median_01, nan=0.0, posinf=0.0, neginf=0.0)
    img_1_0 = np.nan_to_num(median_10, nan=0.0, posinf=0.0, neginf=0.0)


    # ============================================================
    # Feature-length category definitions
    # ============================================================

    feature_ranges = {
        "Sky Fibres (Left)": list(range(1, 3)),
        "Sky Fibres (Right)": list(range(3, 5)),
        "Star Fibres": list(range(35, 40)),
    }


    # ============================================================
    # Step 1: Create robust common mask
    # ============================================================

    final_mask = make_robust_mask(
        img_1_0,
        jump_factor=JUMP_FACTOR,
        smooth_window=SMOOTH_WINDOW,
    )


    # ============================================================
    # Step 2: Apply mask to median images
    # ============================================================

    masked_image_0_1 = final_mask * img_0_1
    masked_image_1_0 = final_mask * img_1_0


    # ============================================================
    # Step 3: Generate feature-length masks
    # ============================================================

    long_masks = create_masks(masked_image_1_0, feature_ranges)
    short_masks = create_masks(masked_image_0_1, feature_ranges)

    for d in (long_masks, short_masks):
        d["Sky Fibres (Left)"] = straighten_fibres(
            d["Sky Fibres (Left)"],
            width=5,
            smooth_kernel=11,
        )
        d["Sky Fibres (Right)"] = straighten_fibres(
            d["Sky Fibres (Right)"],
            width=5,
            smooth_kernel=11,
        )

    plot_masks(long_masks, output_dir, name = "long_exp", title_prefix="Long Exposure: ")
    plot_masks(short_masks, output_dir, name = "short_exp", title_prefix="Short Exposure: ")


    # ============================================================
    # Step 4: Store individual feature masks
    # ============================================================

    short_sky_left = short_masks["Sky Fibres (Left)"]
    short_stars = short_masks["Star Fibres"]
    short_sky_right = short_masks["Sky Fibres (Right)"]

    long_sky_left = long_masks["Sky Fibres (Left)"]
    long_stars = long_masks["Star Fibres"]
    long_sky_right = long_masks["Sky Fibres (Right)"]

    masks = {
        "long_sky_left": long_sky_left,
        "long_stars": long_stars,
        "long_sky_right": long_sky_right,
    }

    region_mask = (masks["long_sky_left"] | masks["long_stars"] | masks["long_sky_right"])

    
    # Consolidate all masks and save
    all_masks_to_save = {
        "final_mask_common": final_mask,
        "long_sky_left": long_sky_left,
        "long_stars": long_stars,
        "long_sky_right": long_sky_right,
        "short_sky_left": short_sky_left,
        "short_stars": short_stars,
        "short_sky_right": short_sky_right,
        "region_mask": region_mask
    }

    if SAVE_MASK:
        save_masks_as_fits(all_masks_to_save, SAVE_DIR, prefix="")  
    
    
    return masks, region_mask



def normalise_fits(EXP_DIRS, output_dir, region_mask, valid_min, valid_max):

    for exp_dir in EXP_DIRS:

        images = images_dict[exp_dir]
        headers = headers_dict[exp_dir]

        # --- Per-exposure containers ---
        normalized_images_dict[exp_dir] = {}
        norm_factors_dict[exp_dir] = {}
        master_flats[exp_dir] = {}

        # Store UTMJD once per exposure (same order for all quadrants)
        utmjd_dict[exp_dir] = np.array([hdr.get("UTMJD", i) for i, hdr in enumerate(headers)])

        # ============================================================
        # KEY CHANGE: stack exposure once
        # images_stack shape = (N, fullH, fullW)
        # ============================================================
        images_stack = np.stack(images, axis=0).astype(float)

        # --- 2x2 figure ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle(f"Normalisation Factor vs UTMJD\nExposure: {float(exp_dir.replace('_', '.'))}s", fontsize=16)

        for idx, qname in enumerate(quadrants):
            ys, xs = quadrant_slices[qname]

            # ============================================================
            # Extract quadrant stack (vectorised)
            # quad_stack shape = (N, Hq, Wq)
            # ============================================================
            quad_stack = images_stack[:, ys, xs]

            # Apply region mask (vectorised)
            reg = region_mask[ys, xs].astype(bool)
            quad_stack = quad_stack * reg[None, :, :]

            # ============================================================
            # Build master flat from valid pixels only
            # ============================================================
            valid = (quad_stack > valid_min) & (quad_stack < valid_max) & (quad_stack != 0)
            masked_stack = np.where(valid, quad_stack, np.nan)

            master_flat = np.nanmedian(masked_stack, axis=0)
            master_flats[exp_dir][qname] = master_flat

            # ============================================================
            # Normalise each frame
            # Store as cube (N,H,W) NOT list
            # ============================================================
            N = quad_stack.shape[0]
            norm_images = np.empty_like(quad_stack, dtype=float)
            norm_factors = np.empty(N, dtype=float)

            for i in range(N):
                img = quad_stack[i]

                # Flat-field
                norm_img = img / master_flat

                # Scalar factor per frame
                factor = np.nanmedian(norm_img)
                norm_factors[i] = factor

                # Normalise image by its own factor (median ~1)
                if np.isfinite(factor) and factor != 0:
                    norm_images[i] = img / factor
                else:
                    norm_images[i] = np.nan

            # Store
            norm_factors_dict[exp_dir][qname] = norm_factors
            normalized_images_dict[exp_dir][qname] = norm_images

            # ============================================================
            # Store NF outputs for Excel (median + perfile)
            # ============================================================
            nf_median = np.nanmedian(norm_factors)

            if exp_dir not in results_nf:
                results_nf[exp_dir] = {}
            results_nf[exp_dir][qname] = nf_median

            # store utmjd for this exposure (once)
            results_utmjd[exp_dir] = utmjd_dict[exp_dir]

            # store per-file NFs
            if exp_dir not in results_nf_perfile:
                results_nf_perfile[exp_dir] = {}
            results_nf_perfile[exp_dir][qname] = norm_factors.tolist()

            # store summary NF (median)
            if exp_dir not in results_nf_summary:
                results_nf_summary[exp_dir] = {}
            results_nf_summary[exp_dir][qname] = nf_median

            # ============================================================
            # Plot
            # ============================================================
            utmjds = utmjd_dict[exp_dir]
            mask = np.isfinite(norm_factors) & np.isfinite(utmjds)

            utmjds_clean = utmjds[mask]
            norm_factors_clean = norm_factors[mask]

            ax = axes.flat[idx]

            if len(norm_factors_clean) >= 2:
                std_err = np.nanstd(norm_factors_clean) / np.sqrt(len(norm_factors_clean))
                popt, pcov = curve_fit(linear, utmjds_clean, norm_factors_clean)

                ax.plot(
                    utmjds_clean,
                    linear(utmjds_clean, *popt),
                    "-",
                    label=f"Best-fit: m={popt[0]:.2e}, c={popt[1]:.2e}"
                )
            else:
                std_err = None

            ax.errorbar(
                utmjds_clean,
                norm_factors_clean,
                yerr=std_err,
                fmt="o"
            )
            
            ax.grid(True)
            ax.set_xlabel("UTMJD")
            ax.set_ylabel("Normalisation Factor")
            ax.set_title(qname)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"Norm_Fact_{exp_dir}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def nonlinearity_uncertainty(params, cov, M_fit, n_samples=5000):

    M0 = M_fit[0]
    Mmax = M_fit[-1]

    # Report the NL evaluated at the best-fit parameters as the central value.
    G_best = G_poly_even(np.array([M0, Mmax]), params)
    G_initial_best, G_final_best = G_best
    nl_best = (G_initial_best - G_final_best) * 100 / G_initial_best

    # Propagate the full fitted-parameter covariance to NL by Monte Carlo.
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(params, cov, n_samples)

    nl_samples = []

    for p in samples:
        G = G_poly_even(np.array([M0, Mmax]), p)
        G_initial, G_final = G

        if np.isfinite(G_initial) and abs(G_initial) > 1e-12:
            nl = (G_initial - G_final) * 100 / G_initial
            nl_samples.append(nl)

    nl_samples = np.asarray(nl_samples)

    return nl_best, np.nanstd(nl_samples, ddof=1)
        
def var_vs_mean(region_mask, output_dir, valid_min, valid_max,
                mean_min_threshold, mean_max_threshold, s, fit_mode="binned"):

    results = {}
    filtered_data = {}

    # ============================================================
    # Helper: compute mean/variance maps once per quadrant
    # ============================================================
    def compute_means_vars_for_quadrant(qname):
        imgs = normalized_images_dict["1_0"][qname]


        # Value range mask
        imgs = np.where((imgs >= valid_min) & (imgs <= valid_max), imgs, np.nan)

        # Spatial region mask
        ys, xs = quadrant_slices[qname]
        reg = region_mask[ys, xs]  # (Hq, Wq)
        imgs = np.where(reg[None, :, :], imgs, np.nan)

        means = np.nanmean(imgs, axis=0)
        variances = np.nanvar(imgs, axis=0, ddof=1)

        return means, variances

    # ============================================================
    # 1) Compute and store results (means/vars + flattened)
    # ============================================================
    def process_quadrant_vm(qname):
        means, variances = compute_means_vars_for_quadrant(qname)

        mean_mask = (
            np.isfinite(means) &
            np.isfinite(variances) &
            (means >= mean_min_threshold) &
            (means <= mean_max_threshold)
        )

        means_flat = means[mean_mask].ravel()
        variances_flat = variances[mean_mask].ravel()

        return qname, {
            "means": means,
            "variances": variances,
            "means_flat": means_flat,
            "variances_flat": variances_flat
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        for qname, out in ex.map(process_quadrant_vm, quadrants):
            results[qname] = out


    # ============================================================
    # 2) Plot 1: raw scatter
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            qname = plot_order[i][j]
            ax = axes[i, j]

            means_flat = results[qname]["means_flat"]
            variances_flat = results[qname]["variances_flat"]

            ax.scatter(means_flat, variances_flat, s=1, alpha=0.3)
            ax.set_title(qname)
            ax.grid(True)

    fig.supxlabel("Mean Pixel Value")
    fig.supylabel("Variance of Pixel Value")
    fig.suptitle("Variance vs Mean", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "var_mean_raw.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # 3) Build filtered_data (only once)
    # ============================================================
    for qname in quadrants:
        means_flat = results[qname]["means_flat"]
        variances_flat = results[qname]["variances_flat"]

        # Convert variance -> stddev
        var_sqrt = np.sqrt(variances_flat)

        # Robust outlier removal
        median_var = np.median(var_sqrt)
        mad_var = np.median(np.abs(var_sqrt - median_var))
        sigma_var = 1.4826 * mad_var

        if sigma_var > 0 and np.isfinite(sigma_var):
            outlier_mask = np.abs(var_sqrt - median_var) <= 10 * sigma_var
        else:
            outlier_mask = np.ones_like(var_sqrt, dtype=bool)

        means_final = means_flat[outlier_mask]
        var_final = var_sqrt[outlier_mask]

        # Store variance squared (as you did)
        filtered_data[qname] = (means_final, var_final**2)

    # ============================================================
    # 4) Plot 2: hexbin of mean vs sqrt(variance)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            qname = plot_order[i][j]
            ax = axes[i, j]

            means_final, var_sq = filtered_data[qname]
            var_final = np.sqrt(var_sq)

            hb = ax.hexbin(means_final, var_final, gridsize=150, mincnt=1, bins="log")
            hb.set_rasterized(False)
            ax.set_title(qname)
            ax.grid(True)

    fig.supxlabel("Mean (ADU)")
    fig.supylabel(r'$\sqrt{Variance}$ (ADU)')
    fig.suptitle("Variance vs Mean", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "var_mean_hex.png"), dpi=150, bbox_inches='tight')
    plt.close()


    # ============================================================
    # 5) Fit model (PARALLEL) + store fit outputs
    # ============================================================

    def fit_one_quadrant_vm(q):
        x_filt, y_filt = filtered_data[q]

        mask = (x_filt > 0) & (y_filt > 0) & (x_filt < 62000)
        M = x_filt[mask]
        y = np.sqrt(y_filt[mask])

        if M.size < 20:
            return q, None

        M_bin, y_bin, scatter_bin, yerr_bin, n_bin = robust_equal_width_bins(
            M, y, n_bins=25, min_count=30, statistic="mode",
            mode_method="histogram"
        )
        if M_bin.size < 8:
            return q, None

        # Prevent a handful of enormous bins from claiming unrealistically
        # tiny errors when neighbouring pixels are spatially correlated.
        error_floor = 0.02 * np.median(scatter_bin)
        yerr_fit = np.maximum(yerr_bin, error_floor)

        if fit_mode == "binned":
            fit_M, fit_y, fit_sigma = M_bin, y_bin, yerr_fit
        elif fit_mode == "raw":
            n_frames = normalized_images_dict["1_0"][q].shape[0]
            # Approximate standard error of a sample standard deviation.
            raw_sigma = y / np.sqrt(2.0 * max(n_frames - 1, 1))
            positive = raw_sigma[np.isfinite(raw_sigma) & (raw_sigma > 0)]
            floor = np.percentile(positive, 5) if positive.size else 1.0
            fit_M, fit_y = M, y
            fit_sigma = np.maximum(raw_sigma, floor)
        else:
            raise ValueError("fit_mode must be 'binned' or 'raw'")

        params, errs, cov = fit_quadrant_even(fit_M, fit_y, fit_sigma, s)

        if not np.all(np.isfinite(params)) or cov is None:
            return q, None

        return q, dict(
            M=M, y=y, params=params, errs=errs, cov=cov,
            M_bin=M_bin, y_bin=y_bin, scatter_bin=scatter_bin,
            yerr_bin=yerr_bin, yerr_fit=yerr_fit, n_bin=n_bin
        )
        

    fit_outputs = {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for q, out in ex.map(fit_one_quadrant_vm, quadrants):
            fit_outputs[q] = out


    # ============================================================
    # 6) Plot + store results (SERIAL)
    # ============================================================

    def model_uncertainty(M_vals, params, cov, s, model_func):
        eps = 1e-6
        n_params = len(params)
        J = np.zeros((len(M_vals), n_params))

        for i in range(n_params):
            dp = np.zeros_like(params)
            dp[i] = eps
            y_plus = model_func(M_vals, params + dp, s)
            y_minus = model_func(M_vals, params - dp, s)
            J[:, i] = (y_plus - y_minus) / (2 * eps)

        var = np.einsum("ij,jk,ik->i", J, cov, J)
        return np.sqrt(np.maximum(var, 0))


    def gain_uncertainty(M_vals, params, cov):
        eps = 1e-6
        n_params = len(params)
        J = np.zeros((len(M_vals), n_params))

        for i in range(n_params):
            dp = np.zeros_like(params)
            dp[i] = eps
            g_plus = G_poly_even(M_vals, params + dp)
            g_minus = G_poly_even(M_vals, params - dp)
            J[:, i] = (g_plus - g_minus) / (2 * eps)

        var = np.einsum("ij,jk,ik->i", J, cov, J)
        return np.sqrt(np.maximum(var, 0))


    for q in quadrants:

        out = fit_outputs.get(q)

        if out is None:
            print(f"Skipping {q}: fit failed or not enough points")
            continue

        M = out["M"]
        y = out["y"]
        params = out["params"]
        errs = out["errs"]
        cov = out["cov"]
        M_bin = out["M_bin"]
        y_bin = out["y_bin"]
        scatter_bin = out["scatter_bin"]
        yerr_bin = out["yerr_bin"]
        yerr_fit = out["yerr_fit"]

        M_fit = np.linspace(M.min(), M.max(), 600)
        y_fit = y_model_even(M_fit, params, s)
        y_unc = model_uncertainty(M_fit, params, cov, s, y_model_even)
        
        # Save csv file for model var vs mean curve
        csv_file = os.path.join(output_dir, "fit_curves.csv")

        new_df = pd.DataFrame({
            f"M_fit_{q}": M_fit,
            f"y_fit_{q}": y_fit
        })

        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)

            # Append new columns instead of overwriting
            combined_df = pd.concat([existing_df, new_df], axis=1)

            # Remove duplicate columns if script is rerun for same quadrant
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        else:
            combined_df = new_df

        combined_df.to_csv(csv_file, index=False)
        
        
        raw_resid = y - y_model_even(M, params, s)

        # ---- Figure ----
        fig, axs = plt.subplots(2, 2, figsize=(14, 6), sharex=True)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # ========================================================
        # (a) RAW DATA
        # ========================================================
        hb = axs[0, 0].hexbin(M, y, gridsize=150, mincnt=1, bins='log')
        hb.set_rasterized(False)

        axs[0, 0].plot(
            M_fit, y_fit,
            'r', lw=2,
            label=(
                f"Fit to binned data"
            )
        )

        axs[0, 0].set_title(f"{q}", fontsize = 15)
        axs[0, 0].set_ylabel(r"$\sqrt{Variance}$ (ADU)", fontsize = 15)
        axs[0, 0].legend(fontsize = 15)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].tick_params(axis="both", which="major", labelsize=15)

        axs[0, 0].text(0.02, 0.95, "(a)", transform=axs[0, 0].transAxes,
                       va="top", ha="left", fontweight="bold")

        y_model_bin = y_model_even(M_bin, params, s)
        resid_bin = y_bin - y_model_bin
        # Plot the population scatter, not the much smaller uncertainty used
        # internally to weight the fitted bin centre.
        resid_yerr = scatter_bin

        valid = np.isfinite(yerr_fit) & (yerr_fit > 0)
        chi2 = np.sum(((y_bin[valid] - y_model_bin[valid]) / yerr_fit[valid])**2)
        dof = np.sum(valid) - len(params)
        chi2_red = chi2 / dof if dof > 0 else np.nan

        # ========================================================
        # (b) BINNED DATA
        # ========================================================
        axs[0, 1].errorbar(
            M_bin, y_bin,
            yerr=scatter_bin,
            fmt='o', capsize=3,
            label="Binned data"
        )

        axs[0, 1].plot(M_fit, y_fit, 'r', lw=2, label="Fit to binned data")

        #axs[1, 0].fill_between(
        #    M_fit,
        #    y_fit - y_unc,
        #    y_fit + y_unc,
        #    color='r',
        #    alpha=0.2,
        #    label="1σ model uncertainty"
        #)

        axs[0, 1].set_title("Binned Data", fontsize = 15)
        
        axs[0, 1].set_ylabel(r"$\sqrt{Variance}$ (ADU)", fontsize = 15)
        axs[0, 1].legend(fontsize = 15)
        axs[0, 1].grid(True, ls='--', alpha=0.5)
        axs[0, 1].tick_params(axis="both", which="major", labelsize=15)

        axs[0, 1].text(0.02, 0.95, "(b)", transform=axs[0, 1].transAxes,
                       va="top", ha="left", fontweight="bold")

        # ========================================================
        # (d) GAIN CURVE
        # ========================================================
        G_fit = G_poly_even(M_fit, params)
        G_unc = gain_uncertainty(M_fit, params, cov)

        # Non-linearity endpoints use the BINNED signal range (M_bin.min()/max()),
        # not the raw M.min()/max() used for the plotted M_fit curve. This matches
        # the convention used in exp_rat (bin_x.min()/max()), so the PTC and
        # exposure-ratio non-linearity numbers are evaluated over the same portion
        # of the dynamic range.
        M_lo, M_hi = M_bin.min(), M_bin.max()
        M_endpoints = np.array([M_lo, M_hi])
        G_initial, G_final = G_poly_even(M_endpoints, params)

        nonlinearity = (G_initial - G_final) * 100 / G_initial
        nl_mean, nl_err = nonlinearity_uncertainty(params, cov, M_endpoints) 

        results_vm[q] = {
            "g0": params[0],
            "g6": params[1],
            "g0_err": errs[0],
            "g6_err": errs[1],
            "vm_nl": nl_mean,
            "vm_nl_err": nl_err
        }

        axs[1, 1].plot(
            M_fit, G_fit,
            lw=2,
            label=(
                rf'$\mathrm{{NL}}_{{\max}}^{{\rm PTC}}'
                rf' = {nl_mean:.0f} \pm {nl_err:.0f}\%$'
            )
        )

        """axs[0, 1].fill_between(
            M_fit,
            G_fit - G_unc,
            G_fit + G_unc,
            alpha=0.2,
            label="1σ model uncertainty"
        )"""

        axs[1, 1].set_title(r'Non-linear Gain $G(\bar{x})$', fontsize = 15)
        axs[1, 1].set_ylabel(r'$G(\bar{x})$ (e$^{-}$/ADU)', fontsize = 15)
        axs[1, 1].set_xlabel("Mean (ADU)", fontsize = 15)
        axs[1, 1].legend(fontsize = 15)
        axs[1, 1].grid(True, ls='--', alpha=0.5)
        axs[1, 1].tick_params(axis="both", which="major", labelsize=15)

        axs[1, 1].text(0.02, 0.95, "(d)", transform=axs[1, 1].transAxes,
                       va="top", ha="left", fontweight="bold")

        # ========================================================
        # (c) RESIDUALS
        # ========================================================
        axs[1, 0].errorbar(
            M_bin,
            resid_bin,
            yerr=resid_yerr,
            fmt='o',
            capsize=3
        )

        axs[1, 0].axhline(0, color='r', ls='--')
        axs[1, 0].set_title(f"Binned Data Residuals", fontsize = 15) #  (χ²ᵣ = {chi2_red:.2f})")
        axs[1, 0].set_xlabel("Mean (ADU)", fontsize = 15)
        axs[1, 0].set_ylabel("Residuals (ADU)", fontsize = 15)
        axs[1, 0].grid(True, ls='--', alpha=0.5)
        axs[1, 0].tick_params(axis="both", which="major", labelsize=15)

        axs[1, 0].text(0.02, 0.95, "(c)", transform=axs[1, 0].transAxes,
                       va="top", ha="left", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"var_mean_best_fit_{q}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()



                   
def exp_rat(output_dir, mean_min_threshold, mean_max_threshold, region_mask,
            fit_mode="binned"):

    # Containers
    results_rat_init = {}
    results_rat = {}

    # ============================================================
    # 1) Per-quadrant ratio extraction + gaussian clipping
    # ============================================================
    def process_quadrant_ratio(qname):

        imgs_10 = normalized_images_dict["1_0"][qname]
        median_10 = np.nanmedian(imgs_10, axis=0)
        mad_10 = 1.4826 * np.nanmedian(
            np.abs(imgs_10 - median_10[None, :, :]), axis=0
        )
        n_10 = np.sum(np.isfinite(imgs_10), axis=0)
        sem_10 = 1.2533 * mad_10 / np.sqrt(np.maximum(n_10, 1))

        imgs_01 = normalized_images_dict["0_1"][qname]
        median_01 = np.nanmedian(imgs_01, axis=0)
        mad_01 = 1.4826 * np.nanmedian(
            np.abs(imgs_01 - median_01[None, :, :]), axis=0
        )
        n_01 = np.sum(np.isfinite(imgs_01), axis=0)
        sem_01 = 1.2533 * mad_01 / np.sqrt(np.maximum(n_01, 1))

        ys, xs = quadrant_slices[qname]
        region_mask_q = region_mask[ys, xs]

        value_mask = (median_10 > mean_min_threshold) & (median_10 <= mean_max_threshold)
        combined_mask = value_mask & region_mask_q

        median_10_masked = median_10[combined_mask]
        median_01_masked = median_01[combined_mask]
        sem_10_masked = sem_10[combined_mask]
        sem_01_masked = sem_01[combined_mask]

        good = (
            np.isfinite(median_10_masked) &
            np.isfinite(median_01_masked) &
            (median_01_masked != 0)
        )

        median_10_masked = median_10_masked[good]
        median_01_masked = median_01_masked[good]
        sem_10_masked = sem_10_masked[good]
        sem_01_masked = sem_01_masked[good]

        if median_10_masked.size < 10:
            return qname, dict(
                median_10_masked=np.array([]),
                ratio_i=np.array([]),
                x=np.array([]),
                ratio=np.array([]),
                ratio_err=np.array([])
            )

        ratio_i = median_10_masked / median_01_masked
        ratio_err_i = np.abs(ratio_i) * np.sqrt(
            (sem_10_masked / median_10_masked)**2 +
            (sem_01_masked / median_01_masked)**2
        )
        x, ratio, clip_mask = gaussian_clip(
            median_10_masked, ratio_i, return_mask=True
        )
        ratio_err = ratio_err_i[clip_mask]

        return qname, dict(
            median_10_masked=median_10_masked,
            ratio_i=ratio_i,
            x=x,
            ratio=ratio,
            ratio_err=ratio_err
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        for qname, out in ex.map(process_quadrant_ratio, quadrants):
            results_rat_init[qname] = out

   
    # ============================================================
    # 2) Plot 1: median_10_masked vs ratio_i
    # ============================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            qname = plot_order[i][j]
            ax = axes1[i, j]
            data = results_rat_init[qname]

            if data["median_10_masked"].size == 0:
                ax.set_title(f"{qname} (no data)")
                ax.axis("off")
                continue

            hb = ax.hexbin(
                data["median_10_masked"],
                data["ratio_i"],
                gridsize=150,
                mincnt=1,
                bins="log"
            )
            hb.set_rasterized(False)
            ax.set_title(qname)
            ax.set_xlabel("Median Pixel Value in 1.0s Exposure")
            ax.set_ylabel("Ratio")
            ax.grid(True)

    fig1.suptitle(f"Ratio of Median Images (1.0 / 0.1)", fontsize=14)
    plt.savefig(os.path.join(output_dir, "ratio_raw.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
        
    # ============================================================
    # 3) Plot 2: clipped x vs clipped ratio
    # ============================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            qname = plot_order[i][j]
            ax = axes2[i, j]
            data = results_rat_init[qname]

            if data["x"].size == 0:
                ax.set_title(f"{qname} (no data)")
                ax.axis("off")
                continue

            hb = ax.hexbin(
                data["x"],
                data["ratio"],
                gridsize=150,
                mincnt=1,
                bins="log"
            )
            hb.set_rasterized(False)
            ax.set_title(qname)
            ax.set_xlabel("Median Pixel Value in 1.0s Exposure")
            ax.set_ylabel("Ratio")
            ax.grid(True)

    fig2.suptitle(f"Clipped Ratio of Median Images (1.0 / 0.1)", fontsize=14)
    plt.savefig(os.path.join(output_dir, "ratio_clipped.png"), dpi=150, bbox_inches='tight')
    plt.close()
    

    # ============================================================
    # 4) Fit model per quadrant (same as your original)
    # ============================================================
    def fit_quadrant_er(qname):
        xg = np.asarray(results_rat_init[qname]["x"])
        rg = np.asarray(results_rat_init[qname]["ratio"])
        raw_err = np.asarray(results_rat_init[qname]["ratio_err"])

        mask = np.isfinite(xg) & np.isfinite(rg) & np.isfinite(raw_err) & (raw_err > 0)
        xg, rg, raw_err = xg[mask], rg[mask], raw_err[mask]

        if xg.size < 20:
            return qname, None

        if qname in ["Q1", "Q2"]:
            norm_mask = (xg > 10000) & (xg < 25000)
        else:
            norm_mask = xg < 25000

        rv = rg[norm_mask]
        rv = rv[np.isfinite(rv)]

        if rv.size < 5:
            return qname, None

        kde = gaussian_kde(rv)
        grid = np.linspace(rv.min(), rv.max(), 1000)
        norm_factor = grid[np.argmax(kde(grid))]

        if not np.isfinite(norm_factor) or norm_factor == 0:
            return qname, None

        r_norm = rg / norm_factor

        bin_x, bin_y, bin_scatter, bin_sem, bin_n = robust_equal_width_bins(
            xg, r_norm, n_bins=25, min_count=30, statistic="mode",
            mode_method="kde"
        )
        if bin_x.size < 8:
            return qname, None

        # Fit the same robust bin centres that are plotted.  The model remains
        # anchored at R=1 and contains only one sixth-order coefficient.
        x_lo, x_hi = bin_x.min(), bin_x.max()
        error_floor = 0.02 * np.median(bin_scatter)
        fit_err = np.maximum(bin_sem, error_floor)
        if fit_mode == "binned":
            fit_x, fit_y, fit_sigma = bin_x, bin_y, fit_err
        elif fit_mode == "raw":
            positive = raw_err[np.isfinite(raw_err) & (raw_err > 0)]
            floor = np.percentile(positive, 5) if positive.size else 1e-3
            fit_x, fit_y = xg, r_norm
            fit_sigma = np.maximum(raw_err / norm_factor, floor / norm_factor)
        else:
            raise ValueError("fit_mode must be 'binned' or 'raw'")

        popt, pcov = curve_fit(
            model_ratio_x6, fit_x, fit_y,
            sigma=fit_sigma, absolute_sigma=True,
            p0=[0.0],
            maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))

        model_bin = model_ratio_x6(bin_x, *popt)
        resid_mean = bin_y - model_bin
     

        return qname, dict(
            x=xg,
            r=r_norm,
            r_raw=rg,
            raw_err=raw_err,
            norm_factor=norm_factor,
            qname=qname,
            fit_mode=fit_mode,
            popt=popt,
            perr=perr,
            bin_x=bin_x,
            bin_y=bin_y,
            bin_err=bin_scatter,
            bin_fit_err=fit_err,
            resid_y=resid_mean,
            resid_err=bin_scatter,
            pcov=pcov,
            x_lo=x_lo,
            x_hi=x_hi
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        for qname, out in ex.map(fit_quadrant_er, quadrants):
            if out is not None:
                results_rat[qname] = out

    def eval_ratio_fit(d, x):
        return model_ratio_x6(x, d["popt"][0])

    def ratio_nonlinearity(d, n_samples=200, max_bootstrap_points=1000):
        # Central value remains the NL of the original best-fit model.
        lo = eval_ratio_fit(d, np.array([d["x_lo"]]))[0]
        hi = eval_ratio_fit(d, np.array([d["x_hi"]]))[0]
        nl = 100.0 * (lo - hi) / lo

        # Bootstrap the complete exposure-ratio fitting path used to determine
        # the NL uncertainty.  Each realisation re-estimates the low-signal KDE
        # normalisation, the KDE mode in every signal bin, and the a6 fit.
        # max_bootstrap_points limits only the bootstrap resample size used for
        # each KDE so this remains practical for very large CCD-pixel samples.
        rng = np.random.default_rng(42)

        xg = np.asarray(d["x"], dtype=float)
        rg = np.asarray(d["r_raw"], dtype=float)
        raw_err = np.asarray(d["raw_err"], dtype=float)

        if d["qname"] in ["Q1", "Q2"]:
            norm_mask = (xg > 10000) & (xg < 25000)
        else:
            norm_mask = xg < 25000

        norm_pool = rg[norm_mask]
        norm_pool = norm_pool[np.isfinite(norm_pool)]

        if norm_pool.size < 5:
            return nl, np.nan

        edges = np.linspace(xg.min(), xg.max(), 26)
        vals_nl = []

        for _ in range(n_samples):
            # ------------------------------------------------------------
            # 1) Re-estimate the low-signal normalisation factor
            # ------------------------------------------------------------
            n_norm = min(norm_pool.size, max_bootstrap_points)
            norm_sample = rng.choice(norm_pool, size=n_norm, replace=True)

            if np.nanmax(norm_sample) <= np.nanmin(norm_sample):
                continue

            try:
                kde = gaussian_kde(norm_sample)
                grid = np.linspace(norm_sample.min(), norm_sample.max(), 200)
                norm_factor_boot = grid[np.argmax(kde(grid))]
            except (ValueError, np.linalg.LinAlgError):
                continue

            if not np.isfinite(norm_factor_boot) or norm_factor_boot == 0:
                continue

            # ------------------------------------------------------------
            # 2) Re-estimate the KDE mode in every signal bin
            # ------------------------------------------------------------
            rows = []

            for i in range(len(edges) - 1):
                use = (
                    (xg >= edges[i]) &
                    (xg <= edges[i + 1] if i == len(edges) - 2 else xg < edges[i + 1])
                )

                vals = rg[use] / norm_factor_boot
                vals = vals[np.isfinite(vals)]

                if vals.size < 30:
                    continue

                n_draw = min(vals.size, max_bootstrap_points)
                vals_boot = rng.choice(vals, size=n_draw, replace=True)

                if np.nanmax(vals_boot) <= np.nanmin(vals_boot):
                    continue

                try:
                    kde_bin = gaussian_kde(vals_boot)
                    grid_bin = np.linspace(vals_boot.min(), vals_boot.max(), 200)
                    centre = grid_bin[np.argmax(kde_bin(grid_bin))]
                except (ValueError, np.linalg.LinAlgError):
                    continue

                scatter = np.std(vals_boot, ddof=1)
                n_effective = min(vals.size, 1000)
                error = scatter / np.sqrt(n_effective)
                x_centre = 0.5 * (edges[i] + edges[i + 1])
                rows.append((x_centre, centre, scatter, error))

            if len(rows) < 8:
                continue

            boot_x = np.asarray([r[0] for r in rows])
            boot_y = np.asarray([r[1] for r in rows])
            boot_scatter = np.asarray([r[2] for r in rows])
            boot_sem = np.asarray([r[3] for r in rows])

            error_floor = 0.02 * np.median(boot_scatter)
            boot_fit_err = np.maximum(boot_sem, error_floor)

            # ------------------------------------------------------------
            # 3) Refit a6 using the same fitting mode as the main analysis
            # ------------------------------------------------------------
            try:
                if d["fit_mode"] == "binned":
                    popt_boot, _ = curve_fit(
                        model_ratio_x6,
                        boot_x,
                        boot_y,
                        sigma=boot_fit_err,
                        absolute_sigma=True,
                        p0=[d["popt"][0]],
                        maxfev=20000
                    )
                else:
                    n_raw = min(xg.size, max_bootstrap_points * 10)
                    idx = rng.integers(0, xg.size, size=n_raw)
                    xb = xg[idx]
                    rb = rg[idx] / norm_factor_boot
                    eb = raw_err[idx] / norm_factor_boot
                    good = np.isfinite(xb) & np.isfinite(rb) & np.isfinite(eb) & (eb > 0)
                    xb, rb, eb = xb[good], rb[good], eb[good]

                    if xb.size < 20:
                        continue

                    positive = eb[np.isfinite(eb) & (eb > 0)]
                    floor = np.percentile(positive, 5) if positive.size else 1e-3
                    sigma_boot = np.maximum(eb, floor)

                    popt_boot, _ = curve_fit(
                        model_ratio_x6,
                        xb,
                        rb,
                        sigma=sigma_boot,
                        absolute_sigma=True,
                        p0=[d["popt"][0]],
                        maxfev=20000
                    )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

            # Keep the NL definition at the same original signal endpoints.
            y0 = model_ratio_x6(d["x_lo"], popt_boot[0])
            y1 = model_ratio_x6(d["x_hi"], popt_boot[0])

            if np.isfinite(y0) and np.isfinite(y1) and y0 != 0:
                vals_nl.append(100.0 * (y0 - y1) / y0)

        vals_nl = np.asarray(vals_nl)

        if vals_nl.size < 2:
            return nl, np.nan

        return nl, np.std(vals_nl, ddof=1)

                
    # ==========================================================
    # Save best-fit model for all quadrants
    # ==========================================================
    csv_rows = []

    for q in quadrants:
        if q not in results_rat:
            continue

        d = results_rat[q]

        x = d["x"]
        model_ratio = eval_ratio_fit(d, x)

        csv_rows.append(pd.DataFrame({
            "Quadrant": q,
            "Pixel_Value_Long_Exposure": x,
            "Best_Fit_Ratio": model_ratio
        }))

    if csv_rows:
        csv_df = pd.concat(csv_rows, ignore_index=True)
        csv_df = csv_df.sort_values(["Quadrant", "Pixel_Value_Long_Exposure"])
        csv_df.to_csv(
            os.path.join(output_dir, "ratio_best_fit_models.csv"),
            index=False
        )
    

    # ==========================================================
    # 5) Plot: Raw Data + Model
    # ==========================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            q = plot_order[i][j]
            ax = axes[i, j]

            if q not in results_rat:
                ax.set_title(f"{q} (skipped)")
                ax.axis("off")
                continue

            d = results_rat[q]
            hb = ax.hexbin(d["x"], d["r"], gridsize=150, bins="log", mincnt=1)
            hb.set_rasterized(False)

            xx = np.linspace(d["x"].min(), d["x"].max(), 500)
            ax.plot(xx, eval_ratio_fit(d, xx), "b", lw=2)

            ax.set_title(q)
            ax.grid(True)

    fig.supxlabel("Pixel Value in Long Exposure (ADU)")
    fig.supylabel("Normalized Ratio")
    fig.suptitle(f"Density Plot of the Ratios & Fit to binned data", fontsize=14)
    plt.savefig(os.path.join(output_dir, "ratio_hex_best_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================
    # 6) Plot: Binned Data + Model
    # ==========================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            q = plot_order[i][j]
            ax = axes[i, j]

            if q not in results_rat:
                ax.set_title(f"{q} (skipped)")
                ax.axis("off")
                continue

            d = results_rat[q]

            ax.errorbar(d["bin_x"], d["bin_y"], yerr=d["bin_err"],
                        fmt="o", capsize=3, color="black")

            xx = np.linspace(d["x"].min(), d["x"].max(), 500)
            yy = eval_ratio_fit(d, xx)
            nl, nl_err = ratio_nonlinearity(d)

            ax.plot(xx, yy, "b", lw=2,
                    label=rf"$\mathrm{{NL}}_{{\max}}^{{\rm exp}} = {nl:.2f} \pm {nl_err:.2f}\%$")

            ax.legend()
            ax.set_title(q)
            ax.grid(True)

    fig.supxlabel("Pixel Value in Long Exposure (ADU)")
    fig.supylabel("Normalized Ratio")
    fig.suptitle(f"Fit to binned data", fontsize=14)
    plt.savefig(os.path.join(output_dir, "ratio_model_best_fit.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================================
    # 7) Plot: Residuals
    # ==========================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    for i in range(2):
        for j in range(2):
            q = plot_order[i][j]
            ax = axes[i, j]

            if q not in results_rat:
                ax.set_title(f"{q} (skipped)")
                ax.axis("off")
                continue

            d = results_rat[q]
            ax.errorbar(d["bin_x"], d["resid_y"], yerr=d["resid_err"],
                        fmt="o", capsize=3, color="blue")

            ax.axhline(0, color="black", ls="--")
            ax.set_title(q)
            ax.grid(True)

    fig.supxlabel("Pixel Value in Long Exposure (ADU)")
    fig.supylabel("Residual (Data − Model)")
    fig.suptitle(f"Binned Data Residuals", fontsize=14)
    plt.savefig(os.path.join(output_dir, "ratio_residual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    
    for q in plot_order[0] + plot_order[1]:  # flatten the 2x2 list
        if q not in results_rat:
            print(f"{q} skipped")
            continue

        d = results_rat[q]

        # Compute reduced chi-squared for binned data
        #yy_full = model_x6(d["x"], d["popt"][0])
        #chi2 = np.sum((d["r"] - yy_full)**2 / sigma_pixel**2)
        #dof = len(d["bin_y"]) - len(d["popt"])  # N - p
        #chi2_red = chi2 / dof if dof > 0 else np.nan

        TICK_SIZE = 15
        # Create figure with GridSpec: equal-height top & bottom rows
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.4, wspace=0.3)

        # Top row: raw data + model (span both columns)
        ax_hex = fig.add_subplot(gs[0, :])
        hb = ax_hex.hexbin(d["x"], d["r"], gridsize=300, bins="log", mincnt=1)
        hb.set_rasterized(False)
        xx = np.linspace(d["x"].min(), d["x"].max(), 500)
        ax_hex.plot(xx, eval_ratio_fit(d, xx), "b", lw=2)
        ax_hex.set_title(f"{q}: Density Plot of the Ratios & Fit to Binned Data", fontsize = 15)
        ax_hex.set_xlabel("Pixel Value in Long Exposure (ADU)", fontsize = 15)
        ax_hex.set_ylabel("Ratio", fontsize = 15)
        ax_hex.grid(True)
        ax_hex.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        # Bottom-left: binned data + model
        ax_bin = fig.add_subplot(gs[1, 0])
        ax_bin.errorbar(d["bin_x"], d["bin_y"], yerr=d["bin_err"],
                        fmt="o", capsize=3, color="black")
        yy = eval_ratio_fit(d, xx)
        nl, nl_err = ratio_nonlinearity(d)
        
        
        
        results_er[q] = {
            "a6": d["popt"][0],
            "a6_err": d["perr"][0],
            "er_nl": nl,
            "er_nl_err": nl_err
        }
        
        ax_bin.plot(xx, yy, "b", lw=2, label=rf"$\mathrm{{NL}}_{{\max}}^{{\rm exp}} = {nl:.1f} \pm {nl_err:.1f}\%$")
        ax_bin.legend(fontsize = 15)
        ax_bin.set_title("Fit to Binned Data", fontsize = 15)
        ax_bin.set_xlabel("Pixel Value in Long Exposure (ADU)", fontsize = 15)
        ax_bin.set_ylabel("Ratio (Binned)", fontsize = 15)
        ax_bin.grid(True)
        ax_bin.tick_params(axis="both", which="major", labelsize=TICK_SIZE)


        # Bottom-right: residuals
        ax_res = fig.add_subplot(gs[1, 1])
        ax_res.errorbar(d["bin_x"], d["resid_y"], yerr=d["resid_err"],
                        fmt="o", capsize=3, color="blue")
        ax_res.axhline(0, color="black", ls="--")
        ax_res.set_title(f"Binned Data Residuals", fontsize = 15) # (χ²ᵣ = {chi2_red:.2f})")
        ax_res.set_xlabel("Pixel Value in Long Exposure (ADU)", fontsize = 15)
        ax_res.set_ylabel("Residual", fontsize = 15)
        ax_res.grid(True)
        ax_res.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

        # Save figure
        plt.savefig(os.path.join(output_dir, f"ratio_overview_{q}.png"), dpi=150, bbox_inches='tight')
        plt.close()


        
def format_worksheet(ws, col_width=18, font_size=14,
                     sci_fmt="0.000E+00", normal_fmt="0.000"):
    #ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    base_font = Font(size=font_size)
    align = Alignment(horizontal="center", vertical="center", wrap_text=False)

    # get header names
    headers = {col_idx: ws.cell(row=1, column=col_idx).value
               for col_idx in range(1, ws.max_column + 1)}

    # font + alignment for all cells
    for row in ws.iter_rows():
        for cell in row:
            cell.font = base_font
            cell.alignment = align

    # apply number formats based on column
    for col_idx, header in headers.items():
        header_lower = str(header).lower()
        for r in range(2, ws.max_row + 1):
            cell = ws.cell(row=r, column=col_idx)
            val = cell.value
            
            if val is None:
                continue
                
            if header == "Exposure":
                continue
                
            if header in ["Folder Name", "File_Number"]:
                if isinstance(val, (int, float)):
                    cell.number_format = "0"
                continue
                
            if header == "UTMJD" or header.startswith("UTMJD_"):
                if isinstance(val, (int, float)):
                    cell.number_format = normal_fmt
                continue
                
            if header.endswith("%"):
                if isinstance(val, (int, float)):
                    cell.number_format = "0.00"
                continue
                
            if isinstance(val, (int, float)):
                cell.number_format = sci_fmt

    # same column width
    for col_idx in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width


def update_excel_two_sheets(excel_path, folder_name, amp_mode):
    excel_path = Path(excel_path)

    # Always treat folder_name as a clean string
    folder_name = str(folder_name).strip()

    # ==========================================================
    # Build SUMMARY row (1 row per folder)
    # ==========================================================
    def utmjd_minmax(exp):
        arr = np.asarray(results_utmjd.get(exp, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return (np.nan, np.nan)
        return (float(np.min(arr)), float(np.max(arr)))

    ut01_min, ut01_max = utmjd_minmax("0_1")
    ut10_min, ut10_max = utmjd_minmax("1_0")

    summary_row = {
        "Folder Name": folder_name,
        "AMP_MODE": amp_mode,
        "UTMJD_01_min": ut01_min,
        "UTMJD_01_max": ut01_max,
        "UTMJD_10_min": ut10_min,
        "UTMJD_10_max": ut10_max,
    }

    for q in quadrants:
        d = results_vm.get(q, {})
        summary_row[f"{q}_g0"] = d.get("g0", np.nan)
        summary_row[f"{q}_g0_err"] = d.get("g0_err", np.nan)
        summary_row[f"{q}_g6"] = d.get("g6", np.nan)
        summary_row[f"{q}_g6_err"] = d.get("g6_err", np.nan)

    for q in quadrants:
        d = results_vm.get(q, {})
        summary_row[f"VM_{q}_NL_%"] = d.get("vm_nl", np.nan)
        summary_row[f"VM_{q}_NL_ERR%"] = d.get("vm_nl_err", np.nan)

    for q in quadrants:
        d = results_er.get(q, {})
        summary_row[f"{q}_a6"] = d.get("a6", np.nan)
        summary_row[f"{q}_a6_err"] = d.get("a6_err", np.nan)

    for q in quadrants:
        d = results_er.get(q, {})
        summary_row[f"ER_{q}_NL_%"] = d.get("er_nl", np.nan)
        summary_row[f"ER_{q}_NL_ERR_%"] = d.get("er_nl_err", np.nan)

        
    summary_df_new = pd.DataFrame([summary_row])

    # ==========================================================
    # Build PER-FILE NF table
    # ==========================================================
    perfile_rows = []
    for exp in ["0_1", "1_0"]:
        utmjds = np.asarray(results_utmjd.get(exp, []), dtype=float)
        if utmjds.size == 0:
            continue

        n_files = utmjds.size

        for i in range(n_files):
            row = {
                "Folder Name": folder_name,
                "Exposure": str(exp).strip(),
                "File_Number": int(i),
                "UTMJD": float(utmjds[i]) if np.isfinite(utmjds[i]) else np.nan,
            }

            for q in quadrants:
                nf_arr = np.asarray(results_nf_perfile.get(exp, {}).get(q, []), dtype=float)
                if nf_arr.size == n_files and np.isfinite(nf_arr[i]):
                    row[f"NF_{q}"] = float(nf_arr[i])
                else:
                    row[f"NF_{q}"] = np.nan

            perfile_rows.append(row)

    perfile_df_new = pd.DataFrame(perfile_rows)

    # Force correct types (important for matching!)
    if not perfile_df_new.empty:
        perfile_df_new["Folder Name"] = perfile_df_new["Folder Name"].astype(str).str.strip()
        perfile_df_new["Exposure"] = perfile_df_new["Exposure"].astype(str).str.strip()
        perfile_df_new["File_Number"] = pd.to_numeric(perfile_df_new["File_Number"], errors="coerce").astype("Int64")

    # ==========================================================
    # Read existing Excel (if exists and valid)
    # ==========================================================
    if excel_path.exists():
        try:
            with pd.ExcelFile(excel_path) as xls:
                summary_old = (
                    pd.read_excel(xls, sheet_name="Summary")
                    if "Summary" in xls.sheet_names
                    else pd.DataFrame()
                )
                perfile_old = (
                    pd.read_excel(xls, sheet_name="PerFile_NF")
                    if "PerFile_NF" in xls.sheet_names
                    else pd.DataFrame()
                )
        except zipfile.BadZipFile:
            print(f"WARNING: {excel_path} is corrupted. Rebuilding from scratch.")
            summary_old = pd.DataFrame()
            perfile_old = pd.DataFrame()
    else:
        summary_old = pd.DataFrame()
        perfile_old = pd.DataFrame()

    # Normalise key columns in old sheets (CRITICAL)
    if not summary_old.empty and "Folder Name" in summary_old.columns:
        summary_old["Folder Name"] = summary_old["Folder Name"].astype(str).str.strip()

    if not perfile_old.empty:
        if "Folder Name" in perfile_old.columns:
            perfile_old["Folder Name"] = perfile_old["Folder Name"].astype(str).str.strip()
        if "Exposure" in perfile_old.columns:
            perfile_old["Exposure"] = perfile_old["Exposure"].astype(str).str.strip()
        if "File_Number" in perfile_old.columns:
            perfile_old["File_Number"] = pd.to_numeric(perfile_old["File_Number"], errors="coerce").astype("Int64")

    # ==========================================================
    # Update SUMMARY
    # ==========================================================
    if summary_old.empty:
        summary_out = summary_df_new
    else:
        if "Folder Name" not in summary_old.columns:
            raise ValueError("Summary sheet exists but has no 'Folder Name' column.")

        if (summary_old["Folder Name"] == folder_name).any():
            idx = summary_old.index[summary_old["Folder Name"] == folder_name][0]
            for col in summary_df_new.columns:
                summary_old.loc[idx, col] = summary_df_new.loc[0, col]
            summary_out = summary_old
        else:
            summary_out = pd.concat([summary_old, summary_df_new], ignore_index=True)

    # ==========================================================
    # Update PER-FILE
    # ==========================================================
    if perfile_old.empty:
        perfile_out = perfile_df_new
    else:
        required = {"Folder Name", "Exposure", "File_Number"}
        if not required.issubset(perfile_old.columns):
            raise ValueError("PerFile_NF sheet exists but is missing key columns.")

        perfile_out = perfile_old.copy()

        for _, newrow in perfile_df_new.iterrows():
            mask = (
                (perfile_out["Folder Name"] == newrow["Folder Name"]) &
                (perfile_out["Exposure"] == newrow["Exposure"]) &
                (perfile_out["File_Number"] == newrow["File_Number"])
            )

            if mask.any():
                idx = perfile_out.index[mask][0]
                for col in perfile_df_new.columns:
                    perfile_out.loc[idx, col] = newrow[col]
            else:
                perfile_out = pd.concat([perfile_out, pd.DataFrame([newrow])], ignore_index=True)

    # ==========================================================
    # Sort (numerically by folder name)
    # ==========================================================
    # Folder names are numeric-like (e.g. 230417), but keep them as strings for matching.
    def folder_sort_key(series):
        return pd.to_numeric(series, errors="coerce")

    summary_out["Folder Name"] = summary_out["Folder Name"].astype(str).str.strip()
    perfile_out["Folder Name"] = perfile_out["Folder Name"].astype(str).str.strip()

    summary_out = summary_out.sort_values(
        by="Folder Name",
        key=folder_sort_key
    ).reset_index(drop=True)

    perfile_out = perfile_out.sort_values(
        by=["Folder Name", "Exposure", "File_Number"],
        key=lambda s: folder_sort_key(s) if s.name == "Folder Name" else s
    ).reset_index(drop=True)

    # ==========================================================
    # Save Excel with formatting
    # ==========================================================
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        summary_out.to_excel(writer, sheet_name="Summary", index=False)
        perfile_out.to_excel(writer, sheet_name="PerFile_NF", index=False)

        wb = writer.book
        format_worksheet(wb["Summary"], col_width=18, font_size=14)
        format_worksheet(wb["PerFile_NF"], col_width=18, font_size=14)

    print(f"Excel updated: {excel_path}")

        
        
def process_data(BASE_DIR, SAVE_MASK, default_mask, s=3.5, mean_min_threshold=5000,
                 mean_max_threshold=62000, fit_mode="binned"):

    EXP_DIRS = ["0_1", "1_0"]
    valid_min, valid_max = 1000, 64000
    output_dir = BASE_DIR + str("/Output_Figs_vtest")
    os.makedirs(output_dir, exist_ok=True)

    # Read input files
    for exp_dir in EXP_DIRS:
        dir_path = os.path.join(BASE_DIR, exp_dir)
        images, headers = load_fits_images(dir_path)
        images_dict[exp_dir] = images
        headers_dict[exp_dir] = headers
    
    # Detect AMP mode from FITS headers
    amp_mode = detect_amp_mode(headers_dict)
    
    # Generate masks
    masks, region_mask = get_masks(SAVE_MASK, output_dir, default_mask)

    # Run analysis functions (all should now just display plots)
    normalise_fits(EXP_DIRS, output_dir, region_mask, valid_min, valid_max)
    
    var_vs_mean(region_mask, output_dir, valid_min, valid_max,
                mean_min_threshold, mean_max_threshold, s, fit_mode=fit_mode)
    
    exp_rat(output_dir, mean_min_threshold, mean_max_threshold, region_mask,
            fit_mode=fit_mode)

    # Update Excel summary
    folder_name = os.path.basename(os.path.normpath(BASE_DIR))
    excel_path = "results_summary_test.xlsx"
    update_excel_two_sheets(
        excel_path,
        folder_name=folder_name,
        amp_mode=amp_mode
    )
    
    
def read_folder_list(txt_path):
    folders = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            folders.append(line)
    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Run process_data() on a list of folders from a .txt file"
    )
    parser.add_argument("txt_file", help="Path to a .txt file containing folder paths")
    parser.add_argument("--save_mask", action="store_true", default=False)
    parser.add_argument("--dm", action="store_true", default=False)
    parser.add_argument("--s", type=float, default=4.0)
    parser.add_argument("--mean_min", type=float, default=5000)
    parser.add_argument("--mean_max", type=float, default=62000)
    parser.add_argument(
        "--fit_mode", choices=("binned", "raw"), default="binned",
        help="Fit robust binned centres (recommended) or all weighted pixels"
    )
    
    

    args = parser.parse_args()

    folders = read_folder_list(args.txt_file)

    if not folders:
        print("No folders found in the txt file.")
        return

    print(f"Found {len(folders)} folders.\n")

    for i, base_dir in enumerate(folders, start=1):
        print("=" * 70)
        print(f"[{i}/{len(folders)}] Processing: {base_dir}")

        if not os.path.isdir(base_dir):
            print(f"  SKIP: Not a valid directory: {base_dir}")
            continue

        try:
            process_data(
                BASE_DIR=base_dir,
                SAVE_MASK=args.save_mask,
                default_mask=args.dm,
                s=args.s,
                mean_min_threshold=args.mean_min,
                mean_max_threshold=args.mean_max,
                fit_mode=args.fit_mode,
            )
            print("  DONE.")
        except Exception:
            print("  ERROR occurred in this folder:")
            traceback.print_exc()
            print("  Continuing to next folder...")

    print("\nAll processing finished.")


if __name__ == "__main__":
    main()
