#!/usr/bin/env python3
"""
fast_receive.py

Vectorized photoreceptor initialization + per-frame processing.

Usage:
  # Create photoreceptor table (one-time, faster than object-based)
  python fast_receive.py --init --surface_radius 1248 --hex_size 1.0 --out photoreceptors.npz

  # Process one image with the saved photoreceptor table
  python fast_receive.py --process --image path/to/img.jpg --photofile photoreceptors.npz \
        --out_csv out.csv --out_png out.png
"""

import argparse
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# ---------------------------
# Utilities & LUTs (vector)
# ---------------------------
def build_spectral_lut(wmin=380.0, wmax=700.0, n=321):
    w = np.linspace(wmin, wmax, n).astype(np.float32)
    k = 69.7
    # Govardovskii-like nomogram (vectorized)
    gv = np.exp(3.21 * np.log(k / w) - 0.485 * (np.log(k / w)) ** 2 + 9.71e-3 * (np.log(k / w)) ** 3)
    gv = (gv / gv.max()).astype(np.float32)
    return w, gv


W_LUT, SENS_LUT = build_spectral_lut()


def spectral_weight_from_wavelength(wavelengths):
    # wavelengths: np.array (float32)
    return np.interp(wavelengths, W_LUT, SENS_LUT, left=0.0, right=0.0).astype(np.float32)


# HSP brightness vectorized
def hsp_brightness_vec(R, G, B):
    return np.sqrt(0.299 * (R ** 2) + 0.587 * (G ** 2) + 0.114 * (B ** 2)).astype(np.float32)


def rgb_to_luminance_vec(R, G, B, L_max=300.0, gamma=2.2):
    perceived = hsp_brightness_vec(R, G, B)
    max_perceived = hsp_brightness_vec(255.0, 255.0, 255.0)
    norm = perceived / max_perceived
    return (norm ** gamma) * L_max


def luminance_to_photoiso_vec(luminance, pupil_diameter_mm=3.0, conversion_factor=1.0, ocular_transmittance=0.9):
    pupil_area = math.pi * (pupil_diameter_mm / 2.0) ** 2
    trolands = luminance * pupil_area
    return conversion_factor * trolands * ocular_transmittance


# Fast vectorized rgb -> approximate dominant wavelength via hue (no colorsys per-pixel)
def rgb_to_wavelength_vec(R, G, B):
    r = R.astype(np.float32) / 255.0
    g = G.astype(np.float32) / 255.0
    b = B.astype(np.float32) / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-12

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delta > 1e-9
    # max == r
    m_r = (maxc == r) & mask
    h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6.0
    # max == g
    m_g = (maxc == g) & mask
    h[m_g] = ((b[m_g] - r[m_g]) / delta[m_g]) + 2.0
    # max == b
    m_b = (maxc == b) & mask
    h[m_b] = ((r[m_b] - g[m_b]) / delta[m_b]) + 4.0
    h = h / 6.0
    h = np.mod(h, 1.0)
    return (380.0 + h * (700.0 - 380.0)).astype(np.float32)


# ---------------------------
# Fast vectorized photoreceptor grid generator
# ---------------------------
def generate_hex_grid(surface_radius, hex_size):
    """
    Produce a pointy-topped hex grid of centers that cover the circle of given radius.
    Returns array shape (M,2) of (x,y) in microns centered at (0,0).
    Vectorized—no python loops per hex.
    """
    # hexagon geometry
    hex_width = math.sqrt(3.0) * hex_size
    hex_height = 2.0 * hex_size
    v_spacing = 0.75 * hex_height

    # estimate grid limits
    row_min = int(np.floor(-surface_radius / v_spacing))
    row_max = int(np.ceil(surface_radius / v_spacing))
    col_min = int(np.floor(-surface_radius / hex_width))
    col_max = int(np.ceil(surface_radius / hex_width))

    rows = np.arange(row_min, row_max + 1)
    cols = np.arange(col_min, col_max + 1)
    R, C = np.meshgrid(rows, cols, indexing='ij')
    y = R * v_spacing
    offset = (hex_width / 2.0) * (R % 2 != 0)
    x = offset + C * hex_width

    xs = x.ravel()
    ys = y.ravel()
    mask = (xs ** 2 + ys ** 2) <= (surface_radius + 1e-3) ** 2
    centers = np.column_stack([xs[mask], ys[mask]]).astype(np.float32)
    return centers


def create_photoreceptor_table(surface_radius=1248.0, cone_threshold=208.0, hex_size=1.0, seed=42):
    """
    Build a flat photoreceptor table:
      - one row per photoreceptor unit (rods and cone subcells)
      - columns: x_micron, y_micron, type(0=rod,1=cone_sub), subtype(-1 for rod, 0=S,1=M,2=L),
                 lambda_max, threshold, conv_factor
    This function keeps geometry simple and vectorized for speed.
    """
    rng = np.random.default_rng(seed)
    centers = generate_hex_grid(surface_radius, hex_size)  # centers of hex tiles
    distances = np.sqrt((centers[:, 0]) ** 2 + (centers[:, 1]) ** 2)

    # Decide which centers are "foveal" (cones) vs peripheral candidate (mix)
    is_fovea = distances < cone_threshold

    # For peripheral centers we compute rod_probability using the same r_peak idea (vectorized)
    r_peak = (cone_threshold + surface_radius) / 2.0
    inner_mask = (distances <= r_peak) & (~is_fovea)
    outer_mask = (distances > r_peak) & (~is_fovea)

    rod_prob = np.zeros_like(distances, dtype=np.float32)
    # inner side: ((d - cone_threshold)/(r_peak - cone_threshold)) ** inner_exponent
    inner_exponent = 2.0
    outer_exponent = 0.3
    denom_inner = (r_peak - cone_threshold) if (r_peak - cone_threshold) != 0 else 1.0
    rod_prob[inner_mask] = (((distances[inner_mask] - cone_threshold) / denom_inner) ** inner_exponent).astype(
        np.float32)
    denom_outer = (surface_radius - r_peak) if (surface_radius - r_peak) != 0 else 1.0
    rod_prob[outer_mask] = (((surface_radius - distances[outer_mask]) / denom_outer) ** outer_exponent).astype(
        np.float32)

    # Now vectorized creation:
    rows = []
    # For fovea centers: create cones (3 subcells per cone center)
    cone_centers = centers[is_fovea]
    n_cone_centers = cone_centers.shape[0]
    # generate small offsets for sub-cones around the center (triangular-like)
    off = hex_size * 0.25
    offsets = np.array([[0.0, 0.0], [off, 0.0], [-off, 0.0]], dtype=np.float32)
    # tile and add
    if n_cone_centers:
        cone_coords = (cone_centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        # subtype assignment: cycle S,M,L (0,1,2) for each subcell
        subtypes = np.tile(np.array([0, 1, 2], dtype=np.int8), n_cone_centers)
        types = np.ones(len(subtypes), dtype=np.int8)  # cone_sub
        lambda_map = np.array([445.0, 535.0, 565.0], dtype=np.float32)
        lambda_vals = np.tile(lambda_map, n_cone_centers)
        threshold_vals = np.full(len(subtypes), 100.0, dtype=np.float32)  # default cone threshold
        conv_factor = np.full(len(subtypes), 100.0, dtype=np.float32)
        for i in range(len(subtypes)):
            rows.append((float(cone_coords[i, 0]), float(cone_coords[i, 1]), int(types[i]), int(subtypes[i]),
                         float(lambda_vals[i]), float(threshold_vals[i]), float(conv_factor[i])))

    # For peripheral centers: probabilistically create rods (3 per center) or cones nearer r_peak
    periph_centers = centers[~is_fovea]
    periph_probs = rod_prob[~is_fovea]
    n_per = periph_centers.shape[0]
    if n_per:
        draw = rng.random(n_per)
        # where draw < prob -> make rods (three each), else if distance < r_peak create cones
        rod_centers = periph_centers[draw < periph_probs]
        cone_like_centers = periph_centers[(draw >= periph_probs) & (distances[~is_fovea] < r_peak)]
        # rods: create 3 small-parallelogram offsets like original (angles factor 0,1,2)
        if rod_centers.size:
            nrod = rod_centers.shape[0]
            angles = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=np.float32)
            rad = hex_size * 0.5
            rod_offsets = np.column_stack([rad * np.cos(angles), rad * np.sin(angles)])
            rod_coords = (rod_centers[:, None, :] + rod_offsets[None, :, :]).reshape(-1, 2)
            types_r = np.zeros(len(rod_coords), dtype=np.int8)
            subtypes_r = np.full(len(rod_coords), -1, dtype=np.int8)
            lambda_vals_r = np.full(len(rod_coords), 498.0, dtype=np.float32)
            threshold_r = np.full(len(rod_coords), 1.0, dtype=np.float32)
            conv_factor_r = np.full(len(rod_coords), 1.0, dtype=np.float32)
            for i in range(len(rod_coords)):
                rows.append((float(rod_coords[i, 0]), float(rod_coords[i, 1]), int(types_r[i]), int(subtypes_r[i]),
                             float(lambda_vals_r[i]), float(threshold_r[i]), float(conv_factor_r[i])))
        # cone-like in inner periphery: create small cone groups as above (but fewer)
        if cone_like_centers.size:
            ncl = cone_like_centers.shape[0]
            off2 = hex_size * 0.25
            offsets2 = np.array([[0.0, 0.0], [off2, 0.0], [-off2, 0.0]], dtype=np.float32)
            cone_coords2 = (cone_like_centers[:, None, :] + offsets2[None, :, :]).reshape(-1, 2)
            for i in range(len(cone_coords2)):
                sub = i % 3
                rows.append((float(cone_coords2[i, 0]), float(cone_coords2[i, 1]), 1, int(sub),
                             float([445.0, 535.0, 565.0][sub]), 100.0, 100.0))

    # Build table as numpy structured arrays for speed & clarity
    if len(rows) == 0:
        raise RuntimeError("No photoreceptors generated - check parameters")

    arr = np.array(rows, dtype=[('x', 'f4'), ('y', 'f4'), ('is_cone_sub', 'i1'),
                                ('subtype', 'i1'), ('lambda_max', 'f4'),
                                ('threshold', 'f4'), ('conv_factor', 'f4')])
    return arr


# ---------------------------
# Precompute pixel mapping & save
# ---------------------------
def build_and_save_photoreceptors(out_path, surface_radius=1248.0, cone_threshold=208.0, hex_size=1.0,
                                  image_sample_shape=(2048, 2048)):
    """
    Build photoreceptor table and compute pixel mapping for an image shape sample (so mapping works per image scale).
    Save to out_path (.npz).
    """
    print("Building photoreceptors (vectorized)...")
    pr_table = create_photoreceptor_table(surface_radius=surface_radius,
                                          cone_threshold=cone_threshold,
                                          hex_size=hex_size)
    print(f"Generated {len(pr_table)} photoreceptor units (rows include cone subcells + rods).")

    H, W = image_sample_shape
    scale_x = W / (2.0 * surface_radius)
    scale_y = H / (2.0 * surface_radius)
    px = np.clip(((pr_table['x'] + surface_radius) * scale_x).astype(np.int32), 0, W - 1)
    py = np.clip(((pr_table['y'] + surface_radius) * scale_y).astype(np.int32), 0, H - 1)

    np.savez_compressed(out_path,
                        pr_table=pr_table,
                        px=px, py=py,
                        surface_radius=float(surface_radius),
                        hex_size=float(hex_size),
                        cone_threshold=float(cone_threshold))
    print(f"Wrote photoreceptors -> {out_path}")


# ---------------------------
# Per-frame processing (vectorized)
# ---------------------------
def process_image_with_photoreceptors(image_path, photofile, out_csv=None, out_png=None):
    print("Loading photoreceptors:", photofile)
    data = np.load(photofile, allow_pickle=True)
    pr_table = data['pr_table']
    px = data['px'].astype(np.int32)
    py = data['py'].astype(np.int32)
    surface_radius = float(data['surface_radius'])

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # H,W,3
    H_img, W_img = arr.shape[0], arr.shape[1]
    if (W_img, H_img) != (int((2 * surface_radius) * (W_img / (2 * surface_radius))),
                          int((2 * surface_radius) * (H_img / (2 * surface_radius)))):
        # mapping depends only on W/H; to be safe, recompute px/py for this image size
        scale_x = W_img / (2.0 * surface_radius)
        scale_y = H_img / (2.0 * surface_radius)
        px = np.clip(((pr_table['x'] + surface_radius) * scale_x).astype(np.int32), 0, W_img - 1)
        py = np.clip(((pr_table['y'] + surface_radius) * scale_y).astype(np.int32), 0, H_img - 1)

    # sample RGB for all receptors at once
    R = arr[py, px, 0].astype(np.float32)
    G = arr[py, px, 1].astype(np.float32)
    B = arr[py, px, 2].astype(np.float32)

    # wavelengths, luminance, photoisom (vector)
    wavelengths = rgb_to_wavelength_vec(R, G, B)
    luminance = rgb_to_luminance_vec(R, G, B)
    photoiso_base = luminance_to_photoiso_vec(luminance, conversion_factor=1.0)  # base conv

    # Multiply by per-row conv_factor, subtract threshold, apply spectral weight
    conv = pr_table['conv_factor'].astype(np.float32)
    thresh = pr_table['threshold'].astype(np.float32)
    lam = pr_table['lambda_max'].astype(np.float32)

    # compute spectral weight by shifting wavelength for lambda_max per receptor:
    # we approximate spectral_sensitivity(w, lambda_max) by scaling w before LUT lookup
    # effective_wavelength = w * (500.0 / lambda_max)
    eff_w = wavelengths * (500.0 / lam)
    spec_w = spectral_weight_from_wavelength(eff_w)

    photoiso = photoiso_base * conv
    raw_resp = (photoiso * spec_w) - thresh
    raw_resp = np.clip(raw_resp, 0.0, None).astype(np.float32)

    # Build opponent channels for cone parents:
    # We need to map cone subcells that belong to same "parent cone center".
    # We don't have explicit parent indices; but since cones were generated as 3 subcells in contiguous blocks,
    # we can reconstruct parent_idx by integer-dividing the cone-sub index positions.
    # Ensure we have an ndarray (works if pr_table['is_cone_sub'] is scalar or array)
    is_cone = np.asarray(pr_table['is_cone_sub'])
    # Coerce to boolean mask (safe even if is_cone was a scalar bool/int)
    is_cone_mask = (is_cone == 1)

    # Use flatnonzero to get an ndarray of indices (always an ndarray, no ambiguous bool)
    cone_idxs = np.flatnonzero(is_cone_mask)
    n_cones = cone_idxs.size // 3  # because we created triplets
    parent_idx = np.full(pr_table.shape[0], -1, dtype=np.int32)
    if cone_idxs.size > 0:
        # integer parent id per cone-subcell assuming contiguous triplets:
        parent_ids_for_cone_subs = (np.arange(cone_idxs.size, dtype=np.int32) // 3)
        parent_idx[cone_idxs] = parent_ids_for_cone_subs
        for i, global_idx in enumerate(cone_idxs):
            parent_idx[global_idx] = i // 3

    # accumulate L/M/S sums per parent
    n_parents = parent_idx.max() + 1 if parent_idx.max() >= 0 else 0
    L_sum = np.zeros(n_parents, dtype=np.float32)
    M_sum = np.zeros(n_parents, dtype=np.float32)
    S_sum = np.zeros(n_parents, dtype=np.float32)

    if n_parents > 0:
        cone_subtypes = pr_table['subtype'][is_cone].astype(np.int8)
        cone_resps = raw_resp[is_cone]
        cone_parents = parent_idx[is_cone]
        cone_subtypes = np.atleast_1d(cone_subtypes).astype(np.int8)
        cone_resps = np.atleast_1d(cone_resps).astype(np.float32)
        cone_parents = np.atleast_1d(cone_parents).astype(np.int32)

        mask_L = (cone_subtypes == 2)
        mask_M = (cone_subtypes == 1)
        mask_S = (cone_subtypes == 0)

        # Use np.any() to be robust in all cases (scalar or array)
        if np.any(mask_L):
            np.add.at(L_sum, cone_parents[mask_L], cone_resps[mask_L])
        if np.any(mask_M):
            np.add.at(M_sum, cone_parents[mask_M], cone_resps[mask_M])
        if np.any(mask_S):
            np.add.at(S_sum, cone_parents[mask_S], cone_resps[mask_S])

    # Compose output records
    records = []
    # For cones, we will attach opponency values for their parent
    # For rods, we attach raw response directly
    for i in range(pr_table.shape[0]):
        px_i = int(px[i])
        py_i = int(py[i])
        if pr_table['is_cone_sub'][i] == 1:
            pidx = parent_idx[i]
            rg = float(L_sum[pidx] - M_sum[pidx]) if pidx >= 0 else 0.0
            by = float(S_sum[pidx] - (L_sum[pidx] + M_sum[pidx])) if pidx >= 0 else 0.0
            lum = float(L_sum[pidx] + M_sum[pidx]) if pidx >= 0 else 0.0
            records.append({
                'idx': int(i),
                'x_micron': float(pr_table['x'][i]),
                'y_micron': float(pr_table['y'][i]),
                'pixel_x': px_i,
                'pixel_y': py_i,
                'cell_type': 'cone',
                'subtype': int(pr_table['subtype'][i]),
                'response': float(raw_resp[i]),
                'opp_rg': rg, 'opp_by': by, 'opp_lum': lum
            })
        else:
            records.append({
                'idx': int(i),
                'x_micron': float(pr_table['x'][i]),
                'y_micron': float(pr_table['y'][i]),
                'pixel_x': px_i,
                'pixel_y': py_i,
                'cell_type': 'rod',
                'subtype': -1,
                'response': float(raw_resp[i]),
                'opp_rg': 0.0, 'opp_by': 0.0, 'opp_lum': 0.0
            })
    df = None
    # Save CSV via pandas if requested
    try:
        import pandas as pd
        df = pd.DataFrame.from_records(records)
        if out_csv:
            df.to_csv(out_csv, index=False)
            print("Wrote CSV:", out_csv)
    except Exception as e:
        print("pandas not available or failed; skipping CSV write:", e)

    # Optionally create a simple overlay for visualization (non realtime)
    if out_png is not None:
        print("Creating overlay PNG (non-RT) ...")
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        df_for_plot = None
        try:
            df_for_plot = df
        except Exception:
            # fallback build small arrays
            xs = [r['pixel_x'] for r in records]
            ys = [r['pixel_y'] for r in records]
            resp = np.array([r['response'] for r in records], dtype=np.float32)
            plt.scatter(xs, ys, s=np.clip(resp / max(resp.max(), 1.0) * 4.0, 2, 8), alpha=0.6, marker='o')
        else:
            cones = df_for_plot[df_for_plot['cell_type'] == 'cone']
            rods = df_for_plot[df_for_plot['cell_type'] == 'rod']
            if len(cones):
                max_resp = max(cones['response'].max(), 1.0)
                plt.scatter(cones['pixel_x'], cones['pixel_y'],
                            s=(np.clip(cones['response'] / max_resp * 4.0, 2, 8)),
                            marker='o', alpha=0.6)
            if len(rods):
                max_resp = max(rods['response'].max(), 1.0)
                plt.scatter(rods['pixel_x'], rods['pixel_y'],
                            s=(np.clip(rods['response'] / (max_resp) * 3.0, 2, 6)),
                            marker='.', alpha=0.4)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Wrote PNG:", out_png)

    return records


# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true", help="Build and save photoreceptors (one-time).")
    p.add_argument("--process", action="store_true", help="Process a single image using saved photoreceptors.")
    p.add_argument("--photofile", default="photoreceptors.npz")
    p.add_argument("--image", default=None)
    p.add_argument("--out", default="receive_out.csv")
    p.add_argument("--out_png", default="receive_out.png")
    p.add_argument("--surface_radius", type=float, default=1248.0)
    p.add_argument("--cone_threshold", type=float, default=208.0)
    p.add_argument("--hex_size", type=float, default=1.0)
    p.add_argument("--sample_w", type=int, default=2048, help="Sample image width for mapping when init")
    p.add_argument("--sample_h", type=int, default=2048, help="Sample image height for mapping when init")
    args = p.parse_args()

    if args.init:
        build_and_save_photoreceptors(args.photofile,
                                      surface_radius=args.surface_radius,
                                      cone_threshold=args.cone_threshold,
                                      hex_size=args.hex_size,
                                      image_sample_shape=(args.sample_h, args.sample_w))
    if args.process:
        if args.image is None:
            raise SystemExit("Please provide --image to process.")
        process_image_with_photoreceptors(args.image, args.photofile, out_csv=args.out, out_png=args.out_png)


if __name__ == "__main__":
    main()
