#!/usr/bin/env python3
"""
camera_receive_allinone.py

Single-file camera -> photoreceptor front-end that accepts your original photoreceptors.pkl
or a photoreceptors.npz built previously.

Designed for low per-frame latency (vectorized NumPy, no Python-per-receptor loops).

Usage examples:
  # Using your original pickle (default names)
  python camera_receive_allinone.py --photofile /path/to/photoreceptors.pkl --cam 0 --frames 500

  # Using an npz produced by fast_receive
  python camera_receive_allinone.py --photofile photoreceptors.npz --cam 0 --frames 500

Notes:
  - Requires: numpy, opencv-python (cv2). Pillow not required.
  - If cv2 not present the script will error early. Install with: pip install opencv-python
"""

from __future__ import annotations
import argparse
import pickle
import time
import os
import sys
from typing import Dict, Any, Tuple
import numpy as np

# Prefer cv2 for camera capture; error out if missing.
try:
    import cv2
except Exception as e:
    cv2 = None
    # We'll raise later if used


# -------------------------
# LUTs and small utilities
# -------------------------
def build_spectral_lut(wmin: float = 380.0, wmax: float = 700.0, n: int = 321) -> Tuple[np.ndarray, np.ndarray]:
    w = np.linspace(wmin, wmax, n).astype(np.float32)
    k = 69.7
    gv = np.exp(3.21 * np.log(k / w) - 0.485 * (np.log(k / w)) ** 2 + 9.71e-3 * (np.log(k / w)) ** 3)
    gv = (gv / gv.max()).astype(np.float32)
    return w, gv


W_LUT, SENS_LUT = build_spectral_lut()


def spectral_weight_from_wavelength_vec(wavelengths: np.ndarray) -> np.ndarray:
    return np.interp(wavelengths, W_LUT, SENS_LUT, left=0.0, right=0.0).astype(np.float32)


def rgb_to_wavelength_vec(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    # approximate dominant wavelength via hue (vectorized)
    r = R.astype(np.float32) / 255.0
    g = G.astype(np.float32) / 255.0
    b = B.astype(np.float32) / 255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-12

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delta > 1e-9
    m_r = (maxc == r) & mask
    m_g = (maxc == g) & mask
    m_b = (maxc == b) & mask

    # compute hues (safe division)
    h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6.0
    h[m_g] = ((b[m_g] - r[m_g]) / delta[m_g]) + 2.0
    h[m_b] = ((r[m_b] - g[m_b]) / delta[m_b]) + 4.0
    h = h / 6.0
    h = np.mod(h, 1.0)
    return (380.0 + h * (700.0 - 380.0)).astype(np.float32)


def hsp_brightness_vec(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sqrt(0.299 * (R ** 2) + 0.587 * (G ** 2) + 0.114 * (B ** 2)).astype(np.float32)


def rgb_to_luminance_vec(R: np.ndarray, G: np.ndarray, B: np.ndarray, L_max: float = 300.0, gamma: float = 2.2) -> np.ndarray:
    perceived = hsp_brightness_vec(R, G, B)
    max_perceived = hsp_brightness_vec(255.0, 255.0, 255.0)
    norm = perceived / max_perceived
    return (norm ** gamma) * L_max


def luminance_to_photoiso_vec(luminance: np.ndarray, pupil_diameter_mm: float = 3.0,
                              conversion_factor: float = 1.0, ocular_transmittance: float = 0.9) -> np.ndarray:
    pupil_area = np.pi * (pupil_diameter_mm / 2.0) ** 2
    trolands = luminance * pupil_area
    return (conversion_factor * trolands * ocular_transmittance).astype(np.float32)


# -------------------------
# Loader: .npz or .pkl
# -------------------------
def load_photoreceptors_from_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    pr_table = data['pr_table']
    px = data['px'].astype(np.int32)
    py = data['py'].astype(np.int32)

    # precompute constants
    conv = pr_table['conv_factor'].astype(np.float32)
    thresh = pr_table['threshold'].astype(np.float32)
    lambda_max = pr_table['lambda_max'].astype(np.float32)
    lambda_max = np.where(lambda_max <= 0.0, 500.0, lambda_max)
    inv_lambda_factor = (500.0 / lambda_max).astype(np.float32)
    is_cone = np.asarray(pr_table['is_cone_sub']).astype(np.int8) == 1
    subtype = np.asarray(pr_table['subtype']).astype(np.int8)

    return {
        'pr_table': pr_table, 'px': px, 'py': py,
        'conv': conv, 'thresh': thresh, 'inv_lambda_factor': inv_lambda_factor,
        'is_cone': is_cone, 'subtype': subtype,
        'surface_radius': float(data['surface_radius']) if 'surface_radius' in data else None
    }


def load_photoreceptors_from_pkl(pkl_path: str, image_sample_shape=(2048, 2048), surface_radius: float = 1248.0) -> Dict[str, Any]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        stored = pickle.load(f)

    rows = []
    # stored is expected to be a list of parent cell objects (cones parents) and rods
    for obj in stored:
        cell_type = getattr(obj, 'cell_type', None)
        if cell_type == 'cone' or (hasattr(obj, 'cells') and any(getattr(c, 'subtype', None) is not None for c in getattr(obj, 'cells', []))):
            # parent cone: iterate subcells
            for c in obj.cells:
                cx = float(getattr(c, 'x', getattr(obj, 'x', 0.0)))
                cy = float(getattr(c, 'y', getattr(obj, 'y', 0.0)))
                subtype = getattr(c, 'subtype', None)
                if isinstance(subtype, str):
                    subtype_map = {'S': 0, 'M': 1, 'L': 2}
                    sidx = subtype_map.get(subtype.upper(), 1)
                else:
                    sidx = int(subtype) if subtype is not None else 1
                lam = float(getattr(c, 'lambda_max', 535.0))
                thresh = float(getattr(c, 'threshold', 100.0))
                conv = float(getattr(c, 'conv_factor', 100.0)) if hasattr(c, 'conv_factor') else 100.0
                rows.append((cx, cy, 1, sidx, lam, thresh, conv))
        elif cell_type == 'rod' or (hasattr(obj, 'cells') and any((getattr(c, 'lambda_max', None) == 498 or getattr(c, 'threshold', None) == 1) for c in getattr(obj, 'cells', []))):
            for c in obj.cells:
                cx = float(getattr(c, 'x', getattr(obj, 'x', 0.0)))
                cy = float(getattr(c, 'y', getattr(obj, 'y', 0.0)))
                lam = float(getattr(c, 'lambda_max', 498.0))
                thresh = float(getattr(c, 'threshold', 1.0))
                conv = float(getattr(c, 'conv_factor', 1.0)) if hasattr(c, 'conv_factor') else 1.0
                rows.append((cx, cy, 0, -1, lam, thresh, conv))
        else:
            # fallback: try to interpret obj.cells if present
            if hasattr(obj, 'cells') and len(obj.cells) > 0:
                for c in obj.cells:
                    subtype = getattr(c, 'subtype', None)
                    cx = float(getattr(c, 'x', getattr(obj, 'x', 0.0)))
                    cy = float(getattr(c, 'y', getattr(obj, 'y', 0.0)))
                    if subtype is not None:
                        if isinstance(subtype, str):
                            subtype_map = {'S': 0, 'M': 1, 'L': 2}
                            sidx = subtype_map.get(subtype.upper(), 1)
                        else:
                            sidx = int(subtype)
                        lam = float(getattr(c, 'lambda_max', 535.0))
                        thresh = float(getattr(c, 'threshold', 100.0))
                        conv = float(getattr(c, 'conv_factor', 100.0)) if hasattr(c, 'conv_factor') else 100.0
                        rows.append((cx, cy, 1, sidx, lam, thresh, conv))
                    else:
                        lam = float(getattr(c, 'lambda_max', 498.0))
                        thresh = float(getattr(c, 'threshold', 1.0))
                        conv = float(getattr(c, 'conv_factor', 1.0)) if hasattr(c, 'conv_factor') else 1.0
                        rows.append((cx, cy, 0, -1, lam, thresh, conv))
            else:
                # if obj itself looks receptor-like
                if hasattr(obj, 'lambda_max') or hasattr(obj, 'subtype'):
                    cx = float(getattr(obj, 'x', 0.0))
                    cy = float(getattr(obj, 'y', 0.0))
                    subtype = getattr(obj, 'subtype', None)
                    if subtype is not None:
                        if isinstance(subtype, str):
                            subtype_map = {'S': 0, 'M': 1, 'L': 2}
                            sidx = subtype_map.get(subtype.upper(), 1)
                        else:
                            sidx = int(subtype)
                        lam = float(getattr(obj, 'lambda_max', 535.0))
                        thresh = float(getattr(obj, 'threshold', 100.0))
                        conv = float(getattr(obj, 'conv_factor', 100.0)) if hasattr(obj, 'conv_factor') else 100.0
                        rows.append((cx, cy, 1, sidx, lam, thresh, conv))
                    else:
                        lam = float(getattr(obj, 'lambda_max', 498.0))
                        thresh = float(getattr(obj, 'threshold', 1.0))
                        conv = float(getattr(obj, 'conv_factor', 1.0)) if hasattr(obj, 'conv_factor') else 1.0
                        rows.append((cx, cy, 0, -1, lam, thresh, conv))
                else:
                    # skip unknown entry
                    continue

    if len(rows) == 0:
        raise RuntimeError("No usable photoreceptor entries found in pickle.")

    pr_table = np.array(rows, dtype=[('x', 'f4'), ('y', 'f4'), ('is_cone_sub', 'i1'),
                                     ('subtype', 'i1'), ('lambda_max', 'f4'),
                                     ('threshold', 'f4'), ('conv_factor', 'f4')])

    H, W = image_sample_shape
    scale_x = W / (2.0 * surface_radius)
    scale_y = H / (2.0 * surface_radius)
    px = np.clip(((pr_table['x'] + surface_radius) * scale_x).astype(np.int32), 0, W - 1)
    py = np.clip(((pr_table['y'] + surface_radius) * scale_y).astype(np.int32), 0, H - 1)

    conv = pr_table['conv_factor'].astype(np.float32)
    thresh = pr_table['threshold'].astype(np.float32)
    lambda_max = pr_table['lambda_max'].astype(np.float32)
    lambda_max = np.where(lambda_max <= 0.0, 500.0, lambda_max)
    inv_lambda_factor = (500.0 / lambda_max).astype(np.float32)
    is_cone = (pr_table['is_cone_sub'].astype(np.int8) == 1)
    subtype = pr_table['subtype'].astype(np.int8)

    return {
        'pr_table': pr_table,
        'px': px, 'py': py,
        'conv': conv, 'thresh': thresh, 'inv_lambda_factor': inv_lambda_factor,
        'is_cone': is_cone, 'subtype': subtype, 'surface_radius': float(surface_radius)
    }


def load_photoreceptors_auto(path: str, image_sample_shape=(2048, 2048), surface_radius: float = 1248.0) -> Dict[str, Any]:
    lower = path.lower()
    if lower.endswith('.npz'):
        return load_photoreceptors_from_npz(path)
    elif lower.endswith('.pkl') or lower.endswith('.pickle'):
        return load_photoreceptors_from_pkl(path, image_sample_shape=image_sample_shape, surface_radius=surface_radius)
    else:
        # try both heuristics
        if os.path.exists(path + '.npz'):
            return load_photoreceptors_from_npz(path + '.npz')
        if os.path.exists(path + '.pkl'):
            return load_photoreceptors_from_pkl(path + '.pkl', image_sample_shape=image_sample_shape, surface_radius=surface_radius)
        raise FileNotFoundError(f"Photoreceptor file not found or unsupported extension: {path}")


# -------------------------
# Camera processing loop
# -------------------------
def process_camera_stream(photofile: str, cam_device: int | str = 0, frame_limit: int | None = None,
                          show: bool = False, rod_alpha: float = 0.2, sample_shape=(2048, 2048),
                          surface_radius: float = 1248.0) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Install via: pip install opencv-python")

    data = load_photoreceptors_auto(photofile, image_sample_shape=sample_shape, surface_radius=surface_radius)
    px = data['px']
    py = data['py']
    conv = data['conv']
    thresh = data['thresh']
    inv_lambda_factor = data['inv_lambda_factor']
    is_cone = data['is_cone']
    subtype = data['subtype']
    n_receptors = px.shape[0]

    has_rods = np.any(~is_cone)
    rod_state = np.zeros(n_receptors, dtype=np.float32) if has_rods else None

    print(f"[camera_receive] loaded photoreceptors: n_receptors={n_receptors}, cones={int(np.sum(is_cone))}, rods={int(np.sum(~is_cone))}")

    cap = cv2.VideoCapture(int(cam_device)) if isinstance(cam_device, int) or cam_device.isdigit() else cv2.VideoCapture(cam_device)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera device or file: {cam_device}")

    remap_needed = False
    frame_count = 0
    t0 = time.perf_counter()
    last_report = t0
    latencies = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[camera_receive] frame read failed / stream ended")
                break

            t_start = time.perf_counter()

            H_img, W_img = frame.shape[0], frame.shape[1]
            # If px/py mapping outside current frame, recompute once
            if (px.max() >= W_img or py.max() >= H_img) and not remap_needed:
                # try to remap using surface_radius if stored, else scale proportionally
                sr = data.get('surface_radius', None)
                if sr is None:
                    # proportional scaling fallback
                    px = np.clip((px * (W_img / max(1, px.max()+1))).astype(np.int32), 0, W_img-1)
                    py = np.clip((py * (H_img / max(1, py.max()+1))).astype(np.int32), 0, H_img-1)
                else:
                    pr_table = data['pr_table']
                    scale_x = W_img / (2.0 * sr)
                    scale_y = H_img / (2.0 * sr)
                    px = np.clip(((pr_table['x'] + sr) * scale_x).astype(np.int32), 0, W_img-1)
                    py = np.clip(((pr_table['y'] + sr) * scale_y).astype(np.int32), 0, H_img-1)
                remap_needed = True

            # frame: BGR uint8
            R = frame[py, px, 2].astype(np.float32)
            G = frame[py, px, 1].astype(np.float32)
            B = frame[py, px, 0].astype(np.float32)

            # photoreceptor computation (vectorized)
            wavelengths = rgb_to_wavelength_vec(R, G, B)
            luminance = rgb_to_luminance_vec(R, G, B)
            photoiso_base = luminance_to_photoiso_vec(luminance, conversion_factor=1.0)

            eff_w = wavelengths * inv_lambda_factor
            spec_w = spectral_weight_from_wavelength_vec(eff_w)

            resp = (photoiso_base * conv * spec_w) - thresh
            resp = np.clip(resp, 0.0, None).astype(np.float32)

            if has_rods:
                mask_rods = ~is_cone
                # in-place EMA for rods
                rod_state[mask_rods] = (1.0 - rod_alpha) * rod_state[mask_rods] + rod_alpha * resp[mask_rods]
                output_resp = resp.copy()
                output_resp[mask_rods] = rod_state[mask_rods]
            else:
                output_resp = resp  # cones only

            # output_resp is the per-receptor signal to hand off to next stage.
            # For minimal latency we do NOT serialize here. If you need IPC, use shared memory or a socket.

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            latencies.append(latency_ms)
            frame_count += 1

            # periodic report every 1s
            now = t_end
            if now - last_report >= 1.0:
                recent = latencies[-max(1, len(latencies)//10):]
                avg_recent = float(np.mean(recent))
                fps = frame_count / (now - t0)
                print(f"[camera_receive] frames={frame_count} fps={fps:.1f} last_latency={latency_ms:.2f} ms recent_avg={avg_recent:.2f} ms")
                last_report = now

            # optional visual debug overlay
            if show:
                vis = frame.copy()
                max_plot = 2000
                n = output_resp.shape[0]
                if n <= max_plot:
                    sel = np.arange(n)
                else:
                    rng = np.random.default_rng(123456)
                    sel = rng.choice(n, size=max_plot, replace=False)
                vals = output_resp[sel]
                vmax = max(vals.max(), 1e-6)
                sizes = np.clip((vals / vmax) * 6.0, 1.0, 8.0).astype(np.int32)
                for ii, ix in enumerate(sel):
                    cx = int(px[ix]); cy = int(py[ix])
                    color = (0, int(min(255, 255 * (vals[ii] / vmax))), 0)
                    cv2.circle(vis, (cx, cy), int(sizes[ii]), color, thickness=-1)
                cv2.imshow("camera_receive (debug)", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if (frame_limit is not None) and (frame_count >= frame_limit):
                break

    finally:
        try:
            cap.release()
            if show:
                cv2.destroyAllWindows()
        except Exception:
            pass

    total_time = time.perf_counter() - t0
    if frame_count > 0:
        print(f"[camera_receive] done frames={frame_count} total_time={total_time:.3f}s avg_latency={(total_time/frame_count*1000.0):.2f} ms avg_fps={(frame_count/total_time):.2f}")


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Camera -> photoreceptor front-end (single-file).")
    p.add_argument("--photofile", default="photoreceptors.pkl", help="path to photoreceptors (.pkl or .npz)")
    p.add_argument("--cam", default="0", help="camera device index (0) or video file path")
    p.add_argument("--frames", type=int, default=None, help="number of frames to process then exit")
    p.add_argument("--show", action="store_true", help="visual debug overlay (slower)")
    p.add_argument("--rod-alpha", type=float, default=0.2, help="EMA alpha for rod temporal integration (0..1)")
    p.add_argument("--sample_w", type=int, default=2048, help="sample image width used when loading .pkl")
    p.add_argument("--sample_h", type=int, default=2048, help="sample image height used when loading .pkl")
    p.add_argument("--surface_radius", type=float, default=1248.0, help="retina radius in microns (if using .pkl)")
    args = p.parse_args()

    cam_device = args.cam
    try:
        cam_device_int = int(cam_device)
        cam_device = cam_device_int
    except Exception:
        # if a file path or non-int string, pass it to cv2.VideoCapture directly
        cam_device = cam_device

    process_camera_stream(
        photofile=args.photofile,
        cam_device=cam_device,
        frame_limit=args.frames,
        show=args.show,
        rod_alpha=args.rod_alpha,
        sample_shape=(args.sample_h, args.sample_w),
        surface_radius=args.surface_radius
    )


if __name__ == "__main__":
    main()
