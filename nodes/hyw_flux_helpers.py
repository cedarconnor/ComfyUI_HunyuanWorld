import os
import sys
from typing import Tuple

import numpy as np
import torch
try:
    import cv2  # Required for projection math and remap
    _CV2_OK = True
    _CV2_ERR = None
except Exception as _e:
    _CV2_OK = False
    _CV2_ERR = _e
from PIL import Image

# Reuse tensor helpers
from .hyw_utils import pil_to_tensor, tensor_to_pil

# Local perspective→equirect projection (no external imports required)
def _project_perspective_to_equirect(img_bgr: np.ndarray, height: int, width: int, fov: float, theta: float, phi: float,
                                     crop_bound: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Project a perspective image onto an equirectangular canvas using FOV/theta/phi.

    Returns (persp_on_equirect_bgr, mask_3c_float01)
    - persp_on_equirect_bgr: HxWx3 uint8 BGR image with zeros where no data
    - mask_3c_float01: HxWx3 float32 mask in [0,1], 1 where data is present
    """
    if not _CV2_OK:
        raise RuntimeError(f"cv2 import failed: {repr(_CV2_ERR)}")

    # Optionally crop boundaries proportionally (mirrors HunyuanWorld behavior)
    if crop_bound:
        h, w = img_bgr.shape[:2]
        y0, y1 = int(h * 0.05), int(h * 0.95)
        x0, x1 = int(w * 0.05), int(w * 0.95)
        img_bgr = img_bgr[y0:y1, x0:x1]

    src_h, src_w = img_bgr.shape[:2]
    # Derive horizontal/vertical FOV based on source aspect
    if src_w > src_h:
        wFOV = fov
        hFOV = (float(src_h) / src_w) * fov
    else:
        wFOV = (float(src_w) / src_h) * fov
        hFOV = fov

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    # Build equirect spherical grid: x=lon [-180,180], y=lat [90,-90]
    lon, lat = np.meshgrid(np.linspace(-180.0, 180.0, width),
                           np.linspace(90.0, -90.0, height))
    # Spherical to Cartesian (unit sphere)
    x_map = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    y_map = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    z_map = np.sin(np.radians(lat))
    xyz = np.stack((x_map, y_map, z_map), axis=2)

    # Rotate by theta around Z, then by -phi around Y (match HunyuanWorld code)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R1, _ = cv2.Rodrigues(z_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))
    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape(height * width, 3).T
    xyz = R2 @ xyz
    xyz = R1 @ xyz
    xyz = xyz.T.reshape(height, width, 3)

    # Keep only forward-facing points (x > 0)
    forward_mask = (xyz[:, :, 0] > 0).astype(np.float32)

    # Perspective division (normalize by x)
    xyz = xyz / np.repeat(xyz[:, :, 0][:, :, None], 3, axis=2)

    # Map to perspective image coordinates
    cond = ((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) &
            (-h_len < xyz[:, :, 2]) & (xyz[:, :, 2] < h_len))
    lon_map = np.where(cond, (xyz[:, :, 1] + w_len) / (2 * w_len) * src_w, 0.0)
    lat_map = np.where(cond, (-xyz[:, :, 2] + h_len) / (2 * h_len) * src_h, 0.0)
    mask = (cond.astype(np.float32) * forward_mask).astype(np.float32)

    persp = cv2.remap(img_bgr,
                      lon_map.astype(np.float32),
                      lat_map.astype(np.float32),
                      interpolation=cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_WRAP)

    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    persp = (persp * mask3).astype(np.uint8)
    return persp, mask3


def _image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    # Expect IMAGE tensor [1, H, W, C] in 0..1
    if image_tensor.dim() == 4:
        arr = image_tensor[0].cpu().numpy()
    else:
        arr = image_tensor.cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def _mask_numpy_to_tensor(mask_np: np.ndarray) -> torch.Tensor:
    # Expect HxW uint8 0..255, return [1, H, W] float 0..1
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask_f = (mask_np.astype(np.float32) / 255.0)
    mask_t = torch.from_numpy(mask_f)[None, ...]
    return mask_t


class HYW_PerspectiveToPanoramaMask:
    """Convert a perspective image into an equirectangular seed + inpaint mask.

    - Outputs an equirect image placed using FOV/theta/phi and a mask where white=to-generate.
    - Suitable for use with VAEEncodeForInpaint → KSampler (FLUX Fill) → VAEDecode workflow.
    """

    CATEGORY = "HunyuanWorld/Adapters"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("panorama_seed", "mask")
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Perspective input image to place onto an equirectangular canvas."}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 16384, "step": 64, "tooltip": "Target panorama width (2:1 recommended)."}),
                "height": ("INT", {"default": 960, "min": 128, "max": 8192, "step": 64, "tooltip": "Target panorama height (half of width recommended)."}),
                "fov": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 180.0, "tooltip": "Field of view of the input image in degrees."}),
                "theta": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "tooltip": "Horizontal placement angle (yaw) in degrees."}),
                "phi": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "tooltip": "Vertical placement angle (pitch) in degrees."}),
            },
            "optional": {
                "erode_radius": ("INT", {"default": 5, "min": 0, "max": 32, "tooltip": "Erode the keep-mask to allow FLUX Fill to regenerate a small border around the seed."}),
                "assume_equirect_if_2to1": ("BOOLEAN", {"default": True, "tooltip": "If input image is ~2:1 aspect, treat it as an equirect panorama and bypass projection."}),
                "equirect_mask_mode": (["none", "all_generate"], {"default": "none", "tooltip": "Mask behavior when bypassing projection: 'none' creates an all-black mask (no inpaint). 'all_generate' creates an all-white mask."}),
            },
        }

    def convert(self, input_image, width, height, fov, theta, phi, erode_radius=5, assume_equirect_if_2to1=True, equirect_mask_mode="none") -> Tuple[torch.Tensor, torch.Tensor]:
        if not _CV2_OK:
            raise RuntimeError("Perspective projection dependency missing: cv2 failed to import.\n"
                               f"cv2 error: {repr(_CV2_ERR)}\n"
                               "Install with: pip install opencv-python-headless")

        # Convert to numpy BGR
        img_np = _image_tensor_to_numpy(input_image)

        # If input is already equirect (≈2:1), pass-through if requested
        h0, w0 = img_np.shape[0], img_np.shape[1]
        if assume_equirect_if_2to1 and h0 > 0 and abs((w0 / max(1, h0)) - 2.0) < 0.03:
            # Resize to target canvas if needed
            if (w0, h0) != (width, height):
                img_np_resized = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
            else:
                img_np_resized = img_np

            seed_pil = Image.fromarray(img_np_resized)
            seed_tensor = pil_to_tensor(seed_pil)

            if equirect_mask_mode == "all_generate":
                mask_u8 = np.full((height, width), 255, dtype=np.uint8)
            else:
                mask_u8 = np.zeros((height, width), dtype=np.uint8)
            mask_tensor = _mask_numpy_to_tensor(mask_u8)
            return (seed_tensor, mask_tensor)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            # If grayscale, expand to BGR
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        h_in, w_in = img_bgr.shape[:2]
        # Optional pre-resize to roughly match FOV coverage
        if w_in > h_in:
            ratio = w_in / h_in
            w = int((fov / 360.0) * width)
            h = max(1, int(w / ratio))
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ratio = h_in / w_in
            h = int((fov / 180.0) * height)
            w = max(1, int(h / ratio))
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # Project to equirectangular canvas (local implementation)
        img_array, mask_array = _project_perspective_to_equirect(
            img_bgr=img_bgr, height=height, width=width, fov=fov, theta=theta, phi=phi, crop_bound=False
        )

        # Erode mask and compute masked seed
        if erode_radius > 0:
            kernel = np.ones((erode_radius, erode_radius), np.uint8)
            # mask_array is 3-channel float [0,1]; convert to single-channel uint8 for morphology
            mask_u8 = (mask_array[:, :, 0] * 255).astype(np.uint8)
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
            mask_array = np.repeat((mask_u8.astype(np.float32) / 255.0)[:, :, None], 3, axis=2)
        else:
            # Ensure mask in float [0,1]
            mask_array = mask_array.astype(np.float32)

        img_array = img_array * mask_array
        # Invert for inpainting: 255 where to generate
        mask_u8 = (mask_array[:, :, 0] * 255).astype(np.uint8)
        mask_inv_u8 = 255 - mask_u8

        seed_rgb = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2RGB)
        seed_pil = Image.fromarray(seed_rgb)
        seed_tensor = pil_to_tensor(seed_pil)
        mask_tensor = _mask_numpy_to_tensor(mask_inv_u8)

        return (seed_tensor, mask_tensor)


class HYW_ShiftPanorama:
    """Shift a panorama horizontally (wrap-around)."""

    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("shifted",)
    FUNCTION = "shift"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE", {"tooltip": "Panorama image to shift horizontally with wrap-around."}),
                "shift_pixels": ("INT", {"default": 0, "min": -32768, "max": 32768, "tooltip": "Positive shifts right; negative shifts left. Use to move the seam."}),
            },
        }

    def shift(self, panorama, shift_pixels):
        img_np = _image_tensor_to_numpy(panorama)
        # Roll horizontally; keep channels if present
        if img_np.ndim == 3:
            shifted = np.roll(img_np, shift_pixels, axis=1)
        else:
            shifted = np.roll(img_np, shift_pixels, axis=1)
        shifted_pil = Image.fromarray(shifted)
        return (pil_to_tensor(shifted_pil),)
