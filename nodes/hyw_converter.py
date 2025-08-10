from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ComfyStockBundleRuntime:
    """Lightweight HYW_RUNTIME-compatible adapter that carries ComfyUI stock components.

    Notes
    - This object is meant to be passed as `HYW_RUNTIME` into HunyuanWorld nodes.
    - It encapsulates references to stock ComfyUI `MODEL`, `CLIP`, and `VAE` objects
      so they can be carried through a single noodle.
    - Generation methods are not implemented here; this adapter only bundles components.
    - Downstream nodes can detect this via `is_comfy_stock_bundle()` and extract components
      with `get_components()`.
    """

    model: Any
    clip: Any
    vae: Optional[Any] = None
    device: str = ""
    dtype: str = ""
    hints: Optional[Dict[str, Any]] = None

    # Introspection helpers for downstream nodes
    def is_comfy_stock_bundle(self) -> bool:
        return True

    def get_components(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "clip": self.clip,
            "vae": self.vae,
            "device": self.device,
            "dtype": self.dtype,
            "hints": self.hints or {},
        }


class HYW_RuntimeFromStock:
    """Convert stock ComfyUI MODEL/CLIP/VAE into a HYW_RUNTIME noodle.

    This node does not load or change models; it only bundles references from
    default ComfyUI loaders into a single object returned as `HYW_RUNTIME` so
    you can connect it into HunyuanWorld nodes expecting a runtime noodle.

    Current limitations
    - This adapter does not implement generation. HunyuanWorld generation nodes
      that directly call diffusers pipelines will not use these components yet.
    - It is intended as an integration bridge so HYW nodes can later detect and
      consume stock components without reloading them.
    """

    CATEGORY = "HunyuanWorld/Adapters"
    RETURN_TYPES = ("HYW_RUNTIME",)
    RETURN_NAMES = ("hyw_runtime",)
    FUNCTION = "bundle"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL from ComfyUI UNET/Checkpoint loader, optionally after LoRA."}),
                "clip": ("CLIP", {"tooltip": "CLIP from DualCLIPLoader (or similar)."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Optional VAE from VAELoader."}),
                "device": ("STRING", {"default": "", "tooltip": "Optional device hint (e.g., cuda:0)."}),
                "dtype": ("STRING", {"default": "", "tooltip": "Optional dtype hint (bfloat16/float16/float32)."}),
            },
        }

    def _extract_hints(self, model: Any, clip: Any, vae: Optional[Any]) -> Dict[str, Any]:
        """Best-effort extraction of useful hints (like filenames) from stock objects.

        This is intentionally defensive; if nothing is found, returns an empty dict.
        """
        hints: Dict[str, Any] = {}

        def maybe_attach(name: str, obj: Any):
            try:
                if obj is None:
                    return
                # Common attributes on Comfy wrappers to surface
                for attr in ("filename", "ckpt_path", "file_path", "path", "model_name"):
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str) and val:
                            hints[f"{name}_{attr}"] = val
                # Some wrappers store an inner object under `.model` or `.clip` with similar attrs
                for inner in ("model", "clip", "vae"):
                    if hasattr(obj, inner):
                        inner_obj = getattr(obj, inner)
                        for attr in ("filename", "ckpt_path", "file_path", "path", "model_name"):
                            if hasattr(inner_obj, attr):
                                val = getattr(inner_obj, attr)
                                if isinstance(val, str) and val:
                                    hints[f"{name}_{inner}_{attr}"] = val
            except Exception:
                # Hints are opportunistic; ignore failures
                pass

        maybe_attach("model", model)
        maybe_attach("clip", clip)
        maybe_attach("vae", vae)
        return hints

    def bundle(self, model, clip, vae=None, device: str = "", dtype: str = ""):
        hints = self._extract_hints(model, clip, vae)
        runtime = ComfyStockBundleRuntime(model=model, clip=clip, vae=vae, device=device, dtype=dtype, hints=hints)
        return (runtime,)
