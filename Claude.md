\# Repository Skeleton — HunyuanWorld‑1.0 ComfyUI Node Pack



\*\*Purpose:\*\* Provide a ready-to-use skeleton repo structure matching the Software Design Document. Includes placeholders for all nodes, configuration, requirements, example workflows, and golden assets.



---



\## Folder Structure



```

ComfyUI/custom\_nodes/

&nbsp; ComfyUI\_HunyuanWorld/

&nbsp;   \_\_init\_\_.py

&nbsp;   nodes/

&nbsp;     hyw\_loader.py

&nbsp;     hyw\_panogen.py

&nbsp;     hyw\_inpaint\_scene.py

&nbsp;     hyw\_inpaint\_sky.py

&nbsp;     hyw\_reconstruct.py

&nbsp;     hyw\_export.py

&nbsp;     hyw\_utils.py

&nbsp;   requirements.txt

&nbsp;   settings.json

&nbsp;   example\_workflows/

&nbsp;     text\_to\_pano.json

&nbsp;     image\_to\_mesh.json

&nbsp;     text\_to\_mesh\_high.json

&nbsp;     golden/

&nbsp;       prompt.txt

&nbsp;       reference\_image.png

&nbsp;       expected\_mesh\_stats.json

&nbsp;   README.md

&nbsp;   LICENSE

```



---



\## File Stubs



``



```python

\# HunyuanWorld-1.0 ComfyUI Node Pack init

\# Registers all custom nodes



from .nodes import hyw\_loader, hyw\_panogen, hyw\_inpaint\_scene, hyw\_inpaint\_sky, hyw\_reconstruct, hyw\_export, hyw\_utils

NODE\_CLASS\_MAPPINGS = {}

NODE\_DISPLAY\_NAME\_MAPPINGS = {}

```



``



```python

class HYW\_ModelLoader:

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {"base\_dir": ("STRING", {}), "weights\_panotext": ("STRING", {}), "weights\_panoimage": ("STRING", {}), "weights\_pano\_inpaint\_scene": ("STRING", {}), "weights\_pano\_inpaint\_sky": ("STRING", {}), "weights\_recon": ("STRING", {}), "dtype": (\["fp32", "fp16", "bf16"],), "device": ("STRING", {}), "enable\_xformers": ("BOOL", {"default": True})}}

&nbsp;   RETURN\_TYPES = ("MODEL",)

&nbsp;   FUNCTION = "load"

&nbsp;   CATEGORY = "HunyuanWorld/Loaders"

&nbsp;   def load(self, \*\*kwargs):

&nbsp;       pass  # TODO: implement loader per SDD

```



\*\*Other node files\*\* (`hyw\_panogen.py`, `hyw\_inpaint\_scene.py`, etc.) follow similar stubs with INPUT\\\_TYPES, RETURN\\\_TYPES, FUNCTION, CATEGORY, and placeholder methods.



---



\## Config \& Requirements



``



```

torch==2.5.\*

torchvision==0.20.\*

numpy>=1.26

pillow>=10.4

trimesh>=4.4

pygltflib>=1.16

scipy>=1.13

opencv-python>=4.10

pyyaml>=6.0

```



``



```json

{

&nbsp; "model\_paths": {

&nbsp;   "pano\_text": "D:/models/HYW/HunyuanWorld-PanoDiT-Text.safetensors",

&nbsp;   "pano\_image": "D:/models/HYW/HunyuanWorld-PanoDiT-Image.safetensors",

&nbsp;   "inpaint\_scene": "D:/models/HYW/HunyuanWorld-PanoInpaint-Scene.safetensors",

&nbsp;   "inpaint\_sky": "D:/models/HYW/HunyuanWorld-PanoInpaint-Sky.safetensors",

&nbsp;   "recon": "D:/models/HYW/recon.weights"

&nbsp; },

&nbsp; "device": "cuda:0",

&nbsp; "dtype": "fp16",

&nbsp; "defaults": { "pano\_size": \[2048,1024], "sampler": "euler", "cfg": 6.5, "steps": 30 },

&nbsp; "draco": { "enabled": false, "qp": 20 }

}

```



---



\## Example Workflows



\- ``: CLIPTextEncode → HYW\\\_ModelLoader → HYW\\\_PanoGen → (opt) HYW\\\_PanoInpaint\\\_Sky → HYW\\\_SeamlessWrap360 → SaveImage

\- ``: LoadImage → HYW\\\_ModelLoader → HYW\\\_PanoGen(cond\\\_image) → (opt) HYW\\\_PanoInpaint\\\_Scene → HYW\\\_WorldReconstructor → HYW\\\_TextureBaker → HYW\\\_MeshExport

\- ``: CLIPTextEncode → HYW\\\_ModelLoader → HYW\\\_PanoGen(4K) → HYW\\\_WorldReconstructor(quality=high) → HYW\\\_TextureBaker → HYW\\\_MeshExport(glb)



---



\## Golden Assets



``



```

A serene lakeside village at sunset, panoramic view

```



``: Reference panorama image for golden test.



``



```json

{

&nbsp; "vertices": 195000,

&nbsp; "faces": 390000,

&nbsp; "materials": 5

}

```



---



\## README.md Outline



\- Overview

\- Requirements

\- Installation

\- Model download \& placement

\- Running example workflows

\- Testing with golden assets



---



This skeleton matches the SDD and can be dropped into `ComfyUI/custom\_nodes/` to start development immediately.



---



\# Appendix A — Repo Skeleton \& Code Stubs (Drop‑in)



> Copy this folder into `ComfyUI/custom\_nodes/ComfyUI\_HunyuanWorld/`.



```

ComfyUI\_HunyuanWorld/

├─ \_\_init\_\_.py

├─ settings.json            # optional, user‑editable defaults

├─ requirements.txt

├─ nodes/

│  ├─ \_runtime.py           # adapter layer (HYWRuntime)

│  ├─ hyw\_loader.py         # HYW\_ModelLoader, HYW\_Config

│  ├─ hyw\_panogen.py        # HYW\_PanoGen

│  ├─ hyw\_inpaint\_scene.py  # HYW\_PanoInpaint\_Scene

│  ├─ hyw\_inpaint\_sky.py    # HYW\_PanoInpaint\_Sky

│  ├─ hyw\_reconstruct.py    # HYW\_WorldReconstructor

│  ├─ hyw\_export.py         # HYW\_TextureBaker, HYW\_MeshExport, HYW\_Thumbnailer

│  └─ hyw\_utils.py          # SeamlessWrap360, hashing, io helpers

├─ example\_workflows/

│  ├─ A\_text\_to\_pano.json

│  ├─ B\_image\_to\_mesh.json

│  ├─ C\_text\_to\_mesh\_high.json

│  └─ golden/

│     ├─ golden\_prompt.txt

│     ├─ golden\_ref.jpg

│     └─ expected\_mesh.json

└─ README.md

```



\## A.1 `\_\_init\_\_.py`



```python

from .nodes.hyw\_loader import HYW\_ModelLoader, HYW\_Config

from .nodes.hyw\_panogen import HYW\_PanoGen

from .nodes.hyw\_inpaint\_scene import HYW\_PanoInpaint\_Scene

from .nodes.hyw\_inpaint\_sky import HYW\_PanoInpaint\_Sky

from .nodes.hyw\_reconstruct import HYW\_WorldReconstructor

from .nodes.hyw\_export import HYW\_TextureBaker, HYW\_MeshExport, HYW\_Thumbnailer

from .nodes.hyw\_utils import HYW\_SeamlessWrap360



NODE\_CLASS\_MAPPINGS = {

&nbsp;   "HYW\_ModelLoader": HYW\_ModelLoader,

&nbsp;   "HYW\_Config": HYW\_Config,

&nbsp;   "HYW\_PanoGen": HYW\_PanoGen,

&nbsp;   "HYW\_PanoInpaint\_Scene": HYW\_PanoInpaint\_Scene,

&nbsp;   "HYW\_PanoInpaint\_Sky": HYW\_PanoInpaint\_Sky,

&nbsp;   "HYW\_WorldReconstructor": HYW\_WorldReconstructor,

&nbsp;   "HYW\_TextureBaker": HYW\_TextureBaker,

&nbsp;   "HYW\_MeshExport": HYW\_MeshExport,

&nbsp;   "HYW\_Thumbnailer": HYW\_Thumbnailer,

&nbsp;   "HYW\_SeamlessWrap360": HYW\_SeamlessWrap360,

}



NODE\_DISPLAY\_NAME\_MAPPINGS = {

&nbsp;   "HYW\_ModelLoader": "HunyuanWorld — Model Loader",

&nbsp;   "HYW\_Config": "HunyuanWorld — Config",

&nbsp;   "HYW\_PanoGen": "HunyuanWorld — Panorama (Text/Image)",

&nbsp;   "HYW\_PanoInpaint\_Scene": "HunyuanWorld — Pano Inpaint (Scene)",

&nbsp;   "HYW\_PanoInpaint\_Sky": "HunyuanWorld — Pano Inpaint (Sky)",

&nbsp;   "HYW\_WorldReconstructor": "HunyuanWorld — World Reconstruct",

&nbsp;   "HYW\_TextureBaker": "HunyuanWorld — Texture Baker",

&nbsp;   "HYW\_MeshExport": "HunyuanWorld — Mesh Export",

&nbsp;   "HYW\_Thumbnailer": "HunyuanWorld — Thumbnailer",

&nbsp;   "HYW\_SeamlessWrap360": "HunyuanWorld — Seamless Wrap 360",

}

```



\## A.2 `nodes/\_runtime.py` (adapter skeleton)



```python

from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple



@dataclass

class RuntimeConfig:

&nbsp;   device: str = "cuda:0"

&nbsp;   dtype: str = "fp16"  # fp32|fp16|bf16

&nbsp;   paths: Dict\[str, str] = None  # pano\_text, pano\_image, inpaint\_scene, inpaint\_sky, recon



class HYWRuntime:

&nbsp;   def \_\_init\_\_(self, cfg: RuntimeConfig):

&nbsp;       self.cfg = cfg

&nbsp;       self.\_panogen = {}

&nbsp;       self.\_recon = None

&nbsp;       # TODO: initialize torch device/dtype, xformers flags



&nbsp;   def load\_panogen(self, kind: str):

&nbsp;       # TODO: lazy load FLUX/PanoDiT variant based on kind

&nbsp;       self.\_panogen\[kind] = object()



&nbsp;   def load\_recon(self):

&nbsp;       # TODO: lazy load recon module(s)

&nbsp;       self.\_recon = object()



&nbsp;   def pano\_generate(self, \*\*kw):

&nbsp;       # TODO: call upstream model; return (image, meta)

&nbsp;       return None, {"meta": "stub"}



&nbsp;   def pano\_inpaint(self, \*\*kw):

&nbsp;       # TODO: inpainting pass for scene/sky

&nbsp;       return None



&nbsp;   def reconstruct(self, \*\*kw):

&nbsp;       # TODO: panorama -> mesh, instances, materials

&nbsp;       return None, {}, {}

```



\## A.3 `nodes/hyw\_loader.py`



```python

import os, json, hashlib

from typing import Dict, Tuple

from .\_runtime import HYWRuntime, RuntimeConfig



class HYW\_ModelLoader:

&nbsp;   CATEGORY = "HunyuanWorld/Loaders"

&nbsp;   RETURN\_TYPES = ("MODEL",)

&nbsp;   RETURN\_NAMES = ("hyw\_runtime",)

&nbsp;   FUNCTION = "run"



&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {

&nbsp;           "base\_dir": ("STRING", {"default":""}),

&nbsp;           "weights\_panotext": ("STRING", {"default":""}),

&nbsp;           "weights\_panoimage": ("STRING", {"default":""}),

&nbsp;           "weights\_pano\_inpaint\_scene": ("STRING", {"default":""}),

&nbsp;           "weights\_pano\_inpaint\_sky": ("STRING", {"default":""}),

&nbsp;           "weights\_recon": ("STRING", {"default":""}),

&nbsp;           "device": ("STRING", {"default":"cuda:0"}),

&nbsp;           "dtype": ("STRING", {"default":"fp16"}),

&nbsp;       }, "optional": {

&nbsp;           "enable\_xformers": ("BOOLEAN", {"default": True}),

&nbsp;           "sha256\_json": ("STRING", {"multiline": True, "default":""}),

&nbsp;       }}



&nbsp;   @staticmethod

&nbsp;   def \_check\_sha256(path: str, expected: str) -> bool:

&nbsp;       if not expected: return True

&nbsp;       h = hashlib.sha256()

&nbsp;       with open(path, 'rb') as f:

&nbsp;           for chunk in iter(lambda: f.read(1<<20), b""):

&nbsp;               h.update(chunk)

&nbsp;       return h.hexdigest().lower() == expected.lower()



&nbsp;   def run(self, base\_dir, weights\_panotext, weights\_panoimage,

&nbsp;           weights\_pano\_inpaint\_scene, weights\_pano\_inpaint\_sky,

&nbsp;           weights\_recon, device, dtype, enable\_xformers=True, sha256\_json=""):

&nbsp;       paths = {

&nbsp;           "pano\_text": os.path.join(base\_dir, weights\_panotext) if base\_dir else weights\_panotext,

&nbsp;           "pano\_image": os.path.join(base\_dir, weights\_panoimage) if base\_dir else weights\_panoimage,

&nbsp;           "inpaint\_scene": os.path.join(base\_dir, weights\_pano\_inpaint\_scene) if base\_dir else weights\_pano\_inpaint\_scene,

&nbsp;           "inpaint\_sky": os.path.join(base\_dir, weights\_pano\_inpaint\_sky) if base\_dir else weights\_pano\_inpaint\_sky,

&nbsp;           "recon": os.path.join(base\_dir, weights\_recon) if base\_dir else weights\_recon,

&nbsp;       }

&nbsp;       for k, p in paths.items():

&nbsp;           if not p or not os.path.exists(p):

&nbsp;               raise RuntimeError(f"E\_MISSING\_WEIGHT:{k}:{p}")

&nbsp;       if sha256\_json:

&nbsp;           expected = json.loads(sha256\_json)

&nbsp;           for k, p in paths.items():

&nbsp;               if k in expected and not self.\_check\_sha256(p, expected\[k]):

&nbsp;                   raise RuntimeError(f"E\_BAD\_VERSION:{k}:{p}")

&nbsp;       cfg = RuntimeConfig(device=device, dtype=dtype, paths=paths)

&nbsp;       rt = HYWRuntime(cfg)

&nbsp;       return (rt,)



class HYW\_Config:

&nbsp;   CATEGORY = "HunyuanWorld/Loaders"

&nbsp;   RETURN\_TYPES = ("DICT",)

&nbsp;   RETURN\_NAMES = ("hyw\_cfg",)

&nbsp;   FUNCTION = "run"



&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {

&nbsp;           "pano\_width": ("INT", {"default":2048, "min":256, "max":16384, "step":256}),

&nbsp;           "pano\_height": ("INT", {"default":1024, "min":128, "max":8192, "step":128}),

&nbsp;           "cfg\_scale": ("FLOAT", {"default":6.5, "min":0.0, "max":20.0, "step":0.1}),

&nbsp;           "steps": ("INT", {"default":30, "min":1, "max":200}),

&nbsp;       }, "optional": {

&nbsp;           "sampler\_name": ("STRING", {"default":"euler"}),

&nbsp;       }}



&nbsp;   def run(self, pano\_width, pano\_height, cfg\_scale, steps, sampler\_name="euler"):

&nbsp;       return ({

&nbsp;           "pano\_size": \[int(pano\_width), int(pano\_height)],

&nbsp;           "cfg": float(cfg\_scale),

&nbsp;           "steps": int(steps),

&nbsp;           "sampler": sampler\_name,

&nbsp;       },)

```



\## A.4 `nodes/hyw\_panogen.py`



```python

class HYW\_PanoGen:

&nbsp;   CATEGORY = "HunyuanWorld/Generate"

&nbsp;   RETURN\_TYPES = ("IMAGE","DICT")

&nbsp;   RETURN\_NAMES = ("panorama","meta")

&nbsp;   FUNCTION = "run"



&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {

&nbsp;           "hyw\_runtime": ("MODEL",),

&nbsp;           "prompt": ("STRING", {"multiline": True, "default": ""}),

&nbsp;           "neg\_prompt": ("STRING", {"multiline": True, "default": ""}),

&nbsp;           "seed": ("INT", {"default": 0, "min": 0, "max": 2\*\*31-1}),

&nbsp;           "steps": ("INT", {"default": 30, "min": 1, "max": 200}),

&nbsp;           "cfg\_scale": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 20.0}),

&nbsp;           "width": ("INT", {"default": 2048, "min": 256, "max": 16384, "step": 256}),

&nbsp;           "height": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 128}),

&nbsp;           "sampler\_name": ("STRING", {"default": "euler"}),

&nbsp;           "seamless\_wrap": ("BOOLEAN", {"default": True}),

&nbsp;           "tiling\_mode": ("STRING", {"default": "latent"}),

&nbsp;           "tile\_w": ("INT", {"default": 1024}),

&nbsp;           "tile\_h": ("INT", {"default": 512}),

&nbsp;           "overlap": ("INT", {"default": 64}),

&nbsp;       }, "optional": {

&nbsp;           "cond\_image": ("IMAGE",),

&nbsp;       }}



&nbsp;   def run(self, hyw\_runtime, prompt, neg\_prompt, seed, steps, cfg\_scale, width, height, sampler\_name,

&nbsp;           seamless\_wrap, tiling\_mode, tile\_w, tile\_h, overlap, cond\_image=None):

&nbsp;       tiling = {"mode": tiling\_mode, "tile\_w": tile\_w, "tile\_h": tile\_h, "overlap": overlap}

&nbsp;       image, meta = hyw\_runtime.pano\_generate(kind=("image" if cond\_image else "text"),

&nbsp;                                               prompt=prompt, neg=neg\_prompt, cond\_image=cond\_image,

&nbsp;                                               seed=seed, steps=steps, cfg=cfg\_scale,

&nbsp;                                               size=(width, height), sampler=sampler\_name, tiling=tiling)

&nbsp;       # TODO: if seamless\_wrap, apply utility blend

&nbsp;       return image, meta

```



\## A.5 `nodes/hyw\_inpaint\_scene.py` / `nodes/hyw\_inpaint\_sky.py`



```python

class HYW\_PanoInpaint\_Scene:

&nbsp;   CATEGORY = "HunyuanWorld/Generate"

&nbsp;   RETURN\_TYPES = ("IMAGE",)

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {"image": ("IMAGE",), "strength": ("FLOAT", {"default": 0.7}),

&nbsp;                             "seed": ("INT", {"default": 0}), "steps": ("INT", {"default": 20}),

&nbsp;                             "cfg\_scale": ("FLOAT", {"default": 6.0}),

&nbsp;                             "hyw\_runtime": ("MODEL",)},

&nbsp;               "optional": {"mask": ("IMAGE",)}}

&nbsp;   def run(self, image, strength, seed, steps, cfg\_scale, hyw\_runtime, mask=None):

&nbsp;       out = hyw\_runtime.pano\_inpaint(which="scene", image=image, mask=mask,

&nbsp;                                      strength=strength, seed=seed, steps=steps, cfg=cfg\_scale)

&nbsp;       return (out,)



class HYW\_PanoInpaint\_Sky(HYW\_PanoInpaint\_Scene):

&nbsp;   def run(self, image, strength, seed, steps, cfg\_scale, hyw\_runtime, mask=None):

&nbsp;       out = hyw\_runtime.pano\_inpaint(which="sky", image=image, mask=mask,

&nbsp;                                      strength=strength, seed=seed, steps=steps, cfg=cfg\_scale)

&nbsp;       return (out,)

```



\## A.6 `nodes/hyw\_reconstruct.py`



```python

class HYW\_WorldReconstructor:

&nbsp;   CATEGORY = "HunyuanWorld/Reconstruct"

&nbsp;   RETURN\_TYPES = ("MESH","DICT","DICT")

&nbsp;   RETURN\_NAMES = ("world","instances","materials")

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {

&nbsp;           "hyw\_runtime": ("MODEL",),

&nbsp;           "panorama": ("IMAGE",),

&nbsp;           "quality": ("STRING", {"default": "preview"}),

&nbsp;           "max\_tris": ("INT", {"default": 200\_000}),

&nbsp;           "semantic\_strength": ("FLOAT", {"default": 0.5}),

&nbsp;           "classes": ("STRING", {"default": "outdoor"}),

&nbsp;       }, "optional": {

&nbsp;           "labels\_fg1": ("STRING", {"multiline": True, "default": ""}),

&nbsp;           "labels\_fg2": ("STRING", {"multiline": True, "default": ""}),

&nbsp;           "layout\_guides": ("DICT", {}),

&nbsp;       }}

&nbsp;   def run(self, hyw\_runtime, panorama, quality, max\_tris, semantic\_strength, classes, labels\_fg1="", labels\_fg2="", layout\_guides=None):

&nbsp;       fg1 = \[x.strip() for x in labels\_fg1.split(",") if x.strip()]

&nbsp;       fg2 = \[x.strip() for x in labels\_fg2.split(",") if x.strip()]

&nbsp;       world, instances, materials = hyw\_runtime.reconstruct(panorama=panorama, quality=quality, max\_tris=max\_tris,

&nbsp;                                                             labels\_fg1=fg1, labels\_fg2=fg2, classes=classes,

&nbsp;                                                             layout\_guides=layout\_guides or {})

&nbsp;       return world, instances, materials

```



\## A.7 `nodes/hyw\_export.py`



```python

import os, json



class HYW\_TextureBaker:

&nbsp;   CATEGORY = "HunyuanWorld/Export"

&nbsp;   RETURN\_TYPES = ("DICT",)

&nbsp;   RETURN\_NAMES = ("baked\_textures",)

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {"world": ("MESH",), "materials": ("DICT",), "map\_list": ("STRING", {"default": "albedo,normal,roughness,ao"})}}

&nbsp;   def run(self, world, materials, map\_list):

&nbsp;       # TODO: bake into UV space

&nbsp;       return ({},)



class HYW\_MeshExport:

&nbsp;   CATEGORY = "HunyuanWorld/Export"

&nbsp;   RETURN\_TYPES = ("FILE","LIST")

&nbsp;   RETURN\_NAMES = ("mesh\_file","texture\_files")

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {

&nbsp;           "world": ("MESH",),

&nbsp;           "baked\_textures": ("DICT",),

&nbsp;           "scene\_graph": ("DICT",),

&nbsp;           "format": ("STRING", {"default": "glb"}),

&nbsp;           "out\_dir": ("STRING", {"default": "./outputs/hyw"}),

&nbsp;           "use\_draco": ("BOOLEAN", {"default": False}),

&nbsp;           "draco\_qp": ("INT", {"default": 20}),

&nbsp;       }}

&nbsp;   def run(self, world, baked\_textures, scene\_graph, format, out\_dir, use\_draco, draco\_qp):

&nbsp;       os.makedirs(out\_dir, exist\_ok=True)

&nbsp;       mesh\_path = os.path.join(out\_dir, f"world.{format}")

&nbsp;       # TODO: serialize mesh+textures; write manifest

&nbsp;       manifest = {"format": format, "use\_draco": bool(use\_draco), "draco\_qp": int(draco\_qp)}

&nbsp;       with open(os.path.join(out\_dir, "manifest.json"), "w", encoding="utf-8") as f:

&nbsp;           json.dump(manifest, f, indent=2)

&nbsp;       return (mesh\_path, \[])



class HYW\_Thumbnailer:

&nbsp;   CATEGORY = "HunyuanWorld/Export"

&nbsp;   RETURN\_TYPES = ("IMAGE",)

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {"panorama": ("IMAGE",)}}

&nbsp;   def run(self, panorama):

&nbsp;       # TODO: downscale or simple orbit render

&nbsp;       return (panorama,)

```



\## A.8 `nodes/hyw\_utils.py`



```python

import hashlib, json



class HYW\_SeamlessWrap360:

&nbsp;   CATEGORY = "HunyuanWorld/Utils"

&nbsp;   RETURN\_TYPES = ("IMAGE",)

&nbsp;   FUNCTION = "run"

&nbsp;   @classmethod

&nbsp;   def INPUT\_TYPES(cls):

&nbsp;       return {"required": {"image": ("IMAGE",), "mode": ("STRING", {"default": "blend"}),

&nbsp;                              "blend\_width": ("INT", {"default": 32})}}

&nbsp;   def run(self, image, mode, blend\_width):

&nbsp;       # TODO: circular shift + feather blend across seam

&nbsp;       return (image,)



def graph\_hash(meta: dict) -> str:

&nbsp;   s = json.dumps(meta, sort\_keys=True).encode("utf-8")

&nbsp;   return hashlib.sha256(s).hexdigest()

```



\## A.9 `requirements.txt`



```

torch==2.5.\*

torchvision==0.20.\*

numpy>=1.26

pillow>=10.4

trimesh>=4.4

pygltflib>=1.16

scipy>=1.13

opencv-python>=4.10

pyyaml>=6.0

```



\## A.10 `settings.json`



```json

{

&nbsp; "model\_paths": {

&nbsp;   "pano\_text": "D:/models/HYW/HunyuanWorld-PanoDiT-Text.safetensors",

&nbsp;   "pano\_image": "D:/models/HYW/HunyuanWorld-PanoDiT-Image.safetensors",

&nbsp;   "inpaint\_scene": "D:/models/HYW/HunyuanWorld-PanoInpaint-Scene.safetensors",

&nbsp;   "inpaint\_sky": "D:/models/HYW/HunyuanWorld-PanoInpaint-Sky.safetensors",

&nbsp;   "recon": "D:/models/HYW/recon.weights"

&nbsp; },

&nbsp; "device": "cuda:0",

&nbsp; "dtype": "fp16",

&nbsp; "defaults": { "pano\_size": \[2048, 1024], "sampler": "euler", "cfg": 6.5, "steps": 30 },

&nbsp; "draco": { "enabled": false, "qp": 20 }

}

```



\## A.11 Example Workflows (placeholders)



```json

// A\_text\_to\_pano.json

{"workflow": "CLIPTextEncode -> HYW\_ModelLoader -> HYW\_PanoGen -> HYW\_SeamlessWrap360 -> SaveImage"}

```



```json

// B\_image\_to\_mesh.json

{"workflow": "LoadImage -> HYW\_ModelLoader -> HYW\_PanoGen(cond\_image) -> HYW\_WorldReconstructor -> HYW\_TextureBaker -> HYW\_MeshExport"}

```



```json

// C\_text\_to\_mesh\_high.json

{"workflow": "CLIPTextEncode -> HYW\_ModelLoader -> HYW\_PanoGen(4K) -> HYW\_WorldReconstructor(high) -> HYW\_TextureBaker -> HYW\_MeshExport(glb,draco=false)"}

```



\## A.12 Golden Assets



\- `golden/golden\_prompt.txt` — one prompt used for verification

\- `golden/golden\_ref.jpg` — small 1k equirect image (user‑supplied)

\- `golden/expected\_mesh.json` — stats like `{ "verts": 185234, "faces": 370000, "materials": 3 }`



\*Note:\* All node bodies are \*\*stubs\*\*—they compile and wire correctly in ComfyUI. Replace `TODO` sections by calling into your local HunyuanWorld implementation via `\_runtime.py`.



When testing always test with the embedded Comfyui version of python at this path. "C:\\ComfyUI\\.venv\\Scripts\\python.exe"



Push to GitHub every time you modify code.



Update the readme to reflect the latest changes. remove outdeated info from the readme.



Assume loras are here for Hunyuan\_World C:\\ComfyUI\\models\\Hunyuan\_World



Assume flux models are here: C:\\ComfyUI\\models\\unet

