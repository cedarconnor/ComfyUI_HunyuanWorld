from .nodes.hyw_loader import HYW_ModelLoader, HYW_Config, HYW_SettingsLoader
from .nodes.hyw_converter import HYW_RuntimeFromStock
from .nodes.hyw_panogen import HYW_PanoGen, HYW_PanoGenBatch
from .nodes.hyw_flux_helpers import HYW_PerspectiveToPanoramaMask, HYW_ShiftPanorama
from .nodes.hyw_inpaint_scene import HYW_PanoInpaint_Scene, HYW_PanoInpaint_Advanced
from .nodes.hyw_inpaint_sky import HYW_PanoInpaint_Sky, HYW_SkyMaskGenerator
from .nodes.hyw_reconstruct import HYW_WorldReconstructor, HYW_MeshProcessor, HYW_MeshAnalyzer
from .nodes.hyw_export import HYW_TextureBaker, HYW_MeshExport, HYW_Thumbnailer
from .nodes.hyw_utils import HYW_SeamlessWrap360, HYW_PanoramaValidator, HYW_MetadataManager

NODE_CLASS_MAPPINGS = {
    "HYW_ModelLoader": HYW_ModelLoader,
    "HYW_Config": HYW_Config,
    "HYW_SettingsLoader": HYW_SettingsLoader,
    "HYW_RuntimeFromStock": HYW_RuntimeFromStock,
    "HYW_PanoGen": HYW_PanoGen,
    "HYW_PanoGenBatch": HYW_PanoGenBatch,
    "HYW_PerspectiveToPanoramaMask": HYW_PerspectiveToPanoramaMask,
    "HYW_ShiftPanorama": HYW_ShiftPanorama,
    "HYW_PanoInpaint_Scene": HYW_PanoInpaint_Scene,
    "HYW_PanoInpaint_Advanced": HYW_PanoInpaint_Advanced,
    "HYW_PanoInpaint_Sky": HYW_PanoInpaint_Sky,
    "HYW_SkyMaskGenerator": HYW_SkyMaskGenerator,
    "HYW_WorldReconstructor": HYW_WorldReconstructor,
    "HYW_MeshProcessor": HYW_MeshProcessor,
    "HYW_MeshAnalyzer": HYW_MeshAnalyzer,
    "HYW_TextureBaker": HYW_TextureBaker,
    "HYW_MeshExport": HYW_MeshExport,
    "HYW_Thumbnailer": HYW_Thumbnailer,
    "HYW_SeamlessWrap360": HYW_SeamlessWrap360,
    "HYW_PanoramaValidator": HYW_PanoramaValidator,
    "HYW_MetadataManager": HYW_MetadataManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYW_ModelLoader": "HunyuanWorld — Model Loader",
    "HYW_Config": "HunyuanWorld — Config",
    "HYW_SettingsLoader": "HunyuanWorld — Settings Loader",
    "HYW_RuntimeFromStock": "HunyuanWorld — Runtime From Stock",
    "HYW_PanoGen": "HunyuanWorld — Panorama (Text/Image)",
    "HYW_PanoGenBatch": "HunyuanWorld — Panorama Batch",
    "HYW_PerspectiveToPanoramaMask": "HunyuanWorld — Perspective→Pano + Mask",
    "HYW_ShiftPanorama": "HunyuanWorld — Shift Panorama",
    "HYW_PanoInpaint_Scene": "HunyuanWorld — Pano Inpaint (Scene)",
    "HYW_PanoInpaint_Advanced": "HunyuanWorld — Pano Inpaint (Advanced)",
    "HYW_PanoInpaint_Sky": "HunyuanWorld — Pano Inpaint (Sky)",
    "HYW_SkyMaskGenerator": "HunyuanWorld — Sky Mask Generator",
    "HYW_WorldReconstructor": "HunyuanWorld — World Reconstruct",
    "HYW_MeshProcessor": "HunyuanWorld — Mesh Processor",
    "HYW_MeshAnalyzer": "HunyuanWorld — Mesh Analyzer",
    "HYW_TextureBaker": "HunyuanWorld — Texture Baker", 
    "HYW_MeshExport": "HunyuanWorld — Mesh Export",
    "HYW_Thumbnailer": "HunyuanWorld — Thumbnailer",
    "HYW_SeamlessWrap360": "HunyuanWorld — Seamless Wrap 360",
    "HYW_PanoramaValidator": "HunyuanWorld — Panorama Validator",
    "HYW_MetadataManager": "HunyuanWorld — Metadata Manager",
}
