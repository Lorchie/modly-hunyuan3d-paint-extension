"""
Hunyuan3D-Paint extension for Modly.

Applies PBR textures to an existing .obj or .glb mesh using a reference
image. 100% local inference via Hunyuan3D-Paint (Tencent).

Reference : https://huggingface.co/tencent/Hunyuan3D-2
GitHub    : https://github.com/Tencent/Hunyuan3D-2

hy3dgen source is either loaded from vendor/ (if build_vendor.py was run)
or downloaded at first load from the Tencent GitHub repository.

The C++ texture-generation extensions (custom_rasterizer, differentiable_renderer)
are compiled automatically by setup.py at install time. If they are missing,
click "Repair" on the Models page to re-run setup.py.
"""
import io
import os
import random
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent
_GITHUB_ZIP    = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"


class Hunyuan3DPaintGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-paint"
    DISPLAY_NAME = "Hunyuan3D Paint"
    VRAM_GB      = 8

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check  # "hunyuan3d-paint-v2-0-turbo" or "hunyuan3d-paint-v2-0"
        paint_dir = self.model_dir / subfolder
        return paint_dir.exists() and any(paint_dir.glob("*.safetensors"))

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_vendor()
        self._ensure_hy3dgen()
        self._check_texgen_extensions()

        import torch
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        subfolder = self.download_check
        print(f"[Hunyuan3DPaintGenerator] Loading pipeline ({subfolder})...")
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
        )
        self._model = pipeline
        print("[Hunyuan3DPaintGenerator] Loaded.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        image_bytes : raw bytes of the reference image (.png / .jpg / .webp).
        params      : {
            "mesh_path"          : str   - absolute or workspace-relative path to .glb / .obj
            "texture_resolution" : int   - 512 / 1024 / 2048
            "num_inference_steps": int   - 15 / 30 / 50
            "guidance_scale"     : float - 1.0-10.0
            "seed"               : int   - -1 for random
          }
        """
        import torch
        import trimesh

        num_steps      = int(params.get("num_inference_steps", 30))
        guidance_scale = float(params.get("guidance_scale", 5.5))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2 ** 31 - 1)

        # 3%: Load mesh
        self._report(progress_cb, 3, "Loading mesh...")

        mesh_path = params.get("mesh_path", "").strip()
        if not mesh_path:
            raise ValueError("No mesh provided. Connect a mesh node to the mesh input.")

        resolved = Path(mesh_path)
        if not resolved.is_absolute():
            workspace = Path(os.environ.get("WORKSPACE_DIR", str(Path.home() / ".modly" / "workspace")))
            resolved = workspace / mesh_path

        if not resolved.exists():
            raise ValueError(f"Mesh file not found: {resolved}")

        loaded = trimesh.load(str(resolved), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError("The mesh file contains no geometry.")
            mesh = trimesh.util.concatenate(geometries)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded)}")

        self._check_cancelled(cancel_event)

        # 10%: Preprocess reference image
        self._report(progress_cb, 10, "Preprocessing reference image...")
        image = self._preprocess(image_bytes)

        self._check_cancelled(cancel_event)

        # 20%: Load model
        self._report(progress_cb, 20, "Loading paint model...")
        self.load()
        self._check_cancelled(cancel_event)

        # 25%: Configure renderer
        self._report(progress_cb, 25, "Configuring renderer...")

        res = int(params.get("texture_resolution", 1024))
        self._model.config.render_size  = res
        self._model.config.texture_size = res
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        self._model.render = MeshRender(default_resolution=res, texture_size=res)

        # Best-effort: expose num_inference_steps / guidance_scale on config
        # so that a future hy3dgen version can pick them up without a code change.
        self._model.config.num_inference_steps = num_steps
        self._model.config.guidance_scale      = guidance_scale

        # Seed global RNG (affects diffusion sampling inside HunyuanPaintPipeline)
        torch.manual_seed(seed)
        random.seed(seed)

        self._check_cancelled(cancel_event)

        # 30-92%: Generate textures
        self._report(progress_cb, 30, "Generating textures...")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 92, "Generating textures...", stop_evt),
                daemon=True,
            )
            t.start()

        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_img.close()  # close before save — required on Windows
        try:
            image.save(tmp_img.name)
            # Hunyuan3DPaintPipeline.__call__(mesh, image) - no extra kwargs.
            # guidance_scale / num_inference_steps are set via config above.
            with torch.no_grad():
                result = self._model(mesh, image=tmp_img.name)
            textured = result[0] if isinstance(result, (list, tuple)) else result
        finally:
            stop_evt.set()
            os.unlink(tmp_img.name)

        self._check_cancelled(cancel_event)

        # 95%: Export GLB
        self._report(progress_cb, 95, "Exporting GLB...")

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_textured.glb"
        path = self.outputs_dir / name
        textured.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        """Remove background and return RGBA image."""
        import rembg

        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            # cuDNN/CUDA incompatibility - fall back to CPU
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _setup_vendor(self) -> None:
        """Add vendor/ to sys.path so hy3dgen is importable without installation."""
        vendor_dir = _EXTENSION_DIR / "vendor"
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

    def _ensure_hy3dgen(self) -> None:
        """
        Ensure hy3dgen is importable.
        Order: vendor/ -> downloaded copy in model_dir/_hy3dgen.
        """
        try:
            import hy3dgen  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            import hy3dgen  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\nOriginal error: {exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        """Download Hunyuan3D-2 source from GitHub and extract hy3dgen/ to dest."""
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3DPaintGenerator] Downloading hy3dgen source from GitHub...")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print("[Hunyuan3DPaintGenerator] Extracting hy3dgen...")

        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[Hunyuan3DPaintGenerator] hy3dgen extracted to {dest}.")

    def _check_texgen_extensions(self) -> None:
        """
        Validate that the C++ texture-generation extensions are compiled.
        Raises RuntimeError with build instructions if they are missing.
        """
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
            return
        except (ImportError, OSError) as exc:
            vendor   = _EXTENSION_DIR / "vendor" / "hy3dgen" / "texgen"
            fallback = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"
            base     = vendor if vendor.exists() else fallback
            raise RuntimeError(
                "C++ extensions for texture generation are not compiled.\n"
                "Build them once with the app's Python:\n\n"
                f"  cd \"{base / 'custom_rasterizer'}\"\n"
                f"  python setup.py install\n\n"
                f"  cd \"{base / 'differentiable_renderer'}\"\n"
                f"  python setup.py install\n\n"
                f"Original error: {exc}"
            ) from exc
