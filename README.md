# modly-hunyuan3d-paint-extension

Modly extension to apply PBR textures to an existing `.obj` or `.glb` mesh
from a reference image, using **Hunyuan3D-Paint** by Tencent — 100% local.

| Variant | Model | VRAM | Speed (RTX 4070) |
|---|---|---|---|
| **Turbo** (default) | `hunyuan3d-paint-v2-0-turbo` | ~8 GB | ~40–60 s |
| **Standard** | `hunyuan3d-paint-v2-0` | ~8 GB | ~90–120 s |

---

## Requirements

- GPU with ≥ 8 GB VRAM (NVIDIA, CUDA 11.8 / 12.4 / 12.8)
- Modly ≥ 0.x
- Python ≥ 3.10 (provided by Modly)
- MSVC (Windows) or GCC + CUDA toolkit (Linux) for one-time C++ compilation

---

## Installation

### 1 — Install the extension in Modly

> **Extensions → Install from file** → select the `.zip` from the
> [latest release](https://github.com/Lorchie/modly-hunyuan3d-paint-extension/releases/latest).

Modly runs `setup.py` automatically to create a venv and install Python deps.

### 2 — Build vendor/ (one-time, optional but recommended)

Vendoring hy3dgen avoids a ~350 MB download at first use:

```bash
# Run with the app's embedded Python
python build_vendor.py
```

### 3 — Compile C++ extensions (one-time, required)

The texture renderer requires two compiled extensions. After step 2:

**Windows (MSVC + CUDA toolkit)**
```bat
cd vendor\hy3dgen\texgen\custom_rasterizer
python setup.py install

cd ..\differentiable_renderer
python setup.py install
```

**Linux (GCC + CUDA toolkit)**
```bash
cd vendor/hy3dgen/texgen/custom_rasterizer
python setup.py install

cd ../differentiable_renderer
python setup.py install
```

> If you skipped step 2, replace `vendor/` with
> `<model_dir>/_hy3dgen/hy3dgen/texgen/` (path shown in the error dialog).

---

## Usage

1. In Modly, select **Hunyuan3D Paint — Turbo** (or Standard) in the
   model selector.
2. Connect or drag-and-drop a `.glb` / `.obj` mesh as the **primary input**.
3. In the **Reference Image Path** parameter, paste the absolute path to
   your reference image (`.png`, `.jpg`, `.webp`).
4. Adjust optional parameters and click **Generate**.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| Reference Image Path | string | — | Absolute path to the reference image |
| Texture Resolution | select | 1024 | Atlas resolution: 512 / 1024 / 2048 |
| Inference Steps | select | 30 | 15 = Fast · 30 = Balanced · 50 = High Quality |
| Guidance Scale | float | 5.5 | Conditioning strength (1.0–10.0) |
| Seed | int | -1 | -1 = random |

> **Note on Inference Steps / Guidance Scale** — `Hunyuan3DPaintPipeline`
> exposes only `(mesh, image)` at its public API level. These params are
> forwarded as config hints and will take effect when hy3dgen exposes them
> at the outer pipeline level. Seed is applied globally via
> `torch.manual_seed()` and does affect the diffusion sampling.

---

## BLOC D — Input schema note

During development, the Modly TypeScript API was inspected and confirmed:

- `ExtensionNode.input` is a **singleton string** (`'image' | 'text' | 'mesh'`)
  → **CAS (a)** applies.
- `ParamSchema.type` supports `'select' | 'int' | 'float' | 'string'`
  — no `'file'` type exists.

Architecture consequence:
- Each node declares `"input": "mesh"` — the mesh bytes arrive as the
  `image_bytes` argument of `generate()`.
- The reference image is passed via `params["image_path"]` (type `"string"`).

---

## Model weights

Weights are downloaded on first use from
[tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) on
Hugging Face Hub.

| Variant | Downloaded folders |
|---|---|
| Turbo | `hunyuan3d-paint-v2-0-turbo/` + `hunyuan3d-delight-v2-0/` |
| Standard | `hunyuan3d-paint-v2-0/` + `hunyuan3d-delight-v2-0/` |

Shape generation folders (`hunyuan3d-dit-*`, `hunyuan3d-vae-*`) are skipped.

---

## License

MIT — see [LICENSE](LICENSE).
Model weights are subject to the
[Tencent Hunyuan3D Community License](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE).
