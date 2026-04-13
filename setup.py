"""
Hunyuan3D-Paint -- extension setup script.

Zero local compilation. All C++ extensions are installed from prebuilt wheels
hosted on GitHub Releases / wheels/ branch.

Called by Modly at extension install time:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
"""
import io
import json
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GITHUB_ZIP   = "https://github.com/deepbeepmeep/Hunyuan3D-2GP/archive/refs/heads/main.zip"
_REPO         = "Lorchie/modly-hunyuan3d-paint-extension"
_RELEASES_URL = f"https://github.com/{_REPO}/releases/latest/download"
_WHEELS_RAW   = f"https://raw.githubusercontent.com/{_REPO}/main/wheels"

# ---------------------------------------------------------------------------
# CUDA / Torch resolver
# ---------------------------------------------------------------------------

def resolve_cuda(gpu_sm: int) -> tuple:
    """
    Map GPU SM version to (cu_tag, torch_ver, index_url, torch_pkgs).

    gpu_sm >= 100  -> CUDA 12.8  (Blackwell RTX 50xx)
    gpu_sm >= 70   -> CUDA 12.4  (Ampere / Ada)
    gpu_sm <  70   -> CUDA 11.8  (legacy Turing / Pascal)
    gpu_sm == 0    -> CPU fallback
    """
    if gpu_sm >= 100:
        return (
            "cu128", "2.7.0",
            "https://download.pytorch.org/whl/cu128",
            ["torch==2.7.0", "torchvision==0.22.0"],
        )
    if gpu_sm >= 70:
        return (
            "cu124", "2.6.0",
            "https://download.pytorch.org/whl/cu124",
            ["torch==2.6.0", "torchvision==0.21.0"],
        )
    # SM < 70 (Pascal/Volta — GTX 10xx, Titan V): too old, CPU fallback.
    return (
        "cpu", "2.6.0",
        "https://download.pytorch.org/whl/cpu",
        ["torch==2.6.0", "torchvision==0.21.0"],
    )

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def _is_win() -> bool:
    return platform.system() == "Windows"

def _pip(venv: Path) -> Path:
    return venv / ("Scripts/pip.exe" if _is_win() else "bin/pip")

def _exe(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if _is_win() else "bin/python")

def pip(venv: Path, *args: str) -> None:
    subprocess.run([str(_pip(venv)), *args], check=True)

def platform_tag() -> str:
    if _is_win():
        return "win_amd64"
    machine = platform.machine().lower()
    return f"linux_{machine}"

def python_tag(venv: Path) -> str:
    return subprocess.check_output(
        [str(_exe(venv)), "-c",
         "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()

# ---------------------------------------------------------------------------
# Wheel URL builder
# ---------------------------------------------------------------------------

def wheel_candidates(module: str, py_tag: str, plat: str,
                     cu_tag: str, torch_ver: str) -> list:
    """
    Return candidate wheel URLs in priority order:
      1. GitHub Release asset (stable, permanent)
      2. Raw main/wheels/ (rolling, updated by CI)

    Naming convention:
      mesh_processor    : mesh_processor-0.0.0-{py}-{py}-{plat}.whl
      custom_rasterizer : custom_rasterizer-0.1.0+torch{ver}.{cu}-{py}-{py}-{plat}.whl
                          custom_rasterizer-0.1-{py}-{py}-{plat}.whl  (generic fallback)
    """
    torch_label = f"torch{torch_ver.replace('.', '')}.{cu_tag}"

    if module == "mesh_processor":
        names = [f"mesh_processor-0.0.0-{py_tag}-{py_tag}-{plat}.whl"]
    elif module == "custom_rasterizer":
        names = [
            f"custom_rasterizer-0.1.0+{torch_label}-{py_tag}-{py_tag}-{plat}.whl",
            f"custom_rasterizer-0.1-{py_tag}-{py_tag}-{plat}.whl",
        ]
    elif module == "hy3dgen":
        names = ["hy3dgen-2.0.0-py3-none-any.whl"]
    else:
        return []

    urls = []
    for name in names:
        urls.append(f"{_RELEASES_URL}/{name}")
        urls.append(f"{_WHEELS_RAW}/{name}")
    return urls

# ---------------------------------------------------------------------------
# Wheel installer
# ---------------------------------------------------------------------------

def _url_exists(url: str, timeout: int = 10) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False

def install_wheel(venv: Path, module: str, candidates: list) -> bool:
    """
    Try each candidate URL in order until one installs successfully.
    Returns True on success, False if all candidates failed / not found.
    """
    pip_exe = _pip(venv)
    for url in candidates:
        filename = url.split("/")[-1]
        print(f"[setup]   Checking {filename} ...")
        if not _url_exists(url):
            continue
        print(f"[setup]   Installing {filename} ...")
        result = subprocess.run(
            [str(pip_exe), "install", url],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[setup]   OK: {filename}")
            return True
        print(f"[setup]   Failed: {result.stderr.strip()[:300]}")
    return False

# ---------------------------------------------------------------------------
# hy3dgen source installer
# ---------------------------------------------------------------------------

def install_hy3dgen(venv: Path, ext_dir: Path) -> None:
    """
    Install hy3dgen Python package into the venv.
    Priority: prebuilt wheel > vendor/ > GitHub source download.
    """
    pip_exe = _pip(venv)

    # Check if already installed
    result = subprocess.run(
        [str(pip_exe), "show", "hy3dgen"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("[setup] hy3dgen already installed, skipping.")
        return

    # Try prebuilt wheel first
    candidates = wheel_candidates("hy3dgen", "py3", "none-any", "", "")
    print("[setup] Installing hy3dgen from prebuilt wheel ...")
    if install_wheel(venv, "hy3dgen", candidates):
        return

    # Fallback: vendor/ directory
    vendor_src = ext_dir / "vendor" / "hy3dgen"
    if vendor_src.exists():
        print(f"[setup] Installing hy3dgen from vendor/ ...")
        subprocess.run([str(pip_exe), "install", str(vendor_src)], check=True)
        return

    # Last resort: download source from GitHub and install
    print("[setup] Downloading hy3dgen source from GitHub...")
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=300) as resp:
        data = resp.read()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prefix = "Hunyuan3D-2GP-main/hy3dgen/"
        strip  = "Hunyuan3D-2GP-main/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix) and member != f"{strip}setup.py":
                    continue
                rel    = member[len(strip):]
                target = tmp / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        subprocess.run([str(pip_exe), "install", str(tmp)], check=True)

    print("[setup] hy3dgen installed from source.")

# ---------------------------------------------------------------------------
# C++ extension wheel installer (zero compilation)
# ---------------------------------------------------------------------------

def install_extensions(venv: Path, py_tag: str, plat: str,
                        cu_tag: str, torch_ver: str) -> None:
    """
    Install prebuilt C++ extension wheels. No compilation whatsoever.
    Raises RuntimeError if no compatible wheel is available.
    """
    for module in ("mesh_processor", "custom_rasterizer"):
        candidates = wheel_candidates(module, py_tag, plat, cu_tag, torch_ver)
        print(f"[setup] Installing {module} from prebuilt wheel ...")
        if install_wheel(venv, module, candidates):
            continue

        raise RuntimeError(
            f"\nNo compatible prebuilt wheel found for '{module}'.\n"
            f"  Python  : {py_tag}\n"
            f"  Platform: {plat}\n"
            f"  CUDA    : {cu_tag}\n\n"
            f"To fix:\n"
            f"  1. Go to the extension repo: https://github.com/{_REPO}\n"
            f"  2. Trigger Actions > 'Build Wheels' (workflow_dispatch)\n"
            f"  3. Reinstall this extension once the workflow completes.\n"
        )

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ── Resolve CUDA / Torch config ────────────────────────────────────────
    cu_tag, torch_ver, torch_index, torch_pkgs = resolve_cuda(gpu_sm)
    print(f"[setup] GPU SM {gpu_sm} -> {cu_tag}, PyTorch {torch_ver}")

    # ── Install PyTorch (GPU, with CPU fallback) ───────────────────────────
    print(f"[setup] Installing PyTorch {torch_ver} ({cu_tag}) ...")
    try:
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    except subprocess.CalledProcessError:
        print("[setup] GPU PyTorch install failed -> falling back to CPU build.")
        pip(venv, "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu")
        cu_tag = "cpu"

    # ── Core runtime dependencies ──────────────────────────────────────────
    print("[setup] Installing core dependencies ...")
    pip(venv, "install",
        "Pillow",
        "numpy",
        "trimesh",
        "huggingface_hub",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate",
        "einops",
        "scipy",
        "opencv-python-headless",
        "xatlas",
        "pygltflib",
    )

    # ── rembg + onnxruntime ────────────────────────────────────────────────
    # onnxruntime-gpu must match the CUDA version used by torch.
    # cu128 requires onnxruntime-gpu >= 1.21 (first release with CUDA 12.x support).
    # cu124 / cu118 work with onnxruntime-gpu >= 1.17.
    print("[setup] Installing rembg + onnxruntime ...")
    if gpu_sm >= 100:
        ort_pkg = "onnxruntime-gpu>=1.21"
    elif gpu_sm >= 70:
        ort_pkg = "onnxruntime-gpu>=1.17"
    else:
        ort_pkg = "onnxruntime"

    pip(venv, "install", "rembg", ort_pkg)

    # ── hy3dgen source ─────────────────────────────────────────────────────
    install_hy3dgen(venv, ext_dir)

    # ── C++ extensions (prebuilt wheels only, zero compilation) ───────────
    if cu_tag != "cpu":
        py_tag = python_tag(venv)
        plat   = platform_tag()
        install_extensions(venv, py_tag, plat, cu_tag, torch_ver)
    else:
        print("[setup] CPU mode: skipping C++ GPU extensions.")

    print()
    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 0)),
            int(args.get("cuda_version", 0)),
        )
    elif len(sys.argv) >= 4:
        setup(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Usage: python setup.py '{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":86}'")
        sys.exit(1)
