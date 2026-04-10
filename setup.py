"""
Hunyuan3D-Paint -- extension setup script.

Creates an isolated venv, installs all required dependencies, and compiles
the C++ hy3dgen extensions (custom_rasterizer + differentiable_renderer)
automatically for the detected GPU architecture.

Called by Modly at extension install time with:

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

_GITHUB_ZIP = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def python_tag(venv: Path) -> str:
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")
    return subprocess.check_output(
        [str(exe), "-c",
         "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def install_hy3dgen(venv: Path, ext_dir: Path) -> None:
    """
    Copy hy3dgen source from vendor/ (if present) or download it from GitHub,
    then place it in the venv's site-packages.
    """
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")

    site_packages = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    dest = Path(site_packages) / "hy3dgen"

    if dest.exists():
        print("[setup] hy3dgen already in site-packages, skipping.")
        return

    # Prefer vendor/ if build_vendor.py was already run
    vendor_src = ext_dir / "vendor" / "hy3dgen"
    if vendor_src.exists():
        print(f"[setup] Copying hy3dgen from vendor/ -> {dest}")
        import shutil
        shutil.copytree(str(vendor_src), str(dest))
        print("[setup] hy3dgen copied from vendor/.")
        return

    # Fallback: download from GitHub
    print("[setup] Downloading hy3dgen source from GitHub...")
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
        data = resp.read()

    prefix = "Hunyuan3D-2-main/hy3dgen/"
    strip  = "Hunyuan3D-2-main/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = Path(site_packages) / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    print(f"[setup] hy3dgen installed to {site_packages}.")


# ---------------------------------------------------------------------------
# C++ extension compiler
# ---------------------------------------------------------------------------

def _sm_to_arch(gpu_sm: int) -> str:
    """Convert SM integer (e.g. 120) to TORCH_CUDA_ARCH_LIST string (e.g. '12.0')."""
    major = gpu_sm // 10
    minor = gpu_sm % 10
    return f"{major}.{minor}"


def _module_importable(venv: Path, module: str) -> bool:
    """Return True if the module can be imported inside the venv."""
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")
    result = subprocess.run(
        [str(exe), "-c", f"import {module}"],
        capture_output=True,
    )
    return result.returncode == 0


def compile_texgen(venv: Path, ext_dir: Path, gpu_sm: int) -> None:
    """
    Compile custom_rasterizer and differentiable_renderer for the target GPU.
    Raises RuntimeError if compilation fails so the install surfaces the error.
    """
    is_win = platform.system() == "Windows"
    exe    = venv / ("Scripts/python.exe" if is_win else "bin/python")

    # Resolve texgen directory: prefer vendor/, fall back to site-packages
    texgen = ext_dir / "vendor" / "hy3dgen" / "texgen"
    if not texgen.exists():
        try:
            site = subprocess.check_output(
                [str(exe), "-c",
                 "import site; print([p for p in site.getsitepackages()"
                 " if 'site-packages' in p][0])"],
                text=True,
            ).strip()
            texgen = Path(site) / "hy3dgen" / "texgen"
        except Exception as exc:
            print(f"[setup] Could not locate hy3dgen/texgen: {exc}")
            return

    if not texgen.exists():
        print(f"[setup] texgen directory not found at {texgen}, skipping C++ build.")
        return

    arch      = _sm_to_arch(gpu_sm) if gpu_sm > 0 else "8.6"
    build_env = {**os.environ, "TORCH_CUDA_ARCH_LIST": arch}
    if is_win:
        build_env["DISTUTILS_USE_SDK"] = "1"

    # (ext_dir_name, importable_module_name)
    # differentiable_renderer compiles `mesh_processor` (pybind11)
    # custom_rasterizer compiles `custom_rasterizer` package + CUDA kernel
    extensions = [
        ("differentiable_renderer", "mesh_processor"),
        ("custom_rasterizer",       "custom_rasterizer"),
    ]
    for ext_name, module_name in extensions:
        ext_path = texgen / ext_name
        if not ext_path.exists():
            print(f"[setup] {ext_name} not found at {ext_path}, skipping.")
            continue

        # Skip if the compiled module is already importable in the venv
        if _module_importable(venv, module_name):
            print(f"[setup] {module_name} already compiled, skipping.")
            continue

        print(f"[setup] Compiling {ext_name} for sm_{gpu_sm} (arch={arch}) ...")
        result = subprocess.run(
            [str(exe), "setup.py", "install"],
            cwd=str(ext_path),
            env=build_env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"{ext_name} compilation failed (exit {result.returncode}).\n"
                f"Ensure CUDA Toolkit and MSVC Build Tools are installed.\n"
                f"Source directory: {ext_path}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        print(f"[setup] {ext_name} compiled successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # PyTorch selection
    if gpu_sm >= 100 or cuda_version >= 128:
        torch_ver   = "2.7.0"
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)")
    elif gpu_sm == 0 or gpu_sm >= 70:
        torch_ver   = "2.6.0"
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
    else:
        torch_ver   = "2.5.1"
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # Core runtime dependencies
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
        # Required to compile differentiable_renderer (pybind11 C++ extension)
        "pybind11>=2.6.0",
        "ninja",
    )

    # rembg
    print("[setup] Installing rembg ...")
    if gpu_sm >= 70:
        pip(venv, "install", "rembg[gpu]")
    else:
        pip(venv, "install", "rembg", "onnxruntime")

    # hy3dgen source
    install_hy3dgen(venv, ext_dir)

    # C++ extensions
    compile_texgen(venv, ext_dir, gpu_sm)

    print()
    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            args["python_exe"],
            Path(args["ext_dir"]),
            int(args.get("gpu_sm", 86)),
            int(args.get("cuda_version", 0)),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        sys.exit(1)
