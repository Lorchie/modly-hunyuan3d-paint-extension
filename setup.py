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

_GITHUB_ZIP   = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
# Prebuilt wheels hosted in this repo's wheels/ directory (raw GitHub URLs).
# Format: {module: [(label, url_template), ...]}
# {pyver} is replaced with e.g. "311", {platform} with "win_amd64" or "linux_x86_64".
_WHEELS_BASE  = "https://raw.githubusercontent.com/Lorchie/modly-hunyuan3d-paint-extension/main/wheels"
_WHEEL_SPECS  = {
    # pybind11 C++, no CUDA — one wheel per Python version / platform
    "mesh_processor": [
        ("mesh_inpaint_processor-0.0.0-cp{pyver}-cp{pyver}-{platform}.whl", None),
    ],
    # CUDA extension — prefer the torch+cuda-tagged wheel, fall back to generic
    "custom_rasterizer": [
        ("custom_rasterizer-0.1.0+{torch_label}-cp{pyver}-cp{pyver}-{platform}.whl", "{torch_label}"),
        ("custom_rasterizer-0.1-cp{pyver}-cp{pyver}-{platform}.whl", None),
    ],
}


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
# Prebuilt wheel installer
# ---------------------------------------------------------------------------

def _wheel_platform() -> str:
    """Return the platform tag used in wheel filenames."""
    if platform.system() == "Windows":
        return "win_amd64"
    machine = platform.machine().lower()
    return f"linux_{machine}" if machine == "x86_64" else f"linux_{machine}"


def _torch_label(torch_ver: str, cuda_ver: int) -> str:
    """e.g. torch_ver='2.7.0', cuda_ver=128 -> 'torch270.cuda128'"""
    v = torch_ver.replace(".", "")
    return f"torch{v}.cuda{cuda_ver}"


def _try_install_prebuilt(venv: Path, module: str, pyver: str,
                           plat: str, torch_label: str) -> bool:
    """
    Attempt to install a prebuilt wheel for `module` from the wheels/ directory.
    Returns True if successfully installed, False otherwise.
    """
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    specs = _WHEEL_SPECS.get(module, [])

    for template, needs_label in specs:
        if needs_label and not torch_label:
            continue
        filename = template.format(
            pyver=pyver,
            platform=plat,
            torch_label=torch_label,
        )
        url = f"{_WHEELS_BASE}/{filename}"
        print(f"[setup] Trying prebuilt wheel: {filename} ...")
        try:
            # HEAD request to check existence before downloading
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            print(f"[setup]   Not found, skipping.")
            continue

        try:
            subprocess.run(
                [str(pip_exe), "install", url],
                check=True,
            )
            print(f"[setup] Installed prebuilt wheel: {filename}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[setup]   Install failed (exit {exc.returncode}), will compile from source.")
            return False

    return False


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


def _msvc_env(arch: str = "x64") -> dict:
    """
    Return an environment dict with MSVC build tools activated.
    Finds Visual Studio via vswhere.exe and runs vcvarsall.bat.
    Returns os.environ.copy() unchanged if MSVC is already active or not found.
    """
    import shutil

    if platform.system() != "Windows":
        return os.environ.copy()

    # Already active if cl.exe is on PATH
    if shutil.which("cl.exe"):
        print("[setup] MSVC already active on PATH.")
        return os.environ.copy()

    # Locate vswhere.exe (ships with every VS / Build Tools installer)
    prog86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(prog86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        print("[setup] WARNING: vswhere.exe not found — MSVC Build Tools may be missing.")
        return os.environ.copy()

    # Ask vswhere for the latest install that has the C++ x64 tools
    try:
        install_path = subprocess.check_output(
            [
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualCpp.Tools.HostX64.TargetX64",
                "-property", "installationPath",
            ],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        print(f"[setup] WARNING: vswhere query failed: {exc}")
        return os.environ.copy()

    if not install_path:
        print("[setup] WARNING: No Visual Studio installation with C++ tools found.")
        return os.environ.copy()

    vcvarsall = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.exists():
        print(f"[setup] WARNING: vcvarsall.bat not found at {vcvarsall}")
        return os.environ.copy()

    # Run vcvarsall and capture the resulting environment
    try:
        out = subprocess.check_output(
            f'"{vcvarsall}" {arch} && set',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[setup] WARNING: vcvarsall.bat failed: {exc}")
        return os.environ.copy()

    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    print(f"[setup] MSVC activated via {vcvarsall} ({arch})")
    return env


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
    # Activate MSVC so cl.exe is available on Windows
    build_env = {**_msvc_env(), "TORCH_CUDA_ARCH_LIST": arch}
    if is_win:
        build_env["DISTUTILS_USE_SDK"] = "1"

    # Metadata for prebuilt wheel selection
    pyver      = subprocess.check_output(
        [str(exe), "-c", "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()
    plat       = _wheel_platform()
    torch_ver  = subprocess.check_output(
        [str(exe), "-c", "import torch; print(torch.__version__.split('+')[0])"],
        text=True, stderr=subprocess.DEVNULL,
    ).strip() if _module_importable(venv, "torch") else ""
    tlabel     = _torch_label(torch_ver, cuda_version) if torch_ver and cuda_version else ""

    # (ext_dir_name, importable_module_name)
    # differentiable_renderer -> mesh_processor (pybind11, no CUDA)
    # custom_rasterizer       -> custom_rasterizer (CUDA + PyTorch)
    extensions = [
        ("differentiable_renderer", "mesh_processor"),
        ("custom_rasterizer",       "custom_rasterizer"),
    ]
    for ext_name, module_name in extensions:
        ext_path = texgen / ext_name

        # Skip if already importable
        if _module_importable(venv, module_name):
            print(f"[setup] {module_name} already installed, skipping.")
            continue

        # 1. Try prebuilt wheel first (fast, no compiler needed)
        if _try_install_prebuilt(venv, module_name, pyver, plat, tlabel):
            continue

        # 2. Fall back to source compilation
        if not ext_path.exists():
            print(f"[setup] {ext_name} source not found at {ext_path}, skipping.")
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
