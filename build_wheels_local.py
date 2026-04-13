"""
Build prebuilt wheels locally (simulates the GitHub Actions CI).

Usage:
    python build_wheels_local.py
    python build_wheels_local.py --python C:/path/to/python.exe

Outputs wheels into: ./wheels/
"""
import argparse
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_GITHUB_ZIP = "https://github.com/deepbeepmeep/Hunyuan3D-2GP/archive/refs/heads/main.zip"

EXTENSIONS = {
    "differentiable_renderer": {
        "module":  "mesh_processor",
        "prefix":  "Hunyuan3D-2GP-main/hy3dgen/texgen/differentiable_renderer/",
        "needs_cuda": False,
    },
    "custom_rasterizer": {
        "module":  "custom_rasterizer",
        "prefix":  "Hunyuan3D-2GP-main/hy3dgen/texgen/custom_rasterizer/",
        "needs_cuda": True,
    },
}

STRIP = "Hunyuan3D-2GP-main/"


# ---------------------------------------------------------------------------

def download_hy3dgen(dest: Path) -> None:
    """Download Hunyuan3D-2GP zip and extract extension sources + full hy3dgen package."""
    if dest.exists():
        print(f"[local] Source already extracted at {dest}, skipping download.")
        return
    print(f"[local] Downloading Hunyuan3D-2GP source from GitHub ...")
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=300) as r:
        data = r.read()
    print("[local] Extracting ...")
    prefixes = [e["prefix"] for e in EXTENSIONS.values()] + [
        f"{STRIP}hy3dgen/",
        f"{STRIP}setup.py",
    ]
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for m in zf.namelist():
            if not any(m.startswith(p) for p in prefixes):
                continue
            rel = Path(m[len(STRIP):])
            if m.endswith("/"):
                (dest / rel).mkdir(parents=True, exist_ok=True)
            else:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_bytes(zf.read(m))
    print("[local] Done.")


def patch_sources(src_root: Path) -> None:
    """
    Apply Windows-specific patches to the extracted source files.

    differentiable_renderer/setup.py: add /MANIFEST:NO to link_args to avoid
    LNK1158 (link.exe failing to spawn rc.exe for manifest embedding).
    """
    if platform.system() != "Windows":
        return

    dr_setup = src_root / "hy3dgen" / "texgen" / "differentiable_renderer" / "setup.py"
    if not dr_setup.exists():
        return

    content = dr_setup.read_text(encoding="utf-8")
    old = "        link_args = []\n        extra_includes = []"
    new = "        link_args = ['/MANIFEST:NO']  # avoid LNK1158 (rc.exe)\n        extra_includes = []"
    if old in content:
        dr_setup.write_text(content.replace(old, new), encoding="utf-8")
        print("[local] Patched differentiable_renderer/setup.py (/MANIFEST:NO)")


def get_torch_info(python: str) -> tuple:
    """Return (torch_ver, cu_tag) from the given Python executable."""
    try:
        out = subprocess.check_output(
            [python, "-c",
             "import torch; v=torch.__version__; "
             "cuda=torch.version.cuda or ''; "
             "print(v.split('+')[0], cuda.replace('.',''))"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split()
        torch_ver = out[0]
        cuda_raw  = out[1] if len(out) > 1 else ""
        if cuda_raw:
            cu_tag = f"cu{cuda_raw[:3]}"
        else:
            cu_tag = "cpu"
        return torch_ver, cu_tag
    except Exception as exc:
        print(f"[local] WARNING: could not detect torch version: {exc}")
        return "unknown", "cpu"


def get_python_tag(python: str) -> str:
    return subprocess.check_output(
        [python, "-c",
         "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def get_platform_tag(python: str) -> str:
    return subprocess.check_output(
        [python, "-c",
         "import sysconfig; print(sysconfig.get_platform().replace('-','_').replace('.','_'))"],
        text=True,
    ).strip()


def _find_vcvarsall() -> str | None:
    """Return path to vcvarsall.bat, or None if not found."""
    if platform.system() != "Windows":
        return None
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        return None
    try:
        vs_path = subprocess.check_output(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualCpp.Tools.HostX86.TargetX64",
             "-property", "installationPath"],
            text=True,
        ).strip()
    except Exception:
        return None
    if not vs_path:
        return None
    bat = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    return str(bat) if bat.exists() else None


def _find_winsdk_x64_bin() -> str | None:
    """Return the x64 bin dir of the latest installed Windows 10/11 SDK."""
    winsdk = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
    if not winsdk.exists():
        return None
    sdk_dirs = sorted(winsdk.glob("10.0.*"), reverse=True)
    for d in sdk_dirs:
        if (d / "x64" / "rc.exe").exists():
            return str(d / "x64")
    return None


def _find_cuda_extra_includes(cuda_home: str) -> list[str]:
    """
    Return extra INCLUDE paths needed to compile CUDA extensions.

    Some CUDA toolkit installs (e.g. network-mode v12.8) ship an incomplete
    include/ missing cuda_runtime_api.h and thrust/. Fall back to the latest
    other installed version that has the full headers.

    Also adds include/cccl when present (CUDA 12+ moved thrust/cub/libcudacxx
    into include/cccl/, so 'thrust/complex.h' lives at include/cccl/thrust/).

    These paths are for cl.exe (INCLUDE env only) — nvcc runs from cuda_home
    so the torch version-check passes.
    """
    cuda_inc = Path(cuda_home) / "include"
    paths: list[str] = []

    def _collect(inc: Path) -> None:
        paths.append(str(inc))
        cccl = inc / "cccl"
        if cccl.exists():
            paths.append(str(cccl))

    if (cuda_inc / "cuda_runtime_api.h").exists():
        _collect(cuda_inc)
        return paths

    print(f"[local] WARNING: cuda_runtime_api.h not found in {cuda_inc}")
    print("[local] Searching other CUDA installs for complete headers...")
    toolkit_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit_root.exists():
        for d in sorted(toolkit_root.iterdir(), reverse=True):
            candidate = d / "include" / "cuda_runtime_api.h"
            if candidate.exists():
                print(f"[local] Using CUDA headers from: {d / 'include'}")
                _collect(d / "include")
                return paths

    print("[local] WARNING: no complete CUDA headers found — build may fail.")
    _collect(cuda_inc)
    return paths


def build_wheel(python: str, src_path: Path, out_dir: Path,
                vcvarsall: str | None = None,
                cuda_home: str | None = None) -> Path:
    """Build a wheel and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pip_cmd = subprocess.list2cmdline(
        [python, "-m", "pip", "wheel", str(src_path),
         "--no-deps", "--no-build-isolation", "-w", str(out_dir)]
    )

    if vcvarsall and platform.system() == "Windows":
        # Use a temp .bat file so vcvarsall env changes persist to pip wheel.
        winsdk_bin = _find_winsdk_x64_bin() or ""
        lines = [
            "@echo off",
            f'call "{vcvarsall}" amd64',
            "set DISTUTILS_USE_SDK=1",
            "set MSSdk=1",
            # CCCL headers (CUDA 13+) require the conforming MSVC preprocessor.
            # Injecting via CL env var propagates to every cl.exe invocation in this session.
            'set "CL=/Zc:preprocessor %CL%"',
        ]
        if cuda_home:
            cuda_home_w  = str(Path(cuda_home))
            cuda_bin     = str(Path(cuda_home) / "bin")
            extra_incs   = _find_cuda_extra_includes(cuda_home)
            cuda_lib     = str(Path(cuda_home) / "lib" / "x64")
            lines.append(f'set "CUDA_HOME={cuda_home_w}"')
            lines.append(f'set "CUDA_PATH={cuda_home_w}"')
            lines.append(f'set "PATH={cuda_bin};%PATH%"')
            # INCLUDE: cl.exe finds CUDA headers (+ cccl/thrust) even if cuda_home is incomplete
            inc_prepend = ";".join(extra_incs)
            lines.append(f'set "INCLUDE={inc_prepend};%INCLUDE%"')
            lines.append(f'set "LIB={cuda_lib};%LIB%"')
        if winsdk_bin:
            lines.append(f'set "PATH={winsdk_bin};%PATH%"')
        lines.append(pip_cmd)

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bat", delete=False
        ) as f:
            f.write(("\r\n".join(lines) + "\r\n").encode("cp1252"))
            bat_path = f.name

        try:
            subprocess.run(["cmd", "/c", bat_path], check=True)
        finally:
            Path(bat_path).unlink(missing_ok=True)
    else:
        env = os.environ.copy()
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["CUDA_PATH"] = cuda_home
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
        subprocess.run(pip_cmd, shell=True, check=True, env=env)

    wheels = list(out_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel produced in {out_dir}")
    return wheels[0]


def build_hy3dgen_wheel(python: str, src_root: Path, out_dir: Path) -> Path:
    """Build the pure Python hy3dgen wheel from the extracted GP source root."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python, "-m", "pip", "wheel", str(src_root), "--no-deps", "-w", str(out_dir)],
        check=True,
    )
    wheels = sorted(out_dir.glob("hy3dgen-*.whl"))
    if not wheels:
        raise RuntimeError(f"No hy3dgen wheel produced in {out_dir}")
    return wheels[-1]


def tag_custom_rasterizer(whl: Path, torch_ver: str, cu_tag: str, out_dir: Path) -> Path:
    """Rename wheel to include torch+cuda version tag."""
    tv = torch_ver.replace(".", "")
    label = f"torch{tv}.{cu_tag}"
    tagged = re.sub(
        r"^(custom_rasterizer-[^-]+)(-)",
        lambda m: f"{m.group(1)}.0+{label}{m.group(2)}",
        whl.name,
        count=1,
    )
    dest = out_dir / tagged
    shutil.copy(whl, dest)
    whl.unlink()
    return dest


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build hy3dgen C++ wheels locally.")
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--source-dir", default="hy3dgen_src",
        help="Directory to extract hy3dgen source into (default: hy3dgen_src)",
    )
    parser.add_argument(
        "--out-dir", default="wheels",
        help="Output directory for wheels (default: wheels/)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading hy3dgen source (assume already extracted)",
    )
    parser.add_argument(
        "--cuda-home", default=None,
        help="Path to CUDA toolkit (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8)",
    )
    args = parser.parse_args()

    python    = args.python
    src_root  = Path(args.source_dir)
    out_dir   = Path(args.out_dir)
    cuda_home = args.cuda_home

    print(f"[local] Python      : {python}")
    py_tag, plat_tag = get_python_tag(python), get_platform_tag(python)
    torch_ver, cu_tag = get_torch_info(python)
    print(f"[local] Python tag  : {py_tag}")
    print(f"[local] Platform    : {plat_tag}")
    print(f"[local] Torch       : {torch_ver}")
    print(f"[local] CUDA tag    : {cu_tag}")
    print()

    # Ensure build tools are available in the target Python
    subprocess.run(
        [python, "-m", "pip", "install", "pybind11>=2.6.0", "setuptools", "wheel"],
        check=True, stdout=subprocess.DEVNULL,
    )

    # Locate MSVC on Windows
    vcvarsall = _find_vcvarsall()
    if platform.system() == "Windows":
        if vcvarsall:
            print(f"[local] MSVC         : {vcvarsall}")
        else:
            print("[local] WARNING: MSVC not found — C++ build may fail.")
    if cuda_home:
        print(f"[local] CUDA_HOME    : {cuda_home}")
    print()

    # Download / extract sources, then apply Windows patches
    if not args.skip_download:
        download_hy3dgen(src_root)
    patch_sources(src_root)

    # Build each extension
    for ext_name, cfg in EXTENSIONS.items():
        src_path = src_root / "hy3dgen" / "texgen" / ext_name
        if not src_path.exists():
            print(f"[local] Source not found: {src_path} — skipping.")
            continue

        if cfg["needs_cuda"] and cu_tag == "cpu":
            print(f"[local] {ext_name}: CUDA not available, skipping.")
            continue

        print(f"[local] Building {ext_name} ({cfg['module']}) ...")
        raw_dir   = out_dir / "_raw"
        _cuda     = cuda_home if cfg["needs_cuda"] else None
        whl       = build_wheel(python, src_path, raw_dir, vcvarsall, _cuda)
        print(f"[local]   Built: {whl.name}")

        if ext_name == "custom_rasterizer":
            whl = tag_custom_rasterizer(whl, torch_ver, cu_tag, raw_dir)
            print(f"[local]   Tagged: {whl.name}")

        final = out_dir / whl.name
        shutil.copy(whl, final)
        print(f"[local]   -> {final}")

    # Build hy3dgen pure Python wheel
    print("[local] Building hy3dgen (pure Python) ...")
    raw_dir = out_dir / "_raw"
    hy_whl = build_hy3dgen_wheel(python, src_root, raw_dir)
    final_hy = out_dir / hy_whl.name
    shutil.copy(hy_whl, final_hy)
    print(f"[local]   -> {final_hy}")

    # Cleanup raw dir
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    print()
    print("=== Wheels built ===")
    for w in sorted(out_dir.glob("*.whl")):
        print(f"  {w.name}")
    print(f"\nOutput: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
