"""
Build the vendor/ directory for the Hunyuan3D-Paint extension.

Run this script once (with the app's venv active) to populate vendor/.
The resulting vendor/ folder should be committed to the extension repository
so end users never need to download hy3dgen source at runtime.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)
    - Internet access to reach github.com

Note: The C++ texture-generation extensions (custom_rasterizer and
differentiable_renderer) cannot be pre-built for all platforms and CUDA
versions. After running this script, compile them manually with the app's
Python:

    cd vendor/hy3dgen/texgen/custom_rasterizer
    python setup.py install

    cd vendor/hy3dgen/texgen/differentiable_renderer
    python setup.py install

These compiled .pyd / .so files must be installed into the app's Python
environment (site-packages). They cannot be committed to vendor/.
"""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

VENDOR      = Path(__file__).parent / "vendor"
GITHUB_ZIP  = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"

# Pure-Python packages to vendor (no compilation needed).
# torch and torchvision are already provided by the host app - do NOT vendor them.
PURE_PACKAGES = [
    "trimesh",
    "huggingface_hub",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target."""
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_hy3dgen(dest: Path) -> None:
    """
    Download Hunyuan3D-2 source from GitHub and extract the hy3dgen/ package
    into vendor/. Only the pure-Python portions are extracted; the C++
    extensions (custom_rasterizer, differentiable_renderer) must be compiled
    separately - see the module docstring above.
    """
    import urllib.request

    hy3dgen_dest = dest / "hy3dgen"
    if hy3dgen_dest.exists():
        print("  hy3dgen/ already present in vendor/, skipping.")
        return

    print("  Downloading Hunyuan3D-2 source from GitHub...")
    with urllib.request.urlopen(GITHUB_ZIP, timeout=180) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data) // 1024 // 1024} MB.")

    prefix = "Hunyuan3D-2-main/hy3dgen/"
    strip  = "Hunyuan3D-2-main/"

    print("  Extracting hy3dgen/...")
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

    print(f"  hy3dgen/ extracted to {dest}.")
    print()
    print("  IMPORTANT - C++ extensions still need manual compilation:")
    print(f"    cd \"{dest / 'hy3dgen' / 'texgen' / 'custom_rasterizer'}\"")
    print(f"    python setup.py install")
    print()
    print(f"    cd \"{dest / 'hy3dgen' / 'texgen' / 'differentiable_renderer'}\"")
    print(f"    python setup.py install")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    for pkg in PURE_PACKAGES:
        print(f"\n[1] Vendoring {pkg}...")
        vendor_pure_package(pkg, VENDOR)

    # 2. hy3dgen source
    print("\n[2] Vendoring hy3dgen source...")
    vendor_hy3dgen(VENDOR)

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("Remember to compile the C++ extensions before first use.")


if __name__ == "__main__":
    main()
