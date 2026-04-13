"""
Download all prebuilt wheels from the latest 'Build Wheels' CI run.

Usage:
    python get_wheels.py              # download to ./wheels-download/
    python get_wheels.py --out dist/  # custom output folder

Requirements:
    gh CLI installed and authenticated (gh auth login)
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "Lorchie/modly-hunyuan3d-paint-extension"
WORKFLOW = "build-wheels.yml"
ARTIFACT = "all-wheels"


def run(cmd: list) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="wheels-download", help="Output folder")
    parser.add_argument("--run-id", default=None, help="Specific workflow run ID (default: latest)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Find the latest successful run
    if args.run_id:
        run_id = args.run_id
    else:
        print(f"[get_wheels] Finding latest successful run of {WORKFLOW} ...")
        run_id = run([
            "gh", "run", "list",
            "--repo", REPO,
            "--workflow", WORKFLOW,
            "--status", "success",
            "--limit", "1",
            "--json", "databaseId",
            "--jq", ".[0].databaseId",
        ])
        if not run_id:
            print("No successful run found. Trigger the workflow first:", file=sys.stderr)
            print(f"  gh workflow run {WORKFLOW} --repo {REPO}", file=sys.stderr)
            sys.exit(1)

    print(f"[get_wheels] Run ID: {run_id}")

    # Download the all-wheels artifact
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    print(f"[get_wheels] Downloading '{ARTIFACT}' artifact ...")
    run([
        "gh", "run", "download", run_id,
        "--repo", REPO,
        "--name", ARTIFACT,
        "--dir", str(tmp),
    ])

    # Unzip
    zips = list(tmp.glob("*.zip"))
    if not zips:
        print("No zip found in artifact.", file=sys.stderr)
        sys.exit(1)

    print(f"[get_wheels] Extracting {zips[0].name} ...")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(out)

    shutil.rmtree(tmp)

    wheels = sorted(out.glob("*.whl"))
    print(f"\n=== {len(wheels)} wheels in {out.resolve()} ===")
    for w in wheels:
        print(f"  {w.name}")

    print("\nTo upload to a GitHub Release:")
    print(f"  gh release upload <tag> {out}/*.whl --repo {REPO}")


if __name__ == "__main__":
    main()
