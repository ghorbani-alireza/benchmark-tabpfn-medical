"""Optional: recreate the TabPFN source copies already included in vendor/."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


REPO_URL = "https://github.com/PriorLabs/TabPFN.git"
VERSIONS = [
    ("v6.0.0", "tabpfn_v25"),
    ("v7.1.1", "tabpfn_v26"),
    ("v8.5.0", "tabpfn_v3"),
    ("v9.0.0", "tabpfn_v35"),
]


def setup_vendor(main_path=None, force=False):
    """Download the specified versions and copy them into this project's vendor/."""
    root = Path(main_path) if main_path else Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)

    for tag, package in VERSIONS:
        destination = vendor / package
        if destination.exists() and not force:
            print(f"{package} already exists; skipped.")
            continue

        # 1. Download into a temporary folder; never change the working directory.
        with tempfile.TemporaryDirectory(prefix="tabpfn-source-") as temporary:
            clone = Path(temporary) / "source"
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", tag, REPO_URL, str(clone)],
                check=True,
            )
            source = clone / "src" / "tabpfn"
            if not source.is_dir():
                source = clone / "tabpfn"
            if not source.is_dir():
                raise RuntimeError(f"TabPFN source folder not found for {tag}")

            # 2. Rename imports so all four versions can be used together.
            prepared = Path(temporary) / package
            shutil.copytree(source, prepared, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for path in prepared.rglob("*.py"):
                original = path.read_text(encoding="utf-8")
                updated = re.sub(r"(\bfrom\s+)tabpfn(?=[.\s])", rf"\1{package}", original)
                updated = re.sub(r"(\bimport\s+)tabpfn(?=[.\s]|$)", rf"\1{package}", updated)
                updated = re.sub(r'''(["'])tabpfn(?=\.)''', rf"\1{package}", updated)
                if updated != original:
                    notice = f"# Modified for this benchmark: package imports renamed to {package}.\n"
                    path.write_text(notice + updated, encoding="utf-8")

            # 3. Preserve the upstream license and record the source version.
            notices = [path for path in clone.iterdir() if path.is_file()
                       and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))]
            if not any(path.name.upper().startswith("LICENSE") for path in notices):
                raise RuntimeError(f"Upstream license missing for {tag}; existing vendor copy kept")
            for notice in notices:
                shutil.copy2(notice, prepared / notice.name)
            commit = subprocess.run(
                ["git", "-C", str(clone), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            (prepared / "SOURCE.txt").write_text(
                f"Source: {REPO_URL}\nTag: {tag}\nCommit: {commit}\n"
                f"Local package: {package}\nModification: package imports renamed.\n",
                encoding="utf-8",
            )

            # 4. Replace only this package after its new copy is prepared.
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(prepared, destination)
            print(f"Prepared {package} from {tag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing vendor source copies")
    args = parser.parse_args()
    setup_vendor(force=args.force)
