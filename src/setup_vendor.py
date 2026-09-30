"""One-time setup: install four TabPFN versions and copy them to vendor/.
Run once from the notebook: from src.setup_vendor import setup_vendor; setup_vendor(MAIN_PATH)
"""
import os, sys, shutil, subprocess, re

REPO_URL = "https://github.com/PriorLabs/TabPFN.git"

VERSIONS = [
    ("v6.0.0", "tabpfn_v25"),
    ("v7.1.1", "tabpfn_v26"),
    ("v8.5.0", "tabpfn_v3"),
    ("v9.0.0", "tabpfn_v35"),
]


def _install(tag, pkg):
    clone = f"/content/tabpfn_{tag.replace('.', '_')}_source"
    if os.path.isdir(clone):
        shutil.rmtree(clone)

    subprocess.run(["git", "clone", "--depth", "1",
                    "--branch", tag, REPO_URL, clone], check=True)
    os.chdir(clone)

    # Locate the tabpfn package folder
    if os.path.isdir("tabpfn"):
        pkg_dir = "tabpfn"
    elif os.path.isdir("src/tabpfn"):
        pkg_dir = "src/tabpfn"
    else:
        raise RuntimeError(f"tabpfn package not found in {clone}")

    # Rename to the target name
    parent = os.path.dirname(pkg_dir) or "."
    os.rename(pkg_dir, os.path.join(parent, pkg))
    if parent != ".":
        os.rename(os.path.join(parent, pkg), pkg)

    # Fix internal imports
    for pat, rep in [
        (r"from tabpfn\.", f"from {pkg}."),
        (r"from tabpfn import", f"from {pkg} import"),
        (r"import tabpfn\.", f"import {pkg}."),
        (r"^(\s*)import tabpfn\s*$", rf"\1import {pkg}"),
    ]:
        for root, _, files in os.walk(pkg):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                p = os.path.join(root, fname)
                with open(p) as f:
                    txt = f.read()
                new = re.sub(pat, rep, txt, flags=re.MULTILINE)
                if new != txt:
                    with open(p, "w") as f:
                        f.write(new)

    os.chdir("/content")
    return os.path.join(clone, pkg)


def setup_vendor(main_path, force=False):
    """Install the four TabPFN versions and copy them into main_path/vendor/."""
    vendor = os.path.join(main_path, "vendor")
    os.makedirs(vendor, exist_ok=True)

    for tag, pkg in VERSIONS:
        dst = os.path.join(vendor, pkg)

        if os.path.isdir(dst) and not force:
            print(f"✓ {pkg} already in vendor/ — skipping")
            continue

        print(f"→ installing {pkg} ({tag})...")
        src_pkg = _install(tag, pkg)

        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src_pkg, dst)
        print(f"✓ copied to {dst}")

    print("done. Restart the runtime, then continue with dependencies installation (pip install -r requirements.txt).")