#!/usr/bin/env python
import os
import subprocess
import sys
import shutil

REPO_URL = "https://github.com/PriorLabs/TabPFN.git"

# Install four versions of TabPFN side-by-side:
#  - v6.0.0  -> package name: tabpfn_v25   (TabPFN-2.5)
#  - v7.1.1  -> package name: tabpfn_v26   (TabPFN-2.6)
#  - v8.5.0  -> package name: tabpfn_v3    (TabPFN-3)
#  - v9.0.0  -> package name: tabpfn_v35   (TabPFN-3.5)


def install_version(version_tag, new_package_name):
    clone_dir = f"/content/tabpfn_{version_tag.replace('.', '_')}_source"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)

    subprocess.run(["git", "clone", "--depth", "1", "--branch", version_tag,
                REPO_URL, clone_dir], check=True)
    os.chdir(clone_dir)

    # Find where the `tabpfn` package actually lives
    if os.path.isdir("tabpfn"):
        pkg_dir = "tabpfn"
    elif os.path.isdir("src/tabpfn"):
        pkg_dir = "src/tabpfn"
    else:
        raise RuntimeError(f"Could not find `tabpfn` package in {clone_dir}")

    # Rename it
    parent = os.path.dirname(pkg_dir) or "."
    os.rename(pkg_dir, os.path.join(parent, new_package_name))

    # Move it to the root so it's importable as a top-level package
    if parent != ".":
        os.rename(os.path.join(parent, new_package_name),
                  os.path.join(".", new_package_name))

    # Fix internal imports
    subprocess.run(f"find . -type f -name '*.py' -exec sed -i 's/from tabpfn\\./from {new_package_name}./g' {{}} +",
                   shell=True, check=False)
    subprocess.run(f"find . -type f -name '*.py' -exec sed -i 's/import tabpfn\\./import {new_package_name}./g' {{}} +",
                   shell=True, check=False)
    subprocess.run(f"find . -type f -name '*.py' -exec sed -i 's/\\bimport tabpfn\\b/import {new_package_name}/g' {{}} +",
                   shell=True, check=False)

    # Update package name in pyproject.toml
    subprocess.run(f"sed -i 's/^name = \"tabpfn\"$/name = \"{new_package_name}\"/g' pyproject.toml",
                   shell=True, check=False)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    os.chdir("/content")
    print(f"✅ {new_package_name} ({version_tag}) installed.")
