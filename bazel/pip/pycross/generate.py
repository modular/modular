#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


import os
import sys
from typing import Any

import tomllib  # type: ignore
from package import Package
from template import TEMPLATE

_ALLOWED_DUPLICATE_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
}


def _should_ignore(package: dict[str, Any]) -> bool:
    # Ignores pypi torch versions because uv is too aggressive about pulling
    # those in even though a group will always be specified.
    return package["name"] == "bazel-pyproject" or (
        package["name"] in _ALLOWED_DUPLICATE_PACKAGES
        and (
            # Ignore torch versions from pypi that should not be in the lockfile
            "https://pypi.org/simple" in package["source"].get("registry", "")
            or (
                # Ignore torch versions that are not GPU specific but are from the GPU registry and should not be in the lockfile
                "+" not in package["version"]
                and "cpu" not in package["source"].get("registry", "")
            )
        )
    )


def _get_direct_deps(data: dict) -> set[str]:
    direct_deps = set()

    for package in data["package"]:
        if package["name"] == "bazel-pyproject":
            for dep in package["dependencies"]:
                direct_deps.add(dep["name"].lower())

            for group in package["dev-dependencies"].values():
                for dep in group:
                    direct_deps.add(dep["name"].lower())
            break

    return direct_deps


def _main(uv_lock: str, output_path: str) -> None:
    with open(uv_lock, "rb") as f:
        data = tomllib.load(f)

    package_names = set()
    duplicate_packages = set()

    all_versions = {}
    for package in data["package"]:
        if _should_ignore(package):
            continue

        all_versions[package["name"]] = package["version"]
        if package["name"] in package_names:
            duplicate_packages.add(package["name"])
            all_versions[package["name"]] = "multiple"
        package_names.add(package["name"])

    unexpected_duplicates = duplicate_packages - _ALLOWED_DUPLICATE_PACKAGES
    if unexpected_duplicates:
        print("\nerror: Found duplicate packages that are not expected:")
        for package in sorted(unexpected_duplicates):
            print(f"  {package}")
        exit(1)

    targets = ""
    all_downloads = set()
    for package in data["package"]:
        if _should_ignore(package):
            continue

        pkg, downloads = Package(package, all_versions).render()
        targets += pkg
        all_downloads |= downloads

    direct_deps = _get_direct_deps(data)
    output = TEMPLATE.format(
        pins="\n".join(
            f'    "{name}": "{name}@{target}",'
            for name, target in sorted(all_versions.items())
            if name.lower() in direct_deps
        ),
        targets=targets,
        repositories="\n".join(
            download.render() for download in sorted(all_downloads)
        ),
    )

    with open(output_path, "w") as f:
        f.write(output.strip() + "\n")


if __name__ == "__main__":
    if directory := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    _main(sys.argv[1], sys.argv[2])
