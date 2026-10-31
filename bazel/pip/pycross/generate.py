#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import tomllib  # type: ignore
from package import Package
from pip_platform import ALL_PLATFORMS
from template import TEMPLATE

_TORCH_PACKAGES = {
    "torch",
    "torchaudio",
    "torchvision",
    "triton",  # pytorch-triton-rocm was renamed to triton
}

# Must match DEFAULT_PYTHON_VERSION in common.MODULE.bazel.
_DEFAULT_PYTHON_VERSION = "3.13"

# Packages that only targets on the default Python version use, along with
# everything reachable only through them. Their wheels are built per Python
# version and an unconfigured `bazel query` fetches every select() branch, so
# other versions get no branch at all. Most of the savings is in torch, cutlass
# and the vllm/sglang kernels.
_DEFAULT_PYTHON_VERSION_ONLY = _TORCH_PACKAGES | {
    "flashinfer-python",
    "nvidia-cutlass-dsl",
    "sglang",
    "vllm",
}

_DEFAULT_PYTHON_VERSION_PLATFORMS = frozenset(
    platform
    for platform in ALL_PLATFORMS
    if platform.python_version == _DEFAULT_PYTHON_VERSION
)

# Force the @multiple label, mainly for adding extra constraints to targets
_FORCE_MULTIPLE = {
    "aiperf",
    "sglang",
    "vllm",
}

_ALLOWED_DUPLICATE_PACKAGES = (
    _TORCH_PACKAGES
    | _FORCE_MULTIPLE
    | {
        # Split based on Python versions
        "numpy",
        "scipy",
        # Unresolvable conflicts between dependency groups.
        # Only add these here if they are not globally
        # resolvable in `override-dependencies` (i.e. we
        # are required to diverge).
        "apache-tvm-ffi",  # vllm 0.30.0 pins 0.1.11; MAX and sglang use 0.1.9
        "click",  # vllm/sglang need >=8.4.2 via huggingface-hub; MAX tooling needs <8.2
        "compressed-tensors",  # vllm/sglang need 0.17.0; MAX stays on 0.15.0.1
        "cuda-bindings",  # vllm/sglang use CUDA 13; the default groups use CUDA 12
        "cuda-pathfinder",  # Follows cuda-python's split
        "cuda-python",  # vllm/sglang use CUDA 13; the default groups use CUDA 12
        "cuda-toolkit",  # Differs between vllm's and sglang's torch versions
        "fastapi",  # vllm 0.24.0 caps fastapi below what other groups resolve to
        "flash-attn-4",  # sglang 0.5.12.post1 needs 4.0.0b9; the benchmarks use 4.0.0b31
        "flashinfer-cubin",  # sglang pins 0.6.11.post1; MAX uses 0.6.8.post1
        "flashinfer-python",  # MAX, vllm and sglang each pin a different version
        "hf-xet",  # Follows huggingface-hub's split
        "huggingface-hub",  # vllm 0.30.0 needs >=1.31, which forces click>=8.4.2
        "idna",  # Follows mcp's split
        "llguidance",  # Not used by MAX directly; vllm pulls 1.7.5, sglang pins 0.7.30
        "mcp",  # vllm 0.30.0 needs mcp 2.x; MAX and sglang use 1.30.0
        "nvidia-cudnn-frontend",  # flashinfer 0.7 needs >=1.29 (bench, vllm); MAX and sglang use 1.26.0
        "nvidia-cublas",  # Differs between vllm's and sglang's torch versions
        "nvidia-cudnn-cu13",  # Differs between vllm's and sglang's torch versions
        "nvidia-cusparselt-cu13",  # Differs between vllm's and sglang's torch versions
        "nvidia-cutlass-dsl",  # MAX, vllm and sglang each pin a different version
        "nvidia-cutlass-dsl-libs-base",  # Follows nvidia-cutlass-dsl's split
        "nvidia-cutlass-dsl-libs-core",  # Follows nvidia-cutlass-dsl's split
        "nvidia-cutlass-dsl-libs-cu12",  # Follows nvidia-cutlass-dsl's split
        "nvidia-cutlass-dsl-libs-cu13",  # Follows nvidia-cutlass-dsl's split
        "nvidia-nccl-cu13",  # Differs between vllm's and sglang's torch versions
        "nvidia-nvjitlink",  # Differs between vllm's and sglang's torch versions
        "openai",  # vllm 0.30.0 needs >=2.25; sglang pins 2.6.1; MAX uses 2.11.0
        "outlines-core",  # Conflicts between vllm and sglang
        "quack-kernels",  # vllm pins 0.6.5, sglang 0.4.1; flash-attn-4 (aarch64/darwin) pins >=0.3.3
        "sse-starlette",  # Follows mcp's split
        "tilelang",  # MAX itself doesn't use tilelang, but the default environment group does; vllm 0.30.0 hard-pins 0.1.12
        "tokenspeed-mla",  # vllm pins ==0.1.8, sglang pins ==0.1.1
        "torchcodec",  # vllm pins a newer version than sglang
        "xgrammar",  # vllm 0.31.0 pins 0.2.7; sglang pins 0.2.0
        "transformers",  # MAX pins 5.12.x; sglang pins 5.3.0, vllm aligns on 5.12.x
        "vllm",
        "sglang",
        # aiperf (SemiAnalysis fork) pins different versions for these.
        "datasets",
        "mteb",
        "plotly",
        "pyzmq",
        "rich",
        "tqdm",
    }
)


# Optional deps are forced into each package's own target, so an extra whose
# dep depends back on the package forms a cycle: aiperf's textual requests
# markdown-it-py[plugins], and mdit-py-plugins depends on markdown-it-py.
_SKIP_OPTIONAL_DEPS = {("markdown-it-py", "mdit-py-plugins")}


def _should_ignore(
    package: dict[str, Any],
    cpu_versions: set[tuple[str, str]],
) -> bool:
    # Ignores pypi torch versions because uv is too aggressive about pulling
    # those in even though a group will always be specified.
    registry = package["source"].get("registry", "")
    return package["name"] == "bazel-pyproject" or (
        package["name"] in _TORCH_PACKAGES
        and (
            # Ignore torch versions from pypi that should not be in the lockfile
            "https://pypi.org/simple" in registry
            or (
                # Ignore plain-versioned torch packages from non-cpu registries
                # only if the same version is already available from the cpu
                # registry (avoid dropping versions that only exist in cu128).
                "+" not in package["version"]
                and "cpu" not in registry
                and (package["name"], package["version"]) in cpu_versions
            )
        )
    )


def _get_direct_deps(data: dict) -> set[str]:  # type: ignore[type-arg]
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


def _reachable(
    starts: Iterable[str],
    graph: Mapping[str, set[str]],
    blocked: Collection[str] = (),
) -> set[str]:
    seen: set[str] = set()
    stack = [name for name in starts if name not in blocked]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(dep for dep in graph[name] if dep not in blocked)
    return seen


def _default_python_version_only(
    data: Mapping[str, Any],
    direct_deps: set[str],
    roots: set[str],
) -> set[str]:
    """Returns the roots and every package reachable only through them."""
    graph: dict[str, set[str]] = defaultdict(set)
    for package in data["package"]:
        name = package["name"]
        graph[name].update(
            dep["name"] for dep in package.get("dependencies", [])
        )
        for deps in package.get("optional-dependencies", {}).values():
            graph[name].update(
                dep["name"]
                for dep in deps
                if (name, dep["name"]) not in _SKIP_OPTIONAL_DEPS
            )

    everything = _reachable(direct_deps, graph)
    shared = _reachable(direct_deps - roots, graph, blocked=roots)
    return everything - shared


def _main(uv_lock: str, output_path: str) -> None:
    with open(uv_lock, "rb") as f:
        data = tomllib.load(f)

    # Collect plain-versioned torch packages available from the cpu registry so
    # we can deduplicate non-cpu registry entries that resolve to the same version.
    cpu_versions = {
        (pkg["name"], pkg["version"])
        for pkg in data["package"]
        if pkg["name"] in _TORCH_PACKAGES
        and "cpu" in pkg["source"].get("registry", "")
        and "+" not in pkg["version"]
    }

    package_names = set()
    duplicate_packages = set()

    all_versions = {}
    for package in data["package"]:
        if _should_ignore(package, cpu_versions):
            continue

        name = package["name"]
        all_versions[name] = package["version"]
        if name in package_names or name in _FORCE_MULTIPLE:
            duplicate_packages.add(name)
            all_versions[name] = "multiple"
        package_names.add(name)

    unexpected_duplicates = duplicate_packages - _ALLOWED_DUPLICATE_PACKAGES
    if unexpected_duplicates:
        print("\nerror: Found duplicate packages that are not expected:")
        for package in sorted(unexpected_duplicates):
            print(f"  {package}")
        sys.exit(1)

    direct_deps = _get_direct_deps(data)
    default_python_version_only = _default_python_version_only(
        data, direct_deps, _DEFAULT_PYTHON_VERSION_ONLY
    )

    packages = [
        Package(
            package,
            all_versions,
            _SKIP_OPTIONAL_DEPS,
            _DEFAULT_PYTHON_VERSION_PLATFORMS
            if package["name"] in default_python_version_only
            else ALL_PLATFORMS,
        )
        for package in data["package"]
        if not _should_ignore(package, cpu_versions)
    ]

    targets = ""
    all_downloads = set()
    for package in packages:
        pkg, downloads = package.render()
        targets += pkg
        all_downloads |= downloads

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
