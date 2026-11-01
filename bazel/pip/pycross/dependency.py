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

from typing import Any

from packaging.markers import Marker
from pip_platform import ALL_PLATFORMS, Platform
from utils import assert_keys

_EXTRA_PREFIX = "group-15-bazel-pyproject-"
# Groups that target no GPU, so they share the plain (cpu) constraint.
_CPU_TYPES = ("cpu", "aiperf")
# Maps dependency-group extra names to their GPU constraint suffix.
# nvidia-bench, nvidia-sglang and nvidia-vllm are separate conflict groups but
# still target nvidia hardware, so they share the "nvidia" constraint.
_GPU_TYPES: dict[str, str] = {
    "amd": "amd",
    "nvidia": "nvidia",
    "nvidia-bench": "nvidia",
    "nvidia-sglang": "nvidia",
    "nvidia-vllm": "nvidia",
}


def _system_environment(platform: Platform, extra: str) -> dict[str, str]:
    return {
        "extra": extra,
        "platform_machine": platform.arch,
        "platform_system": platform.operating_system,
        "python_full_version": platform.python_version + ".0",
        "python_version": platform.python_version,
        "sys_platform": platform.operating_system,
    }


class Dependency:
    def __init__(self, blob: dict[str, Any], all_versions: dict[str, str]):
        assert_keys(
            blob,
            required={"name"},
            optional={"marker", "version", "source", "extra"},
        )

        version = blob.get("version") or all_versions[blob["name"]]
        self.name = blob["name"] + "@" + version

        self.marker_str = blob.get("marker", "")
        # NOTE: Assume if the dependency is a duplicate, it is supported on all platforms
        if all_versions[blob["name"]] == "multiple":
            self.name = blob["name"] + "@multiple"
            self.marker_str = ""

    def merge_markers(self, other: "Dependency") -> None:
        if not self.marker_str:
            self.marker_str = other.marker_str
            return
        if not other.marker_str:
            return

        self.marker_str = f"({self.marker_str}) or ({other.marker_str})"

    def constraints(self) -> tuple[list[str], list[str]]:
        if not self.marker_str:
            return [""], []

        marker = Marker(self.marker_str)

        def _matches(platform: Platform) -> bool:
            return any(
                marker.evaluate(
                    _system_environment(platform, _EXTRA_PREFIX + group)
                )
                for group in _CPU_TYPES
            )

        universal = True
        for platform in ALL_PLATFORMS:
            if not _matches(platform):
                universal = False
                break

        if universal:
            return [""], []

        constraints = set()
        for platform in ALL_PLATFORMS:
            if _matches(platform):
                constraints.add(platform.constraint)

        gpu_constraints = set()
        for platform in ALL_PLATFORMS:
            if not platform.supports_gpu or _matches(platform):
                continue
            for gpu_extra_suffix, constraint_suffix in _GPU_TYPES.items():
                gpu_extra = _EXTRA_PREFIX + gpu_extra_suffix
                if marker.evaluate(_system_environment(platform, gpu_extra)):
                    gpu_constraints.add(
                        f"{platform.constraint}_{constraint_suffix}_gpu"
                    )

        return sorted(constraints), sorted(gpu_constraints)
