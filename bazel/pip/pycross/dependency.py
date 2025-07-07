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

from typing import Any

from packaging.markers import Marker
from pip_platform import ALL_PLATFORMS, Platform
from utils import assert_keys

_EXTRA_PREFIX = "group-15-bazel-pyproject-"
_CPU_EXTRA = _EXTRA_PREFIX + "cpu"
_GPU_TYPES = ["amd", "nvidia"]


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
        universal = True
        for platform in ALL_PLATFORMS:
            if not marker.evaluate(_system_environment(platform, _CPU_EXTRA)):
                universal = False
                break

        if universal:
            return [""], []

        constraints = set()
        for platform in ALL_PLATFORMS:
            if marker.evaluate(_system_environment(platform, _CPU_EXTRA)):
                constraints.add(platform.constraint)

        if constraints:
            return sorted(constraints), []

        gpu_constraints = set()
        for platform in ALL_PLATFORMS:
            if not platform.supports_gpu:
                continue
            for gpu in _GPU_TYPES:
                gpu_extra = _EXTRA_PREFIX + gpu
                if marker.evaluate(_system_environment(platform, gpu_extra)):
                    gpu_constraints.add(f"{platform.constraint}_{gpu}_gpu")

        return [], sorted(gpu_constraints)
