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

import functools

from download import Download
from packaging.tags import Tag, compatible_tags, cpython_tags, mac_platforms

SUPPORTED_PYTHON_VERSIONS = {"3.9", "3.10", "3.11", "3.12", "3.13"}
SUPPORTED_PLATFORMS = {
    ("linux", "aarch64"),
    ("linux", "x86_64"),
    ("darwin", "arm64"),
}

# NOTE: Order matters, earlier ones will be preferred. This list must be
# generated on the oldest OS version we support with the oldest supported
# Python version by calling 'packaging.tags.platform_tags()'.
_LINUX_PLATFORM_TAGS = [
    "manylinux_2_35_{arch}",
    "manylinux_2_34_{arch}",
    "manylinux_2_33_{arch}",
    "manylinux_2_32_{arch}",
    "manylinux_2_31_{arch}",
    "manylinux_2_30_{arch}",
    "manylinux_2_29_{arch}",
    "manylinux_2_28_{arch}",
    "manylinux_2_27_{arch}",
    "manylinux_2_26_{arch}",
    "manylinux_2_25_{arch}",
    "manylinux_2_24_{arch}",
    "manylinux_2_23_{arch}",
    "manylinux_2_22_{arch}",
    "manylinux_2_21_{arch}",
    "manylinux_2_20_{arch}",
    "manylinux_2_19_{arch}",
    "manylinux_2_18_{arch}",
    "manylinux_2_17_{arch}",
    "manylinux2014_{arch}",
]

# NOTE: These tags are not supported by aarch64 linux
_LINUX_X86_64_ONLY_PLATFORM_TAGS = [
    "manylinux_2_16_x86_64",
    "manylinux_2_15_x86_64",
    "manylinux_2_14_x86_64",
    "manylinux_2_13_x86_64",
    "manylinux_2_12_x86_64",
    "manylinux2010_x86_64",
    "manylinux_2_11_x86_64",
    "manylinux_2_10_x86_64",
    "manylinux_2_9_x86_64",
    "manylinux_2_8_x86_64",
    "manylinux_2_7_x86_64",
    "manylinux_2_6_x86_64",
    "manylinux_2_5_x86_64",
    "manylinux1_x86_64",
]


class Platform:
    python_version: str
    operating_system: str
    arch: str

    def __init__(self, python_version: str, operating_system: str, arch: str):
        self.operating_system = operating_system
        self.python_version = python_version
        self.arch = arch

    def __hash__(self) -> int:
        return hash((self.python_version, self.operating_system, self.arch))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Platform):
            return NotImplemented

        return (
            self.python_version == other.python_version
            and self.operating_system == other.operating_system
            and self.arch == other.arch
        )

    def __repr__(self) -> str:
        return f"Platform(python_version={self.python_version!r}, operating_system={self.operating_system!r}, arch={self.arch!r})"

    @property
    def tag(self) -> str:
        return f"cp{self.python_version.replace('.', '')}"

    @functools.cached_property
    def tags(self) -> list[Tag]:
        if self.operating_system == "darwin":
            # NOTE: The version here must match the oldest macOS version we support for developers
            platforms = list(mac_platforms(version=(12, 0), arch=self.arch))
        else:
            platforms = [x.format(arch=self.arch) for x in _LINUX_PLATFORM_TAGS]
            if self.arch == "x86_64":
                platforms += _LINUX_X86_64_ONLY_PLATFORM_TAGS
            # NOTE: Added last, after x86_64 tags
            platforms.append(f"linux_{self.arch}")

        python_version = tuple(map(int, self.python_version.split(".")))
        return list(
            cpython_tags(
                python_version=python_version,
                abis=[
                    "abi3",
                    "none",
                    "cp" + self.python_version.replace(".", ""),
                ],
                platforms=platforms,
            )
        ) + list(
            compatible_tags(
                python_version=python_version,
                interpreter="cp",
                platforms=platforms,
            )
        )

    def is_compatible_with(self, tag: Download) -> bool:
        return bool(set(self.tags) & tag.tags)

    def first_matching_wheel_tag(self, downloads: list[Download]) -> Download:
        for tag in self.tags:
            for download in downloads:
                assert download.is_wheel
                if tag in download.tags:
                    return download

        raise ValueError("Wheels passed here should already be compatible")

    @property
    def constraint(self) -> str:
        """Return the platform constraint for bazel."""
        if self.operating_system == "darwin":
            return f":_env_python_{self.python_version}_aarch64-apple-darwin"
        return (
            f":_env_python_{self.python_version}_{self.arch}-unknown-linux-gnu"
        )

    @property
    def supports_gpu(self) -> bool:
        return self.operating_system == "linux" and self.arch == "x86_64"


ALL_PLATFORMS = frozenset(
    {
        Platform(
            python_version=py, operating_system=operating_system, arch=arch
        )
        for py in SUPPORTED_PYTHON_VERSIONS
        for operating_system, arch in SUPPORTED_PLATFORMS
    }
)
