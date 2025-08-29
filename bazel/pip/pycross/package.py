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

from collections import defaultdict
from typing import Any

from dependency import Dependency
from download import Download
from pip_platform import ALL_PLATFORMS, Platform
from render import render_constrained_deps
from utils import assert_keys


class Package:
    def __init__(self, blob: dict[str, Any], all_versions: dict[str, str]):
        assert_keys(
            blob,
            required={"name", "version"},
            optional={
                "dependencies",
                "metadata",
                "optional-dependencies",
                "resolution-markers",
                "sdist",
                "source",
                "wheels",
            },
        )

        self.name = blob["name"]
        self.version = blob["version"]
        self.sdist = Download(blob["sdist"]) if "sdist" in blob else None
        self.wheels = [Download(wheel) for wheel in blob.get("wheels", [])]

        self.library_name = self.name + "@" + self.version
        self.wheel_target_name = "_wheel_" + self.name + "@" + self.version
        self.sdist_target_name = "_sdist_" + self.name + "@" + self.version
        self.sdist_build_target_name = (
            "_build_" + self.name + "@" + self.version
        )
        deps_prefix = (
            f"_{self.name}_{self.version}".replace("-", "_")
            .replace(".", "_")
            .replace("+", "_")
        )
        self.deps_var_name = f"{deps_prefix}_deps"
        self.build_deps_var_name = f"{deps_prefix}_build_deps"

        all_dependencies = [
            Dependency(dep, all_versions)
            for dep in blob.get("dependencies", [])
        ] + [
            # NOTE: Force all optional dependencies to be included
            Dependency(dep, all_versions)
            for deps in blob.get("optional-dependencies", {}).values()
            for dep in deps
        ]
        unique_dependencies: dict[str, Dependency] = {}
        for dep in all_dependencies:
            if dep.name in unique_dependencies:
                # TODO: should not be possible besides torch?
                unique_dependencies[dep.name].merge_markers(dep)
            else:
                unique_dependencies[dep.name] = dep

        self.dependencies = list(unique_dependencies.values())

    def render(self) -> tuple[str, set[Download]]:
        deps_by_constraints: dict[str, list[str]] = defaultdict(list)
        deps_by_gpu_constraints: dict[str, list[str]] = defaultdict(list)
        for dep in self.dependencies:
            constraints, gpu_constraints = dep.constraints()
            for constraint in constraints:
                deps_by_constraints[constraint].append(dep.name)
            for gpu_constraint in gpu_constraints:
                deps_by_gpu_constraints[gpu_constraint].append(dep.name)

        package = render_constrained_deps(
            self.deps_var_name, deps_by_constraints, deps_by_gpu_constraints
        )

        downloads_by_platform: dict[Platform, Download] = {}
        sdist_platforms = set()
        for platform in ALL_PLATFORMS:
            potential_wheels = [
                whl for whl in self.wheels if platform.is_compatible_with(whl)
            ]
            if not potential_wheels:
                sdist_platforms.add(platform)
                continue

            best_match = platform.first_matching_wheel_tag(potential_wheels)
            downloads_by_platform[platform] = best_match

        build_deps_line = f"deps = {self.build_deps_var_name}"
        if package:
            deps_line = f"""
        deps = {self.deps_var_name},"""
            build_deps_line = (
                f"deps = {self.deps_var_name} + {self.build_deps_var_name}"
            )
        else:
            deps_line = ""

        testonly_line = f"""
        testonly = "{self.name}" in _TESTONLY_DEPS,"""

        all_downloads = set(downloads_by_platform.values())
        needs_sdist = len(sdist_platforms) > 0
        # No sdist is potentially ok as long as the package is never used on platforms without wheels
        if needs_sdist and not self.sdist:
            sdist_platforms = set()
        elif needs_sdist and self.sdist:
            all_downloads.add(self.sdist)
            package += f"""\
    {self.build_deps_var_name} = [
        ":setuptools",
        ":wheel",
    ]

    native.alias(
        name = "{self.sdist_target_name}",
        actual = "@{self.sdist.name}//file",
    )

    pycross_wheel_build(
        name = "{self.sdist_build_target_name}",
        sdist = ":{self.sdist_target_name}",
        target_environment = _target,
        {build_deps_line},{testonly_line}
        **extra_build_args
    )

"""
        assert all_downloads, (
            f"No downloads for package {self.name} with wheels: {self.wheels}."
        )

        select_values = {}
        for platform, download in downloads_by_platform.items():
            assert download.is_wheel
            select_values[platform.constraint] = f"@{download.name}//file"

        for platform in sdist_platforms:
            select_values[platform.constraint] = (
                f":{self.sdist_build_target_name}"
            )

        if not select_values:
            raise ValueError(
                f"No supported platforms for package {self.name} with wheels: {self.wheels}, -------- {downloads_by_platform}."
            )

        unique_downloads = set(select_values.values())
        if len(unique_downloads) == 1:
            actual = f'"{next(iter(unique_downloads))}",'
        else:
            actual = f"""select({{
            {",\n            ".join(sorted(f'"{k}": "{v}"' for k, v in select_values.items()))},
        }}),"""

        tags_line = ""
        if self.library_name.startswith("torch@"):
            tags_line = """
        tags = ["no-remote"],
        exec_compatible_with = HOST_CONSTRAINTS,"""

        package += f"""\
    native.alias(
        name = "{self.wheel_target_name}",
        actual = {actual}
    )

    pycross_wheel_library(
        name = "{self.library_name}",{deps_line}
        wheel = ":{self.wheel_target_name}",{tags_line}{testonly_line}
    )

"""

        return (package, all_downloads)
