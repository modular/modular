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

import pytest
from dependency import Dependency


def test_throws_missing_keys() -> None:
    with pytest.raises(ValueError, match="Missing required keys in blob"):
        Dependency({}, {})


def test_initialize_dependency_without_version() -> None:
    blob = {
        "name": "example_package",
        "marker": "sys_platform == 'linux'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    assert dependency.name == "example_package@1.0.0"
    assert dependency.marker_str == "sys_platform == 'linux'"


def test_initialize_dependency_with_version() -> None:
    blob = {
        "name": "example_package",
        "marker": "sys_platform == 'linux' or sys_platform == 'darwin'",
        "version": "1.1.0",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    assert dependency.name == "example_package@1.1.0"
    assert (
        dependency.marker_str
        == "sys_platform == 'linux' or sys_platform == 'darwin'"
    )


def test_no_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    assert dependency.constraints() == ([""], [])


def test_all_platforms_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "sys_platform == 'linux' or sys_platform == 'darwin'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    assert dependency.constraints() == ([""], [])


def test_universal_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "(sys_platform == 'linux' or sys_platform == 'darwin') and python_version >= '3.9'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    assert dependency.constraints() == ([""], [])


def test_python_version_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "python_version < '3.10'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    constraints, gpu_constraints = dependency.constraints()
    assert not gpu_constraints
    assert constraints == [
        ":_env_python_3.9_aarch64-apple-darwin",
        ":_env_python_3.9_aarch64-unknown-linux-gnu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu",
    ]


def test_pypy_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "python_implementation == 'PyPy'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)
    constraints, gpu_constraints = dependency.constraints()
    assert not constraints
    assert not gpu_constraints


def test_wild_constraint() -> None:
    blob = {
        "name": "example_package",
        "marker": "(python_full_version < '3.10' and sys_platform == 'darwin') or (python_full_version < '3.10' and sys_platform == 'linux') or (sys_platform != 'darwin' and sys_platform != 'linux' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-cpu') or (sys_platform != 'darwin' and sys_platform != 'linux' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-nvidia') or (sys_platform != 'darwin' and sys_platform != 'linux' and extra == 'group-15-bazel-pyproject-cpu' and extra == 'group-15-bazel-pyproject-nvidia') or (sys_platform == 'darwin' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-cpu') or (sys_platform == 'darwin' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-nvidia') or (sys_platform == 'darwin' and extra == 'group-15-bazel-pyproject-cpu' and extra == 'group-15-bazel-pyproject-nvidia') or (sys_platform == 'linux' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-cpu') or (sys_platform == 'linux' and extra == 'group-15-bazel-pyproject-amd' and extra == 'group-15-bazel-pyproject-nvidia') or (sys_platform == 'linux' and extra == 'group-15-bazel-pyproject-cpu' and extra == 'group-15-bazel-pyproject-nvidia')",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    constraints, gpu_constraints = dependency.constraints()
    assert not gpu_constraints
    assert constraints == [
        ":_env_python_3.9_aarch64-apple-darwin",
        ":_env_python_3.9_aarch64-unknown-linux-gnu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu",
    ]


def test_nvidia_only_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "sys_platform == 'linux' and extra == 'group-15-bazel-pyproject-nvidia'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    constraints, gpu_constraints = dependency.constraints()
    assert not constraints
    assert gpu_constraints == [
        ":_env_python_3.10_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.11_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.12_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.13_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu_nvidia_gpu",
    ]


def test_amd_only_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "sys_platform == 'linux' and extra == 'group-15-bazel-pyproject-amd'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    constraints, gpu_constraints = dependency.constraints()
    assert not constraints
    assert gpu_constraints == [
        ":_env_python_3.10_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.11_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.12_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.13_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu_amd_gpu",
    ]


def test_multi_gpu_constraints() -> None:
    blob = {
        "name": "example_package",
        "marker": "extra == 'group-15-bazel-pyproject-nvidia' or extra == 'group-15-bazel-pyproject-amd'",
    }
    all_versions = {"example_package": "1.0.0"}
    dependency = Dependency(blob, all_versions)

    constraints, gpu_constraints = dependency.constraints()
    assert not constraints
    assert gpu_constraints == [
        ":_env_python_3.10_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.10_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.11_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.11_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.12_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.12_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.13_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.13_x86_64-unknown-linux-gnu_nvidia_gpu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu_amd_gpu",
        ":_env_python_3.9_x86_64-unknown-linux-gnu_nvidia_gpu",
    ]
