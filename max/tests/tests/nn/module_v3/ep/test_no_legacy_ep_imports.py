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
"""ModuleV3 code reaches expert parallelism only through ``max.nn.ep``.

The ModuleV2 models stay on the legacy exchange, ``max.nn.legacy.comm.ep``,
and so do the ``pipelines/lib`` files that set it up for them. Everything
else this test scans must not import it: the ModuleV3 architectures,
``max.nn`` outside ``max.nn.legacy``, ``max.pipelines.lib`` and the tests
and fixtures of the ModuleV3 expert-parallel families. The ``internal``
package scans ``mach/tools/gen-mef``, which is not public. The MiniMax-M3
ModuleV3 model in ``max_private`` stays on the legacy exchange by decision,
so no test scans it.
"""

from __future__ import annotations

import pathlib

import max.nn.ep
import max.pipelines.lib
import pytest
from legacy_ep_scan import legacy_ep_imports, scan

_MODULEV2_ARCHITECTURES = frozenset(
    {
        "deepseekV3",
        "deepseekV3_2",
        "deepseekV3_2_nextn",
        "deepseekV3_nextn",
        "deepseekV41",
        "hy_v3",
        "kimik2_5",
        "laguna",
        "llama3",
        "minimax_m2",
        "qwen3",
        "step3p5",
    }
)
"""The architecture packages that build ModuleV2 models, which stay on the
legacy exchange. Every other architecture package is scanned, and each of
these must still import the legacy exchange."""

_MODULEV2_LIB_FILES = frozenset(
    {
        "interfaces/pipeline_model.py",
        "registration_graph_api.py",
        "pipeline_variants/spec_decode_pipeline_model.py",
    }
)
"""The ``max.pipelines.lib`` files that set up the legacy exchange for a
ModuleV2 model. Each must still import it."""

_FAMILY_TEST_DIRS = (
    "beam_modulev3",
    "deepseekV3",
    "deepseekV3_2",
    "deepseekV41_modulev3",
    "dflash2_beam_modulev3",
    "dflash_mimo_v2_modulev3",
    "glm5_1",
    "glm5_next",
    "kimi_k3_modulev3",
    "kimik2_5_modulev3",
    "mimo_v2_modulev3",
    "mistral_large4",
    "mtp_deepseekV3",
    "mtp_glm5_next",
    "muse_spark",
)
"""The test directories, under ``max/tests/integration/architectures``, of
the ModuleV3 expert-parallel families. Each package in them that holds
Python files exports them as ``py_srcs`` for this test."""

_NN = pathlib.Path(max.nn.ep.__file__).parents[1]
_PIPELINES = pathlib.Path(max.pipelines.lib.__file__).parents[1]
_ARCHITECTURES = _PIPELINES / "architectures"
_FAMILY_TESTS = (
    pathlib.Path(__file__).parents[4] / "integration" / "architectures"
)

# The scan reads what this target's data puts in runfiles. Below these
# counts, assume a tree is missing and fail rather than pass on an empty
# walk.
_MIN_ARCHITECTURE_FILES = 800
_MIN_LIB_FILES = 50
_MIN_NN_FILES = 60


def _architecture_files() -> list[pathlib.Path]:
    return [
        path
        for path in sorted(_ARCHITECTURES.rglob("*.py"))
        if path.relative_to(_ARCHITECTURES).parts[0]
        not in _MODULEV2_ARCHITECTURES
    ]


def _lib_files() -> list[pathlib.Path]:
    lib = _PIPELINES / "lib"
    return [
        path
        for path in sorted(lib.rglob("*.py"))
        if path.relative_to(lib).as_posix() not in _MODULEV2_LIB_FILES
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "import max.nn.legacy.comm.ep",
        "import max.nn.legacy.comm.ep.ep_manager as m",
        "from max.nn.legacy.comm.ep import EPBatchManager",
        "from max.nn.legacy.comm.ep.ep_kernels import ep_dispatch",
        "from max.nn.legacy.comm import ep",
        "from max.nn.legacy.comm import EPCommBuffers",
        "def f():\n    from max.nn.legacy.comm.ep import EPCommInitializer",
        "from ..legacy.comm.ep.ep_kernels import ep_dispatch",
    ],
)
def test_finds_each_form_of_legacy_ep_import(statement: str) -> None:
    assert legacy_ep_imports(statement, "x.py")


@pytest.mark.parametrize(
    "statement",
    [
        "from max.nn.ep import EPBuffers",
        "from max.nn.legacy.comm import Signals",
        "from max.nn.legacy.graph_groups import EPPayload",
        '"""Replaces the legacy EPBatchManager (max.nn.legacy.comm.ep)."""',
    ],
)
def test_ignores_what_is_not_a_legacy_ep_import(statement: str) -> None:
    assert not legacy_ep_imports(statement, "x.py")


@pytest.mark.parametrize("name", sorted(_MODULEV2_ARCHITECTURES))
def test_exempt_architectures_still_import_legacy_ep(name: str) -> None:
    """An exemption that covers no legacy import would hide new code."""
    package = _ARCHITECTURES / name
    assert package.is_dir(), f"exempt architecture {name} does not exist"
    assert scan(sorted(package.rglob("*.py"))), (
        f"exempt architecture {name} no longer imports the legacy EP"
        " exchange: drop it from _MODULEV2_ARCHITECTURES"
    )


@pytest.mark.parametrize("name", sorted(_MODULEV2_LIB_FILES))
def test_exempt_lib_files_still_import_legacy_ep(name: str) -> None:
    """An exemption that covers no legacy import would hide new code."""
    path = _PIPELINES / "lib" / name
    assert path.is_file(), f"exempt pipelines/lib file {name} does not exist"
    assert scan([path]), (
        f"exempt pipelines/lib file {name} no longer imports the legacy EP"
        " exchange: drop it from _MODULEV2_LIB_FILES"
    )


def test_modulev3_architectures_do_not_import_legacy_ep() -> None:
    paths = _architecture_files()
    assert len(paths) >= _MIN_ARCHITECTURE_FILES, (
        f"only {len(paths)} architecture files were scanned; the"
        " architectures are not in runfiles"
    )
    assert not scan(paths)


def test_nn_outside_legacy_does_not_import_legacy_ep() -> None:
    paths = [
        path
        for path in sorted(_NN.rglob("*.py"))
        if path.relative_to(_NN).parts[0] != "legacy"
    ]
    assert len(paths) >= _MIN_NN_FILES, (
        f"only {len(paths)} max.nn files were scanned; the package is not"
        " in runfiles"
    )
    scanned = {path.relative_to(_NN).as_posix() for path in paths}
    for name in (
        "ep_config.py",
        "ep/exchange.py",
        "common_layers/quant_moe.py",
        "_graph_ops/ep.py",
        "kv_cache/__init__.py",
    ):
        assert name in scanned, f"max.nn/{name} is not in runfiles"
    assert not scan(paths)


def test_pipelines_lib_imports_legacy_ep_only_for_modulev2() -> None:
    paths = _lib_files()
    assert len(paths) >= _MIN_LIB_FILES, (
        f"only {len(paths)} pipelines/lib files were scanned; the package"
        " is not in runfiles"
    )
    assert not scan(paths)


@pytest.mark.parametrize("family", _FAMILY_TEST_DIRS)
def test_modulev3_family_tests_do_not_import_legacy_ep(family: str) -> None:
    paths = sorted((_FAMILY_TESTS / family).rglob("*.py"))
    assert paths, f"no tests of {family} are in runfiles"
    assert not scan(paths)
