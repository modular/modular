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
"""Tests the max.nn.legacy deprecation warning.

The warning fires once per process, so every case runs in a fresh
interpreter.
"""

import json
import os
import subprocess
import sys
import textwrap

_RECORD = """
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
{body}
print(json.dumps([
    [str(w.message), w.category.__name__, w.filename]
    for w in caught
    if "max.nn.legacy" in str(w.message)
]))
"""


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def _legacy_warnings(body: str) -> list[list[str]]:
    return json.loads(
        _run(
            _RECORD.format(body=textwrap.indent(textwrap.dedent(body), "    "))
        )
    )


def _assert_one_warning_at_caller(warnings: list[list[str]]) -> None:
    assert len(warnings) == 1, warnings
    message, category, filename = warnings[0]
    assert category == "DeprecationWarning"
    assert "removed in MAX 27.1" in message
    assert "Use max.nn instead" in message
    # Attributed to the code that used the module, not to MAX internals.
    assert filename == "<string>"


def test_using_legacy_warns() -> None:
    _assert_one_warning_at_caller(
        _legacy_warnings("""
            import max.nn.legacy as nn
            nn.Linear
        """)
    )


def test_importing_new_nn_does_not_warn() -> None:
    # max.nn imports max.nn.legacy itself; that internal use stays quiet.
    assert _legacy_warnings("import max.nn") == []


def test_importing_legacy_without_using_it_does_not_warn() -> None:
    # Loading always goes through max.nn, so only a use reaches the hook.
    assert _legacy_warnings("import max.nn.legacy") == []


def test_using_legacy_after_new_nn_loaded_it_warns() -> None:
    _assert_one_warning_at_caller(
        _legacy_warnings("""
            import max.nn
            from max.nn.legacy import Linear
        """)
    )


def test_importing_a_legacy_submodule_warns() -> None:
    _assert_one_warning_at_caller(
        _legacy_warnings("""
            import max.nn
            from max.nn.legacy.kv_cache import KVCacheParams
        """)
    )


def test_warns_once_per_process() -> None:
    _assert_one_warning_at_caller(
        _legacy_warnings("""
            import max.nn.legacy
            from max.nn.legacy import Linear, Module
            from max.nn.legacy.kernels import rope_ragged
            max.nn.legacy.Sequential
        """)
    )


def test_every_legacy_module_is_hooked() -> None:
    unhooked = json.loads(
        _run("""
import importlib, json, pkgutil, sys, types
import max.nn.legacy
for info in pkgutil.walk_packages(max.nn.legacy.__path__, "max.nn.legacy."):
    importlib.import_module(info.name)
print(json.dumps(sorted(
    name for name, module in sys.modules.items()
    if (name == "max.nn.legacy" or name.startswith("max.nn.legacy."))
    and type(module) is types.ModuleType
)))
""")
    )
    assert unhooked == []
