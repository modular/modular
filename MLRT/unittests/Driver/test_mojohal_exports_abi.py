# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Checks the MojoHAL_* C declarations against their Mojo definitions.

Neither compiler sees the other side of this ABI, so a parameter whose type
drifts on one side still links and fails only at run time. This test compares
each export's name, return type and parameter types, in order.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# The C type each Mojo spelling crosses the ABI as. A Mojo raw-memory pointer
# carries no constness, so it matches both C spellings.
_MOJO_TO_C: dict[str, tuple[str, ...]] = {
    "_CString[]": ("const char *",),
    "Bool": ("bool",),
    "Int32": ("int32_t",),
    "Int": ("intptr_t",),
    "Int64": ("int64_t",),
    "UInt64": ("uint64_t",),
    "_DeviceContextBoxPtr": ("MojoHALContextBox *",),
    "_DeviceBufferBoxPtr": ("MojoHALBufferBox *",),
    "_DeviceStreamBoxPtr": ("MojoHALStreamBox *",),
    "_DeviceEventBoxPtr": ("MojoHALEventBox *",),
    "_DeviceContextScopeBoxPtr": ("MojoHALScopeBox *",),
    "_CompletionFlagBoxPtr": ("MojoHALCompletionFlagBox *",),
    "_RawMemoryPtr": ("void *", "const void *"),
    "OpaquePointer[MutUntrackedOrigin]": ("void *",),
    "Pointer[c_char, MutUntrackedOrigin]": ("char *",),
    "Pointer[mut=False, c_char, ImmUntrackedOrigin]": ("const char *",),
    "def(OpaquePointer[MutUntrackedOrigin]) thin -> None": (
        "void (*)(void *)",
    ),
}

_OUT_PTR = re.compile(r"^_OutPtr\[(.*)\]$")


def _c_types_for(mojo_type: str) -> tuple[str, ...]:
    mojo_type = " ".join(mojo_type.split())
    if match := _OUT_PTR.match(mojo_type):
        return tuple(
            t + "*" if t.endswith("*") else t + " *"
            for t in _c_types_for(match.group(1))
        )
    if mojo_type not in _MOJO_TO_C:
        raise AssertionError(f"no C spelling known for Mojo type {mojo_type}")
    return _MOJO_TO_C[mojo_type]


def _split_top_level(text: str) -> list[str]:
    """Splits on commas outside brackets and parentheses."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _normalize_c_type(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"\s*\*", " *", text)
    text = re.sub(r"\* \*", "**", text)
    return text.replace("( *)", "(*)")


def _c_param_type(param: str) -> str:
    """Drops the parameter name from a C parameter declaration."""
    if "(*" in param:
        return _normalize_c_type(re.sub(r"\(\*\s*\w+\)", "(*)", param))
    return _normalize_c_type(re.sub(r"\w+$", "", param.strip()))


def _mojo_exports(source: str) -> dict[str, tuple[str, list[str]]]:
    exports: dict[str, tuple[str, list[str]]] = {}
    pattern = re.compile(
        r"@export\s+def\s+(MojoHAL_\w+)\((.*?)\)\s+abi\(\"C\"\)"
        r"(?:\s*->\s*(.*?))?:\n",
        re.DOTALL,
    )
    for name, params, result in pattern.findall(source):
        param_types = [
            param.split(":", 1)[1].strip() for param in _split_top_level(params)
        ]
        exports[name] = (result.strip() or "None", param_types)
    return exports


def _c_declarations(header: str) -> dict[str, tuple[str, list[str]]]:
    header = re.sub(r"//[^\n]*", "", header)
    declarations: dict[str, tuple[str, list[str]]] = {}
    pattern = re.compile(
        r"([\w\s\*]+?)\b(MojoHAL_\w+)\s*\(((?:[^()]|\([^()]*\))*)\)\s*;"
    )
    for result, name, params in pattern.findall(header):
        param_list = (
            [] if params.strip() == "void" else _split_top_level(params)
        )
        declarations[name] = (
            _normalize_c_type(result),
            [_c_param_type(param) for param in param_list],
        )
    return declarations


def test_every_export_matches_its_declaration() -> None:
    exports = _mojo_exports(
        Path(os.environ["MOJOHAL_EXPORTS_MOJO"]).read_text()
    )
    declarations = _c_declarations(
        Path(os.environ["MOJOHAL_EXPORTS_H"]).read_text()
    )
    assert exports, "found no MojoHAL_* exports"

    assert sorted(exports) == sorted(declarations), (
        "exported but not declared:"
        f" {sorted(set(exports) - set(declarations))}; declared but not"
        f" exported: {sorted(set(declarations) - set(exports))}"
    )
    for name, (mojo_result, mojo_params) in exports.items():
        c_result, c_params = declarations[name]
        expected_result = (
            ("void",) if mojo_result == "None" else _c_types_for(mojo_result)
        )
        assert c_result in expected_result, (
            f"{name} returns {c_result} in C, {mojo_result} in Mojo"
        )
        assert len(c_params) == len(mojo_params), (
            f"{name} takes {len(c_params)} parameters in C,"
            f" {len(mojo_params)} in Mojo"
        )
        for index, (c_param, mojo_param) in enumerate(
            zip(c_params, mojo_params, strict=True)
        ):
            assert c_param in _c_types_for(mojo_param), (
                f"{name} parameter {index} is {c_param} in C, {mojo_param}"
                " in Mojo"
            )


def test_drifted_parameter_is_reported() -> None:
    exports = _mojo_exports(
        '@export\ndef MojoHAL_f(out: _OutPtr[_RawMemoryPtr]) abi("C"):\n'
    )
    declarations = _c_declarations("void MojoHAL_f(MojoHALStreamBox **out);")
    (_, (mojo_param,)) = exports["MojoHAL_f"]
    (_, (c_param,)) = declarations["MojoHAL_f"]
    assert c_param not in _c_types_for(mojo_param)
