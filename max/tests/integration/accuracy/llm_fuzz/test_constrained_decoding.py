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

import json
from typing import Any

import pytest
from scenarios.constrained_decoding import (
    _ROOT_SCOPED_KEYWORDS,
    _SCHEMA_DIR,
    _inline_refs,
    _is_root_scoped,
    _jsonschema_check,
    _tool_parameters,
    _wrap_root,
)


@pytest.mark.parametrize(
    "schema,expected",
    [
        ({"type": "string"}, False),
        ({"$ref": "#/definitions/a"}, True),
        ({"properties": {"a": {"$ref": "#"}}}, True),
        ({"allOf": [{"type": "integer"}, {"$ref": "#/definitions/a"}]}, True),
        ({"$id": "http://example.com/a.json", "type": "object"}, True),
        (
            {"items": {"$schema": "http://json-schema.org/draft-07/schema#"}},
            True,
        ),
        (True, False),
    ],
)
def test_is_root_scoped(schema: Any, expected: bool) -> None:
    assert _is_root_scoped(schema) is expected


def test_tool_parameters_wraps_only_ref_free_schemas() -> None:
    ref_free = {"type": "string"}
    assert _tool_parameters(ref_free) == _wrap_root(ref_free)

    with_ref = {
        "properties": {"a": {"$ref": "#/definitions/a"}},
        "definitions": {"a": {"type": "integer"}},
    }
    assert _tool_parameters(with_ref) is with_ref


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(_contains_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_contains_key(v, key) for v in node)
    return False


def _suite_groups() -> list[tuple[str, dict[str, Any]]]:
    groups = []
    for path in sorted(_SCHEMA_DIR.rglob("*.json")):
        rel = path.relative_to(_SCHEMA_DIR).with_suffix("").as_posix()
        for i, group in enumerate(json.loads(path.read_text())):
            if isinstance(group, dict) and "schema" in group:
                groups.append((f"{rel}[{i}]", group))
    return groups


def test_tool_parameters_preserve_suite_verdicts() -> None:
    """Tool ``parameters`` accept exactly the suite instances the schema does.

    Every draft 7 suite instance must get the same verdict from the original
    schema as from the tool ``parameters`` built for it, with the instance
    wrapped under ``value`` when the schema was.
    """
    failures: list[str] = []
    checked = 0
    for label, group in _suite_groups():
        schema = group["schema"]
        inlined = _inline_refs(schema)
        params = _tool_parameters(inlined)
        wrapped = params is not inlined
        if wrapped:
            leaked = [
                k for k in _ROOT_SCOPED_KEYWORDS if _contains_key(params, k)
            ]
            if leaked:
                failures.append(f"{label}: wrapped schema contains {leaked}")
            ok, detail = _jsonschema_check(params, {})
            if ok is not False:
                failures.append(f"{label}: wrapper accepts {{}} ({detail})")

        for test in group.get("tests", []):
            data = test["data"]
            expected, _ = _jsonschema_check(schema, data)
            if expected is None:
                # The oracle itself cannot run, e.g. Python's ``re`` rejects an
                # ECMA-262 escape the pattern uses.
                continue
            checked += 1
            actual, detail = _jsonschema_check(
                params, {"value": data} if wrapped else data
            )
            if actual is not expected:
                failures.append(
                    f"{label} {test['description']!r}: expected {expected}, "
                    f"got {actual} ({detail})"
                )
            if wrapped and expected:
                ok, detail = _jsonschema_check(
                    params, {"value": data, "extra": 0}
                )
                if ok is not False:
                    failures.append(
                        f"{label}: wrapper accepts an extra key ({detail})"
                    )

    assert checked > 0
    assert not failures, "\n".join(failures)
