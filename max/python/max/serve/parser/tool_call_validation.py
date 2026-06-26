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
"""Observability-only schema-conformance check for generated tool calls.

Runs JSON Schema validation on parsed tool-call arguments purely to emit
structured, PII-free signals to the serve log. It never mutates the response,
never repairs arguments, and never raises into the request path. The goal is to
turn the coarse production "Schema Mismatch" rate into a per-keyword/per-path
distribution so the failure modes can be sized.

PII: only schema-defined names reach the log -- the function name, the failing
validator keyword, and the JSON path (object/property names, which come from
the developer-supplied schema). Argument *values* and validator error messages
(which embed the offending value) are never logged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

logger = logging.getLogger("max.serve")

# Cap errors recorded per call so a deeply-wrong object can't bloat a log line.
_MAX_ERRORS_PER_CALL = 5

# Compiled validators keyed by a stable schema string. Building a validator is
# the only non-trivial cost (validation itself is microseconds); caching keeps
# this off the hot path. Dict writes are atomic under the GIL and a concurrent
# double-build is idempotent, so no lock is needed.
_VALIDATOR_CACHE: dict[str, Validator] = {}

Outcome = Literal["valid", "invalid_json", "unknown_tool", "schema_mismatch"]


@dataclass
class ToolCallConformance:
    """Result of validating one tool call's arguments against its schema."""

    function: str
    outcome: Outcome
    # "keyword@json_path" pairs; schema-defined names only, no instance values.
    errors: list[str] = field(default_factory=list)


def _validator_for(schema: Mapping[str, Any]) -> Validator | None:
    """Returns a cached validator for *schema*, or ``None`` if unbuildable.

    Validates under the 2020-12 dialect, which is what ``jsonschema`` selects
    by default for schemas that omit ``$schema`` (as tool-parameter schemas
    do), matching the dialect a plain ``jsonschema.validate`` would use.
    """
    try:
        key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    cached = _VALIDATOR_CACHE.get(key)
    if cached is not None:
        return cached
    # check_schema rejects an invalid tool-definition schema (SchemaError); a
    # bad tool definition is not a model failure, so skip rather than blame it.
    # Broad except keeps this observability path from ever raising.
    try:
        Draft202012Validator.check_schema(dict(schema))
        validator: Validator = Draft202012Validator(dict(schema))
    except Exception:
        return None
    _VALIDATOR_CACHE[key] = validator
    return validator


def check_tool_call_conformance(
    calls: list[tuple[str, object]],
    schemas_by_name: Mapping[str, Mapping[str, Any]],
) -> list[ToolCallConformance]:
    """Validates each ``(name, arguments)`` call against its declared schema.

    Pure and side-effect free; never raises. ``arguments`` is the raw JSON
    string emitted by the model (an already-decoded mapping is also accepted).
    An empty/whitespace argument string is treated as ``{}`` (a no-arg call).
    A schema that cannot be compiled yields ``valid`` -- this check never
    invents a failure it cannot substantiate.
    """
    results: list[ToolCallConformance] = []
    for name, raw_args in calls:
        schema = schemas_by_name.get(name)
        if schema is None:
            results.append(ToolCallConformance(name, "unknown_tool"))
            continue

        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                results.append(ToolCallConformance(name, "invalid_json"))
                continue
        else:
            parsed = raw_args

        validator = _validator_for(schema)
        if validator is None:
            results.append(ToolCallConformance(name, "valid"))
            continue

        errors: list[str] = []
        for err in validator.iter_errors(parsed):
            errors.append(f"{err.validator}@{err.json_path}")
            if len(errors) >= _MAX_ERRORS_PER_CALL:
                break
        results.append(
            ToolCallConformance(
                name, "schema_mismatch" if errors else "valid", errors
            )
        )
    return results


def log_tool_call_conformance(
    calls: list[tuple[str, object]],
    schemas_by_name: Mapping[str, Mapping[str, Any]],
    *,
    request_id: str,
    streaming: bool,
) -> None:
    """Logs schema-conformance of generated tool calls (observability only).

    Emits one WARNING line per *non-conforming* call; conforming calls are
    silent to keep log volume proportional to failures. Never raises into the
    caller. See the module docstring for the PII guarantee.
    """
    try:
        results = check_tool_call_conformance(calls, schemas_by_name)
    except Exception:
        logger.debug("tool_call_conformance check failed", exc_info=True)
        return
    for r in results:
        if r.outcome == "valid":
            continue
        logger.warning(
            "tool_call_conformance req=%s stream=%s fn=%s outcome=%s errors=%s",
            request_id,
            streaming,
            r.function,
            r.outcome,
            ",".join(r.errors) if r.errors else "-",
        )
