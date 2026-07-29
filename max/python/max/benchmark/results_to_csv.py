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

"""Read and flatten benchmark result JSON blobs into CSV-ready rows.

The serving benchmark treats its JSON output as the single source of truth: a
``save_result_json`` blob carries *every* metric the run produced. Turning that
into a spreadsheet-friendly CSV starts by flattening each nested blob into a
flat ``dotted.key`` column namespace. This module provides that ingestion step;
column selection and CSV writing build on top of it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeGuard

from typing_extensions import TypedDict

# A JSON value as produced by ``json.load``. Recursive: objects nest objects
# and lists. Using this instead of a bare ``object`` lets the flattener's
# branches typecheck (``json.dumps`` only ever sees a non-str scalar or list).
JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | Mapping[str, "JSONValue"]
)
# A JSON object: the top level of a single result blob.
JSONObject = Mapping[str, JSONValue]
# One flattened output row: a string cell per (flattened) column name.
Row = dict[str, str]


class ResultEntry(TypedDict, total=False):
    """One ``results`` entry of a ``ResultSetInContext`` document."""

    iteration_config: JSONObject
    result: JSONValue


class ResultSetDocument(TypedDict, total=False):
    """The ``{run_context, results}`` envelope emitted by the JSON reporter."""

    run_context: JSONObject
    results: list[ResultEntry]


def flatten(obj: JSONObject) -> Row:
    """Flattens a nested JSON object into a flat ``dotted.key -> cell`` mapping.

    Nested objects are expanded recursively with dot-separated keys. Lists and
    other non-string scalars are JSON-encoded into a single cell (matching the
    streaming CSV reporter's convention), ``None`` becomes an empty string, and
    strings pass through unchanged.

    Args:
        obj:
            The parsed JSON object (a single result blob) to flatten.

    Returns:
        A mapping from flattened column name to its string cell value.
    """
    flat: Row = {}

    def _walk(value: JSONValue, prefix: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                flat[prefix] = "{}"
                return
            for k, v in value.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                _walk(v, child)
        elif value is None:
            flat[prefix] = ""
        elif isinstance(value, str):
            flat[prefix] = value
        else:
            # Lists, ints, floats, bools: a single JSON-encoded cell. Lists are
            # variable-length per row and cannot become their own columns.
            flat[prefix] = json.dumps(value)

    _walk(obj, "")
    return flat


def _is_result_set_document(
    data: Mapping[str, object],
) -> TypeGuard[ResultSetDocument]:
    """Returns True when ``data`` is a ``ResultSetInContext`` envelope.

    Detected by the presence of a ``results`` list; ``run_context`` and each
    entry's fields are validated where they are read.
    """
    return isinstance(data.get("results"), list)


def _rows_from_result_set(document: ResultSetDocument) -> list[Row]:
    """Flattens a ``ResultSetInContext`` document into one row per result.

    Each row merges the shared ``run_context`` with the entry's
    ``iteration_config`` and ``result``.
    """
    run_context = document.get("run_context")
    rows: list[Row] = []
    for entry in document["results"]:
        record: dict[str, JSONValue] = {}
        if isinstance(run_context, Mapping):
            record["run_context"] = run_context
        iteration_config = entry.get("iteration_config")
        if isinstance(iteration_config, Mapping):
            record["iteration_config"] = iteration_config
        if "result" in entry:
            record["result"] = entry["result"]
        rows.append(flatten(record))
    return rows


def load_result_rows(path: Path) -> list[Row]:
    """Loads a result JSON file and returns one flattened row per result.

    Supports the shapes the serving benchmark emits:

    - a single ``save_result_json`` blob (one row);
    - a JSON array of such blobs (one row each);
    - a ``ResultSetInContext`` document with a ``results`` list, where each
      entry is merged with the shared ``run_context`` and its
      ``iteration_config`` (one row each).

    Args:
        path:
            The path to the result JSON file.

    Returns:
        A list of flattened rows (each a ``column -> cell`` mapping).

    Raises:
        ValueError: If the top-level JSON is not an object or array.
    """
    with open(path) as f:
        data: object = json.load(f)

    if isinstance(data, list):
        return [flatten(item) for item in data]
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{path}: expected a JSON object or array at the top level, "
            f"got {type(data).__name__}"
        )
    if _is_result_set_document(data):
        return _rows_from_result_set(data)
    return [flatten(data)]
