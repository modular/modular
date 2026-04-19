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

"""Dump MO IR for a harness layer graph.

Usage::

    bazel run //max/python/layer_benchmarks:dump_ir -- \
        --harness attention_with_rope \
        --dtype bfloat16 \
        --mo-path /tmp/attention.mo.mlir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cyclopts import App

import testbed.harnesses  # noqa: F401 — trigger registration
from testbed.harness import dict_to_dataclass
from testbed.ir_dump import dump_mo_ir
from testbed.registry import HARNESS_REGISTRY
from testbed.runner import create_session

app = App(name="dump-ir", help="Dump MO IR for a harness layer graph.")


def _parse_static_params(params_json: str | None) -> dict[str, Any]:
    if params_json is None:
        return {}
    return json.loads(params_json)


@app.default
def main(
    *,
    harness: str,
    mo_path: str,
    dtype: str | None = None,
    params: str | None = None,
) -> None:
    """Dump MO IR for a harness layer graph.

    Args:
        harness: Harness name (e.g. attention_with_rope, rms_norm).
        mo_path: Output path for the .mo.mlir file.
        dtype: Override harness dtype.
        params: Static params as a JSON object.
    """
    if harness not in HARNESS_REGISTRY:
        raise ValueError(
            f"Unknown harness '{harness}', "
            f"expected one of {list(HARNESS_REGISTRY.keys())}"
        )

    raw_params = _parse_static_params(params)
    if dtype:
        raw_params["dtype"] = dtype

    harness_cls = HARNESS_REGISTRY[harness]
    static_params = dict_to_dataclass(
        harness_cls.static_params_type(), raw_params
    )

    session, device = create_session()
    harness_instance = harness_cls(static_params, session, device)

    graph, _ = harness_instance.build_graph()
    out_path = dump_mo_ir(graph, Path(mo_path))
    print(f"MO IR written to {out_path}")


if __name__ == "__main__":
    app()
