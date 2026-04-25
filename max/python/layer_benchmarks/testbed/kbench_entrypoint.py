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

"""Unified kbench entry point for the model test bed.

Usage::

    kbench bench_attention_with_rope.yaml

The YAML config must include ``$harness`` (and optionally ``$dtype``) as
params. These are extracted from the config bundle to select the harness
class; remaining keys are split into static params (model config) and
dynamic params (per-shape dimensions).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from benchmark_utils import dump_kbench_csv, print_results_table
from cyclopts import App

# Import harnesses to trigger registration.
import testbed.harnesses  # noqa: F401
from testbed.harness import dict_to_dataclass
from testbed.registry import HARNESS_REGISTRY
from testbed.runner import LayerTestRunner, create_session

# Dynamic param keys recognized as per-shape dimensions.
# TODO: If new harnesses introduce novel dynamic keys (e.g. num_experts,
# image_tokens), consider adding a dynamic_param_keys property to the
# harness ABC so each harness declares its own shape keys.
_SHAPE_KEYS = {"batch_size", "seq_len", "ctx_len", "context_len"}

# Keys extracted from config entries to control the runner, not passed
# to the harness as static params.
_CONTROL_KEYS = {"harness"}

app = App(
    name="kbench-testbed",
    help="Model test bed: unified kbench entry point",
)


def parse_testbed_config_bundle(
    bundle_json: str,
    shape_keys: set[str] | None = None,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Parse a kbench config bundle into (static_params, shapes) groups.

    Entries with the same static params are grouped together so the graph
    is compiled once per unique config.

    Args:
        bundle_json: JSON string from kbench --single-invoke.
        shape_keys: Set of keys to treat as dynamic (per-shape) params.
            Defaults to _SHAPE_KEYS.

    Returns:
        List of (static_params, list_of_dynamic_params) tuples.
    """
    if shape_keys is None:
        shape_keys = _SHAPE_KEYS

    entries = json.loads(bundle_json)
    groups: OrderedDict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = (
        OrderedDict()
    )

    for entry in entries:
        static = {}
        dynamic = {}
        for k, v in entry.items():
            if k in shape_keys:
                dynamic[k] = v
            else:
                static[k] = v

        # Use a hashable key for grouping.
        group_key = json.dumps(static, sort_keys=True)
        if group_key not in groups:
            groups[group_key] = (static, [])
        groups[group_key][1].append(dynamic)

    return list(groups.values())


@app.default
def main(
    *,
    config_bundle: str,
    iterations: int = 50,
    warmup: int = 5,
    output: str | None = None,
) -> None:
    """Run benchmarks from a kbench config bundle.

    Args:
        config_bundle: JSON config bundle from kbench single-invoke mode.
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations.
        output: Output CSV path (injected by kbench).
    """
    configs = parse_testbed_config_bundle(config_bundle)

    all_results: list[tuple[str, Any]] = []
    last_harness_name = ""

    for static_params, shapes in configs:
        # Extract harness name from the config bundle.
        harness_name = static_params.get("harness")
        if harness_name is None:
            raise ValueError(
                "Config bundle entries must include a 'harness' key "
                f"(one of {list(HARNESS_REGISTRY.keys())})"
            )
        if harness_name not in HARNESS_REGISTRY:
            raise ValueError(
                f"Unknown harness '{harness_name}', "
                f"expected one of {list(HARNESS_REGISTRY.keys())}"
            )

        harness_cls = HARNESS_REGISTRY[harness_name]
        # Filter out control keys before passing to the harness.
        harness_params = {
            k: v for k, v in static_params.items() if k not in _CONTROL_KEYS
        }

        # Convert config dicts to typed dataclasses.
        harness_params = dict_to_dataclass(
            harness_cls.static_params_type(), harness_params
        )

        session, device = create_session()
        harness = harness_cls(harness_params, session, device)
        runner = LayerTestRunner(harness)

        shapes = [
            dict_to_dataclass(harness_cls.dynamic_params_type(), s)
            for s in shapes
        ]

        print(f"Compiling {harness.name} graph...")
        results = runner.benchmark(shapes, iterations, warmup)
        all_results.extend(results)

        config_name = static_params.get(
            "model_name", static_params.get("name", harness.name)
        )
        last_harness_name = harness.name
        print_results_table(f"{harness.name} [{config_name}]", results)

    if output:
        dump_kbench_csv(last_harness_name, all_results, output)


if __name__ == "__main__":
    app()
