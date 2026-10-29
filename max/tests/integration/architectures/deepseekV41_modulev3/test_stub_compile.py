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
"""Compiles the V4.1 stub on SM100 virtual devices and checks what it holds.

Linux only: the graph compiler has no NVPTX target on macOS. Each cell runs
in a fresh process, since virtual devices latch their count and target at the
first device a process creates.
"""

from __future__ import annotations

import multiprocessing
import re
from typing import Any

import abi_contract
import pytest
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    parameter_specs,
)

_SHARD_SUFFIX = re.compile(r"\._shard\.\d+$")
_SUBGRAPHS = [
    "swa_only",
    "full_r2",
    "reuse_r2",
    "full_r1_cand",
    "reindex_r1",
    "reuse_r1",
    "engram",
]


def _run(model: str, tp: int, out: Any) -> None:
    out.put(abi_contract.run_cell(model, tp, count_bodies=True))


@pytest.mark.parametrize(
    ("model", "tp"),
    [("f1", 1), ("f1", 2), ("f1", 4), ("f1", 8), ("full", 8)],
)
def test_compiles_and_registry_names_equal_the_tree(
    model: str, tp: int
) -> None:
    queue = multiprocessing.get_context("spawn").Queue()
    process = multiprocessing.get_context("spawn").Process(
        target=_run, args=(model, tp, queue)
    )
    process.start()
    result, registry = queue.get()
    process.join()
    assert process.exitcode == 0

    config = V41Config.from_hf(abi_contract.hf_config(model))
    tree = {spec.name for spec in parameter_specs(config, tp)}
    assert {_SHARD_SUFFIX.sub("", name) for name in registry} == tree
    assert sorted(result["subgraphs"]) == sorted(_SUBGRAPHS)
