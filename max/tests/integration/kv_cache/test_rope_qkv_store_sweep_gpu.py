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
"""The rotation and store parity sweep: table dtype, rotary width, value scale.

The fused store's parity is gated by the Mojo kernel test in
``max/kernels/test/gpu/kv_cache/test_rope_qkv_store_paged.mojo`` and, on the
graph side, by the CPU op-surface and mutation-chain check in
``test_rope_qkv_store_ops_cpu.py``; this target carries the rest of the sweep
and is manual. It lays every case out as the shipping geometry's ragged
batch, where a row lands across a page boundary and a batch starts at a page
boundary.
"""

from __future__ import annotations

import pytest
from rope_qkv_store_testlib import (
    FUSED_KV,
    FUSED_QKV,
    ROPE_KV,
    ROPE_QKV,
    SHAPES,
    Compiled,
    assert_parity,
    case_inputs,
    compile_arms,
    fill_caches,
    graph_inputs,
)

PROMPT_LENS = [1, 2, 5, 3]
CACHE_LENS = [0, 127, 128, 255]


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((SHAPES[1], 0.1), id="drafter-tp4-bf16-table"),
        pytest.param((SHAPES[2], 0.1), id="full-rotary"),
        pytest.param((SHAPES[0], 1.00390625), id="drafter-tp4-fp32-near-one"),
        pytest.param((SHAPES[0], -2.0), id="drafter-tp4-fp32-negative"),
    ],
)
def compiled(request: pytest.FixtureRequest) -> Compiled:
    shape, scale = request.param
    return compile_arms(shape, scale)


@pytest.mark.parametrize(
    ("reference", "fused"),
    [(ROPE_QKV, FUSED_QKV), (ROPE_KV, FUSED_KV)],
    ids=["with-query", "no-query"],
)
def test_the_fused_store_matches_the_unfused_sequence(
    reference: str, fused: str, compiled: Compiled
) -> None:
    values, caches = case_inputs(
        compiled, sum(PROMPT_LENS), PROMPT_LENS, CACHE_LENS, num_caches=2
    )
    initial = fill_caches(caches, compiled.device)
    outputs = [
        tuple(model.execute(*graph_inputs(mode, values, cache)))
        for mode, model, cache in zip(
            (reference, fused),
            (compiled.models[reference], compiled.models[fused]),
            caches,
            strict=True,
        )
    ]
    assert_parity(fused, PROMPT_LENS, CACHE_LENS, caches, outputs, initial)
