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
"""Bitwise parity for the head-major fused rotation and K/V store.

Each fused op is graded against the unfused sequence it replaces, both arms
built by ``rope_qkv_store_testlib``. This target grades the drafter's shipping
geometry and passes a ragged batch across page boundaries, where a row can land
on a page the batch does not start on; the sweep in
``test_rope_qkv_store_sweep_gpu`` carries the table dtypes, rotary widths and
value scales. Both are manual: they compare bytes against a reference chain.
The ops' surface without a GPU -- one op each, and the store's place in the
cache's mutation chain -- is ``test_rope_qkv_store_ops_cpu``, which is what
presubmit runs.

Why this is a Python test rather than a Mojo kernel test: what is under test
is the composition -- projection layout, position, one BF16 rounding, and the
order of the writes against a graph consumer of the mutation chain -- where
the primitives it composes (``rope_value``, the paged store) already carry
Mojo coverage. The shapes are the drafter's decode sizes on one B200.
"""

from __future__ import annotations

import pytest
import torch
from rope_qkv_store_testlib import (
    FUSED_KV,
    FUSED_QKV,
    ROPE_KV,
    ROPE_QKV,
    SHAPES,
    Compiled,
    assert_parity,
    bits,
    case_inputs,
    compile_arms,
    fill_caches,
    graph_inputs,
)


@pytest.fixture(
    scope="module",
    params=[pytest.param((SHAPES[0], 0.1), id="drafter-tp4-fp32-0.1")],
)
def compiled(request: pytest.FixtureRequest) -> Compiled:
    shape, scale = request.param
    return compile_arms(shape, scale)


@pytest.mark.parametrize(
    ("prompt_lens", "cache_lens"),
    [([1], [129]), ([1, 2, 5, 3], [0, 127, 128, 255])],
    ids=["single-decode", "ragged-page-boundaries"],
)
@pytest.mark.parametrize(
    ("reference", "fused"),
    [(ROPE_QKV, FUSED_QKV), (ROPE_KV, FUSED_KV)],
    ids=["with-query", "no-query"],
)
def test_the_fused_store_matches_the_unfused_sequence(
    reference: str,
    fused: str,
    prompt_lens: list[int],
    cache_lens: list[int],
    compiled: Compiled,
) -> None:
    tokens = sum(prompt_lens)
    values, caches = case_inputs(
        compiled, tokens, prompt_lens, cache_lens, num_caches=2
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
    assert_parity(fused, prompt_lens, cache_lens, caches, outputs, initial)


@pytest.mark.parametrize("mode", [FUSED_QKV, FUSED_KV])
def test_a_batch_that_ends_before_its_tokens_writes_nothing_more(
    mode: str, compiled: Compiled
) -> None:
    """The grid covers the token extent, so a batch that ends before it
    leaves a tail that must return before its first store."""
    prompt_lens = [1, 2]
    owned_rows = sum(prompt_lens)
    owned_values, caches = case_inputs(
        compiled, owned_rows, prompt_lens, [128, 128], num_caches=2
    )
    # The same operands with one more token row, which no batch owns. The
    # graph's token extent is dynamic, so one model serves both runs.
    over_values, _ = case_inputs(
        compiled, owned_rows + 1, prompt_lens, [128, 128]
    )
    fill_caches(caches, compiled.device)
    model = compiled.models[mode]
    model.execute(*graph_inputs(mode, owned_values, caches[0]))
    model.execute(*graph_inputs(mode, over_values, caches[1]))
    assert torch.equal(bits(caches[0].kv_blocks), bits(caches[1].kv_blocks)), (
        f"{mode} wrote the token row past the final offset"
    )
