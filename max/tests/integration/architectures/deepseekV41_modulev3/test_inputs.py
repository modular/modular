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
"""Tests the graph-input order `DeepseekV41Inputs` fixes."""

from __future__ import annotations

import pytest
from max.driver import Buffer
from max.dtype import DType
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheInputsPerDevice
from max.pipelines.architectures.deepseekV41_modulev3.inputs import (
    DeepseekV41Inputs,
)


def _buf() -> Buffer:
    return Buffer.zeros((1,), DType.int64)


def _kv_tree() -> tuple[KVCacheInputs[Buffer, Buffer], tuple[Buffer, ...]]:
    """A one-device paged-cache tree and its leaves, in field order."""
    leaves = tuple(_buf() for _ in range(6))
    device = KVCacheInputsPerDevice[Buffer, Buffer](
        kv_blocks=leaves[0],
        cache_lengths=leaves[1],
        lookup_table=leaves[2],
        max_prompt_length=leaves[3],
        max_cache_length=leaves[4],
        page_stride=leaves[5],
    )
    return (device,), leaves


def _inputs(
    kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    data_parallel_splits: Buffer | None,
    input_row_offsets_i64: Buffer | None,
    *,
    ep_inputs: tuple[Buffer, ...] = (),
) -> DeepseekV41Inputs:
    """A populated inputs object over the given KV tree."""
    return DeepseekV41Inputs(
        tokens=_buf(),
        input_row_offsets=_buf(),
        return_n_logits=_buf(),
        extras=(_buf(), _buf()),
        ep_inputs=ep_inputs,
        kv_cache_inputs=kv_cache_inputs,
        data_parallel_splits=data_parallel_splits,
        input_row_offsets_i64=input_row_offsets_i64,
    )


def test_buffers_order_without_dp_pair() -> None:
    kv_cache_inputs, kv_leaves = _kv_tree()
    inputs = _inputs(kv_cache_inputs, None, None)
    assert inputs.buffers == (
        inputs.tokens,
        inputs.return_n_logits,
        inputs.input_row_offsets,
        *kv_leaves,
        *inputs.extras,
    )


def test_buffers_appends_ep_tail_after_extras() -> None:
    kv_cache_inputs, kv_leaves = _kv_tree()
    inputs = _inputs(kv_cache_inputs, None, None, ep_inputs=(_buf(), _buf()))
    # The EP comm buffers trail the extras, where the compile input list
    # appends ``EPBatchManager.input_types`` -- the forward's variadic
    # tail the batch processor binds from the EPCommInitializer.
    assert inputs.buffers == (
        inputs.tokens,
        inputs.return_n_logits,
        inputs.input_row_offsets,
        *kv_leaves,
        *inputs.extras,
        *inputs.ep_inputs,
    )


def test_buffers_splices_dp_pair_after_row_offsets() -> None:
    kv_cache_inputs, kv_leaves = _kv_tree()
    inputs = _inputs(kv_cache_inputs, _buf(), _buf())
    assert inputs.buffers == (
        inputs.tokens,
        inputs.return_n_logits,
        inputs.input_row_offsets,
        # The CPU DP split pair sits between the row offsets and the KV
        # leaves, matching the forward's variadic order.
        inputs.data_parallel_splits,
        inputs.input_row_offsets_i64,
        *kv_leaves,
        *inputs.extras,
    )


@pytest.mark.parametrize("splits_set", [True, False])
def test_half_set_dp_pair_fails_loudly(splits_set: bool) -> None:
    kv_cache_inputs, _ = _kv_tree()
    half = (_buf(), None) if splits_set else (None, _buf())
    with pytest.raises(ValueError, match="set together"):
        _inputs(kv_cache_inputs, *half)
