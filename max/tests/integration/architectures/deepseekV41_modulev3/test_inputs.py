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

from max.driver import Buffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputs, KVCacheInputsPerDevice
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
) -> DeepseekV41Inputs:
    """A populated inputs object over the given KV tree."""
    return DeepseekV41Inputs(
        tokens=_buf(),
        input_row_offsets=_buf(),
        return_n_logits=_buf(),
        extras=(_buf(), _buf()),
        kv_cache_inputs=kv_cache_inputs,
    )


def test_buffers_order() -> None:
    kv_cache_inputs, kv_leaves = _kv_tree()
    inputs = _inputs(kv_cache_inputs)
    assert inputs.buffers == (
        inputs.tokens,
        inputs.return_n_logits,
        inputs.input_row_offsets,
        *kv_leaves,
        *inputs.extras,
    )
