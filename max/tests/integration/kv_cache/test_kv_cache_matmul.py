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

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.kv_cache import PagedKVCacheManager
from max.mlir import StringAttr
from max.nn.kernels import (
    fused_qkv_ragged_matmul,
    matmul_k_cache_ragged,
    matmul_kv_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.pipelines import TextContext
from modular_graph_test import modular_graph_test
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack


class KeyOrValue(Enum):
    KEY = 0
    VALUE = 1


def _dump_k_cache_to_torch_tensor(
    cache: PagedKVCacheManager, ctx: TextContext, device_id: int = 0
) -> torch.Tensor:
    """
    Returns a torch tensor of the shape [seq_len, num_layers, n_heads, head_dim]

    This should only be used for testing purposes.
    """
    return _dump_k_or_v_cache_to_torch_tensor(
        cache, ctx, device_id, KeyOrValue.KEY
    )


def _dump_v_cache_to_torch_tensor(
    cache: PagedKVCacheManager, ctx: TextContext, device_id: int = 0
) -> torch.Tensor:
    """
    Returns a torch tensor of the shape [seq_len, num_layers, n_heads, head_dim]

    This should only be used for testing purposes.
    """
    return _dump_k_or_v_cache_to_torch_tensor(
        cache, ctx, device_id, KeyOrValue.VALUE
    )


def _dump_k_or_v_cache_to_torch_tensor(
    cache: PagedKVCacheManager,
    ctx: TextContext,
    device_id: int = 0,
    key_or_value: KeyOrValue = KeyOrValue.KEY,
) -> torch.Tensor:
    """
    Returns a torch tensor of the shape [seq_len, num_layers, n_heads, head_dim]

    This should only be used for testing purposes.
    """
    req_blocks = cache.get_req_blocks(ctx.request_id, replica_idx=0)

    torch_dtype = cache.params.dtype.to_torch()

    # [total_num_pages, kv_dim, num_layers, page_size, n_heads, head_dim]
    device_tensor = cache.get_device_tensors(replica_idx=0)[device_id]
    device_tensor_torch = from_dlpack(device_tensor).to(torch_dtype).cpu()

    # [total_num_pages, num_layers, page_size, n_heads, head_dim]
    device_tensor_torch = device_tensor_torch[:, key_or_value.value, :, :, :, :]

    # [seq_len, num_layers, n_heads, head_dim]
    seq_len = ctx.tokens.processed_length
    res = torch.empty(
        (
            seq_len,
            cache.params.num_layers,
            cache.params.n_kv_heads_per_device,
            cache.params.head_dim,
        ),
        dtype=torch_dtype,
    )

    for start_idx in range(0, seq_len, cache.page_size):
        end_idx = min(start_idx + cache.page_size, seq_len)

        block_id = req_blocks[start_idx // cache.page_size]

        # [num_layers, page_size, n_heads, head_dim]
        block_torch = device_tensor_torch[block_id, :]

        for token_idx in range(start_idx, end_idx):
            res[token_idx, :, :, :] = block_torch[
                :, token_idx % cache.page_size, :, :
            ]

    return res


def test_fused_qkv_ragged_matmul(session: InferenceSession) -> None:
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.float32,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wqkv_type = TensorType(
        DType.float32,
        [
            num_q_heads * kv_params.head_dim,
            (num_q_heads + 2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        [
            "input_row_offsets_len",
        ],
        device=DeviceRef.CPU(),
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )
    blocks_type, cache_lengths_type, lookup_table_type, is_cache_empty_type = (
        kv_params.get_symbolic_inputs()[0]
    )

    def construct() -> Graph:
        with Graph(
            "call_ragged_qkv_matmul",
            input_types=[
                input_type,
                input_row_offsets_type,
                wqkv_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                is_cache_empty_type,
            ],
        ) as g:
            (
                input,
                input_row_offsets,
                wqkv,
                blocks,
                cache_lengths,
                lookup_table,
                is_cache_empty,
            ) = g.inputs
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                is_cache_empty.tensor,
            )
            result = fused_qkv_ragged_matmul(
                kv_params,
                input.tensor,
                input_row_offsets.tensor,
                wqkv.tensor,
                kv_collection,
                layer_idx,
                num_q_heads,
            )
            g.output(result)
        return g

    g = construct()

    # Create contexts and claim seq_ids in cache
    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum
    blocks, cache_lengths, lookup_table_tensor, is_cache_empty_buf = (
        kv_manager.runtime_inputs([batch])[0]
    )

    @modular_graph_test(
        session,
        g,
        static_dims={
            "total_seq_len": total_seq_len,
            "input_row_offsets_len": len(prompt_lens) + 1,
        },
        provided_inputs={
            1: input_row_offsets,
            3: blocks,
            4: cache_lengths,
            5: lookup_table_tensor,
            6: is_cache_empty_buf,
        },
    )
    def test_runs_without_nan(
        execute: Callable[[Sequence[Buffer]], Buffer],
        inputs: Sequence[Buffer],
        torch_inputs: Sequence[torch.Tensor],
    ) -> None:
        inputs = list(inputs)
        result = execute(inputs).to_numpy()
        assert np.all(np.isfinite(result))


@dataclass(frozen=True)
class MatmulKVRaggedModel:
    """Model containing a single matmul KV ragged op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *kv_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_kv_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=PagedCacheValues(
                kv_blocks=kv_inputs[0].buffer,
                cache_lengths=kv_inputs[1].tensor,
                lookup_table=kv_inputs[2].tensor,
                max_lengths=kv_inputs[3].tensor,
            ),
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
        )


@pytest.mark.parametrize(
    "dtype",
    [
        DType.float32,
        # TODO(bduke): support converting to torch tensor from bfloat16 driver
        # tensor.
        # DType.bfloat16,
    ],
)
def test_matmul_kv_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_kv_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_params.get_symbolic_inputs()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Create contexts and claim seq_ids in cache.
    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[i] = running_sum
    kv_inputs = kv_manager.runtime_inputs([batch])[0]
    kv_blocks = kv_inputs[0]
    # First check that the KV cache was zeroed out on initialization.
    assert not kv_blocks.to_numpy().any()

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wkv = torch.randn(size=wkv_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wkv, *kv_inputs)

    # Check that the matmul wrote output to the KV cache.
    assert kv_blocks.to_numpy().any()


@dataclass(frozen=True)
class MatmulKRaggedModel:
    """Model containing a single matmul KV ragged op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        hidden_states: TensorValue,
        input_row_offsets: TensorValue,
        weight: TensorValue,
        *kv_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        matmul_k_cache_ragged(
            self.kv_params,
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection=PagedCacheValues(
                kv_blocks=kv_inputs[0].buffer,
                cache_lengths=kv_inputs[1].tensor,
                lookup_table=kv_inputs[2].tensor,
                max_lengths=kv_inputs[3].tensor,
            ),
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
        )


@pytest.mark.parametrize("dtype", [DType.float32])
def test_matmul_k_ragged(session: InferenceSession, dtype: DType) -> None:
    """Tests the matmul_k_cache_ragged custom op."""
    # Set up hyperparameters for the test.
    page_size = 128
    torch_dtype = {
        DType.float32: torch.float32,
        DType.bfloat16: torch.bfloat16,
    }[dtype]
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        cache_strategy="paged",
        page_size=page_size,
        devices=[DeviceRef.CPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wk_type = TensorType(
        dtype,
        [
            kv_params.n_kv_heads * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )
    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    graph = Graph(
        "matmul_k_cache_ragged",
        forward=MatmulKRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wk_type,
            *kv_params.get_symbolic_inputs()[0],
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Create contexts and claim seq_ids in cache.
    batch = []
    for i in range(batch_size):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context.request_id, replica_idx=0)
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        batch.append(context)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    kv_inputs = kv_manager.runtime_inputs([batch])[0]

    hidden_states = torch.randn(
        size=[total_seq_len, num_q_heads * kv_params.head_dim],
        dtype=torch_dtype,
    )
    wk = torch.randn(size=wk_type.shape.static_dims, dtype=torch_dtype)
    model(hidden_states, input_row_offsets, wk, *kv_inputs)

    ref_results = hidden_states @ wk.T

    for batch_idx, ctx in enumerate(batch):
        ctx.update(999)
        k_cache = _dump_k_cache_to_torch_tensor(kv_manager, ctx)

        # Calculate starting position for this batch
        seq_start = (
            int(np.cumsum(prompt_lens[:batch_idx])) if batch_idx != 0 else 0
        )
        seq_len = prompt_lens[batch_idx]

        expected = ref_results[seq_start : (seq_start + seq_len), :]

        torch.testing.assert_close(
            k_cache.reshape([seq_len, -1]),
            expected,
            rtol=5e-4,
            atol=5e-4,
        )


@pytest.mark.parametrize(
    "dtype",
    [DType.float32, DType.bfloat16],
)
def test_matmul_kv_cache_ragged_chains(dtype: DType) -> None:
    """Tests that staging matmul_kv_cache_ragged threads chains."""
    # Set up hyperparameters for the test.
    num_q_heads = 32
    kv_params = KVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        cache_strategy="paged",
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    # Set MLIR types for the graph.
    hidden_state_type = TensorType(
        dtype,
        ["total_seq_len", num_q_heads * kv_params.head_dim],
        device=DeviceRef.CPU(),
    )
    wkv_type = TensorType(
        dtype,
        [
            (2 * (kv_params.n_kv_heads)) * kv_params.head_dim,
            num_q_heads * kv_params.head_dim,
        ],
        device=DeviceRef.CPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        ["input_row_offsets_len"],
        device=DeviceRef.CPU(),
    )

    kv_manager = PagedKVCacheManager(
        kv_params,
        total_num_pages=8,
        session=InferenceSession(devices=[CPU()]),
        max_batch_size=128,
    )
    # Stage the fetch op + custom matmul KV cache ragged op graph.
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=MatmulKVRaggedModel(kv_params, layer_idx=0),
        input_types=[
            hidden_state_type,
            input_row_offsets_type,
            wkv_type,
            *kv_params.get_symbolic_inputs()[0],
        ],
    )
    matmul_kv_cache_op = [
        op
        for op in graph._mlir_op.regions[0].blocks[0].operations
        if op.name == "mo.custom"
        and "kv_matmul" in StringAttr(op.attributes["symbol"]).value
    ][0]
    assert len(matmul_kv_cache_op.results) == 1
    assert "!mo.chain" in str(matmul_kv_cache_op.results[-1].type)

    matmul_args = matmul_kv_cache_op.operands
    assert "!mo.chain" in str(matmul_args[-1].type)
