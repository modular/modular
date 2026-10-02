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

import numpy as np
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    store_k_cache_padded,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

TOTAL_NUM_PAGES = 16


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=32,
        devices=[DeviceRef.GPU()],
    )


def test_kv_cache_store_ragged_executes() -> None:
    device = Accelerator()
    kv_params = _kv_params()

    prompt_lens = [33, 66, 1]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    x_cache_type = TensorType(
        DType.float32,
        [total_seq_len, kv_params.n_kv_heads, kv_params.head_dim],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=DeviceRef.GPU(),
    )

    with Graph(
        "kv_cache_store_ragged",
        input_types=[
            x_cache_type,
            offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        x_cache_in, input_row_offsets_in, *_kv_rest = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(
            kv_collection,
            x_cache_in.tensor,
            input_row_offsets_in.tensor,
            layer_idx,
        )
        graph.output(x_cache_in.tensor)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=TOTAL_NUM_PAGES
    )
    assert not runtime_inputs.kv_blocks.to_numpy().any()

    offsets = np.array(
        [0, prompt_lens[0], prompt_lens[0] + prompt_lens[1], total_seq_len],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(0)
    x_cache_np = rng.standard_normal(
        x_cache_type.shape.static_dims, dtype=np.float32
    )
    x_cache_data = Buffer.from_numpy(x_cache_np).to(device)
    offsets_data = Buffer.from_numpy(offsets).to(device)
    model(
        x_cache_data,
        offsets_data,
        *tree.leaves(runtime_inputs),
    )

    assert runtime_inputs.kv_blocks.to_numpy().any()


def test_kv_cache_store_padded_executes() -> None:
    device = Accelerator()
    kv_params = _kv_params()

    valid_lengths = [33, 66, 1]
    batch_size = len(valid_lengths)
    padded_seq_len = max(valid_lengths)

    x_cache_type = TensorType(
        DType.float32,
        [batch_size, padded_seq_len, kv_params.n_kv_heads, kv_params.head_dim],
        device=DeviceRef.GPU(),
    )
    valid_lengths_type = TensorType(
        DType.uint32,
        [batch_size],
        device=DeviceRef.GPU(),
    )

    with Graph(
        "kv_cache_store_padded",
        input_types=[
            x_cache_type,
            valid_lengths_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        x_cache_in, valid_lengths_in, *_kv_rest = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_padded(
            kv_collection,
            x_cache_in.tensor,
            valid_lengths_in.tensor,
            layer_idx,
        )
        graph.output(x_cache_in.tensor)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, valid_lengths, total_num_pages=TOTAL_NUM_PAGES
    )
    assert not runtime_inputs.kv_blocks.to_numpy().any()

    lengths = np.array(valid_lengths, dtype=np.uint32)
    rng = np.random.default_rng(1)
    x_cache_np = rng.standard_normal(
        x_cache_type.shape.static_dims, dtype=np.float32
    )
    x_cache_data = Buffer.from_numpy(x_cache_np).to(device)
    lengths_data = Buffer.from_numpy(lengths).to(device)
    model(
        x_cache_data,
        lengths_data,
        *tree.leaves(runtime_inputs),
    )

    assert runtime_inputs.kv_blocks.to_numpy().any()


def _kv_params_fp8() -> MHAKVCacheParams:
    """Page layout for an FP8 quantized cache, which also carries kv_scales."""
    return MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=128,
        ),
    )


def test_store_k_scale_cache_executes() -> None:
    """Test that store_k_scale_cache kernel executes and writes to kv_scales buffer."""
    device = Accelerator()
    kv_params = _kv_params_fp8()

    prompt_lens = [33, 66, 1]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    assert kv_params.kvcache_quant_config is not None
    quantization_granularity = (
        kv_params.kvcache_quant_config.quantization_granularity
    )
    head_dim_granularity = kv_params.head_dim // quantization_granularity

    x_k_scale_type = TensorType(
        DType.float32,
        [total_seq_len, kv_params.n_kv_heads, head_dim_granularity],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=DeviceRef.GPU(),
    )

    kv_symbolic_inputs = kv_params.get_symbolic_inputs()[0]

    with Graph(
        "store_k_scale_cache",
        input_types=[
            x_k_scale_type,
            offsets_type,
            *tree.leaves(kv_symbolic_inputs),
        ],
    ) as graph:
        x_k_scale_in = graph.inputs[0].tensor
        input_row_offsets_in = graph.inputs[1].tensor

        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]

        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_scale_cache_ragged(
            kv_collection,
            x_k_scale_in,
            input_row_offsets_in,
            layer_idx,
            quantization_granularity,
        )
        graph.output(x_k_scale_in)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )
    assert runtime_inputs.kv_scales is not None
    assert not runtime_inputs.kv_scales.to_numpy().any()

    offsets = np.array(
        [0, prompt_lens[0], prompt_lens[0] + prompt_lens[1], total_seq_len],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(42)
    x_k_scale_np = rng.standard_normal(
        x_k_scale_type.shape.static_dims, dtype=np.float32
    )
    x_k_scale_data = Buffer.from_numpy(x_k_scale_np).to(device)
    offsets_data = Buffer.from_numpy(offsets).to(device)

    model(
        x_k_scale_data,
        offsets_data,
        *tree.leaves(runtime_inputs),
    )

    assert runtime_inputs.kv_scales.to_numpy().any()


def _kv_params_narrow() -> MHAKVCacheParams:
    """Small enough that every written element can be checked by index."""
    return MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=4,
        num_layers=1,
        page_size=16,
        devices=[DeviceRef.GPU()],
    )


# Covered rows per request, then the rows past `input_row_offsets[-1]`.
_COVERED = [2, 3]
_SURPLUS = 4


def test_store_ragged_ignores_rows_past_the_last_offset() -> None:
    """Surplus source rows must write nothing, not extend the last request.

    A producer whose output row count is only an upper bound on a
    data-dependent one hands the store rows that belong to no request --
    `mla_kpool_compress`, whose pool count is a sum of per-request floors, is
    one. The store grids on the source shape, so it sees those rows, and the
    batch search answers with the last request for every one of them. Before
    the bound below existed they were written at `cache_length + (row -
    offsets[batch - 1])`, walking off the end of that request's span and into
    whatever pages follow. Only a debug `assert` in
    `get_batch_from_row_offsets` noticed, and it compiles out at the default
    assert level.
    """
    device = Accelerator()
    kv_params = _kv_params_narrow()

    batch_size = len(_COVERED)
    covered_rows = sum(_COVERED)
    total_rows = covered_rows + _SURPLUS
    # A page-aligned start puts each request's new rows at slot 0 of its
    # second page, so an expected position is a page id and a slot.
    cache_lengths = [kv_params.page_size] * batch_size

    x_cache_type = TensorType(
        DType.float32,
        [total_rows, kv_params.n_kv_heads, kv_params.head_dim],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32, [batch_size + 1], device=DeviceRef.GPU()
    )

    with Graph(
        "kv_cache_store_ragged_surplus_rows",
        input_types=[
            x_cache_type,
            offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        x_cache_in, input_row_offsets_in, *_rest = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]
        store_k_cache_ragged(
            kv_collection,
            x_cache_in.tensor,
            input_row_offsets_in.tensor,
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
        )
        graph.output(x_cache_in.tensor)

    model = InferenceSession(devices=[device]).load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, _COVERED, cache_lengths=cache_lengths
    )
    assert not runtime_inputs.kv_blocks.to_numpy().any()

    # Row `i` carries the value `i + 1`, so a misplaced row is identifiable
    # and no written row can be mistaken for the zero-filled pool.
    x_cache_np = np.broadcast_to(
        np.arange(1, total_rows + 1, dtype=np.float32).reshape(-1, 1, 1),
        x_cache_type.shape.static_dims,
    ).copy()
    offsets = np.array([0, *np.cumsum(_COVERED)], dtype=np.uint32)

    model(
        Buffer.from_numpy(x_cache_np).to(device),
        Buffer.from_numpy(offsets).to(device),
        *tree.leaves(runtime_inputs),
    )

    blocks = runtime_inputs.kv_blocks.to_numpy()
    lookup_table = runtime_inputs.lookup_table.to_numpy()

    for request, n_covered in enumerate(_COVERED):
        page = lookup_table[
            request, cache_lengths[request] // kv_params.page_size
        ]
        for slot in range(n_covered):
            expected = float(offsets[request] + slot + 1)
            np.testing.assert_array_equal(
                blocks[page, 0, 0, slot, 0, :],
                np.full(kv_params.head_dim, expected, dtype=np.float32),
            )

    # The surplus rows carry distinct non-zero values, so the written element
    # count is what separates "ignored" from "written somewhere".
    assert np.count_nonzero(blocks) == covered_rows * kv_params.head_dim


def test_store_k_scale_ragged_ignores_rows_past_the_last_offset() -> None:
    """The scale store shares the value store's bound and its failure mode."""
    device = Accelerator()
    kv_params = _kv_params_fp8()

    batch_size = len(_COVERED)
    covered_rows = sum(_COVERED)
    total_rows = covered_rows + _SURPLUS
    cache_lengths = [kv_params.page_size] * batch_size

    assert kv_params.kvcache_quant_config is not None
    quantization_granularity = (
        kv_params.kvcache_quant_config.quantization_granularity
    )
    head_dim_granularity = kv_params.head_dim // quantization_granularity

    x_k_scale_type = TensorType(
        DType.float32,
        [total_rows, kv_params.n_kv_heads, head_dim_granularity],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32, [batch_size + 1], device=DeviceRef.GPU()
    )

    with Graph(
        "store_k_scale_cache_surplus_rows",
        input_types=[
            x_k_scale_type,
            offsets_type,
            *tree.leaves(kv_params.get_symbolic_inputs()[0]),
        ],
    ) as graph:
        kv_collection = kv_params.unflatten_kv_inputs(iter(graph.inputs[2:]))[0]
        store_k_scale_cache_ragged(
            kv_collection,
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            quantization_granularity,
        )
        graph.output(graph.inputs[0].tensor)

    model = InferenceSession(devices=[device]).load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, _COVERED, cache_lengths=cache_lengths
    )
    assert runtime_inputs.kv_scales is not None
    assert not runtime_inputs.kv_scales.to_numpy().any()

    x_k_scale_np = np.broadcast_to(
        np.arange(1, total_rows + 1, dtype=np.float32).reshape(-1, 1, 1),
        x_k_scale_type.shape.static_dims,
    ).copy()
    offsets = np.array([0, *np.cumsum(_COVERED)], dtype=np.uint32)

    model(
        Buffer.from_numpy(x_k_scale_np).to(device),
        Buffer.from_numpy(offsets).to(device),
        *tree.leaves(runtime_inputs),
    )

    scales = runtime_inputs.kv_scales.to_numpy()
    assert np.count_nonzero(scales) == covered_rows * head_dim_granularity
