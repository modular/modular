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
"""Fixtures shared by the fused rotation-and-store tests and its benchmark.

Four arms cover the two fused ops and the unfused sequences they replace:

* ``rope-qkv`` -- store the raw key, store the float32-scaled value, then
  rotate the key in place and the query through ``fused_qk_ragged_rope``.
  ``fused_rope_qkv_store_ragged`` replaces that for the drafter's block.
* ``rope-kv`` -- the same minus the query, with the key standing in the
  query's slot and the discarded rotation paid for. The context writer's
  ``fused_k_rope_v_store_ragged`` replaces it.

So a parity pass states that the fusion moves work and not numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import (
    fused_k_rope_v_store_ragged,
    fused_qk_ragged_rope,
    fused_rope_qkv_store_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheInputsPerDevice, MHAKVCacheParams
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

# The block's fused op and the sequence it replaces, then the context
# writer's op and its own.
ROPE_QKV = "rope-qkv"
FUSED_QKV = "fused-qkv"
ROPE_KV = "rope-kv"
FUSED_KV = "fused-kv"
MODES = (ROPE_QKV, FUSED_QKV, ROPE_KV, FUSED_KV)


@dataclass(frozen=True)
class Shape:
    """One head geometry and the RoPE table's element type."""

    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    frequency_dtype: DType


SHAPES = [
    # The drafter's published geometry at TP4, rotating through the float32
    # table its context writer builds.
    Shape("drafter-tp4-fp32", 16, 2, 128, 64, DType.float32),
    # The same head rotated through a bfloat16 table, which is the dtype the
    # target's layers carry.
    Shape("drafter-tp4-bf16", 16, 2, 128, 64, DType.bfloat16),
    # Full rotary, where no prefix passes through unrotated.
    Shape("full-rotary", 8, 1, 128, 128, DType.float32),
]

Caches = list[KVCacheInputsPerDevice[Buffer, Buffer]]
Values = tuple[Buffer, Buffer, Buffer, Buffer, Buffer]


def params_of(shape: Shape) -> MHAKVCacheParams:
    """The cache a case's layer stores into."""
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )


def build_graph(
    params: MHAKVCacheParams,
    shape: Shape,
    scale: float,
    *,
    mode: str,
    observe_cache: bool = True,
    frequency_positions: int = 512,
) -> Graph:
    """Builds one arm: a fused op, or the unfused sequence it replaces."""
    with_q = mode.endswith("qkv")
    head_counts = (
        (shape.num_q_heads, shape.num_kv_heads, shape.num_kv_heads)
        if with_q
        else (shape.num_kv_heads, shape.num_kv_heads)
    )
    # The projections, then the offsets and the RoPE table, then the cache.
    leading = len(head_counts) + 2
    with Graph(
        f"rope_qkv_{shape.name}_{mode}_{scale}",
        input_types=[
            *(
                TensorType(
                    DType.bfloat16,
                    ["tokens", heads, shape.head_dim],
                    DeviceRef.GPU(),
                )
                for heads in head_counts
            ),
            TensorType(DType.uint32, ["offsets"], DeviceRef.GPU()),
            TensorType(
                shape.frequency_dtype,
                [frequency_positions, shape.rotary_dim],
                DeviceRef.GPU(),
            ),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        # The cache's own operands are buffer values, which
        # `unflatten_kv_inputs` reads; everything before them is a tensor.
        values = [value.tensor for value in graph.inputs[:leading]]
        cache = params.unflatten_kv_inputs(iter(graph.inputs[leading:]))[0]
        k_index = 1 if with_q else 0
        q = values[0] if with_q else values[k_index]
        k, v = values[k_index], values[k_index + 1]
        offsets, freqs = values[leading - 2], values[leading - 1]
        layer = ops.constant(1, DType.uint32, device=DeviceRef.CPU())
        roped = None
        if mode == FUSED_QKV:
            roped = fused_rope_qkv_store_ragged(
                q, k, v, offsets, freqs, cache, layer, scale
            )
        elif mode == FUSED_KV:
            fused_k_rope_v_store_ragged(
                k, v, offsets, freqs, cache, layer, scale
            )
        else:
            store_k_cache_ragged(cache, k, offsets, layer)
            # A bfloat16 constant would round the scale itself
            # (0.612 -> 0.6133), so the reference scales before rounding.
            scaled = (v.cast(DType.float32) * scale).cast(v.dtype)
            store_v_cache_ragged(cache, scaled, offsets, layer)
            roped = fused_qk_ragged_rope(
                params,
                q,
                offsets,
                cache,
                freqs,
                layer,
                interleaved=False,
            )
        # A graph consumer of the mutated cache, not a host readback after
        # execution: buffer_load reads the device's mutation chain.
        if observe_cache:
            consumed = ops.cast(ops.buffer_load(cache.kv_blocks), DType.float32)
            if roped is None:
                graph.output(consumed)
            else:
                graph.output(roped, consumed)
        elif roped is not None:
            graph.output(roped)
        else:
            # The query-less fused arm returns nothing, and the benchmark
            # builds it without a cache consumer: hand the key through so the
            # graph has an output and the frame stays one launch.
            graph.output(k)
    return graph


def graph_inputs(
    mode: str, values: Values, cache: KVCacheInputsPerDevice[Buffer, Buffer]
) -> list[Buffer]:
    """The operands one arm takes: queries only where it ropes them."""
    q, k, v, offsets, freqs = values
    tensors = [q, k, v] if mode.endswith("qkv") else [k, v]
    return [*tensors, offsets, freqs, *tree.leaves(cache)]


def bits(buffer: Buffer) -> torch.Tensor:
    """The buffer's contents as raw 16-bit patterns, on the host."""
    return torch.from_dlpack(buffer).cpu().view(torch.int16)


def projection(tokens: int, epoch: int, heads: int, width: int) -> torch.Tensor:
    """Distinct bfloat16 bits per (token, head, dim), with edge values first.

    The leading lanes carry values the rotation and the value scale are
    sensitive to: a pair that differs only in the last mantissa bit, a
    negative pair, and -0. K and V are built with different epochs, so a
    kernel that read one where it meant the other could not pass.
    """
    token = np.arange(tokens, dtype=np.int64)[:, None, None]
    head = np.arange(heads, dtype=np.int64)[None, :, None]
    dim = np.arange(width, dtype=np.int64)[None, None, :]
    values = (
        0x3C00 + (token * 137 + head * 19 + dim * 3 + epoch * 11) % 1024
    ).astype(np.uint16)
    values |= np.where((token + head + dim) % 2, 0x8000, 0).astype(np.uint16)
    sensitive = np.array(
        [0x3F81, 0x3EAB, 0xBF83, 0x8000, 0x3F7F, 0x4081], np.uint16
    )
    values = np.where(
        dim < 8, sensitive[(dim + epoch) % len(sensitive)], values
    )
    return torch.from_numpy(values.view(np.int16)).view(torch.bfloat16)


def rope_freqs(shape: Shape, positions: int = 512) -> torch.Tensor:
    """Interleaved NeoX cos/sin coefficients, in the shape's own dtype."""
    position = torch.arange(positions, dtype=torch.float32)[:, None]
    inv_freq = 10000.0 ** (
        -torch.arange(0, shape.rotary_dim, 2, dtype=torch.float32)
        / shape.rotary_dim
    )
    angle = position * inv_freq
    table = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1).reshape(
        positions, shape.rotary_dim
    )
    if shape.frequency_dtype == DType.float32:
        return table
    return table.to(torch.bfloat16)


@dataclass
class Compiled:
    """One (shape, scale) case's loaded arms."""

    device: Accelerator
    params: MHAKVCacheParams
    shape: Shape
    models: dict[str, Model]


def compile_arms(
    shape: Shape, scale: float, modes: tuple[str, ...] = MODES
) -> Compiled:
    """Loads the named arms of one (shape, scale) case, one model each."""
    if not is_b100_b200():
        pytest.skip("The fused RoPE K/V store is validated on B200")
    device = Accelerator()
    session = InferenceSession(devices=[device])
    params = params_of(shape)
    models = {
        mode: session.load(build_graph(params, shape, scale, mode=mode))
        for mode in modes
    }
    return Compiled(device, params, shape, models)


def case_inputs(
    compiled: Compiled,
    tokens: int,
    prompt_lens: list[int],
    cache_lens: list[int],
    *,
    num_caches: int = 1,
) -> tuple[Values, Caches]:
    """One ragged case's device tensors and one fresh cache per arm."""
    device = compiled.device
    shape = compiled.shape
    values = (
        Buffer.from_dlpack(
            projection(tokens, 0, shape.num_q_heads, shape.head_dim)
        ).to(device),
        Buffer.from_dlpack(
            projection(tokens, 0, shape.num_kv_heads, shape.head_dim)
        ).to(device),
        # A different epoch from K, so a swapped load could not pass.
        Buffer.from_dlpack(
            projection(tokens, 5, shape.num_kv_heads, shape.head_dim)
        ).to(device),
        Buffer.from_numpy(np.array([0, *np.cumsum(prompt_lens)], np.uint32)).to(
            device
        ),
        Buffer.from_dlpack(rope_freqs(shape)).to(device),
    )
    caches = [
        paged_kv_cache_inputs(
            compiled.params,
            prompt_lens,
            cache_lengths=cache_lens,
            total_num_pages=16,
        )
        for _ in range(num_caches)
    ]
    return values, caches


def fill_caches(caches: Caches, device: Accelerator) -> torch.Tensor:
    """Gives every cache the same initial contents, and returns them."""
    block_shape = tuple(caches[0].kv_blocks.shape)
    initial = torch.arange(int(np.prod(block_shape)), dtype=torch.float32)
    initial = (
        (initial.remainder(173) / 16 - 5)
        .to(torch.bfloat16)
        .reshape(block_shape)
    )
    initial_buf = Buffer.from_dlpack(initial).to(device)
    for cache in caches:
        cache.kv_blocks.inplace_copy_from(initial_buf)
    return initial


def assert_parity(
    fused: str,
    prompt_lens: list[int],
    cache_lens: list[int],
    caches: Caches,
    outputs: list[tuple[Buffer, ...]],
    initial: torch.Tensor,
) -> None:
    """Grades a fused arm against the reference run beside it.

    Both caches must agree bit for bit, the arm's queries must too when it
    ropes any, the in-graph consumer of the mutation chain must see the final
    writes, and nothing outside the batch's own rows may move.
    """
    if fused == FUSED_QKV:
        assert torch.equal(bits(outputs[0][0]), bits(outputs[1][0])), (
            "the fused op rotated Q differently"
        )
    # The caches agree bit for bit, including every lane of every head: the
    # fused ops round the value scale in the kernel, where the reference
    # rounded it into a tensor first.
    actual = bits(caches[1].kv_blocks)
    assert torch.equal(bits(caches[0].kv_blocks), actual), (
        "the fused op wrote a different KV cache"
    )
    # And the in-graph consumer of the mutation chain saw those writes. It is
    # each arm's last output: the query rotation precedes it where there is
    # one.
    for output, cache in zip(outputs, caches, strict=True):
        expected = torch.from_dlpack(cache.kv_blocks).cpu().float()
        assert torch.equal(
            torch.from_dlpack(output[-1]).cpu().view(torch.int32),
            expected.view(torch.int32),
        ), "the in-graph consumer did not see the final writes"
    # The rows the batch owns are the only ones written.
    touched = torch.zeros(tuple(caches[1].kv_blocks.shape), dtype=torch.bool)
    lookup = caches[1].lookup_table.to_numpy()
    for batch, length in enumerate(prompt_lens):
        for local in range(length):
            pos = cache_lens[batch] + local
            touched[int(lookup[batch, pos // 128]), :, 1, pos % 128, :, :] = (
                True
            )
    assert torch.equal(actual[~touched], initial.view(torch.int16)[~touched]), (
        "the fused op wrote rows the batch does not own"
    )
