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
"""Checks chunked QKV preparation, cache mutation and metadata replay.

MiMo's unfused preparation helper supplies the production reference sequence;
its chunk/head dimensions are explicit, so the same reference covers the
operator's layouts beyond the published checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import fused_chunked_qkv_rope_store_ragged
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.mimo_v2.layers.attention import (
    QkvChunkLayout,
    prepare_qkv_unfused,
)
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs


@dataclass(frozen=True)
class _Layout:
    name: str
    chunks: int
    num_q_heads: int
    num_kv_heads: int
    qk_head_dim: int = 192
    v_head_dim: int = 128
    cache_head_dim: int = 256
    rotary_dim: int = 64
    chunk_padding: int = 0

    @property
    def q_rows(self) -> int:
        return self.num_q_heads // self.chunks * self.qk_head_dim

    @property
    def k_rows(self) -> int:
        return self.num_kv_heads // self.chunks * self.qk_head_dim

    @property
    def v_rows(self) -> int:
        return self.num_kv_heads // self.chunks * self.v_head_dim


_LAYOUTS = [
    _Layout("tiny-two-chunks", 2, 8, 4),
    _Layout("published-tp2", 2, 32, 4),
    _Layout("published-tp4-one-chunk", 1, 16, 2),
    _Layout("published-tp1-four-chunks", 4, 64, 8),
    _Layout("no-padding-full-rotary", 2, 8, 4, 128, 128, 128, 128),
    _Layout("full-published-tp2", 2, 32, 2, chunk_padding=64),
    _Layout("full-published-tp4", 1, 16, 1, chunk_padding=64),
    _Layout("full-published-tp1", 4, 64, 4, chunk_padding=64),
]


def _projection(tokens: int, epoch: int, layout: _Layout) -> torch.Tensor:
    token = np.arange(tokens, dtype=np.int64)[:, None, None]
    chunk = np.arange(layout.chunks, dtype=np.int64)[None, :, None]
    q_rows, k_rows = layout.q_rows, layout.k_rows
    chunk_width = q_rows + k_rows + layout.v_rows + layout.chunk_padding
    row = np.arange(chunk_width, dtype=np.int64)[None, None, :]
    q_heads = layout.num_q_heads // layout.chunks
    kv_heads = layout.num_kv_heads // layout.chunks
    head = np.where(
        row < q_rows,
        row // layout.qk_head_dim,
        np.where(
            row < q_rows + k_rows,
            (row - q_rows) // layout.qk_head_dim + q_heads,
            (row - q_rows - k_rows) // layout.v_head_dim + q_heads + kv_heads,
        ),
    )
    dim = np.where(
        row < q_rows + k_rows,
        row % layout.qk_head_dim,
        (row - q_rows - k_rows) % layout.v_head_dim,
    )
    bits = (
        0x3C00
        + (token * 137 + chunk * 67 + head * 19 + dim * 3 + epoch * 11) % 1024
    ).astype(np.uint16)
    bits |= np.where((token + chunk + head) % 2, 0x8000, 0).astype(np.uint16)
    sensitive = np.array(
        [0x3F81, 0x3EAB, 0xBF83, 0x8000, 0x3F7F, 0x4081], np.uint16
    )
    bits = np.where(dim < 8, sensitive[(dim + epoch) % len(sensitive)], bits)
    if layout.chunk_padding:
        bits[..., -layout.chunk_padding :] = 0x7FC0
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)


def _graph(
    params: MHAKVCacheParams,
    scale: float,
    *,
    fused: bool,
    layout: _Layout,
    observe_cache: bool = True,
    frequency_positions: int = 512,
) -> Graph:
    q_rows, k_rows, v_rows = layout.q_rows, layout.k_rows, layout.v_rows
    with Graph(
        f"chunked_qkv_{layout.name}_{fused}_{scale}",
        input_types=[
            TensorType(
                DType.bfloat16,
                [
                    "tokens",
                    layout.chunks,
                    q_rows + k_rows + v_rows + layout.chunk_padding,
                ],
                DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, ["offsets"], DeviceRef.GPU()),
            TensorType(
                DType.bfloat16,
                [frequency_positions, layout.rotary_dim],
                DeviceRef.GPU(),
            ),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        qkv, offsets, freqs = (value.tensor for value in graph.inputs[:3])
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        layer = ops.constant(1, DType.uint32, device=DeviceRef.CPU())
        if fused:
            q = fused_chunked_qkv_rope_store_ragged(
                qkv,
                offsets,
                freqs,
                cache,
                layer,
                scale,
                num_q_heads=layout.num_q_heads,
                qk_head_dim=layout.qk_head_dim,
                v_head_dim=layout.v_head_dim,
                chunk_padding=layout.chunk_padding,
            )
        else:
            q = prepare_qkv_unfused(
                qkv,
                offsets,
                freqs,
                cache,
                layer,
                scale,
                kv_params=params,
                layout=QkvChunkLayout(layout.chunks, q_rows, k_rows, v_rows),
                head_dim=layout.qk_head_dim,
                v_head_dim=layout.v_head_dim,
            )
        # This is a graph consumer of the mutated cache, not a host readback
        # after execution. buffer_load reads the device's mutation chain.
        if observe_cache:
            consumed = ops.cast(ops.buffer_load(cache.kv_blocks), DType.float32)
            graph.output(q, consumed)
        else:
            graph.output(q)
    return graph


def _bits(buffer: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buffer).cpu().view(torch.int16)


@dataclass
class _CompiledFixture:
    device: Accelerator
    session: InferenceSession
    params: MHAKVCacheParams
    layout: _Layout
    models: tuple[Model, Model]


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((layout, scale), id=f"{layout.name}-{scale_name}")
        for layout in _LAYOUTS
        for scale, scale_name in (
            [(1.00390625, "bf16-tie"), (0.1, "decimal"), (-2.0, "negative")]
            if layout in _LAYOUTS[:2]
            else [(0.1, "decimal")]
        )
    ],
)
def compiled_models(request: pytest.FixtureRequest) -> _CompiledFixture:
    if not is_b100_b200():
        pytest.skip("Chunked QKV preparation is validated on B200")
    layout, scale = request.param
    device = Accelerator()
    session = InferenceSession(devices=[device])
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=layout.num_kv_heads,
        head_dim=layout.cache_head_dim,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    return _CompiledFixture(
        device,
        session,
        params,
        layout,
        (
            session.load(_graph(params, scale, fused=False, layout=layout)),
            session.load(_graph(params, scale, fused=True, layout=layout)),
        ),
    )


@pytest.mark.parametrize(
    ("prompt_lens", "cache_lens"),
    [([1], [129]), ([1, 2, 5, 3], [0, 127, 128, 255]), ([0], [129])],
    ids=["single-decode", "ragged-page-boundaries", "empty"],
)
def test_registered_chunked_qkv_rope_store(
    prompt_lens: list[int],
    cache_lens: list[int],
    compiled_models: _CompiledFixture,
) -> None:
    device = compiled_models.device
    params = compiled_models.params
    layout = compiled_models.layout
    models = compiled_models.models
    tokens = sum(prompt_lens)
    # Allocate room for all three epochs before capture; only values change.
    final_cache_lens = [length + 2 for length in cache_lens]
    caches = [
        paged_kv_cache_inputs(
            params,
            prompt_lens,
            cache_lengths=final_cache_lens,
            total_num_pages=16,
        )
        for _ in models
    ]
    shape = tuple(caches[0].kv_blocks.shape)
    initial = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(
        shape
    )
    initial = (initial.remainder(173) / 16 - 5).to(torch.bfloat16)
    initial_buf = Buffer.from_dlpack(initial).to(device)
    offsets = np.array([0, *np.cumsum(prompt_lens)], np.uint32)
    position = torch.arange(512, dtype=torch.float32)[:, None]
    inv_freq = 10000.0 ** (
        -torch.arange(0, layout.rotary_dim, 2, dtype=torch.float32)
        / layout.rotary_dim
    )
    angle = position * inv_freq
    freqs = (
        torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        .reshape(512, layout.rotary_dim)
        .to(torch.bfloat16)
    )
    inputs = [
        [
            Buffer.from_dlpack(_projection(tokens, 0, layout)).to(device),
            Buffer.from_numpy(offsets).to(device),
            Buffer.from_dlpack(freqs).to(device),
            *tree.leaves(cache),
        ]
        for cache in caches
    ]
    captured: list[tuple[Buffer, ...]] = []
    try:
        for epoch in range(3):
            max_cache = max(
                length + epoch + prompt
                for length, prompt in zip(cache_lens, prompt_lens, strict=True)
            )
            for cache, values in zip(caches, inputs, strict=True):
                cache.kv_blocks.inplace_copy_from(initial_buf)
                cache.cache_lengths.inplace_copy_from(
                    Buffer.from_numpy(
                        np.array(
                            [length + epoch for length in cache_lens], np.uint32
                        )
                    ).to(device)
                )
                # Change CPU bounds after capture as well as GPU lengths/data.
                # These are valid upper bounds, independent of the logical rows.
                cache.max_prompt_length.inplace_copy_from(
                    Buffer.from_numpy(
                        np.array([max(prompt_lens) + (epoch == 2)], np.uint32)
                    )
                )
                cache.max_cache_length.inplace_copy_from(
                    Buffer.from_numpy(
                        np.array([max_cache + (epoch == 2)], np.uint32)
                    )
                )
                values[0].inplace_copy_from(
                    Buffer.from_dlpack(_projection(tokens, epoch, layout)).to(
                        device
                    )
                )
            if epoch == 0:
                outputs = [
                    tuple(model.execute(*values))
                    for model, values in zip(models, inputs, strict=True)
                ]
            elif epoch == 1:
                captured = [
                    tuple(model.capture(631, *values))
                    for model, values in zip(models, inputs, strict=True)
                ]
                # Capture may execute the append; reset and grade a real replay.
                for cache in caches:
                    cache.kv_blocks.inplace_copy_from(initial_buf)
                for model, values in zip(models, inputs, strict=True):
                    model.replay(631, *values)
                outputs = captured
            else:
                for model, values in zip(models, inputs, strict=True):
                    model.debug_verify_replay(631, *values)
                    model.replay(631, *values)
                outputs = captured
            assert torch.equal(_bits(outputs[0][0]), _bits(outputs[1][0]))
            observed = [
                torch.from_dlpack(values[1]).cpu() for values in outputs
            ]
            assert torch.equal(
                observed[0].view(torch.int32), observed[1].view(torch.int32)
            )
            actual = [_bits(cache.kv_blocks) for cache in caches]
            assert torch.equal(actual[0], actual[1])
            q_bits = _bits(outputs[1][0])
            assert torch.count_nonzero(q_bits[..., layout.qk_head_dim :]) == 0
            lut = caches[1].lookup_table.to_numpy()
            for batch, length in enumerate(prompt_lens):
                for local in range(length):
                    pos = cache_lens[batch] + epoch + local
                    page = int(lut[batch, pos // 128])
                    for plane, real_depth in (
                        (0, layout.qk_head_dim),
                        (1, layout.v_head_dim),
                    ):
                        assert (
                            torch.count_nonzero(
                                actual[1][
                                    page, plane, 1, pos % 128, :, real_depth:
                                ]
                            )
                            == 0
                        )
            # The in-graph consumer must see the final writes, including the
            # FP32 value scale supplied as a CPU operand.
            for consumed, cache in zip(observed, caches, strict=True):
                expected = torch.from_dlpack(cache.kv_blocks).cpu().float()
                assert torch.equal(
                    consumed.view(torch.int32), expected.view(torch.int32)
                )
            touched = torch.zeros(shape, dtype=torch.bool)
            lut = caches[0].lookup_table.to_numpy()
            for batch, length in enumerate(prompt_lens):
                for local in range(length):
                    pos = cache_lens[batch] + epoch + local
                    page = int(lut[batch, pos // 128])
                    touched[page, :, 1, pos % 128, :, :] = True
            assert torch.equal(
                actual[1][~touched], initial.view(torch.int16)[~touched]
            )
            if tokens == 0:
                assert tuple(outputs[1][0].shape) == (
                    0,
                    layout.num_q_heads,
                    layout.cache_head_dim,
                )
                assert torch.equal(actual[1], initial.view(torch.int16))
    finally:
        for model in models:
            model.release_captured_graph(631)


def _unchecked_graph(
    device: DeviceRef,
    *,
    chunk_width: int = 1408,
    output_width: int = 256,
    rotary_width: int = 64,
    chunk_padding: int = 0,
) -> Graph:
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=256,
        num_layers=2,
        page_size=128,
        devices=[device],
    )
    with Graph(
        "unchecked_chunked_qkv",
        input_types=[
            TensorType(DType.bfloat16, [1, 2, chunk_width], device),
            TensorType(DType.uint32, [2], device),
            TensorType(DType.bfloat16, [512, rotary_width], device),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        qkv, offsets, freqs = (value.tensor for value in graph.inputs[:3])
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        q = ops.inplace_custom(
            "mo.chunked_qkv_rope_store.ragged.paged",
            device=device,
            values=[
                qkv,
                offsets,
                freqs,
                *cache.flatten_without_attention_dispatch_metadata(),
                ops.constant(1, DType.uint32, device=DeviceRef.CPU()),
                ops.constant(0.1, DType.float32, device=DeviceRef.CPU()),
            ],
            out_types=[
                TensorType(DType.bfloat16, [1, 8, output_width], device)
            ],
            parameters={
                "qk_head_dim": 192,
                "v_head_dim": 128,
                "chunk_padding": chunk_padding,
            },
        )[0].tensor
        graph.output(q)
    return graph


@pytest.mark.compile_on_gpu(
    "Invalid graphs produce no MEF; this test requires the actual compiler diagnostic"
)
@pytest.mark.parametrize(
    "case", ["chunk_width", "output_width", "rotary_width", "chunk_padding"]
)
def test_registered_chunked_qkv_rejects_malformed_shapes(case: str) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    # Bypass the Python wrapper: the builtin must protect its own static layout.
    arguments = {
        "chunk_width": 1408,
        "output_width": 256,
        "rotary_width": 64,
        "chunk_padding": 0,
    }
    arguments[case] -= 2
    diagnostic = {
        "chunk_width": "Invalid QKV chunk width",
        "output_width": "Query output width must match cache head size",
        "rotary_width": "cache head and rotary widths must be aligned to 8 elements",
        "chunk_padding": "QKV chunk padding must be nonnegative and aligned to 8 elements",
    }[case]
    with pytest.raises(
        RuntimeError, match="Failed to compile the model"
    ) as error:
        session.load(_unchecked_graph(DeviceRef.GPU(), **arguments))
    cause = error.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert f"constraint failed: {diagnostic}" in str(cause)


@pytest.mark.compile_on_gpu(
    "CPU-target rejection must run the compiler, not replay a nonexistent MEF"
)
def test_registered_chunked_qkv_rejects_cpu_compilation() -> None:
    session = InferenceSession(devices=[CPU()])
    with pytest.raises(
        RuntimeError, match="Failed to compile the model"
    ) as error:
        session.load(_unchecked_graph(DeviceRef.CPU()))
    cause = error.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert "constraint failed: chunked QKV preparation requires a GPU" in str(
        cause
    )


@pytest.mark.parametrize(
    ("chunk_width", "rotary_width", "device", "message"),
    [
        (1400, 64, DeviceRef.GPU(), "chunk width must be 1408"),
        (1408, 66, DeviceRef.GPU(), "aligned to 8"),
        (1408, 64, DeviceRef.CPU(), "requires a GPU"),
    ],
)
def test_chunked_qkv_rejects_unsupported_layout(
    chunk_width: int,
    rotary_width: int,
    device: DeviceRef,
    message: str,
) -> None:
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=256,
        num_layers=2,
        page_size=128,
        devices=[device],
    )
    with Graph(
        "invalid_chunked_qkv",
        input_types=[
            TensorType(DType.bfloat16, [1, 2, chunk_width], device),
            TensorType(DType.uint32, [2], device),
            TensorType(DType.bfloat16, [512, rotary_width], device),
            *params.flattened_kv_inputs(),
        ],
    ) as graph:
        qkv, offsets, freqs = (value.tensor for value in graph.inputs[:3])
        cache = params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        layer = ops.constant(1, DType.uint32, device=DeviceRef.CPU())
        with pytest.raises(ValueError, match=message):
            fused_chunked_qkv_rope_store_ragged(
                qkv,
                offsets,
                freqs,
                cache,
                layer,
                0.1,
                num_q_heads=8,
                qk_head_dim=192,
                v_head_dim=128,
            )
        graph.output(qkv)
