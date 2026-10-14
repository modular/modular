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
"""Tests the MiMo-V2 DFlash drafter's attention against a direct reference.

On the real checkpoint the drafter's sinks take about 3e-5 of the softmax
mass, and the per-query window differs from a uniform one in at most 7 far
keys, so removing either leaves draft agreement with vLLM unchanged. These
tests make both matter: sinks comparable to the attention logits, and large
values on the keys at the window's edge.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.dflash_mimo_v2 import (
    DFlashContextWriter,
    DFlashMiMoV2,
    DFlashMiMoV2Config,
)
from max.pipelines.architectures.dflash_mimo_v2.context_writer import (
    DFlashContextLayer,
)
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    MASK_EMBEDDING,
    drafter_tensor_shapes,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 128
ROTARY = 64
WINDOW = 1024
BLOCK = 8
V_SCALE = 0.612
THETA = 1e4
EPS = 1e-6
# The anchor, far enough in that the window is full and starts past 0.
ANCHOR = 1100


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to the nearest bfloat16 and returns its bits."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _round(x: np.ndarray) -> np.ndarray:
    return _f32(_bf16(x))


def _buffer(x: np.ndarray) -> Buffer:
    return Buffer.from_numpy(_bf16(x)).view(DType.bfloat16)


def _config(kv_params: MHAKVCacheParams) -> DFlashMiMoV2Config:
    return DFlashMiMoV2Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=HIDDEN,
        num_hidden_layers=1,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        partial_rotary_factor=ROTARY / HEAD_DIM,
        sliding_window=WINDOW,
        block_size=BLOCK,
        target_layer_ids=[0, 1],
        mask_token_id=7,
        attention_value_scale=V_SCALE,
        max_seq_len=4096,
        devices=[DeviceRef.GPU()],
        kv_params=kv_params,
    )


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )


def _weights(
    config: DFlashMiMoV2Config, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """BF16-exact random weights: projections that keep activations O(1)."""
    weights = {}
    for name, shape in drafter_tensor_shapes(config).items():
        if len(shape) == 2:
            w = rng.standard_normal(shape) / math.sqrt(shape[1])
        else:
            w = 1.0 + 0.1 * rng.standard_normal(shape)
        weights[name] = _round(w.astype(np.float32))
    # Sinks near the log of the summed key mass, so each takes a sizeable
    # share of its head's softmax.
    weights["layers.0.self_attn.attention_sink_bias"] = _round(
        np.array([6.5, 7.0, 7.5, 8.0], np.float32)
    )
    weights[MASK_EMBEDDING] = _round(
        rng.standard_normal(HIDDEN).astype(np.float32)
    )
    return weights


def _state(weights: dict[str, np.ndarray]) -> dict[str, WeightData]:
    return {
        name: WeightData(_buffer(w), name, DType.bfloat16, Shape(w.shape))
        for name, w in weights.items()
    }


def _rms(x: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * gain


def _rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """NeoX rotation of the leading ``ROTARY`` dims of ``[rows, heads, dim]``."""
    inv_freq = 1.0 / THETA ** (np.arange(0, ROTARY, 2) / ROTARY)
    angles = positions[:, None, None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    half = ROTARY // 2
    x1, x2 = x[..., :half], x[..., half:ROTARY]
    return np.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., ROTARY:]], axis=-1
    )


def _project_kv(
    w: dict[str, np.ndarray], x: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    attn = "layers.0.self_attn."
    k = (x @ w[attn + "k_proj.weight"].T).reshape(-1, KV_HEADS, HEAD_DIM)
    k = _rope(_rms(k, w[attn + "k_norm.weight"]), positions)
    v = V_SCALE * (x @ w[attn + "v_proj.weight"].T).reshape(
        -1, KV_HEADS, HEAD_DIM
    )
    return k, v


def _attention_reference(
    w: dict[str, np.ndarray],
    x: np.ndarray,
    ctx_k: np.ndarray,
    ctx_v: np.ndarray,
    *,
    sinks: bool = True,
    window: int = WINDOW,
    uniform_window: bool = False,
) -> np.ndarray:
    """The block's attention output, with context at ``[0, ANCHOR)``."""
    attn = "layers.0.self_attn."
    positions = np.arange(ANCHOR, ANCHOR + BLOCK)
    q = (x @ w[attn + "q_proj.weight"].T).reshape(BLOCK, HEADS, HEAD_DIM)
    q = _rope(_rms(q, w[attn + "q_norm.weight"]), positions)
    block_k, block_v = _project_kv(w, x, positions)
    k = np.concatenate([ctx_k, block_k])
    v = np.concatenate([ctx_v, block_v])
    group = HEADS // KV_HEADS
    k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) / math.sqrt(HEAD_DIM)
    keys = np.arange(ANCHOR + BLOCK)
    queries = ANCHOR if uniform_window else positions
    visible = (
        keys[None, :] + window > np.broadcast_to(queries, (BLOCK,))[:, None]
    )
    scores = np.where(visible[None], scores, -np.inf)
    peak = scores.max(-1, keepdims=True)
    mass = np.exp(scores - peak)
    total = mass.sum(-1, keepdims=True)
    if sinks:
        sink = w[attn + "attention_sink_bias"][:, None, None]
        peak_all = np.maximum(peak, sink)
        mass = np.exp(scores - peak_all)
        total = mass.sum(-1, keepdims=True) + np.exp(sink - peak_all)
    out = np.einsum("hqk,khd->qhd", mass / total, v).reshape(BLOCK, -1)
    return out @ w[attn + "o_proj.weight"].T


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _read_kv(blocks: Buffer, page_size: int) -> np.ndarray:
    """``[positions, 2, heads, dim]`` K and V of a one-request cache."""
    bits = np.from_dlpack(blocks.to(CPU()).view(DType.uint16))
    pages = _f32(bits[:-1, :, 0])  # [page, kv, slot, head, dim]
    return pages.transpose(0, 2, 1, 3, 4).reshape(-1, 2, KV_HEADS, HEAD_DIM)


@pytest.fixture(scope="module")
def setup() -> tuple[DFlashMiMoV2Config, dict[str, np.ndarray]]:
    kv_params = _kv_params()
    return _config(kv_params), _weights(
        _config(kv_params), np.random.default_rng(0)
    )


def test_context_writer_matches_reference(
    setup: tuple[DFlashMiMoV2Config, dict[str, np.ndarray]],
) -> None:
    config, w = setup
    rng = np.random.default_rng(1)
    starts, rows = [0, 300], [5, 40]
    taps = _round(
        rng.standard_normal((sum(rows), 2, HIDDEN)).astype(np.float32)
    )

    writer = DFlashContextWriter(config)
    writer.load_state_dict(
        {k: v for k, v in _state(w).items() if k in writer.raw_state_dict()},
        strict=True,
    )
    kv_params = config.kv_params
    with Graph(
        "writer",
        input_types=[
            TensorType(DType.bfloat16, ["rows", 2, HIDDEN], DeviceRef.GPU()),
            TensorType(DType.uint32, ["offsets"], DeviceRef.GPU()),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        x, offsets, *kv = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv))[0]
        writer(
            [[x.tensor[:, 0, :], x.tensor[:, 1, :]]],
            [offsets.tensor],
            [kv_collection],
        )
        graph.output(ops.constant(0, DType.int32, DeviceRef.CPU()))

    device = Accelerator()
    model = InferenceSession(devices=[device]).load(
        graph, weights_registry=writer.state_dict()
    )
    cache = paged_kv_cache_inputs(kv_params, rows, cache_lengths=starts)
    model.execute(
        _buffer(taps).to(device),
        Buffer.from_numpy(np.array([0, rows[0], sum(rows)], np.uint32)).to(
            device
        ),
        *tree.leaves(cache),
    )

    # Request 1's pages follow request 0's single page.
    cached = _read_kv(cache.kv_blocks, 128)
    got = np.concatenate(
        [cached[: rows[0]], cached[128 + starts[1] : 128 + starts[1] + rows[1]]]
    )
    positions = np.concatenate(
        [np.arange(s, s + n) for s, n in zip(starts, rows, strict=True)]
    )
    ctx = _rms(
        taps.reshape(len(taps), -1) @ w["fc.weight"].T, w["hidden_norm.weight"]
    )
    k, v = _project_kv(w, ctx, positions)
    assert _rel(got[:, 0], k) < 1.5e-2
    assert _rel(got[:, 1], v) < 1.5e-2
    # The tolerance separates the value scale from its absence.
    assert _rel(got[:, 1], v / V_SCALE) > 0.3


def test_block_attention_sinks_and_window(
    setup: tuple[DFlashMiMoV2Config, dict[str, np.ndarray]],
) -> None:
    config, w = setup
    rng = np.random.default_rng(2)
    x = _round(rng.standard_normal((BLOCK, HIDDEN)).astype(np.float32))
    ctx_k = _round(
        _rope(
            rng.standard_normal((ANCHOR, KV_HEADS, HEAD_DIM)), np.arange(ANCHOR)
        ).astype(np.float32)
    )
    ctx_v = 0.1 * rng.standard_normal((ANCHOR, KV_HEADS, HEAD_DIM))
    # Block query i sees the context from ANCHOR + i - 1023 on, so these nine
    # keys straddle the window edge of every query.
    edge = slice(ANCHOR - WINDOW, ANCHOR - WINDOW + BLOCK + 1)
    ctx_v[edge] = rng.choice([-64.0, 64.0], size=ctx_v[edge].shape)
    ctx_v = _round(ctx_v.astype(np.float32))

    drafter = DFlashMiMoV2(config)
    drafter.load_state_dict(_state(w), strict=True)
    kv_params = config.kv_params
    with Graph(
        "attention",
        input_types=[
            TensorType(DType.bfloat16, [BLOCK, HIDDEN], DeviceRef.GPU()),
            TensorType(DType.uint32, [2], DeviceRef.GPU()),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        block, offsets, *kv = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv))[0]
        layer = drafter.layers[0]
        assert isinstance(layer, DFlashContextLayer)
        attn = layer.self_attn_shards[0]
        out = attn(
            ops.constant(0, DType.uint32, DeviceRef.CPU()),
            block.tensor,
            kv_collection,
            drafter.rope.freqs_cis.to(DeviceRef.GPU()),
            offsets.tensor,
        )
        graph.output(ops.cast(out, DType.float32))

    device = Accelerator()
    model = InferenceSession(devices=[device]).load(
        graph, weights_registry=drafter.state_dict()
    )
    cache = paged_kv_cache_inputs(kv_params, [BLOCK], cache_lengths=[ANCHOR])
    blocks = np.zeros(cache.kv_blocks.shape, np.uint16)
    for pos in range(ANCHOR):
        blocks[pos // 128, 0, 0, pos % 128] = _bf16(ctx_k[pos])
        blocks[pos // 128, 1, 0, pos % 128] = _bf16(ctx_v[pos])
    cache = dataclasses.replace(
        cache,
        kv_blocks=Buffer.from_numpy(blocks).view(DType.bfloat16).to(device),
    )
    (result,) = model.execute(
        _buffer(x).to(device),
        Buffer.from_numpy(np.array([0, BLOCK], np.uint32)).to(device),
        *tree.leaves(cache),
    )
    assert isinstance(result, Buffer)
    got = np.from_dlpack(result.to(CPU()))

    tolerance = 2e-2
    assert _rel(got, _attention_reference(w, x, ctx_k, ctx_v)) < tolerance
    # Each convention the fixture cannot see moves this output well past the
    # tolerance.
    for wrong in (
        dict(sinks=False),
        dict(uniform_window=True),
        dict(window=WINDOW - 1),
        dict(window=WINDOW + 1),
    ):
        reference = _attention_reference(w, x, ctx_k, ctx_v, **wrong)
        assert _rel(got, reference) > 10 * tolerance, wrong


def test_block_embeddings_use_mask_embedding(
    setup: tuple[DFlashMiMoV2Config, dict[str, np.ndarray]],
) -> None:
    config, w = setup
    anchors = _round(
        np.random.default_rng(3).standard_normal((2, HIDDEN)).astype(np.float32)
    )
    drafter = DFlashMiMoV2(config)
    drafter.load_state_dict(_state(w), strict=True)
    with Graph(
        "embeddings",
        input_types=[TensorType(DType.bfloat16, [2, HIDDEN], DeviceRef.GPU())],
    ) as graph:
        (rows,) = drafter.block_embeddings([graph.inputs[0].tensor])
        graph.output(ops.cast(rows, DType.float32))

    device = Accelerator()
    model = InferenceSession(devices=[device]).load(
        graph, weights_registry=drafter.state_dict()
    )
    (result,) = model.execute(_buffer(anchors).to(device))
    assert isinstance(result, Buffer)
    got = np.from_dlpack(result.to(CPU())).reshape(2, BLOCK, HIDDEN)
    np.testing.assert_array_equal(got[:, 0], anchors)
    np.testing.assert_array_equal(
        got[:, 1:], np.broadcast_to(w[MASK_EMBEDDING], (2, BLOCK - 1, HIDDEN))
    )
