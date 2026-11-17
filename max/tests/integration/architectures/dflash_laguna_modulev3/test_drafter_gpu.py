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
"""The Laguna DFlash drafter on a GPU against a numpy reference.

Two layers at the checkpoint's head geometry (72 query heads, 8 KV heads of
128, a 512-key window) with non-trivial norm gammas, so each convention the
reference transcribes from vLLM's drafter moves the output: the per-tap
norms, each layer's ``input_layernorm`` on the context, the causal window
and the per-head gate.

It compiles and runs on a GPU because the two entries meet only in the paged
cache: the positions the context is roped at and where the window's edge
falls come from the cache lengths at run time, which no staged-IR check sees.
A drafter that gets them wrong still serves correct tokens, only fewer of
them accepted, so no accuracy eval or smoke run catches it after merge.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import ml_dtypes
import numpy as np
import pytest
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.kv_cache import KVCacheInputsPerDevice, MHAKVCacheParams
from max.pipelines.architectures.dflash_laguna_modulev3 import (
    DFlashLaguna,
    DFlashLagunaConfig,
)
from max.pipelines.lib.module_registration import device_mesh
from max.tensor import Tensor, default_device, default_dtype
from test_common.simple_kv_cache import paged_kv_cache_inputs

HIDDEN = 256
HEADS = 72
KV_HEADS = 8
HEAD_DIM = 128
MLP_DIM = 512
LAYERS = 2
TAPS = 6
WINDOW = 512
THETA = 5e5
EPS = 1e-6
# Around and past the window, and one so short the block's own keys
# dominate, which makes causality show.
CONTEXT = [17, 300, 511, 513, 2000]

KV_PARAMS = MHAKVCacheParams(
    dtype=DType.bfloat16,
    n_kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    num_layers=LAYERS,
    page_size=128,
    devices=[DeviceRef.GPU()],
)


def _round(x: np.ndarray) -> np.ndarray:
    return (
        np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    )


def _buffer(x: np.ndarray) -> Buffer:
    bits = np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
    return Buffer.from_numpy(bits).view(DType.bfloat16)


def _weights(config: DFlashLagunaConfig) -> dict[str, np.ndarray]:
    """BF16-exact weights that keep activations O(1)."""
    rng = np.random.default_rng(0)
    with (
        F.lazy(),
        default_dtype(DType.bfloat16),
        default_device(device_mesh([CPU()], 1)),
    ):
        names = {
            name: tuple(int(d) for d in p.shape)
            for name, p in DFlashLaguna(config).parameters
        }
    weights = {}
    for name, shape in names.items():
        if len(shape) == 2:
            w = rng.standard_normal(shape) / math.sqrt(shape[1])
        else:
            w = 1.0 + 0.5 * rng.standard_normal(shape)
        weights[name] = _round(w)
    return weights


def _rms(x: np.ndarray, gain: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * gain


def _rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """NeoX rotation of every lane of ``[rows, heads, head_dim]``."""
    inv_freq = 1.0 / THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
    angles = positions[:, None, None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _kv(
    w: dict[str, np.ndarray], i: int, x: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    attn = f"layers.{i}.self_attn."
    q_dim = HEADS * HEAD_DIM
    kv = (x @ w[attn + "qkv_proj.weight"][q_dim:].T).reshape(
        -1, 2 * KV_HEADS, HEAD_DIM
    )
    k = _rope(_rms(kv[:, :KV_HEADS], w[attn + "k_norm.weight"]), positions)
    return k, kv[:, KV_HEADS:]


def _reference(
    w: dict[str, np.ndarray],
    taps: np.ndarray,
    block: np.ndarray,
    width: int,
    *,
    context_layernorm: bool = True,
    causal: bool = True,
    gate: bool = True,
) -> np.ndarray:
    """vLLM's drafter: context K/V for every request, then its block."""
    split = np.split(taps, np.cumsum(CONTEXT)[:-1])
    blocks = np.split(block, len(CONTEXT))
    out = []
    for request_taps, h, start in zip(split, blocks, CONTEXT, strict=True):
        normed = [
            _rms(t, w[f"aux_hidden_norms.{i}.weight"])
            for i, t in enumerate(np.split(request_taps, TAPS, axis=-1))
        ]
        ctx = _rms(
            np.concatenate(normed, -1) @ w["fc.weight"].T,
            w["hidden_norm.weight"],
        )
        ctx_positions = np.arange(start)
        positions = np.arange(start, start + width)
        for i in range(LAYERS):
            p = f"layers.{i}."
            ctx_in = _rms(ctx, w[p + "input_layernorm.weight"])
            ctx_k, ctx_v = _kv(
                w, i, ctx_in if context_layernorm else ctx, ctx_positions
            )
            x = _rms(h, w[p + "input_layernorm.weight"])
            q = x @ w[p + "self_attn.qkv_proj.weight"][: HEADS * HEAD_DIM].T
            q = _rope(
                _rms(
                    q.reshape(width, HEADS, HEAD_DIM),
                    w[p + "self_attn.q_norm.weight"],
                ),
                positions,
            )
            block_k, block_v = _kv(w, i, x, positions)
            # [kv_head, group, row, dim]: query head h reads KV head h // group.
            q = q.reshape(width, KV_HEADS, -1, HEAD_DIM).transpose(1, 2, 0, 3)
            k = np.concatenate([ctx_k, block_k]).transpose(1, 2, 0)[:, None]
            v = np.concatenate([ctx_v, block_v]).transpose(1, 0, 2)[:, None]
            scores = q @ k / math.sqrt(HEAD_DIM)
            keys = np.arange(start + width)
            visible = keys[None, :] > positions[:, None] - WINDOW
            if causal:
                visible &= keys[None, :] <= positions[:, None]
            scores = np.where(visible, scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            attn = (probs @ v).transpose(2, 0, 1, 3).reshape(width, HEADS, -1)
            if gate:
                g = x @ w[p + "self_attn.g_proj.weight"].T
                attn = attn * np.logaddexp(0.0, g)[..., None]
            h = h + attn.reshape(width, -1) @ w[p + "self_attn.o_proj.weight"].T
            m = _rms(h, w[p + "post_attention_layernorm.weight"])
            gate_up = m @ w[p + "mlp.gate_proj.weight"].T
            act = (
                gate_up
                / (1.0 + np.exp(-gate_up))
                * (m @ w[p + "mlp.up_proj.weight"].T)
            )
            h = h + act @ w[p + "mlp.down_proj.weight"].T
        out.append(_rms(h, w["norm.weight"]))
    return np.concatenate(out)


class _Harness(Module[..., Tensor]):
    """Drives one drafter entry from flattened KV graph inputs."""

    def __init__(self, drafter: DFlashLaguna, entry: str) -> None:
        self.drafter = drafter
        self.entry = entry
        self.mesh = device_mesh([Accelerator()], 1)

    def forward(self, x: Tensor, offsets: Tensor, *kv: Tensor) -> Tensor:
        x, offsets = x.to(self.mesh), offsets.to(self.mesh)
        caches = KV_PARAMS.unflatten_kv_inputs(iter(t._graph_value for t in kv))
        if self.entry == "context":
            self.drafter.write_context(x, offsets, caches)
            return offsets
        return self.drafter.forward_block(x, caches, offsets).cast(
            DType.float32
        )


def _config(width: int = 16) -> DFlashLagunaConfig:
    return DFlashLagunaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=MLP_DIM,
        num_hidden_layers=LAYERS,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        sliding_window=WINDOW,
        vocab_size=1024,
        block_size=width,
        trained_block_size=16,
        mask_token_id=12,
        target_layer_ids=[1, 10, 19, 29, 38, 47],
        num_target_layers=48,
        max_seq_len=4096,
        devices=[DeviceRef.GPU()],
        kv_params=KV_PARAMS,
    )


@pytest.fixture(scope="module")
def drafter() -> dict[str, Any]:
    config = _config()
    w = _weights(config)
    width = TAPS * HIDDEN
    runs = {}
    for entry, columns in (("context", width), ("block", HIDDEN)):
        mesh = device_mesh([Accelerator()], 1)
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            harness = _Harness(DFlashLaguna(config), entry)
        runs[entry] = harness.compile(
            TensorType(DType.bfloat16, ["rows", columns], DeviceRef.GPU()),
            TensorType(DType.uint32, ["offsets"], DeviceRef.GPU()),
            *KV_PARAMS.flattened_kv_inputs(),
            weights={f"drafter.{k}": _buffer(v) for k, v in w.items()},
        )
    rng = np.random.default_rng(1)
    taps = _round(rng.standard_normal((sum(CONTEXT), width)))
    # One page table sized for the widest block serves every call, so the
    # blocks read the context the first call wrote.
    pages = paged_kv_cache_inputs(
        KV_PARAMS, [16] * len(CONTEXT), cache_lengths=CONTEXT
    )
    _run(
        runs["context"], taps, CONTEXT, pages, cache_lengths=[0] * len(CONTEXT)
    )
    return dict(runs=runs, weights=w, taps=taps, pages=pages)


def _run(
    run: Any,
    x: np.ndarray,
    rows: Sequence[int],
    pages: KVCacheInputsPerDevice[Buffer, Buffer],
    cache_lengths: Sequence[int],
) -> np.ndarray:
    device = Accelerator()
    cache = dataclasses.replace(
        paged_kv_cache_inputs(KV_PARAMS, rows, cache_lengths=cache_lengths),
        kv_blocks=pages.kv_blocks,
        lookup_table=pages.lookup_table,
    )
    offsets = np.concatenate([[0], np.cumsum(rows)]).astype(np.uint32)
    (out,) = run.execute_raw(
        _buffer(x).to(device),
        Buffer.from_numpy(offsets).to(device),
        *tree.leaves(cache),
    )
    assert isinstance(out, Buffer)
    return np.from_dlpack(out.to(CPU()))


def _block(
    drafter: dict[str, Any], width: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    embeds = _round(rng.standard_normal((len(CONTEXT) * 16, HIDDEN)))
    embeds = embeds.reshape(len(CONTEXT), 16, HIDDEN)[:, :width].reshape(
        -1, HIDDEN
    )
    got = _run(
        drafter["runs"]["block"],
        embeds,
        [width] * len(CONTEXT),
        drafter["pages"],
        cache_lengths=CONTEXT,
    )
    return embeds, got


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_the_block_matches_the_reference(drafter: dict[str, Any]) -> None:
    embeds, got = _block(drafter, 16)
    w, taps = drafter["weights"], drafter["taps"]
    tolerance = 2e-2
    assert _rel(got, _reference(w, taps, embeds, 16)) < tolerance
    # Each convention moves the output well past the tolerance.
    for wrong in (
        dict(context_layernorm=False),
        dict(causal=False),
        dict(gate=False),
    ):
        reference = _reference(w, taps, embeds, 16, **wrong)
        assert _rel(got, reference) > 5 * tolerance, wrong


def test_a_narrower_block_is_the_trained_blocks_prefix(
    drafter: dict[str, Any],
) -> None:
    """K = 7 serves a block of 8, which a causal block computes as the first
    8 rows of its trained 16."""
    _, wide = _block(drafter, 16)
    _, narrow = _block(drafter, 8)
    prefix = wide.reshape(len(CONTEXT), 16, HIDDEN)[:, :8].reshape(-1, HIDDEN)
    assert _rel(narrow, prefix) < 1e-2
    head = np.random.default_rng(3).standard_normal((1024, HIDDEN))
    np.testing.assert_array_equal(
        np.argmax(narrow @ head.T, -1), np.argmax(prefix @ head.T, -1)
    )
