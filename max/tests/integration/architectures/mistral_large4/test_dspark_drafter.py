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
"""Host-only checks of the DSpark drafter: its window mask and its head chain.

Both are places where a plausible-looking implementation is wrong and a full
model run would not say so:

* the block attends the last ``min(2048, anchor + 1)`` context latents **plus
  all of its own rows**, and nothing in the block is causal. A causal mask, or
  a per-row shifted sliding one, still produces finite logits -- it just
  measures a different model, and the draft-token agreement it costs is
  invisible on the fixture (``dspark-reference.md`` §11.1).
* the recurrent head is a *gated* cell whose joint projection chunks into
  ``gate | candidate | emitted`` in that order, with the concat order
  ``[state | markov_embedding(prev) | hidden]``. Each of those is a different
  trained parameter.

These run on the CPU with a geometry small enough to compile in a second, and
compare against a plain numpy transcription of the reference:
``.agentwork/bringup/mistral_4/dspark/dspark-reference.md`` §5 and §9.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.pipelines.architectures.mistral_large4_modulev3 import (
    checkpoint as modulev3_checkpoint,
)
from max.pipelines.architectures.mistral_large4_modulev3.config import (
    _fitted_taps,
    _served_taps,
)
from max.pipelines.architectures.mistral_large4_modulev3.dspark import (
    BLOCK_POSITION_OFFSET_TRTLLM,
    BLOCK_POSITION_OFFSET_VLLM,
    DSparkDrafter,
    DSparkDrafterConfig,
    build_dspark_kv_params,
    convert_dspark_state_dict,
    dspark_tensor_names,
    window_start,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.dspark import (
    DSparkAttention,
    device_scalar,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    MistralLarge4NormalizedConfig,
    MistralLarge4Yarn,
)
from max.pipelines.lib import KVCacheConfig, MAXModelConfig
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from numpy.typing import NDArray

TESTDATA = Path(__file__).parent / "testdata"

HIDDEN = 8
HEADS = 2
NOPE = 4
ROPE = 4
V_HEAD = 4
LATENT = 6
MARKOV_RANK = 4
VOCAB = 12
DEPTH = 3
WINDOW = 4

_Attention = CompiledCallable[..., Tensor]
"""One compiled drafter attention layer, as the mask checks call it."""


def _yarn() -> MistralLarge4Yarn:
    """The drafter's rope geometry, in a shape small enough to compile fast."""
    return MistralLarge4Yarn(
        rope_theta=10000.0,
        factor=2048.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        attention_factor=1.0,
        lerope=False,
        lerope_alpha_dim=ROPE // 2,
    )


def _config(
    *, block_position_offset: int = BLOCK_POSITION_OFFSET_TRTLLM
) -> DSparkDrafterConfig:
    """A miniature drafter: one layer, three rows, a four-key window."""
    return DSparkDrafterConfig(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        kv_lora_rank=LATENT,
        intermediate_size=2 * HIDDEN,
        num_layers=1,
        depth=DEPTH,
        window=WINDOW,
        target_layers=(1,),
        markov_rank=MARKOV_RANK,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        softcap=30.0,
        yarn=_yarn(),
        block_position_offset=block_position_offset,
    )


def _params() -> dict[str, object]:
    """The full checkpoint's ``params.json``, as the M0 tests read it."""
    return json.loads((TESTDATA / "ml4_nvfp4_params.json").read_text())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand(shape: tuple[int, ...], seed: int, scale: float = 0.2) -> np.ndarray:
    return (_rng(seed).standard_normal(shape) * scale).astype(np.float32)


@contextmanager
def _cpu_float32() -> Iterator[None]:
    """Builds a module's parameters on the CPU in float32.

    A box's default device is an accelerator and its default parameter dtype
    bfloat16; this file's numpy references are exact in float32 and its
    session is a CPU one.
    """
    with default_device(CPU()), default_dtype(DType.float32):
        yield


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    """One CPU session for every compiled graph in this file."""
    return InferenceSession(devices=[CPU()])


@pytest.fixture(scope="module")
def attention(
    session: InferenceSession,
) -> tuple[_Attention, DSparkDrafterConfig]:
    """One compiled drafter attention layer, shared by its three checks.

    A CPU compile costs about half a minute, so every property of the block's
    mask is measured on one graph and one session.
    """
    return _compile_attention(session, seed=7)


def _tensor(values: np.ndarray) -> Tensor:
    """A CPU tensor of ``values``, in its own numpy dtype.

    ``Module.compile`` takes and returns :class:`Tensor`, and one built from a
    CPU buffer stays on the CPU rather than following the box's default
    device.
    """
    return Tensor(storage=Buffer.from_numpy(np.ascontiguousarray(values)))


def test_window_start_is_the_last_window_up_to_the_anchor() -> None:
    """The window is the last ``window`` context positions of the pool."""
    assert window_start(0, WINDOW) == 0
    # The anchor's own position is in the window, so a pool one past it drops
    # exactly the window's first row.
    assert window_start(WINDOW - 1, WINDOW) == 0
    assert window_start(WINDOW, WINDOW) == 1
    assert window_start(2047, 2048) == 0
    assert window_start(5000, 2048) == 2953


def _attention_weights() -> dict[str, tuple[int, ...]]:
    return {
        "wq.weight": (HEADS * (NOPE + ROPE), HIDDEN),
        "wkv_a_with_mqa.weight": (LATENT + ROPE, HIDDEN),
        "kv_a_norm.weight": (LATENT,),
        "wkv_b.weight": (HEADS * (NOPE + V_HEAD), LATENT),
        "wo.weight": (HIDDEN, HEADS * V_HEAD),
    }


def _compile_attention(
    session: InferenceSession, seed: int
) -> tuple[_Attention, DSparkDrafterConfig]:
    """Compiles one drafter attention layer on the CPU with random weights."""
    config = _config()
    with _cpu_float32():
        layer = DSparkAttention(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            qk_nope_head_dim=NOPE,
            qk_rope_head_dim=ROPE,
            v_head_dim=V_HEAD,
            kv_lora_rank=LATENT,
            scale=config.attention_scale,
            norm_eps=config.rms_norm_eps,
            rope_inv_freq=config.rope_inv_freq,
        )
    weights = {
        name: _rand(shape, seed + index)
        for index, (name, shape) in enumerate(_attention_weights().items())
    }
    cpu = DeviceRef.CPU()
    run = layer.compile(
        TensorType(DType.float32, ["rows", HIDDEN], device=cpu),
        TensorType(DType.float32, ["ctx", config.latent_dim], device=cpu),
        TensorType(DType.int32, ["rows"], device=cpu),
        TensorType(DType.bool, ["rows", "keys"], device=cpu),
        session=session,
        weights=weights,
    )
    return run, config


def _block_mask(
    rows: int, n_ctx: int, anchor: int, *, window: int = WINDOW
) -> np.ndarray:
    """The reference's mask for one request: window ``[start, anchor]`` plus
    every block row."""
    start = window_start(anchor, window)
    context = np.arange(n_ctx)
    keep = ((context >= start) & (context <= anchor)).astype(bool)
    return np.concatenate(
        [np.broadcast_to(keep, (rows, n_ctx)), np.ones((rows, rows), bool)],
        axis=1,
    )


def _attend(
    run: _Attention,
    x: np.ndarray,
    keys: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Runs the compiled layer on one CPU batch."""
    out = run(
        _tensor(x),
        _tensor(keys),
        _tensor(positions.astype(np.int32)),
        _tensor(mask),
    )
    return out.to_numpy()


def test_block_attention_masks_the_context_window_on_both_ends(
    attention: tuple[_Attention, DSparkDrafterConfig],
) -> None:
    """Neither a row before the window nor one past the anchor reaches a row."""
    run, config = attention
    n_ctx = 6
    keys = _rand((n_ctx, config.latent_dim), seed=1, scale=0.3)
    positions = np.arange(4, 4 + DEPTH, dtype=np.int32)
    x = _rand((DEPTH, HIDDEN), seed=2, scale=0.3)
    # Anchor at 4 with a four-row window: the pool is [1, 4], and row 5 is the
    # token after the anchor -- the leak this mask exists to stop.
    mask = _block_mask(DEPTH, n_ctx, 4)

    base = _attend(run, x, keys, positions, mask)
    for dropped, label in ((0, "before the window"), (5, "after the anchor")):
        perturbed = keys.copy()
        perturbed[dropped, :] = 1e3
        np.testing.assert_array_equal(
            _attend(run, x, perturbed, positions, mask),
            base,
            err_msg=f"a context row {label} changed the block's output",
        )
    for kept in (1, 4):
        perturbed = keys.copy()
        perturbed[kept, :] = 1e3
        assert not np.allclose(
            _attend(run, x, perturbed, positions, mask), base
        ), f"a windowed context row ({kept}) has no effect on the block"


def test_block_attention_is_not_causal_inside_the_block(
    attention: tuple[_Attention, DSparkDrafterConfig],
) -> None:
    """The block's last row reaches its first: no causal mask in the block."""
    run, config = attention
    n_ctx = 4
    keys = _rand((n_ctx, config.latent_dim), seed=3, scale=0.3)
    positions = np.arange(n_ctx, n_ctx + DEPTH, dtype=np.int32)
    x = _rand((DEPTH, HIDDEN), seed=4, scale=0.3)
    mask = _block_mask(DEPTH, n_ctx, n_ctx - 1)

    base = _attend(run, x, keys, positions, mask)
    perturbed = x.copy()
    perturbed[-1, :] = 1e3
    moved = _attend(run, perturbed, keys, positions, mask)
    assert not np.allclose(moved[0], base[0]), (
        "the block's last row does not reach its first, so the block forward"
        " is causal"
    )


def _drafter_weights(config: DSparkDrafterConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter of the miniature drafter, with its shape."""
    shapes = {
        "backbone.target_projection.weight": (HIDDEN, config.tap_width),
        "backbone.target_norm.weight": (HIDDEN,),
        "backbone.final_norm.weight": (HIDDEN,),
        "backbone.mask_embedding": (DEPTH - 1, HIDDEN),
        "autoregressive_head.embedding.weight": (VOCAB, MARKOV_RANK),
        "autoregressive_head.output.weight": (VOCAB, MARKOV_RANK),
        "autoregressive_head.joint_proj.weight": (
            3 * MARKOV_RANK,
            HIDDEN + 2 * MARKOV_RANK,
        ),
        "autoregressive_head.joint_proj.bias": (3 * MARKOV_RANK,),
    }
    for name, shape in _attention_weights().items():
        shapes[f"backbone.layers.0.attention.{name}"] = shape
    shapes["backbone.layers.0.attention_norm.weight"] = (HIDDEN,)
    shapes["backbone.layers.0.ffn_norm.weight"] = (HIDDEN,)
    shapes["backbone.layers.0.feed_forward.w1.weight"] = (2 * HIDDEN, HIDDEN)
    shapes["backbone.layers.0.feed_forward.w3.weight"] = (2 * HIDDEN, HIDDEN)
    shapes["backbone.layers.0.feed_forward.w2.weight"] = (HIDDEN, 2 * HIDDEN)
    return shapes


@dataclass(frozen=True)
class CompiledDrafter:
    """One compiled miniature drafter and the arrays it was given."""

    run: Callable[..., Sequence[Tensor]]
    weights: dict[str, NDArray[Any]]
    config: DSparkDrafterConfig
    embed_weight: NDArray[Any]
    head_weight: NDArray[Any]


@pytest.fixture(scope="module")
def drafter(session: InferenceSession) -> CompiledDrafter:
    """One compiled draft step, shared by every whole-step check.

    A CPU compile of the drafter costs about a minute, so the chain, the
    window's upper bound and the batch layout are all measured on this graph.
    """
    config = _config()
    with _cpu_float32():
        module = DSparkDrafter(config, emit_intermediates=True)
    weights = {
        name: _rand(shape, seed=100 + index)
        for index, (name, shape) in enumerate(_drafter_weights(config).items())
    }
    cpu = DeviceRef.CPU()
    run = module.compile(
        TensorType(DType.float32, ["rows", config.tap_width], device=cpu),
        TensorType(DType.int32, ["rows"], device=cpu),
        TensorType(DType.int32, ["batch"], device=cpu),
        TensorType(DType.int32, ["batch"], device=cpu),
        TensorType(DType.float32, [VOCAB, HIDDEN], device=cpu),
        TensorType(DType.float32, [VOCAB, HIDDEN], device=cpu),
        session=session,
        weights=weights,
    )
    return CompiledDrafter(
        run=run,
        weights=weights,
        config=config,
        embed_weight=_rand((VOCAB, HIDDEN), seed=6),
        head_weight=_rand((VOCAB, HIDDEN), seed=7),
    )


def _per_request(
    values: NDArray[Any], index: int, request: int
) -> NDArray[Any]:
    """Splits one whole-step output into a request's piece of it.

    The head's outputs are ``[batch, depth, ...]`` and the block-row outputs
    ``[batch * depth, ...]``, both request-major. The context latents are per
    context row, which the whole batch shares.
    """
    if index in _PER_REQUEST_OUTPUTS:
        return values[request]
    if index in _BLOCK_ROW_OUTPUTS:
        return values[request * DEPTH : (request + 1) * DEPTH]
    return values


_PER_REQUEST_OUTPUTS = (2, 3, 4)
"""The whole-step outputs laid out per request: logits, tokens, states."""

_BLOCK_ROW_OUTPUTS = (0, 1)
"""The outputs laid out per block row: the head hidden and the base logits."""


def _step(
    compiled: CompiledDrafter,
    taps: NDArray[Any],
    anchor_positions: NDArray[Any],
    anchor_ids: NDArray[Any],
) -> list[NDArray[Any]]:
    """Runs one draft step over ``taps`` and returns every output as numpy."""
    outputs = compiled.run(
        _tensor(taps),
        _tensor(np.arange(taps.shape[0], dtype=np.int32)),
        _tensor(anchor_positions.astype(np.int32)),
        _tensor(anchor_ids.astype(np.int32)),
        _tensor(compiled.embed_weight),
        _tensor(compiled.head_weight),
    )
    return [out.to_numpy() for out in outputs]


def _reference_chain(
    weights: dict[str, np.ndarray],
    base: np.ndarray,
    hidden: np.ndarray,
    anchor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The reference's recurrent chain, in numpy, over one batch.

    Args:
        weights: The compiled drafter's own weights.
        base: ``[batch, depth, vocab]`` soft-capped base logits.
        hidden: ``[batch, depth, hidden]`` block rows.
        anchor_ids: ``[batch]`` anchors.

    Returns:
        ``(tokens [batch, depth], logits [batch, depth, vocab],
        states [batch, depth, rank])``.
    """
    batch, depth, _ = base.shape
    state = np.zeros((batch, MARKOV_RANK), dtype=np.float32)
    previous = anchor_ids.astype(np.int64)
    table = weights["autoregressive_head.embedding.weight"]
    states, tokens, logits = [], [], []
    for step in range(depth):
        states.append(state)
        # An anchor outside the vocabulary contributes no embedding, as the
        # reference's mask-token path does.
        valid = (previous >= 0) & (previous < table.shape[0])
        embedded = np.where(
            valid[:, None], table[np.clip(previous, 0, table.shape[0] - 1)], 0.0
        )
        joint = (
            np.concatenate([state, embedded, hidden[:, step, :]], axis=-1)
            @ weights["autoregressive_head.joint_proj.weight"].T
            + weights["autoregressive_head.joint_proj.bias"]
        )
        gate, candidate, emitted = np.split(joint, 3, axis=-1)
        gate = 1.0 / (1.0 + np.exp(-gate))
        state = gate * state + (1.0 - gate) * np.tanh(candidate)
        bias = np.tanh(emitted) @ weights["autoregressive_head.output.weight"].T
        # The bias is added to the capped logits and the sum is never capped.
        row = base[:, step, :] + bias
        previous = np.argmax(row, axis=-1)
        tokens.append(previous)
        logits.append(row)
    return (
        np.stack(tokens, axis=1),
        np.stack(logits, axis=1),
        np.stack(states, axis=1),
    )


def test_recurrent_head_chains_state_and_tokens(
    drafter: CompiledDrafter,
) -> None:
    """The whole step's chain matches the reference: state, bias, argmax."""
    config = drafter.config
    n_ctx = 6
    taps = _rand((n_ctx, config.tap_width), seed=5, scale=0.5)
    head_hidden, base, logits, tokens, states, *latents = _step(
        drafter, taps, np.array([n_ctx - 1]), np.array([3])
    )
    assert head_hidden.shape == (DEPTH, HIDDEN)
    assert base.shape == (DEPTH, VOCAB)
    assert len(latents) == config.num_layers

    expected_tokens, expected_logits, expected_states = _reference_chain(
        drafter.weights,
        base.reshape(1, DEPTH, VOCAB),
        head_hidden.reshape(1, DEPTH, HIDDEN),
        np.array([3], dtype=np.int64),
    )
    np.testing.assert_allclose(
        states.reshape(1, DEPTH, MARKOV_RANK),
        expected_states,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        logits.reshape(1, DEPTH, VOCAB), expected_logits, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(tokens.reshape(1, DEPTH), expected_tokens)
    # The soft cap covers the shared head only: the biased rows are free to
    # leave it, and the reference's own fixture rows do.
    assert np.max(np.abs(base)) <= config.softcap + 1e-5


def test_rows_after_the_anchor_cannot_reach_the_block(
    drafter: CompiledDrafter,
) -> None:
    """A pool holding future rows must draft exactly as the anchor's own does.

    This is the leak the mask exists to stop: a caller holding one prompt's
    latents hands the drafter every row it has, and an unbounded window let
    every block read the tokens *after* its anchor. Perturbing those rows may
    not move a single logit; perturbing the anchor's own row must.
    """
    config = drafter.config
    n_ctx = 8
    anchor = 5
    taps = _rand((n_ctx, config.tap_width), seed=11, scale=0.5)
    anchors = np.array([anchor])
    ids = np.array([3])

    base_logits, base_tokens = _step(drafter, taps, anchors, ids)[2:4]
    future = taps.copy()
    future[anchor + 1 :, :] = 1e3
    logits, tokens = _step(drafter, future, anchors, ids)[2:4]
    np.testing.assert_array_equal(logits, base_logits)
    np.testing.assert_array_equal(tokens, base_tokens)

    inside = taps.copy()
    inside[anchor, :] = 1e3
    moved = _step(drafter, inside, anchors, ids)[2]
    assert not np.allclose(moved, base_logits), (
        "the anchor's own context row has no effect on the block, so the mask"
        " is not reading the window it claims"
    )


def test_rows_before_the_window_cannot_reach_the_block(
    drafter: CompiledDrafter,
) -> None:
    """Only the last ``window`` context rows are keys, not the whole pool.

    The other half of the bound: an anchor far enough into a prompt leaves
    earlier rows outside its window, and a mask that reads them anyway measures
    a different model (``dspark-reference.md`` §5).
    """
    config = drafter.config
    n_ctx = 10
    anchor = 8
    assert window_start(anchor, config.window) > 0
    taps = _rand((n_ctx, config.tap_width), seed=21, scale=0.5)
    anchors = np.array([anchor])
    ids = np.array([3])

    base_logits, base_tokens = _step(drafter, taps, anchors, ids)[2:4]
    stale = taps.copy()
    stale[: window_start(anchor, config.window), :] = 1e3
    logits, tokens = _step(drafter, stale, anchors, ids)[2:4]
    np.testing.assert_array_equal(logits, base_logits)
    np.testing.assert_array_equal(tokens, base_tokens)

    # The window's first row is a key, so moving it must move the block.
    inside = taps.copy()
    inside[window_start(anchor, config.window), :] = 1e3
    assert not np.allclose(_step(drafter, inside, anchors, ids)[2], base_logits)


class _AnchorBounds(Module[..., Tensor]):
    """Compares anchors against the scalars the helper builds from them."""

    def forward(self, anchors: Tensor) -> Tensor:
        """Returns ``0 <= anchor < VOCAB``, one bool per row."""
        return (anchors >= device_scalar(0, anchors)) & (
            anchors < device_scalar(VOCAB, anchors)
        )


def test_an_int64_anchor_compares_at_its_own_dtype(
    session: InferenceSession,
) -> None:
    """The scalar bounds take the anchor tensor's dtype, not a fixed int32.

    The block driver hands the drafter int64 token ids, and an elementwise
    comparison refuses two dtypes: a scalar built as int32 would fail the graph
    build rather than clamp an out-of-range anchor.
    """
    module = _AnchorBounds()
    run = module.compile(
        TensorType(DType.int64, ["batch"], device=DeviceRef.CPU()),
        session=session,
        weights={},
    )
    anchors = np.array([-2, 0, 3, VOCAB, VOCAB + 3], dtype=np.int64)
    produced = run(_tensor(anchors))
    got = (produced if isinstance(produced, Tensor) else produced[0]).to_numpy()
    np.testing.assert_array_equal(got, (anchors >= 0) & (anchors < VOCAB))


def test_two_anchors_over_one_shared_context_draft_independently(
    drafter: CompiledDrafter,
) -> None:
    """Two anchors on one context pool draft as each would alone.

    The rows are request-major and each block is bounded by *its own* anchor,
    so batching two anchors over the same context may not let one block's
    window, or its three rows, reach the other's. That is the whole of what a
    batch means to this mask: one pool, many anchors (the one-prompt case in
    :meth:`DSparkDrafter.forward`).
    """
    config = drafter.config
    n_ctx = 8
    taps = _rand((n_ctx, config.tap_width), seed=12, scale=0.5)
    anchors = np.array([2, 5])
    ids = np.array([3, 7])

    batched = _step(drafter, taps, anchors, ids)
    alone = [
        _step(drafter, taps, np.array([anchor]), np.array([token]))
        for anchor, token in zip(anchors, ids, strict=True)
    ]
    for index, together in enumerate(batched):
        for request, single_outputs in enumerate(alone):
            np.testing.assert_allclose(
                _per_request(together, index, request),
                _per_request(single_outputs[index], index, 0),
                rtol=1e-6,
                atol=1e-6,
            )
    # Two anchors that share a context but not a position draft differently.
    assert not np.array_equal(batched[3][0], batched[3][1])


class _BlockPositions(Module[..., Tensor]):
    """Traces one :meth:`DSparkDrafter.block_positions` call."""

    def __init__(self, config: DSparkDrafterConfig) -> None:
        """Wraps a drafter whose parameters are never read."""
        super().__init__()
        self.drafter = DSparkDrafter(config)

    def forward(self, anchor_positions: Tensor) -> Tensor:
        """Returns each block row's rotated position."""
        return self.drafter.block_positions(anchor_positions)


def test_the_block_positions_are_computed_by_the_module(
    session: InferenceSession,
) -> None:
    """The offset reaches the module's own arithmetic, at both conventions.

    A formula in the test would only check itself, so this calls the module:
    row ``j`` of a block anchored at ``p`` is rotated at ``p + offset + j``.
    """
    resolved: dict[int, np.ndarray] = {}
    for offset in (BLOCK_POSITION_OFFSET_TRTLLM, BLOCK_POSITION_OFFSET_VLLM):
        config = _config(block_position_offset=offset)
        with _cpu_float32():
            tracer = _BlockPositions(config)
        run = tracer.compile(
            TensorType(DType.int32, ["batch"], device=DeviceRef.CPU()),
            session=session,
            weights={
                f"drafter.{name}": _rand(shape, seed=100 + index)
                for index, (name, shape) in enumerate(
                    _drafter_weights(config).items()
                )
            },
        )
        anchors = np.array([10, 40], dtype=np.int32)
        produced = run(_tensor(anchors))
        # A single output comes back bare rather than in a list.
        resolved[offset] = (
            produced if isinstance(produced, Tensor) else produced[0]
        ).to_numpy()
        expected = np.stack(
            [np.arange(p + offset, p + offset + DEPTH) for p in anchors]
        ).reshape(-1)
        np.testing.assert_array_equal(resolved[offset], expected)
    assert not np.array_equal(
        resolved[BLOCK_POSITION_OFFSET_TRTLLM],
        resolved[BLOCK_POSITION_OFFSET_VLLM],
    )


def test_the_block_position_offset_is_a_config_constant() -> None:
    """The rotation offset is one validated constant, TRT-LLM's by default."""
    assert _config().block_position_offset == BLOCK_POSITION_OFFSET_TRTLLM
    with pytest.raises(ValueError, match="block_position_offset"):
        _config(block_position_offset=2)


def test_config_is_read_from_the_checkpoints_mtp_block() -> None:
    """A full checkpoint's ``mtp`` block resolves to the shipped geometry."""
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    config = DSparkDrafterConfig.from_normalized(normalized)
    assert config.target_layers == (1, 18, 36, 54, 71)
    assert config.tap_width == 5 * config.hidden_size
    assert config.latent_dim == 512 + 64
    assert config.num_layers == 3
    assert config.depth == 3
    assert config.window == 2048
    assert config.markov_rank == 512
    assert config.block_position_offset == BLOCK_POSITION_OFFSET_TRTLLM
    assert (
        DSparkDrafterConfig.from_normalized(
            normalized, block_position_offset=BLOCK_POSITION_OFFSET_VLLM
        ).block_position_offset
        == BLOCK_POSITION_OFFSET_VLLM
    )


def test_the_drafters_rope_is_plain_yarn() -> None:
    """LeRoPE is off for the drafter: the table is YaRN with alpha at zero."""
    config = DSparkDrafterConfig.from_normalized(
        MistralLarge4NormalizedConfig.from_params_json(_params())
    )
    # The first six entries of the reference's 64-wide table, so a changed
    # theta, ramp or LeRoPE term fails here rather than moving every draft
    # logit.
    np.testing.assert_allclose(
        config.rope_inv_freq[:6],
        [
            1.0,
            0.7498942018,
            0.5623413324,
            0.4216965139,
            0.2919144332,
            0.2006725073,
        ],
        rtol=1e-7,
    )
    assert config.rope_inv_freq.shape == (config.qk_rope_head_dim // 2,)


def test_the_drafter_cache_is_one_plane_of_latents() -> None:
    """The drafter's cache is 576-wide, unquantized, one row per layer."""
    leaf = build_dspark_kv_params(
        KVCacheConfig(kv_cache_page_size=128),
        devices=[DeviceRef.CPU()],
        config=_config(),
    )
    assert leaf.dtype == DType.bfloat16
    assert leaf.head_dim == LATENT + ROPE
    assert leaf.num_layers == 1
    assert leaf.n_kv_heads_per_device == 1
    # One plane per slot -- no separate value half -- of 576-wide latents.
    assert leaf.shape_per_block == [1, 1, 128, 1, LATENT + ROPE]


def test_the_weight_names_are_the_checkpoints_own() -> None:
    """The module's parameter names are the checkpoint block minus its prefix."""
    names = dspark_tensor_names(3)
    assert len(names) == 3 * 10 + 8
    assert "mtp_dspark.backbone.layers.2.feed_forward.w2.weight" in names
    assert "mtp_dspark.backbone.mask_embedding" in names
    assert "mtp_dspark.autoregressive_head.joint_proj.bias" in names
    converted = convert_dspark_state_dict(
        {name: np.zeros(1, dtype=np.float32) for name in names}, 3
    )
    assert len(converted) == len(names)
    assert "backbone.layers.0.attention.wq.weight" in converted
    with pytest.raises(ValueError, match="missing"):
        convert_dspark_state_dict({names[0]: np.zeros(1, dtype=np.float32)}, 3)


def test_an_anchor_outside_the_vocabulary_embeds_to_zero(
    drafter: CompiledDrafter,
) -> None:
    """A mask-token anchor is looked up out of range on purpose, and survives.

    The checkpoint's own drafting code uses an id past ``vocab_size`` for the
    row-0 sentinel; the reference zeroes that embedding rather than reading a
    row the table does not have, and this checks the same path through the
    compiled module.
    """
    config = drafter.config
    n_ctx = 5
    taps = _rand((n_ctx, config.tap_width), seed=13, scale=0.5)
    head_hidden, base, logits, tokens, states = _step(
        drafter, taps, np.array([n_ctx - 1]), np.array([VOCAB + 3])
    )[:5]
    expected_tokens, expected_logits, expected_states = _reference_chain(
        drafter.weights,
        base.reshape(1, DEPTH, VOCAB),
        head_hidden.reshape(1, DEPTH, HIDDEN),
        np.array([VOCAB + 3]),
    )
    np.testing.assert_allclose(
        logits.reshape(1, DEPTH, VOCAB), expected_logits, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        states.reshape(1, DEPTH, MARKOV_RANK),
        expected_states,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(tokens.reshape(1, DEPTH), expected_tokens)
    assert np.all(np.isfinite(logits))


@dataclass(frozen=True)
class _TapTextConfig:
    """A generated ``config.json``'s text config, as ``initialize`` sees it."""

    num_hidden_layers: int
    dspark_target_activation_layers: list[int] | None = None
    mtp: dict[str, object] | None = None


@dataclass(frozen=True)
class _Repo:
    """A weights repo handle, as far as the tap resolution reads it.

    ``HuggingFaceRepo`` satisfies this shape; a Hub serve would build it from
    the model id, so ``repo_type`` is what decides whether the directory is on
    disk already or has to be looked up in the cache.
    """

    repo_type: str = "local"
    local_path: str = ""
    repo_id: str = "mistralai/Mistral-Large-4"
    revision: str = "main"
    subfolder: str | None = None


@dataclass(frozen=True)
class _LocalModel:
    """A served checkpoint, as the tap resolution reads it.

    The field types mirror ``MAXModelConfig``'s, which
    ``test_the_local_model_fixture_mirrors_the_served_config_types`` checks:
    this stand-in previously declared ``weight_path`` as a single string, and
    the real field is a list of repo-relative weight filenames, so the mismatch
    hid a ``TypeError`` on the standard serve path.
    """

    model_path: str = ""
    weight_path: list[Path] = field(default_factory=list)
    huggingface_weight_repo: _Repo = field(default_factory=_Repo)


def test_the_local_model_fixture_mirrors_the_served_config_types() -> None:
    """The stand-in this file hands the resolver declares the real shape."""
    hints = get_type_hints(_LocalModel)
    fields = MAXModelConfig.model_fields
    assert hints["model_path"] == fields["model_path"].annotation
    assert hints["weight_path"] == fields["weight_path"].annotation
    # The repo handle is a computed field on the real config, so its presence
    # is what is checked rather than a stored annotation.
    assert "huggingface_weight_repo" in MAXModelConfig.model_computed_fields


def _cold_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes a Hub repo with nothing snapshotted answer "no local files".

    Both halves of the lookup are closed: the cache snapshot resolves to
    nothing, and the fetch of a ``params.json`` the snapshot does not hold
    reaches no network and finds no file.
    """

    def missing(repo_id: str, revision: str = "main") -> str:
        raise FileNotFoundError(f"{repo_id}@{revision} is not in the HF cache")

    monkeypatch.setattr(
        modulev3_checkpoint, "generate_local_model_path", missing
    )
    monkeypatch.setattr(
        modulev3_checkpoint,
        "fetch_repo_file",
        lambda repo, filename: None,
    )


def _stage_params(directory: Path, **mtp: object) -> Path:
    """Writes a ``params.json`` carrying an ``mtp`` block into ``directory``."""
    params = _params()
    block = params["mtp"]
    assert isinstance(block, dict)
    staged = dict(block)
    staged.update(mtp)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "params.json"
    path.write_text(json.dumps({"mtp": staged}))
    return path


def _staged(directory: Path) -> _LocalModel:
    """A served config whose weights sit in a staged checkpoint directory."""
    return _LocalModel(huggingface_weight_repo=_Repo(local_path=str(directory)))


def test_the_served_config_reaches_the_checkpoints_taps(tmp_path: Path) -> None:
    """A served graph captures the checkpoint's five taps, not none.

    The generated ``config.json`` carries no tap list, so the served path has
    to read the staged checkpoint's ``params.json``: without this a full-model
    initialize captures nothing and the drafter starves.
    """
    staged = _stage_params(tmp_path / "checkpoint")
    assert _served_taps(_TapTextConfig(72), _staged(staged.parent)) == [
        1,
        18,
        36,
        54,
        71,
    ]
    # A declared list wins over the checkpoint and needs no local directory.
    assert _served_taps(
        _TapTextConfig(72, dspark_target_activation_layers=[2, 9]),
        _LocalModel(),
    ) == [2, 9]


def test_a_hub_repo_reads_its_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hub serve reads the snapshot directory the cache resolves to.

    This is the standard serve path: ``model_path`` is a repo id that is not on
    disk, and ``weight_path`` holds repo-relative filenames, so neither names a
    directory. The snapshot the loader downloaded them into does.
    """
    snapshot = tmp_path / "hub" / "models--mistralai--Mistral-Large-4"
    _stage_params(snapshot)
    monkeypatch.setattr(
        modulev3_checkpoint,
        "generate_local_model_path",
        lambda repo_id, revision="main": str(snapshot),
    )
    served = _served_taps(
        _TapTextConfig(72),
        _LocalModel(
            model_path="mistralai/Mistral-Large-4",
            weight_path=[
                Path("model-00001-of-00002.safetensors"),
                Path("model-00002-of-00002.safetensors"),
            ],
            huggingface_weight_repo=_Repo(repo_type="online"),
        ),
    )
    assert served == [1, 18, 36, 54, 71]


def test_a_hub_repo_reads_its_snapshot_subfolder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo whose weights live under a subfolder is read under it."""
    snapshot = tmp_path / "snapshot"
    _stage_params(snapshot / "language_model")
    monkeypatch.setattr(
        modulev3_checkpoint,
        "generate_local_model_path",
        lambda repo_id, revision="main": str(snapshot),
    )
    assert _served_taps(
        _TapTextConfig(72),
        _LocalModel(
            model_path="mistralai/Mistral-Large-4",
            huggingface_weight_repo=_Repo(
                repo_type="online", subfolder="language_model"
            ),
        ),
    ) == [1, 18, 36, 54, 71]


def test_a_relative_weight_path_never_reads_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo-relative weight filename does not make the CWD the checkpoint.

    ``Path("model.safetensors").parent`` is ``.``, so deriving a directory from
    a Hub serve's weight path pointed the tap resolution at the process working
    directory -- where any stray ``params.json`` would silently describe a
    checkpoint this process is not serving. Nothing local means no taps.
    """
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "params.json").write_text(
        json.dumps({"mtp": {"target_activation_layers": [9]}})
    )
    monkeypatch.chdir(decoy)
    _cold_cache(monkeypatch)
    assert (
        _served_taps(
            _TapTextConfig(72),
            _LocalModel(
                model_path="mistralai/Mistral-Large-4",
                weight_path=[Path("model.safetensors")],
                huggingface_weight_repo=_Repo(repo_type="online"),
            ),
        )
        == []
    )


def test_an_absolute_weight_file_names_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly given local weight file does say where the shards are."""
    weights = tmp_path / "weights"
    shard = weights / "model-00001-of-00002.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"")
    _stage_params(weights)
    _cold_cache(monkeypatch)
    assert _served_taps(
        _TapTextConfig(72),
        _LocalModel(
            model_path="mistralai/Mistral-Large-4",
            weight_path=[shard, weights / "model-00002-of-00002.safetensors"],
            huggingface_weight_repo=_Repo(repo_type="online"),
        ),
    ) == [1, 18, 36, 54, 71]


def test_a_model_with_no_local_params_json_serves_without_taps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo id with nothing on disk resolves to no taps, and does not raise.

    Most served checkpoints carry no in-checkpoint drafter at all, so the miss
    has to be a quiet empty list rather than an exception on the way to serve.
    """
    _cold_cache(monkeypatch)
    shard = tmp_path / "cache" / "model.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"")
    assert (
        _served_taps(
            _TapTextConfig(72),
            _LocalModel(
                model_path="mistralai/Mistral-Large-4",
                weight_path=[Path("model.safetensors")],
                huggingface_weight_repo=_Repo(repo_type="online"),
            ),
        )
        == []
    )
    assert _served_taps(_TapTextConfig(72), _LocalModel()) == []


def test_a_params_json_without_an_mtp_block_tries_the_next_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo's ``params.json`` need not be the one that carries ``mtp``.

    A staged or re-exported directory can carry the family's ``params.json``
    while the weights come from the checkpoint, so a first candidate without an
    ``mtp`` block is a miss, not the answer.
    """
    staged = tmp_path / "checkpoint"
    staged.mkdir()
    (staged / "params.json").write_text(json.dumps({"max_position": 4096}))
    weights = tmp_path / "weights"
    shard = weights / "model.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"")
    _stage_params(weights)
    _cold_cache(monkeypatch)
    assert _served_taps(
        _TapTextConfig(72),
        _LocalModel(
            model_path="mistralai/Mistral-Large-4",
            weight_path=[shard],
            huggingface_weight_repo=_Repo(local_path=str(staged)),
        ),
    ) == [1, 18, 36, 54, 71]


def test_the_shard_directory_wins_over_the_model_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both name a checkpoint, the one holding the weights is served.

    A ``model_path`` directory can be a staging leftover while the weights are
    read from elsewhere; the taps have to describe the weights, so the shards'
    directory is tried first and the model directory only after it.
    """
    staged = _stage_params(tmp_path / "model_dir", target_activation_layers=[3])
    weights = _stage_params(
        tmp_path / "weights", target_activation_layers=[1, 5]
    )
    _cold_cache(monkeypatch)
    # Only the model directory names one here, so that is what is read.
    assert _served_taps(
        _TapTextConfig(72), _LocalModel(str(staged.parent))
    ) == [3]
    assert _served_taps(
        _TapTextConfig(72),
        _LocalModel(
            str(staged.parent),
            huggingface_weight_repo=_Repo(local_path=str(weights.parent)),
        ),
    ) == [1, 5]


def test_a_malformed_params_json_names_the_file_and_raises(
    tmp_path: Path,
) -> None:
    """A corrupt ``params.json`` is reported, not served past silently.

    A missing file is a miss (the model carries no drafter); a file that is
    there and unreadable is a broken checkpoint, and the message has to say
    which file, because the resolver looks in more than one directory.
    """
    staged = tmp_path / "checkpoint"
    staged.mkdir()
    path = staged / "params.json"
    path.write_text("{ not json")
    with pytest.raises(ValueError) as unreadable:
        _served_taps(_TapTextConfig(72), _staged(staged))
    assert str(path) in str(unreadable.value)

    path.write_text(json.dumps({"mtp": {"target_activation_layers": ["taps"]}}))
    with pytest.raises(ValueError) as not_indices:
        _served_taps(_TapTextConfig(72), _staged(staged))
    assert str(path) in str(not_indices.value)


def test_a_cut_schedule_captures_no_taps(tmp_path: Path) -> None:
    """A graph shorter than the last tap captures nothing, not a subset.

    A partial concat would misalign the drafter's projection columns against
    the checkpoint's, so both the builder and the config validator refuse it:
    one drops the whole list, the other rejects a list that names a layer the
    schedule does not have.
    """
    staged = tmp_path / "checkpoint"
    staged.mkdir()
    (staged / "params.json").write_text(json.dumps({"mtp": _params()["mtp"]}))
    assert _served_taps(_TapTextConfig(4), _LocalModel(str(staged))) == []
    normalized = MistralLarge4NormalizedConfig.from_params_json(_params())
    schedule = normalized.layers
    cut = replace(
        normalized,
        n_layers=4,
        layers=replace(
            schedule,
            rope=schedule.rope[:4],
            owns_indexer=schedule.owns_indexer[:4],
            indexer_source=schedule.indexer_source[:4],
            kv_source=schedule.kv_source[:4],
            moe=schedule.moe[:4],
        ),
    )
    assert _fitted_taps(cut) == []


def test_the_drafter_rejects_a_head_or_block_input_it_cannot_run() -> None:
    """The checkpoint's declared head, block input and norm are validated.

    ``head_type`` picks between a gated recurrent cell and the vanilla Markov
    head, ``draft_input`` between the learned mask rows and re-embedded draft
    tokens, and ``norm_strategy`` between eagle's re-norming and another
    schedule: each is a different trained parameter, so a checkpoint that
    declares another value is refused at config time.
    """
    for key, value in (
        ("head_type", "markov"),
        ("draft_input", "token"),
        ("norm_strategy", "pre_norm"),
    ):
        params = copy.deepcopy(_params())
        block = params["mtp"]
        assert isinstance(block, dict)
        # ``head_type`` and ``draft_input`` sit under ``mtp.dspark``;
        # ``norm_strategy`` is the checkpoint's one ``mtp``-level norm setting.
        target = block["dspark"] if key != "norm_strategy" else block
        assert isinstance(target, dict)
        target[key] = value
        normalized = MistralLarge4NormalizedConfig.from_params_json(params)
        with pytest.raises(ValueError, match=key):
            DSparkDrafterConfig.from_normalized(normalized)
