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
"""Verifies the latent ragged MLA path matches the padded parity form.

Builds one layer with real weights and runs both ``forward`` (padded,
already proven against the reference checkpoint) and
``forward_latent_ragged`` (a single prefill into an empty cache) over the
same input, then compares outputs -- the only numerical difference between
the two is which kernel computes the same attention, so they should match
closely.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_architecture_name,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    MIXED_STEP_SPLIT_SIZE,
    MLAKVCacheParams,
    padded_lut_cols,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from max.tensor import Tensor, compilation, default_device
from max.tensor.sharding import BufferLayout, TensorLayout

# The precompile record cannot take a `-k` filter, so the split rides
# this env var. On NVIDIA, the oversized-extent control covers the
# forward path in presubmit because the stale-extent RED only runs on gfx9.
_FULL_SWEEP = os.environ.get("K3_LATENT_RAGGED_FULL_SWEEP") == "1"

_HIDDEN = 256
_NUM_HEADS = 8
_QK_NOPE_HEAD_DIM = 64
_V_HEAD_DIM = 64
_Q_LORA_RANK = 128
_SEQ_LEN = 6
_PAGE_SIZE = 128
_BUFFER_SIZE = 256
_STALE_EXTENT = 128
"""The extent a cache-0 warmup batch computes for this step's shape --
the value a captured graph would freeze if it read the extent at capture
and never refreshed it, and the undersized bound this file's capture
control must show corrupting."""


def _weighted_latent_layer(
    device: Accelerator, arr: Callable[..., np.ndarray]
) -> KimiK3MLAAttention:
    """Builds the fused-MLA layer with real weights, drawing seven of them.

    ``arr`` draws each weight in the order the assignments run, so a caller
    passing its own generator keeps its draws byte-identical to a build
    that assigned the weights inline.
    """
    layer = KimiK3MLAAttention(
        hidden_size=_HIDDEN,
        num_heads=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        kv_params=MLAKVCacheParams(
            dtype=DType.bfloat16,
            head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
            num_layers=1,
            devices=[DeviceRef.from_device(device)],
            page_size=_PAGE_SIZE,
            num_q_heads=_NUM_HEADS,
        ),
        layer_idx=0,
    )
    layer.q_a_proj.weight = Tensor(
        arr(_Q_LORA_RANK, _HIDDEN), dtype=DType.float32
    ).cast(DType.bfloat16)
    layer.q_a_layernorm.weight = Tensor(
        np.ones(_Q_LORA_RANK, dtype=np.float32), dtype=DType.float32
    ).cast(DType.bfloat16)
    layer.q_b_proj.weight = Tensor(
        arr(
            _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
            _Q_LORA_RANK,
        ),
        dtype=DType.float32,
    ).cast(DType.bfloat16)
    layer.kv_a_proj_with_mqa.weight = Tensor(
        arr(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN), dtype=DType.float32
    ).cast(DType.bfloat16)
    layer.kv_a_layernorm.weight = Tensor(
        np.ones(_KV_LORA_RANK, dtype=np.float32), dtype=DType.float32
    ).cast(DType.bfloat16)
    layer.kv_b_proj.weight = Tensor(
        arr(_NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK),
        dtype=DType.float32,
    ).cast(DType.bfloat16)
    layer.g_proj.weight = Tensor(
        arr(_NUM_HEADS * _V_HEAD_DIM, _HIDDEN), dtype=DType.float32
    ).cast(DType.bfloat16)
    layer.o_proj.weight = Tensor(
        arr(_HIDDEN, _NUM_HEADS * _V_HEAD_DIM), dtype=DType.float32
    ).cast(DType.bfloat16)
    return layer


# The fused MLA prefill/decode kernel only instantiates for a 576-wide latent
# cache (512 + 64) -- `flareMLA_decoding`'s `constraint failed: ... head_dim
# 576 (rotary) or 512 (NoPE)`. Real K3 and K3-mini are exactly this shape; see
# `kimi_k3/model_config.py`'s `KimiK3Config._MLA_KERNEL_LATENT_DIM`.
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64


@pytest.mark.compile_on_gpu(
    "attention.mha.no_cache is staged as an undeclared custom op the interpreter can't bind"
)
@pytest.mark.skipif(
    not _FULL_SWEEP,
    reason="runs in the postsubmit sweep target",
)
def test_forward_latent_ragged_matches_forward() -> None:
    rng = np.random.default_rng(0)

    def arr(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    device = Accelerator()
    with default_device(device):
        layer = _weighted_latent_layer(device, arr)

        x_np = arr(_SEQ_LEN, _HIDDEN)
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)

        padded = Tensor(
            x_np.reshape(1, _SEQ_LEN, _HIDDEN), dtype=DType.float32
        ).cast(DType.bfloat16)
        out_padded = layer(padded).cast(DType.float32).to_numpy()[0]

        assert isinstance(layer.kv_params, MLAKVCacheParams)
        kv_collection = build_paged_cache(layer.kv_params, [_SEQ_LEN], device)
        input_row_offsets = Tensor(
            np.array([0, _SEQ_LEN], dtype=np.uint32), dtype=DType.uint32
        )
        prefill_plan = kimi_k3_mla_prefill_plan(
            kv_params=layer.kv_params,
            kv_collection=kv_collection,
            input_row_offsets=input_row_offsets,
            buffer_size=_PAGE_SIZE,
            buffer_length=Tensor(
                np.array([_PAGE_SIZE], dtype=np.int32), device=CPU()
            ),
        )
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=_SEQ_LEN,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            dtype=DType.bfloat16,
            device=device,
        )
        out_latent_ragged = (
            layer.forward_latent_ragged(
                x, kv_collection, input_row_offsets, prefill_plan, freqs_cis
            )
            .cast(DType.float32)
            .to_numpy()
        )

    err = float(
        np.max(np.abs(out_latent_ragged - out_padded))
        / max(np.max(np.abs(out_padded)), 1e-12)
    )
    assert err < 1e-2, f"latent ragged MLA vs padded parity diverges: {err}"


@pytest.mark.skipif(_FULL_SWEEP, reason="runs in the presubmit target")
def test_prefill_plan_capture_replays_across_page_boundaries() -> None:
    device = Accelerator()
    params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        num_q_heads=_NUM_HEADS,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.from_device(device)],
        page_size=_PAGE_SIZE,
    )
    cache_types = {
        "kv_blocks": BufferLayout(
            params.dtype, [3, *params.shape_per_block], device
        ),
        "cache_lengths": TensorLayout(DType.uint32, [1], device),
        "lookup_table": TensorLayout(
            DType.uint32, [1, padded_lut_cols(2)], device
        ),
        "max_prompt_length": TensorLayout(DType.uint32, [1], CPU()),
        "max_cache_length": TensorLayout(DType.uint32, [1], CPU()),
        "page_stride": TensorLayout(DType.int64, [1], CPU()),
        "attention_dispatch_metadata": TensorLayout(DType.int64, [3], device),
        "mla_num_partitions": TensorLayout(DType.int64, [1], CPU()),
        "mixed_step_split": TensorLayout(
            DType.uint32, [MIXED_STEP_SPLIT_SIZE], CPU()
        ),
    }

    def plan(
        row_offsets: Tensor, host_length: Tensor, *values: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        fields = dict(zip(cache_types, values, strict=True))
        collection = PagedCacheValues(
            kv_blocks=fields["kv_blocks"],
            cache_lengths=fields["cache_lengths"],
            lookup_table=fields["lookup_table"],
            max_prompt_length=fields["max_prompt_length"],
            max_cache_length=fields["max_cache_length"],
            page_stride=fields["page_stride"],
            attention_dispatch_metadata=fields["attention_dispatch_metadata"],
            mla_num_partitions=fields["mla_num_partitions"],
            mixed_step_split=fields["mixed_step_split"],
        )
        result = kimi_k3_mla_prefill_plan(
            kv_params=params,
            kv_collection=collection,
            input_row_offsets=row_offsets,
            buffer_size=256,
            buffer_length=host_length,
        )
        # The plan reserves batch+1 cache-offset columns but writes only batch.
        return (
            result.buffer_row_offsets,
            result.cache_offsets[:, :1],
            result.buffer_lengths,
        )

    run = compilation.compile(plan)(
        TensorLayout(DType.uint32, [2], device),
        TensorLayout(DType.int32, [1], CPU()),
        *cache_types.values(),
    )
    cache = build_paged_cache(params, [2], device, cache_lengths=[254])
    cache_fields = {
        name: value
        for name, value in vars(cache).items()
        if isinstance(value, Tensor)
    }
    offsets = Tensor(np.array([0, 2], dtype=np.uint32), device=device)
    host_length = Tensor(np.array([256], dtype=np.int32), device=CPU())
    inputs = [offsets, host_length, *cache_fields.values()]
    buffers = [value.buffers[0] for value in inputs]
    captured = run.engine_model.capture(1, *buffers)
    for cache_length in [126, 127, 254]:
        cache.cache_lengths.buffers[0].inplace_copy_from(
            Buffer.from_numpy(np.array([cache_length], dtype=np.uint32))
        )
        cache.max_cache_length.buffers[0].to_numpy()[0] = cache_length + 2
        host_length.buffers[0].to_numpy()[0] = (
            (cache_length + 2 + _PAGE_SIZE - 1) // _PAGE_SIZE
        ) * _PAGE_SIZE
        run.engine_model.replay(1, *buffers)
        actual = [value.to_numpy().copy() for value in captured]
        expected = run(*inputs)
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(got, want.to_numpy())


@dataclass
class _PrefillArmHarness:
    """The compiled prefill arm and its live cache, shared by the controls.

    Both capture controls grade the same compiled step, so they share one
    build: the drawn-weight layer, the compiled ``forward`` that plans and
    runs one ragged step, and a cache holding one request with 254 cached
    tokens and 2 new ones. The step's real window is
    ``align_up(254 + 2, 128) = 256`` rows -- exactly one buffer -- so
    every extent below it undersizes the decompression copy and every
    extent above it only adds rows the attention never reads.
    """

    run: compilation.CompiledCallable[..., Any]
    device: Accelerator
    host_length: Tensor
    kv_blocks: Tensor
    inputs: list[Tensor]
    buffers: list[Buffer]

    def set_host_extent(self, extent: int) -> None:
        """Installs ``extent`` as the live host value the plan reads."""
        self.host_length.buffers[0].to_numpy()[0] = extent

    def rewrite_cache_blocks(self, arr: Callable[..., np.ndarray]) -> None:
        """Overwrites every latent cache block with a fresh draw.

        The blocks are an input buffer the captured graph reads through,
        so a rewrite between a capture and its replay is exactly the live
        data a replaying step decompresses.
        """
        shape = tuple(int(dim) for dim in self.kv_blocks.shape)
        self.kv_blocks.buffers[0].inplace_copy_from(
            Tensor(arr(*shape), dtype=DType.float32)
            .to(self.device)
            .cast(DType.bfloat16)
            .buffers[0]
        )


@pytest.fixture(scope="module")
def prefill_arm_harness() -> _PrefillArmHarness:
    """Builds the compiled prefill arm once; the compile is the cost."""
    rng = np.random.default_rng(1)

    def arr(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    device = Accelerator()
    with default_device(device):
        layer = _weighted_latent_layer(device, arr)
        assert isinstance(layer.kv_params, MLAKVCacheParams)
        params = layer.kv_params
        # One request holding 254 cached tokens and 2 new ones: the plan's
        # real total is align_up(254 + 2, 128) = 256 rows, exactly one
        # buffer, and every extent below it undersizes the window.
        cache_types = {
            "kv_blocks": BufferLayout(
                params.dtype, [3, *params.shape_per_block], device
            ),
            "cache_lengths": TensorLayout(DType.uint32, [1], device),
            "lookup_table": TensorLayout(
                DType.uint32, [1, padded_lut_cols(2)], device
            ),
            "max_prompt_length": TensorLayout(DType.uint32, [1], CPU()),
            "max_cache_length": TensorLayout(DType.uint32, [1], CPU()),
            "page_stride": TensorLayout(DType.int64, [1], CPU()),
            "attention_dispatch_metadata": TensorLayout(
                DType.int64, [3], device
            ),
            "mla_num_partitions": TensorLayout(DType.int64, [1], CPU()),
            "mixed_step_split": TensorLayout(
                DType.uint32, [MIXED_STEP_SPLIT_SIZE], CPU()
            ),
        }
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=_BUFFER_SIZE,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            dtype=DType.bfloat16,
            device=device,
        )

        def forward(
            x: Tensor, row_offsets: Tensor, host_length: Tensor, *values: Tensor
        ) -> tuple[Tensor]:
            fields = dict(zip(cache_types, values, strict=True))
            collection = PagedCacheValues(
                kv_blocks=fields["kv_blocks"],
                cache_lengths=fields["cache_lengths"],
                lookup_table=fields["lookup_table"],
                max_prompt_length=fields["max_prompt_length"],
                max_cache_length=fields["max_cache_length"],
                page_stride=fields["page_stride"],
                attention_dispatch_metadata=fields[
                    "attention_dispatch_metadata"
                ],
                mla_num_partitions=fields["mla_num_partitions"],
                mixed_step_split=fields["mixed_step_split"],
            )
            prefill_plan = kimi_k3_mla_prefill_plan(
                kv_params=params,
                kv_collection=collection,
                input_row_offsets=row_offsets,
                buffer_size=_BUFFER_SIZE,
                buffer_length=host_length,
            )
            return (
                layer.forward_latent_ragged(
                    x, collection, row_offsets, prefill_plan, freqs_cis
                ).cast(DType.float32),
            )

        run = compilation.compile(forward)(
            TensorLayout(DType.bfloat16, [2, _HIDDEN], device),
            TensorLayout(DType.uint32, [2], device),
            TensorLayout(DType.int32, [1], CPU()),
            *cache_types.values(),
        )
        cache = build_paged_cache(params, [2], device, cache_lengths=[254])
        cache_fields = {
            name: value
            for name, value in vars(cache).items()
            if isinstance(value, Tensor)
        }
        x = Tensor(arr(2, _HIDDEN), dtype=DType.float32).cast(DType.bfloat16)
        offsets = Tensor(np.array([0, 2], dtype=np.uint32), device=device)
        host_length = Tensor(
            np.array([_BUFFER_SIZE], dtype=np.int32), device=CPU()
        )
        inputs = [x, offsets, host_length, *cache_fields.values()]
        buffers = [value.buffers[0] for value in inputs]
        harness = _PrefillArmHarness(
            run=run,
            device=device,
            host_length=host_length,
            kv_blocks=cache_fields["kv_blocks"],
            inputs=inputs,
            buffers=buffers,
        )
        # Real latent rows, like a cache mid-conversation: the stale-extent
        # control rewrites these blocks, and its tail rows must then hold a
        # decompression of data no live run of the rewritten cache produces.
        harness.rewrite_cache_blocks(arr)
        return harness


@pytest.mark.skipif(
    _FULL_SWEEP == (not accelerator_architecture_name().startswith("gfx9")),
    reason="runs in presubmit on NVIDIA and postsubmit on gfx9",
)
def test_prefill_arm_oversized_extent_capture_replays_eager(
    prefill_arm_harness: _PrefillArmHarness,
) -> None:
    """A capture frozen at an oversized extent replays the eager answer.

    The prefill arm's decompression launch reads the extent on the host,
    so a captured graph replays the extent it was captured at. The
    oversized direction is the safe one: the copy zero-fills the rows
    past the plan's real total and the attention never reads them, so a
    capture frozen oversized replays the eager answer exactly. That is
    also this control's limit -- with the extent held at one value, a
    frozen replay and a live one launch the same copy, so it can never
    catch a stale extent. Catching one is the stale-extent control's
    job, and only where the host extent bounds the replayed
    decompression window: gfx9, per that control's skip.
    """
    harness = prefill_arm_harness
    reference = harness.run(*harness.inputs)[0].to_numpy().copy()
    # An oversized extent costs the copy's slack rows and nothing else:
    # the rows past the real total are zero-filled and the
    # attention never reads them.
    harness.set_host_extent(2 * _BUFFER_SIZE)
    np.testing.assert_array_equal(
        harness.run(*harness.inputs)[0].to_numpy(), reference
    )

    captured = harness.run.engine_model.capture(1, *harness.buffers)
    harness.run.engine_model.replay(1, *harness.buffers)
    np.testing.assert_array_equal(
        captured[0].to_numpy(),
        reference,
        err_msg=(
            "a capture frozen at an oversized host extent replayed a"
            " different answer than eager"
        ),
    )


@pytest.mark.skipif(
    _FULL_SWEEP or not accelerator_architecture_name().startswith("gfx9"),
    reason=(
        "the stale extent does not bound this vendor's replayed"
        " decompression window: on B200 a capture frozen at a stale"
        " extent replayed the live answer bit for bit even with every"
        " cache block rewritten between the capture and its replay, so"
        " the sabotage cannot move the answer here and no host-extent"
        " control can fail"
    ),
)
def test_prefill_arm_stale_extent_capture_corrupts_replay(
    prefill_arm_harness: _PrefillArmHarness,
) -> None:
    """A capture frozen at a stale extent corrupts the replayed answer.

    The captured graph replays the copy launch it recorded, which covers
    only the frozen extent's rows, while the attention still reads the
    plan's full real window -- so the rows in between hold whatever the
    working buffer's tail held, and the replayed step silently attends
    it. The cache blocks are rewritten between the stale capture and its
    replay to make that corruption deterministic: the replay's copy
    re-decompresses rows [0, stale extent) from the rewritten cache,
    while rows [stale extent, real total) keep the pre-rewrite
    decompression -- a stale-but-different answer whatever the vendor's
    buffer reuse leaves in a tail. The stale capture's replayed answer
    must then diverge from the live answer by more than half the live
    answer's own scale, and a capture taken at the live extent over the
    same rewritten cache must replay the live answer bit for bit, which
    grades the corruption to the stale extent itself.

    Runs on gfx9 alone: the host extent must bound the replayed
    decompression window for the sabotage to bite, which holds on gfx9.
    On B200 a capture frozen at the stale extent replays the live
    answer bit for bit even over a rewritten cache, so the undersized
    direction does not bind NVIDIA's replayed window and no tail
    rewrite can move the answer -- NVIDIA is unguarded by this control
    against a capture-time stale extent.
    """
    harness = prefill_arm_harness
    rng = np.random.default_rng(2)

    def fresh(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    # Capture at the stale extent while the cache still holds its original
    # blocks, so the capture's own execution seeds its working buffer's
    # tail with the pre-rewrite decompression.
    harness.set_host_extent(_STALE_EXTENT)
    stale = harness.run.engine_model.capture(2, *harness.buffers)
    # Rewrite every block: the replay's copy re-decompresses only rows
    # [0, stale extent) of this data, so the tail rows the frozen launch
    # skips hold a decompression no run of the rewritten cache produces.
    harness.rewrite_cache_blocks(fresh)
    harness.run.engine_model.replay(2, *harness.buffers)
    undersized = stale[0].to_numpy().copy()

    harness.set_host_extent(_BUFFER_SIZE)
    live = harness.run(*harness.inputs)[0].to_numpy().copy()
    captured = harness.run.engine_model.capture(3, *harness.buffers)
    harness.run.engine_model.replay(3, *harness.buffers)
    np.testing.assert_array_equal(
        captured[0].to_numpy(),
        live,
        err_msg=(
            "a capture frozen at the live host extent replayed a different"
            " answer than eager over the same rewritten cache"
        ),
    )

    assert not np.array_equal(undersized, live), (
        "a capture frozen at a stale extent one page short of the real"
        " total still matched the eager answer bit for bit"
    )
    scale = float(np.max(np.abs(live)))
    margin = float(np.max(np.abs(undersized - live)))
    print(
        f"stale-extent replay diverged from the live answer by {margin}"
        f" ({margin / scale:.0%} of the live answer's own scale)"
    )
    # Run alone, the rows past the stale extent can read never-written
    # memory and come back non-finite, which is as corrupt as a large finite
    # move but compares false against any bound.
    assert not np.isfinite(margin) or margin > scale / 2, (
        "a capture frozen at a stale extent one page short of the real"
        f" total barely moved the answer (max |diff| {margin} against a"
        f" live answer scaled at {scale}); the rows between the stale"
        " extent and the real window must move it by more than half its"
        " own scale"
    )
