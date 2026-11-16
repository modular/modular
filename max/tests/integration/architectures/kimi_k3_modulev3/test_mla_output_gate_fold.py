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
"""The MLA layer's folded output gate against its unfused chain.

The layer folds ``attn * sigmoid(gate_pre)`` into the MLA kernel when the
accelerator's kernel has the fold. Each case below compiles the same weighted
latent MLA step twice -- once with the gate passed to the op and once with the
layer's own multiply -- runs both over the same weights, cache and input, and
requires their outputs to match bit for bit.

The decode cases (4 rows, and 32 rows as four requests of eight) use K3's FP8
latent cache, which is the arm that routes a multi-token batch to the decode
branch, where the gate rides the ``w_uv`` projection's store through the
sliced, row-strided view of the fused input projection. The prefill case uses
a BF16 cache and a six-token request, which the kernel routes to its prefill
branch; there it applies the same multiply in one pass of its own.

The mixed-step case writes a real split record -- decode-width rows in one
view, a prefill row in the other -- the way the KV cache manager does, so the
op runs each view as its own self-call and each call must read the gate at
its own ``first_token`` row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from _paged_cache_test_utils import build_paged_cache
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import (
    MIXED_STEP_SPLIT_SIZE,
    MLAAttnKey,
    MLAKVCacheParams,
    padded_lut_cols,
)
from max.nn.kv_cache.input_types import _mixed_step_decode_view
from max.pipelines.architectures.kimi_k3_modulev3.layers import mla_attention
from max.pipelines.architectures.kimi_k3_modulev3.layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from max.pipelines.kv_cache.paged_kv_cache._mixed_step_split import (
    mixed_step_decode_width,
    mixed_step_split_record,
)
from max.support.math import ceildiv
from max.tensor import Tensor, compilation, default_device
from max.tensor.realization_context import ensure_context
from max.tensor.sharding import BufferLayout, TensorLayout

_HIDDEN = 256
_NUM_HEADS = 8
_QK_NOPE_HEAD_DIM = 64
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 64
_Q_LORA_RANK = 128
# The fused MLA prefill/decode kernel only instantiates for a 576-wide latent
# cache (512 + 64), which is K3's own shape.
_KV_LORA_RANK = 512
_PAGE_SIZE = 128
_BUFFER_SIZE = 256


def _params(device: Accelerator, cache_dtype: DType) -> MLAKVCacheParams:
    return MLAKVCacheParams(
        dtype=cache_dtype,
        head_dim=_KV_LORA_RANK + _QK_ROPE_HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.from_device(device)],
        page_size=_PAGE_SIZE,
        num_q_heads=_NUM_HEADS,
    )


def _weights(seed: int) -> dict[str, np.ndarray]:
    """Draws the layer's six weights at the layer's small-Gaussian scale."""
    rng = np.random.default_rng(seed)

    def draw(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    return {
        "qkv_a_g": draw(
            _Q_LORA_RANK
            + _KV_LORA_RANK
            + _QK_ROPE_HEAD_DIM
            + _NUM_HEADS * _V_HEAD_DIM,
            _HIDDEN,
        ),
        "q_a_norm": np.ones(_Q_LORA_RANK, dtype=np.float32),
        "q_b": draw(
            _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM), _Q_LORA_RANK
        ),
        "kv_a_norm": np.ones(_KV_LORA_RANK, dtype=np.float32),
        "kv_b": draw(
            _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK
        ),
        "o": draw(_HIDDEN, _NUM_HEADS * _V_HEAD_DIM),
    }


def _weight_layouts(device: Device) -> dict[str, TensorLayout]:
    """The layouts of the weights `_weights` draws, in its order."""
    return {
        "qkv_a_g": TensorLayout(
            DType.bfloat16,
            [
                _Q_LORA_RANK
                + _KV_LORA_RANK
                + _QK_ROPE_HEAD_DIM
                + _NUM_HEADS * _V_HEAD_DIM,
                _HIDDEN,
            ],
            device,
        ),
        "q_a_norm": TensorLayout(DType.bfloat16, [_Q_LORA_RANK], device),
        "q_b": TensorLayout(
            DType.bfloat16,
            [
                _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
                _Q_LORA_RANK,
            ],
            device,
        ),
        "kv_a_norm": TensorLayout(DType.bfloat16, [_KV_LORA_RANK], device),
        "kv_b": TensorLayout(
            DType.bfloat16,
            [_NUM_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM), _KV_LORA_RANK],
            device,
        ),
        "o": TensorLayout(
            DType.bfloat16, [_HIDDEN, _NUM_HEADS * _V_HEAD_DIM], device
        ),
    }


def _cache_layouts(
    params: MLAKVCacheParams, device: Accelerator, seq_lens: Sequence[int]
) -> dict[str, BufferLayout | TensorLayout]:
    """The layouts of a freshly built cache for this step, in field order.

    ``seq_lens`` are the requests' total lengths -- their query widths, since
    every case runs over a cache that starts empty.
    """
    pages = sum(ceildiv(length, params.page_size) for length in seq_lens)
    spec = MLAAttnKey.dispatch_metadata_spec()
    return {
        "kv_blocks": BufferLayout(
            params.dtype, [pages + 1, *params.shape_per_block], device
        ),
        "cache_lengths": TensorLayout(DType.uint32, [len(seq_lens)], device),
        "lookup_table": TensorLayout(
            DType.uint32,
            [
                len(seq_lens),
                padded_lut_cols(ceildiv(max(seq_lens), params.page_size)),
            ],
            device,
        ),
        "max_prompt_length": TensorLayout(DType.uint32, [1], CPU()),
        "max_cache_length": TensorLayout(DType.uint32, [1], CPU()),
        "page_stride": TensorLayout(DType.int64, [1], CPU()),
        "attention_dispatch_metadata": TensorLayout(
            spec.dtype, list(spec.shape), device
        ),
        "mla_num_partitions": TensorLayout(DType.int64, [2], CPU()),
        "mixed_step_split": TensorLayout(
            DType.uint32, [MIXED_STEP_SPLIT_SIZE], CPU()
        ),
    }


def _compile_step(
    params: MLAKVCacheParams,
    device: Accelerator,
    *,
    widths: Sequence[int],
    fold: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> compilation.CompiledCallable[..., Any]:
    """Compiles one ragged latent step with or without the folded gate.

    The weights are graph inputs, so nothing is staged on the device before
    the compile. The patched decision records that the layer consulted it, so
    an arm cannot silently run the other arm's chain.
    """
    consulted: list[bool] = []

    def decide() -> bool:
        consulted.append(fold)
        return fold

    monkeypatch.setattr(mla_attention, "_folds_output_gate", decide)
    weight_types = _weight_layouts(device)
    cache_types = _cache_layouts(params, device, widths)

    def forward(
        w_qkv_a_g: Tensor,
        w_q_a_norm: Tensor,
        w_q_b: Tensor,
        w_kv_a_norm: Tensor,
        w_kv_b: Tensor,
        w_o: Tensor,
        x: Tensor,
        row_offsets: Tensor,
        buffer_length: Tensor,
        *values: Tensor,
    ) -> tuple[Tensor]:
        # The layer's own initializers draw from `ops.random`; the weights it
        # would draw are graph inputs here.
        ops.random.set_seed(0)
        freqs_cis = kimi_k3_nope_freqs_cis(
            max_seq_len=_BUFFER_SIZE,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            dtype=DType.bfloat16,
            device=device,
        )
        layer = KimiK3MLAAttention(
            hidden_size=_HIDDEN,
            num_heads=_NUM_HEADS,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            v_head_dim=_V_HEAD_DIM,
            q_lora_rank=_Q_LORA_RANK,
            kv_lora_rank=_KV_LORA_RANK,
            kv_params=params,
            layer_idx=0,
        )
        layer.qkv_a_g_proj.weight = w_qkv_a_g
        layer.q_a_layernorm.weight = w_q_a_norm
        layer.q_b_proj.weight = w_q_b
        layer.kv_a_layernorm.weight = w_kv_a_norm
        layer.kv_b_proj.weight = w_kv_b
        layer.o_proj.weight = w_o
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
        prefill_plan = kimi_k3_mla_prefill_plan(
            kv_params=params,
            kv_collection=collection,
            input_row_offsets=row_offsets,
            buffer_size=_PAGE_SIZE,
            buffer_length=buffer_length,
        )
        return (
            layer.forward_latent_ragged(
                x, collection, row_offsets, prefill_plan, freqs_cis
            ).cast(DType.float32),
        )

    run = compilation.compile(forward)(
        *weight_types.values(),
        TensorLayout(DType.bfloat16, [sum(widths), _HIDDEN], device),
        TensorLayout(DType.uint32, [len(widths) + 1], device),
        TensorLayout(DType.int32, [1], CPU()),
        *cache_types.values(),
    )
    assert consulted, "the layer never consulted the fold decision"
    return run


def _split_cache_fields(
    params: MLAKVCacheParams, device: Accelerator, widths: Sequence[int]
) -> dict[str, Tensor]:
    """The cache fields that turn a mixed step's split on, as the manager
    writes them.

    The record itself comes from ``mixed_step_split_record``, and the decode
    view's row count, query width and split-K count land in the dispatch
    metadata and ``mla_num_partitions`` -- the fields the op reads before it
    takes a split at all. The cache starts empty, so each row's valid length
    is its own query width.
    """
    record = mixed_step_split_record(
        widths, widths, mixed_step_decode_width(num_draft_tokens=0)
    )
    assert record[0] != 0, (
        "the batch must split: decode-width rows beside a prefill row"
    )
    key = params.resolve_attn_key(len(widths), max(widths), max(widths))
    assert isinstance(key, MLAAttnKey)
    view = _mixed_step_decode_view(record, len(widths))
    assert view is not None, "the record's decode view is missing"
    decode_key = params.resolve_attn_key(*view)
    assert isinstance(decode_key, MLAAttnKey)
    key = replace(
        key,
        decode_batch_size=decode_key.batch_size,
        decode_max_prompt_length=decode_key.max_prompt_length,
        decode_num_partitions=decode_key.num_partitions,
    )
    with ensure_context():
        return {
            "attention_dispatch_metadata": Tensor(
                storage=key.pack_into_buffer(device, max(widths))
            ),
            "mla_num_partitions": Tensor(
                storage=Buffer.from_numpy(
                    np.array(
                        [key.num_partitions, key.decode_num_partitions],
                        np.int64,
                    )
                )
            ),
            "mixed_step_split": Tensor(
                storage=Buffer.from_numpy(record.astype(np.uint32))
            ),
        }


def _compare_arms(
    device: Accelerator,
    *,
    widths: Sequence[int],
    cache_dtype: DType,
    monkeypatch: pytest.MonkeyPatch,
    split: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the folded and unfused steps over one layer and returns both.

    ``split`` fills the cache's mixed-step record the way the KV cache
    manager does, so the op runs the batch as one self-call per view.
    """
    params = _params(device, cache_dtype)
    rng = np.random.default_rng(0)
    drawn = _weights(seed=0)
    x_np = (rng.standard_normal((sum(widths), _HIDDEN)) * 0.02).astype(
        np.float32
    )
    row_offsets_np = np.concatenate(([0], np.cumsum(widths))).astype(np.uint32)
    with default_device(device):
        runs = [
            _compile_step(
                params,
                device,
                widths=widths,
                fold=fold,
                monkeypatch=monkeypatch,
            )
            for fold in (True, False)
        ]
        weights = [
            Tensor(value, dtype=DType.float32).cast(DType.bfloat16)
            for value in drawn.values()
        ]
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        row_offsets = Tensor(row_offsets_np, dtype=DType.uint32)
        buffer_length = Tensor(
            np.array([_BUFFER_SIZE], dtype=np.int32), device=CPU()
        )
        outs = []
        for run in runs:
            # A fresh cache per arm: the step writes its tokens' latent into
            # the cache, so two arms sharing one would not read the same keys.
            cache = build_paged_cache(params, widths, device)
            fields = {
                name: value
                for name, value in vars(cache).items()
                if isinstance(value, (Tensor, Buffer))
            }
            assert list(fields) == list(
                _cache_layouts(params, device, widths)
            ), "the cache's fields moved; update `_cache_layouts` with it"
            if split:
                fields.update(_split_cache_fields(params, device, widths))
            out = run(
                *weights, x, row_offsets, buffer_length, *fields.values()
            )[0]
            outs.append(out.to_numpy())
    return outs[0], outs[1]


@pytest.mark.parametrize(
    "batch,tokens",
    [(4, 1), (4, 8)],
    ids=["decode-4-rows", "verify-32-rows"],
)
def test_the_folded_gate_matches_the_unfused_chain_on_decode(
    batch: int, tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decode branch's gated projection store is bit-exact.

    An FP8 latent cache keeps an eight-token request on the decode branch,
    so both of K3's serve shapes exercise ``has_output_gate``, the gate's
    sliced row stride and the fold itself.
    """
    device = Accelerator()
    fused, unfused = _compare_arms(
        device,
        widths=[tokens] * batch,
        cache_dtype=DType.float8_e4m3fn,
        monkeypatch=monkeypatch,
    )
    np.testing.assert_array_equal(fused, unfused)


def test_the_folded_gate_matches_the_unfused_chain_on_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefill branch's own gated pass is bit-exact.

    A BF16 cache routes any multi-token request to the prefill branch, which
    has no ``w_uv`` projection store to fold the gate into and applies it as
    one pass instead -- the same operation the layer would have run.
    """
    device = Accelerator()
    fused, unfused = _compare_arms(
        device,
        widths=[6],
        cache_dtype=DType.bfloat16,
        monkeypatch=monkeypatch,
    )
    np.testing.assert_array_equal(fused, unfused)


def test_the_folded_gate_matches_the_unfused_chain_on_a_mixed_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split's self-calls each read the gate at their own first row.

    A continuous-batching step mixes decode-width rows with a prefill row,
    so the record cuts the batch into a decode view and a prefill view and
    the op runs each view as its own self-call. The gate is per token of
    the whole batch, so a view's call must start reading it at the view's
    own first token: the prefill view's gate rows sit past the decode
    rows'. K3 serves exactly this shape on MI355X -- an FP8 latent cache
    at a fold-eligible head count -- where the split runs with a live
    gate.
    """
    device = Accelerator()
    fused, unfused = _compare_arms(
        device,
        widths=[1, 1, 16],
        cache_dtype=DType.float8_e4m3fn,
        monkeypatch=monkeypatch,
        split=True,
    )
    np.testing.assert_array_equal(fused, unfused)
