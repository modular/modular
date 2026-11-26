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
"""Which K/V path a traced Inkling 2 attention block takes, tracing only.

With ``INKLING2_FUSED_KV_PROLOGUE`` unset the block must trace exactly as it
does without the fused op existing; with it set, a local layer's K/V path is
the one fused op, whatever the batch's shape, and a global layer is untouched.

The trace runs on the host, where ``flash_attention_ragged`` refuses the
block's token sink, so every trace here replaces it with the query itself:
the K/V path under test comes before it and is traced in full.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.inkling2.layers import attention as attn_mod
from max.pipelines.architectures.inkling2.layers.attention import (
    Inkling2Attention,
)
from max.pipelines.architectures.inkling2.layers.short_convolution import (
    LayerConvState,
    SiteState,
)
from max.pipelines.architectures.inkling2.model_config import (
    Inkling2TextConfig,
)
from max.tensor import Tensor, default_device, default_dtype

_FLAG = "INKLING2_FUSED_KV_PROLOGUE"


@pytest.fixture(autouse=True)
def _host_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        attn_mod,
        "flash_attention_ragged",
        lambda kv_params, *, input, **kwargs: input,
    )


HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 128
KERNEL = 4
RING_LEN = KERNEL - 1
SLOTS = 8

_TEXT = Inkling2TextConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=2,
    rms_norm_eps=1e-6,
    vocab_size=512,
    unpadded_vocab_size=512,
    num_attention_heads=HEADS,
    head_dim=HEAD_DIM,
    num_key_value_heads=KV_HEADS,
    swa_num_attention_heads=HEADS,
    swa_head_dim=HEAD_DIM,
    swa_num_key_value_heads=KV_HEADS,
    sliding_window_size=64,
    d_rel=16,
    rel_extent=64,
    log_scaling_alpha=0.1,
    log_scaling_n_floor=128,
    qk_l2_clip_threshold=22.627416997969522,
    rel_l2_clip_threshold=8.0,
    sconv_kernel_size=KERNEL,
    local_layer_ids=(0,),
    n_layers_per_attn_res_block=2,
    attn_res_normalize_values=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    intermediate_size=128,
    route_scale=1.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    final_logit_softcapping=30.0,
    model_max_length=4096,
)


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )


class _Block(Module[..., Tensor]):
    """One attention block over graph inputs, as the decoder calls it."""

    def __init__(self, layer_idx: int) -> None:
        self.kv_params = _kv_params()
        self.attention = Inkling2Attention(
            text_config=_TEXT,
            layer_idx=layer_idx,
            kv_params=self.kv_params,
            num_devices=1,
        )

    def forward(self, *inputs: Any) -> Tensor:
        x, offsets, positions, tau, k_ring, v_ring, k_rows, v_rows = inputs[:8]
        (kv,) = self.kv_params.unflatten_kv_inputs(iter(inputs[8:]))
        kv_collection = PagedCacheValues(
            **{f.name: getattr(kv, f.name) for f in fields(kv)}
        )

        def site(ring: Tensor, rows: Tensor, row: int) -> SiteState:
            return SiteState(
                ring=ring,
                rows=rows,
                layer_row=F.constant(row, DType.uint32, device=CPU()),
            )

        k_site = site(k_ring, k_rows, 0)
        v_site = site(v_ring, v_rows, 0)
        return self.attention(
            x,
            x,
            x,
            kv_collection=kv_collection,
            input_row_offsets=offsets,
            positions=positions,
            tau=tau,
            conv=LayerConvState(k_site, v_site, k_site, v_site),
            cache_layer_idx=F.constant(0, DType.uint32, device=CPU()),
        )


def _trace(layer_idx: int, rows: int | str = "rows") -> str:
    cpu = DeviceRef.CPU()
    ring = BufferType(
        DType.float32, [SLOTS, RING_LEN, KV_HEADS * HEAD_DIM], cpu
    )
    table = TensorType(DType.uint32, [1, "batch"], device=cpu)
    with default_device(CPU()), default_dtype(DType.bfloat16):
        block = _Block(layer_idx)
        types = [
            TensorType(DType.bfloat16, [rows, HIDDEN], device=cpu),
            TensorType(DType.uint32, ["offsets"], device=cpu),
            TensorType(DType.uint32, [rows], device=cpu),
            TensorType(DType.float32, [rows], device=cpu),
            ring,
            ring,
            table,
            table,
            *block.kv_params.flattened_kv_inputs(),
        ]
        return str(block.trace(*types))


def _count(mlir: str, op: str) -> int:
    return mlir.count(f'"{op}"')


_FUSED = "inkling2.kv_prologue"
_CONV = "conv.short_conv_ring"
_STORE = "kv_cache.store.paged.ragged"


def test_flag_off_traces_as_without_the_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off, the local block's graph is the composed path, node for node.

    The reference trace replaces the gate with one that declines without
    looking at any tensor, which is the block as it was before the op.
    """
    monkeypatch.delenv(_FLAG, raising=False)
    off = _trace(0)
    monkeypatch.setattr(
        attn_mod, "can_fuse_kv_prologue", lambda *args, **kwargs: False
    )
    assert _trace(0) == off
    assert _count(off, _FUSED) == 0
    assert _count(off, _CONV) == 2
    assert _count(off, _STORE) == 2


@pytest.mark.parametrize("rows", ["rows", 1, 7])
def test_flag_on_fuses_local_layers(
    monkeypatch: pytest.MonkeyPatch, rows: int | str
) -> None:
    """On, a local layer fuses for symbolic, decode and multi-token rows."""
    monkeypatch.setenv(_FLAG, "1")
    on = _trace(0, rows)
    assert _count(on, _FUSED) == 1
    assert _count(on, _CONV) == 0
    assert _count(on, _STORE) == 0


def test_flag_on_leaves_global_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A global layer's dynamic conv is not the op's, so it stays composed."""
    monkeypatch.setenv(_FLAG, "1")
    on = _trace(1)
    monkeypatch.delenv(_FLAG)
    assert on == _trace(1)
    assert _count(on, _FUSED) == 0
    assert _count(on, _STORE) == 2
