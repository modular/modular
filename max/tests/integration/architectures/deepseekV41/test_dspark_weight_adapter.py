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
"""Tests that converted DSpark checkpoint tensors strict-load into the drafter.

A small-width checkpoint with the real ``mtp.*`` names, stored dtypes and
shape relations, plus one trunk layer, is converted and loaded
``strict=True`` into a :class:`DeepseekV41DSparkDrafter`. Values are
placeholders.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41.weight_adapters import (
    convert_dspark_safetensor_state_dict,
    with_dtype,
)
from transformers.configuration_utils import PretrainedConfig

HIDDEN = 128
# ``wo_a`` loads as MXFP8 with tile-interleaved scales, which need whole
# 128-row, 128-column groups: HEADS * HEAD_DIM / O_GROUPS and O_LORA.
HEADS = 4
HEAD_DIM = 64
ROPE_DIM = 32
Q_LORA = 64
O_GROUPS = 2
O_LORA = 128
INTER = 128
EXPERTS = 4
TOPK = 2
VOCAB = 64
MARKOV = 32
STAGES = 3
TRUNK_LAYERS = 1
TAPS = [0]


def _config() -> DeepseekV41Config:
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=MagicMock(num_draft_tokens=0),
        devices=[DeviceRef.CPU()],
        max_seq_len=64,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=TRUNK_LAYERS,
        rms_norm_eps=1e-6,
        sliding_window=8,
        n_routed_experts=8,
        moe_intermediate_size=INTER,
        num_experts_per_tok=2,
        num_dspark_layers=STAGES,
        dspark_target_layer_ids=TAPS,
        dspark_block_size=5,
        dspark_markov_rank=MARKOV,
        dspark_n_routed_experts=EXPERTS,
        dspark_num_experts_per_tok=TOPK,
        compress_ratios=[1] * TRUNK_LAYERS + [0] * STAGES,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
    )


def _hf_config() -> PretrainedConfig:
    return PretrainedConfig(
        text_config=PretrainedConfig(
            num_hidden_layers=TRUNK_LAYERS,
            n_routed_experts=8,
            dspark_n_routed_experts=EXPERTS,
            num_nextn_predict_layers=STAGES,
        )
    )


def _tensor(name: str, dtype: DType, shape: tuple[int, ...]) -> WeightData:
    """Returns placeholder bytes at the checkpoint's stored dtype and shape."""
    if dtype == DType.bfloat16:
        return with_dtype(np.full(shape, 0x3F80, np.uint16), dtype, name)
    if dtype == DType.float32:
        return with_dtype(np.ones(shape, np.float32), dtype, name)
    if dtype == DType.int8:
        return with_dtype(np.zeros(shape, np.int8), dtype, name)
    # float8_e4m3fn values (0x38 == 1.0) and float8_e8m0fnu scales (127).
    fill = 0x38 if dtype == DType.float8_e4m3fn else 127
    return with_dtype(np.full(shape, fill, np.uint8), dtype, name)


def _fp8_linear(base: str, out_dim: int, in_dim: int) -> dict[str, WeightData]:
    """Returns an MXFP8 linear as stored: e4m3 weight, 32x32-block scales."""
    return {
        f"{base}.weight": _tensor(
            f"{base}.weight", DType.float8_e4m3fn, (out_dim, in_dim)
        ),
        f"{base}.scale": _tensor(
            f"{base}.scale", DType.float8_e8m0fnu, (out_dim // 32, in_dim // 32)
        ),
    }


def mtp_checkpoint() -> dict[str, WeightData]:
    """Returns the checkpoint's ``mtp.*`` tensors at this module's widths."""
    sd: dict[str, WeightData] = {}

    def add(name: str, dtype: DType, shape: tuple[int, ...]) -> None:
        sd[name] = _tensor(name, dtype, shape)

    mix = (2 + 4) * 4
    for i in range(STAGES):
        p = f"mtp.{i}"
        add(f"{p}.attn.attn_sink", DType.float32, (HEADS,))
        add(f"{p}.attn.kv_norm.weight", DType.bfloat16, (HEAD_DIM,))
        add(f"{p}.attn.q_norm.weight", DType.bfloat16, (Q_LORA,))
        sd |= _fp8_linear(f"{p}.attn.wkv", HEAD_DIM, HIDDEN)
        sd |= _fp8_linear(
            f"{p}.attn.wo_a", O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS
        )
        sd |= _fp8_linear(f"{p}.attn.wo_b", HIDDEN, O_GROUPS * O_LORA)
        sd |= _fp8_linear(f"{p}.attn.wq_a", Q_LORA, HIDDEN)
        sd |= _fp8_linear(f"{p}.attn.wq_b", HEADS * HEAD_DIM, Q_LORA)
        add(f"{p}.attn_norm.weight", DType.bfloat16, (HIDDEN,))
        add(f"{p}.ffn_norm.weight", DType.bfloat16, (HIDDEN,))
        for e in range(EXPERTS):
            x = f"{p}.ffn.experts.{e}"
            for proj, n, k in (
                ("w1", INTER, HIDDEN),
                ("w3", INTER, HIDDEN),
                ("w2", HIDDEN, INTER),
            ):
                add(f"{x}.{proj}.weight", DType.int8, (n, k // 2))
                add(f"{x}.{proj}.scale", DType.float8_e8m0fnu, (n, k // 32))
        add(f"{p}.ffn.gate.bias", DType.float32, (EXPERTS,))
        add(f"{p}.ffn.gate.bias_vl", DType.float32, (EXPERTS,))
        add(f"{p}.ffn.gate.weight", DType.bfloat16, (EXPERTS, HIDDEN))
        for proj, n, k in (
            ("w1", INTER, HIDDEN),
            ("w3", INTER, HIDDEN),
            ("w2", HIDDEN, INTER),
        ):
            sd |= _fp8_linear(f"{p}.ffn.shared_experts.{proj}", n, k)
        for site in ("attn", "ffn"):
            add(f"{p}.hc_{site}_base", DType.float32, (mix,))
            add(f"{p}.hc_{site}_fn", DType.float32, (mix, 4 * HIDDEN))
            add(f"{p}.hc_{site}_scale", DType.float32, (3,))
    add("mtp.0.main_norm.weight", DType.bfloat16, (HIDDEN,))
    sd |= _fp8_linear("mtp.0.main_proj", HIDDEN, HIDDEN * len(TAPS))
    last = f"mtp.{STAGES - 1}"
    add(f"{last}.norm.weight", DType.bfloat16, (HIDDEN,))
    add(f"{last}.markov_head.embed.weight", DType.bfloat16, (VOCAB, MARKOV))
    add(f"{last}.markov_head.head.weight", DType.bfloat16, (VOCAB, MARKOV))
    add(
        f"{last}.confidence_head.proj.weight",
        DType.bfloat16,
        (1, HIDDEN + MARKOV),
    )
    return sd


def _trunk_attention() -> dict[str, WeightData]:
    """Returns one trunk layer's attention projections plus the embedding."""
    attn = "layers.0.attn"
    sd = _fp8_linear(f"{attn}.wq_a", Q_LORA, HIDDEN)
    sd |= _fp8_linear(f"{attn}.wkv", HEAD_DIM, HIDDEN)
    sd |= _fp8_linear(f"{attn}.wq_b", HEADS * HEAD_DIM, Q_LORA)
    sd[f"{attn}.q_norm.weight"] = _tensor(
        f"{attn}.q_norm.weight", DType.bfloat16, (Q_LORA,)
    )
    sd["embed.weight"] = _tensor(
        "embed.weight", DType.bfloat16, (VOCAB, HIDDEN)
    )
    return sd


def _weights(sd: dict[str, WeightData]) -> dict[str, Any]:
    """Wraps each tensor in a stand-in whose ``data()`` returns it."""
    out: dict[str, Any] = {}
    for name, data in sd.items():
        mock = MagicMock()
        mock.data.return_value = data
        out[name] = mock
    return out


def _convert() -> dict[str, WeightData]:
    return convert_dspark_safetensor_state_dict(
        _weights(mtp_checkpoint() | _trunk_attention()),
        _hf_config(),
    )


def _drafter_part(converted: dict[str, WeightData]) -> dict[str, WeightData]:
    return {
        name.removeprefix("dspark."): data
        for name, data in converted.items()
        if name.startswith("dspark.")
    }


def test_dspark_tensors_strict_load_into_the_drafter() -> None:
    """Checks converted ``dspark.*`` tensors match the drafter's weights."""
    drafter = DeepseekV41DSparkDrafter(_config(), DeviceRef.CPU())
    part = _drafter_part(_convert())
    drafter.load_state_dict(part, weight_alignment=1, strict=True)

    declared = drafter.raw_state_dict()
    assert set(declared) == set(part)
    # `load_state_dict` tolerates a shape mismatch between two uint8
    # tensors (packed encodings), so pin those shapes explicitly.
    for name, weight in declared.items():
        data = part[name]
        assert data.dtype == weight.dtype, name
        assert tuple(int(d) for d in data.shape) == tuple(
            int(d) for d in weight.shape
        ), name


def test_dspark_wo_a_loads_as_mxfp8() -> None:
    """DSpark ``wo_a`` keeps its e4m3 weight and a tile-interleaved scale."""
    converted = _convert()
    for stage in range(STAGES):
        base = f"dspark.stages.{stage}.attn.wo_a"
        assert converted[f"{base}.weight"].dtype == DType.float8_e4m3fn
        scale = converted[f"{base}.weight_scale"]
        assert scale.dtype == DType.float8_e8m0fnu
        assert tuple(int(d) for d in scale.shape) == (
            O_GROUPS * O_LORA,
            HEADS * HEAD_DIM // O_GROUPS // 32,
        )
