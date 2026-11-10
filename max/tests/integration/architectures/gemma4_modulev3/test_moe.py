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
"""Gemma4 ModuleV3 MoE checkpoint keys and TP2 expert placement, traced lazily."""

from __future__ import annotations

import re

import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape
from max.graph.weights import WeightData
from max.nn.activation import GatedActivation
from max.nn.common_layers.quant_tensor import NVFP4Tensor
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.outputs import ReturnLogits
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import Gemma4TextConfig
from max.pipelines.architectures.gemma4_modulev3.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4_modulev3.layers.moe import (
    Gemma4MoE,
    Gemma4StackedExperts,
)
from max.pipelines.architectures.gemma4_modulev3.layers.transformer_block import (
    Gemma4TransformerBlock,
)
from max.pipelines.architectures.gemma4_modulev3.model_config import (
    Gemma4Config,
)
from max.pipelines.architectures.gemma4_modulev3.modules import Gemma4Decoder
from max.pipelines.lib.registration_runtime import ModuleRuntime
from max.pipelines.weights.quant import parse_quant_config
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh, Replicated
from transformers import PretrainedConfig

# ``_MOE_DIM`` splits over TP2 into whole 16-wide NVFP4 scale blocks.
_HIDDEN = 64
_INTERMEDIATE = 64
_MOE_DIM = 32
_NUM_EXPERTS = 8
_TOP_K = 2
_VOCAB = 256
_N_LAYERS = 2
_LAYER_TYPES = ["sliding_attention", "full_attention"]
_N_HEADS = 2
_HEAD_DIM = 32
_N_KV_HEADS = 2
_N_GLOBAL_KV_HEADS = 2
_GLOBAL_HEAD_DIM = 64
_EPS = 1e-6

_GELU_TANH = GatedActivation("gelu_tanh")


# ---------------------------------------------------------------------------
# Configs and models
# ---------------------------------------------------------------------------


def _kv_params(
    devices: list[DeviceRef], n_kv: int, head_dim: int, n: int
) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        num_layers=n,
        devices=devices,
        page_size=128,
    )


def _model_config(
    devices: list[DeviceRef], quant_config: QuantConfig | None
) -> Gemma4Config:
    text_config = Gemma4TextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_N_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=1024,
        max_seq_len=1024,
        rms_norm_eps=_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=_HEAD_DIM,
        sliding_window=512,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=-1,
        dtype=DType.bfloat16,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=_kv_params(devices, _N_KV_HEADS, _HEAD_DIM, _N_LAYERS),
        num_global_key_value_heads=_N_GLOBAL_KV_HEADS,
        global_head_dim=_GLOBAL_HEAD_DIM,
        attention_k_eq_v=True,
        enable_moe_block=True,
        num_experts=_NUM_EXPERTS,
        top_k_experts=_TOP_K,
        moe_intermediate_size=_MOE_DIM,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=1.0
        ),
        global_rope_theta=10000.0,
        sliding_window_rope_theta=10000.0,
        layer_types=_LAYER_TYPES,
        quant_config=quant_config,
    )
    text_config.return_logits = ReturnLogits.LAST_TOKEN
    return Gemma4Config(
        devices=devices,
        # uint8 = packed float4_e2m1fnx2, as the pipeline resolves it.
        dtype=DType.uint8 if quant_config is not None else DType.bfloat16,
        unquantized_dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {
                "sliding_attention": _kv_params(
                    devices, _N_KV_HEADS, _HEAD_DIM, 1
                ),
                "full_attention": _kv_params(
                    devices, _N_GLOBAL_KV_HEADS, _GLOBAL_HEAD_DIM, 1
                ),
            }
        ),
        text_config=text_config,
        vision_config=None,
        image_token_index=0,
        tie_word_embeddings=True,
    )


def _nvfp4_quant_config() -> QuantConfig:
    """The parsed modelopt config of ``nvidia/Gemma-4-26B-A4B-NVFP4``."""
    prefix = "model.language_model.layers"
    hf_config = PretrainedConfig.from_dict(
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                        "weights": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                        "targets": ["Linear"],
                    }
                },
                "ignore": ["lm_head", "model.embed_vision*"]
                + [
                    f"{prefix}.{i}.{module}*"
                    for i in range(_N_LAYERS)
                    for module in ("mlp", "router", "self_attn")
                ],
            }
        }
    )
    hf_config.text_config = PretrainedConfig.from_dict(
        {"num_hidden_layers": _N_LAYERS}
    )
    quant_config = parse_quant_config(
        hf_config,
        {},
        DType.uint8,
        state_dict_name_prefix="model.language_model.",
        ignored_modules_prefix="model.language_model.",
    )
    assert quant_config is not None and quant_config.is_nvfp4
    return quant_config


def _model_on_devices(config: Gemma4Config, n: int) -> Gemma4TextModel:
    """Builds the text model lazily on an ``n``-device CPU mesh."""
    mesh = DeviceMesh((CPU(),) * n, (n,), ("tp",))
    with (
        F.lazy(),
        default_dtype(config.unquantized_dtype),
        default_device(mesh),
    ):
        return Gemma4TextModel(config, mesh)


def _moe(model: Gemma4TextModel, layer: int) -> Gemma4MoE:
    block = model.layers[layer]
    assert isinstance(block, Gemma4TransformerBlock)
    assert isinstance(block.moe_block, Gemma4MoE)
    return block.moe_block


# ---------------------------------------------------------------------------
# Checkpoint keys
# ---------------------------------------------------------------------------


def _zeros(shape: tuple[int, ...], dtype: DType) -> Buffer:
    np_dtype = np.uint8 if dtype == DType.uint8 else np.float32
    return Buffer.from_numpy(np.zeros(shape, np_dtype)).view(dtype)


def _checkpoint(nvfp4: bool) -> dict[str, Buffer]:
    """Language keys laid out as the 26B-A4B checkpoints store them."""
    bf16 = DType.bfloat16
    wm: dict[str, Buffer] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = _zeros((_VOCAB, _HIDDEN), bf16)
    wm[prefix + "norm.weight"] = _zeros((_HIDDEN,), bf16)
    for i in range(_N_LAYERS):
        lp = f"{prefix}layers.{i}."
        n_kv = _N_KV_HEADS if i == 0 else _N_GLOBAL_KV_HEADS
        hd = _HEAD_DIM if i == 0 else _GLOBAL_HEAD_DIM
        for proj, shape in (
            ("q_proj", (_N_HEADS * hd, _HIDDEN)),
            ("k_proj", (n_kv * hd, _HIDDEN)),
            ("o_proj", (_HIDDEN, _N_HEADS * hd)),
        ):
            wm[f"{lp}self_attn.{proj}.weight"] = _zeros(shape, bf16)
        # attention_k_eq_v: only the sliding layer has a v_proj.
        if i == 0:
            wm[f"{lp}self_attn.v_proj.weight"] = _zeros(
                (n_kv * hd, _HIDDEN), bf16
            )
        wm[lp + "self_attn.q_norm.weight"] = _zeros((hd,), bf16)
        wm[lp + "self_attn.k_norm.weight"] = _zeros((hd,), bf16)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ):
            wm[f"{lp}{norm}.weight"] = _zeros((_HIDDEN,), bf16)
        for proj, shape in (
            ("gate_proj", (_INTERMEDIATE, _HIDDEN)),
            ("up_proj", (_INTERMEDIATE, _HIDDEN)),
            ("down_proj", (_HIDDEN, _INTERMEDIATE)),
        ):
            wm[f"{lp}mlp.{proj}.weight"] = _zeros(shape, bf16)
        wm[lp + "router.proj.weight"] = _zeros((_NUM_EXPERTS, _HIDDEN), bf16)
        wm[lp + "router.scale"] = _zeros((_HIDDEN,), bf16)
        wm[lp + "router.per_expert_scale"] = _zeros((_NUM_EXPERTS,), bf16)
        if not nvfp4:
            wm[lp + "experts.gate_up_proj"] = _zeros(
                (_NUM_EXPERTS, 2 * _MOE_DIM, _HIDDEN), bf16
            )
            wm[lp + "experts.down_proj"] = _zeros(
                (_NUM_EXPERTS, _HIDDEN, _MOE_DIM), bf16
            )
        for e in range(_NUM_EXPERTS if nvfp4 else 0):
            for proj, (rows, cols) in (
                ("gate_proj", (_MOE_DIM, _HIDDEN)),
                ("up_proj", (_MOE_DIM, _HIDDEN)),
                ("down_proj", (_HIDDEN, _MOE_DIM)),
            ):
                ep = f"{lp}experts.{e}.{proj}."
                wm[ep + "weight"] = _zeros((rows, cols // 2), DType.uint8)
                wm[ep + "weight_scale"] = _zeros(
                    (rows, cols // 16), DType.float8_e4m3fn
                )
                wm[ep + "weight_scale_2"] = _zeros((), DType.float32)
                wm[ep + "input_scale"] = _zeros((), DType.float32)
        wm[lp + "layer_scalar"] = _zeros((1,), bf16)
    return wm


def _prepared_decoder_names(config: Gemma4Config, nvfp4: bool) -> set[str]:
    """The names the registered decoder role receives from the checkpoint."""
    checkpoint = {
        name: WeightData(buf, name, buf.dtype, Shape(buf.shape))
        for name, buf in _checkpoint(nvfp4).items()
    }
    prepared = config.prepare_weights(checkpoint)
    assert all(name.startswith("decoder.") for name in prepared)
    return {name.removeprefix("decoder.") for name in prepared}


@pytest.mark.parametrize("nvfp4", [False, True], ids=["bf16", "nvfp4"])
def test_decoder_declares_every_moe_checkpoint_weight(nvfp4: bool) -> None:
    """The registered decoder declares exactly each checkpoint's keys.

    ``Module.compile`` ignores undeclared keys, so a dropped router or expert
    would load silently.
    """
    quant_config = _nvfp4_quant_config() if nvfp4 else None
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    config = _model_config([DeviceRef.CPU()], quant_config).apply_runtime(
        ModuleRuntime(
            mesh=mesh,
            data_parallel_degree=1,
            max_batch_input_tokens=1024,
            max_batch_total_tokens=None,
        )
    )
    with F.lazy(), default_device(mesh):
        decoder = Gemma4Decoder(config)

    declared = {name for name, _ in decoder.parameters}
    assert _prepared_decoder_names(config, nvfp4) == declared

    for i in range(_N_LAYERS):
        moe = f"language_model.layers.{i}.moe_block"
        for name in (
            "gate.gate_score.weight",
            "gate.scale",
            "gate.per_expert_scale",
            "pre_expert_norm.weight",
        ):
            assert f"{moe}.{name}" in declared
        experts = {n for n in declared if n.startswith(f"{moe}.experts.")}
        if nvfp4:
            assert experts == {
                f"{moe}.experts.{e}.{proj}.{leaf}"
                for e in range(_NUM_EXPERTS)
                for proj in ("gate_proj", "up_proj", "down_proj")
                for leaf in (
                    "weight",
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                )
            }
        else:
            assert experts == {
                f"{moe}.experts.gate_up_proj",
                f"{moe}.experts.down_proj",
            }
        for norm in (
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ):
            assert f"language_model.layers.{i}.{norm}.weight" in declared


def test_nvfp4_moe_quantizes_only_the_experts() -> None:
    """The NVFP4 checkpoint's dense MLP and router stay BF16."""
    model = _model_on_devices(
        _model_config([DeviceRef.CPU()], _nvfp4_quant_config()), 1
    )
    weights = dict(model.parameters)
    assert weights["layers.0.mlp.gate_proj.weight"].dtype == DType.bfloat16
    assert weights["layers.0.moe_block.gate.gate_score.weight"].dtype == (
        DType.bfloat16
    )
    expert = "layers.0.moe_block.experts.0.gate_proj"
    assert weights[f"{expert}.weight"].dtype == DType.uint8
    assert weights[f"{expert}.weight_scale"].dtype == DType.float8_e4m3fn
    assert weights[f"{expert}.weight_scale_2"].dtype == DType.float32


# ---------------------------------------------------------------------------
# TP2 placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nvfp4", [False, True], ids=["bf16", "nvfp4"])
def test_tp2_splits_every_expert_moe_dim(nvfp4: bool) -> None:
    """Each device holds half of every expert's gate, up and down widths."""
    quant_config = _nvfp4_quant_config() if nvfp4 else None
    config = _model_config([DeviceRef.CPU()] * 2, quant_config)
    mesh = DeviceMesh((CPU(),) * 2, (2,), ("tp",))
    # Read the bundles in the build context; a later one would realize them.
    with (
        F.lazy(),
        default_dtype(config.unquantized_dtype),
        default_device(mesh),
    ):
        moe = _moe(Gemma4TextModel(config, mesh), 0)
        gate_up, down = moe.gate_up_proj, moe.down_proj
    assert isinstance(moe.experts, Gemma4StackedExperts) != nvfp4
    assert len(gate_up) == len(down) == 2
    half = _MOE_DIM // 2
    for gu, dn in zip(gate_up, down, strict=True):
        if nvfp4:
            assert isinstance(gu, NVFP4Tensor)
            assert isinstance(dn, NVFP4Tensor)
            gu_data, dn_data = gu.weight, dn.weight
            assert list(gu.weight_scale.shape) == [
                _NUM_EXPERTS,
                2 * half,
                _HIDDEN // 16,
            ]
            assert list(dn.weight_scale.shape) == [
                _NUM_EXPERTS,
                _HIDDEN,
                half // 16,
            ]
            # One global scale pair per expert, so each can carry its own
            # up/gate ratio.
            for scale in (dn.weight_scale_2, dn.input_scale):
                assert list(scale.shape) == [_NUM_EXPERTS]
            packed = 2
        else:
            assert isinstance(gu, Tensor) and isinstance(dn, Tensor)
            gu_data, dn_data = gu, dn
            packed = 1
        assert list(gu_data.shape) == [
            _NUM_EXPERTS,
            2 * half,
            _HIDDEN // packed,
        ]
        assert list(dn_data.shape) == [_NUM_EXPERTS, _HIDDEN, half // packed]

    for name, weight in moe.parameters:
        if name.startswith(("gate.", "pre_expert_norm.")):
            assert weight.placements == (Replicated(),), name


def _shard_and_stack_inputs(ir: str) -> list[tuple[list[int], int]]:
    """Each host shard's input shape and split axis, in graph order."""
    return [
        ([int(d) for d in shape.split(",")], int(axis))
        for axis, shape in re.findall(
            r"parameters = \{axis = (\d+) : index\}, "
            r'symbol = "gather_scatter\.shard_and_stack"\}> : '
            r"\(!mo\.tensor<\[([\d, ]+)\]",
            ir,
        )
    ]


def test_tp2_keeps_each_expert_gate_and_up_slices_paired() -> None:
    """Device ``d`` gets slice ``d`` of every expert's gate and of its up.

    The fused rows split as ``[expert, gate|up, moe_dim, hidden]`` along
    ``moe_dim``; a plain split of the ``2 * moe_dim`` rows would give one
    device every gate row. Checks the emitted sharding, not the values.
    """
    mesh = DeviceMesh((CPU(),) * 2, (2,), ("tp",))
    with F.lazy(), default_dtype(DType.float32), default_device(mesh):
        moe = Gemma4MoE(
            _HIDDEN, _NUM_EXPERTS, _TOP_K, _MOE_DIM, _EPS, _GELU_TANH
        )
        assert isinstance(moe.experts, Gemma4StackedExperts)
        assert len(moe.gate_up_proj) == len(moe.down_proj) == 2
        ir = str(Graph.current._mlir_op)

    assert _shard_and_stack_inputs(ir) == [
        ([_NUM_EXPERTS, 2, _MOE_DIM, _HIDDEN], 2),
        ([_NUM_EXPERTS, _HIDDEN, _MOE_DIM], 2),
    ]
