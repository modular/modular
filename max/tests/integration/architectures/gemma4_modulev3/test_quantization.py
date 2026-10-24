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
"""Gemma4 ModuleV3's NVFP4 quantization: config parsing, weight names, TP
scale sharding.

Mirrors ``nvidia/Gemma-4-31B-IT-NVFP4``: modelopt NVFP4 with every
``self_attn`` subtree and ``lm_head`` ignore-listed, so the dense MLPs are
quantized and attention stays BF16.
"""

from __future__ import annotations

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMesh,
    Replicated,
    Sharded,
)
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4_modulev3.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4_modulev3.weight_adapters import (
    convert_language_state_dict_for_module,
)
from max.pipelines.weights.quant import parse_quant_config
from transformers import PretrainedConfig

# Small dims: TP2-divisible in the shapes that matter for scale sharding
# (rows of gate/up, columns of down, and their /16 scale columns).
_HIDDEN = 64
_INTERMEDIATE = 128
_VOCAB = 256
_N_LAYERS = 2
_N_HEADS = 2
_HEAD_DIM = 32
_N_KV_HEADS = 2
_N_GLOBAL_KV_HEADS = 2
_GLOBAL_HEAD_DIM = 64
_RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# Quant-config parsing
# ---------------------------------------------------------------------------


def _nvfp4_quant_config(num_hidden_layers: int = _N_LAYERS) -> QuantConfig:
    """The parsed modelopt/NVFP4 config of an NVFP4 Gemma4 target.

    Mirrors ``nvidia/Gemma-4-31B-IT-NVFP4``'s ``quantization_config`` (every
    ``self_attn`` subtree plus ``lm_head``/embed_vision ignore-listed,
    ``Linear`` targets otherwise) at ``num_hidden_layers`` layers, and passes
    the same name prefixes ``Gemma4ForConditionalGenerationConfig.finalize``
    does, so the per-layer classification comes from the production parser
    rather than a hand-built :class:`QuantConfig`.
    """
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
                    f"model.language_model.layers.{i}.self_attn*"
                    for i in range(num_hidden_layers)
                ],
            }
        }
    )
    # Gemma 4 nests the language tower's depth under text_config, which is
    # what the parser reads to enumerate layers.
    hf_config.text_config = PretrainedConfig.from_dict(
        {"num_hidden_layers": num_hidden_layers}
    )
    quant_config = parse_quant_config(
        hf_config,
        {},
        DType.uint8,
        state_dict_name_prefix="model.language_model.",
        ignored_modules_prefix="model.language_model.",
    )
    assert quant_config is not None
    assert quant_config.is_nvfp4
    assert quant_config.mlp_quantized_layers == set(range(num_hidden_layers))
    assert quant_config.attn_quantized_layers == set()
    return quant_config


def test_parse_quant_config_classifies_gemma4_layers() -> None:
    """The production parser puts every MLP and no attention in NVFP4.

    The Gemma4 prefixes (``model.language_model.``) must reach the parser:
    without them the ignore-listed attention globs never match a real key
    prefix and the per-layer classification silently inverts.
    """
    config = _nvfp4_quant_config()
    assert config.weight_scale.block_size == (1, 16)
    assert config.weight_scale.dtype == DType.float8_e4m3fn
    assert config.input_scale.dtype == DType.float32


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def _make_text_config(
    devices: list[DeviceRef],
    quant_config: QuantConfig | None,
) -> Gemma4TextConfig:
    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=_N_LAYERS,
        devices=devices,
        page_size=128,
    )
    return Gemma4TextConfig(
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
        rms_norm_eps=_RMS_EPS,
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
        kv_params=text_kv,
        num_global_key_value_heads=_N_GLOBAL_KV_HEADS,
        global_head_dim=_GLOBAL_HEAD_DIM,
        attention_k_eq_v=True,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=1.0
        ),
        global_rope_theta=10000.0,
        sliding_window_rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        quant_config=quant_config,
    )


def _make_model_config(
    devices: list[DeviceRef],
    quant_config: QuantConfig | None,
) -> Gemma4ForConditionalGenerationConfig:
    text_config = _make_text_config(devices, quant_config)
    text_config.return_logits = ReturnLogits.LAST_TOKEN
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        # uint8 = packed float4_e2m1fnx2, as the pipeline resolves it.
        dtype=DType.uint8 if quant_config is not None else DType.bfloat16,
        unquantized_dtype=DType.bfloat16,
        kv_params=_make_kv_params(devices),
        text_config=text_config,
        vision_config=None,
        image_token_index=0,
        tie_word_embeddings=True,
    )


def _make_kv_params(devices: list[DeviceRef]) -> MultiKVCacheParams:
    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_N_GLOBAL_KV_HEADS,
        head_dim=_GLOBAL_HEAD_DIM,
        num_layers=1,
        devices=devices,
        page_size=128,
    )
    return MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )


def _model_on_devices(
    config: Gemma4ForConditionalGenerationConfig,
    n: int,
) -> Gemma4TextModel:
    """Builds the text model lazily on an ``n``-device CPU mesh."""
    mesh = DeviceMesh((CPU(),) * n, (n,), ("tp",))
    with (
        F.lazy(),
        default_dtype(config.unquantized_dtype),
        default_device(mesh),
    ):
        return Gemma4TextModel(config, mesh)


# ---------------------------------------------------------------------------
# Weight key remap
# ---------------------------------------------------------------------------


def _weight(array: np.ndarray, dtype: DType) -> Buffer:
    return Buffer.from_numpy(np.ascontiguousarray(array)).view(dtype)


def _nvfp4_checkpoint(quant_config: QuantConfig) -> dict[str, Buffer]:
    """Checkpoint-shaped language keys for the small NVFP4 config.

    Mirrors the HF layout of ``nvidia/Gemma-4-31B-IT-NVFP4``: everything
    under ``model.language_model.``, MLPs in packed uint8 with F8_E4M3 block
    scales (the checkpoint's safetensors header dtype), F32 per-tensor
    scales, attention in BF16.
    """
    wm: dict[str, Buffer] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = _weight(
        np.zeros((_VOCAB, _HIDDEN), np.float32), DType.bfloat16
    )
    wm[prefix + "norm.weight"] = _weight(
        np.zeros(_HIDDEN, np.float32), DType.bfloat16
    )
    for i in range(_N_LAYERS):
        lp = f"{prefix}layers.{i}."
        n_kv = _N_KV_HEADS if i == 0 else _N_GLOBAL_KV_HEADS
        hd = _HEAD_DIM if i == 0 else _GLOBAL_HEAD_DIM
        for proj, rows, cols in (
            ("q_proj", _N_HEADS * hd, _HIDDEN),
            ("k_proj", n_kv * hd, _HIDDEN),
            ("o_proj", _HIDDEN, _N_HEADS * hd),
        ):
            wm[f"{lp}self_attn.{proj}.weight"] = _weight(
                np.zeros((rows, cols), np.float32), DType.bfloat16
            )
        # With attention_k_eq_v the full-attention layer shares K weights
        # for V; the sliding layer has its own v_proj.
        if i == 0:
            wm[f"{lp}self_attn.v_proj.weight"] = _weight(
                np.zeros((n_kv * hd, _HIDDEN), np.float32), DType.bfloat16
            )
        wm[lp + "self_attn.q_norm.weight"] = _weight(
            np.zeros(hd, np.float32), DType.bfloat16
        )
        wm[lp + "self_attn.k_norm.weight"] = _weight(
            np.zeros(hd, np.float32), DType.bfloat16
        )
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = _weight(
                np.zeros(_HIDDEN, np.float32), DType.bfloat16
            )
        for proj, rows, cols in (
            ("gate_proj", _INTERMEDIATE, _HIDDEN),
            ("up_proj", _INTERMEDIATE, _HIDDEN),
            ("down_proj", _HIDDEN, _INTERMEDIATE),
        ):
            wm[f"{lp}mlp.{proj}.weight"] = _weight(
                np.zeros((rows, cols // 2), np.uint8), DType.uint8
            )
            wm[f"{lp}mlp.{proj}.weight_scale"] = _weight(
                np.zeros((rows, cols // 16), np.float32),
                DType.float8_e4m3fn,
            )
            wm[f"{lp}mlp.{proj}.weight_scale_2"] = _weight(
                np.zeros((), np.float32), DType.float32
            )
            wm[f"{lp}mlp.{proj}.input_scale"] = _weight(
                np.zeros((), np.float32), DType.float32
            )
        wm[lp + "layer_scalar"] = _weight(
            np.zeros(1, np.float32), DType.bfloat16
        )
    return wm


def _to_safetensor_weights(
    wm: dict[str, Buffer],
) -> SafetensorWeights:
    return SafetensorWeights(
        [],
        tensors=set(wm.keys()),
        tensors_to_file_idx={},
        _st_weight_map=wm,
    )


def test_nvfp4_checkpoint_keys_match_module_tree() -> None:
    """The NVFP4 adapter's output keys must exactly cover the module tree.

    ``Module.compile(weights=...)`` silently ignores unmatched entries, so
    a misnamed scale would leave a quantized leaf serving its zero
    initialization. The quantized MLP keys keep the checkpoint's names,
    which the ``NVFP4Tensor`` leaves expose, as the BF16 attention and norm
    keys do.
    """
    quant_config = _nvfp4_quant_config()
    config = _make_model_config([DeviceRef.CPU()], quant_config)
    model = _model_on_devices(config, n=1)

    wm = _nvfp4_checkpoint(quant_config)
    converted = set(
        convert_language_state_dict_for_module(
            dict(_to_safetensor_weights(wm).items())
        )
    )

    expected = {f"language_model.{name}" for name, _ in model.parameters}
    assert converted == expected

    proj = "language_model.layers.0.mlp.gate_proj"
    for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        assert f"{proj}.{leaf}" in converted
    # Nothing nests under a projection's ``weight``.
    assert not any(".weight." in name for name in converted)


# ---------------------------------------------------------------------------
# TP2 scale sharding
# ---------------------------------------------------------------------------


def test_nvfp4_mlp_weights_shard_their_scales_too() -> None:
    """Under TP2 the packed data and block scales shard together.

    ``col_parallel``/``row_parallel`` set the sharding intent on the whole
    ``NVFP4Tensor``, whose ``to()`` co-shards ``data`` and ``weight_scale``
    so each device's scale rows describe its own data rows; the per-tensor
    ``weight_scale_2``/``input_scale`` stay replicated CPU scalars.

    The small dims also pin the divisibility contract the 31B checkpoint
    satisfies: gate/up rows (=intermediate) shard on axis 0; down cols
    (=intermediate) shard on axis 1 with scale cols = intermediate // 16,
    both even.
    """
    quant_config = _nvfp4_quant_config()
    config = _make_model_config([DeviceRef.CPU()] * 2, quant_config)
    weights = dict(_model_on_devices(config, n=2).parameters)

    mlp = "layers.0.mlp"
    for proj in ("gate_proj", "up_proj"):
        data = weights[f"{mlp}.{proj}.weight"]
        assert data.dtype == DType.uint8
        assert [int(d) for d in data.shape] == [
            _INTERMEDIATE,
            _HIDDEN // 2,
        ]
        assert data.placements == (Sharded(0),)
        block = weights[f"{mlp}.{proj}.weight_scale"]
        assert block.dtype == DType.float8_e4m3fn
        assert [int(d) for d in block.shape] == [
            _INTERMEDIATE,
            _HIDDEN // 16,
        ]
        assert block.placements == (Sharded(0),)

    data = weights[f"{mlp}.down_proj.weight"]
    assert [int(d) for d in data.shape] == [_HIDDEN, _INTERMEDIATE // 2]
    assert data.placements == (Sharded(1),)
    block = weights[f"{mlp}.down_proj.weight_scale"]
    assert [int(d) for d in block.shape] == [_HIDDEN, _INTERMEDIATE // 16]
    assert block.placements == (Sharded(1),)

    for scale in ("weight_scale_2", "input_scale"):
        tensor = weights[f"{mlp}.gate_proj.{scale}"]
        assert tensor.dtype == DType.float32
        assert tensor.placements == (Replicated(),)


def test_quantized_attention_keeps_bf16_placements() -> None:
    """A quantized attention layer shards like the BF16 one.

    The first-party checkpoint keeps attention BF16, but community quants
    may quantize it; the projections must still carry TP intent and the
    per-projection matmul path must not fuse (each projection's scales
    apply to its own rows).
    """
    quant_config = QuantConfig.nvfp4(
        mlp_quantized_layers=set(range(_N_LAYERS)),
        attn_quantized_layers=set(range(_N_LAYERS)),
    )
    config = _make_model_config([DeviceRef.CPU()] * 2, quant_config)
    weights = dict(_model_on_devices(config, n=2).parameters)

    attn = "layers.0.self_attn"
    for proj, placement in (("q_proj", Sharded(0)), ("o_proj", Sharded(1))):
        data = weights[f"{attn}.{proj}.weight"]
        assert data.dtype == DType.uint8
        assert data.placements == (placement,)
        block = weights[f"{attn}.{proj}.weight_scale"]
        assert block.dtype == DType.float8_e4m3fn
        # The scale shards with the data's axis (0 col-parallel rows,
        # 1 row-parallel cols).
        assert block.placements == data.placements


def test_bf16_checkpoint_keys_still_match_module_tree() -> None:
    """A bf16 checkpoint's keys exactly cover the BF16 module tree."""
    config = _make_model_config([DeviceRef.CPU()], None)
    model = _model_on_devices(config, n=1)

    wm: dict[str, Buffer] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = _weight(
        np.zeros((_VOCAB, _HIDDEN), np.float32), DType.bfloat16
    )
    wm[prefix + "norm.weight"] = _weight(
        np.zeros(_HIDDEN, np.float32), DType.bfloat16
    )
    for i in range(_N_LAYERS):
        lp = f"{prefix}layers.{i}."
        n_kv = _N_KV_HEADS if i == 0 else _N_GLOBAL_KV_HEADS
        hd = _HEAD_DIM if i == 0 else _GLOBAL_HEAD_DIM
        for proj, rows, cols in (
            ("q_proj", _N_HEADS * hd, _HIDDEN),
            ("k_proj", n_kv * hd, _HIDDEN),
            ("o_proj", _HIDDEN, _N_HEADS * hd),
        ):
            wm[lp + f"self_attn.{proj}.weight"] = _weight(
                np.zeros((rows, cols), np.float32), DType.bfloat16
            )
        # attention_k_eq_v: only the sliding layer has a v_proj.
        if i == 0:
            wm[lp + "self_attn.v_proj.weight"] = _weight(
                np.zeros((n_kv * hd, _HIDDEN), np.float32), DType.bfloat16
            )
        wm[lp + "self_attn.q_norm.weight"] = _weight(
            np.zeros(hd, np.float32), DType.bfloat16
        )
        wm[lp + "self_attn.k_norm.weight"] = _weight(
            np.zeros(hd, np.float32), DType.bfloat16
        )
        for proj, rows, cols in (
            ("gate_proj", _INTERMEDIATE, _HIDDEN),
            ("up_proj", _INTERMEDIATE, _HIDDEN),
            ("down_proj", _HIDDEN, _INTERMEDIATE),
        ):
            wm[lp + f"mlp.{proj}.weight"] = _weight(
                np.zeros((rows, cols), np.float32), DType.bfloat16
            )
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = _weight(
                np.zeros(_HIDDEN, np.float32), DType.bfloat16
            )
        wm[lp + "layer_scalar"] = _weight(
            np.zeros(1, np.float32), DType.bfloat16
        )

    converted = set(
        convert_language_state_dict_for_module(
            dict(_to_safetensor_weights(wm).items())
        )
    )
    expected = {f"language_model.{name}" for name, _ in model.parameters}
    assert converted == expected
