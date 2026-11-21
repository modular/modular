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
"""``kimi_k3_weight_rules``: a Kimi K3 checkpoint's names on the text tower.

CPU-only. The modules declare every tensor as the checkpoint stores it, so
the rules only rename, relabel the MXFP4 scales and drop the vision tower.
A small synthetic checkpoint under the real checkpoint's names must cover a
lazily built text tower name for name and shape for shape. The per-device
expert stacks the graph builds from the routed experts are checked on a CPU
mesh.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from max import functional as F
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)
from max.pipelines.architectures.kimi_k3_modulev3.model import (
    KimiK3Model,
    KimiK3ModelConfig,
)
from max.pipelines.architectures.kimi_k3_modulev3.model_config import (
    KimiK3Config,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    kimi_k3_weight_rules,
)
from max.pipelines.lib import WeightTransform
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.support.math import ceildiv
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.sharding import DeviceMesh

_HIDDEN_SIZE = 16
_VOCAB_SIZE = 32
_NUM_HIDDEN_LAYERS = 2
_FIRST_K_DENSE_REPLACE = 1
# 1-indexed in the checkpoint's `linear_attn_config`: layer 0 is KDA (dense
# MLP), layer 1 is MLA (MoE) -- covering both attention kinds and both FFN
# kinds in two layers.
_KDA_LAYERS = [1]
_FULL_ATTN_LAYERS = [2]
_KDA_NUM_HEADS = 2
_KDA_HEAD_DIM = 4
_KDA_CONV_KERNEL_SIZE = 4
_KDA_INNER = _KDA_NUM_HEADS * _KDA_HEAD_DIM
# A_log is stored longer than the active head count, with an exact-zero
# tail -- mirrors the real checkpoint's 128-wide storage for 96 active heads.
_A_LOG_STORED_LEN = _KDA_NUM_HEADS + 2

_NUM_ATTENTION_HEADS = 2
_QK_NOPE_HEAD_DIM = 4
_QK_ROPE_HEAD_DIM = 2
_V_HEAD_DIM = 4
_Q_LORA_RANK = 8
_KV_LORA_RANK = 8

_INTERMEDIATE_SIZE = 12
_ROUTED_EXPERT_HIDDEN_SIZE = 6
_MOE_INTERMEDIATE_SIZE = 5
_NUM_EXPERTS = 3
_NUM_EXPERTS_PER_TOKEN = 2
_NUM_SHARED_EXPERTS = 1

# Even, so both axes a packed expert splits land on a whole byte and a
# whole 32-element E8M0 scale block.
_PACKED_ROUTED_EXPERT_HIDDEN_SIZE = 8
_PACKED_MOE_INTERMEDIATE_SIZE = 4

_MAX_SEQ_LEN = 128

_TEXT = "language_model.model"
_RNG = np.random.default_rng(0)


class _FakeTextConfig(SimpleNamespace):
    """Minimal stand-in for the HuggingFace text sub-config; see
    ``test_model_config.py``'s identical fixture."""

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _text_config(
    *,
    routed_expert_hidden_size: int = _ROUTED_EXPERT_HIDDEN_SIZE,
    moe_intermediate_size: int = _MOE_INTERMEDIATE_SIZE,
) -> _FakeTextConfig:
    return _FakeTextConfig(
        hidden_size=_HIDDEN_SIZE,
        vocab_size=_VOCAB_SIZE,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        first_k_dense_replace=_FIRST_K_DENSE_REPLACE,
        linear_attn_config={
            "kda_layers": _KDA_LAYERS,
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "num_heads": _KDA_NUM_HEADS,
            "head_dim": _KDA_HEAD_DIM,
            "short_conv_kernel_size": _KDA_CONV_KERNEL_SIZE,
        },
        num_attention_heads=_NUM_ATTENTION_HEADS,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        attn_res_block_size=4,
        intermediate_size=_INTERMEDIATE_SIZE,
        routed_expert_hidden_size=routed_expert_hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
        num_shared_experts=_NUM_SHARED_EXPERTS,
        rms_norm_eps=1e-5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        max_position_embeddings=4096,
    )


def _expert_sizes(packed: bool) -> dict[str, int]:
    """The routed expert widths; a packed checkpoint needs both even."""
    if not packed:
        return {}
    return {
        "routed_expert_hidden_size": _PACKED_ROUTED_EXPERT_HIDDEN_SIZE,
        "moe_intermediate_size": _PACKED_MOE_INTERMEDIATE_SIZE,
    }


def _fp32(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _weight(array: np.ndarray, name: str, dtype: DType = DType.float32) -> Any:
    """A stand-in ``Weights`` whose ``data()`` returns real ``WeightData``."""
    if dtype == DType.bfloat16:
        # Truncate each fp32 value to its bf16 bit pattern, the same
        # round-to-16-bits a real bf16 checkpoint stores, built without a
        # numpy bfloat16 dtype.
        bits = (array.astype(np.float32).view(np.uint32) >> 16).astype(
            np.uint16
        )
        shape = tuple(int(d) for d in array.shape)
        weight_data = WeightData(
            data=Buffer.from_dlpack(bits).view(
                dtype=DType.bfloat16, shape=shape
            ),
            name=name,
            dtype=DType.bfloat16,
            shape=Shape(shape),
        )
    else:
        weight_data = WeightData.from_numpy(array.astype(np.float32), name)
    weights = MagicMock(spec=Weights)
    weights.data.return_value = weight_data
    return weights


def _weight_u8(array: np.ndarray, name: str) -> Any:
    """A stand-in ``Weights`` whose ``data()`` returns raw ``uint8`` data.

    Mirrors the real checkpoint's MXFP4 ``weight_packed``/``weight_scale``
    storage: both ship as plain ``U8`` bytes, since safetensors has no
    packed-FP4 or E8M0 dtype.
    """
    shape = tuple(int(d) for d in array.shape)
    weight_data = WeightData(
        data=Buffer.from_dlpack(array).view(dtype=DType.uint8, shape=shape),
        name=name,
        dtype=DType.uint8,
        shape=Shape(shape),
    )
    weights = MagicMock(spec=Weights)
    weights.data.return_value = weight_data
    return weights


def _to_numpy_u8(data: WeightData) -> np.ndarray:
    """Reads a packed-MXFP4 or E8M0-scale ``WeightData`` as raw ``uint8``
    bytes, for exact byte-identity assertions (no value decoding)."""
    return np.from_dlpack(Buffer.from_dlpack(data.data).view(DType.uint8))


def _checkpoint(
    *,
    routed_expert_hidden_size: int = _ROUTED_EXPERT_HIDDEN_SIZE,
    moe_intermediate_size: int = _MOE_INTERMEDIATE_SIZE,
    packed_experts: bool = False,
) -> dict[str, Any]:
    """A small Kimi K3 checkpoint under the real checkpoint's names: layer 0
    is KDA + dense MLP, layer 1 is MLA + MoE, beside a vision tower the text
    tower does not load.

    Args:
        routed_expert_hidden_size, moe_intermediate_size: Override the module
            globals -- the MXFP4 packed case needs both even (two 4-bit
            values per packed byte), which the globals are not.
        packed_experts: Builds the routed experts MXFP4-packed
            (``weight_packed``/``weight_scale``) instead of dense ``.weight``.
    """
    bf16 = DType.bfloat16
    shared_intermediate_size = moe_intermediate_size * _NUM_SHARED_EXPERTS
    ck: dict[str, Any] = {
        "vision_tower.patch_embed.proj.weight": _weight(
            _fp32((_HIDDEN_SIZE, 3)), "vision", bf16
        ),
        "mm_projector.proj.0.weight": _weight(
            _fp32((_HIDDEN_SIZE, _HIDDEN_SIZE)), "projector", bf16
        ),
        f"{_TEXT}.embed_tokens.weight": _weight(
            _fp32((_VOCAB_SIZE, _HIDDEN_SIZE)), "embed", bf16
        ),
        "language_model.lm_head.weight": _weight(
            _fp32((_VOCAB_SIZE, _HIDDEN_SIZE)), "lm_head", bf16
        ),
        f"{_TEXT}.norm.weight": _weight(_fp32((_HIDDEN_SIZE,)), "norm", bf16),
        f"{_TEXT}.output_attn_res_proj.weight": _weight(
            _fp32((1, _HIDDEN_SIZE)), "out_res_proj", bf16
        ),
        f"{_TEXT}.output_attn_res_norm.weight": _weight(
            _fp32((_HIDDEN_SIZE,)), "out_res_norm", bf16
        ),
    }

    for i in range(_NUM_HIDDEN_LAYERS):
        src = f"{_TEXT}.layers.{i}"
        for leaf, shape in (
            ("input_layernorm", (_HIDDEN_SIZE,)),
            ("post_attention_layernorm", (_HIDDEN_SIZE,)),
            ("self_attention_res_proj", (1, _HIDDEN_SIZE)),
            ("self_attention_res_norm", (_HIDDEN_SIZE,)),
            ("mlp_res_proj", (1, _HIDDEN_SIZE)),
            ("mlp_res_norm", (_HIDDEN_SIZE,)),
        ):
            ck[f"{src}.{leaf}.weight"] = _weight(_fp32(shape), leaf, bf16)

        is_kda = i in {j - 1 for j in _KDA_LAYERS}
        attn_src = f"{src}.self_attn"
        if is_kda:
            for leaf, shape in (
                ("q_proj", (_KDA_INNER, _HIDDEN_SIZE)),
                ("k_proj", (_KDA_INNER, _HIDDEN_SIZE)),
                ("v_proj", (_KDA_INNER, _HIDDEN_SIZE)),
                ("f_a_proj", (_KDA_HEAD_DIM, _HIDDEN_SIZE)),
                ("f_b_proj", (_KDA_INNER, _KDA_HEAD_DIM)),
                ("b_proj", (_KDA_NUM_HEADS, _HIDDEN_SIZE)),
                ("g_proj", (_KDA_INNER, _HIDDEN_SIZE)),
                ("o_proj", (_HIDDEN_SIZE, _KDA_INNER)),
            ):
                ck[f"{attn_src}.{leaf}.weight"] = _weight(
                    _fp32(shape), leaf, bf16
                )
            ck[f"{attn_src}.o_norm.weight"] = _weight(
                _fp32((_KDA_HEAD_DIM,)), "o_norm"
            )
            for axis in ("q", "k", "v"):
                ck[f"{attn_src}.{axis}_conv1d.weight"] = _weight(
                    _fp32((_KDA_INNER, 1, _KDA_CONV_KERNEL_SIZE)),
                    f"{axis}_conv1d",
                )
            a_log = np.zeros(_A_LOG_STORED_LEN, dtype=np.float32)
            a_log[:_KDA_NUM_HEADS] = _fp32((_KDA_NUM_HEADS,))
            ck[f"{attn_src}.A_log"] = _weight(a_log, "A_log")
            ck[f"{attn_src}.dt_bias"] = _weight(_fp32((_KDA_INNER,)), "dt_bias")
        else:
            q_head_dim = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
            for leaf, shape in (
                ("q_a_proj", (_Q_LORA_RANK, _HIDDEN_SIZE)),
                ("q_a_layernorm", (_Q_LORA_RANK,)),
                ("q_b_proj", (_NUM_ATTENTION_HEADS * q_head_dim, _Q_LORA_RANK)),
                (
                    "kv_a_proj_with_mqa",
                    (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM, _HIDDEN_SIZE),
                ),
                ("kv_a_layernorm", (_KV_LORA_RANK,)),
                (
                    "kv_b_proj",
                    (
                        _NUM_ATTENTION_HEADS
                        * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
                        _KV_LORA_RANK,
                    ),
                ),
                ("g_proj", (_NUM_ATTENTION_HEADS * _V_HEAD_DIM, _HIDDEN_SIZE)),
                ("o_proj", (_HIDDEN_SIZE, _NUM_ATTENTION_HEADS * _V_HEAD_DIM)),
            ):
                ck[f"{attn_src}.{leaf}.weight"] = _weight(
                    _fp32(shape), leaf, bf16
                )

        if i < _FIRST_K_DENSE_REPLACE:
            for leaf, shape in (
                ("gate_proj", (_INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
                ("up_proj", (_INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
                ("down_proj", (_HIDDEN_SIZE, _INTERMEDIATE_SIZE)),
            ):
                ck[f"{src}.mlp.{leaf}.weight"] = _weight(
                    _fp32(shape), leaf, bf16
                )
        else:
            moe = f"{src}.block_sparse_moe"
            ck[f"{moe}.gate.weight"] = _weight(
                _fp32((_NUM_EXPERTS, _HIDDEN_SIZE)), "gate", bf16
            )
            ck[f"{moe}.gate.e_score_correction_bias"] = _weight(
                _fp32((_NUM_EXPERTS,)), "bias"
            )
            ck[f"{moe}.routed_expert_down_proj.weight"] = _weight(
                _fp32((routed_expert_hidden_size, _HIDDEN_SIZE)),
                "down_proj",
                bf16,
            )
            ck[f"{moe}.routed_expert_up_proj.weight"] = _weight(
                _fp32((_HIDDEN_SIZE, routed_expert_hidden_size)),
                "up_proj",
                bf16,
            )
            ck[f"{moe}.routed_expert_norm.weight"] = _weight(
                _fp32((routed_expert_hidden_size,)), "norm", bf16
            )
            for e in range(_NUM_EXPERTS):
                if packed_experts:
                    # w1/w3 pack along the latent (K) axis; w2 packs along
                    # the FFN (K) axis -- two 4-bit values per byte, one
                    # E8M0 scale per 32 unpacked elements.
                    for leaf, rows, k in (
                        (
                            "w1",
                            moe_intermediate_size,
                            routed_expert_hidden_size,
                        ),
                        (
                            "w2",
                            routed_expert_hidden_size,
                            moe_intermediate_size,
                        ),
                        (
                            "w3",
                            moe_intermediate_size,
                            routed_expert_hidden_size,
                        ),
                    ):
                        ck[f"{moe}.experts.{e}.{leaf}.weight_packed"] = (
                            _weight_u8(
                                _RNG.integers(
                                    0, 256, size=(rows, k // 2), dtype=np.uint8
                                ),
                                f"{leaf}.{e}.packed",
                            )
                        )
                        ck[f"{moe}.experts.{e}.{leaf}.weight_scale"] = (
                            _weight_u8(
                                # Real E8M0 bias is 127; stay near it.
                                _RNG.integers(
                                    120, 134, size=(rows, ceildiv(k, 32))
                                ).astype(np.uint8),
                                f"{leaf}.{e}.scale",
                            )
                        )
                    continue
                ck[f"{moe}.experts.{e}.w1.weight"] = _weight(
                    _fp32((moe_intermediate_size, routed_expert_hidden_size)),
                    f"w1.{e}",
                    bf16,
                )
                ck[f"{moe}.experts.{e}.w2.weight"] = _weight(
                    _fp32((routed_expert_hidden_size, moe_intermediate_size)),
                    f"w2.{e}",
                    bf16,
                )
                ck[f"{moe}.experts.{e}.w3.weight"] = _weight(
                    _fp32((moe_intermediate_size, routed_expert_hidden_size)),
                    f"w3.{e}",
                    bf16,
                )
            ck[f"{moe}.shared_experts.gate_proj.weight"] = _weight(
                _fp32((shared_intermediate_size, _HIDDEN_SIZE)), "sg", bf16
            )
            ck[f"{moe}.shared_experts.up_proj.weight"] = _weight(
                _fp32((shared_intermediate_size, _HIDDEN_SIZE)), "su", bf16
            )
            ck[f"{moe}.shared_experts.down_proj.weight"] = _weight(
                _fp32((_HIDDEN_SIZE, shared_intermediate_size)), "sd", bf16
            )
    return ck


def _convert(checkpoint: Mapping[str, Weights]) -> dict[str, WeightData]:
    return WeightTransform(kimi_k3_weight_rules(_text_config()))(checkpoint)


def _shape(shape: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _finalized_config(
    packed: bool, state_dict: Mapping[str, WeightData]
) -> KimiK3Config:
    """The registry config, completed with the converted checkpoint."""
    text_config = _text_config(**_expert_sizes(packed))
    config = KimiK3Config(
        # A packed checkpoint's routed experts resolve to uint8.
        dtype=DType.uint8 if packed else DType.bfloat16,
        kv_params=MagicMock(),
        devices=[DeviceRef.CPU()],
        max_seq_len=_MAX_SEQ_LEN,
        model=KimiK3ModelConfig.from_huggingface_config(
            {"text_config": text_config.to_dict()}
        ),
        max_batch_context_length=_MAX_SEQ_LEN,
    )
    return config.finalize(text_config, state_dict, ModelOutputConfig())


@contextmanager
def _virtual_gfx950() -> Iterator[None]:
    """A gfx950 virtual device: the MXFP4 experts resolve their AMD path
    with no GPU attached."""
    set_virtual_device_api("hip")
    set_virtual_device_target_arch("gfx950")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _module_shapes(config: KimiK3Config) -> dict[str, tuple[int, ...]]:
    """The text tower's parameters, built lazily from ``config``."""
    with (
        _virtual_gfx950(),
        F.lazy(),
        default_device(Accelerator(0)),
        default_dtype(DType.bfloat16),
    ):
        model = KimiK3Model(config.model, quant_config=config.quant_config)
    return {name: _shape(weight.shape) for name, weight in model.parameters}


@pytest.mark.parametrize(
    "packed", [False, True], ids=["bf16_experts", "mxfp4_experts"]
)
def test_checkpoint_covers_the_module(packed: bool) -> None:
    """The renamed checkpoint is the module's parameters, name for name and
    shape for shape, and the vision tower drops out.

    A missing parameter already fails ``Module.compile``; this also catches
    a checkpoint tensor the module would silently skip.
    """
    converted = _convert(
        _checkpoint(packed_experts=packed, **_expert_sizes(packed))
    )
    expected = _module_shapes(_finalized_config(packed, converted))
    assert {
        name: _shape(data.shape) for name, data in converted.items()
    } == expected


def test_finalize_takes_the_stored_a_log_length() -> None:
    """The KDA layers declare ``A_log`` at the length the checkpoint stores
    it, zero tail included, so the load binds it whole."""
    config = _finalized_config(False, _convert(_checkpoint()))
    assert config.model.kda_a_log_size == _A_LOG_STORED_LEN


def test_finalize_refuses_a_short_a_log() -> None:
    """Fewer coefficients than heads would leave a head on a stale decay."""
    checkpoint = _checkpoint()
    checkpoint[f"{_TEXT}.layers.0.self_attn.A_log"] = _weight(
        np.ones(_KDA_NUM_HEADS - 1, dtype=np.float32), "A_log"
    )
    with pytest.raises(ValueError, match="fewer than"):
        _finalized_config(False, _convert(checkpoint))


def test_mxfp4_scales_are_relabeled_e8m0() -> None:
    """The scales load as zero-copy E8M0 views of the checkpoint's ``uint8``
    bytes, and the packed weights as the checkpoint's own buffers."""
    checkpoint = _checkpoint(packed_experts=True, **_expert_sizes(True))
    converted = _convert(checkpoint)
    moe = f"{_TEXT}.layers.1.block_sparse_moe"
    for e in range(_NUM_EXPERTS):
        for w in ("w1", "w2", "w3"):
            dst = f"layers.1.ffn.experts.{e}.{w}"
            src = f"{moe}.experts.{e}.{w}"
            assert (
                converted[f"{dst}.weight"]
                is checkpoint[f"{src}.weight_packed"].data()
            )
            scale = converted[f"{dst}.weight_scale"]
            assert scale.dtype == DType.float8_e8m0fnu
            raw = _to_numpy_u8(checkpoint[f"{src}.weight_scale"].data())
            got = _to_numpy_u8(scale)
            np.testing.assert_array_equal(got, raw)
            assert np.shares_memory(got, raw)


def _cpu_mesh_moe(tp_size: int) -> KimiK3LatentMoE:
    """A packed latent MoE block whose replicated weights sit on a CPU mesh.

    MegaFFN is off because its preshuffled stacking runs a GPU kernel.
    """
    mesh = DeviceMesh((CPU(),) * tp_size, (tp_size,), ("tp",))
    # The E8M0 zeros the experts start with would compile eagerly.
    with F.lazy(), default_device(mesh):
        return KimiK3LatentMoE(
            hidden_size=_HIDDEN_SIZE,
            latent_size=64,
            moe_intermediate_size=128,
            shared_intermediate_size=128,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            beta=4.0,
            linear_beta=None,
            rms_norm_eps=1e-5,
            routed_scaling_factor=1.0,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
            dtype=DType.bfloat16,
            quant_config=QuantConfig.mxfp4(mlp_quantized_layers={0}),
            amd_w4a8=True,
            amd_mega_ffn=False,
        )


def _rank_interleaved(parts: Sequence[np.ndarray], tp_size: int) -> np.ndarray:
    """``parts`` stacked rank by rank: each rank's rows of every part, in
    order, so rank ``r``'s contiguous slice is its share of each."""
    shards = [np.split(part, tp_size) for part in parts]
    return np.concatenate(
        [shard[rank] for rank in range(tp_size) for shard in shards]
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_graph_expert_stacks_split_per_rank(tp_size: int) -> None:
    """Each device's gate/up stack holds, per expert, its rank's rows of the
    gate and then of the up projection, and its down stack its slice of the
    FFN (K) axis, for weights and scales alike.

    The host backend cannot compile E8M0, so random ``uint8`` codes stand in
    for the scales: the stacking only moves bytes.
    """
    layer = _cpu_mesh_moe(tp_size)
    rng = np.random.default_rng(tp_size)
    ref: dict[tuple[str, str], list[np.ndarray]] = {}
    for expert in layer.experts:
        for w in ("w1", "w2", "w3"):
            projection = getattr(expert, w)
            for leaf in ("weight", "weight_scale"):
                shape = [int(d) for d in getattr(projection, leaf).shape]
                codes = rng.integers(0, 256, shape, dtype=np.uint8)
                setattr(projection, leaf, Tensor.from_dlpack(codes))
                ref.setdefault((w, leaf), []).append(codes)

    for leaf, suffix in (("weight", ""), ("weight_scale", "_scale")):
        gate_up = np.stack(
            [
                _rank_interleaved([g, u], tp_size)
                for g, u in zip(ref["w1", leaf], ref["w3", leaf], strict=True)
            ]
        )
        down = np.stack(ref["w2", leaf])
        got_gate_up = getattr(layer, f"experts_gate_up_proj{suffix}")
        got_down = getattr(layer, f"experts_down_proj{suffix}")
        for name, got, want, axis in (
            ("gate_up", got_gate_up, gate_up, 1),
            ("down", got_down, down, 2),
        ):
            shards = [shard.to_numpy() for shard in got.local_shards]
            assert len(shards) == tp_size
            for d, shard in enumerate(shards):
                width = want.shape[axis] // tp_size
                np.testing.assert_array_equal(
                    shard,
                    np.take(want, range(d * width, (d + 1) * width), axis),
                    err_msg=f"{name}{suffix} on device {d}",
                )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "vision_tower.patch_embed.proj.weight",
            "vision_encoder.patch_embed.proj.weight",
        ),
        (
            "vision_tower.encoder.final_layernorm.weight",
            "vision_encoder.encoder.norm.weight",
        ),
        (
            "vision_tower.encoder.blocks.0.wqkv.weight",
            "vision_encoder.encoder.blocks.0.attn.wqkv.weight",
        ),
        (
            "vision_tower.encoder.blocks.0.wo.weight",
            "vision_encoder.encoder.blocks.0.attn.wo.weight",
        ),
        (
            "vision_tower.encoder.blocks.0.mlp.fc0.weight",
            "vision_encoder.encoder.blocks.0.mlp.up_proj.weight",
        ),
        (
            "vision_tower.encoder.blocks.0.mlp.fc1.weight",
            "vision_encoder.encoder.blocks.0.mlp.down_proj.weight",
        ),
        (
            "mm_projector.proj.0.weight",
            "vision_encoder.patch_merger.linear1.weight",
        ),
        (
            "mm_projector.proj.2.weight",
            "vision_encoder.patch_merger.linear2.weight",
        ),
        (
            "mm_projector.post_norm.weight",
            "vision_encoder.patch_merger.post_norm.weight",
        ),
        (
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "layers.0.ffn.gate_proj.weight",
        ),
    ],
)
def test_vision_checkpoint_names_preserve_text_and_vision_mlp_paths(
    source: str, target: str
) -> None:
    config = _text_config()
    config.vision_config = SimpleNamespace(patch_size=14)
    weight = _weight(np.zeros((1,), dtype=np.float32), source)
    transformed = WeightTransform(kimi_k3_weight_rules(config))(
        {source: weight}
    )
    assert list(transformed) == [target]
    assert transformed[target].shape == weight.data().shape
    assert transformed[target].dtype == weight.data().dtype
