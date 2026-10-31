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
"""``convert_kimi_k3_state_dict`` (ModuleV3): key naming and shape transforms.

CPU-only: everything here is dict-to-dict conversion over real (small,
synthetic) tensors, not graph construction. The shape transforms (fused qkv
projection, fused qkv conv, fused gate/up, stacked experts, de-padded
``A_log``) are checked numerically against ``model.py``'s own numpy helpers
-- the same transforms applied to the same raw arrays -- rather than against
a hand-computed expectation, so the two implementations cannot silently
drift apart.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.architectures.kimi_k3_modulev3 import model as km
from max.pipelines.architectures.kimi_k3_modulev3 import (
    weight_adapters as wa,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    convert_kimi_k3_state_dict,
    interleave_for_tp,
    tp_degree,
)
from max.support.math import ceildiv

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
_SHARED_INTERMEDIATE_SIZE = _MOE_INTERMEDIATE_SIZE * _NUM_SHARED_EXPERTS

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


def _fp32(shape: tuple[int, ...]) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _weight(array: np.ndarray, name: str, dtype: DType = DType.float32) -> Any:
    """A stand-in ``Weights`` whose ``data()`` returns real ``WeightData``."""
    if dtype == DType.bfloat16:
        # Truncate each fp32 value to its bf16 bit pattern, the same
        # round-to-16-bits a real bf16 checkpoint stores -- built without a
        # numpy bfloat16 dtype, matching `_rewrap`'s own view-based pattern.
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
    storage: both ship as plain ``U8`` bytes (safetensors has no packed-FP4
    or E8M0 dtype), reinterpreted downstream by `_stack_expert_weights` (for
    `weight_packed`, unchanged) and `_as_e8m0` (for `weight_scale`).
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


def _to_numpy(data: WeightData) -> np.ndarray:
    """Decodes a (possibly bf16) ``WeightData`` back to fp32 numpy, for
    assertions. Mirrors ``_elements``' bf16-as-uint16 view, then widens."""
    buffer = Buffer.from_dlpack(data.data)
    if data.dtype == DType.bfloat16:
        bits = np.from_dlpack(buffer.view(DType.uint16)).astype(np.uint32)
        return (bits << 16).view(np.float32)
    return np.from_dlpack(buffer)


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
    """A small, real Kimi K3 checkpoint: layer 0 is KDA + dense MLP, layer 1
    is MLA + MoE. Stored fp32 throughout except where a comment says
    otherwise, matching the real checkpoint, so the adapter's dtype
    resolution (see `test_unpinned_weights_resolve_to_the_model_dtype`) is
    exercised rather than hidden behind an already-bf16 fixture.

    Args:
        routed_expert_hidden_size, moe_intermediate_size: Override the module
            globals -- the MXFP4 packed case needs both even (two 4-bit
            values per packed byte), which the globals (picked for the dense
            tests) are not.
        packed_experts: Builds the routed experts MXFP4-packed
            (``weight_packed``/``weight_scale``) instead of dense ``.weight``.
    """
    bf16 = DType.bfloat16
    ck: dict[str, Any] = {
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
                ("g_proj", (_KDA_INNER, _HIDDEN_SIZE)),
                ("o_proj", (_HIDDEN_SIZE, _KDA_INNER)),
            ):
                ck[f"{attn_src}.{leaf}.weight"] = _weight(
                    _fp32(shape), leaf, bf16
                )
            # Stored fp32 in the real checkpoint; not pinned, so this proves
            # the adapter resolves it to the model dtype like everything else.
            ck[f"{attn_src}.b_proj.weight"] = _weight(
                _fp32((_KDA_NUM_HEADS, _HIDDEN_SIZE)), "b_proj"
            )
            ck[f"{attn_src}.o_norm.weight"] = _weight(
                _fp32((_KDA_HEAD_DIM,)), "o_norm", bf16
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
                _fp32((_NUM_EXPERTS, _HIDDEN_SIZE)), "gate"
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
                                # Real E8M0 bias is 127; stay near it so a
                                # decode doesn't overflow fp64 in tests that
                                # read these back.
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
                _fp32((_SHARED_INTERMEDIATE_SIZE, _HIDDEN_SIZE)), "sg", bf16
            )
            ck[f"{moe}.shared_experts.up_proj.weight"] = _weight(
                _fp32((_SHARED_INTERMEDIATE_SIZE, _HIDDEN_SIZE)), "su", bf16
            )
            ck[f"{moe}.shared_experts.down_proj.weight"] = _weight(
                _fp32((_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)), "sd", bf16
            )
    return ck


@pytest.fixture
def checkpoint() -> dict[str, Any]:
    return _checkpoint()


@pytest.fixture
def converted(checkpoint: dict[str, Any]) -> dict[str, WeightData]:
    return convert_kimi_k3_state_dict(checkpoint, _text_config())


def test_key_set_matches_the_module_tree(
    converted: dict[str, WeightData],
) -> None:
    expected = {
        "embed_tokens.weight",
        "lm_head_weight",
        "final_norm_weight",
        "output_attn_res_proj_weight",
        "output_attn_res_norm_weight",
    }
    for i in range(_NUM_HIDDEN_LAYERS):
        dst = f"layers.{i}"
        expected |= {
            f"{dst}.input_layernorm_weight",
            f"{dst}.post_attention_layernorm_weight",
            f"{dst}.self_attention_res_proj_weight",
            f"{dst}.self_attention_res_norm_weight",
            f"{dst}.mlp_res_proj_weight",
            f"{dst}.mlp_res_norm_weight",
        }
        is_kda = i in {j - 1 for j in _KDA_LAYERS}
        attn = f"{dst}.attention"
        if is_kda:
            expected |= {
                f"{attn}.qkv_proj.weight",
                f"{attn}.f_a_proj.weight",
                f"{attn}.f_b_proj.weight",
                f"{attn}.b_proj.weight",
                f"{attn}.o_norm.weight",
                f"{attn}.g_proj.weight",
                f"{attn}.o_proj.weight",
                f"{attn}.conv_weight",
                f"{attn}.a_log",
                f"{attn}.dt_bias",
            }
        else:
            expected |= {
                f"{attn}.q_a_proj.weight",
                f"{attn}.q_a_layernorm.weight",
                f"{attn}.q_b_proj.weight",
                f"{attn}.kv_a_proj_with_mqa.weight",
                f"{attn}.kv_a_layernorm.weight",
                f"{attn}.kv_b_proj.weight",
                f"{attn}.g_proj.weight",
                f"{attn}.o_proj.weight",
            }
        ffn = f"{dst}.ffn"
        if i < _FIRST_K_DENSE_REPLACE:
            expected |= {
                f"{ffn}.gate_up_proj.weight",
                f"{ffn}.down_proj.weight",
            }
        else:
            expected |= {
                f"{ffn}.gate.gate_score.weight",
                f"{ffn}.gate.e_score_correction_bias",
                f"{ffn}.routed_expert_down_proj.weight",
                f"{ffn}.routed_expert_up_proj.weight",
                f"{ffn}.routed_expert_norm.weight",
                f"{ffn}.experts_gate_up_proj",
                f"{ffn}.experts_down_proj",
                f"{ffn}.shared_experts.gate_up_proj.weight",
                f"{ffn}.shared_experts.down_proj.weight",
            }
    assert set(converted) == expected


def test_fused_qkv_proj_matches_model_py(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    """At the loader's own TP degree (1) the adapter's merged ``qkv_proj``
    must agree with ``model.py``'s eager-loader helper."""
    attn_src = f"{_TEXT}.layers.0.self_attn"
    q, k, v = (
        _to_numpy(checkpoint[f"{attn_src}.{axis}_proj.weight"].data())
        for axis in ("q", "k", "v")
    )
    expected = _to_numpy(
        WeightData.from_numpy(km._fuse_qkv_proj(q, k, v), "ref").astype(
            DType.bfloat16
        )
    )
    actual = _to_numpy(converted["layers.0.attention.qkv_proj.weight"])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_fused_qkv_proj_interleaves_per_rank(tp_size: int) -> None:
    """Rank ``r``'s contiguous shard of the merged weight is its slice of
    q, k and v -- not whole projections.

    Parts carry disjoint values, so a plain ``[q; k; v]`` concat is
    distinguishable from the interleave at every TP degree above 1.
    """
    rows, cols = 8, 5
    arrays = [
        _RNG.standard_normal((rows, cols)).astype(np.float32) + 100 * i
        for i in range(3)
    ]
    q, k, v = (
        WeightData.from_numpy(a, axis)
        for a, axis in zip(arrays, ("q", "k", "v"), strict=True)
    )
    fused = wa._fuse_qkv_proj(q, k, v, "qkv_proj.weight", tp_size)
    actual = _to_numpy(fused)
    np.testing.assert_array_equal(actual, interleave_for_tp(arrays, tp_size))
    # The production property, stated directly: each rank's contiguous
    # shard holds its rows of every part, in q, k, v order.
    per_rank = rows // tp_size
    for r in range(tp_size):
        shard = actual[r * 3 * per_rank : (r + 1) * 3 * per_rank]
        for p, part in enumerate(arrays):
            np.testing.assert_array_equal(
                shard[p * per_rank : (p + 1) * per_rank],
                part[r * per_rank : (r + 1) * per_rank],
            )


def test_fused_qkv_proj_spilled_output_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the spill threshold at zero the merged weight is file-backed
    but holds the same bytes as the in-memory one."""
    arrays = [_fp32((8, 5)) + 100 * i for i in range(3)]
    q, k, v = (
        WeightData.from_numpy(a, axis)
        for a, axis in zip(arrays, ("q", "k", "v"), strict=True)
    )
    in_memory = _to_numpy(wa._fuse_qkv_proj(q, k, v, "qkv", 2))
    monkeypatch.setattr(wa, "_SPILL_BYTES", 0)
    spilled = _to_numpy(wa._fuse_qkv_proj(q, k, v, "qkv", 2))
    np.testing.assert_array_equal(spilled, in_memory)


def test_fused_qkv_proj_rejects_mismatched_parts() -> None:
    """A checkpoint whose q/k/v projections disagree in shape or dtype must
    fail loudly, not produce a silently mis-merged weight."""
    q = _weight(_fp32((8, 5)), "q_proj").data()
    k_bad_shape = _weight(_fp32((8, 4)), "k_proj").data()
    v_bad_dtype = _weight(_fp32((8, 5)), "v_proj", DType.bfloat16).data()
    k = _weight(_fp32((8, 5)), "k_proj").data()
    with pytest.raises(ValueError, match="k_proj has dtype"):
        wa._fuse_qkv_proj(q, k_bad_shape, k, "qkv_proj.weight", 2)
    with pytest.raises(ValueError, match="v_proj has dtype"):
        wa._fuse_qkv_proj(q, k, v_bad_dtype, "qkv_proj.weight", 2)


def test_fused_qkv_proj_casts_mixed_float_parts_to_the_model_dtype() -> None:
    """Parts stored in different float dtypes each cast to the model dtype
    before merging, as the final cast pass would; a non-float part still
    fails loudly."""
    arrays = [_fp32((8, 5)) + 100 * i for i in range(3)]
    q = _weight(arrays[0], "q_proj", DType.float32).data()
    k = _weight(arrays[1], "k_proj", DType.bfloat16).data()
    v = _weight(arrays[2], "v_proj", DType.float16).data()
    fused = wa._fuse_qkv_proj(q, k, v, "qkv_proj.weight", 2, DType.float32)
    assert fused.dtype == DType.float32
    expected = interleave_for_tp(
        [_to_numpy(part.astype(DType.float32)) for part in (q, k, v)], 2
    )
    np.testing.assert_array_equal(_to_numpy(fused), expected)

    v_int = _weight_u8(np.zeros((8, 5), np.uint8), "v_proj").data()
    with pytest.raises(ValueError, match="v_proj has dtype"):
        wa._fuse_qkv_proj(q, k, v_int, "qkv_proj.weight", 2, DType.float32)


def test_fused_qkv_conv_matches_model_py(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    attn_src = f"{_TEXT}.layers.0.self_attn"
    q, k, v = (
        _to_numpy(checkpoint[f"{attn_src}.{axis}_conv1d.weight"].data())
        for axis in ("q", "k", "v")
    )
    fused = km._fuse_qkv_conv_weight(q, k, v)
    # `conv_weight` isn't fp32-pinned, so the adapter resolves it to the
    # model dtype (bf16) like the real value is; round the reference the
    # same way, through the same `WeightData.astype` the adapter itself
    # uses, so this compares the fusion math, not bf16 rounding noise.
    expected = _to_numpy(
        WeightData.from_numpy(fused, "ref").astype(DType.bfloat16)
    )
    actual = _to_numpy(converted["layers.0.attention.conv_weight"])
    np.testing.assert_array_equal(actual, expected)


def test_fused_gate_up_matches_model_py(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    gate = _to_numpy(
        checkpoint[f"{_TEXT}.layers.0.mlp.gate_proj.weight"].data()
    )
    up = _to_numpy(checkpoint[f"{_TEXT}.layers.0.mlp.up_proj.weight"].data())
    expected = km._fuse_gate_up(gate, up)
    actual = _to_numpy(converted["layers.0.ffn.gate_up_proj.weight"])
    np.testing.assert_array_equal(actual, expected)


def test_canonical_a_log_matches_model_py(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    raw = _to_numpy(checkpoint[f"{_TEXT}.layers.0.self_attn.A_log"].data())
    expected = km._canonical_a_log(raw, _KDA_NUM_HEADS, _KDA_HEAD_DIM)
    actual = _to_numpy(converted["layers.0.attention.a_log"])
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (_KDA_NUM_HEADS, _KDA_HEAD_DIM)


def test_dt_bias_reshaped_to_heads_by_head_dim(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    raw = _to_numpy(checkpoint[f"{_TEXT}.layers.0.self_attn.dt_bias"].data())
    actual = _to_numpy(converted["layers.0.attention.dt_bias"])
    np.testing.assert_array_equal(
        actual, raw.reshape(_KDA_NUM_HEADS, _KDA_HEAD_DIM)
    )


def test_stacked_experts_match_model_py(
    checkpoint: dict[str, Any], converted: dict[str, WeightData]
) -> None:
    moe = f"{_TEXT}.layers.1.block_sparse_moe"
    w1 = [
        _to_numpy(checkpoint[f"{moe}.experts.{e}.w1.weight"].data())
        for e in range(_NUM_EXPERTS)
    ]
    w2 = [
        _to_numpy(checkpoint[f"{moe}.experts.{e}.w2.weight"].data())
        for e in range(_NUM_EXPERTS)
    ]
    w3 = [
        _to_numpy(checkpoint[f"{moe}.experts.{e}.w3.weight"].data())
        for e in range(_NUM_EXPERTS)
    ]
    expected_gate_up, expected_down = km._stack_expert_weights(w1, w2, w3)
    actual_gate_up = _to_numpy(converted["layers.1.ffn.experts_gate_up_proj"])
    actual_down = _to_numpy(converted["layers.1.ffn.experts_down_proj"])
    np.testing.assert_array_equal(actual_gate_up, expected_gate_up)
    np.testing.assert_array_equal(actual_down, expected_down)
    assert actual_gate_up.shape == (
        _NUM_EXPERTS,
        2 * _MOE_INTERMEDIATE_SIZE,
        _ROUTED_EXPERT_HIDDEN_SIZE,
    )
    assert actual_down.shape == (
        _NUM_EXPERTS,
        _ROUTED_EXPERT_HIDDEN_SIZE,
        _MOE_INTERMEDIATE_SIZE,
    )


def test_routed_entry_and_exit_are_not_swapped(
    converted: dict[str, WeightData],
) -> None:
    """``routed_expert_down_proj`` is the ENTRY (hidden -> latent); a swap
    with ``routed_expert_up_proj`` would show up as a transposed shape."""
    down = converted["layers.1.ffn.routed_expert_down_proj.weight"]
    up = converted["layers.1.ffn.routed_expert_up_proj.weight"]
    assert down.shape == Shape((_ROUTED_EXPERT_HIDDEN_SIZE, _HIDDEN_SIZE))
    assert up.shape == Shape((_HIDDEN_SIZE, _ROUTED_EXPERT_HIDDEN_SIZE))


@pytest.mark.parametrize(
    "key",
    [
        "layers.0.attention.a_log",
        "layers.0.attention.dt_bias",
        "layers.1.ffn.gate.gate_score.weight",
        "layers.1.ffn.gate.e_score_correction_bias",
    ],
)
def test_fp32_pins_are_forced_even_from_bf16(
    converted: dict[str, WeightData], key: str
) -> None:
    """Every pinned tensor is fp32 on output, whatever dtype it arrived in."""
    assert converted[key].dtype == DType.float32


@pytest.mark.parametrize(
    "key",
    [
        "layers.0.attention.b_proj.weight",
        "layers.0.attention.conv_weight",
        "layers.0.attention.o_norm.weight",
    ],
)
def test_unpinned_weights_resolve_to_the_model_dtype(
    converted: dict[str, WeightData], key: str
) -> None:
    """These parameters are only cast to fp32 transiently at compute time
    (see ``kda_attention.py``); the stored parameter itself follows the
    model's working dtype like everything else, so the adapter must resolve
    it there too rather than pin it fp32."""
    assert converted[key].dtype == DType.bfloat16


def test_non_pinned_weights_keep_their_stored_dtype(
    converted: dict[str, WeightData],
) -> None:
    assert converted["layers.0.attention.qkv_proj.weight"].dtype == (
        DType.bfloat16
    )
    assert converted["embed_tokens.weight"].dtype == DType.bfloat16


# Even, so both axes a packed expert splits land on a whole byte and a
# whole 32-element E8M0 scale block.
_PACKED_ROUTED_EXPERT_HIDDEN_SIZE = 8
_PACKED_MOE_INTERMEDIATE_SIZE = 4


@pytest.fixture
def packed_checkpoint() -> dict[str, Any]:
    return _checkpoint(
        routed_expert_hidden_size=_PACKED_ROUTED_EXPERT_HIDDEN_SIZE,
        moe_intermediate_size=_PACKED_MOE_INTERMEDIATE_SIZE,
        packed_experts=True,
    )


@pytest.fixture
def packed_converted(
    packed_checkpoint: dict[str, Any],
) -> dict[str, WeightData]:
    return convert_kimi_k3_state_dict(
        packed_checkpoint,
        _text_config(
            routed_expert_hidden_size=_PACKED_ROUTED_EXPERT_HIDDEN_SIZE,
            moe_intermediate_size=_PACKED_MOE_INTERMEDIATE_SIZE,
        ),
    )


def test_loads_mxfp4_packed_experts(
    packed_checkpoint: dict[str, Any],
    packed_converted: dict[str, WeightData],
) -> None:
    """Packed weights and their E8M0 scales load byte-identical and stacked.

    The packed VALUES are not transformed, only stacked across experts and
    fused gate-then-up -- same mechanism as the dense case -- so every byte
    of the real checkpoint's per-expert tensors must reappear unchanged at
    its stacked position.
    """
    moe = f"{_TEXT}.layers.1.block_sparse_moe"
    gate_up = _to_numpy_u8(
        packed_converted["layers.1.ffn.experts_gate_up_proj"]
    )
    down = _to_numpy_u8(packed_converted["layers.1.ffn.experts_down_proj"])
    gate_up_scale = _to_numpy_u8(
        packed_converted["layers.1.ffn.experts_gate_up_proj_scale"]
    )
    down_scale = _to_numpy_u8(
        packed_converted["layers.1.ffn.experts_down_proj_scale"]
    )

    gate_rows = _PACKED_MOE_INTERMEDIATE_SIZE
    for e in range(_NUM_EXPERTS):
        w1 = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w1.weight_packed"].data()
        )
        w3 = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w3.weight_packed"].data()
        )
        w2 = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w2.weight_packed"].data()
        )
        np.testing.assert_array_equal(gate_up[e, :gate_rows], w1)
        np.testing.assert_array_equal(gate_up[e, gate_rows:], w3)
        np.testing.assert_array_equal(down[e], w2)

        w1_scale = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w1.weight_scale"].data()
        )
        w3_scale = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w3.weight_scale"].data()
        )
        w2_scale = _to_numpy_u8(
            packed_checkpoint[f"{moe}.experts.{e}.w2.weight_scale"].data()
        )
        np.testing.assert_array_equal(gate_up_scale[e, :gate_rows], w1_scale)
        np.testing.assert_array_equal(gate_up_scale[e, gate_rows:], w3_scale)
        np.testing.assert_array_equal(down_scale[e], w2_scale)

    assert gate_up.dtype == np.uint8
    assert down.dtype == np.uint8
    assert (
        packed_converted["layers.1.ffn.experts_gate_up_proj"].dtype
        == DType.uint8
    )
    assert (
        packed_converted["layers.1.ffn.experts_gate_up_proj_scale"].dtype
        == DType.float8_e8m0fnu
    )
    assert (
        packed_converted["layers.1.ffn.experts_down_proj_scale"].dtype
        == DType.float8_e8m0fnu
    )


def test_dense_checkpoint_has_no_quantized_expert_keys(
    converted: dict[str, WeightData],
) -> None:
    """The existing dense (BF16) fixture never produces scale tensors."""
    assert "layers.1.ffn.experts_gate_up_proj_scale" not in (converted)


@pytest.mark.parametrize(
    "tp_size, expected",
    [
        (1, [0, 1, 2, 3, 10, 11, 12, 13]),
        (2, [0, 1, 10, 11, 2, 3, 12, 13]),
        (4, [0, 10, 1, 11, 2, 12, 3, 13]),
    ],
)
def test_interleave_for_tp_groups_rows_per_rank(
    tp_size: int, expected: list[int]
) -> None:
    gate = np.arange(4)[:, None] + np.zeros((1, 3), dtype=np.int64)
    up = gate + 10
    fused = interleave_for_tp([gate, up], tp_size)
    np.testing.assert_array_equal(fused[:, 0], expected)
    assert fused.shape == (8, 3)


def test_interleave_for_tp_three_blocks_on_a_later_axis() -> None:
    q, k, v = (np.arange(4)[None, :] + 10 * b for b in range(3))
    fused = interleave_for_tp([q, k, v], 2, axis=1)
    np.testing.assert_array_equal(
        fused[0], [0, 1, 10, 11, 20, 21, 2, 3, 12, 13, 22, 23]
    )


def test_interleave_for_tp_rejects_a_non_divisible_width() -> None:
    with pytest.raises(ValueError, match="not divisible by the TP degree 4"):
        interleave_for_tp([np.zeros((6, 2)), np.zeros((6, 2))], 4)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_stacked_gate_up_matches_interleave_for_tp(tp_size: int) -> None:
    gate = [_fp32((8, 3)) for _ in range(_NUM_EXPERTS)]
    up = [_fp32((8, 3)) for _ in range(_NUM_EXPERTS)]
    stacked = wa._stack_gate_up_arrays(gate, up, tp_size)
    for e in range(_NUM_EXPERTS):
        np.testing.assert_array_equal(
            stacked[e], interleave_for_tp([gate[e], up[e]], tp_size)
        )


def test_stacked_scales_follow_their_weights_rows() -> None:
    """Packed weights and their E8M0 scales get the same row permutation.

    Each row of both tensors stores its own source tag (expert, block, row),
    so the stacked tensors must agree tag for tag.
    """
    rows, tp_size = 8, 2

    def tagged(block: int, cols: int) -> list[WeightData]:
        return [
            WeightData.from_numpy(
                np.repeat(
                    (e * 64 + block * 16 + np.arange(rows, dtype=np.uint8))[
                        :, None
                    ],
                    cols,
                    axis=1,
                ),
                f"w{block}.{e}",
            )
            for e in range(_NUM_EXPERTS)
        ]

    weight_gate_up, _ = wa._stack_expert_weights(
        tagged(1, 4), tagged(2, 4), tagged(3, 4), "w", tp_size
    )
    scale_gate_up, _ = wa._stack_expert_weights(
        tagged(1, 1), tagged(2, 1), tagged(3, 1), "s", tp_size
    )
    weight_tags = _to_numpy_u8(weight_gate_up)[..., 0]
    scale_tags = _to_numpy_u8(scale_gate_up)[..., 0]
    np.testing.assert_array_equal(weight_tags, scale_tags)
    # Rank 0's shard is gate rows 0-3 then up rows 0-3.
    np.testing.assert_array_equal(
        weight_tags[0, :rows], [16, 17, 18, 19, 48, 49, 50, 51]
    )


def _pipeline_config(
    n_devices: int, ep_size: int = 1, data_parallel_degree: int = 1
) -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(
            device_specs=[object()] * n_devices,
            data_parallel_degree=data_parallel_degree,
        ),
        runtime=SimpleNamespace(ep_size=ep_size),
    )


def test_tp_degree_follows_the_mesh() -> None:
    assert tp_degree(None) == 1
    assert tp_degree(_pipeline_config(8)) == 8
    # An ``ep`` mesh shards nothing along TP.
    assert tp_degree(_pipeline_config(8, ep_size=8)) == 1
    assert tp_degree(_pipeline_config(8, data_parallel_degree=8)) == 1
