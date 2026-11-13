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
"""The MAX-side adapter applies the contract's names, shapes and skips.

`test_weight_contract.py` checks the table; this file checks that
``convert_safetensor_state_dict`` produces exactly that table from a state dict,
including the FP8 block-scale reshape actually carrying the stored values.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.mistral_large4_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
    is_skipped_checkpoint_tensor,
)
from transformers import PretrainedConfig


class _Weight:
    """The one surface the adapter reads: ``data()``."""

    def __init__(self, data: WeightData) -> None:
        self._data = data

    def data(self) -> WeightData:
        """Returns the weight's data and metadata."""
        return self._data


def _labelled(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: DType,
    values: np.ndarray | None = None,
) -> WeightData:
    """A ``WeightData`` whose storage width matches ``dtype``'s."""
    width = (
        4 if dtype == DType.float32 else (2 if dtype == DType.bfloat16 else 1)
    )
    raw = np.zeros(shape, dtype=np.uint8 if width == 1 else np.uint32)
    if values is not None:
        raw = np.ascontiguousarray(values).view(
            np.float32 if dtype == DType.float32 else np.uint32
        )
    return WeightData(data=raw, name=name, dtype=dtype, shape=Shape(shape))


def _state_dict() -> dict[str, Weights]:
    """The state dict under test, in the ``Weights`` shape the adapter reads.

    ``_Weight`` carries only ``data()``, one of the seven members the
    ``Weights`` protocol declares, so the dict needs a cast to type-check and an
    ``isinstance`` check would fail for the same reason.
    """
    scale = np.arange(4, dtype=np.float32).reshape(2, 1, 2, 1)
    return cast(
        dict[str, Weights],
        {
            "layers.0.attention.wq.weight": _Weight(
                _labelled(
                    "layers.0.attention.wq.weight",
                    (256, 256),
                    dtype=DType.float8_e4m3fn,
                )
            ),
            "layers.0.attention.wq.weight_scale": _Weight(
                _labelled(
                    "layers.0.attention.wq.weight_scale",
                    (2, 1, 2, 1),
                    dtype=DType.float32,
                    values=scale,
                )
            ),
            "layers.1.experts.0.w1.weight": _Weight(
                _labelled(
                    "layers.1.experts.0.w1.weight", (4, 8), dtype=DType.uint8
                )
            ),
            "layers.1.experts.0.w1.weight_scale": _Weight(
                _labelled(
                    "layers.1.experts.0.w1.weight_scale",
                    (4, 1),
                    dtype=DType.float8_e4m3fn,
                )
            ),
            "layers.1.experts.0.w1.weight_scale_2": _Weight(
                _labelled(
                    "layers.1.experts.0.w1.weight_scale_2",
                    (),
                    dtype=DType.float32,
                )
            ),
            "layers.1.experts.0.w1.input_scale": _Weight(
                _labelled(
                    "layers.1.experts.0.w1.input_scale",
                    (),
                    dtype=DType.float32,
                )
            ),
            "layers.0.attention_norm.weight": _Weight(
                _labelled(
                    "layers.0.attention_norm.weight", (8,), dtype=DType.bfloat16
                )
            ),
            "layers.1.gate.weight": _Weight(
                _labelled("layers.1.gate.weight", (2, 8), dtype=DType.bfloat16)
            ),
            "output.weight": _Weight(
                _labelled("output.weight", (8, 8), dtype=DType.bfloat16)
            ),
            "lerope.alpha": _Weight(
                _labelled("lerope.alpha", (4,), dtype=DType.bfloat16)
            ),
            "mtp_dspark.backbone.target_projection.weight": _Weight(
                _labelled(
                    "mtp_dspark.backbone.target_projection.weight",
                    (8, 8),
                    dtype=DType.bfloat16,
                )
            ),
            "vision_encoder.patch_conv.weight": _Weight(
                _labelled(
                    "vision_encoder.patch_conv.weight",
                    (2, 3, 2, 2),
                    dtype=DType.bfloat16,
                )
            ),
        },
    )


def test_the_adapter_renames_and_reshapes_exactly_as_the_contract_says() -> (
    None
):
    adapted = convert_safetensor_state_dict(_state_dict(), PretrainedConfig())
    assert set(adapted) == {
        "language_model.layers.0.self_attn.wq.weight",
        "language_model.layers.0.self_attn.wq.weight_scale_inv",
        "language_model.layers.1.mlp.experts.0.w1.weight",
        "language_model.layers.1.mlp.experts.0.w1.weight_scale",
        "language_model.layers.1.mlp.experts.0.w1.weight_scale_2",
        "language_model.layers.1.mlp.experts.0.w1.input_scale",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.1.mlp.gate.gate_score.weight",
        "language_model.lm_head.weight",
        "language_model.lerope.alpha",
        "vision_encoder.patch_conv",
    }, "the drafter tensors are dropped; the vision ones bind by name"

    scale = adapted["language_model.layers.0.self_attn.wq.weight_scale_inv"]
    assert tuple(scale.shape) == (2, 2)
    assert scale.dtype == DType.float32
    # The reshape preserves element order: [rows/128, 1, cols/128, 1] reads out
    # as the [rows, cols] block grid.
    expected = np.arange(4, dtype=np.float32).reshape(2, 2)
    np.testing.assert_array_equal(np.from_dlpack(scale), expected)


def test_the_adapter_keeps_nvfp4_tensor_names_and_shapes() -> None:
    adapted = convert_safetensor_state_dict(_state_dict(), PretrainedConfig())
    group_scale = adapted[
        "language_model.layers.1.mlp.experts.0.w1.weight_scale"
    ]
    assert tuple(group_scale.shape) == (4, 1), "NVFP4 group scales stay 2-D"
    assert group_scale.dtype == DType.float8_e4m3fn
    # A per-tensor global keeps its scalar shape, so ``NVFP4Tensor`` can read it.
    assert (
        tuple(
            adapted[
                "language_model.layers.1.mlp.experts.0.w1.weight_scale_2"
            ].shape
        )
        == ()
    )


def test_the_adapter_refuses_the_base_checkpoints_scale_convention() -> None:
    state = _state_dict()
    state["layers.0.attention.wq.qscale_weight"] = cast(
        Weights,
        _Weight(
            _labelled(
                "layers.0.attention.wq.qscale_weight",
                (2, 2),
                dtype=DType.float32,
            )
        ),
    )
    with pytest.raises(ValueError, match="qscale_weight"):
        convert_safetensor_state_dict(state, PretrainedConfig())


def test_the_skip_predicate_agrees_with_the_reasons() -> None:
    assert is_skipped_checkpoint_tensor(
        "mtp_dspark.backbone.target_norm.weight"
    )
    assert not is_skipped_checkpoint_tensor("vision_encoder.ln_pre.weight")
    assert not is_skipped_checkpoint_tensor("layers.0.attention.wq.weight")


def test_the_adapter_routes_the_vision_subtree_by_name() -> None:
    """The vision weights survive the adapter, under the vision names.

    The conv patch embed is the one rename; everything else keeps its
    checkpoint name, which is what lets the vision graph bind without a
    rename table of its own.
    """
    adapted = convert_safetensor_state_dict(_state_dict(), PretrainedConfig())
    patch_conv = adapted["vision_encoder.patch_conv"]
    assert patch_conv.dtype == DType.bfloat16
    assert tuple(patch_conv.shape) == (2, 3, 2, 2)
    assert patch_conv.name == "vision_encoder.patch_conv"


def test_the_adapter_refuses_a_quantized_vision_tensor() -> None:
    """A packed vision weight never reaches the BF16 vision graph."""
    state = _state_dict()
    state["vision_encoder.ln_pre.weight"] = cast(
        Weights,
        _Weight(
            _labelled(
                "vision_encoder.ln_pre.weight",
                (8,),
                dtype=DType.uint8,
            )
        ),
    )
    with pytest.raises(ValueError, match="unquantized BF16"):
        convert_safetensor_state_dict(state, PretrainedConfig())


def test_the_adapter_permutes_vision_qk_rows_into_the_rotate_half_layout() -> (
    None
):
    """The tower's Q/K rows arrive interleaved-RoPE and load rotate-half.

    The checkpoint stores each head's rope rows as adjacent pairs
    ``(x0, x1), (x2, x3), ...`` sharing one frequency; the tower applies
    rotate-half rope, so the adapter moves each head's even rows ahead of
    its odd ones -- transformers' ``PermuteForRope`` vision rule. Row labels
    are bf16-encoded small integers, so the permutation reads off the
    adapted bits directly.
    """
    heads, head_dim, hidden = 16, 128, 6
    rows = heads * head_dim
    labels = np.arange(rows, dtype=np.float32)
    bits = np.right_shift(labels.view(np.uint32), np.uint32(16)).astype(
        np.uint16
    )
    raw = np.ascontiguousarray(np.repeat(bits[:, None], hidden, axis=1))
    name = "vision_encoder.transformer.layers.3.attention.wq.weight"
    state = _state_dict()
    state[name] = cast(
        Weights,
        _Weight(
            WeightData(
                data=raw,
                name=name,
                dtype=DType.bfloat16,
                shape=Shape((rows, hidden)),
            )
        ),
    )

    adapted = convert_safetensor_state_dict(state, PretrainedConfig())
    assert name in adapted
    weight = adapted[name]
    assert weight.dtype == DType.bfloat16
    assert tuple(weight.shape) == (rows, hidden)
    got = np.from_dlpack(Buffer.from_dlpack(weight.data).view(DType.uint16))

    # The expected map is written from the row labels, not from the
    # implementation's reshape: adapted row h*128 + p*64 + i (head h,
    # rotate-half half p, index i within it) holds the checkpoint's row
    # h*128 + 2*i + p. An inverted permutation fails this.
    adapted_row = np.arange(rows)
    head, rest = np.divmod(adapted_row, head_dim)
    half, index = np.divmod(rest, head_dim // 2)
    checkpoint_row = head * head_dim + 2 * index + half
    np.testing.assert_array_equal(got, raw[checkpoint_row])
    # Spot checks, spelled out: the first row stays put, and the
    # checkpoint's second row lands behind the first half-head of head 0
    # and of head 5.
    np.testing.assert_array_equal(got[0], raw[0])
    np.testing.assert_array_equal(got[64], raw[1])
    np.testing.assert_array_equal(got[5 * 128 + 64], raw[5 * 128 + 1])
