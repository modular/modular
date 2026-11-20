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

"""Name-remapping checks for the DeepseekV3 ModuleV3 weight adapters.

The quantized layers name their parameters as the checkpoint does (see
``max/tests/tests/nn/module_v3/quant_layers``), so the adapter remaps prefixes,
drops what MAX doesn't load, and strips ``.weight`` only from bf16 raw-tensor
MLA projections; the assertions here are the parameter names the layers
expose.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from max.dtype import DType
from max.graph.weights import Weights
from max.pipelines.architectures.deepseekV3_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)

_NUM_LAYERS = 1
# Layer index _NUM_LAYERS is the MTP layer, which the adapter drops.
_MTP_LAYER = _NUM_LAYERS


def _weight(dtype: DType) -> Any:
    """A stand-in ``Weights`` whose ``data()`` reports only its dtype."""
    weights = MagicMock(spec=Weights)
    weights.data.return_value = SimpleNamespace(dtype=dtype)
    return weights


def _convert(state_dict: dict[str, Any]) -> dict[str, Any]:
    return convert_safetensor_state_dict(
        state_dict, DeepseekV3Config(num_hidden_layers=_NUM_LAYERS)
    )


def test_nvfp4_projection_keeps_the_checkpoint_names() -> None:
    """Each modelopt entry keeps its name, which ``NVFP4Tensor`` exposes."""
    converted = _convert(
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.uint8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.mlp.experts.0.gate_proj.input_scale": _weight(
                DType.float32
            ),
        }
    )

    proj = "language_model.layers.0.mlp.experts.0.gate_proj"
    assert set(converted) == {
        f"{proj}.weight",
        f"{proj}.weight_scale",
        f"{proj}.weight_scale_2",
        f"{proj}.input_scale",
    }


def test_nvfp4_quantizes_o_proj_but_not_the_latent_projections() -> None:
    """In an NVFP4 checkpoint the MLA q/kv projections stay bf16 raw tensors."""
    converted = _convert(
        {
            "model.layers.0.self_attn.o_proj.weight": _weight(DType.uint8),
            "model.layers.0.self_attn.o_proj.weight_scale": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.self_attn.o_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.self_attn.q_a_proj.weight": _weight(DType.bfloat16),
            "model.layers.0.self_attn.kv_b_proj.weight": _weight(
                DType.bfloat16
            ),
        }
    )

    assert set(converted) == {
        "language_model.layers.0.self_attn.o_proj.weight",
        "language_model.layers.0.self_attn.o_proj.weight_scale",
        "language_model.layers.0.self_attn.o_proj.weight_scale_2",
        # Raw-tensor MLA projections drop the ``.weight`` suffix entirely.
        "language_model.layers.0.self_attn.q_a_proj",
        "language_model.layers.0.self_attn.kv_b_proj",
    }


def test_nvfp4_applies_the_shared_bf16_renames() -> None:
    """The cross-cutting V3 renames still run on an NVFP4 checkpoint."""
    converted = _convert(
        {
            "model.layers.0.mlp.gate.weight_scale_2": _weight(DType.float32),
            "model.layers.0.mlp.gate.weight": _weight(DType.bfloat16),
            "model.layers.0.self_attn.kv_a_layernorm.weight": _weight(
                DType.bfloat16
            ),
            "model.layers.0.input_layernorm.weight": _weight(DType.bfloat16),
            "model.embed_tokens.weight": _weight(DType.bfloat16),
            "lm_head.weight": _weight(DType.bfloat16),
        }
    )

    assert "language_model.layers.0.mlp.gate.gate_score.weight" in converted
    assert "language_model.layers.0.self_attn.kv_a_proj_layernorm" in converted
    assert "language_model.layers.0.input_layernorm.weight" in converted
    assert "language_model.embed_tokens.weight" in converted
    assert "language_model.lm_head.weight" in converted


def test_nvfp4_drops_kv_cache_scales_and_mtp_layer() -> None:
    """modelopt emits KV-cache scales and an MTP layer that MAX doesn't load.

    An MTP layer kept in FP8 doesn't make the checkpoint FP8: the bf16 MLA
    projections still drop ``.weight``.
    """
    converted = _convert(
        {
            "model.layers.0.self_attn.o_proj.weight_scale_2": _weight(
                DType.float32
            ),
            "model.layers.0.self_attn.q_a_proj.weight": _weight(DType.bfloat16),
            "model.layers.0.self_attn.k_proj.k_scale": _weight(DType.float32),
            "model.layers.0.self_attn.v_proj.v_scale": _weight(DType.float32),
            f"model.layers.{_MTP_LAYER}.mlp.experts.0.gate_proj.weight": _weight(
                DType.uint8
            ),
            f"model.layers.{_MTP_LAYER}.input_layernorm.weight": _weight(
                DType.bfloat16
            ),
            f"model.layers.{_MTP_LAYER}.self_attn.q_a_proj.weight_scale_inv": (
                _weight(DType.float32)
            ),
        }
    )

    assert set(converted) == {
        "language_model.layers.0.self_attn.o_proj.weight_scale_2",
        "language_model.layers.0.self_attn.q_a_proj",
    }


def test_fp8_latent_projections_keep_the_checkpoint_names() -> None:
    """An FP8 checkpoint quantizes the MLA projections, so they keep ``.weight``.

    A bf16 latent projection is a plain tensor at ``<proj>`` and drops the
    suffix; a quantized one keys its packed data ``weight``, beside
    ``weight_scale_inv``, as an expert projection does.
    """
    fp8 = _convert(
        {
            "model.layers.0.self_attn.q_a_proj.weight": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": _weight(
                DType.float32
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.float8_e4m3fn
            ),
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": _weight(
                DType.float32
            ),
        }
    )
    attn = "language_model.layers.0.self_attn.q_a_proj"
    expert = "language_model.layers.0.mlp.experts.0.gate_proj"
    assert set(fp8) == {
        f"{attn}.weight",
        f"{attn}.weight_scale_inv",
        f"{expert}.weight",
        f"{expert}.weight_scale_inv",
    }

    bf16 = _convert(
        {
            "model.layers.0.self_attn.q_a_proj.weight": _weight(DType.bfloat16),
            "model.layers.0.mlp.experts.0.gate_proj.weight": _weight(
                DType.bfloat16
            ),
        }
    )
    assert set(bf16) == {attn, f"{expert}.weight"}
