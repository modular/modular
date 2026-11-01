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
"""Maps the modelopt MXFP8 Hy4 safetensors onto the ModuleV3 Hy4 names.

Beyond renames, the adapter changes storage in four places:

* Every ``*weight_scale`` / ``*_scale`` (U8 E8M0, one per 32 elements along
  K, ``[.., N, K/32]``) keeps its bits and layout as ``float8_e8m0fnu``, the
  contract of ``MXFP8Tensor``; the block-scaled matmuls interleave them for
  the kernel. FP8 data passes through, routed
  experts included: ``gate_up_proj [E, 2I, H]`` (gate rows then up rows) and
  ``down_proj [E, H, I]`` are already the ``[E, N, K]`` layout the grouped
  matmul takes.
* ``kv_b_proj`` is dequantized to BF16 on the host (MLA absorb has no MXFP8
  path). An e4m3 value times a power of two fits BF16 exactly.
* The indexer's ``wq_b``/``wk`` rows (and their scales) and ``k_norm`` are
  permuted per head so the 64 rope dims come first, as SGLang's
  ``permute_hyv4_indexer_weight`` does; the checkpoint stores them last.
* ``indexer.*`` of a ``shared`` layer is dropped (vLLM's
  ``is_skip_topk_indexer_weight``). The published checkpoints carry none; the
  rule guards a re-upload that adds them.

Any other name raises, so a re-upload that adds a tensor fails at load.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.weights._fp8 import fp8_e4m3fn_to_float32
from transformers.configuration_utils import PretrainedConfig

_MXFP8_BLOCK = 32
_PREFIX = "language_model."
_DRAFT_PREFIX = "draft."

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_MTP_LAYER = re.compile(r"^model\.mtp_layers\.0\.(.+)$")

_MXFP8 = (
    r"self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj"
    r"|indexer\.(?:wq_b|wk))"
    r"|mlp\.(?:shared_experts\.)?(?:gate_proj|up_proj|down_proj)"
)

_Rule = tuple[re.Pattern[str], str, bool]


def _rules(*rules: tuple[str, str, bool]) -> list[_Rule]:
    return [(re.compile(p), r, s) for p, r, s in rules]


# (checkpoint suffix, module suffix, is an E8M0 scale), inside one block.
_BLOCK_RULES = _rules(
    (r"hc_(attn|mlp)_layer\.hc_pre\.hc_(fn|base|scale)", r"hc_\1.\2", False),
    (
        r"(input_layernorm|post_attention_layernorm"
        r"|self_attn\.(?:q_a_layernorm|linear_gate|indexer\.weights_proj"
        r"|indexer\.k_norm))\.weight",
        r"\1.weight",
        False,
    ),
    (
        r"self_attn\.indexer\.k_norm\.bias",
        r"self_attn.indexer.k_norm.bias",
        False,
    ),
    (
        r"self_attn\.learnable_sink_param",
        r"self_attn.learnable_sink_param",
        False,
    ),
    (
        r"self_attn\.kv_a_layernorm\.weight",
        "self_attn.kv_a_proj_layernorm",
        False,
    ),
    (rf"({_MXFP8})\.weight", r"\1.weight", False),
    (rf"({_MXFP8})\.weight_scale", r"\1.weight_scale", True),
    (r"mlp\.gate\.weight", "mlp.gate.gate_score.weight", False),
    (
        r"mlp\.gate\.e_score_correction_bias",
        r"mlp.gate.e_score_correction_bias",
        False,
    ),
    (
        r"mlp\.experts\.(gate_up_proj|down_proj)",
        r"mlp.experts.\1.weight",
        False,
    ),
    (
        r"mlp\.experts\.(gate_up_proj|down_proj)_scale",
        r"mlp.experts.\1.weight_scale",
        True,
    ),
)

_MTP_RULES = _rules(
    (r"(enorm|hnorm|eh_proj|final_layernorm)\.weight", r"\1.weight", False),
)

_TOP_RULES = _rules(
    (r"model\.(embed_tokens|norm)\.weight", r"\1.weight", False),
    (r"lm_head\.weight", "lm_head.weight", False),
    (r"model\.hc_head\.hc_head_(fn|base|scale)", r"hc_head.hc_\1", False),
)

_KV_B = "self_attn.kv_b_proj"
_INDEXER = re.compile(r"self_attn\.indexer\.(wq_b|wk|k_norm)\.")
_UNSIGNED = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32}


def e8m0_to_float32(raw: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """``2 ** (e - 127)``, exact in float32 for every E8M0 code but 255."""
    # ldexp, not exp2: numpy's float32 exp2 is off by an ulp on x86.
    return np.ldexp(np.float32(1.0), raw.astype(np.int32) - 127)


def _bits(weight: WeightData) -> np.ndarray:
    """The tensor's raw bits as an unsigned integer array of its shape."""
    buffer = weight.to_buffer()
    return np.from_dlpack(buffer.view(_UNSIGNED[weight.dtype.size_in_bytes]))


def _from_bits(
    bits: npt.NDArray[np.generic], dtype: DType, name: str
) -> WeightData:
    buffer = Buffer.from_numpy(np.ascontiguousarray(bits))
    if dtype != DType.from_numpy(bits.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(buffer.shape))


def _e8m0_scale(scale: WeightData, name: str) -> WeightData:
    return _from_bits(_bits(scale), DType.float8_e8m0fnu, name)


def _dequantize_kv_b(
    weight: WeightData, scale: WeightData, name: str
) -> WeightData:
    """MXFP8 ``[N, K]`` with E8M0 ``[N, K/32]`` to BF16."""
    values = fp8_e4m3fn_to_float32(weight)
    scales = np.repeat(e8m0_to_float32(_bits(scale)), _MXFP8_BLOCK, axis=1)
    product = np.ascontiguousarray(values * scales, dtype=np.float32)
    return _from_bits(
        (product.view(np.uint32) >> 16).astype(np.uint16), DType.bfloat16, name
    )


def _rope_first(
    weight: WeightData, groups: int, config: PretrainedConfig
) -> WeightData:
    """Moves each head's last ``qk_rope_head_dim`` rows to the front."""
    bits = _bits(weight)
    heads = bits.reshape(groups, config.index_head_dim, *bits.shape[1:])
    rope = config.qk_rope_head_dim
    permuted = np.concatenate((heads[:, -rope:], heads[:, :-rope]), axis=1)
    return _from_bits(permuted.reshape(bits.shape), weight.dtype, weight.name)


def _match(rules: list[_Rule], suffix: str) -> tuple[str, bool] | None:
    for pattern, replacement, is_scale in rules:
        if pattern.fullmatch(suffix):
            return pattern.sub(replacement, suffix), is_scale
    return None


class _BlockConverter:
    """Converts the tensors of one decoder block (a main layer or the MTP)."""

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config
        self.kv_b: dict[str, dict[str, WeightData]] = {}

    def convert(
        self,
        key: str,
        prefix: str,
        suffix: str,
        data: WeightData,
        rules: list[_Rule],
    ) -> dict[str, WeightData]:
        if suffix in (f"{_KV_B}.weight", f"{_KV_B}.weight_scale"):
            parts = self.kv_b.setdefault(prefix, {})
            parts[suffix.rsplit(".", 1)[1]] = data
            if len(parts) < 2:
                return {}
            name = f"{prefix}{_KV_B}"
            return {
                name: _dequantize_kv_b(
                    parts["weight"], parts["weight_scale"], name
                )
            }

        matched = _match(rules, suffix)
        if matched is None:
            raise ValueError(
                f"Hy4 weight adapter: unexpected checkpoint tensor {key!r}"
            )
        target, is_scale = matched
        name = prefix + target
        indexer = _INDEXER.match(suffix)
        if indexer is not None:
            groups = self.config.index_n_heads if indexer[1] == "wq_b" else 1
            data = _rope_first(data, groups, self.config)
        if is_scale:
            return {name: _e8m0_scale(data, name)}
        return {name: dataclasses.replace(data, name=name)}

    def check_done(self) -> None:
        for prefix, parts in self.kv_b.items():
            if len(parts) < 2:
                raise ValueError(
                    f"Hy4 weight adapter: {prefix}{_KV_B} lacks a weight or scale"
                )


def _convert(
    state_dict: Mapping[str, Weights],
    config: PretrainedConfig,
    route: Callable[[str], tuple[str, str, list[_Rule]] | None],
) -> dict[str, WeightData]:
    converter = _BlockConverter(config)
    out: dict[str, WeightData] = {}
    for key, value in state_dict.items():
        routed = route(key)
        if routed is None:
            continue
        prefix, suffix, rules = routed
        out.update(converter.convert(key, prefix, suffix, value.data(), rules))
    converter.check_done()
    return out


def convert_safetensor_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Adapts the target model's tensors; ``model.mtp_layers.*`` is skipped."""
    shared = {
        i
        for i, kind in enumerate(huggingface_config.indexer_types)
        if kind == "shared"
    }

    def route(key: str) -> tuple[str, str, list[_Rule]] | None:
        if key.startswith("model.mtp_layers."):
            return None
        layer = _LAYER.match(key)
        if layer is None:
            return _PREFIX, key, _TOP_RULES
        if ".indexer." in key and int(layer[1]) in shared:
            return None
        return f"{_PREFIX}layers.{layer[1]}.", layer[2], _BLOCK_RULES

    return _convert(state_dict, huggingface_config, route)


def convert_mtp_state_dict(
    state_dict: Mapping[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Adapts ``model.mtp_layers.0.*`` to the draft's ``draft.*`` names.

    Only the MTP layer's own tensors are returned; the draft shares the
    target's ``embed_tokens`` and ``lm_head``. Its ``learnable_sink_param``
    stays BF16 as stored.
    """

    def route(key: str) -> tuple[str, str, list[_Rule]] | None:
        mtp = _MTP_LAYER.match(key)
        if mtp is None:
            if key.startswith("model.mtp_layers."):
                raise ValueError(
                    f"Hy4 weight adapter: unexpected checkpoint tensor {key!r}"
                )
            return None
        return _DRAFT_PREFIX, mtp[1], _BLOCK_RULES + _MTP_RULES

    return _convert(state_dict, huggingface_config, route)
