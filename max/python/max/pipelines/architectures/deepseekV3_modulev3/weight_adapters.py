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

from __future__ import annotations

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# MLA projections the attention module holds raw, not in a
# ``QuantizedLinear``. A bf16 one is a plain tensor at ``<proj>``, so its
# checkpoint ``<proj>.weight`` drops the suffix; a quantized one keys its
# packed data ``weight`` and keeps the checkpoint name.
_MLA_RAW_PROJECTIONS: frozenset[str] = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "q_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    }
)


def _remap_bf16_prefix(name: str) -> str:
    """Apply the cross-cutting V3 path remap (HF prefix → V3 attribute path).

    These rules are safe to run on both bf16 and FP8 entries: they only
    affect the leading ``model.`` / ``lm_head.`` prefixes and the
    ``mlp.gate.weight`` → ``mlp.gate.gate_score.weight`` rename. They do
    not strip any trailing ``.weight`` suffixes (which would collide with
    ``.weight_scale_inv``).
    """
    out = name
    out = out.replace("mlp.gate.weight", "mlp.gate.gate_score.weight")
    out = out.replace(
        "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_layernorm"
    )
    if out.startswith("model."):
        out = "language_model." + out[len("model.") :]
    elif out.startswith("lm_head."):
        out = "language_model.lm_head." + out[len("lm_head.") :]
    return out


def _strip_weight_suffix_for_raw_mla(name: str) -> str:
    """Strip ``.weight`` from raw-tensor MLA projection names.

    Only for checkpoints whose MLA projections are bf16 (bf16 and NVFP4);
    an FP8 checkpoint quantizes them, and a quantized projection keeps its
    checkpoint name.
    """
    for proj in _MLA_RAW_PROJECTIONS:
        suffix = f"self_attn.{proj}.weight"
        if name.endswith(suffix):
            return name[: -len(".weight")]
    return name


def _convert_bf16_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Adapt a bf16 DeepseekV3 checkpoint to the V3 module hierarchy."""
    out: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        max_name = _remap_bf16_prefix(name)
        max_name = _strip_weight_suffix_for_raw_mla(max_name)
        # Drop FP8 KV-cache static scales emitted by modelopt NVFP4
        # checkpoints (e.g. `k_proj.k_scale`, `v_proj.v_scale`). MAX reads
        # KV cache scales from a separate configuration path, so these
        # keys would otherwise trigger a strict load_state_dict failure.
        if max_name.endswith((".k_scale", ".v_scale")):
            continue
        out[max_name] = value.data()
    return out


def _convert_fp8_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Adapt an FP8 block-scaled DeepseekV3 checkpoint.

    Every quantized projection keeps its checkpoint names: an
    :class:`FP8BlockTensor` keys its packed data ``weight`` beside
    ``weight_scale_inv``, whether a ``QuantizedLinear`` holds it or the MLA
    module holds it raw, so only the bf16 prefix remap applies.
    """
    out: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        max_name = _remap_bf16_prefix(name)
        if max_name.endswith((".k_scale", ".v_scale")):
            continue
        data = value.data()
        # DeepSeek-V3.1-Terminus stores norm gammas (``model.norm.weight``,
        # the MLA ``kv_a_layernorm.weight``) in float32; the module declares
        # them bf16 and the ModuleV3 loader validates dtypes strictly.
        if data.dtype == DType.float32 and name.endswith("norm.weight"):
            data = data.astype(DType.bfloat16)
        out[max_name] = data
    return out


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Remap an HF safetensors state dict to DeepseekV3 ModuleV3 names.

    Dispatches to the FP8 block-scaled path on any ``*.weight_scale_inv``
    entry, unless a ``*.weight_scale_2`` entry marks the checkpoint NVFP4;
    otherwise applies the bf16 remap (raw-tensor MLA projections,
    language_model prefix, lm_head), which also serves NVFP4 checkpoints:
    their quantized projections keep the checkpoint names and their MLA
    projections are bf16, even when an FP8-kept layer such as the MTP
    layer brings ``*.weight_scale_inv`` entries along.
    """
    is_nvfp4 = any(name.endswith(".weight_scale_2") for name in state_dict)
    is_fp8_blockscaled = any(
        name.endswith(".weight_scale_inv") for name in state_dict
    )

    if is_fp8_blockscaled and not is_nvfp4:
        new_state_dict = _convert_fp8_state_dict(state_dict)
    else:
        new_state_dict = _convert_bf16_state_dict(state_dict)

    # TODO(E2EOPT-673): Support MTP. We currently delete the MTP weights.
    # This is also done in the official DeepSeek HF checkpoint converter:
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/4592be48c07f036b32ef971474068aebc489e3e7/inference/convert.py#L53-L54
    mtp_layer_idx = huggingface_config.num_hidden_layers
    mtp_prefix = f"language_model.layers.{mtp_layer_idx}."
    for key in list(new_state_dict.keys()):
        if key.startswith(mtp_prefix):
            del new_state_dict[key]
    return new_state_dict
