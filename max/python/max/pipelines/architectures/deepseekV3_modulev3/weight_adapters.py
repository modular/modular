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

from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

# MLA projections that are stored as raw ``FP8BlockTensor`` parameters
# (not wrapped in a ``QuantizedLinear``). Used to decide whether the FP8
# adapter should emit ``<proj>.data`` / ``<proj>.scale_inv`` versus
# ``<proj>.weight.data`` / ``<proj>.weight.scale_inv``.
_MLA_RAW_PROJECTIONS: frozenset[str] = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "q_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    }
)

# Projections backed by ``QuantizedLinear`` (one inner ``weight`` attribute
# that's itself an :class:`FP8BlockTensor` module).
_FP8_LINEAR_PROJECTIONS: frozenset[str] = frozenset(
    {"gate_proj", "up_proj", "down_proj", "o_proj"}
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

    Only fires for the bf16 (non-FP8) path; the FP8 path handles the
    ``.weight`` / ``.weight_scale_inv`` rewrites explicitly so it can
    distinguish the two.
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
        if max_name.endswith(".k_scale") or max_name.endswith(".v_scale"):
            continue
        out[max_name] = value.data()
    return out


def _convert_fp8_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Adapt an FP8 block-scaled DeepseekV3 checkpoint.

    For each FP8 weight, both the packed data and the per-block inverse
    scale need to land on the matching :class:`FP8BlockTensor` module
    parameters:

    * MLA raw-tensor projections (stored directly on the attention
      module, not wrapped in a Linear)::

          self_attn.q_a_proj.weight          → self_attn.q_a_proj.data
          self_attn.q_a_proj.weight_scale_inv → self_attn.q_a_proj.scale_inv

    * Linear-backed projections (QuantizedLinear → weight: FP8BlockTensor)::

          mlp.gate_proj.weight               → mlp.gate_proj.weight.data
          mlp.gate_proj.weight_scale_inv     → mlp.gate_proj.weight.scale_inv
          self_attn.o_proj.weight            → self_attn.o_proj.weight.data
          ...
    """
    out: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # Detect quantized weight / scale before remapping suffixes.
        is_scale = name.endswith(".weight_scale_inv")
        if is_scale:
            base = name[: -len(".weight_scale_inv")]
        elif name.endswith(".weight"):
            base = name[: -len(".weight")]
        else:
            base = None

        if base is None:
            # Non-quantized weight (e.g. layernorm, norm gamma, embedding).
            max_name = _remap_bf16_prefix(name)
            max_name = _strip_weight_suffix_for_raw_mla(max_name)
        else:
            base = _remap_bf16_prefix(base)
            tail = base.rsplit(".", 1)[-1]
            if tail in _MLA_RAW_PROJECTIONS:
                # Raw-tensor MLA: drop ``.weight`` entirely, then attach
                # ``.data`` / ``.scale_inv`` for the FP8BlockTensor params.
                inner = "scale_inv" if is_scale else "data"
                max_name = f"{base}.{inner}"
            elif tail in _FP8_LINEAR_PROJECTIONS:
                inner = "scale_inv" if is_scale else "data"
                max_name = f"{base}.weight.{inner}"
            elif not is_scale:
                # ``.weight`` entry that's not FP8-quantized: layernorm
                # gamma, q_a_layernorm, kv_a_layernorm, router gate
                # weight, embedding, lm_head, etc. These stay bf16 in
                # FP8 checkpoints. Run the full bf16 prefix remap on the
                # original full name (this is safe because we already
                # excluded ``.weight_scale_inv`` from the bf16 path).
                max_name = _remap_bf16_prefix(name)
            else:
                # Unrecognized scale-bearing weight; surface a clear error
                # by routing to a debug path so the load layer can report.
                max_name = f"{base}.scale_inv"

        if max_name.endswith(".k_scale") or max_name.endswith(".v_scale"):
            continue
        out[max_name] = value.data()
    return out


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Remap an HF safetensors state dict to DeepseekV3 ModuleV3 names.

    Dispatches to the FP8 block-scaled path when the input contains any
    ``*.weight_scale_inv`` entry; otherwise falls back to the bf16-only
    remap (raw-tensor MLA projections, language_model prefix, lm_head).
    """
    is_fp8_blockscaled = any(
        name.endswith(".weight_scale_inv") for name in state_dict
    )

    if is_fp8_blockscaled:
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
