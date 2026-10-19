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
"""Weight adapters for Gemma4 ModuleV3: reuse the graph arch's converters."""

from __future__ import annotations

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

# Packed NVFP4 weight data dtypes (``float4_e2m1fnx2`` is stored as uint8).
_FP4_WEIGHT_DTYPES = frozenset({DType.uint8, DType.float4_e2m1fn})

# Per-projection NVFP4 scale tensors, mapped onto the ``NVFP4Tensor`` leaves
# of the backing ``QuantizedLinear.weight``.
_NVFP4_SCALE_SUFFIXES: tuple[str, ...] = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
)


def _convert_nvfp4_language_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Remap bare NVFP4 projection entries onto ``NVFP4Tensor`` leaves.

    The graph converter's output names the quantized projections bare
    (``layers.0.mlp.gate_proj.weight``, ``...weight_scale``, ...); the
    ModuleV3 tree nests each projection's packed data and scales under a
    single ``weight`` attribute::

        <proj>.weight         (uint8)    -> <proj>.weight.data
        <proj>.weight_scale   (f8_e4m3)  -> <proj>.weight.weight_scale
        <proj>.weight_scale_2 (fp32)     -> <proj>.weight.weight_scale_2
        <proj>.input_scale    (fp32)     -> <proj>.weight.input_scale

    The checkpoint stores ``weight_scale`` as ``F8_E4M3`` in its safetensors
    header, matching ``NVFP4Tensor.weight_scale``'s declared dtype exactly,
    so the scale entries pass through without a cast.
    """
    out: dict[str, WeightData] = {}
    for name, data in state_dict.items():
        scale_suffix = next(
            (s for s in _NVFP4_SCALE_SUFFIXES if name.endswith(s)), None
        )
        if scale_suffix is not None:
            out[f"{name[: -len(scale_suffix)]}.weight.{scale_suffix[1:]}"] = (
                data
            )
        elif name.endswith(".weight") and data.dtype in _FP4_WEIGHT_DTYPES:
            out[f"{name[: -len('.weight')]}.weight.data"] = data
        else:
            out[name] = data
    return out


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Full-checkpoint adapter used by the registry: language + vision keys,
    each re-prefixed to match the eager module trees.

    The two slices never collide: the language converter only accepts
    ``(model.)language_model.*`` keys and the vision converter only accepts
    ``model.vision_tower.*`` / ``model.embed_vision.*``.
    """
    converted = convert_language_state_dict_for_module(state_dict)
    converted.update(convert_vision_state_dict_for_module(state_dict))
    return converted


def convert_language_state_dict_for_module(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Language slice, re-prefixed under ``language_model.``.

    The graph arch's converter filters checkpoint keys down to
    `model.language_model.*` / `language_model.*` and strips those prefixes
    (the graph arch's module tree consumes bare, unprefixed parameter paths).
    Our ModuleV3 module tree nests everything under `language_model`, so
    re-add the prefix here to match the module's parameter paths.

    NVFP4 checkpoints (any ``*.weight_scale_2`` entry) additionally get
    their packed weights and per-projection scales remapped onto the
    ``NVFP4Tensor`` leaves before the prefix is re-added.
    """
    language = convert_safetensor_language_state_dict(state_dict)
    is_nvfp4 = any(name.endswith(".weight_scale_2") for name in language)
    if is_nvfp4:
        language = _convert_nvfp4_language_state_dict(language)
    return {f"language_model.{name}": data for name, data in language.items()}


def convert_vision_state_dict_for_module(
    state_dict: dict[str, Weights], **unused_kwargs
) -> dict[str, WeightData]:
    """Vision slice, unprefixed.

    The eager vision tower mirrors the graph tower's attribute names, so
    the graph converter's output keys (``patch_embedder.*``, ``encoder.layers.*``,
    ``embed_vision.*``, ``std_bias``/``std_scale``) already match
    ``Gemma4VisionModel``'s parameter paths. Text-only checkpoints simply
    match no key, yielding an empty dict.
    """
    return dict(convert_safetensor_vision_state_dict(state_dict))
