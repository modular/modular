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
"""Weight adapter for GLM-5.3-Flash (``glm5_next``) safetensors checkpoints.

Four transforms beyond prefix stripping:

* **The QKV conv concat.** The checkpoint stores ``q_conv1d`` / ``k_conv1d`` /
  ``v_conv1d`` separately, but the KDA layer runs one depthwise convolution
  over the concatenated QKV. The reference's conversion mapping is the
  authority on order -- ``Concatenate(dim=0)`` over ``[q, k, v]`` in that
  order, ``src/transformers/conversion_mapping.py``, the ``"Glm5Next"`` entry.
  Permuting them produces a model that runs and answers slightly badly, so the
  order is pinned by test rather than by comment.
* **``weight_scale_inv`` to ``weight_scale``**, matching the DeepSeek naming
  MAX's blockscaled-FP8 path reads. :mod:`.quantization` derives the whole
  precision map from which tensors have one.
* **float32 promotion** for the parameters the model computes in float32:
  ``A_log``, ``dt_bias``, the mHC ``base`` and ``scale``, the router's
  correction bias, and the KDA output gated norm.
* **Conv3d patch embedding to Linear** for the vision tower.
"""

from __future__ import annotations

import re

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from transformers import AutoConfig

__all__ = ["convert_glm5_next_state_dict"]

# Checkpoint prefix for the vision tower, and the MAX module it loads into.
_VISION_CHECKPOINT_PREFIX = "model.visual."
_VISION_MAX_PREFIX = "vision_encoder."

# Language-model prefix map, longest match first: the multimodal prefix has to
# be tried before the plain one or `model.` would strip only half of it.
_LM_PREFIX_MAP = (
    ("model.language_model.", ""),
    ("model.", ""),
)

# Parameters the model computes in float32. `A_log` and `dt_bias` already are;
# the casts are no-ops there and guard against a variant that stored bf16.
# `o_norm` backs the KDA output gated RMSNorm, which normalises in float32 --
# without this it would be downcast by a global bf16 cast.
_FLOAT32_SUFFIXES = (
    ".A_log",
    ".dt_bias",
    "_base",
    "_scale",
    ".o_norm.weight",
    ".e_score_correction_bias",
)

# Ordered, and the order is load-bearing. See the module docstring.
_CONV_PARTS = ("q_conv1d", "k_conv1d", "v_conv1d")

_KDA_CONV = re.compile(r"^(layers\.\d+\.self_attn)\.([qkv])_conv1d\.weight$")


def _concat_conv_channels(parts: list[WeightData], new_name: str) -> WeightData:
    """Concatenates depthwise conv weights ``[C, 1, K]`` along the channel axis.

    bfloat16 is reinterpreted as uint16 for the numpy concat -- numpy has no
    native bfloat16, and a concat along the leading axis of contiguous tensors
    is a pure byte append -- then reinterpreted back. No values change.
    """
    dtype = parts[0].dtype
    tail = tuple(int(d) for d in parts[0].shape[1:])
    for part in parts[1:]:
        if tuple(int(d) for d in part.shape[1:]) != tail:
            raise ValueError(
                f"Cannot concatenate '{new_name}': parts disagree on the "
                f"non-channel dims, {tail} vs "
                f"{tuple(int(d) for d in part.shape[1:])}."
            )
        if part.dtype != dtype:
            raise ValueError(
                f"Cannot concatenate '{new_name}': mixed dtypes "
                f"{dtype} and {part.dtype}."
            )
    total = sum(int(p.shape[0]) for p in parts)
    arrays = [
        np.from_dlpack(
            Buffer.from_dlpack(p.data).view(DType.uint16)
            if dtype == DType.bfloat16
            else Buffer.from_dlpack(p.data)
        )
        for p in parts
    ]
    concatenated = np.ascontiguousarray(np.concatenate(arrays, axis=0))
    shape = (total, *tail)
    return WeightData(
        data=Buffer.from_dlpack(concatenated).view(dtype=dtype, shape=shape),
        name=new_name,
        dtype=dtype,
        shape=Shape(list(shape)),
        quantization_encoding=parts[0].quantization_encoding,
    )


def _as_depthwise_conv(weight: WeightData) -> WeightData:
    """Reshapes a 2-D ``[C, K]`` conv weight to the ``[C, 1, K]`` MAX expects.

    GLM-5.3-Flash ships the conv weights already 3-D; this covers a variant
    that flattened the singleton group axis away.
    """
    if len(weight.shape) != 2:
        return weight
    channels, kernel = (int(d) for d in weight.shape)
    shape = (channels, 1, kernel)
    return WeightData(
        data=Buffer.from_dlpack(weight.data).view(
            dtype=weight.dtype, shape=shape
        ),
        name=weight.name,
        dtype=weight.dtype,
        shape=Shape(list(shape)),
        quantization_encoding=weight.quantization_encoding,
    )


def _conv3d_to_linear(weight: WeightData) -> WeightData:
    """Flattens the Conv3d patch projection into a Linear weight.

    HuggingFace stores ``(out_channels, in_channels, kT, kH, kW)``; the MAX
    patch embedding is an equivalent Linear over the flattened patch.
    """
    if len(weight.shape) != 5:
        return weight
    out_c, in_c, kt, kh, kw = (int(d) for d in weight.shape)
    flat = in_c * kt * kh * kw
    return WeightData(
        data=Buffer.from_dlpack(weight.data).view(
            dtype=weight.dtype, shape=(out_c, flat)
        ),
        name=weight.name,
        dtype=weight.dtype,
        shape=Shape([out_c, flat]),
        quantization_encoding=weight.quantization_encoding,
    )


def _strip_lm_prefix(name: str) -> str:
    for before, after in _LM_PREFIX_MAP:
        if name.startswith(before):
            return after + name[len(before) :]
    return name


def _rename(name: str) -> str:
    """Applies the MAX-side renames to an already-prefix-stripped name."""
    # The checkpoint stores the pool-compression gate as a bare `[128, 4096]`
    # tensor, which the reference applies as `F.linear(hidden_states, gate)`.
    # MAX declares it as a bias-less `Linear`, so it needs the `.weight` leaf.
    # Its sibling `index_kpool_compress_ape` is genuinely a bare parameter --
    # it is *added* to the gate logits, not multiplied -- and keeps its name.
    name = name.replace(
        "indexer.index_kpool_compress_gate",
        "indexer.index_kpool_compress_gate.weight",
    )
    # The MoE router lives behind `gate_score` in MAX's MoE gate. Anchored on
    # `mlp.gate.` so it cannot touch a dense layer's `mlp.gate_proj.weight`
    # or the router's `mlp.gate.e_score_correction_bias`.
    name = name.replace("mlp.gate.weight", "mlp.gate.gate_score.weight")
    return name.replace("weight_scale_inv", "weight_scale")


def convert_glm5_next_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts a GLM-5.3-Flash checkpoint to MAX weight names.

    Args:
        state_dict: The raw checkpoint weights.
        huggingface_config: The checkpoint's HuggingFace config, read for the
            layer count that identifies the MTP draft layer.

    Returns:
        The transformed weights.

    Raises:
        ValueError: If a KDA layer is missing one of its three conv tensors,
            or if the three disagree on dtype or kernel width.
    """
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    new_state_dict: dict[str, WeightData] = {}
    # Buffered per-layer conv parts, keyed by the `layers.N.self_attn` prefix.
    conv_parts: dict[str, dict[str, WeightData]] = {}

    for checkpoint_name, value in state_dict.items():
        weight = value.data()

        if checkpoint_name.startswith(_VISION_CHECKPOINT_PREFIX):
            max_name = (
                _VISION_MAX_PREFIX
                + checkpoint_name[len(_VISION_CHECKPOINT_PREFIX) :]
            )
            # The tower's stacked QKV loads into a StackedLinear.
            max_name = max_name.replace("attn.qkv.", "attn.qkv_proj.")
            if max_name == f"{_VISION_MAX_PREFIX}patch_embed.proj.weight":
                weight = _conv3d_to_linear(weight)
            # NOTE: `downsample.weight` is left 4-D, `[out, in, 2, 2]`. The 2x2
            # strided conv is flattened into a matmul in the tower, but whether
            # the flattened layout is (c, h, w) or (h, w, c) depends on how the
            # module gathers each merge patch -- a wrong guess here is a silent
            # accuracy loss, not a shape error. The vision lane owns that
            # contract and reshapes there.
            new_state_dict[max_name] = weight
            continue

        max_name = _rename(_strip_lm_prefix(checkpoint_name))

        conv_match = _KDA_CONV.match(max_name)
        if conv_match is not None:
            prefix, which = conv_match.groups()
            conv_parts.setdefault(prefix, {})[f"{which}_conv1d"] = (
                _as_depthwise_conv(weight)
            )
            continue

        if max_name.endswith(_FLOAT32_SUFFIXES):
            weight = weight.astype(DType.float32)

        new_state_dict[max_name] = weight

    for prefix, parts in conv_parts.items():
        missing = [name for name in _CONV_PARTS if name not in parts]
        if missing:
            raise ValueError(
                f"KDA layer '{prefix}' is missing {missing}. All three conv "
                "tensors are needed to build the concatenated-QKV convolution."
            )
        new_state_dict[f"{prefix}.conv1d.weight"] = _concat_conv_channels(
            [parts[name] for name in _CONV_PARTS], f"{prefix}.conv1d.weight"
        )

    # TODO(GLM53-MTP): the MTP draft layer's weights are dropped here, matching
    # DeepSeek-V3.2 and GLM-5.2, whose MTP lives in a separate registered
    # architecture (`unified_mtp_glm5_2`) that consumes them. The GLM-5.3-Flash
    # equivalent needs the KDA state rollback from `unified_mtp_qwen3_5`, so
    # until that lands these weights have no consumer and a strict load would
    # reject them.
    # TODO(GLM53-VISION): the graph is text-only -- the architecture is
    # registered as TEXT_GENERATION with a `TextContext`, and no vision tower
    # or placeholder substitution is wired. Dropping these keeps a strict load
    # honest about what the graph actually declares; the alternative is a
    # non-strict load that would also hide real regressions.
    vision_keys = [
        key for key in new_state_dict if key.startswith(_VISION_MAX_PREFIX)
    ]
    for key in vision_keys:
        del new_state_dict[key]

    mtp_prefix = f"layers.{text_config.num_hidden_layers}."
    for key in [k for k in new_state_dict if k.startswith(mtp_prefix)]:
        del new_state_dict[key]

    return new_state_dict
