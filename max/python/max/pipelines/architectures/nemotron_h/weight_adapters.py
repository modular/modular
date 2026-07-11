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
"""Weight adapter for Nemotron-H checkpoints.

Maps HuggingFace ``NemotronHForCausalLM`` weight names to the MAX module names
and applies dtype/shape fixups:

* strip ``backbone.`` prefix; ``backbone.embeddings`` -> ``embed_tokens``;
  ``backbone.norm_f`` -> ``norm_f``; ``backbone.layers.N`` -> ``blocks.N``.
* conv1d weight ``[dim, 1, K]`` is kept 3-D (MAX expects depthwise [dim,1,K]).
* ``A_log`` / ``D`` / ``dt_bias`` cast to float32 (per-head scalars); the gated
  ``norm.weight`` cast to float32.
* FP8 (modelopt per-tensor static): F8_E4M3 weights are kept as-is; scale
  tensors (``weight_scale`` / ``input_scale``) cast to float32. Excluded
  modules (lm_head, attn q/k/v/o, the mamba in/out_proj at [11,16,23,31], all
  conv1d) stay bf16 — they simply have no scale tensors in the checkpoint.
"""

from __future__ import annotations

import re

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# The mamba ``in_proj`` is ONE fused matmul in the nn.Module (matching the HF
# reference + vLLM), so its checkpoint tensors map 1:1: ``in_proj.weight`` stays
# F8_E4M3 via the generic FP8 path, and ``in_proj.weight_scale`` /
# ``in_proj.input_scale`` fall through to the generic scale->fp32 path. No
# row-slicing into separate projections is needed.

# Attention q/k/v projections (always bf16 in the FP8 checkpoint) are fused into
# one ``qkv_proj.weight`` (concat order q, k, v along the out-dim) to match the
# single fused-QKV ``Linear`` in the nn.Module. ``o_proj`` is untouched.
_QKV_RE = re.compile(r"^(blocks\.\d+\.mixer\.)([qkv])_proj\.weight$")

# Ordered prefix/name rewrites (applied in sequence; first match wins per
# group). The real FP8 checkpoint uses the ``backbone.`` prefix; the installed
# transformers ``NemotronHForCausalLM`` (used as the logit-verify reference)
# uses the ``model.`` prefix instead. Handle both.
_RENAMES: list[tuple[str, str]] = [
    ("backbone.embeddings.", "embed_tokens."),
    ("backbone.norm_f.", "norm_f."),
    ("backbone.layers.", "blocks."),
    ("backbone.", ""),
    ("model.embeddings.", "embed_tokens."),
    ("model.norm_f.", "norm_f."),
    ("model.layers.", "blocks."),
    ("model.", ""),
]

# All ``mixer.*`` names map 1:1 onto the MAX mixer Weights. (The gated-norm
# weight's MAX Weight is declared with name ``norm.weight``, so it matches the
# checkpoint's ``mixer.norm.weight`` directly — no rename.)
_MIXER_RENAMES: list[tuple[str, str]] = []

# fp32 params: per-head SSM scalars + the mamba gated-norm weight. The block
# pre-norm (``blocks.{i}.norm.weight``) and final ``norm_f.weight`` stay bf16,
# so the gated-norm suffix is the specific ``.mixer.norm.weight``.
_FP32_SUFFIXES = (".A_log", ".D", ".dt_bias", ".mixer.norm.weight")


def _concat_rows(parts: list[WeightData], new_name: str) -> WeightData:
    """Concatenate 2-D weights ``[out_i, in]`` along the out-dim (axis 0).

    All parts share the same ``in`` dim and dtype. bf16 is reinterpreted as
    uint16 for the numpy concat (numpy has no native bf16), since a row-concat
    of contiguous ``[out, in]`` matrices is a pure byte append; the result is
    reinterpreted back to the original dtype. No values change.
    """
    in_dim = int(parts[0].shape[1])
    dtype = parts[0].dtype
    total_out = sum(int(p.shape[0]) for p in parts)
    arrs = [
        np.from_dlpack(Buffer.from_dlpack(p.data).view(DType.uint16))
        if dtype == DType.bfloat16
        else np.from_dlpack(Buffer.from_dlpack(p.data))
        for p in parts
    ]
    cat = np.concatenate(arrs, axis=0)
    buf = Buffer.from_dlpack(cat).view(dtype=dtype, shape=(total_out, in_dim))
    return WeightData(
        data=buf,
        name=new_name,
        dtype=dtype,
        shape=Shape([total_out, in_dim]),
        quantization_encoding=parts[0].quantization_encoding,
    )


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert a Nemotron-H checkpoint to MAX module weight names.

    The mamba ``in_proj`` is ONE fused matmul in the nn.Module, so its
    checkpoint tensors map 1:1 (``in_proj.weight`` -> generic FP8 path;
    ``in_proj.weight_scale`` / ``in_proj.input_scale`` -> generic scale->fp32).
    The attention q/k/v weights are concatenated into one fused
    ``qkv_proj.weight``.
    """
    new_state_dict: dict[str, WeightData] = {}
    # Per-attention-layer q/k/v weights buffered for fusion into qkv_proj.
    qkv_parts: dict[str, dict[str, WeightData]] = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in _RENAMES:
            if before in max_name:
                max_name = max_name.replace(before, after)
        for before, after in _MIXER_RENAMES:
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # Buffer attention q/k/v for fusion into a single qkv_proj.weight.
        qm = _QKV_RE.match(max_name)
        if qm is not None:
            prefix, which = qm.group(1), qm.group(2)
            qkv_parts.setdefault(prefix, {})[which] = weight_data
            continue

        # Scale tensors -> float32 (FP8 kernels require f32 scales). The fused
        # mamba ``in_proj`` scales pass through here 1:1.
        if max_name.endswith(
            ("weight_scale", "input_scale", "weight_scale_inv")
        ):
            if max_name.endswith("weight_scale_inv"):
                max_name = max_name[: -len("weight_scale_inv")] + "weight_scale"
            weight_data = weight_data.astype(DType.float32)
            new_state_dict[max_name] = weight_data
            continue

        # FP8 weights stay F8_E4M3.
        if weight_data.dtype == DType.float8_e4m3fn:
            new_state_dict[max_name] = weight_data
            continue

        # Per-head scalars + gated norm weight -> float32.
        if max_name.endswith(_FP32_SUFFIXES):
            weight_data = weight_data.astype(DType.float32)

        # conv1d weight: keep [dim, 1, K]. HF stores it as [dim, 1, K] already
        # (a Conv1d depthwise weight); if it ever comes in 2-D, expand.
        if max_name.endswith(".conv1d.weight") and len(weight_data.shape) == 2:
            d0, d1 = weight_data.shape
            buf = Buffer.from_dlpack(weight_data.data).view(
                dtype=weight_data.dtype, shape=(int(d0), 1, int(d1))
            )
            weight_data = WeightData(
                data=buf,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=Shape([d0, 1, d1]),
                quantization_encoding=weight_data.quantization_encoding,
            )

        new_state_dict[max_name] = weight_data

    # Emit the fused qkv_proj.weight per attention layer (concat order q,k,v).
    for prefix, parts in qkv_parts.items():
        fused_name = f"{prefix}qkv_proj.weight"
        new_state_dict[fused_name] = _concat_rows(
            [parts["q"], parts["k"], parts["v"]], fused_name
        )

    return new_state_dict
