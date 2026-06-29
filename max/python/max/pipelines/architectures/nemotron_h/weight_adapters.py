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

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

_IN_PROJ_RE = re.compile(
    r"^(blocks\.\d+\.mixer\.)in_proj\.(weight|weight_scale|input_scale)$"
)

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


def _row_slice(
    wd: WeightData, start: int, end: int, new_name: str
) -> WeightData:
    """Row-slice a 2-D weight ``[out, in]`` to ``[end-start, in]`` (contiguous).

    Works for fp8 (``Buffer.from_dlpack(...)[start:end, :]`` keeps a contiguous
    sub-range).
    """
    in_dim = int(wd.shape[1])
    sliced = Buffer.from_dlpack(wd.data)[start:end, :]
    return WeightData(
        data=sliced,
        name=new_name,
        dtype=wd.dtype,
        shape=Shape([end - start, in_dim]),
        quantization_encoding=wd.quantization_encoding,
    )


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert a Nemotron-H checkpoint to MAX module weight names.

    The fused mamba ``in_proj`` is split into three ``in_proj_{gate,
    hidden_BC,dt}`` projections (the nn.Module uses three matmuls to keep the
    gate/hidden_BC/dt outputs contiguous). The weight is row-sliced; the
    per-tensor FP8 ``weight_scale`` / ``input_scale`` scalars are replicated to
    all three.
    """
    # Mamba in_proj split sizes.
    intermediate = (
        huggingface_config.mamba_num_heads * huggingface_config.mamba_head_dim
    )
    conv_dim = intermediate + 2 * (
        huggingface_config.n_groups * huggingface_config.ssm_state_size
    )
    nheads = huggingface_config.mamba_num_heads
    splits = [
        ("in_proj_gate", 0, intermediate),
        ("in_proj_hidden_BC", intermediate, intermediate + conv_dim),
        (
            "in_proj_dt",
            intermediate + conv_dim,
            intermediate + conv_dim + nheads,
        ),
    ]

    new_state_dict: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in _RENAMES:
            if before in max_name:
                max_name = max_name.replace(before, after)
        for before, after in _MIXER_RENAMES:
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # Split the fused mamba in_proj into three projections.
        m = _IN_PROJ_RE.match(max_name)
        if m is not None:
            prefix, kind = m.group(1), m.group(2)
            if kind == "weight":
                for sub, lo, hi in splits:
                    sub_name = f"{prefix}{sub}.weight"
                    new_state_dict[sub_name] = _row_slice(
                        weight_data, lo, hi, sub_name
                    )
            else:  # weight_scale / input_scale: per-tensor scalar -> replicate
                sd = weight_data.astype(DType.float32)
                for sub, _lo, _hi in splits:
                    new_state_dict[f"{prefix}{sub}.{kind}"] = sd
            continue

        # Scale tensors -> float32 (FP8 kernels require f32 scales).
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

    return new_state_dict
