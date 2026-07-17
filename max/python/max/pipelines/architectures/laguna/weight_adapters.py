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
"""Weight adapter for Laguna-M.1-NVFP4 (``poolside/Laguna-M.1-NVFP4``).

The checkpoint is compressed-tensors NVFP4: MLP/MoE ``gate_proj``/``up_proj``/
``down_proj`` (dense layer 0 + the routed experts + the shared expert) are
4-bit, each quantized Linear carrying four tensors:

.. code-block:: text

    weight_packed        packed FP4 (uint8, two fp4 per byte)
    weight_scale         per-group (16) FP8-e4m3 scale
    weight_global_scale  FP32 global scale
    input_global_scale   FP32 activation scale

MAX's ``MoEQuantized`` / ``Linear(quant_config=...)`` declare each expert's
projection as a per-expert quantized Linear expecting ``.weight`` (packed),
``.weight_scale``, ``.weight_scale_2`` (global), ``.input_scale``. The suffixes
map as follows; the two global scales are also reciprocated into MAX's
multiplicative convention:

.. code-block:: text

    .weight_packed        -> .weight
    .weight_global_scale  -> .weight_scale_2
    .input_global_scale   -> .input_scale
    .weight_scale         kept

Attention projections (q/k/v/o/g_proj) are in the quant ``ignore`` list and ship
bf16 ``.weight`` (passed through). The checkpoint's static FP8 KV-cache scales
``self_attn.{k,v}_scale`` are DROPPED: MAX computes KV-cache scales dynamically
at runtime and never consumes checkpoint static scales (same as deepseekV3/Kimi),
so a strict load would otherwise reject them as unexpected keys. Laguna structural
renames (router gate, expert correction bias, shared-expert pluralization) match
the bf16 donor.
"""

from __future__ import annotations

from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from max.pipelines.weights._compressed_tensors import (
    convert_compressed_tensors_nvfp4_weight,
)
from transformers import AutoConfig

# Checkpoint -> MAX weight name mapping, applied with ``str.replace`` semantics.
LAGUNA_SAFETENSOR_MAP: dict[str, str] = {
    "model.": "",
    # Laguna structural renames (same as the bf16 donor).
    ".mlp.gate.weight": ".mlp.gate.gate_score.weight",
    ".mlp.experts.e_score_correction_bias": ".mlp.gate.e_score_correction_bias",
    ".mlp.shared_expert.": ".mlp.shared_experts.",
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Converts Laguna-M.1-NVFP4 safetensor weights to MAX names.

    Renames the compressed-tensors NVFP4 suffixes to the per-expert quantized
    Linear names MAX's ``MoEQuantized`` expects, applies the Laguna structural
    renames, and reinterprets uint8 per-group scales as ``float8_e4m3fn``.
    """
    del huggingface_config, pipeline_config
    out: dict[str, WeightData] = {}
    for hf_name, value in state_dict.items():
        # FP8 KV-cache scales: the graph uses a bf16 KV cache, so these are
        # unused (strict load would reject them as unexpected). Drop them.
        if hf_name.endswith(".k_scale") or hf_name.endswith(".v_scale"):
            continue
        max_name = hf_name
        for before, after in LAGUNA_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        data = value.data()
        max_name, data = convert_compressed_tensors_nvfp4_weight(max_name, data)
        out[max_name] = data
    return out
