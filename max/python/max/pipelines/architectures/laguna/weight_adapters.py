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
"""Weight adapter for the Laguna checkpoint family.

Every released Laguna checkpoint mixes precisions: some linear layers ship
compressed-tensors NVFP4 weights, the rest ship a plain ``.weight`` tensor.
Which is which varies per layer and per module, but the adapter renames by
suffix and so never has to know. Per-layer precision is resolved downstream in
:meth:`~max.pipelines.architectures.laguna.model.LagunaModel._resolve_nvfp4_quant_config`,
from the scales this adapter emits.

An NVFP4 Linear carries four tensors:

.. code-block:: text

    weight_packed        packed FP4 (uint8, two fp4 per byte)
    weight_scale         per-group (16) FP8-e4m3 scale
    weight_global_scale  FP32 global scale
    input_global_scale   FP32 activation scale

MAX's ``MoEQuantized`` / ``Linear(quant_config=...)`` declare each expert's
projection as a per-expert quantized Linear expecting ``.weight`` (packed),
``.weight_scale``, and the global scales under their stored names, which the
layers invert in the graph. So the NVFP4 mapping is one suffix rename:

.. code-block:: text

    .weight_packed        -> .weight
    .weight_scale         kept
    .weight_global_scale  kept
    .input_global_scale   kept

The bf16 modules carry an unsuffixed ``.weight``, which matches none of those
suffixes and passes through as-is: attention (q/k/v/o/g_proj), norms, router
gates, correction biases, and whichever dense MLPs, shared experts and routed
experts a given checkpoint left unquantized.

Beyond the suffixes, the adapter strips the ``model.`` prefix and applies the
Laguna structural renames (router gate, expert correction bias, shared-expert
pluralization) that match the bf16 donor. The checkpoint's static FP8 KV-cache
scales ``self_attn.{k,v}_scale`` are DROPPED: MAX computes KV-cache scales
dynamically at runtime and never consumes checkpoint static scales (same as
deepseekV3/Kimi), so a strict load would otherwise reject them as unexpected
keys.
"""

from __future__ import annotations

import dataclasses

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Checkpoint -> MAX weight name mapping, applied in order with ``str.replace``
# semantics (``dict`` preserves insertion order). Order matters: do the NVFP4
# suffix renames before the structural ones so neither re-matches the other.
LAGUNA_SAFETENSOR_MAP: dict[str, str] = {
    # NVFP4 compressed-tensors packed weight -> MAX quantized-Linear weight.
    ".weight_packed": ".weight",
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
    """Converts Laguna safetensor weights to MAX names.

    Renames the compressed-tensors NVFP4 packed weight to the per-expert
    quantized Linear name MAX's ``MoEQuantized`` expects, applies the Laguna
    structural renames, and reinterprets uint8 per-group scales as
    ``float8_e4m3fn``. A tensor carrying none of those suffixes keeps its name
    and its contents.
    """
    del huggingface_config, pipeline_config
    out: dict[str, WeightData] = {}
    for hf_name, value in state_dict.items():
        # FP8 KV-cache scales: the graph uses a bf16 KV cache, so these are
        # unused (strict load would reject them as unexpected). Drop them.
        if hf_name.endswith((".k_scale", ".v_scale")):
            continue
        # Only the leading ``model.`` is a prefix; a substring replace would
        # also rewrite anything containing it further along the name.
        max_name = hf_name.removeprefix("model.")
        for before, after in LAGUNA_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        data = value.data()
        # Per-group NVFP4 scales are float8_e4m3fn; some exporters store them
        # as raw uint8. Reinterpret so the kernel reads the right dtype. (This
        # is the NVFP4 analogue of the MXFP4 path, where per-group scales are
        # instead reinterpreted as float8_e8m0fnu — see minimax_m2.)
        if max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e4m3fn)
        out[max_name] = data
    return out
