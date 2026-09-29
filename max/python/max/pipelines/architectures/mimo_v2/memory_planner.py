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
"""Memory planning for MiMo-V2.6-Flash."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import override

from .quant import FP8_BLOCK, MXFP4_BLOCK, moe_layers
from .weight_adapters import qkv_chunk_layout

_BF16 = DType.bfloat16.size_in_bytes
_F32 = DType.float32.size_in_bytes


def adapted_weights_size(config: PretrainedConfig, num_devices: int) -> int:
    """Returns the bytes the adapted text-model weights take on the devices.

    The count is of the adapter's output, not of the checkpoint files: dense
    F32 weights are FP8 with float32 128x128 block scales, the experts are
    MXFP4 with one E8M0 scale per 32 elements, and the vision, audio and MTP
    tensors are never loaded. Tensor parallelism splits every weight except
    the norms and the router, which each device holds whole.

    Args:
        config: The checkpoint's top-level Hugging Face config.
        num_devices: The tensor-parallel degree.

    Returns:
        The total over all devices, in bytes.
    """
    hidden = config.hidden_size
    # The embedding and the untied head; the final norm.
    split = 2 * config.vocab_size * hidden * _BF16
    replicated = hidden * _BF16
    moe = set(moe_layers(config))
    for layer in range(config.num_hidden_layers):
        sliding = config.hybrid_layer_pattern[layer] == 1
        layout = qkv_chunk_layout(config, sliding)
        qkv_rows = layout.chunks * layout.padded_rows
        heads = layout.chunks * layout.q_rows // config.head_dim
        blocks = (qkv_rows // FP8_BLOCK) * (hidden // FP8_BLOCK)
        split += qkv_rows * hidden + blocks * _F32
        split += hidden * heads * config.v_head_dim * _BF16
        sinks = (
            "add_swa_attention_sink_bias"
            if sliding
            else "add_full_attention_sink_bias"
        )
        if getattr(config, sinks, False):
            split += heads * _BF16
        replicated += 2 * hidden * _BF16
        if layer not in moe:
            width = config.intermediate_size
            blocks = (width // FP8_BLOCK) * (hidden // FP8_BLOCK)
            split += 3 * (width * hidden + blocks * _F32)
            continue
        experts, width = config.n_routed_experts, config.moe_intermediate_size
        replicated += experts * hidden * _F32 + experts * _F32
        # Gate, up and down: 4-bit codes and one E8M0 byte per 32 elements.
        elements = 3 * experts * width * hidden
        split += elements // 2 + elements // MXFP4_BLOCK
    return split + replicated * num_devices


class MiMoV2MemoryPlanner(PagedMemoryPlanner):
    """Plans the KV cache around the adapted weights.

    The default weight estimate sums the safetensors files, which for this
    checkpoint are larger than what reaches the devices and include the
    audio tokenizer and DFlash drafter shipped beside the text model.

    No activation memory is reserved. At the default 8,192-token prefill
    budget the largest transient is the MoE's, under 4 GB per device even if
    none of its tensors is fused or freed early, well inside the headroom
    that ``device_memory_utilization`` leaves (10% of each device).
    """

    _always_signal_buffers = True

    @override
    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Returns :func:`adapted_weights_size` for the pipeline's devices."""
        model = pipeline_config.model
        assert model.huggingface_config is not None
        return adapted_weights_size(
            model.huggingface_config, len(model.device_specs)
        )
