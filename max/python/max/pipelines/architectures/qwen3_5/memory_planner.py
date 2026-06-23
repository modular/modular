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

"""Memory planner for the Qwen3.5 (GatedDeltaNet) architecture."""

from __future__ import annotations

from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model_config import Qwen3_5Config


class Qwen3_5MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Qwen3.5 GatedDeltaNet SSM models.

    Accounts for the slot-indexed recurrent state pool.
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserve GPU memory for the GatedDeltaNet state pool.

        The slot-indexed SSM kernels mutate the conv and recurrent pools in
        place; there are no working buffers and no graph-output pool, so
        peak footprint is a single ``max_batch x per_req`` allocation.

        ``Qwen3_5Config.initialize_from_config`` pre-sets ``max_batch_size``
        before this method runs, so it is always known here.
        """
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        layer_types = Qwen3_5Config._get_layer_types(text_config)
        num_linear = sum(1 for lt in layer_types if lt == "linear_attention")

        nk = getattr(text_config, "linear_num_key_heads", 16)
        nv = getattr(text_config, "linear_num_value_heads", 48)
        kd = getattr(text_config, "linear_key_head_dim", 128)
        vd = getattr(text_config, "linear_value_head_dim", 128)
        kernel = getattr(text_config, "linear_conv_kernel_dim", 4)

        conv_dim = 2 * kd * nk + vd * nv
        # Determine state dtype bytes: states stored in model dtype (typically bfloat16).
        encoding = pipeline_config.model.quantization_encoding
        state_dtype = (
            supported_encoding_dtype(encoding)
            if encoding is not None
            else DType.bfloat16
        )
        dtype_bytes = state_dtype.size_in_bytes
        bytes_per_layer = (
            conv_dim * (kernel - 1) * dtype_bytes + nv * kd * vd * dtype_bytes
        )
        per_req = num_linear * bytes_per_layer

        max_batch = pipeline_config.runtime.max_batch_size
        assert max_batch is not None, (
            "Qwen3_5Config.initialize_from_config must set max_batch_size "
            "before estimate_activation_memory runs"
        )
        # 1x: single in-place pool — kernels mutate it via slot_idx.
        return max_batch * per_req if num_linear > 0 else 0
