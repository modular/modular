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
"""Memory planner for the Mamba2 (SSD) architecture."""

from __future__ import annotations

from max.driver import Device
from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model_config import Mamba2Config


class Mamba2MemoryPlanner(PagedMemoryPlanner):
    """Paged planner that also reserves the per-request SSM state pool.

    Mamba2 keeps its real per-request state in
    :class:`~.ssm_cache.Mamba2SSMStateCache`, not in the (dummy) paged KV
    cache. Surfacing the pool size as activation memory keeps the cache
    allocator's budget consistent with what the model actually allocates.
    """

    #: Set by :meth:`infer_max_batch_size`; consumed by
    #: :meth:`estimate_activation_memory` when the user left
    #: ``max_batch_size`` unset.
    _inferred_max_batch_size: int | None = None

    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        inferred = super().infer_max_batch_size(
            pipeline_config, devices, weights_size
        )
        self._inferred_max_batch_size = inferred
        return inferred

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserve GPU memory for the per-request SSM state pool.

        The SSM cache holds ``num_layers`` x ``(conv_state + ssm_state)``
        per slot:

        * ``conv_state``: ``conv_dim * (d_conv - 1)`` elements
        * ``ssm_state``:  ``nheads * head_dim * d_state`` elements
        """
        cfg = Mamba2Config.from_hf_config(huggingface_config)

        max_batch = pipeline_config.runtime.max_batch_size
        if max_batch is None:
            max_batch = self._inferred_max_batch_size
        if max_batch is None:
            max_batch = 1

        encoding = pipeline_config.model.quantization_encoding
        state_dtype = (
            supported_encoding_dtype(encoding)
            if encoding is not None
            else DType.float32
        )
        dtype_bytes = state_dtype.size_in_bytes

        conv_state_elems = cfg.conv_dim * (cfg.d_conv - 1)
        ssm_state_elems = cfg.nheads * cfg.headdim * cfg.d_state
        per_layer_bytes = (conv_state_elems + ssm_state_elems) * dtype_bytes
        return max_batch * cfg.n_layer * per_layer_bytes
