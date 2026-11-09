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
"""The GLM-5.x decoder (ModuleV3), registered for the ``DECODER`` role."""

from __future__ import annotations

from typing import ClassVar

from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import DP
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.nn.legacy.comm.ep import EPBatchManager, EPCommBuffers
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import (
    DeviceMapping,
    Replicated,
    Unknown,
)

from ..deepseekV3_2_modulev3.deepseekV3_2 import (
    DeepseekV3_2TextModel,
    replica_context_lengths,
)
from .model_config import Glm5_1Config


class Glm5_1Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """GLM-5.x's sparse-MLA decoder, shared with DeepSeek-V3.2.

    It runs tensor-parallel attention over a ``tp`` mesh, one attention
    replica per device over a ``dp`` mesh, or, under expert parallelism,
    tensor-parallel replicas over a ``(dp, tp)`` mesh, with its experts
    tensor-parallel or, under expert parallelism, spread across the devices.
    """

    graph_name: ClassVar[str] = "DeepseekV3_2"
    supports_data_parallel: ClassVar[bool] = True
    supports_multi_device_replicas: ClassVar[bool] = True

    def __init__(self, config: Glm5_1Config) -> None:
        super().__init__()
        if config.mesh is None:
            raise ValueError("GLM-5.x needs a mesh: apply the runtime first")
        self.ep_batch_manager = (
            None
            if config.ep_config is None
            else EPBatchManager(config.ep_config)
        )
        # Quantized checkpoints keep norms, biases and embeddings in bf16.
        dtype = (
            DType.bfloat16 if config.quant_config is not None else config.dtype
        )
        with default_dtype(dtype):
            self.language_model = DeepseekV3_2TextModel(
                config, self.ep_batch_manager
            )
        self.config = config
        self.kv_params = config.kv_params

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the decoder over the batch."""
        mesh = self.config.mesh
        assert mesh is not None
        assert inputs.batch_context_lengths is not None

        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        dp_degree = self.config.data_parallel_degree
        if dp_degree > 1:
            assert inputs.host_input_row_offsets is not None
            batch_context_length = replica_context_lengths(
                inputs.batch_context_lengths, mesh
            )
            data_parallel_splits = inputs.data_parallel_splits
            input_row_offsets_i64 = F.cast(
                inputs.host_input_row_offsets, DType.int64
            )
        else:
            batch_context_length = inputs.batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in tree.leaves(inputs.kv))
        # A ``MultiKVCacheParams`` tree unflattens to a dict keyed by child
        # name, so the two caches are addressed by name rather than position.
        kv_tree = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_tree, dict)
        # Replicas serve different requests, so their caches vary;
        # tensor-parallel devices all store the same latent cache.
        kv_mapping = DeviceMapping(
            mesh,
            tuple(
                Unknown() if name == DP else Replicated()
                for name in mesh.axis_names
            ),
        )

        def collection(name: str) -> PagedCacheValues:
            leaves = tree.leaves(kv_tree[name], leaf=KVCacheInputsPerDevice)
            return PagedCacheValues.from_upstream(leaves, kv_mapping)

        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            assert inputs.ep is not None
            comm_buffers = self.ep_batch_manager.comm_buffers(
                [x._graph_value for x in tree.leaves(inputs.ep)]
            )
        outputs = self.language_model(
            inputs.tokens,
            collection("mla"),
            collection("indexer"),
            inputs.return_n_logits,
            inputs.input_row_offsets,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )
        return self.language_model.module_outputs(outputs)
