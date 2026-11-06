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
"""The registered Hy4 decoder role."""

from __future__ import annotations

from typing import ClassVar

from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.tensor import default_dtype

from .hy_v4 import HYV4TextModel
from .model_config import HYV4Config


class HYV4Decoder(Module[[DecoderInputs], DecoderOutputs]):
    """Runs Hy4 with the shared decoder input and output trees."""

    graph_name: ClassVar[str] = "HYV4"
    input_layout: ClassVar[tuple[str, ...]] = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "batch_context_lengths",
        "kv",
    )

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.cache_params: list[KVCacheParams] = []
        for params in config.kv_params.children.values():
            assert isinstance(params, KVCacheParams)
            self.cache_params.append(params)
        # The config dtype selects FP8 quantization; unquantized parameters
        # are BF16, with FP32 weights declared explicitly by their layers.
        with default_dtype(DType.bfloat16):
            self.language_model = HYV4TextModel(config)
        if self.language_model.embed_tokens.weight.mesh.num_devices > 1:
            self.language_model.shard()

    def forward(self, inputs: DecoderInputs) -> DecoderOutputs:
        """Runs the decoder with the existing graph signature and weights."""
        assert inputs.batch_context_lengths is not None
        (batch_context_length,) = inputs.batch_context_lengths
        mesh = self.language_model.embed_tokens.weight.mesh
        tokens = inputs.tokens
        input_row_offsets = inputs.input_row_offsets
        if mesh.num_devices > 1:
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)
        values = iter(t._graph_value for t in tree.leaves(inputs.kv))
        mla, indexer = (
            PagedCacheValues.from_upstream(
                params.unflatten_kv_inputs(values), tokens.mapping
            )
            for params in self.cache_params
        )
        logits = self.language_model(
            tokens, mla, indexer, input_row_offsets, batch_context_length
        )
        return DecoderOutputs(next_token_logits=logits.to(mesh.devices[0]))
