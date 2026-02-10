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
"""Nemotron pipeline model.

Reuses the ``LlamaModelBase`` pipeline infrastructure (KV cache management,
input preparation, execution) and overrides ``_build_graph`` to assemble a
``Nemotron`` model graph instead of a ``Llama3`` one.
"""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.driver import Device
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
)
from transformers import AutoConfig

from ..llama3.model import LlamaModelBase
from .model_config import NemotronConfig
from .nemotron import Nemotron


class NemotronModel(LlamaModelBase):
    """Nemotron pipeline model implementation.

    Extends ``LlamaModelBase`` which provides the full pipeline lifecycle:
    input preparation, KV-cache plumbing, execution, and log-probability
    computation.  Only ``_build_graph`` is overridden to wire up the
    Nemotron-specific graph (LayerNorm, partial RoPE, squared-ReLU MLP).
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

    # ------------------------------------------------------------------
    # KV params delegate
    # ------------------------------------------------------------------

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return NemotronConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return NemotronConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> dict[str, WeightData]:
        huggingface_config = self.huggingface_config
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        return state_dict

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = self._get_state_dict(weights, adapter)

        model_config = NemotronConfig.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )

        # Single-GPU execution path.
        model = Nemotron(model_config)

        model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        self.state_dict: dict[str, Any] = model.state_dict()

        with Graph(
            "nemotron",
            input_types=model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                *kv_cache_inputs,
            ) = graph.inputs

            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            outputs = model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph
