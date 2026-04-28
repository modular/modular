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

from __future__ import annotations

import logging
from typing import Any, Literal

from max._core.engine import Model
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig

from ..llama3.model import LlamaModelBase
from .model_config import Step3p5Config
from .step3p5 import Step3p5

logger = logging.getLogger("max.pipelines")


class Step3p5Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Step-3.5-Flash pipeline model implementation."""

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Step3p5Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )
        model_config = Step3p5Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        if self.pipeline_config.runtime.ep_size > 1:
            raise NotImplementedError(
                "Expert parallelism is not yet supported for Step-3.5."
            )

        nn_model = Step3p5(model_config)

        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )

        self.state_dict = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph("step3p5", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]

            kv_cache_inputs = variadic_args[num_devices:]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
            )

            graph.output(*outputs)
            return graph
