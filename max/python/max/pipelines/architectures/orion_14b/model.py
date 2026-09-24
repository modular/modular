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

"""Implements the Orion pipeline model."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from max.engine import InferenceSession
from max.graph import Graph
from max.pipelines.architectures.llama3.model import Llama3Model

from .model_config import OrionConfig
from .orion_14b import Orion


class OrionModel(Llama3Model):
    """Orion pipeline model implementation.

    ``norm_method`` stays at the donor default: neither donor branch produces
    Orion's normalization -- :class:`Orion` replaces all three norm sites after
    construction regardless -- and only ``"rms_norm"`` constructs for a BF16
    model, since ``ConstantLayerNorm`` builds its affines through NumPy, which
    has no native bfloat16.
    """

    model_config_cls: ClassVar[type[Any]] = OrionConfig
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = OrionConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, OrionConfig)

        if model_config.data_parallel_degree > 1:
            raise NotImplementedError(
                "Orion does not support data parallelism: "
                "create_data_parallel_graph builds the donor's RMSNorm graph, "
                "which does not match Orion's learned LayerNorm."
            )
        if len(self.devices) > 1:
            raise NotImplementedError(
                "Orion is single-device only: DistributedLlama3 builds the "
                "donor's RMSNorm graph, which does not match Orion's learned "
                "LayerNorm. Serve with a single --devices entry."
            )
        if self._lora_manager:
            raise NotImplementedError(
                "Orion does not support LoRA: the adapter path is untested "
                "against this port's replaced norm layers."
            )

        return self._build_single_device_graph_for_compile(
            state_dict, model_config
        )

    def _build_single_device_graph_for_compile(
        self,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        assert isinstance(model_config, OrionConfig)
        single_model = Orion(model_config)

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = single_model.state_dict()

        with Graph(
            "orion",
            input_types=single_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                *rest,
            ) = graph.inputs
            kv_collections = self._unflatten_kv_inputs(rest)
            outputs = single_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry
