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
"""A dummy Llama architecture that registers its graph through graph_specs.

Pass it as a custom architecture to build a pipeline that ``max export`` can
export without a real checkpoint or tokenizer. Its decoder names its inputs to
a batch processor, so the export records the named input tree and the
per-step dims a serving engine sizes its inputs by.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, TensorType, Value
from max.nn import Module
from max.nn.graph_groups import Logits, RaggedBatch
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.transformer import DecoderOutputs, ReturnLogits
from max.pipelines import PipelineConfig
from max.pipelines.context import TextContext
from max.pipelines.lib import GraphRole, GraphSpec, ModelInputs, ModelOutputs
from max.pipelines.lib.interfaces import (
    ArchConfig,
    BatchProcessor,
    BatchProcessorRuntime,
)
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)


class _EchoTokens(Module):
    """Returns the tokens as the logits, ignoring the KV cache inputs."""

    def __call__(
        self, tokens: Value[Any], *kv_inputs: Value[Any]
    ) -> DecoderOutputs:
        return DecoderOutputs(logits=Logits(next_token_logits=tokens.tensor))


class _NamesInputs(BatchProcessor[TextContext, ModelInputs, ArchConfig]):
    """Holds the decoder's named inputs; the dummy batches nothing itself."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        raise NotImplementedError

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        raise NotImplementedError

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        raise NotImplementedError


class DummyGraphSpecsPipelineModel(DummyLlamaPipelineModel):
    """The dummy Llama model, loaded through :meth:`load_graphs`."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(pipeline_config, session, *args, **kwargs)
        self._batch_processor = _NamesInputs(
            self.arch_config,
            BatchProcessorRuntime(
                pipeline_config=pipeline_config,
                devices=self.devices,
                return_logits=ReturnLogits.LAST_TOKEN,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
            ),
        )
        self.model = self.load_model(session)

    def _load_state_dict(self) -> dict[str, Any]:
        return {}

    def _create_model_config(self, state_dict: dict[str, Any]) -> None:
        return None

    def graph_specs(self, model_config: None) -> Mapping[str, GraphSpec]:
        return {
            GraphRole.DECODER: GraphSpec(
                _EchoTokens(),
                "dummy",
                self._bind_graph_inputs(
                    OrderedDict(
                        batch=RaggedBatch(
                            tokens=TensorType(
                                DType.int64,
                                shape=["total_seq_len"],
                                device=DeviceRef.CPU(),
                            )
                        ),
                        kv=self.kv_params.get_symbolic_inputs(),
                    )
                ),
                weights={},
            )
        }

    def load_model(self, session: InferenceSession) -> Model:
        return self.load_graphs(session)[GraphRole.DECODER]


ARCHITECTURES = [
    replace(DUMMY_LLAMA_ARCH, pipeline_model=DummyGraphSpecsPipelineModel)
]
