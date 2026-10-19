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
export without a real checkpoint or tokenizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, TensorType, Value
from max.nn import Module
from max.pipelines import PipelineConfig
from max.pipelines.lib import GraphRole, GraphSpec
from test_common.pipeline_model_dummy import (
    DUMMY_LLAMA_ARCH,
    DummyLlamaPipelineModel,
)


class _EchoTokens(Module):
    """Returns the tokens, ignoring the KV cache inputs."""

    def __call__(
        self, tokens: Value[Any], *kv_inputs: Value[Any]
    ) -> Value[Any]:
        return tokens


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
                [
                    TensorType(
                        DType.int64,
                        shape=["batch_size"],
                        device=DeviceRef.CPU(),
                    ),
                    *self.kv_params.flattened_kv_inputs(),
                ],
                weights={},
            )
        }

    def load_model(self, session: InferenceSession) -> Model:
        return self.load_graphs(session)[GraphRole.DECODER]


ARCHITECTURES = [
    replace(DUMMY_LLAMA_ARCH, pipeline_model=DummyGraphSpecsPipelineModel)
]
