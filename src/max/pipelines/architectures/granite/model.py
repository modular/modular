# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max.driver import Device
from max.engine import InferenceSession
from max.graph import ops
from max.graph.weights import Weights
from max.pipelines import KVCacheConfig, PipelineConfig, SupportedEncoding
from transformers import AutoConfig

from ..llama3.model import Llama3Model


class GraniteModel(Llama3Model):
    """Granite pipeline model implementation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
        )

        logits_scaling = getattr(huggingface_config, "logits_scaling", 1.0)

        if logits_scaling != 1.0:
            self.logits_processor = lambda logits: logits / ops.constant(
                logits_scaling, logits.dtype
            )
