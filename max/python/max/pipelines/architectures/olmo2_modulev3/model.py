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
from collections.abc import Callable
from typing import Any, ClassVar

from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from ..llama3_modulev3.model import Llama3Model
from .model_config import Olmo2Config
from .olmo2 import Olmo2

logger = logging.getLogger("max.pipelines")


class Olmo2Model(Llama3Model):
    """An Olmo2 pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = Olmo2Config
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )

    def load_model(self) -> Callable[..., Any]:
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }
        model_config = Olmo2Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        with F.lazy(), default_dtype(model_config.dtype):
            nn_model = Olmo2(model_config, self.kv_params)
            nn_model.to(self.devices[0])

        assert self.batch_processor is not None
        compile_input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params,
            device_refs=[device_ref],
        )

        compiled_model = nn_model.compile(
            *compile_input_types,
            weights=state_dict,
        )

        return compiled_model
