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

from typing import Any

from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)

from .model_config import MuseSparkConfig


# TODO: replace with the ModuleV3 text model.
class MuseSparkModel(PipelineModel[TextContext, MuseSparkConfig, Any]):
    """Registers the architecture; raises until the text model exists."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The Muse Spark pipeline model is not built yet"
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError(
            "The Muse Spark pipeline model is not built yet"
        )
