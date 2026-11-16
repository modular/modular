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
"""Implements the GLM-5.1 (GlmMoeDsa) pipeline model."""

from __future__ import annotations

from typing import Any, ClassVar

from max.pipelines.architectures.deepseekV3_2.model import DeepseekV3_2Model
from max.pipelines.lib import CheckpointRole, WeightEntry, WeightTransform
from max.pipelines.lib.graph_specs import GraphRole

from .model_config import Glm5_1Config


class Glm5_1Model(DeepseekV3_2Model):
    """GLM-5.1 pipeline model.

    Skeleton alias of :class:`~max.pipelines.architectures.deepseekV3_2.model.DeepseekV3_2Model`
    until GLM-specific bring-up diverges from DeepSeek-V3.2.
    """

    model_config_cls: ClassVar[type[Any]] = Glm5_1Config

    # Without allreduce, attention reduce-scatters tokens across the TP ranks
    # before the expert dispatch, which is the layout GLM-5.3's recipe serves.
    # Exporters pick the matching EP backend from it.
    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"

    def _load_state_dict(self) -> dict[str, Any]:
        state_dict = super()._load_state_dict()
        # The architecture's rules name the decoder's weights.
        assert isinstance(self.adapter, WeightTransform)
        self.weight_naming = {
            GraphRole.DECODER: (
                WeightEntry(CheckpointRole.TARGET, self.adapter),
            )
        }
        return state_dict
