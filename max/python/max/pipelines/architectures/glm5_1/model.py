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

from .model_config import Glm5_1Config


class Glm5_1Model(DeepseekV3_2Model):
    """GLM-5.1 pipeline model.

    Skeleton alias of :class:`~max.pipelines.architectures.deepseekV3_2.model.DeepseekV3_2Model`
    until GLM-specific bring-up diverges from DeepSeek-V3.2.
    """

    model_config_cls: ClassVar[type[Any]] = Glm5_1Config
