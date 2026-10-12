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
"""Implements the GLM-5.x (GlmMoeDsa) pipeline model, in the ModuleV3 API."""

from __future__ import annotations

from typing import Any, ClassVar

from ..deepseekV3_2_modulev3.model import DeepseekV3_2Model
from .model_config import Glm5_1Config


class Glm5_1Model(DeepseekV3_2Model):
    """GLM-5.x pipeline model (ModuleV3).

    GLM shares DeepSeek-V3.2's sparse-MLA decoder; only the config differs.
    """

    model_config_cls: ClassVar[type[Any]] = Glm5_1Config
