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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max.driver import Device
from max.engine import Model
from max.graph.weights import Weights

if TYPE_CHECKING:
    from max.pipelines.lib import SupportedEncoding


class MaxModel(ABC):
    """Base interface for pipeline models with weight-backed execution."""

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        self.config = config
        self.encoding = encoding
        self.devices = devices
        self.weights = weights

    @abstractmethod
    def load_model(self) -> Model:
        """Load and return a runtime model instance."""
        ...
