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


import numpy as np


class FlowMatchEulerDiscreteScheduler:
    """Minimal stub for FlowMatchEulerDiscreteScheduler."""

    def __init__(self, **kwargs) -> None:
        self.config = type("Config", (), {"use_flow_sigmas": False})()
        self.timesteps = np.array([], dtype=np.float32)
        self.sigmas = np.array([], dtype=np.float32)
        self.order = 1

    def set_timesteps(
        self, num_inference_steps: int, device: str | None = None, **kwargs
    ) -> None:
        """Stub for set_timesteps."""
        self.timesteps = np.linspace(
            1000, 0, num_inference_steps, dtype=np.float32
        )
        self.sigmas = np.linspace(1.0, 0, num_inference_steps, dtype=np.float32)
