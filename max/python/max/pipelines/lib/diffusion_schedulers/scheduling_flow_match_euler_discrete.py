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


import numpy as np
import numpy.typing as npt


class FlowMatchEulerDiscreteScheduler:
    """Minimal stub for FlowMatchEulerDiscreteScheduler."""

    def __init__(self, **kwargs) -> None:
        self.config = type("Config", (), {"use_flow_sigmas": False})()
        self.timesteps = np.array([], dtype=np.float32)
        self.sigmas = np.array([], dtype=np.float32)
        self.order = 1

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        sigmas: npt.NDArray[np.float32] | None = None,
        **kwargs,
    ) -> None:
        """Set the timesteps and sigmas for the diffusion process.

        Args:
            num_inference_steps: Number of inference steps. Used to generate
                timesteps if sigmas is not provided.
            sigmas: Custom sigma schedule. If provided, timesteps are derived
                from sigmas.
            **kwargs: Additional arguments (accepted for compatibility).
        """
        if sigmas is not None:
            # Use provided sigmas and derive timesteps
            # Append final sigma of 0.0 for the last scheduler step
            # (scheduler step accesses sigmas[i+1], so we need n+1 elements)
            self.sigmas = np.append(sigmas, np.float32(0.0))
            self.timesteps = sigmas * 1000.0
        elif num_inference_steps is not None:
            # Generate default timesteps
            self.timesteps = np.linspace(
                0, 1000, num_inference_steps, dtype=np.float32
            )
            self.sigmas = self.timesteps / 1000.0
