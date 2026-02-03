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
"""Flow Match Euler Discrete Scheduler for diffusion models."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _time_shift_exponential(
    mu: float, sigma_param: float, t: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Resolution-dependent timestep shift (diffusers FlowMatchEulerDiscreteScheduler)."""
    t_safe = np.clip(t.astype(np.float64), 1e-7, 1.0)
    out = np.exp(mu) / (np.exp(mu) + (1.0 / t_safe - 1.0) ** sigma_param)
    return out.astype(np.float32)


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    use_flow_sigmas: bool = False
    use_dynamic_shifting: bool = True


class FlowMatchEulerDiscreteScheduler:
    """Minimal Flow Match Euler Discrete Scheduler.

    This scheduler provides timestep and sigma scheduling for flow-matching
    diffusion models. The actual denoising step computation is handled by
    the pipeline (e.g., FluxPipeline._scheduler_step).
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the scheduler.

        Args:
            **kwargs: Configuration parameters (currently unused, accepted
                for compatibility with diffusers config loading).
        """
        self.config = SchedulerConfig(
            use_flow_sigmas=kwargs.get("use_flow_sigmas", False),
            use_dynamic_shifting=kwargs.get("use_dynamic_shifting", True),
        )
        self.timesteps: npt.NDArray[np.float32] = np.array([], dtype=np.float32)
        self.sigmas: npt.NDArray[np.float32] = np.array([], dtype=np.float32)
        self.order: int = 1

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
            **kwargs: mu (float) for resolution-dependent shifting when
                use_dynamic_shifting is True.
        """
        if sigmas is not None:
            sigmas = np.asarray(sigmas, dtype=np.float32)
            if self.config.use_dynamic_shifting and "mu" in kwargs:
                mu = float(kwargs["mu"])
                sigmas = _time_shift_exponential(mu, 1.0, sigmas)
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
