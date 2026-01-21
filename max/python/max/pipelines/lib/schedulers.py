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

"""Scheduler implementations for diffusion pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
import numpy.typing as npt


class Scheduler(ABC):
    """Base class for all diffusion schedulers."""

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Any = None,
        timesteps: npt.NDArray[np.int64] | None = None,
        sigmas: npt.NDArray[np.float32] | None = None,
        **kwargs,
    ) -> None:
        """Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: The number of diffusion steps.
            device: The device to move timesteps to.
            timesteps: Custom timesteps array.
            sigmas: Custom sigmas array.
            kwargs: Additional scheduler-specific parameters.
        """
        pass

    @property
    @abstractmethod
    def timesteps(self) -> npt.NDArray[np.float32]:
        """The computed timestep schedule."""
        pass

    @property
    @abstractmethod
    def sigmas(self) -> npt.NDArray[np.float32]:
        """The computed sigmas schedule."""
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """The scheduler configuration dictionary."""
        pass


class FlowMatchEulerDiscreteScheduler(Scheduler):
    pass


class SchedulerFactory:
    """Factory for creating schedulers from diffusers configuration."""

    _REGISTRY: Dict[str, Type[Scheduler]] = {
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    }

    @classmethod
    def create(
        cls, class_name: str, config_dict: Dict[str, Any] | None = None
    ) -> Scheduler:
        """Create a scheduler instance.

        Args:
            class_name: The diffusers scheduler class name.
            config_dict: Parameters for the scheduler.

        Returns:
            A scheduler instance.

        Raises:
            ValueError: If the scheduler class is not supported.
        """
        if class_name not in cls._REGISTRY:
            # Fallback to FlowMatch if it's a flow-match variant, or error out
            if "FlowMatch" in class_name:
                return FlowMatchEulerDiscreteScheduler(config_dict)

            raise ValueError(f"Unsupported scheduler class: {class_name}")

        return cls._REGISTRY[class_name](config_dict)
