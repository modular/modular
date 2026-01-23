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

"""Scheduler factory for creating schedulers from diffusers configuration."""

from typing import Any


class SchedulerFactory:
    """Factory for creating schedulers from diffusers configuration."""

    @classmethod
    def create(
        cls, class_name: str, config_dict: dict[str, Any] | None = None
    ) -> Any:
        """Create a scheduler instance.

        Args:
            class_name: The diffusers scheduler class name.
            config_dict: Configuration parameters for the scheduler.

        Returns:
            A scheduler instance.

        Raises:
            ValueError: If the scheduler class is not supported.
        """
        if class_name == "FlowMatchEulerDiscreteScheduler":
            from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
            return FlowMatchEulerDiscreteScheduler(config_dict)
        else:
            raise ValueError(f"Unsupported scheduler class: {class_name}")
