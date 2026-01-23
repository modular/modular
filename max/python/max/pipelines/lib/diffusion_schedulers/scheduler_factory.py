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

import importlib
from typing import Any

# Registry mapping scheduler class names to (module_path, class_name) tuples.
# Module paths are relative to this package.
_SCHEDULER_REGISTRY: dict[str, tuple[str, str]] = {
    "FlowMatchEulerDiscreteScheduler": (
        ".scheduling_flow_match_euler_discrete",
        "FlowMatchEulerDiscreteScheduler",
    ),
}


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
        if class_name not in _SCHEDULER_REGISTRY:
            supported = ", ".join(sorted(_SCHEDULER_REGISTRY.keys()))
            raise ValueError(
                f"Unsupported scheduler class: {class_name}. "
                f"Supported schedulers: {supported}"
            )

        module_path, cls_name = _SCHEDULER_REGISTRY[class_name]
        module = importlib.import_module(module_path, package=__package__)
        scheduler_cls = getattr(module, cls_name)
        return scheduler_cls(config_dict)
