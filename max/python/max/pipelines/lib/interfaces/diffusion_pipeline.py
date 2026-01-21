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

"""Pipeline utilities for MAX-optimized pipelines."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from max.config import load_config
from max.driver import load_devices
from max.graph import DeviceRef
from max.graph.weights import load_weights
from max.pipelines.lib.interfaces.max_model import MaxModel
from tqdm import tqdm

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..diffusion_schedulers import FlowMatchEulerDiscreteScheduler


class DiffusionPipeline(ABC):
    config_name: str | None = None
    """
    The name of the config file of the pipeline.

    It can be found in the downloaded path or HuggingFace hub.
    It's usually "model_index.json" or "config.json" for Diffusion models.
    """

    components: (
        dict[str, type[MaxModel] | type[FlowMatchEulerDiscreteScheduler]] | None
    ) = None
    """The components of the pipeline.
    
    It can be found in the downloaded path or HuggingFace hub.
    It's usually contains text_encoder, tokenizer, transformer, vae, etc.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cached_folder: str,
        **kwargs: Any,
    ) -> DiffusionPipeline:
        """Load a pipeline from a pretrained model.

        Args:
            pipeline_config: Pipeline configuration for model and runtime setup.
            cached_folder: Local path to the downloaded model snapshot.
            **kwargs: Additional pipeline-specific arguments.
        """
        self.pipeline_config = pipeline_config
        self.devices = load_devices(pipeline_config.model.device_specs)

        # Load sub models
        loaded_sub_models = self.load_sub_models(cached_folder)
        for name, model in loaded_sub_models.items():
            setattr(self, name, model)

        self.init_remaining_components()

    @abstractmethod
    def init_remaining_components(self) -> None:
        pass

    def load_sub_models(
        self,
        pretrained_model_name_or_path: str | os.PathLike,
    ) -> dict:
        """Load sub-models for the pipeline.

        Args:
            pretrained_model_name_or_path: Path to pretrained model.

        Returns:
            Dictionary containing the loaded sub-models.
        """
        loaded_sub_models = {}
        if self.components is None:
            raise ValueError(
                f"`components` for {self.__class__.__name__} pipeline is not set. "
                "Please set proper components based on its sub-directories in the downloaded path."
            )
        for name, component_class in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            component_path = os.path.join(pretrained_model_name_or_path, name)
            if "tokenizer" in name:
                # NOTE: Currently, we are using tokenizers from transformers.
                # TODO(minkyu): Check if we can use Tokenizer in Max,
                # and remove this conditional path.
                loaded_sub_models[name] = component_class.from_pretrained(
                    component_path
                )
                continue

            if (
                not hasattr(component_class, "config_name")
                or component_class.config_name is None
            ):
                raise ValueError(
                    f"`config_name` for {component_class.__name__} is not set. "
                    "Please set proper config file name in the downloaded path."
                )
            config = load_config(
                f"{component_path}/{component_class.config_name}"
            )
            if issubclass(component_class, MaxModel):
                weight_paths = [
                    Path(pretrained_model_name_or_path) / weight_path
                    for weight_path in self.pipeline_config.model.weight_path
                    if weight_path.split("/")[0] == name
                ]
                loaded_sub_models[name] = component_class(
                    config=config,
                    encoding=self.pipeline_config.model.quantization_encoding,
                    devices=self.devices,
                    weights=load_weights(weight_paths),
                )
            else:
                loaded_sub_models[name] = component_class(
                    **config,
                    device=DeviceRef.from_device(self.devices[0]),
                    dtype=self.pipeline_config.model.quantization_encoding.dtype,
                )

        return loaded_sub_models

    def finalize_pipeline_config(self) -> None:
        return

    def _execution_device(self) -> DeviceRef:
        r"""Returns the device on which the pipeline's models will be executed.

        This property checks pipeline components to determine the execution device.
        It supports MAX models (with DeviceRef device attribute).
        Similar structure to diffusers' _execution_device but returns DeviceRef instead of DeviceRef.

        Returns:
            DeviceRef: The execution device (GPU if available, otherwise CPU).
        """
        # Check MAX models - prioritize GPU
        # Similar to diffusers' _execution_device but for MAX models (not torch.nn.Module)
        sub_models = {k: getattr(self, k) for k in self.components}
        for name, model in sub_models.items():
            exclude_from_cpu_offload = getattr(
                self, "_exclude_from_cpu_offload", set()
            )
            if name in exclude_from_cpu_offload:
                continue

            if hasattr(model, "device") and isinstance(model.device, DeviceRef):
                return model.device

        if hasattr(self, "device"):
            try:
                device = self.device
                if isinstance(device, DeviceRef):
                    return device
            except Exception:
                pass

        return DeviceRef.CPU()
