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
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.config import load_config
from max.driver import load_devices
from max.graph import DeviceRef
from max.graph.weights import load_weights
from max.interfaces.tokens import TokenBuffer
from max.pipelines.lib.interfaces.max_model import MaxModel
from tqdm import tqdm

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from ..diffusion_schedulers import FlowMatchEulerDiscreteScheduler


T = TypeVar("T", bound="PixelModelInputs")


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


@dataclass(kw_only=True)
class PixelModelInputs:
    """
    A common input container for pixel-generation models.

    This dataclass is designed to provide a consistent set of fields used across multiple pixel pipelines/models.
    """

    tokens: TokenBuffer
    """
    Primary encoder token buffer.
    This is the main prompt representation consumed by the model's text encoder.
    Required for all models.
    """

    tokens_2: TokenBuffer | None = None
    """
    Secondary encoder token buffer (for dual-encoder models).
    Examples: architectures that have a second text encoder stream or pooled embeddings.
    If the model is single-encoder, leave as None.
    """

    negative_tokens: TokenBuffer | None = None
    """
    Negative prompt tokens for the primary encoder.
    Used for classifier-free guidance (CFG) or similar conditioning schemes.
    If your pipeline does not use negative prompts, leave as None.
    """

    negative_tokens_2: TokenBuffer | None = None
    """
    Negative prompt tokens for the secondary encoder (for dual-encoder models).
    If the model is single-encoder or you do not use negative prompts, leave as None.
    """

    extra_params: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    """
    A bag of model-specific numeric parameters not represented as explicit fields.

    Typical uses:
    - Architecture-specific knobs (e.g., cfg_normalization arrays, scaling vectors)
    - Precomputed per-step values not worth standardizing across all models
    - Small numeric tensors that are easier to carry as named extras than formal fields

    Values are expected to be numpy arrays (ndarray) to keep the contract consistent,
    but you can relax this if your codebase needs non-array values.
    """

    timesteps: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """
    Precomputed denoising timestep schedule.

    - Usually a 1D float32 numpy array of length `num_inference_steps`
      (exact semantics depend on your scheduler).
    - If your pipeline precomputes the scheduler trajectory, you pass it here.
    - Some models may not require explicit timesteps; in that case it may remain empty.
      (Model-specific subclasses can enforce non-empty via __post_init__.)
    """

    sigmas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """
    Precomputed sigma schedule for denoising.

    - Usually a 1D float32 numpy array of length `num_inference_steps`
      corresponding to the noise level per step.
    - Some schedulers are sigma-based; others are timestep-based; some use both.
    - If unused, it may remain empty unless your model subclass requires it.
    """

    latents: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """
    Initial latent noise tensor (or initial latent state).

    - For diffusion/flow models, this is typically random noise seeded per request.
    - Shape depends on model: commonly [B, C, H/8, W/8] for image latents,
      or [B, T, C, H/8, W/8] for video latents.
    - If your pipeline generates latents internally, you may leave it empty.
      (Model-specific subclasses can enforce non-empty via __post_init__.)
    """

    latent_image_ids: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """
    Optional latent image IDs / positional identifiers for latents.

    - Some pipelines attach per-latent identifiers for caching, routing, or conditioning.
    - Often used to avoid recomputation of image-id embeddings across steps.
    - If unused, it may remain empty.
    """

    height: int = 1024
    """
    Output height in pixels.

    - This is a required scalar (not None).
    - If a context provides `height=None`, `from_context()` treats that as "not provided"
      and substitutes this default value (or a subclass override).
    """

    width: int = 1024
    """
    Output width in pixels.

    - This is a required scalar (not None).
    - If a context provides `width=None`, `from_context()` treats that as "not provided"
      and substitutes this default value (or a subclass override).
    """

    num_inference_steps: int = 50
    """
    Number of denoising/inference steps.

    - This is a required scalar (not None).
    - If a context provides `num_inference_steps=None`, `from_context()` treats that as
      "not provided" and substitutes this default value (or a subclass override).
    """

    guidance_scale: float = 3.5
    """
    Guidance scale for classifier-free guidance (CFG).

    - A higher value typically increases adherence to the prompt but can reduce diversity.
    - This is expected to be a real float (not None).
    - If a context provides `guidance_scale=None`, `from_context()` substitutes the default.
    """

    guidance: npt.NDArray[np.float32] | None = None
    """
    Optional guidance tensor.

    - Some pipelines precompute guidance weights/tensors (e.g., per-token weights, per-step weights).
    - None is meaningful here: it means "no explicit guidance tensor supplied".
    - Unlike scalar fields, None is preserved (not replaced).
    """

    true_cfg_scale: float = 1.0
    """
    "True CFG" scale used by certain pipelines/models.

    - Some architectures distinguish between the user-facing guidance_scale and an internal
      scale applied to a different normalization or conditioning pathway.
    - Defaults to 1.0 for pipelines that do not use this feature.
    """

    num_warmup_steps: int = 0
    """
    Number of warmup steps.

    - Used in some schedulers/pipelines to handle initial steps differently
      (e.g., scheduler stabilization, cache warmup, etc.).
    - Must be >= 0.
    """

    num_images_per_prompt: int = 1
    """
    Number of images/videos to generate per prompt.

    - Commonly used for "same prompt, multiple samples" behavior.
    - Must be > 0.
    - For video generation, the naming may still be used for historical compatibility.
    """

    def __post_init__(self) -> None:
        """
        Basic invariant checks for core scalar fields.

        Model-specific subclasses may override __post_init__ and call super().__post_init__()
        to add stricter validations (e.g., requiring timesteps/sigmas/latents to be non-empty).
        """
        if not isinstance(self.height, int) or self.height <= 0:
            raise ValueError(
                f"height must be a positive int. Got {self.height!r}"
            )
        if not isinstance(self.width, int) or self.width <= 0:
            raise ValueError(
                f"width must be a positive int. Got {self.width!r}"
            )
        if (
            not isinstance(self.num_inference_steps, int)
            or self.num_inference_steps <= 0
        ):
            raise ValueError(
                f"num_inference_steps must be a positive int. Got {self.num_inference_steps!r}"
            )

        if (
            not isinstance(self.num_warmup_steps, int)
            or self.num_warmup_steps < 0
        ):
            raise ValueError(
                f"num_warmup_steps must be >= 0. Got {self.num_warmup_steps!r}"
            )
        if (
            not isinstance(self.num_images_per_prompt, int)
            or self.num_images_per_prompt <= 0
        ):
            raise ValueError(
                f"num_images_per_prompt must be > 0. Got {self.num_images_per_prompt!r}"
            )

        required_arrays = {
            "timesteps": self.timesteps,
            "latents": self.latents,
        }

        missing = [
            name
            for name, arr in required_arrays.items()
            if not isinstance(arr, np.ndarray) or arr.size == 0
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires non-empty numpy arrays for: {', '.join(missing)}"
            )

    @classmethod
    def from_context(cls, context: dict[str, Any]) -> PixelModelInputs:
        """
        Build an instance from a context-like dict.

        Policy:
        - If a key is missing: the dataclass default applies automatically.
        - If a key is present with value None: treat as missing and substitute the class default
          (including subclass overrides).

        """
        fmap = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        for k, v in context.items():
            if k not in fmap:
                continue

            if v is None:
                f = fmap[k]
                if f.default is not MISSING:
                    kwargs[k] = f.default
                elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
                    kwargs[k] = f.default_factory()  # type: ignore[misc]
                else:
                    # No default -> keep None; for required fields this should fail downstream.
                    kwargs[k] = None
            else:
                kwargs[k] = v

        return cls(**kwargs)
