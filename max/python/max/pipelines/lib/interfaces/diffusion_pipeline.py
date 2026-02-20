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

"""Pipeline utilities for MAX-optimized diffusion pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt
from max._core.driver import Device
from max.driver import CPU, Accelerator
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext
from max.interfaces.tokens import TokenBuffer
from max.nn import Module
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.tensor import Tensor
from PIL import Image
from tqdm import tqdm
from typing_extensions import Self

if TYPE_CHECKING:
    from max.engine import InferenceSession

    from ..config import PipelineConfig

CompileTarget: TypeAlias = Callable[..., Any] | Module[..., Any]
CompileDecorator: TypeAlias = Callable[[CompileTarget], "CompileWrapper"]


class DiffusionPipeline(ABC):
    """Base class for diffusion pipelines.

    Subclasses must define `components` mapping component names to ComponentModel types.
    """

    components: dict[str, type[ComponentModel]] | None = None

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        weight_paths: list[Path],
        **kwargs: Any,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.session = session
        self.devices = devices

        for name, model in self._load_sub_models(weight_paths).items():
            setattr(self, name, model)

        self.init_remaining_components()

    @abstractmethod
    def init_remaining_components(self) -> None:
        """Initialize non-ComponentModel components (e.g., image processors)."""

    @abstractmethod
    def prepare_inputs(
        self, context: PixelGenerationContext
    ) -> PixelModelInputs:
        """Prepare inputs for the pipeline."""
        raise NotImplementedError(
            f"prepare_inputs is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def execute(self, model_inputs: PixelModelInputs, **kwargs: Any) -> Any:
        """Execute the pipeline with the given model inputs.

        Args:
            model_inputs: Prepared model inputs from prepare_inputs.
            **kwargs: Additional pipeline-specific execution parameters.

        Returns:
            Pipeline-specific output (e.g., generated images).
        """
        raise NotImplementedError(
            f"execute is not implemented for {self.__class__.__name__}"
        )

    def _load_sub_models(
        self, weight_paths: list[Path]
    ) -> dict[str, ComponentModel]:
        """Load all ComponentModel sub-components defined in `components`."""
        if not self.components:
            raise ValueError(
                f"{self.__class__.__name__}.components is not set."
            )

        diffusers_config = self.pipeline_config.model.diffusers_config
        if not diffusers_config:
            raise ValueError(
                "diffusers_config is required for DiffusionPipeline."
            )

        components_config = diffusers_config.get("components")
        if not components_config:
            raise ValueError("diffusers_config['components'] is missing.")

        relative_paths = self._resolve_relative_component_paths()
        loaded_sub_models: dict[str, ComponentModel] = {}

        for name, component_cls in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            if not issubclass(component_cls, ComponentModel):
                continue

            config_dict = self._get_component_config_dict(
                components_config, name
            )
            abs_paths = self._resolve_absolute_paths(
                weight_paths, relative_paths[name]
            )

            loaded_sub_models[name] = component_cls(
                config=config_dict,
                encoding=self.pipeline_config.model.quantization_encoding,
                devices=self.devices,
                weights=load_weights(abs_paths),
            )

        return loaded_sub_models

    def _get_component_config_dict(
        self, components_config: dict[str, Any], name: str
    ) -> dict[str, Any]:
        """Extract config_dict for a named component."""
        component = components_config.get(name)
        if not component:
            raise ValueError(f"Missing config for component '{name}'.")

        config_dict = component.get("config_dict")
        if not config_dict:
            raise ValueError(f"Missing config_dict for component '{name}'.")

        return config_dict

    def _resolve_relative_component_paths(self) -> dict[str, list[str]]:
        """Group weight paths by component name (first path segment)."""
        result: dict[str, list[str]] = {}

        for path in self.pipeline_config.model.weight_path:
            path_str = str(path)
            parts = path_str.split("/")
            if len(parts) >= 2:
                component = parts[0]
                result.setdefault(component, []).append(path_str)

        if not result:
            raise ValueError(
                "No component weights found. Expected format: <component>/<file>"
            )
        return result

    def _resolve_absolute_paths(
        self, weight_paths: list[Path], relative_paths: list[str]
    ) -> list[Path]:
        """Match relative component paths to absolute weight paths."""
        absolute_paths = [
            abs_path
            for abs_path in weight_paths
            for rel_path in relative_paths
            if rel_path in str(abs_path)
        ]

        if not absolute_paths:
            raise ValueError(f"Component weights not found: {relative_paths}")
        return absolute_paths


@dataclass(kw_only=True)
class PixelModelInputs:
    """A common input container for pixel-generation models.

    This dataclass is designed to provide a consistent set of fields used
    across multiple pixel pipelines/models.
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
    input_image: Image.Image | None = None
    """
    Optional input image for image-to-image generation (PIL.Image.Image).
    """

    def __post_init__(self) -> None:
        """Basic invariant checks for core scalar fields.

        Model-specific subclasses may override __post_init__ and call
        super().__post_init__() to add stricter validations (for example,
        requiring timesteps/sigmas/latents to be non-empty).
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
    def from_context(cls, context: PixelGenerationContext) -> Self:
        """Build an instance from a context-like dict.

        Policy:

        - If a key is missing: the dataclass default applies automatically.
        - If a key is present with value None: treat as missing and substitute the class default
          (including subclass overrides).
        """
        fmap = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        for dataclass_field in fields(cls):
            name = dataclass_field.name
            if not hasattr(context, name):
                continue
            v = getattr(context, name)

            if v is None:
                if dataclass_field.default is not MISSING:
                    kwargs[name] = dataclass_field.default
                elif dataclass_field.default_factory is not MISSING:
                    kwargs[name] = dataclass_field.default_factory()
                else:
                    # No default -> keep None; for required fields this should fail downstream.
                    kwargs[name] = None
            else:
                kwargs[name] = v

        return cls(**kwargs)


class CompileWrapper:
    def __init__(
        self,
        compile_target: CompileTarget,
        input_types: Iterable[TensorType] | None = None,
    ) -> None:
        """Initialize the CompileWrapper.

        Args:
            compile_target: The function or module to be compiled.
            input_types: A list of input types (TensorTypes) required for compilation.

        Raises:
            ValueError: If input_types is not provided.
        """
        target_name = (
            compile_target.__name__
            if not isinstance(compile_target, Module)
            else type(compile_target).__name__
        )
        if input_types is None:
            raise ValueError(
                f"input_types must be provided for compilation of {target_name}."
            )

        input_types_tuple = tuple(input_types)
        self._compiled_module: Callable[..., Any] | None = None
        self._compiled_model: Model | None = None

        if isinstance(compile_target, Module):
            self._compiled_module = compile_target.compile(*input_types_tuple)
            return

        with Graph(
            compile_target.__name__, input_types=input_types_tuple
        ) as graph:
            output = compile_target(*graph.inputs)
            if isinstance(output, Iterable):
                graph.output(*output)
            else:
                graph.output(output)
            compiled_graph = graph

        device: CPU | Accelerator
        if any(input_type.device.is_gpu() for input_type in input_types_tuple):
            device = Accelerator()
        else:
            device = CPU()
        session = InferenceSession([device])
        self._compiled_model = session.load(compiled_graph)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | list[Tensor]:
        """Execute the compiled session with the given arguments.

        Args:
            *args: Positional arguments to pass to the session.
            **kwargs: Keyword arguments to pass to the session.

        Returns:
            The result of the session execution.
        """
        if self._compiled_module is not None:
            return self._compiled_module(*args, **kwargs)

        if self._compiled_model is None:
            raise RuntimeError("CompileWrapper has no compiled target.")

        normalized_args = tuple(self._unwrap_tensor(arg) for arg in args)
        normalized_kwargs = {
            key: self._unwrap_tensor(val) for key, val in kwargs.items()
        }
        buffers = self._compiled_model(*normalized_args, **normalized_kwargs)
        outputs = [Tensor.from_dlpack(buffer) for buffer in buffers]
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def _unwrap_tensor(value: Any) -> Any:
        try:
            if hasattr(value, "driver_tensor"):
                return value.driver_tensor
            return value
        except TypeError:
            return value


def max_compile(
    compile_target: CompileTarget | None = None,
    input_types: Iterable[TensorType] | None = None,
) -> CompileDecorator | CompileWrapper:
    """Decorator or function to compile a target with specified input types.

    Args:
        compile_target: The function or module to compile. If None, returns a decorator.
        input_types: The input types for the compilation.

    Returns:
        A CompileWrapper instance if compile_target is provided, otherwise a decorator.
    """
    if compile_target is None:

        def decorator(f: CompileTarget) -> CompileWrapper:
            return CompileWrapper(f, input_types)

        return decorator

    return CompileWrapper(compile_target, input_types)
