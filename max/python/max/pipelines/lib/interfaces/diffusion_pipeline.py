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

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload

import numpy as np
import numpy.typing as npt
from max._core.driver import Device
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import Graph, TensorType
from max.graph.weights import load_weights
from max.interfaces import PixelGenerationContext
from max.pipelines.lib.interfaces.component_model import ComponentModel
from tqdm import tqdm

if TYPE_CHECKING:
    from ..config import PipelineConfig
    from .cache_mixin import DenoisingCacheConfig, DenoisingCacheState

logger = logging.getLogger("max.pipelines")

CompileTarget: TypeAlias = Callable[..., Any] | Module[..., Any]
CompileDecorator: TypeAlias = Callable[[CompileTarget], "CompileWrapper"]


@dataclass
class DiffusionPipelineOutput:
    """Output of a diffusion pipeline.

    Attributes:
        images: NHWC uint8 NumPy array of shape (B, H, W, C) with values
            in [0, 255].
    """

    images: npt.NDArray[np.uint8]


class DiffusionPipeline(ABC):
    """Base class for diffusion pipelines.

    Subclasses must define `components` mapping component names to ComponentModel types.
    """

    components: dict[str, type[ComponentModel]] | None = None

    unprefixed_weight_component: str | None = None
    """When set, weight files without a ``<component>/`` prefix are assigned to
    this component.  This supports multi-repo layouts where quantized weights
    for one component (e.g. the transformer) are shipped as flat files in a
    separate repo while the remaining components use the base model repo."""

    default_num_inference_steps: int = 50
    """Default number of denoising steps when the user does not specify one.

    Subclasses may override this to provide a model-appropriate default.
    """

    default_residual_threshold: float = 0.05
    """Model-specific default for the FBCache relative difference threshold.

    Subclasses may override this to provide a model-appropriate default.
    Used when the request does not specify a ``residual_threshold``.
    """

    default_taylorseer_cache_interval: int = 5
    """Model-specific default for the TaylorSeer cache interval.

    Subclasses may override this to provide a model-appropriate default.
    Used when ``DenoisingCacheConfig.taylorseer_cache_interval`` is ``None``.
    """

    default_taylorseer_warmup_steps: int = 9
    """Model-specific default for the TaylorSeer warmup steps.

    Subclasses may override this to provide a model-appropriate default.
    Used when ``DenoisingCacheConfig.taylorseer_warmup_steps`` is ``None``.
    """

    default_taylorseer_max_order: int = 1
    """Model-specific default for the TaylorSeer expansion order.

    Subclasses may override this to provide a model-appropriate default.
    Used when ``DenoisingCacheConfig.taylorseer_max_order`` is ``None``.
    """

    default_teacache_rel_l1_thresh: float = 0.4
    """Model-specific default for the TeaCache relative-L1 threshold.

    Subclasses may override this to provide a model-appropriate default.
    Used when ``DenoisingCacheConfig.teacache_rel_l1_thresh`` is ``None``.
    """

    default_teacache_coefficients: tuple[float, ...] = (
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    )
    """Default TeaCache polynomial coefficients for FLUX-style rescaling."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        weight_paths: list[Path],
        cache_config: DenoisingCacheConfig | None = None,
        **kwargs: Any,
    ) -> None:
        from .cache_mixin import DenoisingCacheConfig

        self.cache_config: DenoisingCacheConfig = (
            cache_config or DenoisingCacheConfig()
        )
        self._resolve_cache_defaults()
        self.pipeline_config = pipeline_config
        self.session = session
        self.devices = devices

        for name, model in self._load_sub_models(weight_paths).items():
            setattr(self, name, model)

        self.init_remaining_components()

    def _resolve_cache_defaults(self) -> None:
        """Resolve nullable DenoisingCacheConfig fields using pipeline defaults.

        Uses class-level ``default_*`` attributes so that subclasses can
        override model-specific defaults.  Called before ``_load_sub_models()``
        so that ComponentModels receive a fully-resolved cache config.
        """
        # Mutates in-place; DenoisingCacheConfig is unfrozen.
        cc = self.cache_config
        if cc.taylorseer_cache_interval is None:
            cc.taylorseer_cache_interval = (
                self.default_taylorseer_cache_interval
            )
        if cc.taylorseer_warmup_steps is None:
            cc.taylorseer_warmup_steps = self.default_taylorseer_warmup_steps
        if cc.taylorseer_max_order is None:
            cc.taylorseer_max_order = self.default_taylorseer_max_order
        if cc.teacache_rel_l1_thresh is None:
            cc.teacache_rel_l1_thresh = self.default_teacache_rel_l1_thresh
        if cc.teacache_coefficients is None:
            cc.teacache_coefficients = list(self.default_teacache_coefficients)

    @abstractmethod
    def init_remaining_components(self) -> None:
        """Initialize non-ComponentModel components (e.g., image processors)."""

    @abstractmethod
    def prepare_inputs(self, context: PixelGenerationContext) -> Any:
        """Prepare inputs for the pipeline."""
        raise NotImplementedError(
            f"prepare_inputs is not implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def execute(
        self, model_inputs: Any, **kwargs: Any
    ) -> DiffusionPipelineOutput:
        """Execute the pipeline with the given model inputs.

        Args:
            model_inputs: Prepared model inputs from prepare_inputs.
            **kwargs: Additional pipeline-specific execution parameters.

        Returns:
            A DiffusionPipelineOutput containing NHWC uint8 images.
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
        pipeline_encoding = self.pipeline_config.model.quantization_encoding

        for name, component_cls in tqdm(
            self.components.items(), desc="Loading sub models"
        ):
            if not issubclass(component_cls, ComponentModel):
                continue

            config_dict = self._get_component_config_dict(
                components_config, name
            )

            if name in relative_paths:
                abs_paths = self._resolve_absolute_paths(
                    weight_paths, relative_paths[name]
                )
                encoding = pipeline_encoding
            else:
                # Component weights not provided — download from base model
                # repo.  These components (e.g. VAE, text encoder) always use
                # bfloat16 regardless of the quantization applied to the
                # primary component.
                abs_paths = self._download_component_weights(name)
                encoding = "bfloat16"

            init_params = inspect.signature(component_cls.__init__).parameters
            init_kwargs: dict[str, Any] = {
                "config": config_dict,
                "encoding": encoding,
                "devices": self.devices,
                "weights": load_weights(abs_paths),
            }
            if "session" in init_params:
                init_kwargs["session"] = self.session
            if "cache_config" in init_params:
                init_kwargs["cache_config"] = self.cache_config

            loaded_sub_models[name] = component_cls(**init_kwargs)

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
        """Group weight paths by component name (first path segment).

        Files without a ``/`` separator (i.e. flat filenames) are assigned to
        :attr:`unprefixed_weight_component` when it is set.
        """
        result: dict[str, list[str]] = {}

        for path in self.pipeline_config.model.weight_path:
            path_str = str(path)
            parts = path_str.split("/")
            if len(parts) >= 2:
                component = parts[0]
                result.setdefault(component, []).append(path_str)
            elif self.unprefixed_weight_component:
                result.setdefault(self.unprefixed_weight_component, []).append(
                    path_str
                )

        if not result:
            raise ValueError(
                "No component weights found. Expected format:"
                " <component>/<file>"
            )
        return result

    def _download_component_weights(self, component_name: str) -> list[Path]:
        """Download weight files for a component from the base model repo."""
        from max.graph.weights import WeightsFormat

        from ..hf_utils import download_weight_files

        model_repo = self.pipeline_config.model.huggingface_model_repo
        all_safetensors = model_repo.weight_files.get(
            WeightsFormat.safetensors, []
        )
        component_files = [
            f for f in all_safetensors if f.startswith(f"{component_name}/")
        ]
        if not component_files:
            raise ValueError(
                f"No weight files found for component '{component_name}' "
                f"in base model repo '{model_repo.repo_id}'."
            )
        if model_repo.repo_type == "online":
            return download_weight_files(
                huggingface_model_id=model_repo.repo_id,
                filenames=component_files,
                revision=self.pipeline_config.model.huggingface_model_revision,
                force_download=self.pipeline_config.model.force_download,
            )
        else:
            local_path = Path(model_repo.repo_id)
            return [local_path / f for f in component_files]

    # -----------------------------------------------------------------
    # Denoising cache support (FBCache + TaylorSeer)
    # -----------------------------------------------------------------

    _cache_taylor_max_order_tensor: Tensor | None = None
    _cache_dtype: DType
    _cache_device: Device

    def _init_cache_state(self, dtype: DType, device: Device) -> None:
        """Initialize pipeline-level cache tensors and TaylorSeer graphs.

        Call once during ``init_remaining_components()``, after the
        transformer has been loaded and compiled.
        """
        self._cache_taylor_max_order_tensor = None
        if self.cache_config.taylorseer:
            self._cache_taylor_max_order_tensor = Tensor(
                storage=Buffer.from_dlpack(
                    np.array(
                        [self.cache_config.taylorseer_max_order],
                        dtype=np.int32,
                    )
                ).to(device)
            )

        self._cache_dtype = dtype
        self._cache_device = device

        if self.cache_config.taylorseer:
            self.build_taylorseer(dtype, device)

    def create_cache_state(
        self,
        batch_size: int,
        seq_len: int,
        transformer_config: Any,
        text_seq_len: int = 0,
    ) -> DenoisingCacheState:
        """Create per-request cache state with fresh tensors.

        Args:
            batch_size: Batch dimension (from prompt_embeds).
            seq_len: Sequence length (from latents).
            transformer_config: Transformer config carrying dimension info.
                Must have ``num_attention_heads``, ``attention_head_dim``,
                ``patch_size``, ``out_channels``, and ``in_channels`` attributes.
            text_seq_len: Text sequence length. Reserved for cache modes that
                require text-aware allocations.
        """
        from .cache_mixin import DenoisingCacheState

        for attr in (
            "num_attention_heads",
            "attention_head_dim",
            "patch_size",
            "out_channels",
            "in_channels",
        ):
            assert hasattr(transformer_config, attr), (
                f"transformer_config missing required attribute '{attr}'"
            )

        residual_dim = (
            transformer_config.num_attention_heads
            * transformer_config.attention_head_dim
        )
        output_dim = (
            transformer_config.patch_size
            * transformer_config.patch_size
            * (
                transformer_config.out_channels
                or transformer_config.in_channels
            )
        )

        state = DenoisingCacheState()

        def _device_zeros(shape: tuple[int, ...]) -> Tensor:
            return Tensor(
                storage=Buffer.zeros(
                    shape, self._cache_dtype, device=self._cache_device
                )
            )

        if self.cache_config.first_block_caching:
            state.prev_residual = _device_zeros(
                (batch_size, seq_len, residual_dim)
            )
            state.prev_output = _device_zeros((batch_size, seq_len, output_dim))

        if self.cache_config.taylorseer:
            for attr in (
                "taylor_factor_0",
                "taylor_factor_1",
                "taylor_factor_2",
            ):
                setattr(
                    state,
                    attr,
                    _device_zeros((batch_size, seq_len, output_dim)),
                )

        if self.cache_config.teacache:
            state.teacache_prev_modulated_input = _device_zeros(
                (batch_size, seq_len, residual_dim)
            )
            state.teacache_cached_residual = _device_zeros(
                (batch_size, seq_len, residual_dim)
            )
            state.teacache_accumulated_rel_l1 = Tensor(
                storage=Buffer.from_dlpack(
                    np.array([0.0], dtype=np.float32)
                ).to(self._cache_device)
            )

        return state

    def build_taylorseer(self, dtype: DType, device: Device) -> None:
        """Build compiled graphs for TaylorSeer predict and update."""
        tensor_type = TensorType(
            dtype, shape=["batch", "seq", "channels"], device=device
        )
        scalar_type = TensorType(DType.float32, shape=[1], device=device)
        order_type = TensorType(DType.int32, shape=[1], device=device)

        self.__dict__["taylor_predict"] = max_compile(
            self.taylor_predict,
            input_types=[
                tensor_type,  # factor_0
                tensor_type,  # factor_1
                tensor_type,  # factor_2
                scalar_type,  # step_offset
                order_type,  # max_order
            ],
        )
        self.__dict__["taylor_update"] = max_compile(
            self.taylor_update,
            input_types=[
                tensor_type,  # new_output
                tensor_type,  # old_factor_0
                tensor_type,  # old_factor_1
                scalar_type,  # delta_step
                order_type,  # max_order
            ],
        )

    @staticmethod
    def taylor_predict(
        factor_0: Tensor,
        factor_1: Tensor,
        factor_2: Tensor,
        step_offset: Tensor,
        max_order: Tensor,
    ) -> Tensor:
        """Taylor series prediction: f(t+dt) ~ f(t) + f'(t)*dt + f''(t)*dt^2/2."""
        offset = F.cast(step_offset, factor_0.dtype)
        result = factor_0 + factor_1 * offset
        offset_sq_half = (
            offset
            * offset
            * F.constant(0.5, factor_0.dtype, device=factor_0.device)
        )
        order2_term = factor_2 * offset_sq_half
        use_order2 = max_order >= F.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = F.cast(
            F.broadcast_to(use_order2, order2_term.shape), order2_term.dtype
        )
        result = result + order2_term * use_order2_cast
        return result

    @staticmethod
    def taylor_update(
        new_output: Tensor,
        old_factor_0: Tensor,
        old_factor_1: Tensor,
        delta_step: Tensor,
        max_order: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Taylor factors via divided differences."""
        delta = F.cast(delta_step, new_output.dtype)
        eps = F.constant(1e-9, new_output.dtype, device=new_output.device)
        safe_delta = delta + eps

        new_factor_0 = new_output
        new_factor_1 = (new_output - old_factor_0) / safe_delta
        new_factor_2 = (new_factor_1 - old_factor_1) / safe_delta
        use_order2 = max_order >= F.constant(
            2, DType.int32, device=max_order.device
        )
        use_order2_cast = F.cast(
            F.broadcast_to(use_order2, new_factor_2.shape),
            new_factor_2.dtype,
        )
        new_factor_2 = new_factor_2 * use_order2_cast

        return new_factor_0, new_factor_1, new_factor_2

    @staticmethod
    def taylorseer_skip_transformer(
        step: int, warmup_steps: int, cache_interval: int
    ) -> bool:
        """Return True when the full transformer pass can be skipped at *step*."""
        if step < warmup_steps:
            return False
        return (step - warmup_steps - 1) % cache_interval != 0

    def run_transformer(
        self,
        cache_state: DenoisingCacheState,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        """Run the transformer for one denoising step.

        Subclasses must override this to call their transformer with the
        appropriate model-specific arguments.  The method should return
        ``(noise_pred,)`` when first_block_caching is disabled, or
        ``(new_residual, noise_pred)`` when first_block_caching is enabled.

        Args:
            cache_state: Per-request mutable cache state for this stream.
            **kwargs: Model-specific arguments forwarded from
                ``run_denoising_step``.
        """
        raise NotImplementedError

    def run_denoising_step(
        self,
        step: int,
        cache_state: DenoisingCacheState,
        device: Device,
        **kwargs: Any,
    ) -> Tensor:
        """Execute one denoising step with caching logic.

        Delegates the actual transformer call to ``self.run_transformer()``,
        which subclasses override with model-specific arguments.

        Args:
            step: Current step index.
            cache_state: Per-request mutable cache state for this stream.
            device: Target device.
            **kwargs: Model-specific arguments forwarded to
                ``run_transformer``.

        Returns:
            noise_pred tensor for this step.
        """
        cache_config = self.cache_config

        # 1. TaylorSeer scheduling decision
        skip_transformer = False
        warmup_steps = cache_config.taylorseer_warmup_steps
        cache_interval = cache_config.taylorseer_cache_interval
        max_order_tensor = self._cache_taylor_max_order_tensor
        if cache_config.taylorseer:
            assert warmup_steps is not None
            assert cache_interval is not None
            skip_transformer = self.taylorseer_skip_transformer(
                step,
                warmup_steps,
                cache_interval,
            )

        # 2. Compute TaylorSeer step delta
        taylor_delta_tensor: Tensor | None = None
        if cache_config.taylorseer:
            assert max_order_tensor is not None
            assert cache_state.taylor_factor_0 is not None
            assert cache_state.taylor_factor_1 is not None
            delta = (
                float(step - cache_state.taylor_last_compute_step)
                if cache_state.taylor_last_compute_step is not None
                else 1.0
            )
            taylor_delta_tensor = Tensor(
                storage=Buffer.from_dlpack(
                    np.array([delta], dtype=np.float32)
                ).to(device)
            )

        # 3. Predict path (skip transformer)
        if cache_config.taylorseer and skip_transformer:
            assert cache_state.taylor_factor_0 is not None
            assert cache_state.taylor_factor_1 is not None
            assert cache_state.taylor_factor_2 is not None
            assert taylor_delta_tensor is not None
            assert max_order_tensor is not None
            return self.taylor_predict(
                cache_state.taylor_factor_0,
                cache_state.taylor_factor_1,
                cache_state.taylor_factor_2,
                taylor_delta_tensor,
                max_order_tensor,
            )

        # 4. Full compute path
        result = self.run_transformer(cache_state, **kwargs)
        if cache_config.teacache:
            (
                cache_state.teacache_prev_modulated_input,
                cache_state.teacache_cached_residual,
                cache_state.teacache_accumulated_rel_l1,
                noise_pred,
            ) = result
        elif cache_config.first_block_caching:
            new_residual, noise_pred = result
            cache_state.prev_residual = new_residual
            cache_state.prev_output = noise_pred
        else:
            noise_pred = result[0]

        # 5. TaylorSeer factor update
        if cache_config.taylorseer:
            assert cache_state.taylor_factor_0 is not None
            assert cache_state.taylor_factor_1 is not None
            assert taylor_delta_tensor is not None
            assert max_order_tensor is not None
            (
                cache_state.taylor_factor_0,
                cache_state.taylor_factor_1,
                cache_state.taylor_factor_2,
            ) = self.taylor_update(
                noise_pred,
                cache_state.taylor_factor_0,
                cache_state.taylor_factor_1,
                taylor_delta_tensor,
                max_order_tensor,
            )
            cache_state.taylor_last_compute_step = step

        return noise_pred

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


class CompileWrapper:
    """Wraps a compile target with optional input type annotations."""

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
        target_name = getattr(
            compile_target, "__name__", type(compile_target).__name__
        )
        if input_types is None:
            raise ValueError(
                f"input_types must be provided for compilation of {target_name}."
            )

        input_types_tuple = tuple(input_types)
        self._compiled_model: Model | None = None
        self._compiled_module = None

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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


@overload
def max_compile(
    compile_target: CompileTarget,
    input_types: Iterable[TensorType] | None = ...,
) -> CompileWrapper: ...


@overload
def max_compile(
    compile_target: None = ...,
    input_types: Iterable[TensorType] | None = ...,
) -> CompileDecorator: ...


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
