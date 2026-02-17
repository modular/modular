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

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.legacy import ReturnLogits
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from max.profiler import traced
from transformers import AutoConfig

from .model_config import MambaConfig

logger = logging.getLogger("max.pipelines")

# Environment variable name for Mojo import paths set by the build system

_MODULAR_MOJO_MAX_IMPORT_PATH = "MODULAR_MOJO_MAX_IMPORT_PATH"


def _get_kernel_library_paths() -> list[Path]:
    """Returns kernel library paths from the build environment.

    Reads the ``MODULAR_MOJO_MAX_IMPORT_PATH`` environment variable set by the
    Bazel build system and extracts paths to ``.mojopkg`` kernel libraries.
    This is required for Mamba models because they use custom kernels
    (causal_conv1d, selective_scan_fwd) that must be explicitly loaded.

    The function looks for the state_space package which contains the
    mamba-specific kernels.

    Returns:
        A list of Path objects pointing to ``.mojopkg`` kernel libraries.
        Returns an empty list if the environment variable is not set.
    """
    import_path_env = os.environ.get(_MODULAR_MOJO_MAX_IMPORT_PATH, "")
    if not import_path_env:
        logger.warning(
            "MODULAR_MOJO_MAX_IMPORT_PATH not set, no custom kernels will be loaded"
        )
        return []

    paths: list[Path] = []

    for entry in import_path_env.split(","):
        if not entry.strip():
            continue

        entry_path = Path(entry.strip())

        # Handle relative paths - try to resolve them relative to current working directory
        if not entry_path.is_absolute():
            # Try resolving relative to current directory first
            resolved = Path.cwd() / entry_path
            if not resolved.exists():
                # If that doesn't work, try as-is (might be relative to runfiles root)
                resolved = entry_path
            entry_path = resolved

        if not entry_path.exists():
            continue

        # If it's already a .mojopkg file, check if it's state_space
        if entry_path.suffix == ".mojopkg":
            if "state_space" in entry_path.name:
                resolved_path = entry_path.resolve()
                logger.info(f"Loading kernel library: {resolved_path}")
                paths.append(resolved_path)
            continue

        # If it's a directory, search recursively for state_space.mojopkg files
        if entry_path.is_dir():
            for mojopkg in entry_path.rglob("*.mojopkg"):
                if "state_space" in mojopkg.name and (
                    mojopkg.is_file() or mojopkg.is_symlink()
                ):
                    resolved_path = mojopkg.resolve()
                    logger.info(f"Loading kernel library: {resolved_path}")
                    paths.append(resolved_path)

    if not paths:
        logger.warning(
            f"No state_space.mojopkg found in MODULAR_MOJO_MAX_IMPORT_PATH: {import_path_env}"
        )
    else:
        logger.info(
            f"Found {len(paths)} state_space.mojopkg file(s): {[str(p) for p in paths]}"
        )
    return paths


class MambaModelInputs(ModelInputs):
    """Inputs for the Mamba pipeline model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    is_prefill: bool
    layer_states: list[Buffer]

    def __init__(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        is_prefill: bool = True,
        layer_states: list[Buffer] | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.is_prefill = is_prefill
        self.layer_states = layer_states or []


class MambaModel(PipelineModel[TextContext]):
    """Mamba pipeline model with incremental SSM state caching.

    Uses separate compiled prefill and step models. The prefill model
    processes the full prompt and extracts per-layer conv/ssm states.
    The step model processes single new tokens using cached states.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self._prefill_model, self._step_model = self._load_models()
        self.logprobs_device = devices[0]
        self.logprobs_model = self._load_logprobs_model(session)

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return MambaConfig.get_num_layers(huggingface_config)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return MambaConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    def _load_logprobs_model(self, session: InferenceSession) -> Model:
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    @staticmethod
    def _compile_module_with_extensions(
        module: Any,
        input_types: list[Any],
        weights: dict[str, Any],
        custom_extensions: list[Any],
    ) -> Any:
        """Compile a Module into a callable, loading custom kernel extensions.

        This replicates Module.compile() but creates the Graph with
        custom_extensions so state_space kernels are available for
        verification during graph construction.
        """
        import functools

        from max import functional as F
        from max._realization_context import (
            GraphRealizationContext,
            _session,
        )
        from max.driver import CPU
        from max.graph import Graph, TensorType
        from max.tensor import Tensor, realization_context

        graph = Graph(
            type(module).__qualname__,
            input_types=input_types,
            custom_extensions=custom_extensions,
        )
        with realization_context(GraphRealizationContext(graph)) as ctx, ctx:
            inputs = [Tensor.from_graph_value(inp) for inp in graph.inputs]

            weight_names_used: list[str] = []

            def as_weight(name: str, tensor: Tensor) -> Any:
                weight_names_used.append(name)
                wtype = TensorType(tensor.dtype, tensor.shape, CPU())
                return F.constant_external(name, wtype).to(tensor.device)

            with module._mapped_parameters(as_weight):
                outputs = module(*inputs)

            weight_keys = set(weights.keys())
            used_keys = set(weight_names_used)
            missing = used_keys - weight_keys
            extra = weight_keys - used_keys
            if missing:
                logger.warning(
                    f"Module needs weights not in state_dict: {missing}"
                )
            if extra:
                logger.info(f"State dict has unused weights: {len(extra)} keys")

            if isinstance(outputs, Tensor):
                graph.output(outputs)
                unary = True
            else:
                graph.output(*outputs)
                unary = False

        session = _session()
        compiled = F.functional(session.load(graph, weights_registry=weights))

        if unary:
            return functools.wraps(module)(lambda *inputs: compiled(*inputs)[0])
        return compiled

    @traced
    def _load_models(self) -> tuple[Any, Any]:
        from max import functional as F
        from max.dtype import DType
        from max.graph import TensorType
        from max.pipelines.lib import CompilationTimer

        from .mamba_module import MambaPrefill, MambaStep

        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        # Prepare state dict
        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        model_config = MambaConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=1,
            norm_method="rms_norm",
            cache_dtype=None,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # Store config values for state tensor types
        self._model_config = model_config
        num_layers = model_config.num_hidden_layers
        intermediate = model_config.intermediate_size
        d_state = model_config.d_state
        conv_width = model_config.conv_kernel

        # Get kernel library paths for custom extensions
        kernel_paths = _get_kernel_library_paths()

        # --- Compile prefill model ---
        timer = CompilationTimer("prefill")
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        row_offsets_type = TensorType(
            DType.uint32, shape=["row_offsets_len"], device=device_ref
        )

        from max.tensor import default_dtype

        with F.lazy(), default_dtype(self.dtype):
            prefill_module = MambaPrefill(model_config)
            prefill_module.to(device0)

        timer.mark_build_complete()
        prefill_model = self._compile_module_with_extensions(
            prefill_module,
            [tokens_type, row_offsets_type],
            state_dict,
            kernel_paths,
        )
        timer.done()

        # --- Compile step model ---
        timer = CompilationTimer("step")
        step_tokens_type = TensorType(
            DType.int64, shape=["batch"], device=device_ref
        )
        layer_state_types: list[TensorType] = []
        for _ in range(num_layers):
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", intermediate, conv_width],
                    device=device_ref,
                )
            )
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", intermediate, d_state],
                    device=device_ref,
                )
            )

        with F.lazy(), default_dtype(self.dtype):
            step_module = MambaStep(model_config)
            step_module.to(device0)

        timer.mark_build_complete()
        step_model = self._compile_module_with_extensions(
            step_module,
            [step_tokens_type, *layer_state_types],
            state_dict,
            kernel_paths,
        )
        timer.done()

        return prefill_model, step_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MambaModelInputs)

        if model_inputs.is_prefill:
            outputs = self._prefill_model(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
            )
        else:
            outputs = self._step_model(
                model_inputs.tokens,
                *model_inputs.layer_states,
            )

        # First output is logits, rest are layer states
        logits = cast(Buffer, outputs[0].driver_tensor)

        # Store layer states for next step
        self._layer_states = [s.driver_tensor for s in outputs[1:]]

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> MambaModelInputs:
        if len(replica_batches) != 1:
            raise ValueError("Mamba does not support DP>1")

        context_batch = replica_batches[0]

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        return MambaModelInputs(
            tokens=Buffer.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            is_prefill=True,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> MambaModelInputs:
        prev = cast(MambaModelInputs, prev_model_inputs)

        return MambaModelInputs(
            tokens=next_tokens,
            input_row_offsets=prev.input_row_offsets,
            return_n_logits=prev.return_n_logits,
            is_prefill=False,
            layer_states=self._layer_states,
        )

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, MambaModelInputs)

        sampled_tokens = next_tokens.to_numpy()
        tokens = model_inputs.tokens.to_numpy()
        input_row_offsets = model_inputs.input_row_offsets.to_numpy()

        return compute_log_probabilities_ragged(
            self.logprobs_device,
            self.logprobs_model,
            input_row_offsets=input_row_offsets,
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,
        )
