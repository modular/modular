# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from max.driver import Device, Tensor
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn import ReturnHiddenStates, ReturnLogits
from max.nn.kv_cache import KVCacheInputs
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
from max.support.algorithm import flatten2d
from transformers import AutoConfig

from .data_parallel_mamba import compute_data_parallel_splits
from .data_parallel_mamba import create_graph as create_data_parallel_graph
from .distributed_mamba import DistributedMamba
from .mamba import Mamba
from .model_config import MambaConfig

logger = logging.getLogger("max.pipelines")

# Environment variable name for Mojo import paths set by the build system.
_MODULAR_MOJO_MAX_IMPORT_PATH = "MODULAR_MOJO_MAX_IMPORT_PATH"


def _get_kernel_library_paths() -> list[Path]:
    """Returns kernel library paths from the build environment.

    Reads the ``MODULAR_MOJO_MAX_IMPORT_PATH`` environment variable set by the
    Bazel build system and extracts paths to ``.mojopkg`` kernel libraries.
    This is required for Mamba models because they use custom kernels
    (causal_conv1d, selective_scan_fwd) that must be explicitly loaded.

    The function looks for MambaKernelAPI which contains only the mamba-specific
    kernels. This avoids conflicts with default kernels like rms_norm that are
    already registered by the runtime.

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

        # If it's already a .mojopkg file, check if it's MambaKernelAPI
        if entry_path.suffix == ".mojopkg":
            if "MambaKernelAPI" in entry_path.name:
                resolved_path = entry_path.resolve()
                logger.info(f"Loading kernel library: {resolved_path}")
                paths.append(resolved_path)
            continue

        # If it's a directory, search recursively for MambaKernelAPI.mojopkg files
        if entry_path.is_dir():
            # Search recursively for MambaKernelAPI.mojopkg files
            found = False
            for mojopkg in entry_path.rglob("*.mojopkg"):
                if "MambaKernelAPI" in mojopkg.name and (
                    mojopkg.is_file() or mojopkg.is_symlink()
                ):
                    resolved_path = mojopkg.resolve()
                    logger.info(f"Loading kernel library: {resolved_path}")
                    paths.append(resolved_path)
                    found = True
                    # Continue searching in case there are multiple (shouldn't happen, but be safe)

            if not found:
                pass

    if not paths:
        logger.warning(
            f"No MambaKernelAPI.mojopkg found in MODULAR_MOJO_MAX_IMPORT_PATH: {import_path_env}"
        )
    else:
        logger.info(
            f"Found {len(paths)} MambaKernelAPI.mojopkg file(s): {[str(p) for p in paths]}"
        )
    return paths


class MambaInputs(ModelInputs):
    """A class representing inputs for the Mamba model.

    This class encapsulates the input tensors required for the Mamba model
    execution.
    """

    tokens: Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets: Tensor
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Tensor

    data_parallel_splits: Tensor | Sequence[Sequence[int]] | None = None
    """Tensor containing the data parallel splits."""

    # For Mamba without SSM state caching, we need to track all tokens
    # so we can reprocess the full sequence each step
    accumulated_tokens: Tensor | None = None
    """All tokens seen so far (prompt + generated). Used for reprocessing in step mode."""

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor,
        signal_buffers: list[Tensor],
        return_n_logits: Tensor,
        lora_ids: Tensor | None = None,
        lora_ranks: Tensor | None = None,
        lora_grouped_offsets: Tensor | None = None,
        num_active_loras: Tensor | None = None,
        lora_end_idx: Tensor | None = None,
        batch_seq_len: Tensor | None = None,
        lora_ids_kv: Tensor | None = None,
        lora_grouped_offsets_kv: Tensor | None = None,
        data_parallel_splits: Tensor | Sequence[Sequence[int]] | None = None,
        accumulated_tokens: Tensor | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets: Input row offsets (ragged tensors).
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
            accumulated_tokens: All tokens seen so far for reprocessing.
        """
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.signal_buffers = signal_buffers
        self.return_n_logits = return_n_logits
        self.lora_ids = lora_ids
        self.lora_ranks = lora_ranks
        self.lora_grouped_offsets = lora_grouped_offsets
        self.num_active_loras = num_active_loras
        self.lora_end_idx = lora_end_idx
        self.batch_seq_len = batch_seq_len
        self.lora_ids_kv = lora_ids_kv
        self.lora_grouped_offsets_kv = lora_grouped_offsets_kv
        self.data_parallel_splits = data_parallel_splits
        self.accumulated_tokens = accumulated_tokens


class MambaModelBase(PipelineModel[TextContext]):
    """Base Mamba pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
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
            return_hidden_states,
        )
        self.model = self.load_model(session)
        self.logprobs_device = devices[0]
        self.logprobs_model = self.load_logprobs_model(session)

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return MambaConfig.get_num_layers(huggingface_config)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MambaInputs)

        if self.pipeline_config.model.data_parallel_degree > 1:
            assert model_inputs.data_parallel_splits is not None
            # Convert data_parallel_splits to Tensor if needed
            if isinstance(model_inputs.data_parallel_splits, Tensor):
                splits_tensor = model_inputs.data_parallel_splits
            else:
                # Convert Sequence[Sequence[int]] to flat array
                splits_array = np.concatenate(
                    [
                        np.array(s, dtype=np.int64)
                        for s in model_inputs.data_parallel_splits
                    ]
                )
                splits_tensor = Tensor.from_numpy(splits_array).to(
                    self.devices[0]
                )
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                splits_tensor,
            )
        elif self._lora_manager:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.lora_ids,  # type: ignore
                model_inputs.lora_ranks,  # type: ignore
                model_inputs.lora_grouped_offsets,  # type: ignore
                model_inputs.num_active_loras,  # type: ignore
                model_inputs.lora_end_idx,  # type: ignore
                model_inputs.batch_seq_len,  # type: ignore
                model_inputs.lora_ids_kv,  # type: ignore
                model_inputs.lora_grouped_offsets_kv,  # type: ignore
                *model_inputs.signal_buffers,
            )
        else:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                *model_inputs.signal_buffers,
            )

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        assert isinstance(model_outputs[0], Tensor)
        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            assert isinstance(model_outputs[3], Tensor)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            assert isinstance(model_outputs[1], Tensor)
            assert isinstance(model_outputs[2], Tensor)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            assert isinstance(model_outputs[1], Tensor)
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
                hidden_states=model_outputs[1],
            )
        else:
            assert len(model_outputs) == 1
            return ModelOutputs(
                logits=model_outputs[0],
                next_token_logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs
        | None = None,  # Mamba doesn't use KV cache
        return_n_logits: int = 1,
    ) -> MambaInputs:
        """Prepare the inputs for the first pass in multistep execution."""
        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        context_batch = flatten2d(replica_batches)

        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = Tensor.from_numpy(
            np.concatenate([ctx.tokens.active for ctx in context_batch])
        ).to(self.devices[0])

        # Constructs splits for the data parallel execution.
        if dp > 1:
            data_parallel_splits = Tensor.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        inputs = MambaInputs(
            tokens=tokens,
            input_row_offsets=Tensor.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=self.signal_buffers,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=data_parallel_splits,
            # Store accumulated tokens for reprocessing in subsequent steps
            accumulated_tokens=tokens,
        )

        # Map model names to LoRA graph inputs
        if self._lora_manager:
            (
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
            ) = self._lora_manager.get_lora_graph_inputs(
                context_batch, input_row_offsets, self.devices[0]
            )

            inputs.lora_ids = lora_ids
            inputs.lora_ranks = lora_ranks
            inputs.lora_grouped_offsets = lora_grouped_offsets
            inputs.num_active_loras = num_active_loras
            inputs.lora_end_idx = lora_end_idx
            inputs.batch_seq_len = batch_seq_len
            inputs.lora_ids_kv = lora_ids_kv
            inputs.lora_grouped_offsets_kv = lora_grouped_offsets_kv

        return inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> MambaInputs:
        """Prepare the inputs for the next token in multistep execution.

        For Mamba models without SSM state caching, we need to reprocess
        the entire sequence (prompt + all generated tokens) each step.
        This is necessary because Mamba relies on sequential state that
        is not persisted between calls without explicit state caching.
        """
        assert isinstance(prev_model_inputs, MambaInputs)

        # Concatenate new token(s) to accumulated tokens
        # This ensures the model sees the full context each step
        if prev_model_inputs.accumulated_tokens is not None:
            # Get previous tokens as numpy, append new token, convert back
            prev_tokens_np = prev_model_inputs.accumulated_tokens.to_numpy()
            next_tokens_np = next_tokens.to_numpy()
            accumulated_np = np.concatenate([prev_tokens_np, next_tokens_np])
            accumulated_tokens = Tensor.from_numpy(accumulated_np).to(
                self.devices[0]
            )
        else:
            accumulated_tokens = next_tokens

        # Update row offsets for the accumulated sequence
        # For batch_size=1: offsets = [0, accumulated_length]
        batch_size = prev_model_inputs.input_row_offsets.shape[0] - 1
        accumulated_length = accumulated_tokens.shape[0] // batch_size

        # Create row offsets for the accumulated sequence
        # Assuming batch_size=1 for now (most common case)
        row_offsets = np.array(
            [i * accumulated_length for i in range(batch_size + 1)],
            dtype=np.uint32,
        )
        input_row_offsets = Tensor.from_numpy(row_offsets).to(self.devices[0])

        return MambaInputs(
            tokens=accumulated_tokens,
            input_row_offsets=input_row_offsets,
            signal_buffers=self.signal_buffers,
            return_n_logits=prev_model_inputs.return_n_logits,
            lora_ids=prev_model_inputs.lora_ids,
            lora_ranks=prev_model_inputs.lora_ranks,
            lora_grouped_offsets=prev_model_inputs.lora_grouped_offsets,
            num_active_loras=prev_model_inputs.num_active_loras,
            lora_end_idx=prev_model_inputs.lora_end_idx,
            batch_seq_len=prev_model_inputs.batch_seq_len,
            lora_ids_kv=prev_model_inputs.lora_ids_kv,
            lora_grouped_offsets_kv=prev_model_inputs.lora_grouped_offsets_kv,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
            accumulated_tokens=accumulated_tokens,
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return MambaConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph(self.weights, self.adapter)
        after_build = time.perf_counter()

        logger.info(f"Building graph took {after_build - before:.6f} seconds")

        before_compile = time.perf_counter()
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()

        logger.info(
            f"Compiling model took {after - before_compile:.6f} seconds"
        )

        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )

        return model

    @traced
    def load_logprobs_model(self, session: InferenceSession) -> Model:
        # TODO: Perhaps 'levels' ought to be configurable.
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def _get_state_dict(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> dict[str, WeightData]:
        # Get Config
        huggingface_config = self.huggingface_config
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}

        return state_dict

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        # Retrieve config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = MambaConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            norm_method=self.norm_method,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

        if model_config.data_parallel_degree > 1:
            graph, new_state_dict = create_data_parallel_graph(
                model_config, state_dict
            )
            self.state_dict = new_state_dict
            return graph

        # Tensor Parallel case
        if len(self.devices) > 1:
            dist_model: DistributedMamba = DistributedMamba(model_config)

            # Load weights.
            dist_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,
            )

            self.state_dict = dist_model.state_dict()

            with Graph(
                getattr(self.huggingface_config, "model_type", "mamba"),
                input_types=dist_model.input_types(),
                custom_extensions=_get_kernel_library_paths(),
            ) as graph:
                tokens, input_row_offsets, return_n_logits, *variadic_args = (
                    graph.inputs
                )

                # Multi-GPU passes a signal buffer per device: unmarshal these.
                signal_buffers = [
                    v.buffer for v in variadic_args[: len(self.devices)]
                ]

                outputs = dist_model(
                    tokens.tensor,
                    signal_buffers,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph

        # Single GPU case
        else:
            single_model: Mamba = Mamba(model_config)

            if self._lora_manager:
                self._lora_manager.init_weights(single_model, state_dict)

            # Load weights.
            logger.info(f"Loading {len(state_dict)} weights into Mamba model")

            # Check for weight name mismatches
            model_weights = set(single_model.state_dict().keys())
            provided_weights = set(state_dict.keys())
            missing = model_weights - provided_weights
            extra = provided_weights - model_weights
            # Log model weights containing 'output' or 'embedding'
            emb_out_weights = [
                w for w in model_weights if "output" in w or "embedding" in w
            ]
            logger.info(
                f"Model embedding/output weights: {sorted(emb_out_weights)}"
            )
            if missing:
                logger.info(
                    f"Weights expected but not in state_dict (will use defaults): {sorted(missing)}"
                )
            if extra:
                logger.warning(
                    f"Extra weights (provided but not expected): {list(extra)[:5]}{'...' if len(extra) > 5 else ''}"
                )

            single_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,
            )
            logger.info(
                f"Model state dict has {len(single_model.state_dict())} weights after loading"
            )
            self.state_dict = single_model.state_dict()

            with Graph(
                "mamba",
                input_types=single_model.input_types(self._lora_manager),
                custom_extensions=_get_kernel_library_paths(),
            ) as graph:
                if self._lora_manager:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        lora_ids,
                        lora_ranks,
                        lora_grouped_offsets,
                        num_active_loras,
                        lora_end_idx,
                        batch_seq_len,
                        lora_ids_kv,
                        lora_grouped_offsets_kv,
                    ) = graph.inputs
                    self._lora_manager.set_graph_info(
                        lora_ids.tensor,
                        lora_ranks.tensor,
                        lora_grouped_offsets.tensor,
                        num_active_loras.tensor,
                        lora_end_idx.tensor,
                        batch_seq_len.tensor,
                        lora_ids_kv.tensor,
                        lora_grouped_offsets_kv.tensor,
                    )
                else:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                    ) = graph.inputs
                outputs = single_model(
                    tokens.tensor,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, MambaInputs)
        mamba_inputs: MambaInputs = model_inputs

        sampled_tokens = next_tokens.to_numpy()
        tokens = mamba_inputs.tokens.to_numpy()
        input_row_offsets = mamba_inputs.input_row_offsets.to_numpy()

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


class MambaModel(MambaModelBase):
    """Mamba pipeline model implementation."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states,
        )
