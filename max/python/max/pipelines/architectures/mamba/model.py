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
from typing import Any, Literal, cast

import numpy as np
from max.driver import Buffer, DLPackArray, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.interfaces import LogProbabilities
from max.nn.legacy import ReturnHiddenStates, ReturnLogits
from max.nn.legacy.kv_cache import KVCacheInputs
from .ssm_state_cache import SSMStateCacheInputs, SSMStateValues
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


class MambaInputs(ModelInputs):
    """A class representing inputs for the Mamba model.

    This class encapsulates the input tensors required for the Mamba model
    execution.
    """

    tokens: Buffer
    """Buffer containing the input token IDs."""

    input_row_offsets: Buffer
    """Buffer containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer

    data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None
    """Buffer containing the data parallel splits."""

    # For Mamba without SSM state caching, we need to track all tokens
    # so we can reprocess the full sequence each step
    accumulated_tokens: Buffer | None = None
    """All tokens seen so far (prompt + generated). Used for reprocessing in step mode."""

    # SSM state cache for efficient autoregressive generation
    ssm_state_cache: SSMStateCacheInputs | None = None
    """SSM state cache for structured state management during autoregressive generation."""

    def __init__(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        signal_buffers: list[Buffer],
        return_n_logits: Buffer,
        lora_ids: Buffer | None = None,
        lora_ranks: Buffer | None = None,
        lora_grouped_offsets: Buffer | None = None,
        num_active_loras: Buffer | None = None,
        lora_end_idx: Buffer | None = None,
        batch_seq_len: Buffer | None = None,
        lora_ids_kv: Buffer | None = None,
        lora_grouped_offsets_kv: Buffer | None = None,
        data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None,
        accumulated_tokens: Buffer | None = None,
        ssm_state_cache: SSMStateCacheInputs | None = None,
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
        self.ssm_state_cache = ssm_state_cache


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
            # Convert data_parallel_splits to Buffer if needed
            if isinstance(model_inputs.data_parallel_splits, DLPackArray):
                splits_tensor = model_inputs.data_parallel_splits
            else:
                # Convert Sequence[Sequence[int]] to flat array
                splits = cast(
                    Sequence[Sequence[int]],
                    model_inputs.data_parallel_splits,
                )
                splits_array = np.concatenate(
                    [
                        np.array(s, dtype=np.int64)
                        for s in splits
                    ]
                )
                splits_tensor = Buffer.from_numpy(splits_array).to(
                    self.devices[0]
                )
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                splits_tensor,
                *model_inputs.ssm_state_cache,  # type: ignore[misc]
            )
        elif self._lora_manager:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                model_inputs.lora_ids,  # type: ignore[arg-type]
                model_inputs.lora_ranks,  # type: ignore[arg-type]
                model_inputs.lora_grouped_offsets,  # type: ignore[arg-type]
                model_inputs.lora_end_idx,  # type: ignore[arg-type]
                model_inputs.batch_seq_len,  # type: ignore[arg-type]
                model_inputs.lora_ids_kv,  # type: ignore[arg-type]
                model_inputs.lora_grouped_offsets_kv,  # type: ignore[arg-type]
                *model_inputs.ssm_state_cache,  # type: ignore[misc]
                *model_inputs.signal_buffers,
            )
        else:
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                *model_inputs.ssm_state_cache,  # type: ignore[misc]
                *model_inputs.signal_buffers,
            )

        has_offsets = self.return_logits in (
            ReturnLogits.VARIABLE,
            ReturnLogits.ALL,
        )
        has_hidden_states = self.return_hidden_states != ReturnHiddenStates.NONE

        assert isinstance(model_outputs[0], DLPackArray)
        if has_offsets and has_hidden_states:
            assert len(model_outputs) == 4
            assert isinstance(model_outputs[1], DLPackArray)
            assert isinstance(model_outputs[2], DLPackArray)
            assert isinstance(model_outputs[3], DLPackArray)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif has_offsets:
            assert len(model_outputs) == 3
            assert isinstance(model_outputs[1], DLPackArray)
            assert isinstance(model_outputs[2], DLPackArray)
            return ModelOutputs(
                logits=model_outputs[1],
                next_token_logits=model_outputs[0],
                logit_offsets=model_outputs[2],
            )
        elif has_hidden_states:
            assert len(model_outputs) == 2
            assert isinstance(model_outputs[1], DLPackArray)
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
        # For Mamba, use current_position to include all tokens (prompt + generated)
        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.current_position for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        # For Mamba (no KV cache), we need ALL tokens including generated ones
        # because the model reprocesses the entire sequence each step.
        # Use current_position to get all tokens (prompt + generated)
        tokens_list = []
        for ctx in context_batch:
            # Get all tokens up to current position (prompt + generated)
            all_tokens = ctx.tokens.array[: ctx.tokens.current_position]
            tokens_list.append(all_tokens)

        tokens_np = np.concatenate(tokens_list)
        tokens = Buffer.from_numpy(tokens_np).to(self.devices[0])

        # Constructs splits for the data parallel execution.
        if dp > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
        else:
            data_parallel_splits = None

        # Allocate SSM state cache for Mamba
        from .ssm_state_cache import SSMStateCacheParams

        batch_size = len(context_batch)
        ssm_cache_params = SSMStateCacheParams(
            dtype=self.dtype,
            num_layers=self.huggingface_config.num_hidden_layers,
            intermediate_size=self.huggingface_config.intermediate_size,
            d_state=self.huggingface_config.state_size,
            conv_kernel=self.huggingface_config.conv_kernel,
            device=DeviceRef.from_device(self.devices[0]),
        )
        ssm_state_cache = ssm_cache_params.allocate_cache(batch_size=batch_size)

        inputs = MambaInputs(
            tokens=tokens,
            input_row_offsets=Buffer.from_numpy(input_row_offsets).to(
                self.devices[0]
            ),
            signal_buffers=self.signal_buffers,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=data_parallel_splits,
            # Store accumulated tokens for reprocessing in subsequent steps
            accumulated_tokens=tokens,
            ssm_state_cache=ssm_state_cache,
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
        next_tokens: Buffer,
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
            accumulated_tokens = Buffer.from_numpy(accumulated_np).to(
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
        input_row_offsets = Buffer.from_numpy(row_offsets).to(self.devices[0])

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
            ssm_state_cache=prev_model_inputs.ssm_state_cache,
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
        self._input_row_offsets_prealloc = Buffer.from_numpy(
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
            cache_dtype=None,
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

        # Buffer Parallel case
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
                        conv_state,
                        ssm_state,
                        seqlen_offset,
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
                    # Construct SSM state cache values for graph execution
                    ssm_state_cache = SSMStateValues(
                        conv_state=conv_state,  # type: ignore[arg-type]
                        ssm_state=ssm_state,  # type: ignore[arg-type]
                        seqlen_offset=seqlen_offset,  # type: ignore[arg-type]
                    )
                    outputs = single_model(
                        tokens.tensor,
                        return_n_logits.tensor,
                        input_row_offsets.tensor,
                        ssm_state_cache=ssm_state_cache,
                    )
                else:
                    (
                        tokens,
                        input_row_offsets,
                        return_n_logits,
                        conv_state,
                        ssm_state,
                        seqlen_offset,
                    ) = graph.inputs
                    # Construct SSM state cache values for graph execution
                    ssm_state_cache = SSMStateValues(
                        conv_state=conv_state,  # type: ignore[arg-type]
                        ssm_state=ssm_state,  # type: ignore[arg-type]
                        seqlen_offset=seqlen_offset,  # type: ignore[arg-type]
                    )
                    outputs = single_model(
                        tokens.tensor,
                        return_n_logits.tensor,
                        input_row_offsets.tensor,
                        ssm_state_cache=ssm_state_cache,
                    )
                graph.output(*outputs)
                return graph

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


class MambaModelNewInputs(ModelInputs):
    """Inputs for the non-legacy Mamba pipeline model."""

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


class MambaModelNew(PipelineModel[TextContext]):
    """Non-legacy Mamba pipeline model with incremental SSM state caching.

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
        module,
        input_types,
        weights,
        custom_extensions,
    ):
        """Compile a Module into a callable, loading custom kernel extensions.

        This replicates Module.compile() but creates the Graph with
        custom_extensions so state_space kernels are available for
        verification during graph construction.
        """
        import functools

        from max import functional as F
        from max.driver import CPU
        from max.graph import Graph, TensorType
        from max._realization_context import (
            GraphRealizationContext,
            _session,
        )
        from max.tensor import Tensor, realization_context

        graph = Graph(
            type(module).__qualname__,
            input_types=input_types,
            custom_extensions=custom_extensions,
        )
        with realization_context(GraphRealizationContext(graph)) as ctx, ctx:
            inputs = [
                Tensor.from_graph_value(inp) for inp in graph.inputs
            ]

            weight_names_used: list[str] = []

            def as_weight(name: str, tensor: Tensor):
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
                logger.info(
                    f"State dict has unused weights: {len(extra)} keys"
                )

            if isinstance(outputs, Tensor):
                graph.output(outputs)
                unary = True
            else:
                graph.output(*outputs)
                unary = False

        session = _session()
        compiled = F.functional(
            session.load(graph, weights_registry=weights)
        )

        if unary:
            return functools.wraps(module)(
                lambda *inputs: compiled(*inputs)[0]
            )
        return compiled

    @traced
    def _load_models(self):
        from max import functional as F
        from max.dtype import DType
        from max.graph import TensorType
        from max.pipelines.lib import CompilationTimer

        from .mamba_module import MambaPrefill, MambaStep

        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.max_batch_size + 1, dtype=np.uint32
            )
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
        assert isinstance(model_inputs, MambaModelNewInputs)

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
    ) -> MambaModelNewInputs:
        if len(replica_batches) != 1:
            raise ValueError("Non-legacy Mamba does not support DP>1")

        context_batch = replica_batches[0]

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate(
            [ctx.tokens.active for ctx in context_batch]
        )

        return MambaModelNewInputs(
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
    ) -> MambaModelNewInputs:
        prev = cast(MambaModelNewInputs, prev_model_inputs)

        return MambaModelNewInputs(
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

        assert isinstance(model_inputs, MambaModelNewInputs)

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
