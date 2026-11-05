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
"""DeepSeek V4.1 roles, deployment constants, and request-state batching.

The registered graph has a new ABI name: deployment constants precede the
per-request state inputs instead of interleaving the original variadic tail.
The legacy module and DSpark builder remain available to existing pairings.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar, cast

import numpy as np
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferType, TensorType
from max.nn.ep_config import EPConfig
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.nn.legacy.comm.ep import EPBatchManager
from max.nn.outputs import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ModelOutputConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    RaggableContext,
    build_single_replica_ragged_token_arrays,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.module_registration import (
    DecoderBatchProcessor,
    DecoderModelInputs,
)
from max.pipelines.lib.registration_runtime import (
    ConstantInputs,
    ExpertParallelDeploymentConfig,
    ModuleRuntime,
)
from max.pipelines.lib.request_state import RequestStateStore
from max.pipelines.lib.role_contracts import DecoderInputs, DecoderOutputs
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.pipelines.modeling.types import RequestID
from max.tensor import Tensor, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig
from typing_extensions import Self

from .batch_processor import DeepseekV41BatchProcessor
from .deepseekV41 import DeepseekV41
from .inputs import v41_config
from .interface import extra_input_names, extra_inputs
from .model import (
    build_ep_config,
    check_parallel_layout,
    engram_static_inputs,
    serving_token_map,
    validate_engram_tables,
)
from .model_config import DeepseekV41Config


def _is_constant(name: str) -> bool:
    return name == "engram_token_map" or name.startswith("engram_table_")


@dataclass(kw_only=True)
class RegisteredDeepseekV41Config(
    DeepseekV41Config, ExpertParallelDeploymentConfig
):
    """Completes the registered decoder's checkpoint and deployment inputs."""

    ep_dispatch_token_layout: ClassVar[str] = "tp_sharded"
    export_ep_init: ClassVar[bool] = True

    pipeline_config: PipelineConfig | None = field(default=None, repr=False)
    text_config: AutoConfig | None = field(default=None, repr=False)
    token_map: np.ndarray | None = field(default=None, repr=False)
    ep: EPConfig | None = field(default=None, repr=False)

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: Any = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Retains the checkpoint source used to verify deployment constants."""
        check_parallel_layout(pipeline_config)
        config = DeepseekV41Config.initialize(
            pipeline_config, model_config, max_seq_len=max_seq_len
        )
        return cls(
            **{
                item.name: getattr(config, item.name) for item in fields(config)
            },
            pipeline_config=pipeline_config,
            text_config=DeepseekV41Config.get_text_config(
                pipeline_config.model.huggingface_config
            ),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, Any],
        outputs: ModelOutputConfig,
    ) -> Self:
        """Pins adapter sidecars without copying them into module weights."""
        del huggingface_config
        tables = getattr(state_dict, "engram_tables", None)
        if tables is None:
            raise ValueError(
                "DeepSeek-V4.1 requires its safetensors weight adapter "
                "and Engram table sidecar"
            )
        assert self.pipeline_config is not None
        validate_engram_tables(tables, self)
        token_map = serving_token_map(self.pipeline_config, self)
        constants = engram_static_inputs(
            tables, token_map, self.devices[0].to_device()
        )
        # Drop checkpoint table views once their pinned copies own the bytes.
        tables.clear()
        return replace(
            self,
            token_map=token_map,
            engram_static_inputs=constants,
            return_logits=outputs.return_logits,
            return_hidden_states=outputs.return_hidden_states,
        )

    def apply_model_load_context(self, context: ModelLoadContext) -> Self:
        """Preserves the DSpark target taps and verify input declarations."""
        layers = list(context.target_layer_ids)
        if layers and layers != self.dspark_target_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 collects only its configured "
                f"dspark_target_layer_ids {self.dspark_target_layer_ids}; "
                f"got {layers}"
            )
        assert self.pipeline_config is not None
        return replace(
            self,
            # The target adapter gathers verify logits/taps itself. Its
            # trunk keeps next-token outputs, including under DP where the
            # standalone all-row and hidden-state heads are unsupported.
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.NONE,
            dspark_trailing_inputs=extra_inputs(
                v41_config(self.pipeline_config.model.huggingface_config),
                self.devices,
                self.data_parallel_degree,
                verify_rows=True,
            ),
        )

    def expert_parallel_config(self, runtime: ModuleRuntime) -> EPConfig:
        """Sizes the same exchange as the legacy serving path."""
        assert self.pipeline_config is not None
        return build_ep_config(
            self.pipeline_config,
            self,
            runtime.mesh.num_devices,
            self.routed_expert_format,
        )

    def apply_runtime(self, runtime: ModuleRuntime) -> Self:
        """Applies initialized EP before constructing the decoder."""
        assert self.pipeline_config is not None
        check_parallel_layout(self.pipeline_config)
        return replace(self, ep=runtime.ep)


class DeepseekV41RequestState(RequestStateStore):
    """Owns Engram history and the request/replay metadata of each step."""

    def __init__(
        self,
        config: RegisteredDeepseekV41Config,
        *,
        max_batch_size: int,
        devices: Sequence[Device],
    ) -> None:
        assert config.pipeline_config is not None
        runtime = BatchProcessorRuntime(
            pipeline_config=config.pipeline_config,
            devices=list(devices),
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
            max_batch_size=max_batch_size,
            max_seq_len=config.max_seq_len,
            max_batch_input_tokens=(
                config.pipeline_config.runtime.max_batch_input_tokens
            ),
            enable_chunked_prefill=(
                config.pipeline_config.runtime.enable_chunked_prefill
            ),
            data_parallel_degree=config.data_parallel_degree,
        )
        self.processor = DeepseekV41BatchProcessor(config, runtime)
        self._declarations = extra_inputs(
            v41_config(config.pipeline_config.model.huggingface_config),
            config.devices,
            config.data_parallel_degree,
            verify_rows=bool(config.dspark_trailing_inputs),
        )
        self._dynamic_indices = tuple(
            i
            for i, (name, _) in enumerate(self._declarations)
            if not _is_constant(name)
        )
        self._splits: np.ndarray | None = None
        self._return_n_logits = 1
        self._window = config.sliding_window
        self._verify_width = (
            config.dspark_block_size if config.dspark_trailing_inputs else 0
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Declares dynamic metadata in the interface's named field order."""
        return tuple(self._declarations[i][1] for i in self._dynamic_indices)

    def max_dims(self) -> Mapping[str, int]:
        """Bounds history and replay from the scheduler's actual batch limits."""
        runtime = self.processor.runtime
        batch = runtime.max_global_batch_size
        tokens = runtime.max_batch_active_tokens
        return {
            "replica_0_batch_size": batch,
            "global_batch_rows": batch,
            "total_seq_len": tokens,
            "merged_seq_len": tokens + batch * self._verify_width,
            "decoder_rows": min(tokens, batch * self._window),
            "engram_history_rows": batch,
            "kept_batch_plus_one": batch + 1,
            "replay_kept_rows": tokens + batch * self._verify_width,
        }

    def set_layout(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        return_n_logits: int,
    ) -> None:
        """Records replica boundaries before the shared batcher prepares state."""
        self._return_n_logits = return_n_logits
        self._splits = (
            compute_data_parallel_splits(replica_batches)
            if self.processor.runtime.data_parallel_degree > 1
            else None
        )

    def prepare(self, batch: Sequence[TextContext]) -> tuple[Buffer, ...]:
        """Stages request metadata while reusing the pinned constants."""
        if batch:
            _, offsets = build_single_replica_ragged_token_arrays(
                cast(Sequence[RaggableContext], batch)
            )
        else:
            offsets = np.zeros(1, np.uint32)
        all_inputs = self.processor._extras(
            batch, offsets, self._return_n_logits, self._splits
        )
        return tuple(all_inputs[i] for i in self._dynamic_indices)

    def release(self, request_id: RequestID) -> None:
        """Drops a completed request's previous-batch history source."""
        # History rows are reused by batch row, not by request allocation.
        # Clearing a source cannot free the shared graph-visible allocation.
        self.processor._lookback.release(request_id)


@dataclass(kw_only=True)
class RegisteredDeepseekV41Inputs(DecoderModelInputs):
    """Shared serving inputs with the DSpark adapter's legacy extras view."""

    extra_names: tuple[str, ...] = ()

    @property
    def extras(self) -> tuple[Buffer, ...]:
        """Reconstructs the speculative adapter tail without duplicate buffers."""
        dynamic = iter(self.state)
        assert isinstance(self.constants, Mapping)
        extras: list[Buffer] = []
        for name in self.extra_names:
            if _is_constant(name):
                value = self.constants[name]
            else:
                value = next(dynamic)
            assert isinstance(value, Buffer)
            extras.append(value)
        return tuple(extras)


class RegisteredDeepseekV41BatchProcessor(DecoderBatchProcessor):
    """Adds request metadata to the framework's ragged batching and outputs."""

    def __init__(
        self,
        config: RegisteredDeepseekV41Config,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        if config.dspark_trailing_inputs:
            # A composed graph builds the target batching directly; it never
            # calls the standalone decoder's graph_specs to bind providers.
            self.state_store = DeepseekV41RequestState(
                config,
                max_batch_size=runtime.max_batch_size,
                devices=runtime.devices,
            )
            self.constant_inputs = RegisteredDeepseekV41Decoder.constant_inputs(
                config,
                devices=runtime.devices,
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DecoderModelInputs:
        """Records replica replay bounds before preparing shared inputs."""
        assert isinstance(self.state_store, DeepseekV41RequestState)
        self.state_store.set_layout(replica_batches, return_n_logits)
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        inputs.signal_buffers = tuple(self.runtime.signal_buffers)
        assert isinstance(self.config, RegisteredDeepseekV41Config)
        assert self.config.pipeline_config is not None
        return RegisteredDeepseekV41Inputs(
            **{
                item.name: getattr(inputs, item.name) for item in fields(inputs)
            },
            extra_names=tuple(
                extra_input_names(
                    v41_config(
                        self.config.pipeline_config.model.huggingface_config
                    ),
                    self.runtime.data_parallel_degree,
                )
            ),
        )


class RegisteredDeepseekV41Decoder(DeepseekV41):
    """Accepts shared semantic inputs while preserving checkpoint weight paths."""

    graph_name: ClassVar[str] = "DeepseekV41_registered_v1"
    supports_data_parallel: ClassVar[bool] = True
    composes_request_state: ClassVar[bool] = True
    input_layout: ClassVar[tuple[str, ...]] = (
        "tokens",
        "return_n_logits",
        "input_row_offsets",
        "data_parallel_splits",
        "host_input_row_offsets",
        "kv",
        "constants",
        "state",
        "ep",
    )

    def __init__(self, config: RegisteredDeepseekV41Config) -> None:
        assert config.pipeline_config is not None
        assert config.token_map is not None
        assert config.text_config is not None
        assert isinstance(config.kv_params, MultiKVCacheParams)
        axis = "dp" if config.data_parallel_degree > 1 else "tp"
        devices = tuple(ref.to_device() for ref in config.devices)
        mesh = DeviceMesh(devices, (len(devices),), (axis,))
        with default_dtype(DType.bfloat16):
            super().__init__(
                config,
                config.kv_params,
                mesh,
                text_config=config.text_config,
                token_map=config.token_map,
                data_parallel_degree=config.data_parallel_degree,
                ep_batch_manager=EPBatchManager(config.ep)
                if config.ep is not None
                else None,
            )

    @classmethod
    def constant_inputs(
        cls, config: RegisteredDeepseekV41Config, *, devices: Sequence[Device]
    ) -> ConstantInputs:
        """Declares pinned host tables and token map as named constants."""
        del cls, devices
        assert config.pipeline_config is not None
        declarations: OrderedDict[str, tree.Tree[TensorType | BufferType]] = (
            OrderedDict(
                (name, value_type)
                for name, value_type in extra_inputs(
                    v41_config(config.pipeline_config.model.huggingface_config),
                    config.devices,
                    config.data_parallel_degree,
                )
                if _is_constant(name)
            )
        )
        return ConstantInputs(
            input_types=declarations,
            buffers=OrderedDict(
                (name, config.engram_static_inputs[name])
                for name in declarations
            ),
        )

    @classmethod
    def state_store(
        cls,
        config: RegisteredDeepseekV41Config,
        *,
        max_batch_size: int,
        devices: Sequence[Device],
    ) -> DeepseekV41RequestState:
        """Builds the request history lifecycle shared by serving and capture."""
        del cls
        return DeepseekV41RequestState(
            config, max_batch_size=max_batch_size, devices=devices
        )

    def forward(
        self,
        tokens: Tensor | DecoderInputs,
        return_n_logits: Tensor | None = None,
        input_row_offsets: Tensor | None = None,
        *rest: Tensor,
    ) -> DecoderOutputs:
        """Reconstructs the original trunk call from the shared role fields."""
        if isinstance(tokens, Tensor):
            assert return_n_logits is not None and input_row_offsets is not None
            return super().forward(
                tokens, return_n_logits, input_row_offsets, *rest
            )
        inputs = tokens
        names = extra_input_names(self.interface_config, self.dp)
        dynamic = iter(inputs.state or ())
        assert inputs.constants is not None
        extras = tuple(
            inputs.constants[name] if _is_constant(name) else next(dynamic)
            for name in names
        )
        dp_inputs: tuple[Tensor, ...] = ()
        if self.dp > 1:
            assert inputs.data_parallel_splits is not None
            assert inputs.host_input_row_offsets is not None
            # Host offsets are int64 in this graph's original DP contract.
            dp_inputs = (
                inputs.data_parallel_splits,
                inputs.host_input_row_offsets.cast(DType.int64),
            )
        return super().forward(
            inputs.tokens,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            *dp_inputs,
            *tree.leaves(inputs.kv),
            *extras,
            *tree.leaves(inputs.ep),
        )
