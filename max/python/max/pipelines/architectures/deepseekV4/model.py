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

"""Implements the DeepSeek-V4-Flash PipelineModel."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    RaggableContext,
    SingleReplicaRaggedBatchProcessor,
    build_single_replica_ragged_token_arrays,
    ragged_kv_symbolic_inputs,
    single_replica_context_batch,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .deepseekV4 import DeepseekV4
from .layers import DeepseekV4Cache
from .layers.quantization import fp8_block_quant_config
from .layers.ragged import window_count
from .model_config import DeepseekV4Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV4Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4 model execution."""

    tokens: Buffer
    """Input token IDs, ragged."""

    input_row_offsets: Buffer
    """Offsets of each sequence in the ragged ``tokens``."""

    return_n_logits: Buffer
    """Number of trailing logits to return."""

    window_rows: list[Buffer] = field(default_factory=list)
    """Host, one per compression ratio; only the length is read, the window
    count (:func:`~.layers.ragged.window_count`)."""

    signal_buffers: list[Buffer] = field(default_factory=list)
    """One per device when there is more than one, else empty."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.window_rows,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
        )


class DeepseekV4BatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextContext, DeepseekV4Inputs]
):
    """Ragged single-replica batching; the stock ragged KV input order."""

    _include_signal_buffers: ClassVar[bool] = True

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        # A single device keeps the signal-free signature, and with it the
        # compiled graph, of the single-device bringup.
        inputs = ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=(
                self._include_signal_buffers and len(device_refs) > 1
            ),
        )
        window_rows = [
            TensorType(DType.uint8, [f"windows_r{r}"], device=DeviceRef.CPU())
            for r in self._window_ratios
        ]
        return [*inputs[:3], *window_rows, *inputs[3:]]

    @property
    def _window_ratios(self) -> Sequence[int]:
        assert isinstance(self.config, DeepseekV4Config)
        return self.config.window_ratios

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV4Inputs:
        context_batch = single_replica_context_batch(
            replica_batches, processor_name=type(self).__qualname__
        )
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        lengths = np.diff(offsets_np)
        device0 = self.runtime.devices[0]
        return DeepseekV4Inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            window_rows=[
                Buffer.from_numpy(
                    np.zeros(window_count(lengths, r), dtype=np.uint8)
                )
                for r in self._window_ratios
            ],
            kv_cache_inputs=kv_cache_inputs,
            signal_buffers=list(self.runtime.signal_buffers),
        )


class DeepseekV4Model(GraphPipelineModelWithKVCache[TextContext]):
    """A DeepSeek-V4-Flash pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV4Config
    batch_processor_cls: ClassVar[type[DeepseekV4BatchProcessor]] = (
        DeepseekV4BatchProcessor
    )

    model: Model
    """The compiled and initialized MAX Engine model."""

    _strict_state_dict_loading = True

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV4Config:
        model_config = self.arch_config_as(DeepseekV4Config)
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states
        # ``norm.weight`` is the one norm every V4 checkpoint has, hash-routed
        # or not, so it is the safe probe for the norm storage dtype.
        model_config.norm_dtype = state_dict["norm.weight"].dtype
        if model_config.quantization_encoding == "float8_e4m3fn":
            # The encoding names the fp8 projections only. The activation
            # dtype is the reference's default dtype, bf16, which is also
            # what the checkpoint stores ``embed`` / ``ffn.gate`` /
            # ``indexer.weights_proj`` / ``wo_a`` in and the reference declares
            # them as; those follow ``config.dtype``. The weights the reference
            # declares float32 (norms, head, compressor ``wkv`` / ``wgate``)
            # are declared so by the modules and upcast by the adapter.
            quant_config = fp8_block_quant_config(
                getattr(self.huggingface_config, "quantization_config", None),
                model_config.num_hidden_layers,
            )
            if quant_config is None:
                raise ValueError(
                    "quantization_encoding float8_e4m3fn needs a checkpoint "
                    "that declares its fp8 quantization_config"
                )
            model_config.quant_config = quant_config
            model_config.dtype = state_dict["embed.weight"].dtype
        # Not a speculative graph: the adapter drops the ``mtp.*`` weights.
        model_config.dspark_stages = False
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: DeepseekV4Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        nn_model = DeepseekV4(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        weights_registry = nn_model.state_dict(auto_initialize=False)

        assert self.batch_processor is not None
        input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params, device_refs=self.device_refs
        )
        with Graph("deepseekV4", input_types=input_types) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )
            n_ratios = len(model_config.window_ratios)
            windows = {
                r: v.tensor.shape[0]
                for r, v in zip(
                    model_config.window_ratios,
                    variadic_args[:n_ratios],
                    strict=True,
                )
            }
            variadic_args = variadic_args[n_ratios:]
            assert isinstance(self.kv_params, MultiKVCacheParams)
            n_dev = len(self.device_refs)
            if n_dev == 1:
                cache = DeepseekV4Cache.from_groups(
                    model_config,
                    self.kv_params.unflatten_basic_kv_tree(iter(variadic_args)),
                )
                outputs = nn_model.serve(
                    tokens.tensor,
                    input_row_offsets.tensor,
                    return_n_logits.tensor,
                    windows,
                    cache,
                )
            else:
                signal_buffers = [v.buffer for v in variadic_args[:n_dev]]
                groups = self.kv_params.unflatten_basic_kv_tree(
                    iter(variadic_args[n_dev:])
                )
                caches = [
                    DeepseekV4Cache.from_groups(
                        model_config, groups, device_idx=i
                    )
                    for i in range(n_dev)
                ]
                tokens_per_dev = ops.distributed_broadcast(
                    tokens.tensor, signal_buffers
                )
                offsets_per_dev = ops.distributed_broadcast(
                    input_row_offsets.tensor, signal_buffers
                )
                outputs = DeepseekV4.serve_tensor_parallel(
                    nn_model.tensor_parallel_replicas(self.device_refs),
                    tokens_per_dev,
                    offsets_per_dev,
                    return_n_logits.tensor,
                    windows,
                    caches,
                    signal_buffers,
                )
            graph.output(*outputs)
        return graph, weights_registry

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(DeepseekV4Inputs, model_inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)
