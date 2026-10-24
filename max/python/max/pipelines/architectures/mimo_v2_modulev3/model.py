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
"""MiMo-V2.6-Flash pipeline model on ModuleV3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from max import tree
from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    MemoryPlan,
    ModelInputs,
    ModuleV3PipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
    build_single_replica_ragged_token_arrays,
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from typing_extensions import override

from ..mimo_v2.model import text_model_weights
from .mimo_v2 import MiMoV2
from .model_config import MiMoV2Config
from .quant import MXFP8_DISPATCH_QUANT_CONFIG


@dataclass
class MiMoV2Inputs(ModelInputs):
    """A class representing inputs for the MiMo-V2 ModuleV3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    ep_inputs: tuple[Buffer, ...] = ()
    """The EP atomic counters and buffer-pointer tables, after the KV
    leaves. Empty without expert parallelism."""

    data_parallel_splits: Buffer | None = None
    """Int64 ``[replicas + 1]`` cumulative request boundaries, between the
    batch inputs and the KV leaves when DP attention runs. The forward
    reads them to slice each replica's batch."""

    host_input_row_offsets: Buffer | None = None
    """Host-side row offsets, right after :attr:`data_parallel_splits`."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The graph's inputs, in the order it declares them.

        ModuleV3's ABI is ``(tokens, return_n_logits, input_row_offsets,
        *kv)``; the KV leaves follow in the same order the symbolic inputs
        declare them, then the EP tail. Under DP attention the batch-split
        tensors sit between the batch inputs and the KV leaves, where
        MiniMax-M2 declares them.
        """
        dp_splits = self.data_parallel_splits
        dp_buffers: tuple[Buffer, ...] = ()
        if dp_splits is not None:
            host_row_offsets = self.host_input_row_offsets
            assert isinstance(dp_splits, Buffer)
            assert host_row_offsets is not None
            dp_buffers = (dp_splits, host_row_offsets)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *dp_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.ep_inputs,
            *self.lora_buffers,
        )


class MiMoV2ModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, MiMoV2Inputs]
):
    """Ragged batching for MiMo-V2 ModuleV3 models (single replica, no
    signals)."""

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires the EP buffers the model's load created."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    @override
    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Declares the DP batch-split tensors between the batch and KV
        leaves, where MiniMax-M2 places them.

        Both DP tensors are host-resident: the forward reads the splits
        and the row offsets while it slices each replica's batch, and
        `prepare_initial_token_inputs` produces them at runtime.
        """
        inputs = modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params, device_refs=device_refs
        )
        if self.runtime.data_parallel_degree > 1:
            inputs[3:3] = [
                TensorType(
                    DType.int64,
                    shape=[self.runtime.data_parallel_degree + 1],
                    device=DeviceRef.CPU(),
                ),
                TensorType(
                    DType.uint32,
                    shape=["input_row_offsets_len"],
                    device=DeviceRef.CPU(),
                ),
            ]
        return inputs

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> MiMoV2Inputs:
        """Packs the ragged batch, with the DP split tensors the forward
        reads.

        Under data parallelism every replica's requests concatenate into one
        device-0 batch, beside the CPU split boundaries and host row offsets
        the graph slices it by -- the runtime half of the DP inputs
        ``get_symbolic_inputs`` declares, as the DeepSeek ModuleV3 port's
        batch processor does.
        """
        assert kv_cache_inputs is not None
        device0 = self.runtime.devices[0]
        dp_degree = self.runtime.data_parallel_degree
        if dp_degree > 1 and len(replica_batches) != dp_degree:
            raise ValueError(
                f"MiMo-V2 data parallelism expects {dp_degree} replica "
                f"batches, got {len(replica_batches)}."
            )
        # An empty replica contributes a zero-width boundary; the in-graph
        # split still hands it a 0-token shard.
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        if context_batch:
            tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
                context_batch
            )
        else:
            tokens_np = np.empty(0, dtype=np.int64)
            offsets_np = np.zeros(1, dtype=np.uint32)

        data_parallel_splits = None
        host_input_row_offsets = None
        if dp_degree > 1:
            data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
            host_input_row_offsets = Buffer.from_numpy(
                offsets_np.astype(np.uint32)
            )

        return MiMoV2Inputs(
            tokens=Buffer.from_numpy(tokens_np.astype(np.int64)).to(device0),
            input_row_offsets=Buffer.from_numpy(
                offsets_np.astype(np.uint32)
            ).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=self._ep_inputs(),
            data_parallel_splits=data_parallel_splits,
            host_input_row_offsets=host_input_row_offsets,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> MiMoV2Inputs:
        return MiMoV2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=self._ep_inputs(),
        )


def _validate_data_parallel_attention(
    dp_degree: int, ep_size: int, n_devices: int
) -> None:
    """Checks the DP-attention settings before any EP state is built.

    DP attention splits each replica's batch across the devices, so the
    experts must spread over the same devices (EP over all of them), and the
    device count must split into whole replicas.

    Raises:
        ValueError: When ``dp_degree > 1`` without EP over every device,
            or when it is not the device count (DP attention serves
            one-device replicas only; ``dp x tp`` meshes are not built).
    """
    if dp_degree <= 1:
        return
    if ep_size != n_devices:
        raise ValueError(
            f"MiMo-V2 data-parallel attention requires expert parallelism "
            f"over every device: ep_size ({ep_size}) must equal the device "
            f"count ({n_devices}) when data_parallel_degree "
            f"({dp_degree}) > 1."
        )
    if dp_degree != n_devices:
        # The module build refuses anything else (no ``dp x tp`` meshes),
        # so fail here -- before the EP config and its buffers exist.
        raise ValueError(
            f"MiMo-V2 data_parallel_degree ({dp_degree}) must equal the "
            f"device count ({n_devices}): data-parallel attention serves "
            "one-device replicas only."
        )


class MiMoV2ModuleV3Model(
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """Pipeline model for MiMo-V2.6-Flash on ModuleV3.

    The graph takes ModuleV3's inputs: tokens, the logit count and row
    offsets on the first device, then the ``{sliding_attention,
    full_attention}`` KV tree, one collection per device. With
    ``runtime.ep_size > 1`` the EP communication buffers follow the KV
    leaves as an extra input tail (atomic counters plus the buffer-pointer
    tensors), the routed experts shard whole across the devices, and the
    one-time ``ep_init`` graph is compiled alongside the forward. With
    ``model.data_parallel_degree > 1`` -- which requires EP over every
    device -- the graph also declares the DP batch-split inputs between the
    batch inputs and the KV leaves, and the forward splits each
    replica's batch with them at runtime.
    """

    model_config_cls: ClassVar[type[Any]] = MiMoV2Config
    # The lib root's own width: the fused spec model subclass binds a batch
    # processor from a different family here, which a concrete declaration
    # would make an override violation.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        MiMoV2ModuleV3BatchProcessor
    )
    # Below data parallelism the dispatch tokens are one replicated batch
    # (the allreduce backend); above it each replica's dispatches cross
    # every expert rank, which only the P2P backend collects. gen_mef
    # picks the matching backend from this at export time.
    ep_dispatch_token_layout: ClassVar[str] = "dp_replicated"

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        # Captured so _init_distributed_runtime can initialize the EP
        # communication buffers, which the forward then reads as inputs.
        self.session = session
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            memory_plan=memory_plan,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
        )

    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        del state_dict
        config = self.arch_config_as(MiMoV2Config)
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        config.data_parallel_degree = (
            self.pipeline_config.model.data_parallel_degree
        )
        return config

    def _build_ep_config(
        self, ep_size: int, n_devices: int, model_config: MiMoV2Config
    ) -> EPConfig:
        """Builds the expert-parallel config from the pipeline ``ep_size``.

        The backend follows the residual stream's placement. With
        replicated (TP) attention the residual is replicated, so the P2P
        dispatch -- which assumes each token lives on one device -- would
        dispatch and combine every replica of a token; the allreduce
        backend routes each device's own tokens to its local experts and
        lets the residual allreduce sum the partial combines. With
        data-parallel attention each token lives on exactly one replica,
        which is the P2P dispatch's precondition, and no whole-batch
        allreduce exists to complete an allreduce-backend combine.
        """
        if ep_size != n_devices:
            raise ValueError(
                f"MiMo-V2 serves expert parallelism on one node only: "
                f"ep_size ({ep_size}) must equal the device count "
                f"({n_devices})."
            )
        dp_degree = self.pipeline_config.model.data_parallel_degree
        use_allreduce = dp_degree <= 1
        if self.pipeline_config.runtime.ep_use_allreduce != use_allreduce:
            wanted = (
                "pass --ep-use-allreduce"
                if use_allreduce
                else "omit --ep-use-allreduce"
            )
            raise NotImplementedError(
                f"MiMo-V2 expert parallelism needs the {'allreduce' if use_allreduce else 'P2P'} "
                f"backend at data_parallel_degree={dp_degree}: "
                f"{wanted}."
            )
        if not self.pipeline_config.runtime.enable_chunked_prefill:
            # The dispatch/combine buffers are sized to max_tokens_per_rank
            # (max_batch_input_tokens on the allreduce backend), which the
            # batch constructor treats as a soft limit: with chunking off
            # it admits an oversized prompt whole, and the dispatch
            # overflows the fixed buffers.
            raise ValueError(
                "MiMo-V2 expert parallelism requires chunked prefill: the "
                "EP dispatch buffers are sized to max_batch_input_tokens, "
                "which without chunking is a soft limit a longer prompt "
                "exceeds."
            )
        if model_config.n_routed_experts % ep_size:
            raise ValueError(
                f"MiMo-V2: {ep_size} devices do not divide the "
                f"{model_config.n_routed_experts} routed experts."
            )
        return EPConfig(
            # The dispatch quantizes tokens to the experts' MXFP8 activation
            # format in transit, so the receiving grouped matmuls consume
            # them unchanged (see MXFP8_DISPATCH_QUANT_CONFIG).
            dispatch_dtype=DType.float8_e4m3fn,
            dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.hidden_size,
            top_k=model_config.num_experts_per_tok,
            n_experts=model_config.n_routed_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=(
                    self.pipeline_config.runtime.max_batch_input_tokens
                ),
                ep_size=ep_size,
                data_parallel_degree=dp_degree,
                use_allreduce=use_allreduce,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=1,
            use_allreduce=use_allreduce,
        )

    @override
    def _init_distributed_runtime(self, model_config: Any) -> None:
        # The base no-op defaults the EP input tail to empty, which the
        # plain (no-EP) compile path reads; call it first so an early
        # return below cannot leave the attribute unset.
        super()._init_distributed_runtime(model_config)
        self._ep_batch_manager: EPBatchManager | None = None
        dp_degree = self.pipeline_config.model.data_parallel_degree
        ep_size = self.pipeline_config.runtime.ep_size
        _validate_data_parallel_attention(dp_degree, ep_size, len(self.devices))
        if ep_size <= 1:
            return

        # Expert parallelism: ep_size > 1 shards whole routed experts across
        # the devices. The communication buffers are allocated once here
        # and threaded through the graph as the EP input tail. Under DP
        # attention the same config serves both: the EP max_tokens_per_rank
        # is already DP-aware (see _build_ep_config), and the DP batch-split
        # inputs are declared between the batch and KV leaves.
        n_devices = len(self.devices)
        ep_config = self._build_ep_config(ep_size, n_devices, model_config)
        model_config.ep_config = ep_config
        self._ep_batch_manager = EPBatchManager(ep_config)
        self._modulev3_extra_input_types = self._ep_batch_manager.input_types()
        if not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(ep_config)
            self.ep_comm_initializer.ep_init(self.session)
            ep_config.node_id = self.ep_comm_initializer.config.node_id

    def _load_state_dict(self) -> dict[str, Any]:
        """Adapts the text model's tensors, :func:`text_model_weights`.

        The adapter must see exactly those to prove every one is consumed or
        ignored.
        """
        assert self.adapter is not None
        self._checkpoint = self.adapter(
            text_model_weights(
                self.pipeline_config.model.huggingface_weight_repo,
                self.weights,
            ),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        return self._checkpoint

    def _instantiate_module(self, model_config: MiMoV2Config) -> MiMoV2:
        n_devices = len(self.devices)
        # One axis, named for what it carries: the batch axis under data
        # parallelism (the DP forward's split placement keys on "dp"),
        # the tensor axis otherwise, as the DeepSeek ModuleV3 port does.
        dp_degree = self.pipeline_config.model.data_parallel_degree
        axis_name = "dp" if dp_degree > 1 else "tp"
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), (axis_name,))
        with default_device(mesh):
            return MiMoV2(model_config, self._ep_batch_manager)
