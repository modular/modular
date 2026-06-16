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
"""Implements the DeepseekV3 nn.model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, ops
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.kv_cache import KVCacheInputs, KVCacheInputsInterface
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    ModelInputs,
    ModelOutputs,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.pipelines.weights.quant import parse_quant_config
from max.support.algorithm import flatten2d
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Inputs, DeepseekV2Model
from .deepseekV3 import DeepseekV3
from .memory_planner import (
    _ep_max_rank_send_tokens_for_pipeline,
    _get_mtp_draft_ep_dispatch_dtype,
)
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV3Inputs(DeepseekV2Inputs):
    """A class representing inputs for the DeepseekV3 model."""

    host_input_row_offsets: Buffer
    """Tensor containing the host input row offsets."""

    batch_context_lengths: list[Buffer]
    """List of tensors containing the context length of each batch."""

    data_parallel_splits: Buffer = field(kw_only=True)
    """Tensor containing the data parallel splits for the MLA layer."""

    ep_inputs: tuple[Buffer, ...] = field(kw_only=True, default=())
    """Expert parallel communication buffers (atomic counters and device pointers)."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class DeepseekV3Model(AlwaysSignalBuffersMixin, DeepseekV2Model):
    """A DeepseekV3 model."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        # data_parallel_degree controls the attention strategy:
        #   == num_devices  ->  DP attention  (each device owns a batch shard)
        #   == 1            ->  TP attention  (heads sharded, tokens replicated)
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree
        max_batch_total_tokens = (
            self.pipeline_config.runtime.max_batch_total_tokens
        )
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif self.pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = self.dtype
        if dtype in (DType.float8_e4m3fn, DType.uint8, DType.float4_e2m1fn):
            quant_config = parse_quant_config(config, state_dict, dtype)
        else:
            quant_config = None

        # Check if EP should be configured
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % len(self.devices) != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({len(self.devices)}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={len(self.devices)}."
                )
            # TODO: Support TP attention for FP8 Deepseek-V3 models.
            if quant_config is not None and not quant_config.is_nvfp4:
                if ep_size > data_parallel_degree:
                    raise ValueError(
                        "TP attention is not supported for FP8 Deepseek-V3 models."
                    )

            n_nodes = ep_size // len(self.devices)

            ep_max_rank_send_tokens = _ep_max_rank_send_tokens_for_pipeline(
                self.pipeline_config
            )

            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_gpus_per_node=len(self.devices),
                n_nodes=n_nodes,
                dispatch_quant_config=None,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            )

            if config.n_shared_experts == 1:
                # Only enable shared expert fusion if the shared expert is of
                # the same shape as routed experts.
                ep_kwargs["fused_shared_expert"] = True

            if quant_config is not None:
                ep_kwargs["dispatch_quant_config"] = quant_config

            ep_config = EPConfig(**ep_kwargs)

        norm_dtype = state_dict[
            "layers.0.self_attn.kv_a_layernorm.weight"
        ].dtype

        # Extract gate dtype from actual weights (may differ from norm_dtype).
        gate_dtype_key = None
        for k in state_dict:
            if k.endswith("gate.gate_score.weight"):
                gate_dtype_key = k
                break
        gate_dtype = (
            state_dict[gate_dtype_key].dtype
            if gate_dtype_key is not None
            else None
        )

        if config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            correction_bias_dtype = state_dict[correction_bias_key].dtype
        else:
            correction_bias_dtype = None

        # Initialize config with parameters from pipeline_config
        model_config = DeepseekV3Config.initialize(self.pipeline_config)

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.gate_dtype = gate_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states

        num_devices = len(self.devices)
        if num_devices > 1:
            if ep_size > 1:
                attn_strategy = "TP" if data_parallel_degree == 1 else "DP"
                moe_strategy = "EP"
            else:
                attn_strategy = "TP"
                moe_strategy = "TP"
            logger.info(
                f"DeepSeekV3: data_parallel_degree={data_parallel_degree},"
                f" ep_size={ep_size}. Use {attn_strategy}-attention +"
                f" {moe_strategy}-MoE strategy."
            )

        return model_config

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the model with the given weights."""

        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        # `_host_input_row_offsets_prealloc` tensor needs to reserve space for
        # `max_batch_size` of requests on each DP rank.
        dp_size = self.pipeline_config.model.data_parallel_degree
        max_batch_size *= dp_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(self.devices[0])
        )

        # create batch context lengths tensor for each device
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        with CompilationTimer("model") as timer:
            if self.adapter:
                state_dict = self.adapter(
                    dict(self.weights.items()),
                    huggingface_config=self.huggingface_config,
                    pipeline_config=self.pipeline_config,
                )
            else:
                state_dict = {
                    key: value.data() for key, value in self.weights.items()
                }
            # Create the model
            config = self._create_model_config(state_dict)

            self.ep_comm_initializer: EPCommInitializer | None = None
            # Skip EP initialization in virtual device mode (compilation-only)
            # since NVSHMEM functions cannot be linked without real GPU devices.
            # We still keep ep_config to generate the correct graph structure.
            if config.ep_config is not None and not is_virtual_device_mode():
                ep_alloc_config = config.ep_config
                # When EAGLE/MTP speculative decoding shares EP buffers between
                # target (FP4) and draft (BF16) models, allocate buffers
                # large enough for the draft model's dispatch dtype.
                draft_ep_dtype = _get_mtp_draft_ep_dispatch_dtype(
                    self.pipeline_config
                )
                if draft_ep_dtype is not None:
                    ep_alloc_config = replace(
                        config.ep_config,
                        dispatch_dtype=draft_ep_dtype,
                        dispatch_quant_config=None,
                    )
                    logger.info(
                        f"Upsizing EP buffers for draft model dispatch dtype: {draft_ep_dtype}"
                    )
                self.ep_comm_initializer = EPCommInitializer(ep_alloc_config)
                self.ep_comm_initializer.ep_init(session)
                # ep_init() sets node_id on the initializer's config; propagate
                # it back to the model's ep_config (which may be a different
                # object when we created a copy above).
                config.ep_config.node_id = ep_alloc_config.node_id
                if config.ep_config.node_id == -1:
                    raise ValueError(
                        "EP node ID is not set. Please check if the EP initialization is successful."
                    )

            nn_model = DeepseekV3(config)
            nn_model.load_state_dict(
                state_dict, weight_alignment=1, strict=True
            )
            self.state_dict = nn_model.state_dict()

            # Create the graph
            with Graph(
                "deepseekV3_graph",
                input_types=nn_model.input_types(self.kv_params),
            ) as graph:
                (
                    tokens,
                    devices_input_row_offsets,
                    host_input_row_offsets,
                    return_n_logits,
                    data_parallel_splits,
                    *variadic_args,
                ) = graph.inputs

                variadic_args_iter = iter(variadic_args)
                # Multi-GPU passes a signal buffer per device: unmarshal these.
                signal_buffers = [
                    next(variadic_args_iter).buffer
                    for _ in range(len(self.devices))
                ]

                # Unmarshal the KV cache arguments.
                kv_inputs = self.kv_params.unflatten_kv_inputs(
                    variadic_args_iter
                )
                assert isinstance(kv_inputs, KVCacheInputs)
                kv_caches_per_dev = list(kv_inputs.inputs)

                # Unmarshal the batch context lengths
                batch_context_lengths = [
                    next(variadic_args_iter).tensor
                    for _ in range(len(self.devices))
                ]

                # all remaining arguments are for EP inputs
                ep_model_inputs = list(variadic_args_iter)

                # DeepseekV3.__call__ expects a per-device list for
                # input_row_offsets
                input_row_offsets_per_dev = list(
                    ops.distributed_broadcast(
                        devices_input_row_offsets.tensor, signal_buffers
                    )
                )
                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets_per_dev,
                    host_input_row_offsets.tensor,
                    data_parallel_splits.tensor,
                    batch_context_lengths,
                    ep_model_inputs,
                )

                graph.output(*outputs)

            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV3Inputs)

        model_outputs = self.model.execute(*model_inputs.buffers)
        num_outputs = len(model_outputs)

        # Possible output configurations:
        # - 4 outputs: next_token_logits, logits, logit_offsets + hidden_states
        # - 3 outputs: next_token_logits, logits, logit_offsets (variable logits)
        # - 2 outputs: next_token_logits + hidden_states
        # - 1 output: next_token_logits only

        if num_outputs == 4:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            assert isinstance(model_outputs[3], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif num_outputs == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        elif num_outputs == 2:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
                hidden_states=model_outputs[1],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        dp = self.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        # Allocate the model inputs on pinned memory for faster h2d
        # transfer speeds. If model is on host, then fall back to normal
        # pageable memory. We initialize these empty max tensors by exporting
        # to numpy over dlpack and using numpy methods.
        # TODO: move rest of inputs to pinned memory
        device0 = self.devices[0]
        pinned = not device0.is_host

        # If we are not in decode only mode, we need to create a list of
        # tensors containing the context length of each batch. Need by MLA
        # prefill.
        if self.pipeline_config.runtime.pipeline_role != "decode_only":

            def align_length(length: int) -> int:
                page_size = self.kv_cache_config.kv_cache_page_size
                return (length + page_size - 1) // page_size * page_size

            for i, batch in enumerate(replica_batches):
                curr_length = sum(
                    [align_length(ctx.tokens.current_position) for ctx in batch]
                )
                self._batch_context_lengths_prealloc_cpu[i][0] = curr_length

            if dp != len(self.devices):
                assert dp == 1
                # Duplicate the batch context lengths for each device.
                for dev_idx in range(1, len(self.devices)):
                    self._batch_context_lengths_prealloc_cpu[dev_idx][0] = (
                        self._batch_context_lengths_prealloc_cpu[0][0].item()
                    )

        context_batch = flatten2d(replica_batches)
        # Create tokens
        tokens: Buffer
        pinned_input_row_offsets: Buffer
        if len(context_batch) == 0:
            if pinned:
                tokens = DevicePinnedBuffer(
                    shape=[0], dtype=DType.int64, device=device0
                )
            else:
                tokens = Buffer(shape=[0], dtype=DType.int64, device=device0)
            host_input_row_offsets = Buffer.zeros(shape=[1], dtype=DType.uint32)

            if pinned:
                pinned_input_row_offsets = DevicePinnedBuffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            else:
                pinned_input_row_offsets = Buffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            device_input_row_offsets = pinned_input_row_offsets.to(device0)
        else:
            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            num_tokens = sum(ctx.tokens.active_length for ctx in context_batch)
            tokens_host: Buffer
            if pinned:
                tokens_host = DevicePinnedBuffer(
                    shape=(num_tokens,),
                    dtype=DType.int64,
                    device=device0,
                )
            else:
                tokens_host = Buffer(
                    shape=(num_tokens,),
                    dtype=DType.int64,
                    device=device0,
                )
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_host.to_numpy(),
            )
            tokens = tokens_host.to(device0)

            # Create a ragged token vector of length: sum(len(t) for t in tokens).
            # Get input_row_offsets: start and end position of each batch in the
            # combined total_seq_len dimension.
            input_row_offsets = np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )

            # FIXME GEX-3121: There is a bug when using pinned buffer as graph cpu input:
            # `Expected Device(type=cpu,id=0), but was on device Device(type=gpu,id=0)`
            # Thus we set up both a non-pinned and a pinned cpu buffer as workaround.
            host_input_row_offsets = Buffer(
                shape=(len(context_batch) + 1,),
                dtype=DType.uint32,
            )
            host_input_row_offsets.to_numpy()[:] = input_row_offsets[:]

            if pinned:
                pinned_input_row_offsets = DevicePinnedBuffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            else:
                pinned_input_row_offsets = Buffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            pinned_input_row_offsets.to_numpy()[:] = input_row_offsets[:]
            device_input_row_offsets = pinned_input_row_offsets.to(device0)

        data_parallel_splits = Buffer.from_numpy(
            compute_data_parallel_splits(replica_batches)
        )

        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else tuple(self.ep_comm_initializer.model_inputs())
        )

        return DeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=device_input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=data_parallel_splits,
            ep_inputs=ep_inputs,
        )
