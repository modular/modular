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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import NUM_GROUPS, estimate_ep_memory_usage
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.config.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from max.pipelines.lib.float8 import parse_float8_config
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.support.algorithm import flatten2d
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Inputs, DeepseekV2Model
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


def _validate_ep_kernel_limits(
    ep_config: EPConfig, *, max_local_experts: int = 32
) -> None:
    n_ranks = ep_config.n_gpus_per_node * ep_config.n_nodes
    n_local_experts = ep_config.n_experts // n_ranks
    if n_local_experts > max_local_experts:
        raise ValueError(
            "Expert-parallel local experts per device "
            f"({n_local_experts}) exceeds kernel limit "
            f"({max_local_experts}). "
            "Use more expert-parallel ranks or disable EP."
        )


@dataclass
class DeepseekV3Inputs(DeepseekV2Inputs):
    """A class representing inputs for the DeepseekV3 model."""

    host_input_row_offsets: Buffer
    """Tensor containing the host input row offsets."""

    batch_context_lengths: list[Buffer]
    """List of tensors containing the context length of each batch."""

    data_parallel_splits: Buffer = field(kw_only=True)
    """Tensor containing the data parallel splits for the MLA layer."""


class DeepseekV3Model(AlwaysSignalBuffersMixin, DeepseekV2Model):
    """A DeepseekV3 model."""

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        encoding = pipeline_config.model.quantization_encoding
        if encoding is not None and is_float4_encoding(encoding):
            cache_dtype = DType.bfloat16
        return DeepseekV3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3Config:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        max_batch_total_tokens = self.pipeline_config.max_batch_total_tokens
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif self.pipeline_config.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = self.dtype
        if dtype in (DType.float8_e4m3fn, DType.uint8, DType.float4_e2m1fn):
            float8_config = parse_float8_config(config, state_dict, dtype)
        else:
            float8_config = None

        # Check if EP should be configured
        if self.pipeline_config.ep_size == 1:
            ep_config = None
        else:
            if self.pipeline_config.ep_size % len(self.devices) != 0:
                raise ValueError(
                    "If you are running with expert parallelism, ep_size must"
                    " be set to the total number of GPUs across nodes."
                )
            n_nodes = self.pipeline_config.ep_size // len(self.devices)
            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                max_tokens_per_rank=self.pipeline_config.max_batch_input_tokens,
                n_gpus_per_node=len(self.devices),
                n_nodes=n_nodes,
                dispatch_fp8_config=None,
            )

            if config.n_shared_experts == 1:
                # Only enable shared expert fusion if the shared expert is of
                # the same shape as routed experts.
                ep_kwargs["fused_shared_expert"] = True

            if float8_config is not None:
                ep_kwargs["dispatch_fp8_config"] = float8_config

            ep_config = EPConfig(**ep_kwargs)
            _validate_ep_kernel_limits(ep_config)

        # Determine data_parallel_degree: EP requires data-parallel attention
        if ep_config is not None:
            # When EP is used, data parallelism is required for attention
            data_parallel_degree = len(self.devices)
        else:
            # Use the configured value from pipeline_config
            data_parallel_degree = (
                self.pipeline_config.model.data_parallel_degree
            )

        norm_dtype = state_dict[
            "layers.0.self_attn.kv_a_layernorm.weight"
        ].dtype

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
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.float8_config = float8_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states

        return model_config

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of our model."""

        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        n_gpus_per_node = len(model_config.device_specs)

        encoding = pipeline_config.model.quantization_encoding
        assert encoding is not None

        def _n_elems_to_bytes(n_elems: int) -> int:
            dtype = supported_encoding_dtype(encoding).size_in_bytes
            if is_float4_encoding(encoding):
                # Account for the scales. For NVFP4 format, every 16 FP4 elements
                # share one FP8 scale factor. The size of the scales is one
                # eighth of the size of the FP4 quants (8 bits / (16 * 4 bits)).
                return int(n_elems // 2 * dtype * 1.125)
            else:
                return n_elems * dtype

        config = model_config.huggingface_config
        assert config is not None
        n_sparse_layers = (
            config.num_hidden_layers - config.first_k_dense_replace
        )
        n_mtp_layers = config.num_nextn_predict_layers

        # Note: All the following calculations are not exact, but they are
        # better than directly using the raw weights size.

        # First, Calculate the lm_head/embed_tokens size.
        # There are always in Bf16.
        lm_head_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        embed_tokens_size = lm_head_size

        # Subtract the lm_head/embed_tokens size from the weights size
        weights_size -= lm_head_size + embed_tokens_size
        weights_size -= (lm_head_size + embed_tokens_size) * n_mtp_layers

        # We don't use the MTP module for now, so subtract the MTP attn/moe size.
        # Estimate the MTP module size by assuming the MTP layer is of the same
        # size as a sparse model layer.
        weights_size = int(
            weights_size * n_sparse_layers / (n_sparse_layers + n_mtp_layers)
        )

        # Calculate the routing experts and the shared experts size.
        expert_elems = (
            config.moe_intermediate_size * config.hidden_size * 3
        )  # A factor of 3 accounts for the gate/up/down proj weights.
        expert_size = _n_elems_to_bytes(expert_elems)
        routing_experts_size = (
            n_sparse_layers * config.n_routed_experts * expert_size
        )
        shared_experts_size = (
            n_sparse_layers * config.n_shared_experts * expert_size
        )

        # Estimate the size of the attention weights.
        attn_weights_size = (
            weights_size - routing_experts_size - shared_experts_size
        )

        # If we use DP attention, attention weights are duplicated on each DP rank.
        total_size = attn_weights_size * model_config.data_parallel_degree

        # The shared experts are duplicated on each device.
        total_size += shared_experts_size * n_gpus_per_node

        ep_size = max(pipeline_config.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # we don't support mixing EP and TP strategies yet.
            # ep_size must be equal to n_gpus_per_node * n_nodes
            assert ep_size % n_gpus_per_node == 0
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Add back the lm_head/embed_tokens size, they will never be duplicated.
        total_size += lm_head_size + embed_tokens_size

        return total_size

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Estimates the activation memory required for model execution.

        This accounts for temporary memory buffers used during model execution,
        such as intermediate activations and working buffers.

        Args:
            pipeline_config: Pipeline configuration
            huggingface_config: HuggingFace model configuration

        Returns:
            Estimated activation memory in bytes
        """

        encoding = pipeline_config.model.quantization_encoding
        assert encoding is not None
        mla_activation_memory: int = 0
        moe_activation_memory: int = 0

        # During the prefill, we need to up-project all the KV cache for
        # current requests. The total context length of requests in a batch
        # should be limited by max_batch_total_tokens.
        if pipeline_config.pipeline_role != "decode_only":
            max_kv_length: int = 0

            if pipeline_config.max_batch_total_tokens is None:
                # If max_batch_total_tokens is not set, we use max_length.
                max_kv_length = pipeline_config.model.max_length or 0
            else:
                max_kv_length = pipeline_config.max_batch_total_tokens

            mla_activation_memory += (
                pipeline_config.model.data_parallel_degree
                * 2  # 2 for K and V
                * max_kv_length
                * huggingface_config.num_attention_heads
                * huggingface_config.qk_nope_head_dim
                * pipeline_config.model.kv_cache.cache_dtype.size_in_bytes
            )

        # Estimate activation memory during Expert Parallel MoE.
        if pipeline_config.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)
            max_input_len_per_rank = pipeline_config.max_batch_input_tokens

            # Calculate the maximum number of tokens a rank may receive during all-to-all routing.
            max_recv_tokens_per_rank = (
                max_input_len_per_rank * huggingface_config.n_routed_experts
            )

            # The maximal activation memory usage happens at the second
            # grouped_matmul in the MoE layer. The input for that matmul would
            # of shape [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.moe_intermediate_size
                * supported_encoding_dtype(encoding).size_in_bytes
            )

            # The output would be of shape [max_recv_tokens_per_rank, hidden_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.hidden_size
                * DType.bfloat16.size_in_bytes  # output is always bfloat16.
            )

            # Adding 256MB per GPU to account for misc items (e.g. FP8 scalars).
            moe_activation_memory += 256 * 1024 * 1024

            moe_activation_memory *= n_gpus_per_node

        # We only need to consider the maximum of the MLA and MoE activation
        # memories, because the MLA and MoE layers are executed sequentially.
        activation_memory = max(mla_activation_memory, moe_activation_memory)

        # EP SHMEM communication buffers are persistent (allocated once at
        # model init, not freed between layers), so add on top of the
        # per-layer activation peak.
        ep_buffer_memory = 0
        if pipeline_config.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)

            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=huggingface_config.hidden_size,
                dispatch_dtype=supported_encoding_dtype(encoding),
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=pipeline_config.max_batch_input_tokens,
                n_experts=huggingface_config.n_routed_experts,
                top_k=huggingface_config.num_experts_per_tok,
            )
            ep_buffer_memory = (
                per_device_ep_memory * NUM_GROUPS * n_gpus_per_node
            )

            logger.info(
                "Estimated EP SHMEM buffer memory: "
                f"{to_human_readable_bytes(ep_buffer_memory)}"
            )

        activation_memory += ep_buffer_memory

        if activation_memory != 0:
            logger.info(
                f"Estimated activation memory: {to_human_readable_bytes(activation_memory)}"
            )

        return activation_memory

    @override
    def load_model(self, session: InferenceSession) -> Model:
        """Load the model with the given weights."""

        max_batch_size = self.pipeline_config.max_batch_size
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

        timer = CompilationTimer("model")
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

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        self.ep_comm_initializer: EPCommInitializer | None = None
        # Skip EP initialization in virtual device mode (compilation-only)
        # since NVSHMEM functions cannot be linked without real GPU devices.
        # We still keep ep_config to generate the correct graph structure.
        if config.ep_config is not None and not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP initialization is successful."
                )

        nn_model = DeepseekV3(config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
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
            fetch_types = self.kv_params.get_symbolic_inputs()[0]
            len_of_kv_inputs = len(list(fetch_types)) * len(self.devices)
            kv_caches_per_dev = self._unflatten_kv_inputs(
                [next(variadic_args_iter) for _ in range(len_of_kv_inputs)]
            )

            # Unmarshal the batch context lengths
            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            # all remaining arguments are for EP inputs
            ep_model_inputs = list(variadic_args_iter)

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                devices_input_row_offsets.tensor,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)

        timer.mark_build_complete()
        model = session.load(graph, weights_registry=self.state_dict)
        timer.done()

        return model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else self.ep_comm_initializer.model_inputs()
        )

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.host_input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.data_parallel_splits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            *model_inputs.batch_context_lengths,
            *ep_inputs,
        )

        num_hidden_state_outputs = len(self.devices)
        num_outputs = len(model_outputs)

        # Possible output configurations:
        # - 1 output: next_token_logits only
        # - 3 outputs: next_token_logits, logits, logit_offsets (variable logits)
        # - 1 + N: next_token_logits + hidden_states (one per device)
        # - 3 + N: next_token_logits, logits, logit_offsets + hidden_states (one per device)

        if num_outputs == 3 + num_hidden_state_outputs:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            hidden_states_list: list[Buffer] = []
            for i in range(num_hidden_state_outputs):
                hs = model_outputs[3 + i]
                assert isinstance(hs, Buffer)
                hidden_states_list.append(hs)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
                hidden_states=hidden_states_list,
            )
        elif num_outputs == 1 + num_hidden_state_outputs:
            assert isinstance(model_outputs[0], Buffer)
            hidden_states_list = []
            for i in range(num_hidden_state_outputs):
                hs = model_outputs[1 + i]
                assert isinstance(hs, Buffer)
                hidden_states_list.append(hs)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
                hidden_states=hidden_states_list,
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
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
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
        if self.pipeline_config.pipeline_role != "decode_only":

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
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> DeepseekV3Inputs:
        assert isinstance(prev_model_inputs, DeepseekV3Inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._device_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        next_host_input_row_offsets = self._host_input_row_offsets_prealloc[
            :row_offsets_size
        ]
        return DeepseekV3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            host_input_row_offsets=next_host_input_row_offsets,
            batch_context_lengths=self._batch_context_lengths_prealloc_cpu,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            data_parallel_splits=prev_model_inputs.data_parallel_splits,
        )
