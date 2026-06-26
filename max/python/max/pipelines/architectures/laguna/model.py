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
from typing import Any, ClassVar, Literal

import numpy as np
from max._core.engine import Model
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.architectures.llama3.model import LlamaModelBase
from max.pipelines.lib import (
    CompilationTimer,
    PipelineConfig,
    supported_encoding_dtype,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.weights.quant import parse_quant_config
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig
from typing_extensions import override

from .laguna import Laguna
from .model_config import LagunaConfig

logger = logging.getLogger("max.pipelines")


class LagunaModel(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Laguna pipeline model for text generation.

    Uses ``AlwaysSignalBuffersMixin`` since ``VocabParallelEmbedding`` and
    ``ColumnParallelLinear`` always require signal buffers for allreduce.
    """

    model_config_cls: ClassVar[type[Any]] = LagunaConfig

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    # Empirically determined headroom reserved per device when CUDA graph
    # capture is enabled, on top of the activation memory we account for
    # explicitly. Without this, capture can OOM on large MoE models.
    _GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE = 8 * 1024**3

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        encoding = pipeline_config.model.quantization_encoding
        n_gpus_per_node = len(pipeline_config.model.device_specs)
        num_experts = getattr(huggingface_config, "num_local_experts", 256)
        moe_dim = getattr(huggingface_config, "intermediate_size", 1536)
        hidden_size = getattr(huggingface_config, "hidden_size", 3072)
        top_k = getattr(huggingface_config, "num_experts_per_tok", 8)

        ep_buffer_memory = 0
        moe_activation_memory = 0
        ep_size = pipeline_config.runtime.ep_size
        if ep_size > 1 and encoding is not None:
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
            )
            ep_dispatch_dtype = supported_encoding_dtype(encoding)

            # Worst-case tokens received per rank during all-to-all routing.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                num_experts,
                ep_size * top_k,
            )

            # Peak MoE activation: input to second grouped_matmul has shape
            # [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * moe_dim
                * ep_dispatch_dtype.size_in_bytes
            )
            # Output has shape [max_recv_tokens_per_rank, hidden_size] in
            # bfloat16.
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * hidden_size
                * DType.bfloat16.size_in_bytes
            )
            # 256MB per GPU for misc scalar buffers.
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            n_nodes = max(ep_size // n_gpus_per_node, 1)
            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=hidden_size,
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=num_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=top_k,
            )
            # EPCommInitializer double-buffers (NUM_GROUPS=2) the SHMEM
            # dispatch/combine buffers.
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory

        graph_capture_headroom = 0
        if pipeline_config.runtime.device_graph_capture:
            graph_capture_headroom = (
                cls._GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE * n_gpus_per_node
            )
            activation_memory += graph_capture_headroom

        if activation_memory != 0:
            logger.info(
                "Estimated activation memory: %s "
                "(ep_buffers=%s, moe_activation=%s, graph_capture=%s)",
                to_human_readable_bytes(activation_memory),
                to_human_readable_bytes(ep_buffer_memory),
                to_human_readable_bytes(moe_activation_memory),
                to_human_readable_bytes(graph_capture_headroom),
            )

        return activation_memory

    @override
    def load_model(self, session: InferenceSession) -> Model:
        assert self.max_batch_size, "Expected max_batch_size to be set"
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter, session)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)
        return model

    def _resolve_nvfp4_quant_config(
        self, model_config: LagunaConfig, state_dict: dict[str, Any]
    ) -> None:
        """Builds the NVFP4 ``quant_config`` for the compressed-tensors checkpoint.

        The compute dtype is bf16 (embedding, attention, norms, lm_head), so the
        base ``finalize`` calls ``parse_quant_config`` with bf16 and gets
        ``None``. Laguna ships compressed-tensors NVFP4, which is numerically
        identical to modelopt NVFP4 (group_size 16, e4m3 per-group scales, fp32
        global scale) but is not recognised by MAX's FP4 parser, which only
        handles the modelopt flavor. Present a modelopt-shaped
        ``quantization_config`` to the parser so the dense/MoE Linears pack via
        ``quant_config.is_fp4``; the non-quant layers stay bf16 from
        ``config.dtype``.
        """
        hf_qc = getattr(self.huggingface_config, "quantization_config", None)
        if model_config.quant_config is not None or not hf_qc:
            return
        # The MAX parser reads only from ``huggingface_config``, so present the
        # modelopt-shaped config to it and restore the original afterwards.
        # compressed-tensors defines quant scope by ``targets`` (only MLP/MoE
        # gate/up/down_proj are FP4); the modelopt parser uses inverse
        # ``ignore`` semantics (quantize all Linears except ignore), so ignore
        # everything that is NOT a target: attention, the router gate, lm_head.
        orig_qc = self.huggingface_config.quantization_config
        try:
            self.huggingface_config.quantization_config = {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "ignore": [
                    "re:.*self_attn\\..*",
                    "re:.*\\.mlp\\.gate$",
                    "lm_head",
                ],
            }
            model_config.quant_config = parse_quant_config(
                self.huggingface_config, state_dict, DType.uint8
            )
        finally:
            self.huggingface_config.quantization_config = orig_qc

    @staticmethod
    def _detect_state_dict_dtypes(
        model_config: LagunaConfig, state_dict: dict[str, Any]
    ) -> None:
        """Reads the gate, correction-bias, and attention dtypes off the weights.

        These are kept higher-precision than the FP4 experts and vary by
        checkpoint, so detect them from the loaded tensors rather than assuming
        the compute dtype. Each dtype is uniform across layers.
        """
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
            elif k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
            elif k.endswith("self_attn.q_proj.weight"):
                model_config.attn_dtype = v.dtype

    def _configure_expert_parallelism(
        self, model_config: LagunaConfig, session: InferenceSession | None
    ) -> None:
        """Sets up expert parallelism, or disables it on a single GPU.

        EP is only meaningful with 2+ devices to dispatch experts across; the EP
        communication kernels (``max/kernels/src/shmem/ep_comm.mojo``) reject
        ``n_ranks=1``. On a single GPU the MoE block runs all experts locally.
        """
        num_devices = len(self.devices)
        if num_devices <= 1:
            self.ep_comm_initializer = None
            model_config.ep_config = None
            logger.info(
                "EP disabled (single-GPU); MoE runs all "
                f"{model_config.num_local_experts} experts locally"
            )
            return

        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=num_devices,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
        )
        is_mxfp4 = (
            model_config.quant_config is not None
            and model_config.quant_config.is_mxfp4
        )
        model_config.ep_config = EPConfig(
            dispatch_dtype=DType.uint8 if is_mxfp4 else model_config.dtype,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.hidden_size,
            top_k=model_config.num_experts_per_tok,
            n_experts=model_config.num_local_experts,
            max_tokens_per_rank=ep_max_rank_send_tokens,
            n_gpus_per_node=num_devices,
            n_nodes=1,
            dispatch_quant_config=model_config.quant_config,
        )
        assert session is not None
        self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
        self.ep_comm_initializer.ep_init(session)
        logger.info(
            f"EP initialized: node_id={model_config.ep_config.node_id}, "
            f"n_gpus={model_config.ep_config.n_gpus_per_node}, "
            f"n_nodes={model_config.ep_config.n_nodes}, "
            f"n_experts={model_config.ep_config.n_experts}, "
            f"max_tokens_per_rank={model_config.ep_config.max_tokens_per_rank}"
        )

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        session: InferenceSession | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )
        model_config = LagunaConfig.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        self._resolve_nvfp4_quant_config(model_config, state_dict)
        self._detect_state_dict_dtypes(model_config, state_dict)
        self._configure_expert_parallelism(model_config, session)

        nn_model = Laguna(model_config)

        graph_inputs = nn_model.input_types(self.kv_params)

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )

        self.state_dict = nn_model.state_dict()

        num_devices = len(self.devices)
        with Graph("laguna", input_types=graph_inputs) as graph:
            inputs_iter = iter(graph.inputs)
            tokens = next(inputs_iter)
            input_row_offsets = next(inputs_iter)
            return_n_logits = next(inputs_iter)
            # DP-N path: data_parallel_splits + host_input_row_offsets
            # appear in the graph input list (see ``Laguna.input_types``).
            # DP=1 path: skip — the model.forward guards
            # ``split_batch_replicated`` on ``data_parallel_degree > 1``
            # so these args can be ``None`` end-to-end.
            if model_config.data_parallel_degree > 1:
                data_parallel_splits = next(inputs_iter).tensor
                host_input_row_offsets = next(inputs_iter).tensor
            else:
                data_parallel_splits = None
                host_input_row_offsets = None

            signal_buffers = [
                next(inputs_iter).buffer for _ in range(num_devices)
            ]

            num_kv_inputs = len(
                nn_model.kv_params.get_symbolic_inputs().flatten()
            )
            kv_cache_inputs = [next(inputs_iter) for _ in range(num_kv_inputs)]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            # Remaining args are EP inputs (empty list if no EP).
            ep_inputs = list(inputs_iter)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,  # type: ignore[arg-type]
                data_parallel_splits,
                host_input_row_offsets,
            )

            graph.output(*outputs)
            return graph
