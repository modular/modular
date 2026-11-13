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
"""The Laguna pipeline model."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from max._core.engine import Model
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import WeightData
from max.nn.ep_config import (
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.legacy.comm.ep import EPCommInitializer
from max.pipelines.architectures.llama3.model import LlamaModelBase
from max.pipelines.lib.graph_specs import GraphRole, GraphSpec
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.weights.quant import (
    apply_fused_kernel_flags,
    build_modelopt_nvfp4_config,
)
from transformers import AutoConfig
from typing_extensions import override

from .laguna import Laguna
from .model_config import LagunaConfig

logger = logging.getLogger("max.pipelines")

# A layer's routed experts are packed NVFP4 iff they ship per-group block
# scales. Matched against the state dict the weight adapter has already
# renamed, so there is no ``model.`` prefix and ``weight_packed`` is already
# ``weight``.
_EXPERT_BLOCK_SCALE = re.compile(
    r"layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight_scale"
)

# The two other tensors an NVFP4 projection needs alongside its block scales,
# the compressed-tensors global scales every Laguna checkpoint stores.
_NVFP4_SCALE_SIBLINGS = ("weight_global_scale", "input_global_scale")

# Compute dtypes a projection can be in and still be unquantized.
_UNQUANTIZED_DTYPES = (DType.bfloat16, DType.float16, DType.float32)


class LagunaModel(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Laguna pipeline model for text generation.

    Uses ``AlwaysSignalBuffersMixin`` since ``VocabParallelEmbedding`` and
    ``ColumnParallelLinear`` always require signal buffers for allreduce.
    """

    model_config_cls: ClassVar[type[Any]] = LagunaConfig

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> LagunaConfig:
        model_config = LagunaConfig.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )
        # Dtype detection first: the quant resolution reads the sniffed
        # attention dtype to decide whether attention is quantized at all.
        self._detect_state_dict_dtypes(model_config, state_dict)
        self._resolve_nvfp4_quant_config(
            self.huggingface_config, model_config, state_dict
        )
        self._setup_ep_config(model_config)
        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: LagunaConfig,
    ) -> None:
        self.ep_comm_initializer = None
        if model_config.ep_config is None:
            return
        self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
        self.ep_comm_initializer.ep_init(session)
        logger.info(
            f"EP initialized: node_id={model_config.ep_config.node_id}, "
            f"n_gpus={model_config.ep_config.n_gpus_per_node}, "
            f"n_nodes={model_config.ep_config.n_nodes}, "
            f"n_experts={model_config.ep_config.n_experts}, "
            f"max_tokens_per_rank={model_config.ep_config.max_tokens_per_rank}"
        )

    @override
    def graph_specs(self, model_config: LagunaConfig) -> dict[str, GraphSpec]:
        """Returns the decoder."""
        nn_model = Laguna(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)
        nn_model.load_state_dict(
            self.state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "laguna",
                graph_inputs,
                weights=nn_model.state_dict(),
            )
        }

    @staticmethod
    def _resolve_nvfp4_quant_config(
        huggingface_config: AutoConfig,
        model_config: LagunaConfig,
        state_dict: Mapping[str, WeightData],
    ) -> None:
        """Derives which layers are NVFP4 from the weights the checkpoint ships.

        Laguna's compute dtype is bf16 (embeddings, attention, norms, lm_head),
        so the base ``finalize`` asks ``parse_quant_config`` for bf16 and gets
        ``None``. The checkpoint is compressed-tensors NVFP4, numerically
        identical to modelopt NVFP4 (group size 16, e4m3 per-group scales, fp32
        global scales) but written with ``re:``-prefixed regex ``ignore``
        entries that MAX's fnmatch-glob matcher cannot read. So the scope of the
        quantization comes from the state dict instead: a layer's experts are
        packed iff they ship block scales, the only signal that survives a
        checkpoint mixing precisions across layers.

        Args:
            huggingface_config: The HuggingFace model configuration.
            model_config: The config to set ``quant_config`` on.
            state_dict: The checkpoint weights, already renamed by the weight
                adapter.

        Raises:
            ValueError: If a layer ships block scales without the global scales
                the NVFP4 matmul also needs, or if the sparse layers disagree
                about their shared-expert precision.
        """
        if model_config.quant_config is not None:
            return

        quantized_layers: set[int] = set()
        for name in state_dict:
            match = _EXPERT_BLOCK_SCALE.fullmatch(name)
            if match is None:
                continue
            quantized_layers.add(int(match.group(1)))
            module = name.removesuffix(".weight_scale")
            missing = [
                sibling
                for sibling in _NVFP4_SCALE_SIBLINGS
                if f"{module}.{sibling}" not in state_dict
            ]
            if missing:
                raise ValueError(
                    f"{module} ships NVFP4 block scales but not "
                    f"{', '.join(missing)}; the NVFP4 matmul needs all three "
                    "to dequantize."
                )

        if not quantized_layers:
            # A plain bf16 checkpoint: every MLP stays unquantized.
            return

        mlp_layer_types = model_config.mlp_layer_types or (
            ["sparse"] * model_config.num_hidden_layers
        )
        # A dense layer has no experts to sniff, so its own projections decide.
        quantized_layers.update(
            i
            for i, mlp_type in enumerate(mlp_layer_types)
            if mlp_type == "dense"
            and f"layers.{i}.mlp.gate_proj.weight_scale" in state_dict
        )

        # Attention is in every released Laguna's ignore list; trust the
        # weights rather than the list, since packed attention would have to
        # ship a non-float dtype.
        attn_dtype = model_config.attn_dtype or model_config.dtype
        attn_quantized_layers = (
            set()
            if attn_dtype in _UNQUANTIZED_DTYPES
            else set(range(model_config.num_hidden_layers))
        )

        quant_config = build_modelopt_nvfp4_config(
            huggingface_config,
            state_dict,
            # No resolved config: the per-layer sets below replace what its
            # ignore globs would otherwise decide.
            None,
        )
        model_config.quant_config = apply_fused_kernel_flags(
            dataclasses.replace(
                quant_config,
                mlp_quantized_layers=quantized_layers,
                attn_quantized_layers=attn_quantized_layers,
                shared_experts_weight_dtype=(
                    LagunaModel._shared_experts_weight_dtype(
                        state_dict, mlp_layer_types
                    )
                ),
                embedding_output_dtype=model_config.unquantized_dtype,
            ),
            state_dict,
        )

    @staticmethod
    def _shared_experts_weight_dtype(
        state_dict: Mapping[str, WeightData],
        mlp_layer_types: Sequence[str],
    ) -> DType | None:
        """Returns the shared experts' storage dtype, or ``None`` if packed.

        Args:
            state_dict: The checkpoint weights, already renamed by the weight
                adapter.
            mlp_layer_types: Per-layer ``"dense"`` / ``"sparse"`` MLP types.

        Returns:
            The dtype the shared-expert projections are stored in, or ``None``
            when they are packed NVFP4 like the routed experts.

        Raises:
            ValueError: If the sparse layers disagree.
        """
        dtypes = {
            state_dict[name].dtype
            for i, mlp_type in enumerate(mlp_layer_types)
            if mlp_type == "sparse"
            and (name := f"layers.{i}.mlp.shared_experts.gate_proj.weight")
            in state_dict
        }
        if len(dtypes) > 1:
            raise ValueError(
                "Laguna shared experts must share one precision across layers, "
                f"but the checkpoint mixes {sorted(str(d) for d in dtypes)}."
            )
        if not dtypes:
            return None
        shared_dtype = dtypes.pop()
        # Packed FP4 arrives as uint8; anything else is a real storage dtype
        # that keeps the shared experts out of the quantized path.
        return None if shared_dtype == DType.uint8 else shared_dtype

    @staticmethod
    def _detect_state_dict_dtypes(
        model_config: LagunaConfig, state_dict: Mapping[str, WeightData]
    ) -> None:
        """Reads the gate, correction-bias, attention and embedding dtypes.

        These are kept higher-precision than the FP4 experts and vary by
        checkpoint, so detect them from the loaded tensors rather than assuming
        the compute dtype. Each is uniform across layers. Expert precision is
        not: it varies per layer and is resolved separately into
        ``quant_config.mlp_quantized_layers``.
        """
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
            elif k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
            elif k.endswith("self_attn.q_proj.weight"):
                model_config.attn_dtype = v.dtype
        embed = state_dict.get("embed_tokens.weight")
        if embed is not None:
            model_config.unquantized_dtype = embed.dtype

    def _setup_ep_config(self, model_config: LagunaConfig) -> None:
        """Sets ``model_config.ep_config``, or disables EP on a single GPU."""
        num_devices = len(self.devices)
        if num_devices <= 1:
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
