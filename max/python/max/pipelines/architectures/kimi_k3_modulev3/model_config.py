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
"""Registry config for Kimi K3's text tower (ModuleV3).

Wraps :class:`~.model.KimiK3ModelConfig` instead of re-declaring its shape
fields, so the two configs share one source of truth rather than drifting
apart. No vision support, but composable later as a VLM wrapper's
``llm_config``. Every method accepts either the top-level multimodal
``config.json`` or an already-extracted text sub-config, via
:meth:`_get_text_config`.

:attr:`quant_config` and :attr:`ep_config` are built by
``KimiK3PipelineModel`` (``pipeline_model.py``), not here -- neither is
available at ``initialize()`` time. Mutually exclusive: EP doesn't support
MXFP4 routed experts yet.

Deliberately leaves out DSpark/speculative-decoding and checkpoint
weight-size estimation (:mod:`memory_planner`'s job). What stays here is
the KV-cache tree and the attention-geometry byte accounting that's
inseparable from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.experimental.sharding import DeviceMesh
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.utils import upper_bounded_default
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

from .model import KimiK3ModelConfig
from .state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    attn_cache,
    kda_state_regions,
)

__all__ = ["KimiK3Config"]


@dataclass(kw_only=True)
class KimiK3Config(ArchConfigWithKVCache):
    """Registry configuration for Kimi K3's text tower. See the module docstring."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        # The real checkpoint's routed experts ship packed this way; see
        # `weight_adapters.py` and `layers/moe_latent.py`.
        "float4_e2m1fnx2",
    }

    dtype: DType
    kv_params: KVCacheParamInterface
    devices: list[DeviceRef]
    max_seq_len: int
    model: KimiK3ModelConfig
    """The model-shape fields ``KimiK3Model`` consumes directly; see the
    module docstring for why these live here instead of being flattened."""

    max_batch_context_length: int
    """Cap on context tokens resident across a batch; sizes the MLA layers'
    one-shot prefill decompression buffer (see
    :meth:`mla_prefill_scratch_bytes`). Required rather than defaulted:
    ``arch.py``'s ``requires_max_batch_context_length=True`` guarantees the
    registry always resolves a value before this config is built."""

    quantization_encoding: SupportedEncoding | None = None

    mesh: DeviceMesh | None = None
    """Device mesh for sharding across multiple devices."""

    ep_config: EPConfig | None = None
    """Expert-parallel configuration, or ``None`` for replicated/TP experts.

    Built by ``KimiK3PipelineModel._init_distributed_runtime`` once a real
    session exists to initialize the exchange; see the module docstring.
    """

    ep_size: int = 1
    """``runtime.ep_size``: the expert-parallel degree, 1 for the TP split.

    Held separately from :attr:`ep_config` so the byte-accounting methods
    below can tell the two splits apart without one.
    """

    quant_config: QuantConfig | None = None
    """MXFP4 config when the checkpoint's routed experts arrive packed,
    built by ``KimiK3PipelineModel._create_model_config``; ``None`` for a
    bf16 checkpoint. See the module docstring."""

    ssm_dtype: DType = DType.bfloat16
    """Dtype of the KDA convolution history; the convolution kernel requires
    its history and ragged input to agree."""

    recurrent_state_dtype: DType = DType.float32
    """Dtype of the persistent KDA recurrent state. FP32 storage preserves
    recurrence precision across decode steps and prefill chunks; promoting
    only the arithmetic still rounds a BF16 pool on each call."""

    def get_kv_params(self) -> KVCacheParamInterface:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @property
    def attn_kv_params(self) -> KVCacheParams:
        """The MLA layers' cache, out of the ``{attn, state}`` tree.

        :attr:`kv_params` is the whole tree, which also carries the KDA
        state; nothing outside this module should reach into it directly.
        """
        return attn_cache(self.kv_params)

    @property
    def mla_latent_cache(self) -> bool:
        """Whether the MLA layers cache the compressed latent.

        Read off the KV params rather than re-derived, so the layer and the
        cache cannot disagree about which one is allocated.
        """
        return isinstance(self.attn_kv_params, MLAKVCacheParams)

    @property
    def tensor_parallel_degree(self) -> int:
        """Devices the tower is split across."""
        return len(self.devices)

    @property
    def expert_parallel(self) -> bool:
        """Whether the routed expert SET is split rather than its FFN width."""
        return self.ep_size > 1

    @staticmethod
    def _get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
        """Extracts the text sub-config, or returns it unchanged if already bare.

        Kimi K3's top-level ``config.json`` carries only the multimodal
        wrapper and has no ``vocab_size`` of its own, so a caller that skips
        the nesting silently gets a defaulted text model. Returning the input
        unchanged when ``text_config`` is absent lets every method here accept
        either the top-level config or an already-extracted text config (the
        shape a VLM wrapper's :class:`ArchVLConfigWithTextSubconfig` passes).
        """
        return getattr(huggingface_config, "text_config", huggingface_config)

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        text_config = cls._get_text_config(huggingface_config)
        return upper_bounded_default(
            upper_bound=text_config.max_position_embeddings,
            default=model_config.max_length,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return KimiK3Config._get_text_config(
            huggingface_config
        ).num_hidden_layers

    @classmethod
    def declared_state_dtypes(
        cls, pipeline_config: PipelineConfig, kv_cache_config: KVCacheConfig
    ) -> tuple[DType, DType]:
        """Returns ``(conv, recurrent)`` storage dtypes of the KDA state.

        Computed here rather than read off an instance, since the state
        leaves are declared before one exists. FP32 unless
        ``--state-pool-dtype`` overrides it: promoting only the arithmetic
        still rounds a BF16 pool every call.

        Raises:
            ValueError: If ``--state-pool-dtype`` names a dtype the pools
                cannot hold.
        """
        encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        conv = DType.float32 if encoding == "float32" else DType.bfloat16
        pool_dtype_name = kv_cache_config.state_pool_dtype
        pool_dtypes = {"bfloat16": DType.bfloat16, "float32": DType.float32}
        if pool_dtype_name is not None and pool_dtype_name not in pool_dtypes:
            raise ValueError(
                "Kimi K3 state_pool_dtype must be 'bfloat16' or 'float32',"
                f" got {pool_dtype_name!r}"
            )
        recurrent = (
            DType.float32
            if pool_dtype_name is None
            else pool_dtypes[pool_dtype_name]
        )
        return conv, recurrent

    # `flareMLA_decoding` asserts `depth == 576` (`mla.mojo:793`) and the
    # fused MLA graph ops are bf16/fp8; neither is a tuning knob. Real K3 is
    # 512 + 64, matching exactly.
    _MLA_KERNEL_LATENT_DIM = 576

    @staticmethod
    def _mla_latent_dim(text_config: AutoConfig) -> int:
        return text_config.kv_lora_rank + text_config.qk_rope_head_dim

    @staticmethod
    def supports_mla_latent_cache(
        text_config: AutoConfig, cache_dtype: DType, num_devices: int
    ) -> bool:
        """Whether the fused MLA kernels can serve this shape at this TP degree.

        False sends the model down the decompressed cache, which costs 64x
        the KV bytes and is only viable at bring-up context lengths. Every
        clause is a kernel constraint rather than a policy choice:

        * bf16 or fp8 KV, and a :attr:`_MLA_KERNEL_LATENT_DIM`-wide latent;
        * a per-device Q-head count the SM100 MLA decode kernel is correct
          at. Above 64 and not a multiple of 64 leaves a partial tail head
          group, which returns wrong values silently, so the kernel rejects
          it at compile time. K3 is 96 heads, so TP1 lands exactly there and
          only TP2 and above (48/24/12) can take the latent path.
        """
        heads = text_config.num_attention_heads
        if heads % num_devices:
            return False
        per_device = heads // num_devices
        return (
            cache_dtype in (DType.bfloat16, DType.float8_e4m3fn)
            and KimiK3Config._mla_latent_dim(text_config)
            == KimiK3Config._MLA_KERNEL_LATENT_DIM
            and (per_device <= 64 or per_device % 64 == 0)
        )

    @staticmethod
    def _attn_kv_params(
        text_config: AutoConfig,
        model: KimiK3ModelConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        """Builds KV params covering the MLA layers only.

        Latent when the fused MLA kernels can run this checkpoint: one
        compressed vector per token, ``kv_lora_rank + qk_rope_head_dim`` wide
        with a single KV head, rather than decompressed per-head K and V. At
        K3's widths that is 576 bytes/token/layer against ``2 x 96 x 192``,
        64x fewer. KDA layers hold constant-size conv and recurrent state
        instead, which :meth:`construct_kv_params` declares beside this leaf.
        """
        if pipeline_config.model.data_parallel_degree > 1:
            raise ValueError("Data parallelism is not supported for Kimi K3")
        num_full_attn = len(model.full_attn_layer_indices)
        speculative = pipeline_config.speculative
        speculative_method = (
            speculative.speculative_method if speculative else None
        )
        if KimiK3Config.supports_mla_latent_cache(
            text_config, cache_dtype, len(devices)
        ):
            return kv_cache_config.to_params(
                dtype=cache_dtype,
                n_kv_heads=1,
                head_dim=KimiK3Config._mla_latent_dim(text_config),
                num_layers=num_full_attn,
                devices=devices,
                data_parallel_degree=1,
                is_mla=True,
                num_q_heads=text_config.num_attention_heads,
                speculative_method=speculative_method,
            )
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
            n_kv_heads=text_config.num_attention_heads,
            head_dim=text_config.qk_nope_head_dim
            + text_config.qk_rope_head_dim,
            num_layers=num_full_attn,
            devices=devices,
            data_parallel_degree=1,
            speculative_method=speculative_method,
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParamInterface:
        """Builds the cache tree: the MLA layers' leaf beside the KDA state.

        The ``attn`` child is :meth:`_attn_kv_params`; the ``state`` child is
        the KDA conv window and recurrent state. A model with no KDA layers
        gets the attention leaf alone.
        """
        text_config = KimiK3Config._get_text_config(huggingface_config)
        model = KimiK3ModelConfig.from_huggingface_config(
            {"text_config": text_config.to_dict()}
        )
        attn = KimiK3Config._attn_kv_params(
            text_config,
            model,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )
        if not model.kda_layer_indices:
            # A leaf of zero bytes would be a page the pool cannot tile.
            return attn
        conv_dtype, recurrent_dtype = KimiK3Config.declared_state_dtypes(
            pipeline_config, kv_cache_config
        )
        state = RecurrentStateParams(
            regions=kda_state_regions(
                num_kda_layers=len(model.kda_layer_indices),
                conv_dim=3 * model.kda_num_heads * model.kda_head_dim,
                conv_kernel_size=model.kda_conv_kernel_size,
                num_heads=model.kda_num_heads,
                head_dim=model.kda_head_dim,
                conv_dtype=conv_dtype,
                recurrent_dtype=recurrent_dtype,
                num_devices=len(devices),
            ),
            devices=attn.devices,
            data_parallel_degree=attn.data_parallel_degree,
        )
        return MultiKVCacheParams.from_params(
            {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
        )

    def per_request_state_bytes(self) -> int:
        """GPU bytes one request occupies in the KDA state leaves, per device.

        These are pages of the KV pool, not a separate reservation. The
        pools follow the head split, so this is what ONE device pays.
        """
        if not self.model.kda_layer_indices:
            return 0
        tp = self.tensor_parallel_degree
        conv_dim = 3 * self.model.kda_num_heads * self.model.kda_head_dim
        conv = (
            (conv_dim // tp)
            * (self.model.kda_conv_kernel_size - 1)
            * self.ssm_dtype.size_in_bytes
        )
        rec = (
            (self.model.kda_num_heads // tp)
            * self.model.kda_head_dim
            * self.model.kda_head_dim
            * self.recurrent_state_dtype.size_in_bytes
        )
        return len(self.model.kda_layer_indices) * (conv + rec)

    def mla_prefill_scratch_bytes(self, max_batch_context_length: int) -> int:
        """GPU bytes ALL devices together hold the MLA prefill window in.

        The prefill branch decompresses a bounded window back into per-head
        K and V -- the one cost the latent cache's 64x-smaller footprint
        trades for. Counted over the FULL head count, summed across devices
        rather than per-device. Zero without the latent cache, whose padded
        parity form never opens this window.
        """
        if not self.mla_latent_cache or not self.model.full_attn_layer_indices:
            return 0
        return (
            max_batch_context_length
            * self.model.num_attention_heads
            * (self.model.qk_nope_head_dim + self.model.v_head_dim)
            * self.dtype.size_in_bytes
        )

    def routed_expert_row_bound(self, pipeline_config: PipelineConfig) -> int:
        """Returns an upper bound on the tokens one forward can carry.

        Which scheduler budget supplies it depends on the regime: chunked
        prefill makes ``max_batch_input_tokens`` hard; without it,
        ``max_batch_total_tokens`` (or else ``max_seq_len * max_batch_size``)
        bounds it instead. Speculative decoding adds to whichever applies,
        though this resolves to zero overhead with no speculative config.

        Raises:
            ValueError: When no budget bounds a context-encoding batch.
        """
        runtime = pipeline_config.runtime
        floor = runtime.max_batch_input_tokens
        if runtime.enable_chunked_prefill:
            charged = floor
        elif runtime.max_batch_total_tokens is not None:
            charged = max(floor, runtime.max_batch_total_tokens)
        elif runtime.max_batch_size is not None:
            charged = max(floor, self.max_seq_len * runtime.max_batch_size)
        else:
            raise ValueError(
                "Kimi K3: chunked prefill is disabled and neither"
                " max-batch-total-tokens nor max-batch-size is set, so"
                " nothing bounds a context-encoding batch and the routed"
                " experts' matmul cannot size its M grid. Re-enable chunked"
                " prefill, or set --max-batch-total-tokens."
            )
        speculative = pipeline_config.speculative
        num_speculative_tokens = (
            (speculative.num_speculative_tokens or 0)
            if speculative is not None
            else 0
        )
        max_requests = runtime.max_batch_size or floor
        return charged + num_speculative_tokens * max_requests

    def ep_comm_buffer_bytes(
        self, pipeline_config: PipelineConfig, *, use_allreduce: bool
    ) -> int:
        """GPU bytes ONE device holds the EP dispatch/combine buffers in.

        Deliberately not ``estimate_ep_memory_usage``: that helper sizes the
        dispatch buffer at ``n_experts`` rows per token, while what's
        actually allocated caps it at ``ep_size * top_k`` -- the two diverge
        7x at K3's 896 experts over EP8, enough to fail planning outright.
        Mirrors ``get_max_recv_tokens`` instead.

        Single-node only: a multi-node caller must add the combine SEND
        buffer separately.

        Args:
            pipeline_config: Supplies the row bound (see
                :meth:`routed_expert_row_bound`).
            use_allreduce: ``runtime.ep_use_allreduce``. The allreduce variant
                keeps dispatch and combine device-local, so its receive
                buffer covers only the local experts.

        Returns:
            Bytes per device, or 0 when expert parallelism is off.
        """
        if not self.expert_parallel:
            return 0
        n_ranks = self.ep_size
        top_k = self.model.num_experts_per_token
        max_tokens = self.routed_expert_row_bound(pipeline_config)
        if use_allreduce:
            recv_rows = max_tokens * min(
                self.model.num_experts // n_ranks, top_k
            )
        else:
            recv_rows = max_tokens * min(
                self.model.num_experts, n_ranks * top_k
            )
        # The dispatch carries the LATENT, not the hidden state; see
        # `KimiK3LatentMoE`'s class docstring.
        latent = self.model.routed_expert_hidden_size
        dispatch_elem = self.dtype.size_in_bytes
        return (
            (max_tokens + recv_rows) * latent * dispatch_elem
            + top_k * max_tokens * latent * self.dtype.size_in_bytes
        )

    def validate_tensor_parallel_split(self) -> None:
        """Raises unless every split axis divides by the device count.

        Checked once at config time rather than at the first shard, because
        a ``Weight.shard`` on an indivisible axis silently distributes the
        remainder instead of failing.
        """
        tp = self.tensor_parallel_degree
        if tp == 1:
            return
        model = self.model
        for name, value in (
            ("num_attention_heads", model.num_attention_heads),
            ("kda_num_heads", model.kda_num_heads),
            ("intermediate_size", model.intermediate_size),
            ("moe_intermediate_size", model.moe_intermediate_size),
            (
                "moe_intermediate_size * num_shared_experts",
                model.moe_intermediate_size * model.num_shared_experts,
            ),
            ("routed_expert_hidden_size", model.routed_expert_hidden_size),
            ("vocab_size", model.vocab_size),
        ):
            if value % tp:
                raise ValueError(
                    f"Kimi K3 cannot run at tensor-parallel degree {tp}:"
                    f" {name}={value} does not divide by it."
                )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a KimiK3Config instance from pipeline configuration."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be"
                " loaded. Please ensure the model repository contains a"
                " valid config.json file."
            )
        if model_config.data_parallel_degree > 1:
            raise ValueError("Data parallelism is not supported for Kimi K3")

        text_config = cls._get_text_config(huggingface_config)
        model = KimiK3ModelConfig.from_huggingface_config(
            {"text_config": text_config.to_dict()}
        )

        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype,
        )
        conv_state_dtype, recurrent_state_dtype = cls.declared_state_dtypes(
            pipeline_config, model_config.kv_cache
        )

        config_instance = cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            max_seq_len=max_seq_len,
            model=model,
            max_batch_context_length=(
                pipeline_config.runtime.max_batch_total_tokens
                or model_config.max_length
                or max_seq_len
            ),
            quantization_encoding=quantization_encoding,
            ssm_dtype=conv_state_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            ep_size=pipeline_config.runtime.ep_size,
        )
        config_instance.validate_tensor_parallel_split()
        return config_instance
