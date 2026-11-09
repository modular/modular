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
"""The ``K3DSparkModel`` draft's config, which builds its module.

A DSpark checkpoint is only ever served as ``--draft-model`` beside a Kimi
K3 (ModuleV3) target, inside the fused speculative-decoding graph, so this
builds the draft's module and KV leaf for that graph. The leaf is the same
compressed MLA latent the target's own layers cache, at the target's own
cache dtype, so both arms of the fused MLA kernels serve the draft's block
call; the draft reads the checkpoint's own ``max_position_embeddings`` for
its length bound, and the graph's rope table adds the block's rows of slack
past it.
"""

from __future__ import annotations

from dataclasses import dataclass

from max import functional as F
from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import MLAKVCacheParams
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.tensor import default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from transformers import AutoConfig

from ..spec_mla import kimi_k3_mla_buffer_size
from .config import K3DSparkDraftConfig, parse_k3_dspark_config
from .dspark import K3DSpark, tensor_parallel_k3_dspark
from .weight_adapters import convert_k3_dspark_state_dict

__all__ = ["K3DSparkDraftArchConfig"]

# The fused MLA kernels' fixed rotary latent width: `flareMLA_decoding`
# asserts 576 (rotary) or 512 (NoPE), and the drafter's yarn rope slice makes
# its row the rotary one. The published checkpoint is kv_lora 512 + rope 64.
_MLA_KERNEL_LATENT_DIM = 576


def _supports_mla_latent_cache(
    draft: K3DSparkDraftConfig, cache_dtype: DType, num_devices: int
) -> bool:
    """Whether the fused MLA kernels can serve this draft at this TP degree.

    Mirrors the target's ``KimiK3Config.supports_mla_latent_cache``, clause
    for clause -- each is a kernel constraint, not a policy choice. The
    drafter differs only in what a False means: it has no decompressed path
    to fall back to, so its caller refuses the checkpoint instead of routing
    it somewhere slower.
    """
    heads = draft.num_attention_heads
    if not heads or heads % num_devices:
        return False
    per_device = heads // num_devices
    return (
        cache_dtype in (DType.bfloat16, DType.float8_e4m3fn)
        and draft.kv_lora_rank + draft.qk_rope_head_dim
        == _MLA_KERNEL_LATENT_DIM
        and (per_device <= 64 or per_device % 64 == 0)
    )


def _mla_buffer_size(
    context: ModelLoadContext,
    config: K3DSparkDraftConfig,
    kv_params: MLAKVCacheParams,
) -> int:
    """The drafter's own one-shot prefill plan cap.

    The target's sizing helper with the drafter's numbers: this many
    query heads per device, and the drafter's own window width. That
    window is wider than the block's ``block_size`` query rows: the block
    call runs at ``cache_length = committed + accepted``, and the
    accepted count can reach ``block_size + 1`` (the verify window's whole
    width), so the window past ``max_seq_len`` is ``block_size + 1``
    accepted rows plus the block's ``block_size`` -- the same
    ``2 * block_size + 1`` rows of slack the pool itself reserves for a
    block draft (``kv_cache/paged_kv_cache/_common.py``), which the helper
    counts from ``1 + num_draft_tokens``. The helper gates its fold
    shortcut on that whole window rather than the block's own query rows,
    so this cap always covers the prefill arm's TG worst case; where the
    block call folds anyway (``kimi_k3_mla_folds_on_decode_branch``) the
    plan is never read and the wider cap only costs the plan buffer's
    rows. An undersized cap would silently truncate the plan's trailing
    windows into garbage.

    The batch-slot count is the same resolution the target's own cap
    sizes with (``pipeline_model._spec_mla_buffer_size``): the user's
    ``--max-batch-size``, or what memory planning inferred for it -- the
    block call opens its window on every decode step, so a cap sized
    for one request would silently truncate every step past the first.

    Raises:
        ValueError: If no batch size is resolved to size the cap with.
    """
    assert kv_params.page_size is not None
    # ``planned_max_batch_size`` carries what memory planning inferred
    # when the user left ``--max-batch-size`` unset.
    max_batch_size: int | None = (
        context.pipeline_config.runtime.max_batch_size
        or context.memory_plan.planned_max_batch_size
    )
    if max_batch_size is None:
        raise ValueError(
            "Kimi K3: the DSpark draft sizes its MLA plan cap for a worst"
            " case over every batch slot, but max_batch_size is unresolved;"
            " pass --max-batch-size or let memory planning infer one."
        )
    return kimi_k3_mla_buffer_size(
        max_batch_context_length=(
            context.max_batch_total_tokens or context.max_seq_len
        ),
        max_seq_len=context.max_seq_len,
        page_size=kv_params.page_size,
        num_draft_tokens=2 * config.block_size,
        max_batch_size=max_batch_size,
        num_q_heads_per_device=config.num_attention_heads
        // max(1, len(kv_params.devices)),
        cache_dtype=kv_params.dtype,
        device_api=accelerator_api(),
    )


@dataclass
class K3DSparkDraftArchConfig:
    """The config of the ``K3DSparkModel`` draft, which builds its module."""

    max_position_embeddings: int = 1 << 20

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> K3DSparkDraftArchConfig:
        """Builds the config from the checkpoint's position bound."""
        del pipeline_config, max_seq_len
        assert model_config is not None
        hf = model_config.huggingface_config
        assert hf is not None
        return cls(max_position_embeddings=int(hf.max_position_embeddings))

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length the config allows."""
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Returns the checkpoint's ``max_position_embeddings``."""
        del model_config
        return int(huggingface_config.max_position_embeddings)

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[K3DSpark, K3DSparkDraftConfig]:
        """Builds the DSpark draft for the draft checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. The draft is sharded over the
        same devices as its target, with the same ``tp`` mesh axis.

        Raises:
            ValueError: If the method isn't ``dspark``.
        """
        pipeline_config = context.pipeline_config
        speculative = pipeline_config.speculative
        draft_model = pipeline_config.draft_model
        assert speculative is not None and draft_model is not None
        if not speculative.is_dspark():
            raise ValueError(
                "The K3DSparkModel draft is a DSpark draft and requires"
                " --speculative-method dspark, got"
                f" {speculative.speculative_method!r}."
            )
        config = parse_k3_dspark_config(draft_model.huggingface_config)
        kv_params = context.kv_params
        assert isinstance(kv_params, MLAKVCacheParams)
        devices = [ref.to_device() for ref in kv_params.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            draft = K3DSpark(
                config,
                kv_params,
                max_seq_len=context.max_seq_len,
                mla_buffer_size=_mla_buffer_size(context, config, kv_params),
            )
            tensor_parallel_k3_dspark(draft, mesh)
        return draft, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: K3DSparkDraftConfig
    ) -> dict[str, WeightData]:
        """Returns the draft tensors the built module compiles with.

        The module tree mirrors the checkpoint's names, so the conversion
        drops nothing but the unconsumed confidence head.
        """
        return convert_k3_dspark_state_dict(
            {name: weight.data() for name, weight in context.weights.items()},
            config,
        )

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MLAKVCacheParams:
        """Returns the draft's own KV leaf: the compressed MLA latent.

        One ``kv_lora_rank + qk_rope_head_dim`` row per token behind a single
        KV head -- on the published checkpoint 576 bytes at fp8 against the
        decompressed form's ``2 * heads * q_head_dim`` -- with the draft's
        full query-head count the MLA kernels dispatch on. The dtype follows
        the target's own cache (``cache_dtype``, the shared pool's): fp8 in
        production, where the block call folds at TP4 and above, and bf16
        selectable, where every TP takes the one-shot decompression prefill
        arm. Deliberately never a ``KVCacheQuantizationConfig``: the latent is
        stored unscaled, and a quantized leaf would make both fused arms
        unreachable.

        Raises:
            ValueError: If the fused MLA kernels cannot serve this checkpoint
                at this TP degree -- the drafter has no decompressed path to
                fall back to.
        """
        draft = parse_k3_dspark_config(huggingface_config)
        speculative = pipeline_config.speculative
        assert speculative is not None
        if not _supports_mla_latent_cache(draft, cache_dtype, len(devices)):
            raise ValueError(
                "The K3 DSpark draft serves only through the fused MLA"
                " kernels, which need a 576-wide latent (kv_lora_rank 512 +"
                " qk_rope_head_dim 64), a bf16 or fp8 KV cache, and a head"
                f" count that divides by {len(devices)} device(s); got"
                f" kv_lora_rank {draft.kv_lora_rank},"
                f" qk_rope_head_dim {draft.qk_rope_head_dim},"
                f" {draft.num_attention_heads} heads and a"
                f" {cache_dtype} cache. The decompressed draft cache no"
                " longer exists; see dspark/model_config.py."
            )
        params = kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=draft.kv_lora_rank + draft.qk_rope_head_dim,
            num_layers=draft.num_hidden_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            is_mla=True,
            num_q_heads=draft.num_attention_heads,
            speculative_method=speculative.speculative_method,
            num_draft_tokens=speculative.draft_width,
        )
        # `to_params` picks the subclass off ``is_mla``; the assert both
        # narrows it for the type checker and pins the leaf kind the fused
        # kernels' dispatch metadata depends on.
        assert isinstance(params, MLAKVCacheParams)
        return params
