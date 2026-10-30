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
"""Kimi K3's full-depth model (ModuleV3), single-device, weight loading only.

Assembles the per-layer building blocks (KDA/MLA attention, dense/MoE FFN,
the attention-residual mixture, the output head) into the real
``num_hidden_layers``-deep model, stacking a real ``KimiK3DecoderLayer`` per
config-declared layer kind rather than a fixed toy depth. Not yet tied to the
registry -- no ``PipelineModel``, no ``arch.py`` entry, no KV cache manager;
that is later work. This module's job is purely: build the real graph shape
and load real checkpoint weights into it, so the result can be driven with an
actual forward pass.

Mirrors two things from ``kimi_k3/kimi_k3.py``'s top-level ``KimiK3`` easy to
miss by only porting the per-layer pieces:

* a FINAL, model-level ``kimi_k3_attn_res_mix_norm`` after the layer loop,
  over separate ``output_attn_res_proj``/``output_attn_res_norm`` weights
  distinct from any layer's own residual-mix weights -- skipping it silently
  feeds the output head the last layer's raw (unmixed) stream;
* ``lm_head`` is never tied to ``embed_tokens`` on this checkpoint
  (``tie_word_embeddings`` is false), so it gets its own weight.

Weight loading is a direct, from-scratch port of
``kimi_k3/weight_adapters.py``'s ``convert_kimi_k3_state_dict`` onto this
module tree, reading a plain ``dict[str, np.ndarray]`` (e.g. from
``safetensors.numpy.load_file``) instead of MAX's ``Weights``/``WeightData``
plumbing -- V2's own adapter only exists bound to that heavier
``PipelineConfig``/``Weights`` machinery, which this module does not depend
on. The four checkpoint-shape transforms it
needs (fused qkv conv, fused gate/up, stacked experts, de-padded ``A_log``)
are reproduced here rather than imported, so this module has no import
dependency on the ``kimi_k3`` (V2) package.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from max import functional as F
from max import tree
from max.driver import CPU
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.nn import Embedding, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.quant_config import QuantConfig
from max.nn.legacy.transformer import ReturnLogits
from max.nn.sequential import ModuleList
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Sharded,
)

from .layers import ep
from .layers.attn_residual import kimi_k3_attn_res_mix_norm
from .layers.decoder_layer import (
    KimiK3DecoderLayer,
    expert_parallel_decoder_layer,
    tensor_parallel_decoder_layer,
)
from .layers.kda_attention import KimiK3KDAAttention
from .layers.mla_attention import (
    KimiK3MLAAttention,
    kimi_k3_mla_prefill_plan,
    kimi_k3_nope_freqs_cis,
)
from .layers.mlp import KimiK3DenseMLP
from .layers.moe_latent import KimiK3LatentMoE
from .layers.norm import KIMI_RMS_NORM_EPS
from .layers.output_head import (
    kimi_k3_output_head,
    tensor_parallel_lm_head_weight,
)
from .state_cache import ATTN_CACHE_KEY, STATE_CACHE_KEY, attn_cache

_TEXT = "language_model.model"


@dataclass(kw_only=True)
class KimiK3ModelConfig:
    """The fields the full-depth assembly needs, read from ``text_config``.

    Deliberately not V2's ``KimiK3Config``: that class is built from a
    ``PipelineConfig``/``AutoConfig`` pair and carries TP/EP/quantization/KV
    cache fields this single-device, no-registry stage has no use for.
    """

    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    kda_layer_indices: list[int]
    full_attn_layer_indices: list[int]
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_kernel_size: int
    kda_gate_lower_bound: float | None
    num_attention_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    attn_res_block_size: int
    first_k_dense_replace: int
    intermediate_size: int
    routed_expert_hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    rms_norm_eps: float
    situ_beta: float
    situ_linear_beta: float
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1

    @classmethod
    def from_huggingface_config(
        cls, config: Mapping[str, object]
    ) -> KimiK3ModelConfig:
        """Builds the config from a Kimi K3 ``config.json``'s parsed dict.

        Args:
            config: The top-level parsed ``config.json``; this model's own
                fields live under its ``text_config`` key -- the top level
                carries only the multimodal wrapper (see ``model_config.py``).
        """
        text = config["text_config"]
        assert isinstance(text, dict)
        linear_cfg = text.get("linear_attn_config") or {}
        num_layers = text["num_hidden_layers"]
        # 1-indexed in the config; see `model_config.py::_layer_indices`.
        kda = sorted(i - 1 for i in linear_cfg.get("kda_layers", []))
        full = sorted(i - 1 for i in linear_cfg.get("full_attn_layers", []))
        if sorted(kda + full) != list(range(num_layers)):
            raise ValueError(
                "Kimi K3 linear_attn_config must partition all"
                f" {num_layers} layers; got kda={kda} full_attn={full}"
            )
        return cls(
            hidden_size=text["hidden_size"],
            vocab_size=text["vocab_size"],
            num_hidden_layers=num_layers,
            kda_layer_indices=kda,
            full_attn_layer_indices=full,
            kda_num_heads=linear_cfg.get("num_heads", 8),
            kda_head_dim=linear_cfg.get("head_dim", 32),
            kda_conv_kernel_size=linear_cfg.get("short_conv_kernel_size", 4),
            kda_gate_lower_bound=linear_cfg.get("gate_lower_bound"),
            num_attention_heads=text["num_attention_heads"],
            qk_nope_head_dim=text["qk_nope_head_dim"],
            qk_rope_head_dim=text["qk_rope_head_dim"],
            v_head_dim=text["v_head_dim"],
            q_lora_rank=text["q_lora_rank"],
            kv_lora_rank=text["kv_lora_rank"],
            attn_res_block_size=text["attn_res_block_size"],
            first_k_dense_replace=text["first_k_dense_replace"],
            intermediate_size=text["intermediate_size"],
            routed_expert_hidden_size=text["routed_expert_hidden_size"],
            moe_intermediate_size=text["moe_intermediate_size"],
            num_experts=text["num_experts"],
            num_experts_per_token=text["num_experts_per_token"],
            num_shared_experts=text["num_shared_experts"],
            rms_norm_eps=text["rms_norm_eps"],
            situ_beta=text["activation_situ_beta"],
            situ_linear_beta=text["activation_situ_linear_beta"],
            moe_renormalize=text.get("moe_renormalize", True),
            routed_scaling_factor=text.get("routed_scaling_factor", 1.0),
            num_expert_group=text.get("num_expert_group", 1),
            topk_group=text.get("topk_group", 1),
        )


def _distributed_kda_state(
    per_device: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    mapping: DeviceMapping,
) -> RecurrentStateInputsPerDevice[Tensor, Tensor]:
    """Wraps each device's raw KDA state leaves into one Tensor-API state.

    Mirrors ``nemotron_h_modulev3.nemotron_h._distributed_state``: both
    pools are sharded on their channel axis like the heads they hold, and
    the row ids -- the same on every device -- stay replicated.
    """
    leaves: list[RecurrentLeafInputs[Tensor, Tensor]] = []
    for index in range(len(per_device[0].leaves)):
        pools = [device.leaves[index].pool for device in per_device]
        spec = (None, TP) + (None,) * (pools[0].rank - 2)
        leaves.append(
            RecurrentLeafInputs(
                pool=Tensor.from_shard_values(
                    pools, NamedMapping(mapping.mesh, spec)
                ),
                live_row_ids=Tensor.from_shard_values(
                    [
                        device.leaves[index].live_row_ids
                        for device in per_device
                    ],
                    mapping,
                ),
            )
        )
    return RecurrentStateInputsPerDevice(leaves=tuple(leaves))


class KimiK3Model(Module[..., ModuleV3Outputs]):
    """The full-depth Kimi K3 text model: embedding, decoder stack, output head.

    Batch size 1 -- see ``decoder_layer.py``'s module docstring for that
    scope cut, which this inherits unchanged. Single device, tensor-parallel,
    or expert-parallel; see ``tensor_parallel_kimi_k3_model`` and
    ``expert_parallel_kimi_k3_model`` below.
    """

    def __init__(
        self,
        config: KimiK3ModelConfig,
        kv_params: KVCacheParamInterface | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        mla_buffer_size: int = 16384,
        quant_config: QuantConfig | None = None,
        ep_config: EPConfig | None = None,
    ) -> None:
        """Builds the model's layers and weights.

        Args:
            config: The checkpoint-derived model config every shape comes
            from.
            kv_params: This model's full cache tree -- an ``MLAKVCacheParams``
            leaf when there are no KDA layers, otherwise a
            ``MultiKVCacheParams`` of that leaf and the KDA recurrent state
            (see ``model_config.py``'s ``construct_kv_params``). ``None``
            runs every MLA layer padded and takes no KDA state.
            return_logits: Not a shape of ``config`` -- a per-request serving
            choice the registry hands ``PipelineModel`` directly (see
            ``KimiK3PipelineModel._instantiate_module``) -- so it is its own
            argument rather than a field on the checkpoint-derived
            ``KimiK3ModelConfig``.
            mla_buffer_size: One-shot MLA prefill buffer size. Same reasoning as
            ``return_logits``: a serving-time quantity
            (``KimiK3Config.max_batch_context_length``), not a checkpoint
            shape, so it is its own argument.
            quant_config: MXFP4 config for the routed experts. Mutually
            exclusive with ``ep_config``.
            ep_config: The routed experts' expert-parallel exchange setup, or
            ``None`` to run them replicated or TP-sharded. Only the config
            lives here: the exchange's real buffers aren't a checkpoint
            weight, so ``forward`` reads them off the tail of ``kv_inputs``
            instead of a stored attribute (which ``Module.stage`` would
            otherwise declare as one).
        """
        super().__init__()
        self.config = config
        self.kv_params = kv_params
        self.return_logits = return_logits
        self.mla_buffer_size = mla_buffer_size
        self.ep_config = ep_config
        attn_params = attn_cache(kv_params) if kv_params is not None else None
        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size, dim=config.hidden_size
        )

        kda_set = set(config.kda_layer_indices)
        layers: list[KimiK3DecoderLayer] = []
        full_attn_idx = 0
        for i in range(config.num_hidden_layers):
            is_kda = i in kda_set
            is_dense = i < config.first_k_dense_replace

            attention: KimiK3KDAAttention | KimiK3MLAAttention
            if is_kda:
                attention = KimiK3KDAAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.kda_num_heads,
                    head_dim=config.kda_head_dim,
                    conv_kernel_size=config.kda_conv_kernel_size,
                    gate_lower_bound=config.kda_gate_lower_bound,
                )
            else:
                attention = KimiK3MLAAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    kv_params=attn_params,
                    layer_idx=full_attn_idx,
                )
                full_attn_idx += 1

            ffn: KimiK3DenseMLP | KimiK3LatentMoE
            if is_dense:
                ffn = KimiK3DenseMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    beta=config.situ_beta,
                    linear_beta=config.situ_linear_beta,
                )
            else:
                ffn = KimiK3LatentMoE(
                    hidden_size=config.hidden_size,
                    latent_size=config.routed_expert_hidden_size,
                    moe_intermediate_size=config.moe_intermediate_size,
                    shared_intermediate_size=(
                        config.moe_intermediate_size * config.num_shared_experts
                    ),
                    num_experts=config.num_experts,
                    num_experts_per_token=config.num_experts_per_token,
                    beta=config.situ_beta,
                    linear_beta=config.situ_linear_beta,
                    rms_norm_eps=config.rms_norm_eps,
                    routed_scaling_factor=config.routed_scaling_factor,
                    num_expert_group=config.num_expert_group,
                    topk_group=config.topk_group,
                    moe_renormalize=config.moe_renormalize,
                    quant_config=quant_config,
                )

            layers.append(
                KimiK3DecoderLayer(
                    layer_idx=i,
                    attn_res_block_size=config.attn_res_block_size,
                    is_kda=is_kda,
                    is_dense=is_dense,
                    hidden_size=config.hidden_size,
                    attention=attention,
                    ffn=ffn,
                    rms_norm_eps=config.rms_norm_eps,
                )
            )
        self.layers = ModuleList(layers)

        self.final_norm_weight = Tensor.ones([config.hidden_size])
        self.lm_head_weight = Tensor.zeros(
            [config.vocab_size, config.hidden_size]
        )
        # Distinct from every layer's own residual-mix weights -- the
        # model-level mix the V2 top-level `__call__` applies after the last
        # decoder layer and before the output head.
        self.output_attn_res_proj_weight = Tensor.zeros([1, config.hidden_size])
        self.output_attn_res_norm_weight = Tensor.ones([config.hidden_size])

    def _place(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replicates the single-device inputs across the model's mesh.

        A peer-to-peer copy makes one device's stream wait on another's,
        which a captured device graph can't record; the collective syncs
        instead.
        """
        mesh = self.embed_tokens.weight.mesh
        if mesh.num_devices > 1:
            return (
                F.distributed_broadcast(tokens, mesh),
                F.distributed_broadcast(input_row_offsets, mesh),
            )
        return tokens.to(mesh), input_row_offsets.to(mesh)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        """Runs the decoder stack over the tokens.

        Args:
            tokens: ``[total_seq_len]`` token ids, ragged (batch size 1 at this
            scope, so ``total_seq_len == seq_len``).
            return_n_logits: ``[1]``, see ``kimi_k3_output_head``.
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix sums.
            kv_inputs: This model's ``kv_params`` cache tree, flattened. Empty
            runs every MLA layer padded; a KDA layer always needs its state
            here, since it has no padded form.
        """
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        hidden_states = self.embed_tokens(tokens)

        kv_collection = None
        state = None
        mla_prefill_plan = None
        mla_freqs_cis = None
        ep_buffers = None
        if kv_inputs:
            assert self.kv_params is not None, (
                "KimiK3Model needs kv_params to serve a real kv_inputs tensor"
            )
            graph_values = iter(x._graph_value for x in kv_inputs)
            if isinstance(self.kv_params, MultiKVCacheParams):
                kv_tree = self.kv_params.unflatten_kv_inputs(graph_values)
                attn_tree = kv_tree[ATTN_CACHE_KEY]
                state_tree = kv_tree.get(STATE_CACHE_KEY)
            else:
                attn_tree = self.kv_params.unflatten_kv_inputs(graph_values)
                state_tree = None

            kv_collection = PagedCacheValues.from_upstream(
                tree.leaves(attn_tree, leaf=KVCacheInputsPerDevice),
                hidden_states.mapping,
            )
            if state_tree is not None:
                state = _distributed_kda_state(
                    tree.leaves(state_tree, leaf=RecurrentStateInputsPerDevice),
                    hidden_states.mapping,
                )

            attn_params = attn_cache(self.kv_params)
            if isinstance(attn_params, MLAKVCacheParams):
                mla_prefill_plan = kimi_k3_mla_prefill_plan(
                    kv_params=attn_params,
                    kv_collection=kv_collection,
                    input_row_offsets=input_row_offsets,
                    buffer_size=self.mla_buffer_size,
                )
                mla_freqs_cis = kimi_k3_nope_freqs_cis(
                    max_seq_len=self.mla_buffer_size,
                    qk_rope_head_dim=self.config.qk_rope_head_dim,
                    dtype=attn_params.dtype,
                    device=CPU(),
                )
                # A constant every device needs, not one device's value.
                mla_freqs_cis = (
                    mla_freqs_cis.to(hidden_states.mesh)
                    if hidden_states.is_distributed
                    else mla_freqs_cis.to(hidden_states.device)
                )

            if self.ep_config is not None:
                # Not a checkpoint weight, so it arrives via the tail of
                # `kv_inputs` instead of a stored attribute (see `__init__`).
                # `counters` is one graph value per device; the other six
                # leaves share a single CPU-resident pointer table.
                mesh = self.embed_tokens.weight.mesh
                counters = Tensor.from_shard_values(
                    [next(graph_values) for _ in range(mesh.num_devices)],
                    DeviceMapping(mesh, (Sharded(axis=0),) * mesh.ndim),
                )
                ep_buffers = ep.EPBuffers(
                    counters=counters,
                    dispatch=ep.Tables(
                        send=Tensor.from_graph_value(next(graph_values)),
                        recv=Tensor.from_graph_value(next(graph_values)),
                        count=Tensor.from_graph_value(next(graph_values)),
                    ),
                    combine=ep.Tables(
                        send=Tensor.from_graph_value(next(graph_values)),
                        recv=Tensor.from_graph_value(next(graph_values)),
                        count=Tensor.from_graph_value(next(graph_values)),
                    ),
                )

        block_residual: list[Tensor] = []
        kda_idx = 0
        for layer in self.layers:
            if layer.is_kda:
                assert state is not None, (
                    "KimiK3Model needs KDA state in kv_inputs to serve a KDA"
                    " layer"
                )
                conv, recurrent = state.leaves[:2]
                hidden_states, block_residual = layer(
                    hidden_states,
                    block_residual,
                    conv_pool=conv.pool,
                    conv_row_id=conv.live_row_ids[kda_idx],
                    state_pool=recurrent.pool,
                    recurrent_row_id=recurrent.live_row_ids[kda_idx],
                    input_row_offsets=input_row_offsets,
                    ep_config=self.ep_config,
                    ep_buffers=ep_buffers,
                )
                kda_idx += 1
            else:
                hidden_states, block_residual = layer(
                    hidden_states,
                    block_residual,
                    input_row_offsets=input_row_offsets,
                    kv_collection=kv_collection,
                    mla_prefill_plan=mla_prefill_plan,
                    mla_freqs_cis=mla_freqs_cis,
                    ep_config=self.ep_config,
                    ep_buffers=ep_buffers,
                )

        # The final norm rides the fused mix; the output head takes the
        # already-normed hidden state (norm-then-gather is per-row
        # identical to V2's gather-then-norm).
        normed = kimi_k3_attn_res_mix_norm(
            hidden_states,
            block_residual,
            self.output_attn_res_proj_weight,
            self.output_attn_res_norm_weight,
            self.final_norm_weight,
            eps=self.config.rms_norm_eps,
            norm_eps=KIMI_RMS_NORM_EPS,
        )
        output = kimi_k3_output_head(
            normed,
            input_row_offsets,
            return_n_logits,
            self.lm_head_weight,
            self.return_logits,
        )
        mesh = self.embed_tokens.weight.mesh
        if mesh.num_devices > 1:
            # Logits are replicated across the mesh, but a compiled graph's
            # output must be a single device's value.
            output = tuple(t.to(mesh.devices[0]) for t in output)
        if len(output) == 1:
            return ModuleV3Outputs(next_token_logits=output[0])
        last_logits, logits, offsets = output
        return ModuleV3Outputs(
            next_token_logits=last_logits, logits=logits, logit_offsets=offsets
        )


def tensor_parallel_kimi_k3_model(
    model: KimiK3Model, mesh: DeviceMesh
) -> KimiK3Model:
    """Shards a weight-loaded model along the ``tp`` mesh axis.

    The model's weights must already be on ``mesh``. ``embed_tokens`` stays
    replicated: a vocab-parallel embedding needs a masked ``gather`` whose
    result is ``Partial``, and the default ``gather`` rule does not produce
    one.
    """
    for layer in model.layers:
        tensor_parallel_decoder_layer(layer, mesh)
    model.lm_head_weight = tensor_parallel_lm_head_weight(
        model.lm_head_weight, mesh
    )
    return model


def expert_parallel_kimi_k3_model(
    model: KimiK3Model, mesh: DeviceMesh
) -> KimiK3Model:
    """Shards a weight-loaded model's routed experts along the ``ep`` mesh axis.

    Everything else -- attention, dense MLPs, ``lm_head`` -- stays
    replicated; the experts' exchange is this mode's only communication.
    Call :meth:`KimiK3Model.forward` with ``ep_config``/``ep_buffers`` set to
    use the sharded experts.
    """
    for layer in model.layers:
        expert_parallel_decoder_layer(layer, mesh)
    return model


def _fuse_gate_up(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Gate-first concatenation; mirrors ``kimi_k3/layers/mlp.py::fuse_gate_up``."""
    return np.ascontiguousarray(np.concatenate([gate, up], axis=0))


def _fuse_qkv_conv_weight(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Packs three ``[C, 1, K]`` depthwise convs into one ``[3C, K]`` weight.

    ``KimiK3KDAAttention.conv_weight`` takes the packed weight. Mirrors
    ``kimi_k3/layers/short_conv.py::fuse_qkv_conv_weight``.
    """
    return np.ascontiguousarray(
        np.concatenate([w.reshape(w.shape[0], -1) for w in (q, k, v)], axis=0)
    )


# Each expert copy is a memcpy plus a page fault per source page, and numpy
# releases the GIL for both, so overlapping several copies in threads hides
# that latency; bounded low since the win is concurrent faults, not cores.
_STACK_WORKERS = min(16, os.cpu_count() or 8)

# Above this, an expert stack is backed by a file instead of anonymous memory,
# so its pages stay reclaimable under memory pressure.
_SPILL_BYTES = int(os.environ.get("MAX_KIMI_K3_STACK_SPILL_BYTES", 4 << 30))
_SPILL_DIR = os.environ.get("MAX_KIMI_K3_STACK_SPILL_DIR") or None


def _stack_output(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocates an expert stack, spilling to a file past ``_SPILL_BYTES``.

    The file is unlinked as soon as it is mapped: the mapping keeps the inode
    alive, so the space is returned when the array is freed and nothing is
    left behind if the process dies.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes < _SPILL_BYTES:
        return np.empty(shape, dtype=dtype)

    fd, path = tempfile.mkstemp(prefix="max-k3-expert-stack-", dir=_SPILL_DIR)
    os.close(fd)
    try:
        out = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    finally:
        os.unlink(path)
    return out


def _settle(out: np.ndarray) -> np.ndarray:
    """Writes a spilled stack back, so its pages become reclaimable.

    Spilling makes the bytes file-backed, but a freshly written shared mapping
    is DIRTY, and dirty page cache is no more reclaimable than anonymous
    memory -- so one ``msync`` per stack makes the pages clean and lets the
    kernel drop them under pressure.
    """
    if isinstance(out, np.memmap):
        out.flush()
    return out


def _copy_into(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Runs ``dst[...] = src`` for each pair, concurrently.

    Every ``dst`` is a disjoint slice of one preallocated output, so the
    writes do not race and the peak stays one output rather than two.
    """
    if len(pairs) < 2 or _STACK_WORKERS < 2:
        for dst, src in pairs:
            dst[...] = src
        return

    def one(pair: tuple[np.ndarray, np.ndarray]) -> None:
        dst, src = pair
        dst[...] = src

    with ThreadPoolExecutor(max_workers=_STACK_WORKERS) as pool:
        # list() rather than a bare map(): map is lazy, and an exception in a
        # worker only surfaces when its result is consumed.
        list(pool.map(one, pairs))


def _stack_experts(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Stacks per-expert tensors into ``[E, *parts[0].shape]``.

    ``np.stack`` allocates the same output and fills it serially; this fills
    it through :func:`_copy_into` instead.
    """
    if not parts:
        raise ValueError("_stack_experts needs at least one expert")
    out = _stack_output((len(parts), *parts[0].shape), parts[0].dtype)
    _copy_into([(out[e], p) for e, p in enumerate(parts)])
    return _settle(out)


def _stack_gate_up(
    gate: Sequence[np.ndarray], up: Sequence[np.ndarray]
) -> np.ndarray:
    """Stacks per-expert gate and up into ``[E, gate_rows + up_rows, cols]``.

    Writes each expert into a preallocated output instead of fusing every
    expert first and stacking the list, which costs the same result at twice
    the peak memory and serializes every expert's page faults.
    """
    if not gate:
        raise ValueError("_stack_gate_up needs at least one expert")
    split = gate[0].shape[0]
    out = _stack_output(
        (len(gate), split + up[0].shape[0], *gate[0].shape[1:]),
        gate[0].dtype,
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for e, (g, u) in enumerate(zip(gate, up, strict=True)):
        pairs.append((out[e, :split], g))
        pairs.append((out[e, split:], u))
    _copy_into(pairs)
    return _settle(out)


def _stack_expert_weights(
    w1: list[np.ndarray], w2: list[np.ndarray], w3: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Stacks per-expert checkpoint weights into the two grouped tensors.

    ``w1`` is gate, ``w3`` is up, ``w2`` is down -- the checkpoint's own
    numbering does not follow the data flow.
    """
    return _stack_gate_up(w1, w3), _stack_experts(w2)


def _canonical_a_log(
    values: np.ndarray, num_heads: int, head_dim: int
) -> np.ndarray:
    """Broadcasts per-head gate coefficients.

    Exact-zero storage padding is dropped, mirroring
    ``kimi_k3/weight_adapters.py::_canonical_a_log``. That version handles a
    bf16-as-uint16 checkpoint; this fixture stores A_log as fp32.
    """
    if values.ndim != 1 or values.size < num_heads:
        raise ValueError(
            f"Kimi K3: A_log has shape {values.shape}; expected a per-head"
            f" vector with at least {num_heads} coefficients."
        )
    if not np.all(np.isfinite(values[:num_heads])):
        raise ValueError(
            "Kimi K3: active A_log head coefficients must be finite."
        )
    if np.any(values[num_heads:] != 0):
        raise ValueError("Kimi K3: A_log storage padding must be exactly zero.")
    return np.broadcast_to(values[:num_heads, None], (num_heads, head_dim))


def load_kimi_k3_weights(
    model: KimiK3Model, raw: Mapping[str, np.ndarray]
) -> None:
    """Loads a real Kimi K3 checkpoint's tensors onto ``model``'s module tree.

    Args:
        model: A :class:`KimiK3Model` already constructed from the same
            config the checkpoint was published with.
        raw: Checkpoint tensors keyed by their original (``language_model.``-
            prefixed) checkpoint names, e.g. from ``safetensors.numpy.load_file``.
    """
    config = model.config

    def get(key: str) -> np.ndarray:
        return raw[key]

    model.embed_tokens.weight = Tensor(
        get(f"{_TEXT}.embed_tokens.weight"), dtype=DType.float32
    )
    model.lm_head_weight = Tensor(
        get("language_model.lm_head.weight"), dtype=DType.float32
    )
    model.final_norm_weight = Tensor(
        get(f"{_TEXT}.norm.weight"), dtype=DType.float32
    )
    model.output_attn_res_proj_weight = Tensor(
        get(f"{_TEXT}.output_attn_res_proj.weight"), dtype=DType.float32
    )
    model.output_attn_res_norm_weight = Tensor(
        get(f"{_TEXT}.output_attn_res_norm.weight"), dtype=DType.float32
    )

    for i, layer in enumerate(model.layers):
        src = f"{_TEXT}.layers.{i}"
        layer.input_layernorm_weight = Tensor(
            get(f"{src}.input_layernorm.weight"), dtype=DType.float32
        )
        layer.post_attention_layernorm_weight = Tensor(
            get(f"{src}.post_attention_layernorm.weight"), dtype=DType.float32
        )
        layer.self_attention_res_proj_weight = Tensor(
            get(f"{src}.self_attention_res_proj.weight"), dtype=DType.float32
        )
        layer.self_attention_res_norm_weight = Tensor(
            get(f"{src}.self_attention_res_norm.weight"), dtype=DType.float32
        )
        layer.mlp_res_proj_weight = Tensor(
            get(f"{src}.mlp_res_proj.weight"), dtype=DType.float32
        )
        layer.mlp_res_norm_weight = Tensor(
            get(f"{src}.mlp_res_norm.weight"), dtype=DType.float32
        )

        attn_src = f"{src}.self_attn"
        attn = layer.attention
        if layer.is_kda:
            assert isinstance(attn, KimiK3KDAAttention)
            attn.q_proj.weight = Tensor(
                get(f"{attn_src}.q_proj.weight"), dtype=DType.float32
            )
            attn.k_proj.weight = Tensor(
                get(f"{attn_src}.k_proj.weight"), dtype=DType.float32
            )
            attn.v_proj.weight = Tensor(
                get(f"{attn_src}.v_proj.weight"), dtype=DType.float32
            )
            attn.f_a_proj.weight = Tensor(
                get(f"{attn_src}.f_a_proj.weight"), dtype=DType.float32
            )
            attn.f_b_proj.weight = Tensor(
                get(f"{attn_src}.f_b_proj.weight"), dtype=DType.float32
            )
            attn.b_proj.weight = Tensor(
                get(f"{attn_src}.b_proj.weight"), dtype=DType.float32
            )
            attn.g_proj.weight = Tensor(
                get(f"{attn_src}.g_proj.weight"), dtype=DType.float32
            )
            attn.o_proj.weight = Tensor(
                get(f"{attn_src}.o_proj.weight"), dtype=DType.float32
            )
            attn.o_norm.weight = Tensor(
                get(f"{attn_src}.o_norm.weight"), dtype=DType.float32
            )
            conv = [
                get(f"{attn_src}.{axis}_conv1d.weight")
                for axis in ("q", "k", "v")
            ]
            attn.conv_weight = Tensor(
                _fuse_qkv_conv_weight(*conv), dtype=DType.float32
            )
            attn.a_log = Tensor(
                _canonical_a_log(
                    get(f"{attn_src}.A_log"),
                    config.kda_num_heads,
                    config.kda_head_dim,
                ),
                dtype=DType.float32,
            )
            attn.dt_bias = Tensor(
                get(f"{attn_src}.dt_bias").reshape(
                    config.kda_num_heads, config.kda_head_dim
                ),
                dtype=DType.float32,
            )
        else:
            assert isinstance(attn, KimiK3MLAAttention)
            attn.q_a_proj.weight = Tensor(
                get(f"{attn_src}.q_a_proj.weight"), dtype=DType.float32
            )
            attn.q_a_layernorm.weight = Tensor(
                get(f"{attn_src}.q_a_layernorm.weight"), dtype=DType.float32
            )
            attn.q_b_proj.weight = Tensor(
                get(f"{attn_src}.q_b_proj.weight"), dtype=DType.float32
            )
            attn.kv_a_proj_with_mqa.weight = Tensor(
                get(f"{attn_src}.kv_a_proj_with_mqa.weight"),
                dtype=DType.float32,
            )
            attn.kv_a_layernorm.weight = Tensor(
                get(f"{attn_src}.kv_a_layernorm.weight"), dtype=DType.float32
            )
            attn.kv_b_proj.weight = Tensor(
                get(f"{attn_src}.kv_b_proj.weight"), dtype=DType.float32
            )
            attn.g_proj.weight = Tensor(
                get(f"{attn_src}.g_proj.weight"), dtype=DType.float32
            )
            attn.o_proj.weight = Tensor(
                get(f"{attn_src}.o_proj.weight"), dtype=DType.float32
            )

        ffn = layer.ffn
        if layer.is_dense:
            assert isinstance(ffn, KimiK3DenseMLP)
            gate = get(f"{src}.mlp.gate_proj.weight")
            up = get(f"{src}.mlp.up_proj.weight")
            ffn.gate_up_proj.weight = Tensor(
                _fuse_gate_up(gate, up), dtype=DType.float32
            )
            ffn.down_proj.weight = Tensor(
                get(f"{src}.mlp.down_proj.weight"), dtype=DType.float32
            )
        else:
            assert isinstance(ffn, KimiK3LatentMoE)
            moe = f"{src}.block_sparse_moe"
            ffn.gate.gate_score.weight = Tensor(
                get(f"{moe}.gate.weight"), dtype=DType.float32
            )
            ffn.gate.e_score_correction_bias = Tensor(
                get(f"{moe}.gate.e_score_correction_bias"), dtype=DType.float32
            )
            ffn.routed_expert_down_proj.weight = Tensor(
                get(f"{moe}.routed_expert_down_proj.weight"),
                dtype=DType.float32,
            )
            ffn.routed_expert_up_proj.weight = Tensor(
                get(f"{moe}.routed_expert_up_proj.weight"), dtype=DType.float32
            )
            ffn.routed_expert_norm.weight = Tensor(
                get(f"{moe}.routed_expert_norm.weight"), dtype=DType.float32
            )
            w1 = [
                get(f"{moe}.experts.{e}.w1.weight")
                for e in range(config.num_experts)
            ]
            w2 = [
                get(f"{moe}.experts.{e}.w2.weight")
                for e in range(config.num_experts)
            ]
            w3 = [
                get(f"{moe}.experts.{e}.w3.weight")
                for e in range(config.num_experts)
            ]
            gate_up, down = _stack_expert_weights(w1, w2, w3)
            ffn.experts_gate_up_proj = Tensor(gate_up, dtype=DType.float32)
            ffn.experts_down_proj = Tensor(down, dtype=DType.float32)
            shared_gate = get(f"{moe}.shared_experts.gate_proj.weight")
            shared_up = get(f"{moe}.shared_experts.up_proj.weight")
            ffn.shared_experts.gate_up_proj.weight = Tensor(
                _fuse_gate_up(shared_gate, shared_up), dtype=DType.float32
            )
            ffn.shared_experts.down_proj.weight = Tensor(
                get(f"{moe}.shared_experts.down_proj.weight"),
                dtype=DType.float32,
            )
