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
"""Nemotron-H nn.Module layers (hybrid Mamba-2 + NoPE attention + relu2 MLP).

Math mirrors the HuggingFace ``NemotronHForCausalLM`` reference
(``torch_forward`` for the mixer). Translated to idiomatic MAX:

* Block: pre-norm RMSNorm -> mixer -> residual add (residual_in_fp32=False).
* Mamba-2 mixer: in_proj -> [gate, hidden_states_B_C, dt]; depthwise SiLU conv
  over hidden_states_B_C; SSD chunked scan (prefill+decode);
  gated RMSNorm (norm_before_gate=False); out_proj.
* Attention: GQA, NoPE (no rotary), no bias.
* MLP: relu2 = down(relu(up(x))**2), non-gated, no bias.

NOTE: the SSD + varlen-conv ops are not yet in the builtin kernel library, so a
graph that instantiates the mamba mixer cannot compile until that handoff lands
(see KERNEL_HANDOFF_register_ssd_in_builtin.md). The layer math here is written
against the verified HF reference and is ready to logit-verify once invocable.
"""

from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.nn.attention import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits, logits_postprocess

from .functional_ops import (
    causal_conv1d_varlen_fwd,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)
from .model_config import NemotronHConfig


def _relu2(x: TensorValue) -> TensorValue:
    """relu(x) ** 2 (Nemotron-H MLP activation, ``relu2``)."""
    r = ops.relu(x)
    return r * r


def _weight_dtype(
    model_dtype: DType, quant_config: QuantConfig | None
) -> DType:
    """Linear weight storage dtype. FP8 (per-tensor static) stores the weight
    as ``float8_e4m3fn``; the ``quantized_matmul`` kernel quantizes the bf16
    activation with ``input_scale`` and dequantizes. Otherwise the weight is the
    model dtype."""
    if quant_config is not None:
        return DType.float8_e4m3fn
    return model_dtype


class NemotronHMLP(Module):
    """relu2 MLP: ``down(relu(up(x))**2)``. Non-gated, no bias."""

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        wdtype = _weight_dtype(config.dtype, quant_config)
        self.up_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mlp_bias,
            quant_config=quant_config,
        )
        self.down_proj = Linear(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mlp_bias,
            quant_config=quant_config,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.down_proj(_relu2(self.up_proj(x)))


class NemotronHAttention(Module):
    """GQA attention, NoPE (no rotary), no bias.

    Position information flows through the SSM layers, so attention layers add
    no positional encoding (NoPE) — matching the HF reference where
    ``position_embeddings`` is unused.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        kv_layer_idx: int,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        self.kv_params: KVCacheParams = config.kv_params
        self.kv_layer_idx = kv_layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.scale = math.sqrt(1.0 / self.head_dim)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        # Attention projections always stay bf16 (in the FP8 checkpoint's
        # exclude list), so no quantization config is wired here.
        self.q_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=q_dim,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )
        self.k_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=kv_dim,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )
        self.v_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=kv_dim,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )
        self.o_proj = Linear(
            in_dim=q_dim,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        query = ops.reshape(self.q_proj(x), [-1, self.n_heads, self.head_dim])
        key = ops.reshape(self.k_proj(x), [-1, self.n_kv_heads, self.head_dim])
        value = ops.reshape(
            self.v_proj(x), [-1, self.n_kv_heads, self.head_dim]
        )

        # NoPE: write K/V to cache as-is (no rotary), then ragged flash attn.
        store_k_cache_ragged(kv_collection, key, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, value, input_row_offsets, layer_idx)

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=query,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, [total_seq_len, -1])
        return self.o_proj(attn_out)


class NemotronHMamba2Mixer(Module):
    """Mamba-2 mixer (selective state-space), matching ``NemotronHMamba2Mixer``.

    State plumbing (carried by ``model.py``):
    * conv1d state lives in a slot-indexed pool mutated in place by
      ``causal_conv1d_varlen_fwd`` (handles prefill + decode).
    * SSM state lives in a slot-indexed fp32 pool mutated in place by
      ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``: the kernel reads initial
      state from ``ssm_pool[slot_idx[b]]`` and writes the final state back to
      the same slot without any graph-side gather/scatter_nd/buffer_store.
      The SSD kernel serves both prefill and decode (decode = seqlen-1 seqs).
    """

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        self.config = config
        self.dtype = config.dtype
        self.nheads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ngroups = config.n_groups
        self.dstate = config.ssm_state_size
        self.conv_kernel = config.conv_kernel
        self.intermediate = config.mamba_intermediate_size
        self.conv_dim = config.conv_dim
        self.group_size = self.intermediate // self.ngroups
        self.eps = config.layer_norm_epsilon

        # in_proj: hidden -> [gate(intermediate), hidden_states_B_C(conv_dim),
        # dt(nheads)]. The fused HF ``in_proj.weight`` is split into three
        # separate Linears (the adapter row-slices the fused weight + replicates
        # the per-tensor FP8 scale). Three matmuls give contiguous gate /
        # hidden_BC / dt outputs — a single fused matmul + ``ops.split`` yields
        # strided views whose row stride breaks the downstream group-RMSNorm
        # reduce on GPU (CUDA misaligned-address).
        wdtype = _weight_dtype(config.dtype, quant_config)
        self.in_proj_gate = Linear(
            in_dim=config.hidden_size,
            out_dim=self.intermediate,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )
        self.in_proj_hidden_BC = Linear(
            in_dim=config.hidden_size,
            out_dim=self.conv_dim,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )
        self.in_proj_dt = Linear(
            in_dim=config.hidden_size,
            out_dim=self.nheads,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )
        self.out_proj = Linear(
            in_dim=self.intermediate,
            out_dim=config.hidden_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )

        # Depthwise conv1d weight [conv_dim, 1, K] (reshaped in adapter) + bias.
        self.conv1d_weight = Weight(
            "conv1d.weight",
            config.dtype,
            [self.conv_dim, 1, self.conv_kernel],
            device=DeviceRef.CPU(),
        )
        self.conv1d_bias: Weight | None = None
        if config.use_conv_bias:
            self.conv1d_bias = Weight(
                "conv1d.bias",
                config.dtype,
                [self.conv_dim],
                device=DeviceRef.CPU(),
            )

        # Per-head scalar SSM params (fp32).
        self.A_log = Weight(
            "A_log", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )
        self.D = Weight(
            "D", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )

        # Gated RMSNorm weight (group-normed). Direct weight (offset 0).
        self.norm_weight = Weight(
            "norm.weight",
            DType.float32,
            [self.intermediate],
            device=DeviceRef.CPU(),
        )

    @property
    def mamba_in_proj_out(self) -> int:
        return self.intermediate + self.conv_dim + self.nheads

    def _gated_group_rmsnorm(
        self, y: TensorValue, gate: TensorValue
    ) -> TensorValue:
        """HF ``Zamba2RMSNormGated`` with ``norm_before_gate=False``.

        ``y`` and ``gate`` are ``[N, intermediate]``. Matches the reference
        exactly: cast to fp32, ``y = y * silu(gate)``, group RMSNorm over groups
        of ``group_size``, cast back to the input dtype, THEN multiply by the
        (fp32) norm weight (so the final product is fp32, matching
        ``self.weight * hidden_states.to(input_dtype)``).
        """
        device = y.device
        input_dtype = y.dtype
        yf = ops.cast(y, DType.float32)
        gatef = ops.cast(gate, DType.float32)
        yf = yf * ops.silu(gatef)
        # Group RMSNorm over the last (group_size) axis. Use the dedicated
        # ``ops.rms_norm`` kernel (with an all-ones weight = pure normalization)
        # rather than a hand-rolled reshape+mean+rsqrt — the manual reduce over
        # a reshaped/fused tensor misaligns on GPU. rms_norm normalizes the last
        # dim and broadcasts back over the leading axes.
        yg = ops.reshape(yf, [-1, self.group_size])
        ones = ops.constant(
            np.ones((self.group_size,), np.float32), device=device
        )
        yg = ops.rms_norm(yg, ones, self.eps, multiply_before_cast=True)
        yf = ops.reshape(yg, [-1, self.intermediate])
        # Cast to input dtype, then multiply by the fp32 norm weight (upcasts
        # the product to fp32), exactly as the reference does.
        w = self.norm_weight.to(device)
        return w * ops.cast(yf, input_dtype)

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        ssm_pool: BufferValue,
        has_initial_state: TensorValue,
        slot_idx: TensorValue,
        query_start_loc: TensorValue,
    ) -> TensorValue:
        """Returns ``output[N, hidden]``.

        Per-mamba-layer state lives in two mutable graph-input pools:

        * conv: ``causal_conv1d_varlen_fwd`` mutates ``conv_pool`` in place at
          slot ``cache_indices[b] = slot_idx[b]`` (Qwen3.5 conv pattern).
        * SSM: ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` reads initial state
          from ``ssm_pool[slot_idx[b]]`` and writes the updated final state back
          to the same slot in-place (no graph-side gather/scatter_nd/
          buffer_store whole-pool RMW).

        ``query_start_loc`` is the ragged ``input_row_offsets``; both kernels
        need it as int32.
        """
        device = x.device
        query_start_loc = ops.cast(query_start_loc, DType.int32)
        # Three separate projections (contiguous outputs); see __init__ note.
        gate = self.in_proj_gate(x)  # [N, intermediate]
        hidden_BC = self.in_proj_hidden_BC(x)  # [N, conv_dim]
        dt = self.in_proj_dt(x)  # [N, nheads]

        # Depthwise SiLU conv over hidden_BC. Op expects [dim, total_seqlen].
        conv_w = ops.reshape(
            self.conv1d_weight.to(device), [self.conv_dim, self.conv_kernel]
        )
        conv_bias = (
            self.conv1d_bias.to(device)
            if self.conv1d_bias is not None
            else ops.constant(0.0, self.dtype, device=device).broadcast_to(
                [self.conv_dim]
            )
        )
        hidden_BC_t = ops.transpose(hidden_BC, 0, 1)  # [conv_dim, N]
        # Slot-indexed in-place conv: the kernel reads+writes the conv pool at
        # slot ``cache_indices[b] = slot_idx[b]`` (Qwen3.5 GatedDeltaNet conv
        # pattern). No graph-side gather/scatter — the pool is mutated directly.
        conv_out_t = causal_conv1d_varlen_fwd(
            x=hidden_BC_t,
            weight=conv_w,
            bias=conv_bias,
            conv_states=conv_pool,
            query_start_loc=query_start_loc,
            cache_indices=ops.cast(slot_idx, DType.int32),
            has_initial_state=has_initial_state,
            activation="silu",
        )
        conv_out = ops.transpose(conv_out_t, 0, 1)  # [N, conv_dim]

        # Split conv output: [hidden(intermediate), B(ng*ds), C(ng*ds)].
        gtss = self.ngroups * self.dstate
        hidden, B, C = ops.split(
            conv_out, [self.intermediate, gtss, gtss], axis=1
        )

        # Reshape for the SSD kernel.
        x_ssm = ops.reshape(hidden, [-1, self.nheads, self.head_dim])
        B = ops.reshape(B, [-1, self.ngroups, self.dstate])
        C = ops.reshape(C, [-1, self.ngroups, self.dstate])

        # A = -exp(A_log); the kernel applies dt softplus internally.
        A = ops.cast(ops.negate(ops.exp(self.A_log.to(device))), self.dtype)
        D = ops.cast(self.D.to(device), self.dtype)
        dt_bias = ops.cast(self.dt_bias.to(device), self.dtype)

        # In-place SSM-pool RMW: the kernel reads initial state from
        # ssm_pool[slot_idx[b]] (when has_initial_state[b]) and writes the
        # updated final state back to the same slot directly — no graph-side
        # buffer_load/gather/scatter_nd/buffer_store whole-pool round-trip.
        # This eliminates ~30% of decode GPU wall-clock (B200 profile).
        y = mamba2_ssd_chunk_scan_varlen_fwd_inplace(
            x=x_ssm,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            ssm_pool=ssm_pool,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            cache_indices=ops.cast(slot_idx, DType.uint32),
        )
        y = ops.reshape(y, [-1, self.intermediate])

        # Gated group RMSNorm (returns fp32), cast back to model dtype, out_proj
        # — matching the reference ``self.out_proj(scan_output.to(dtype))``.
        y = self._gated_group_rmsnorm(y, gate)
        return self.out_proj(ops.cast(y, self.dtype))


class NemotronHBlock(Module):
    """Pre-norm residual block dispatching to one of the three mixers."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        kv_layer_idx: int,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.kind = config.layer_kinds[layer_idx]
        self.norm = RMSNorm(
            config.hidden_size,
            dtype=config.dtype,
            eps=config.layer_norm_epsilon,
            multiply_before_cast=False,
        )
        self.mixer: Module
        if self.kind == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, quant_config=quant_config)
        elif self.kind == "attention":
            self.mixer = NemotronHAttention(config, kv_layer_idx)
        else:  # mlp
            self.mixer = NemotronHMLP(config, quant_config=quant_config)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "NemotronHBlock dispatches per-kind in NemotronH.__call__; do not "
            "call the block directly."
        )


class NemotronH(Module):
    """Full Nemotron-H decoder: embed -> hybrid blocks -> norm -> lm_head."""

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_params = config.kv_params
        self.return_logits = return_logits
        dev = config.devices[0]

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            dev,
        )

        # Map each attention layer to a sequential KV cache index. FP8 is wired
        # per-module: a Linear is FP8 only if its layer is in the config's FP8
        # set (driven by the checkpoint's exclude list); attention, conv1d,
        # norms and lm_head always stay bf16.
        blocks: list[NemotronHBlock] = []
        self.block_kinds: list[str] = list(config.layer_kinds)
        self.mamba_layer_indices: list[int] = []
        kv_idx = 0
        for li, kind in enumerate(config.layer_kinds):
            kv_layer = kv_idx if kind == "attention" else -1
            qc: QuantConfig | None = None
            if kind == "mamba" and li in config.fp8_mamba_layers:
                qc = quant_config
            elif kind == "mlp" and li in config.fp8_mlp_layers:
                qc = quant_config
            blocks.append(NemotronHBlock(config, li, kv_layer, quant_config=qc))
            if kind == "attention":
                kv_idx += 1
            if kind == "mamba":
                self.mamba_layer_indices.append(li)
        # LayerList so the Module weight-naming traversal prefixes each block's
        # weights as ``blocks.{i}.*`` (a plain Python list is not traversed).
        self.blocks = LayerList(blocks)

        self.norm_f = RMSNorm(
            config.hidden_size,
            dtype=config.dtype,
            eps=config.layer_norm_epsilon,
            multiply_before_cast=False,
        )
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            dtype=config.dtype,
            device=dev,
            has_bias=False,
        )

        # Dims for state pool allocation (model.py reads these).
        self.num_mamba_layers = len(self.mamba_layer_indices)
        self.conv_dim = config.conv_dim
        self.conv_kernel = config.conv_kernel
        self.mamba_nheads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim
        self.dstate = config.ssm_state_size

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        kv_collections: list[PagedCacheValues],
        slot_idx: TensorValue,
        conv_pools: list[BufferValue],
        ssm_pools: list[BufferValue],
        has_initial_state: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Run the hybrid stack.

        Returns the logits tuple from :func:`logits_postprocess`. The conv and
        SSM pools are mutable graph inputs mutated in place at slot
        ``slot_idx[batch_item]`` (the SSD kernel reads/writes the ssm_pool
        directly via ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``), so the
        only graph outputs are the logits.
        """
        h = self.embed_tokens(tokens)

        kv_i = 0
        mamba_i = 0
        for block in self.blocks:
            residual = h
            normed = block.norm(h)
            if block.kind == "mamba":
                out = block.mixer(
                    normed,
                    conv_pools[mamba_i],
                    ssm_pools[mamba_i],
                    has_initial_state,
                    slot_idx,
                    input_row_offsets,
                )
                mamba_i += 1
            elif block.kind == "attention":
                # The KV cache holds all attention layers; ``layer_idx`` selects
                # this layer's slice. There is one PagedCacheValues per device
                # (single-device here), so always index [0] — not [kv_i].
                layer_idx = ops.constant(
                    kv_i, DType.uint32, device=DeviceRef.CPU()
                )
                out = block.mixer(
                    layer_idx,
                    normed,
                    kv_collections[0],
                    input_row_offsets,
                )
                kv_i += 1
            else:  # mlp
                out = block.mixer(normed)
            h = residual + out

        return logits_postprocess(
            h,
            input_row_offsets,
            return_n_logits,
            self.norm_f,
            self.lm_head,
            self.return_logits,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Graph input types for the Nemotron-H language graph.

        Order: ``tokens, input_row_offsets, return_n_logits, *kv_inputs,
        slot_idx, *conv_pools, *ssm_pools, has_initial_state``. The conv pools
        are model-dtype mutable buffers; the SSM pools are fp32 mutable buffers
        mutated in place by ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` at
        slot ``slot_idx[batch_item]``. ``has_initial_state`` is ``[batch]``
        bool (empty for a fresh prefill, all-True for decode).
        """
        dev = self.config.devices[0]
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=dev
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=dev
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        kv_types = list(kv_params.get_symbolic_inputs().flatten())

        slot_idx_type = TensorType(
            DType.uint32, shape=["batch_size"], device=dev
        )
        conv_pool_types: list[TensorType | BufferType] = [
            BufferType(
                self.config.dtype,
                shape=["max_slots", self.conv_dim, self.conv_kernel - 1],
                device=dev,
            )
            for _ in range(self.num_mamba_layers)
        ]
        ssm_pool_types: list[TensorType | BufferType] = [
            BufferType(
                DType.float32,
                shape=[
                    "max_slots",
                    self.mamba_nheads,
                    self.mamba_head_dim,
                    self.dstate,
                ],
                device=dev,
            )
            for _ in range(self.num_mamba_layers)
        ]
        has_initial_state_type = TensorType(
            DType.bool, shape=["has_initial_state_len"], device=dev
        )

        base: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        return tuple(
            base
            + kv_types
            + [slot_idx_type]
            + conv_pool_types
            + ssm_pool_types
            + [has_initial_state_type]
        )
