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
"""The Engram n-gram module: hash+gather through main's op, signed-sqrt gate.

The V4.1 Engram mechanism, as one shared ``engram`` subgraph that runs before
layers ``engram_layer_ids`` (1 and 14 in the real checkpoint):

1. ``mo.engram.hash_gather`` -- main's existing graph API, driven through
   :class:`~...deepseekV41.layers.engram_lookup.EngramLookup` -- hashes the
   n-grams over the compressed token map and gathers FP8 rows out of the
   per-layer pinned host tables, producing this layer's MXFP8 operand
   (codes ``[T,6144]`` e4m3, scales ``[ceil(T/128),48,32,4,4]`` e8m0);
2. ``wkv`` (an MXFP8 GEMM) projects the row to per-copy keys plus a
   shared value;
3. the signed-sqrt gate (:func:`engram_gate`) folds the value into all
   ``hc_mult`` hyper-connection stream copies of ``hidden``.

No kernel or wrapper is duplicated here: the hash, gather and operand
layout all come from main's ``EngramLookup``, which owns the token map as
a trace-time constant. The compressed token map therefore arrives as a
constructor argument rather than a graph input (the loader already holds
it as a numpy array).

Call contract: the stream handed to :meth:`Engram.forward` is the one
**after** the previous layer's mHC post write-back (``post(y, stream)``
with the previous site's ``post``/``comb`` already applied), and the
result is what the **next** layer's mHC gates (``pre``/``post``/``comb``
and the collapse) read. Injecting before the write-back would push the
Engram contribution through the previous ``comb`` and feed the gates a
pre-write-back stream; the contribution does not commute with ``comb``
because each copy has its own key and gate.

The tables are **CPU graph inputs**, never Module weights; the only
parameters this module holds are ``wkv.{weight,scale}`` and
``{q,k}_weight``, whose tree paths must exactly match the checkpoint's
weight-adapter names. The tables bind as bit-preserving
``float8_e4m3fn``/``float8_e8m0fnu`` views of the pinned host rows
(main's loader convention: ``Buffer.from_numpy(bytes).view(dtype)``) --
the op reads raw bytes either way, and a numeric cast would corrupt
them. The hash constants (multipliers, primes, offsets)
come from the shared :class:`EngramLayout`, embedded per device inside
the lookup.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Module
from max.nn.legacy.kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul,
)
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41.layers.engram_lookup import (
    EngramLookup,
    EngramTokens,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from .mhc import claim_replicated

# Codes are 24 heads x 256 dims of FP8; scales one E8M0 byte per 32-element
# block, in the kernel's flattened SM100 atom layout per 128-token row group.
_SF_VECTOR_SIZE = 32


def _hash_gather(
    ids: TensorValue,
    positions: TensorValue,
    query_start_loc: TensorValue,
    dead_mask: TensorValue,
    lookback: TensorValue,
    lookback_dead: TensorValue,
    weight_0: TensorValue,
    scale_0: TensorValue,
    weight_1: TensorValue,
    scale_1: TensorValue,
    *,
    lookup: EngramLookup,
) -> tuple[TensorValue, TensorValue]:
    """Runs main's ``mo.engram.hash_gather`` and stacks both layers' operands.

    A free function wrapped in ``F.functional`` so that under TP,
    ``call_on_mesh`` runs it once per device: each rank embeds its own copy
    of the hash constants while the one pinned host table per layer is
    passed whole to every device.

    Args:
        ids: Int32 ``[T]`` token ids on the output device.
        positions: Int64 ``[T]`` absolute sequence positions.
        query_start_loc: Int32 ``[B+1]`` ragged offsets.
        dead_mask: Bool ``[T]``, True for image/dead tokens.
        lookback: Int32 ``[B,3]`` previous ids, newest first.
        lookback_dead: Bool ``[B,3]`` dead companion.
        weight_0: ``[rows_0,256]`` first layer codes on CPU, as a
            bit-preserving E4M3 view of the pinned host bytes.
        scale_0: ``[rows_0,8]`` first layer scales on CPU, as a
            bit-preserving E8M0 view.
        weight_1: ``[rows_1,256]`` second layer codes, E4M3 view.
        scale_1: ``[rows_1,8]`` second layer scales, E8M0 view.
        lookup: The main ``EngramLookup`` holding this model's layout,
            token map and hash constants.

    Returns:
        ``(codes, scales)``: e4m3 ``[2,T,6144]`` and e8m0
        ``[2,ceil(T/128),48,32,4,4]``, stacked in ``layer_ids`` order.
    """
    device = DeviceRef.from_device(ids.device)
    tokens = EngramTokens(
        ids=ids,
        positions=positions,
        query_start_loc=query_start_loc,
        dead_mask=dead_mask,
        lookback=lookback,
        lookback_dead=lookback_dead,
    )
    operands = lookup.full(
        tokens, [weight_0, scale_0, weight_1, scale_1], device
    )
    # Main's lookup splits the op's [2, ...] outputs into a dict keyed by
    # layer id; stack them back in construction order so the caller can
    # select its layer by table ordinal at runtime.
    ordered = [operands[layer_id] for layer_id in lookup.layer_ids]
    codes = ops.stack([op.codes for op in ordered], axis=0)
    scales = ops.stack([op.scales for op in ordered], axis=0)
    return codes, scales


_hash_gather_op = F.functional(_hash_gather)


def _select_ordinal(
    codes: TensorValue, scales: TensorValue, ordinal: TensorValue
) -> tuple[TensorValue, TensorValue]:
    """Selects one layer's operand by table ordinal.

    The ordinal is a per-call-site scalar, so a one-element axis-0 gather
    followed by a squeeze drops the layer axis dynamically -- the shared
    subgraph body cannot bake a static layer index in. (The gather kernel
    rejects rank-0 indices, hence the ``[1]`` index.)

    Args:
        codes: E4m3 ``[2,T,6144]`` stacked operands.
        scales: E8m0 ``[2,S]`` stacked scales.
        ordinal: Uint32 rank-0 scalar selecting this call site's layer.

    Returns:
        ``(codes, scales)``: e4m3 ``[T,6144]`` and e8m0 ``[S]``.
    """
    device = codes.device
    index = ops.reshape(ops.cast(ordinal, DType.int32).to(device), [1])
    return (
        ops.squeeze(ops.gather(codes, index, axis=0), 0),
        ops.squeeze(ops.gather(scales, index, axis=0), 0),
    )


_select_ordinal_op = F.functional(_select_ordinal)
_interleave_op = F.functional(block_scales_interleave)
_gemm_op = F.functional(dynamic_block_scaled_matmul)


def _gate(
    hidden: TensorValue,
    kv: TensorValue,
    q_weight: TensorValue,
    k_weight: TensorValue,
    alive: TensorValue,
    epsilon: float,
) -> TensorValue:
    """Runs main's ``mo.engram.gate_masked`` kernel, as V2's ``Engram`` does."""
    return ops.custom(
        "mo.engram.gate_masked",
        device=hidden.device,
        values=[hidden, kv, q_weight, k_weight, alive],
        out_types=[hidden.type],
        parameters={"epsilon": epsilon},
    )[0].tensor


_gate_op = F.functional(_gate)


def engram_gate(
    hidden: Tensor,
    kv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    dead_mask: Tensor,
    epsilon: float,
) -> Tensor:
    """The signed-sqrt gate: folds ``kv`` into every stream copy.

    For copy ``c`` with ``h = hidden[t, c]`` and its key ``key``::

        x = sum(h * q[c] * k[c] * key) * rms(h) * rms(key) / sqrt(hidden)
        gate = sigmoid(sign(x) * sqrt(max(|x|, 1e-6)))
        out = h + gate * value

    where ``rms(v) = rsqrt(mean(v * v) + epsilon)``, ``x >= 0`` takes the
    positive branch at ``x == 0``, and a dead position's gate is forced to
    zero. One fused kernel computes it in float32 from the bf16 inputs; it
    supports ``hidden_size`` 5120 and ``hc_mult`` 4.

    Args:
        hidden: ``[T, hc_mult, hidden_size]``, all mHC stream copies at the
            injection point (after the previous layer's post write-back,
            before this layer's gates).
        kv: ``[T, (hc_mult + 1) * hidden_size]`` bf16: one key per copy,
            then the shared value.
        q_weight: ``[hc_mult, hidden_size]`` bf16.
        k_weight: ``[hc_mult, hidden_size]`` bf16.
        dead_mask: Uint8 ``[T]``, 1 for dead or image positions (the kernel
            takes the complement, alive, as a bool).
        epsilon: The RMS epsilon.

    Returns:
        ``[T, hc_mult, hidden_size]`` in ``hidden``'s dtype.
    """
    alive = F.logical_not(dead_mask.cast(DType.bool))
    gated = _gate_op(hidden, kv, q_weight, k_weight, alive, epsilon)
    return claim_replicated(gated, hidden)


class _WkvWeight(Module[..., None]):
    """``wkv``'s MXFP8 weight, held as the checkpoint's two leaves.

    The tree paths ``wkv.weight`` / ``wkv.scale`` are the weight
    adapter's (``_fp8_linear`` of ``engram.wkv``): ``[25600,6144]`` e4m3
    and ``[800,192]`` e8m0.
    """

    weight: Tensor
    scale: Tensor


class Engram(Module[..., list[Tensor]]):
    """The Engram subgraph body over the operand row.

    The hash and gather run through main's ``EngramLookup``
    (``mo.engram.hash_gather``); the token map is a constructor argument,
    embedded as a trace-time constant like every other hash constant.

    Args:
        layer_ids: ``config.engram_layer_ids``, in order (the ids seed
            the multipliers, so any renumbering changes the hash
            constants on purpose).
        num_embeddings: ``config.engram_num_embeddings``, per layer.
        max_ngram_size: The n-gram width (4).
        n_heads: Hash heads per n-gram order (8).
        head_dim: Row width of each head's table slice (256).
        compressed_vocab_size: The 99,092-entry map's size.
        engram_vocab_size: The prime-search bound, one above where the
            layout starts drawing bucket primes (``engram_vocab_size - 1 =
            15,999,999`` real).
        hidden_size: The stream width (5120).
        hc_mult: Hyper-connection copies (4); ``wkv`` writes one key per
            copy plus the shared value, so ``25600 = 5 * 5120``.
        pad_token_id: Whose compressed id hashes where a source is
            missing (2).
        rms_norm_eps: The gate's RMS epsilon (1e-20).
        device: The compute device, or the TP mesh (the parameters are
            replicated over it).
        token_map: ``[vocab]`` int32 compressed-id map; the same array the
            serving model digest-checks and hands the constructor, owned
            here as a trace-time constant.
    """

    wkv: _WkvWeight
    q_weight: Tensor
    k_weight: Tensor

    def __init__(
        self,
        *,
        layer_ids: tuple[int, ...],
        num_embeddings: tuple[int, ...],
        max_ngram_size: int = 4,
        n_heads: int = 8,
        head_dim: int = 256,
        compressed_vocab_size: int = 99092,
        engram_vocab_size: int = 16_000_000,
        hidden_size: int = 5120,
        hc_mult: int = 4,
        pad_token_id: int = 2,
        rms_norm_eps: float = 1e-20,
        device: Device | DeviceMesh,
        token_map: np.ndarray,
    ) -> None:
        """Builds the parameter tree, the lookup and the hash constants."""
        super().__init__()
        self.layout = EngramLayout(
            layer_ids=layer_ids,
            num_embeddings=num_embeddings,
            max_ngram_size=max_ngram_size,
            n_heads=n_heads,
            head_dim=head_dim,
            compressed_vocab_size=compressed_vocab_size,
            pad_token_id=pad_token_id,
            vocab_size=engram_vocab_size,
        )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.rms_norm_eps = rms_norm_eps
        self.lookup = EngramLookup(self.layout, token_map)

        self.wkv = _WkvWeight()
        # The wkv input width is the gathered operand's: n_hash_cols
        # 256-byte table rows, i.e. ``n_hash_cols * head_dim`` (24*256 =
        # 6144), matching the weight adapter's spec.
        code_width = self.layout.n_hash_cols * head_dim
        self.wkv.weight = Tensor.zeros(
            [(hc_mult + 1) * hidden_size, code_width],
            dtype=DType.float8_e4m3fn,
            device=device,
        )
        self.wkv.scale = Tensor.zeros(
            [
                ((hc_mult + 1) * hidden_size) // _SF_VECTOR_SIZE,
                code_width // _SF_VECTOR_SIZE,
            ],
            dtype=DType.float8_e8m0fnu,
            device=device,
        )
        self.q_weight = Tensor.zeros(
            [hc_mult, hidden_size], dtype=DType.bfloat16, device=device
        )
        self.k_weight = Tensor.zeros(
            [hc_mult, hidden_size], dtype=DType.bfloat16, device=device
        )

    def _gather_operand(
        self,
        ids: Tensor,
        positions: Tensor,
        query_start_loc: Tensor,
        tables: list[Tensor],
        ordinal: Tensor,
        lookback: Tensor,
        lookback_dead: Tensor,
        dead_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Hashes and gathers this layer's MXFP8 operand.

        Runs main's hash-gather (once per device under TP), claims the
        rule-less results ``Replicated``, then selects this call site's
        layer by its table ordinal.

        Args:
            ids: Int32 ``[T]`` token ids on the compute device.
            positions: Int64 ``[T]`` absolute sequence positions.
            query_start_loc: Int32 ``[B+1]`` ragged request offsets.
            tables: ``[w0, s0, w1, s1]`` CPU pinned tables in ABI order,
                as bit-preserving E4M3/E8M0 views of the host rows.
            ordinal: Uint32 rank-0 scalar: this call site's layer.
            lookback: Int32 ``[B,3]`` previous ids, newest first.
            lookback_dead: Uint8 ``[B,3]`` dead companion.
            dead_mask: Uint8 ``[T]``, 1 for image/dead tokens.

        Returns:
            ``(codes, scales)``: e4m3 ``[T,6144]`` and e8m0
            ``[ceil(T/128),48,32,4,4]``, both ``Replicated``.
        """
        weight_0, scale_0, weight_1, scale_1 = tables
        codes_u8, scales_u8 = _hash_gather_op(
            ids,
            positions,
            query_start_loc,
            # The ABI carries the masks as u8 (1 = dead); the op wants
            # the bool main's own builder passes it.
            F.not_equal(dead_mask, 0),
            lookback,
            F.not_equal(lookback_dead, 0),
            weight_0,
            scale_0,
            weight_1,
            scale_1,
            lookup=self.lookup,
        )
        # The wrapped op has no sharding rule, so its result placement is
        # Unknown; claim it before any rule-based F op consumes it. Under
        # DP the rows are this rank's own (``ids``), so the claim follows
        # them.
        codes_u8, scales_u8 = (
            claim_replicated(codes_u8, ids),
            claim_replicated(scales_u8, ids),
        )
        codes, scales = _select_ordinal_op(codes_u8, scales_u8, ordinal)
        return (
            claim_replicated(codes, ids),
            claim_replicated(scales, ids),
        )

    def _project(self, codes: Tensor, scales: Tensor) -> Tensor:
        """``wkv``: the MXFP8 GEMM from the gathered row to ``[T,25600]``.

        Args:
            codes: E4m3 ``[T,6144]``.
            scales: E8m0 ``[ceil(T/128),48,32,4,4]``, the kernel's atom
                layout.

        Returns:
            Bf16 ``[T, (hc_mult+1)*hidden_size]``, ``Replicated``.
        """
        # [800,192] -> broadcast each block scale across its 32 output rows
        # -> [25600,192], then interleave into the [200,48,32,4,4] atoms.
        rows, cols = (int(d) for d in self.wkv.scale.shape)
        b_scales_2d = (
            self.wkv.scale.reshape([rows, 1, cols])
            .broadcast_to([rows, _SF_VECTOR_SIZE, cols])
            .reshape([rows * _SF_VECTOR_SIZE, cols])
        )
        b_scales = _interleave_op(b_scales_2d, sf_vector_size=_SF_VECTOR_SIZE)
        kv = _gemm_op(
            codes,
            self.wkv.weight,
            scales,
            b_scales,
            sf_vector_size=_SF_VECTOR_SIZE,
            out_type=DType.bfloat16,
        )
        # The GEMM wrapper has no sharding rule either.
        return claim_replicated(kv, codes)

    def forward(
        self,
        carry: list[Tensor],
        tables: list[Tensor],
        inputs: list[Tensor],
        hidden: Tensor | None = None,
        tokens: Tensor | None = None,
        positions: Tensor | None = None,
        query_start_loc: Tensor | None = None,
    ) -> list[Tensor]:
        """The Engram body over one forward's production operands.

        Args:
            carry: Opaque caller state threaded through the shared body;
                this body does not read it.
            tables: ``[w0, s0, w1, s1]`` pinned CPU tables in ABI order,
                as bit-preserving E4M3/E8M0 views of the host rows.
            inputs: ``[ordinal, lookback, lookback_dead, image_token_mask]``:
                the operand row, plus the u32 scalar that selects this
                call site's layer.
            hidden: ``[T, hc_mult, hidden_size]``, the un-collapsed mHC
                streams after the previous layer's post write-back; the
                output feeds this layer's gates (see the module docstring).
            tokens: Int64 ``[T]`` raw token ids (the op wants i32, so it
                is cast here).
            positions: Int64 ``[T]`` absolute sequence positions.
            query_start_loc: Int32 ``[B+1]`` ragged request offsets.

        Returns:
            ``[out]``: the gated streams, ``[T, hc_mult, hidden_size]``.

        Raises:
            ValueError: If any production operand is missing.
        """
        if (
            hidden is None
            or tokens is None
            or positions is None
            or query_start_loc is None
        ):
            raise ValueError(
                "the Engram body requires the mHC streams and the token "
                "batch (hidden, tokens, positions, query_start_loc)"
            )
        ordinal, lookback, lookback_dead, dead_mask = inputs
        ids = tokens.cast(DType.int32)
        qsl = query_start_loc.cast(DType.int32)
        codes, scales = self._gather_operand(
            ids,
            positions,
            qsl,
            tables,
            ordinal,
            lookback,
            lookback_dead,
            dead_mask,
        )
        kv = self._project(codes, scales)
        return [
            engram_gate(
                hidden,
                kv,
                self.q_weight,
                self.k_weight,
                dead_mask,
                self.rms_norm_eps,
            )
        ]
