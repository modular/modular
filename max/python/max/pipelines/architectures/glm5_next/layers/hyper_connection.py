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

"""Manifold-constrained hyper-connections (mHC) for GLM-5.3-Flash.

Per-architecture, as DeepSeek-V4's ``layers/hyper_connection.py`` is. Both
call the same :func:`~max.nn.kernels.hyper_connection_gates` kernel.

TODO(brodriguez): fold this into :class:`~max.nn.HyperConnection`, which now
carries the same ragged layout. That swap renames the three weights and moves
the RMS scale onto the projection, so it wants its own accuracy run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.kernels import hyper_connection_gates
from max.nn.layer import Module, Shardable

from ..model_config import Glm5NextConfig

__all__ = [
    "HyperConnection",
    "StreamMixing",
    "expand_streams",
    "hyper_connection_site",
    "mean_collapse_streams",
]


def expand_streams(x: TensorValue, hc_mult: int) -> TensorValue:
    """Widens a single residual into ``hc_mult`` identical streams.

    Model entry. The reference is
    ``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)``, so all streams
    start identical and diverge only through the per-site ``post`` and ``comb``.

    Args:
        x: Token embeddings, ``[total_tokens, hidden_size]``.
        hc_mult: Number of parallel residual streams.

    Returns:
        ``[total_tokens, hc_mult, hidden_size]``.
    """
    return ops.broadcast_to(
        ops.unsqueeze(x, -2), (x.shape[0], hc_mult, x.shape[-1])
    )


def mean_collapse_streams(streams: TensorValue) -> TensorValue:
    """Collapses the stream axis by an unweighted mean.

    Model exit, before the final norm. GLM-5.3-Flash's collapse is unweighted;
    DeepSeek-V4 instead learns it through an ``hc_head``, a weight the GLM
    checkpoint does not carry, so an implementation cloned from DeepSeek-V4
    fails to load.

    Args:
        streams: ``[total_tokens, hc_mult, hidden_size]``.

    Returns:
        ``[total_tokens, hidden_size]``.
    """
    # `ops.mean` keeps the rank with the reduced axis at size 1; the squeeze is
    # what drops the stream axis.
    return ops.squeeze(ops.mean(streams, axis=-2), -2)


@dataclass(frozen=True)
class StreamMixing:
    """One mHC site's mapping outputs.

    :attr:`post` and :attr:`comb` stay float32 until the write-back casts them,
    matching the reference: a bias in the projection would compound over all 90
    sites of the model.
    """

    post: TensorValue
    """Placement of the sublayer output into each stream, ``[total_tokens,
    hc_mult]`` float32 in ``[0, 2]``."""

    comb: TensorValue
    """Doubly-stochastic stream mixer, ``[total_tokens, hc_mult, hc_mult]``
    float32. The write-back contracts over its *first* stream axis."""

    xs: TensorValue
    """The sequence the sublayer consumes, ``[total_tokens, hidden_size]`` in
    the model's compute dtype, already normalized when
    :meth:`HyperConnection.__call__` was given a gamma."""


class HyperConnection(Module, Shardable):
    """One manifold-constrained hyper-connection site.

    Replaces a pre-norm block's residual add. Where a standard block computes
    ``x' = x + f(norm(x))`` over one residual, mHC keeps ``hc_mult`` parallel
    streams and makes both the read and the write learned and input-dependent
    (`Xie et al. <https://arxiv.org/abs/2512.24880>`_): the streams are
    flattened, RMS-normalized without a weight, and mapped by one learned
    matrix to ``pre`` (which collapses the streams for the sublayer), ``post``
    (which places the sublayer output back into each stream) and ``comb`` (which
    mixes old streams into new). ``comb`` is Sinkhorn-projected towards doubly
    stochastic -- onto the Birkhoff polytope -- which is what restores the
    identity-mapping property plain hyper-connections give up.

    Call it in two halves around the sublayer::

        mixing = site(streams, input_layernorm.weight)
        y = sublayer(mixing.xs)
        streams = site.write_back(streams, y, mixing)

    The split is the fused-kernel boundary: everything in :meth:`__call__` is one
    kernel's worth of work (stream norm, mix logits, Sinkhorn, collapse and the
    block's input layernorm), while :meth:`write_back` is a handful of
    bandwidth-bound dispatches.

    The streams are replicated on every rank, exactly as a single residual is.
    The mapping is a fraction of a percent of a model's arithmetic, and sharding
    its ``(2 + hc_mult) * hc_mult``-column output would buy a collective at
    every site.

    Args:
        hidden_size: Width of one stream.
        hc_mult: Number of parallel residual streams.
        dtype: Storage dtype of the mapping weight, which the checkpoint holds
            at the model's compute dtype while ``base`` and ``scale`` are
            float32.
        sinkhorn_iters: Column/row normalization passes projecting ``comb``
            towards doubly stochastic.
        eps: Positivity floor on ``pre``, ``comb`` and the Sinkhorn divisors.
        rms_norm_eps: Epsilon of the unweighted RMSNorm over the flattened
            streams. Distinct from ``eps``, and easy to conflate: the reference
            uses the model's ``rms_norm_eps`` here.
        name: Weight-name stem. Weights are ``f"{name}_fn"``, ``f"{name}_base"``
            and ``f"{name}_scale"``, and the module's own attribute name is
            omitted from the FQN so flat checkpoint names load directly.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        dtype: DType,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        rms_norm_eps: float = 1e-5,
        name: str = "hc",
    ) -> None:
        super().__init__()
        if sinkhorn_iters < 1:
            raise ValueError(
                f"sinkhorn_iters must be at least 1, got {sinkhorn_iters}."
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.rms_norm_eps = rms_norm_eps

        mix = (2 + hc_mult) * hc_mult
        self.fn = Weight(
            f"{name}_fn",
            dtype,
            [mix, hc_mult * hidden_size],
            device=DeviceRef.CPU(),
        )
        self.base = Weight(
            f"{name}_base", DType.float32, [mix], device=DeviceRef.CPU()
        )
        # One learned scale per mapping output: `pre`, `post`, `comb`.
        self.scale = Weight(
            f"{name}_scale", DType.float32, [3], device=DeviceRef.CPU()
        )
        self._sharding_strategy: ShardingStrategy | None = None

    @property
    def _omit_module_attr_name(self) -> bool:
        return True

    def __call__(
        self, streams: TensorValue, norm_weight: TensorValue | None = None
    ) -> StreamMixing:
        """Computes ``pre`` / ``post`` / ``comb`` and collapses the streams.

        Args:
            streams: ``[total_tokens, hc_mult, hidden_size]``.
            norm_weight: Gamma of the block's input layernorm, applied to the
                collapsed sequence. Passed in rather than applied by the caller
                so that a fused kernel can fold it, which is where vLLM's
                ``mhc_pre`` puts it too.

        Returns:
            The mapping outputs and the sequence the sublayer consumes.

        Raises:
            ValueError: If ``streams`` is not rank 3 with the configured stream
                count and width.
        """
        expected = (self.hc_mult, self.hidden_size)
        if streams.rank != 3 or tuple(streams.shape[-2:]) != expected:
            raise ValueError(
                "hyper-connection streams must be "
                f"[total_tokens, {self.hc_mult}, {self.hidden_size}], got "
                f"{streams.shape}."
            )
        device = streams.device
        tokens = streams.shape[0]
        hc = self.hc_mult

        flat = ops.cast(
            ops.reshape(streams, (tokens, hc * self.hidden_size)),
            DType.float32,
        )
        # Unweighted RMSNorm, float32 throughout.
        normed = flat * ops.rsqrt(
            ops.mean(flat * flat, axis=-1) + self.rms_norm_eps
        )
        mix = normed @ ops.transpose(
            self.fn.cast(DType.float32).to(device), 0, 1
        )

        # The kernel ends on a column pass, so ``comb``'s columns sum to 1 and
        # its rows only approximately. That asymmetry is load-bearing:
        # :meth:`write_back` contracts over the first stream axis, so it is the
        # column sums that make the write-back a convex combination of the
        # incoming streams.
        pre, post, comb_rows = hyper_connection_gates(
            mix,
            self.base.to(device),
            self.scale.to(device),
            hc_mult=hc,
            hc_eps=self.eps,
            hc_sinkhorn_iters=self.sinkhorn_iters,
        )
        comb = ops.reshape(comb_rows, (tokens, hc, hc))

        weighted = ops.unsqueeze(pre, -1) * ops.cast(streams, DType.float32)
        collapsed = ops.cast(
            ops.squeeze(ops.sum(weighted, axis=-2), -2), streams.dtype
        )
        if norm_weight is not None:
            collapsed = ops.rms_norm(
                collapsed,
                norm_weight.cast(collapsed.dtype).to(device),
                self.rms_norm_eps,
                multiply_before_cast=False,
            )
        return StreamMixing(post=post, comb=comb, xs=collapsed)

    def write_back(
        self, streams: TensorValue, y: TensorValue, mixing: StreamMixing
    ) -> TensorValue:
        """Places a sublayer output back into the residual streams.

        ``streams' = post * y + comb^T @ streams``: an outer product in the
        stream axis plus a small batched matmul, both in the model's compute
        dtype.

        Args:
            streams: The streams that went into :meth:`__call__`,
                ``[total_tokens, hc_mult, hidden_size]``.
            y: The sublayer output, ``[total_tokens, hidden_size]``.
            mixing: The matching :meth:`__call__` result.

        Returns:
            The updated streams, same shape and dtype as ``streams``.
        """
        dtype = streams.dtype
        placed = ops.unsqueeze(
            ops.cast(mixing.post, dtype), -1
        ) * ops.unsqueeze(y, -2)
        mixed = ops.transpose(ops.cast(mixing.comb, dtype), -1, -2) @ streams
        return placed + mixed

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the sharding strategy. Only replication is meaningful."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "HyperConnection only supports replicate: the residual streams "
                "are replicated on every rank, and sharding the mapping's "
                f"{(2 + self.hc_mult) * self.hc_mult}-column output would add a "
                "collective at every site."
            )
        self._sharding_strategy = strategy
        self.fn.sharding_strategy = strategy
        self.base.sharding_strategy = strategy
        self.scale.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[HyperConnection]:
        """Creates replicated views of this site, one per device.

        Args:
            devices: Devices to place the replicas on.

        Returns:
            One :class:`HyperConnection` per device.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        shards = []
        for fn, base, scale in zip(
            self.fn.shard(devices),
            self.base.shard(devices),
            self.scale.shard(devices),
            strict=True,
        ):
            replica = HyperConnection(
                hidden_size=self.hidden_size,
                hc_mult=self.hc_mult,
                dtype=self.fn.dtype,
                sinkhorn_iters=self.sinkhorn_iters,
                eps=self.eps,
                rms_norm_eps=self.rms_norm_eps,
            )
            replica.fn = fn
            replica.base = base
            replica.scale = scale
            shards.append(replica)
        return shards


def hyper_connection_site(config: Glm5NextConfig, name: str) -> HyperConnection:
    """Builds one mHC site.

    Args:
        config: The model config.
        name: Weight-name stem, ``"hc_attn"`` or ``"hc_ffn"``, matching the
            checkpoint's flat per-layer tensor names.

    Returns:
        The site, unsharded.
    """
    return HyperConnection(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        # The checkpoint holds `fn` at the compute dtype while `base` and
        # `scale` are float32, and the whole mapping is evaluated in float32
        # regardless -- so this is a storage choice, not a precision one.
        dtype=config.compute_dtype,
        sinkhorn_iters=config.hc_sinkhorn_iters,
        eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
        name=name,
    )
