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
"""Kimi Delta Attention: GLM-5.3-Flash's 34 linear-attention layers.

Linear attention drops the softmax so the sum over past positions reassociates
into a fixed ``[head_dim, head_dim]`` state, and the delta rule replaces what
that state already predicts for ``k`` rather than adding to it. KDA's one
change from Gated DeltaNet is that the forget gate decays the state **per
channel** -- ``alpha_t`` is a ``head_dim``-wide vector per head, so
``S_t = diag(alpha_t) S_{t-1} + k_t delta_t^T``. MAX's
:mod:`~max.nn.legacy.state_space.gated_delta` recurrence takes one decay per head per
token and cannot express that, which is why this is a sibling of Qwen3.5's
``GatedDeltaNet`` rather than a branch inside it.

Per token: ``q/k/v_proj`` 4096 -> 8192 each, concatenated to 24576 and run
through one depthwise causal conv1d of kernel 4 then ``silu``; the forget gate
``f_a_proj`` -> ``f_b_proj`` plus ``dt_bias`` and ``A_log``; the input gate
``b_proj``; the recurrence over a ``[64, 128, 128]`` state; the output gate
``g_a_proj`` -> ``g_b_proj``; a gated RMSNorm over ``head_dim``; ``o_proj``
8192 -> 4096.

Three things here are silently wrong if reordered, and each is pinned by a test
under ``max/tests/integration/architectures/glm5_next/kda/`` rather than by a
comment:

* **The conv channel order is q, k, v.** The weight adapter concatenates the
  three ``[8192, 1, 4]`` checkpoint tensors in that order and this layer
  concatenates its three projections the same way. Permuting either scrambles
  channels without changing a shape.
* **The gated norm normalizes first.** ``o_norm`` is
  ``rms_norm(y) * weight * sigmoid(gate)`` in float32, not
  ``rms_norm(gate_activation(gate) * y)``, which is what MAX's fused
  ``gated_group_rmsnorm`` computes.
* **Tensor parallelism splits heads inside each of q, k and v.** A contiguous
  slice of the 24576-wide concatenated conv would cut across the projection
  boundary: at TP8 the per-rank width is 3072 either way, but ``8192 / 3072``
  is not integral, so ranks 2 and 5 would get a mix of two projections.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    DimLike,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.graph.weight import Segment
from max.nn.legacy.comm import Allreduce
from max.nn.legacy.layer import Module, Shardable
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_conv1d_verify_fwd,
    kda_chunk,
    kda_decode,
    kda_verify_ring,
)
from max.nn.legacy.state_space.kimi_delta import (
    STATE_LAYOUT,
    kda_chunk_predicate,
    kda_chunk_servable,
    kda_gate_mode,
)

from ..model_config import Glm5NextConfig
from .kda import KdaReplayInputs, KdaRing, KdaSublayerInputs

__all__ = [
    "Glm5NextKdaSublayer",
    "KimiDeltaAttention",
]


class KimiDeltaAttention(Module, Shardable):
    """One KDA layer on one device.

    Args:
        hidden_size: Model width.
        num_heads: KDA heads, shared by q, k and v -- there is no GQA here.
        head_dim: Per-head width of q, k, v and both state axes.
        conv_kernel_size: Depthwise causal conv1d kernel size.
        dtype: Dtype of the projections and of this layer's output.
        device: Device this shard computes on.
        rms_norm_eps: Epsilon of the output gated RMSNorm.
        lower_bound: The checkpoint's bounded-gate lower bound, or ``None``
            for the softplus gate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float,
        lower_bound: float | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.dtype = dtype
        self.device = device
        self.rms_norm_eps = rms_norm_eps
        self.lower_bound = lower_bound
        self.gate_mode = kda_gate_mode(lower_bound)
        self._sharding_strategy: ShardingStrategy | None = None

        self.qkv_dim = num_heads * head_dim
        self.conv_dim = 3 * self.qkv_dim

        # Three separate projections, matching the checkpoint. They are
        # concatenated into one conv input at call time; fusing them into a
        # stacked weight would have to undo the adapter's per-tensor layout.
        def head_major_projection() -> Linear:
            return Linear(
                in_dim=hidden_size,
                out_dim=self.qkv_dim,
                dtype=dtype,
                device=device,
                has_bias=False,
            )

        self.q_proj = head_major_projection()
        self.k_proj = head_major_projection()
        self.v_proj = head_major_projection()

        self.conv1d = Weight(
            "conv1d.weight",
            dtype,
            [self.conv_dim, 1, conv_kernel_size],
            device=DeviceRef.CPU(),
        )

        self.f_a_proj = Linear(
            in_dim=hidden_size,
            out_dim=head_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.f_b_proj = Linear(
            in_dim=head_dim,
            out_dim=self.qkv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        # Both are added to the gate pre-activation inside the kernel, so they
        # are float32 here and stay float32 all the way in.
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [self.qkv_dim], device=DeviceRef.CPU()
        )
        self.A_log = Weight(
            "A_log", DType.float32, [num_heads], device=DeviceRef.CPU()
        )

        self.b_proj = Linear(
            in_dim=hidden_size,
            out_dim=num_heads,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.g_a_proj = Linear(
            in_dim=hidden_size,
            out_dim=head_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.g_b_proj = Linear(
            in_dim=head_dim,
            out_dim=self.qkv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        # The reference's RMSNormGated multiplies by the weight directly rather
        # than by `1 + weight`, and holds the whole computation in float32.
        self.o_norm = RMSNorm(
            head_dim,
            dtype=DType.float32,
            eps=rms_norm_eps,
            weight_offset=0.0,
            multiply_before_cast=False,
        )
        self.o_proj = Linear(
            in_dim=self.qkv_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    @property
    def _omit_module_attr_name(self) -> bool:
        """Keeps the weight FQNs at ``self_attn.q_proj.weight``.

        This layer is always held by :class:`Glm5NextKdaSublayer`, which is what
        the decoder layer binds as ``self_attn``. The checkpoint puts the KDA
        weights directly under ``self_attn``, so the intermediate attribute
        name must not appear in between.
        """
        return True

    def state_row_shapes(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Returns this shard's ``(conv, recurrent)`` shapes for one row.

        One row is one layer's state for one request, which is what a
        recurrent cache leaf is declared in. The recurrent shape is laid out
        for :data:`STATE_LAYOUT`.
        """
        return (
            (self.conv_dim, self.conv_kernel_size - 1),
            (self.num_heads, self.head_dim, self.head_dim),
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the layer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the layer by head and propagates that to every weight.

        Args:
            strategy: Must be tensor-parallel; there is no data-parallel path.

        Raises:
            ValueError: If the strategy is not tensor-parallel, or the device
                count does not divide the head count.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "KimiDeltaAttention supports only tensor-parallel sharding, "
                f"got {strategy}"
            )
        num_devices = strategy.num_devices
        if self.num_heads % num_devices:
            raise ValueError(
                f"KimiDeltaAttention num_heads ({self.num_heads}) must be "
                f"divisible by the device count ({num_devices})"
            )
        self._sharding_strategy = strategy

        # `conv1d` is the only weight whose rows span all three projections:
        # the adapter concatenated q, k and v along the channel axis. Splitting
        # each block at its own head boundary keeps the three aligned with the
        # three per-device projections that feed them. A flat split of
        # `conv_dim` has the same per-device width and the wrong channels.
        head_rows = ShardingStrategy.rowwise(num_devices)
        segment = Segment.head_aware(self.num_heads, self.head_dim)
        self.conv1d.sharding_strategy = ShardingStrategy.segmented(
            num_devices, axis=0, segments=(segment, segment, segment)
        )

        # Head-major along their output axis, so a row split is a head split.
        self.q_proj.sharding_strategy = head_rows
        self.k_proj.sharding_strategy = head_rows
        self.v_proj.sharding_strategy = head_rows
        self.f_b_proj.sharding_strategy = head_rows
        self.g_b_proj.sharding_strategy = head_rows
        self.b_proj.sharding_strategy = head_rows
        self.dt_bias.sharding_strategy = head_rows
        self.A_log.sharding_strategy = head_rows

        # `f_a_proj` and `g_a_proj` project down to `head_dim`, which is not a
        # head axis, and the gated norm reduces inside one head. Neither has
        # anything to split.
        replicate = ShardingStrategy.replicate(num_devices)
        self.f_a_proj.sharding_strategy = replicate
        self.g_a_proj.sharding_strategy = replicate
        self.o_norm.sharding_strategy = replicate

        # Row-parallel: each device holds the columns of its own heads and
        # emits a partial sum the caller all-reduces.
        self.o_proj.sharding_strategy = ShardingStrategy.head_aware_columnwise(
            num_devices, self.num_heads, self.head_dim
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[KimiDeltaAttention]:
        """Creates one per-device view of this layer, split by head.

        Args:
            devices: Devices to place the shards on.

        Returns:
            One :class:`KimiDeltaAttention` per device, dimensioned for its own
            head slice.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "KimiDeltaAttention cannot be sharded because no sharding "
                "strategy was provided."
            )
        devices = list(devices)
        num_devices = len(devices)
        q_proj = self.q_proj.shard(devices)
        k_proj = self.k_proj.shard(devices)
        v_proj = self.v_proj.shard(devices)
        conv1d = self.conv1d.shard(devices)
        f_a_proj = self.f_a_proj.shard(devices)
        f_b_proj = self.f_b_proj.shard(devices)
        dt_bias = self.dt_bias.shard(devices)
        a_log = self.A_log.shard(devices)
        b_proj = self.b_proj.shard(devices)
        g_a_proj = self.g_a_proj.shard(devices)
        g_b_proj = self.g_b_proj.shard(devices)
        o_norm = self.o_norm.shard(devices)
        o_proj = self.o_proj.shard(devices)

        shards: list[KimiDeltaAttention] = []
        for i, device in enumerate(devices):
            shard = KimiDeltaAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads // num_devices,
                head_dim=self.head_dim,
                conv_kernel_size=self.conv_kernel_size,
                dtype=self.dtype,
                device=device,
                rms_norm_eps=self.rms_norm_eps,
                lower_bound=self.lower_bound,
            )
            shard.q_proj = q_proj[i]
            shard.k_proj = k_proj[i]
            shard.v_proj = v_proj[i]
            shard.conv1d = conv1d[i]
            shard.f_a_proj = f_a_proj[i]
            shard.f_b_proj = f_b_proj[i]
            shard.dt_bias = dt_bias[i]
            shard.A_log = a_log[i]
            shard.b_proj = b_proj[i]
            shard.g_a_proj = g_a_proj[i]
            shard.g_b_proj = g_b_proj[i]
            shard.o_norm = o_norm[i]
            shard.o_proj = o_proj[i]
            shards.append(shard)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        conv_row_id: TensorValue,
        recurrent_pool: BufferValue,
        recurrent_row_id: TensorValue,
        input_row_offsets: TensorValue,
        replay_capture: list[KdaReplayInputs] | None = None,
        ring: KdaRing | None = None,
        verify_width: TensorValue | None = None,
    ) -> TensorValue:
        """Runs one KDA layer, mutating both state pools in place.

        Both pools are mutable graph inputs the kernels address at row
        ``row_id[batch_item]``, so there is no state graph output and no
        Python-side gather or scatter.

        Args:
            x: ``[total_tokens, hidden_size]``, already normalized by the
                decoder layer.
            conv_pool: The conv leaf's pool, flat over blocks and layers,
                mutated in place.
            conv_row_id: ``[batch_size]`` row of ``conv_pool`` this layer
                reads and writes. Already folded, so the layer never sees the
                layout.
            recurrent_pool: The recurrent leaf's pool, mutated in place.
            recurrent_row_id: Its row for this layer.
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix offsets.
            replay_capture: When given, this call's :class:`KdaReplayInputs`
                are appended to it so a speculative rollback can re-run the
                state kernels over a shorter prefix.
            ring: The speculative verify's ring, set with ``verify_width``.
                A verify records each token's state update into it instead of
                writing ``recurrent_pool``, and the caller folds the accepted
                records in before the pool is read again.
            verify_width: The speculative graph's
                :func:`~max.nn.legacy.state_space.verify_width_operand`. A launch at
                width zero lands both pools on the forward, and a wider one
                leaves them for the rollback.

        Returns:
            ``[total_tokens, hidden_size]``, a partial sum over the head axis
            when this layer is sharded.
        """
        device = x.device
        heads, head_dim = self.num_heads, self.head_dim
        head_shape = [-1, heads, head_dim]

        # Concatenated q, k, v in that order: the adapter's conv channel
        # layout, and the order the shard's conv weight was sliced for.
        qkv = ops.concat(
            [self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1
        ).cast(DType.float32)

        conv_weight = ops.reshape(
            self.conv1d.to(device).cast(DType.float32),
            [self.conv_dim, self.conv_kernel_size],
        )
        row_offsets_uint32 = input_row_offsets.cast(DType.uint32)
        qkv_window: TensorValue | None = None
        if verify_width is None:
            conv_out = gated_delta_conv1d_fwd(
                qkv_input_ragged=qkv,
                conv_weight=conv_weight,
                conv_state=conv_pool,
                slot_idx=conv_row_id.cast(DType.uint32),
                input_row_offsets=row_offsets_uint32,
            )
        else:
            conv_out, qkv_window = gated_delta_conv1d_verify_fwd(
                qkv_input_ragged=qkv,
                conv_weight=conv_weight,
                conv_state=conv_pool,
                slot_idx=conv_row_id.cast(DType.uint32),
                input_row_offsets=row_offsets_uint32,
                verify_width=verify_width,
                rollback=False,
            )
        conv_out = ops.silu(conv_out)
        q, k, v = (
            ops.reshape(part, head_shape)
            for part in ops.split(conv_out, [self.qkv_dim] * 3, axis=-1)
        )

        # The kernel adds `dt_bias`, applies `exp(A_log)` and the bounded
        # sigmoid, and sigmoids `beta_logits`; all of that is folded in by
        # `gate_mode` and `beta_mode`, so what goes in is the raw
        # pre-activation. Unlike Gated DeltaNet's, it is per channel.
        raw_gate = ops.reshape(
            self.f_b_proj(self.f_a_proj(x)).cast(DType.float32), head_shape
        )
        beta_logits = self.b_proj(x).cast(DType.float32)

        if replay_capture is not None:
            replay_capture.append(
                KdaReplayInputs(
                    qkv=qkv,
                    conv_weight=conv_weight,
                    raw_gate=raw_gate,
                    beta_logits=beta_logits,
                    qkv_window=qkv_window,
                )
            )

        a_log = self.A_log.to(device)
        dt_bias = ops.reshape(self.dt_bias.to(device), [heads, head_dim])
        state_indices = recurrent_row_id.cast(DType.int32)
        cu_seqlens = input_row_offsets.cast(DType.int32)

        # Both ops take the same tensors and differ only in how they walk the
        # recurrence. On gfx950 the chunk-eligible configuration requires a
        # float32 pool, so both ops write float32 state there. On NVIDIA q/k/v
        # and the output stay float32 into the chunk op too. The gfx950 chunk
        # kernels take only bf16 q/k/v and output, so that branch rounds at
        # the op boundary and nowhere else; `test_kda_chunk_prefill.py` grades
        # it against float32 decode. The output returns to float32 because the
        # next stage is a float32 gated norm.
        out_shape: list[DimLike] = [x.shape[0], heads, head_dim]

        io_dtype_chunked = (
            DType.bfloat16 if accelerator_api() == "hip" else DType.float32
        )

        def _recurrence(
            chunked: bool,
            qkv_io: tuple[TensorValue, TensorValue, TensorValue] | None = None,
        ) -> TensorValue:
            # Decode pins its `fma`s so a verify and fold over the same tokens
            # land on its state bit for bit, which the MTP rollback tests
            # compare. Only AMD pays for it; NVIDIA always pins.
            op = (
                kda_chunk
                if chunked
                else functools.partial(kda_decode, pin_fma=True)
            )
            io_dtype = io_dtype_chunked if chunked else DType.float32
            q_io, k_io, v_io = qkv_io or (
                q.cast(io_dtype),
                k.cast(io_dtype),
                v.cast(io_dtype),
            )
            out = op(
                q_io,
                k_io,
                v_io,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                recurrent_pool,
                state_indices,
                output_dtype=io_dtype,
                gate_mode=self.gate_mode,
                beta_mode="logits",
                state_layout=STATE_LAYOUT,
            ).cast(DType.float32)
            return ops.rebind(out, out_shape, "kda recurrence output rebind")

        def _verify_recurrence() -> TensorValue:
            assert verify_width is not None
            assert ring is not None, "a verify needs a ring"
            out = kda_verify_ring(
                q,
                k,
                v,
                raw_gate,
                beta_logits,
                a_log,
                dt_bias,
                cu_seqlens,
                recurrent_pool,
                state_indices,
                ring.pool,
                ring.row_id,
                verify_width,
                output_dtype=DType.float32,
                gate_mode=self.gate_mode,
                beta_mode="logits",
                state_layout=STATE_LAYOUT,
            )
            return ops.rebind(out, out_shape, "kda verify output rebind")

        chunk_servable = kda_chunk_servable(
            head_dim, self.gate_mode, recurrent_pool.dtype
        )
        if verify_width is None and chunk_servable:
            # A decode batch (one token per sequence) must stay on float32
            # `kda_decode`, and on NVIDIA `kda_chunk` also reads `cu_seqlens`
            # back to the host to build its chunk map. `ops.cond` branches are
            # separate regions, so a decode batch never launches the prefill
            # side, and the device graph captured for decode never contains
            # it. A mixed prefill+decode batch takes the chunk side whole,
            # decode rows included.
            total_tokens = ops.shape_to_tensor(x.shape)[0]
            batch_size = ops.shape_to_tensor(input_row_offsets.shape)[0] - 1
            core_out = ops.cond(
                kda_chunk_predicate(total_tokens, batch_size, heads, head_dim),
                [TensorType(DType.float32, out_shape, device)],
                lambda: _recurrence(chunked=True),
                lambda: _recurrence(chunked=False),
            )[0]
        elif verify_width is None:
            core_out = _recurrence(chunked=False)
        elif chunk_servable:
            # A draft-free prefill has nothing to roll back, so it takes the
            # forward's chunked kernel. A verify stays on the sequential
            # recurrence, whose arithmetic the ring records and the fold
            # replays.
            total_tokens = ops.shape_to_tensor(x.shape)[0]
            batch_size = ops.shape_to_tensor(input_row_offsets.shape)[0] - 1
            no_drafts = ops.equal(ops.shape_to_tensor(verify_width.shape)[0], 0)
            # TODO(KERN-3879): the gfx950 bf16 casts are made outside the
            # cond because, made inside this branch, mo-hoist-param-exprs
            # leaves the first cast's result on the outer token dim and the
            # graph fails to verify. Move them back once the pass is fixed.
            qkv_io = (
                q.cast(io_dtype_chunked),
                k.cast(io_dtype_chunked),
                v.cast(io_dtype_chunked),
            )
            core_out = ops.cond(
                ops.logical_and(
                    no_drafts,
                    kda_chunk_predicate(
                        total_tokens, batch_size, heads, head_dim
                    ),
                ),
                [TensorType(DType.float32, out_shape, device)],
                lambda: _recurrence(chunked=True, qkv_io=qkv_io),
                _verify_recurrence,
            )[0]
        else:
            core_out = _verify_recurrence()

        # Normalize, scale by the learned weight, and only then gate. Gating
        # first is a different function of the same tensors.
        gate = ops.reshape(
            self.g_b_proj(self.g_a_proj(x)).cast(DType.float32), head_shape
        )
        normed = self.o_norm(core_out) * ops.sigmoid(gate)
        return self.o_proj(
            ops.reshape(normed.cast(self.dtype), [-1, self.qkv_dim])
        )


class Glm5NextKdaSublayer(Module):
    """The KDA sublayer as the decoder layer calls it, across all devices."""

    replay_capture: list[list[KdaReplayInputs]] | None = None
    """Per-device sink for this layer's state-kernel inputs, or ``None``.

    Set by the speculative graph, which re-runs those kernels over the accepted
    prefix after the verify pass. ``None`` in the base graph.
    """

    verify_width: TensorValue | None = None
    """The speculative graph's verify width operand, or ``None``.

    Set alongside :attr:`replay_capture`. See :class:`KimiDeltaAttention`.
    """

    def __init__(self, config: Glm5NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        devices = config.devices
        self.kda = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.linear_num_heads,
            head_dim=config.linear_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            dtype=config.compute_dtype,
            device=devices[0],
            rms_norm_eps=config.rms_norm_eps,
            lower_bound=config.linear_lower_bound,
        )
        self.kda.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.shards = self.kda.shard(devices)
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self, xs: list[TensorValue], inputs: KdaSublayerInputs
    ) -> list[TensorValue]:
        """Runs the layer on every device and all-reduces the result.

        Args:
            xs: ``[total_tokens, hidden_size]`` per device, normalized.
            inputs: This layer's per-step pools, row ids and offsets.

        Returns:
            ``[total_tokens, hidden_size]`` per device.
        """
        # Only the verify width keeps a captured call off `kda_chunk` and its
        # own state writes, which the rollback would then advance again.
        assert self.replay_capture is None or self.verify_width is not None
        return self.allreduce(
            [
                shard(
                    xs[i],
                    conv_pool=inputs.conv_pools[i],
                    conv_row_id=inputs.conv_row_ids[i],
                    recurrent_pool=inputs.recurrent_pools[i],
                    recurrent_row_id=inputs.recurrent_row_ids[i],
                    input_row_offsets=inputs.input_row_offsets[i],
                    replay_capture=(
                        None
                        if self.replay_capture is None
                        else self.replay_capture[i]
                    ),
                    ring=(
                        KdaRing(inputs.ring_pools[i], inputs.ring_row_ids[i])
                        if inputs.ring_pools
                        else None
                    ),
                    verify_width=self.verify_width,
                )
                for i, shard in enumerate(self.shards)
            ],
            inputs.signal_buffers,
        )
