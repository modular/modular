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
"""Resolves the decode dispatch of any attention family through one entry.

A captured device graph freezes the launch its decode attention kernels pick
from host scalars, so a serving engine keys its captures on that launch. Each
attention family states its launch here as an `AttnDispatchPolicy`, and
`attn_dispatch_resolve` selects the policy by the descriptor's `kind`. An
engine therefore needs no code per attention family.
"""

from max.gpu.host import DeviceContext

from nn.attention.gpu.mha_decode_partition_heuristic import (
    amd_mla_decode_dispatch_scalars,
    mha_decoding_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_split_kv_decode_num_partitions,
)
from nn.attention.gpu.nvidia.sm100.fa4_splitk_combine import P_MAX
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLA_SPARSE_KV_FORM_PLAIN,
    compute_mla_dispatch_key_runtime,
)

comptime ATTN_DISPATCH_UNSET = -1
"""Value of a descriptor field its attention family does not set.

Zero is a valid window and a valid stride, so it cannot mean "unset".
"""

comptime ATTN_DISPATCH_MAX_SCALARS = 8
"""Most scalars a key or a metadata record holds."""

comptime ATTN_KV_DTYPE_OTHER = 0
comptime ATTN_KV_DTYPE_FP8 = 1

comptime ATTN_FLAG_SPLIT_HEAD_DIMS = 1 << 0
"""The cache's key is wider than its value."""
comptime ATTN_FLAG_FIXED_TOPK = 1 << 1
"""A sparse decode passes neither per-token top-k lengths nor a sink."""
comptime ATTN_FLAG_PHYSICAL_INDICES = 1 << 2
"""Sparse rows are physical."""
comptime ATTN_FLAG_ROW_INVARIANT_VERIFY = 1 << 3
"""The launch is the row-invariant pin's own verify."""

comptime _ATTN_FLAGS_KNOWN = (
    ATTN_FLAG_SPLIT_HEAD_DIMS
    | ATTN_FLAG_FIXED_TOPK
    | ATTN_FLAG_PHYSICAL_INDICES
    | ATTN_FLAG_ROW_INVARIANT_VERIFY
)


@fieldwise_init
struct AttnDispatchDesc(Copyable, Movable):
    """The facts of one KV cache that a decode dispatch policy reads.

    The fields are the union over every attention family; a policy ignores
    those it does not read. What the device knows (its API and its
    multiprocessor count) is not here: a policy reads it from the device
    context, the source the launch uses.
    """

    var kind: String
    """The attention family: `"mha"`, `"mla"`, `"msa"`, `"msa_index"`."""
    var num_q_heads: Int
    """Query heads on one device."""
    var num_kv_heads: Int
    """KV heads on one device."""
    var head_dim: Int
    """The cache row width the dispatch assumes."""
    var page_size: Int
    var window_size: Int
    """The attention window, unset when there is none."""
    var sparse_indices_stride: Int
    """The sparse index row stride, unset for a dense cache."""
    var kv_dtype: Int
    """An `ATTN_KV_DTYPE_*` value."""
    var kv_form: Int
    """The sparse MLA cache form (`MLA_SPARSE_KV_FORM_*`)."""
    var flags: Int
    """A bitset of `ATTN_FLAG_*`."""

    def has_flag(self, flag: Int) -> Bool:
        return (self.flags & flag) != 0


struct AttnDispatch(Copyable, Movable):
    """What one decode dispatch resolves to."""

    var key: List[Int]
    """The part of the capture identity the kernels pick: the split count,
    then the launch variant, each present only when the family picks one. A
    caller adds the batch size and the query width."""
    var metadata: List[Int]
    """The scalars staged into the dispatch record, empty when no kernel
    reads one."""

    def __init__(out self, *, var key: List[Int], var metadata: List[Int]):
        self.key = key^
        self.metadata = metadata^


trait AttnDispatchPolicy:
    """States the decode launch of one attention family.

    `resolve` must call the functions the family's launch calls, so that the
    capture key cannot drift from the kernel.
    """

    @staticmethod
    def kind() -> StaticString:
        """Returns the descriptor `kind` this policy resolves."""
        ...

    @staticmethod
    def resolve(
        ctx: DeviceContext,
        desc: AttnDispatchDesc,
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
    ) raises -> AttnDispatch:
        """Returns the dispatch of one decode step.

        Raises:
            If the descriptor leaves a field this policy needs unset, or
            the kernels do not support the shape.
        """
        ...


def _required(
    value: Int, field: StaticString, desc: AttnDispatchDesc
) raises -> Int:
    if value < 0:
        raise Error(
            "attention dispatch kind '"
            + desc.kind
            + "' requires the descriptor field '"
            + String(field)
            + "'"
        )
    return value


def _required_count(
    value: Int, field: StaticString, desc: AttnDispatchDesc
) raises -> Int:
    # A count of zero is a descriptor a C caller zero-initialized, not a
    # cache: it would resolve a key, and a wrong one.
    if _required(value, field, desc) < 1:
        raise Error(
            "attention dispatch kind '"
            + desc.kind
            + "' requires a positive descriptor field '"
            + String(field)
            + "'"
        )
    return value


def _or_zero(value: Int) -> Int:
    return value if value >= 0 else 0


def mha_split_kv_decode_num_partitions_runtime(
    batch_size: Int,
    num_keys: Int,
    kv_heads_per_device: Int,
    sliding_window: Int,
    sm_count: Int,
) raises -> Int:
    """Returns `mha_split_kv_decode_num_partitions` for a runtime SM count.

    Args:
        batch_size: Requests in the batch.
        num_keys: The batch's largest cache length.
        kv_heads_per_device: KV heads on one device.
        sliding_window: The attention window, 0 for none.
        sm_count: The device's multiprocessor count.

    Returns:
        The decode partition count the split-cache kernel launches.
    """
    if sm_count < 1:
        raise Error("sm_count must be positive.")
    # The policy's ladder tops out at the SM count, a compile-time parameter,
    # so it is instantiated for every count up to the combine's `P_MAX`, which
    # also caps the count it returns. A MIG slice gets its own count's ladder.
    # TODO(DISTINF-639): a device with more than `P_MAX` SMs takes the `P_MAX`
    # instantiation, whose wave bound counts `P_MAX` SMs and so may split less
    # than the device could; give the policy a runtime SM count if one ships.
    var policy_sm_count = min(sm_count, P_MAX)
    comptime for ladder_sm_count in range(1, P_MAX + 1):
        if policy_sm_count == ladder_sm_count:
            return mha_split_kv_decode_num_partitions[ladder_sm_count](
                batch_size, num_keys, kv_heads_per_device, sliding_window
            )
    raise Error("unreachable: every SM count up to P_MAX has a policy")


struct MHADispatchPolicy(AttnDispatchPolicy):
    """Multi-head attention: the key is the split-K partition count."""

    @staticmethod
    def kind() -> StaticString:
        return "mha"

    @staticmethod
    def resolve(
        ctx: DeviceContext,
        desc: AttnDispatchDesc,
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
    ) raises -> AttnDispatch:
        var kv_heads = _required_count(desc.num_kv_heads, "num_kv_heads", desc)
        var num_partitions: Int
        if desc.has_flag(ATTN_FLAG_SPLIT_HEAD_DIMS):
            num_partitions = mha_split_kv_decode_num_partitions_runtime(
                batch_size,
                max_cache_valid_length,
                kv_heads,
                _or_zero(desc.window_size),
                ctx.device().multiprocessor_count(),
            )
        else:
            num_partitions = mha_decoding_num_partitions(
                batch_size, max_cache_valid_length, kv_heads, ctx
            )
        return AttnDispatch(
            key=[num_partitions],
            metadata=[
                batch_size,
                q_max_seq_len,
                num_partitions,
                max_cache_valid_length,
            ],
        )


def mla_decode_dispatch_scalars(
    ctx: DeviceContext,
    batch_size: Int,
    max_cache_valid_length: Int,
    q_max_seq_len: Int,
    num_heads: Int,
    is_fp8_kv: Bool,
    sparse_indices_stride: Int,
    kv_form: Int,
    page_size: Int,
    fixed_topk: Bool,
    physical_indices: Bool,
    depth: Int,
    row_invariant_verify: Bool,
) raises -> Tuple[Int, Int, Int, Int]:
    """Returns the MLA decode dispatch scalars of the device's own vendor.

    Args:
        ctx: The device the decode runs on.
        batch_size: Requests in the decode batch.
        max_cache_valid_length: The batch's largest valid cache length.
        q_max_seq_len: Query rows per request.
        num_heads: Query heads on one device.
        is_fp8_kv: Whether the cache stores FP8.
        sparse_indices_stride: The sparse index row stride, 0 for dense.
        kv_form: The sparse cache form (`MLA_SPARSE_KV_FORM_*`).
        page_size: The cache page size.
        fixed_topk: Whether a sparse decode passes neither per-token top-k
            lengths nor a sink.
        physical_indices: Whether sparse rows are physical.
        depth: The cache row width.
        row_invariant_verify: Whether the launch is the row-invariant pin's
            own verify.

    Returns:
        The batch size, the query width, the partition count and the kernel
        variant. The variant separates launches that share a partition
        count; only the SM100 sparse decode has more than one.

    Raises:
        If the device is neither HIP nor CUDA, or the kernels do not
        support the shape.
    """
    var api = ctx.api()
    if api == "hip":
        var amd = amd_mla_decode_dispatch_scalars(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            ctx,
            sparse_indices_stride=sparse_indices_stride,
        )
        return (amd[0], amd[1], amd[2], 0)
    if api == "cuda":
        return compute_mla_dispatch_key_runtime(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            sparse_indices_stride,
            ctx.device().multiprocessor_count(),
            kv_form=kv_form,
            page_size=page_size,
            fixed_topk=fixed_topk,
            physical_indices=physical_indices,
            depth=depth,
            row_invariant_verify=row_invariant_verify,
        )
    raise Error(
        "MLA decode dispatch supports only HIP (AMD) and CUDA (NVIDIA) devices."
    )


struct MLADispatchPolicy(AttnDispatchPolicy):
    """Multi-latent attention: the key is the partition count and the
    kernel variant."""

    @staticmethod
    def kind() -> StaticString:
        return "mla"

    @staticmethod
    def resolve(
        ctx: DeviceContext,
        desc: AttnDispatchDesc,
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
    ) raises -> AttnDispatch:
        var num_heads = _required_count(desc.num_q_heads, "num_q_heads", desc)
        var kv_dtype = _required(desc.kv_dtype, "kv_dtype", desc)
        if kv_dtype != ATTN_KV_DTYPE_FP8 and kv_dtype != ATTN_KV_DTYPE_OTHER:
            raise Error("unknown attention dispatch kv_dtype")
        # Only the CUDA launch reads the page size.
        var page_size = 0
        if ctx.api() == "cuda":
            page_size = _required_count(desc.page_size, "page_size", desc)
        var kv_form = MLA_SPARSE_KV_FORM_PLAIN
        if desc.kv_form >= 0:
            kv_form = desc.kv_form
        var scalars = mla_decode_dispatch_scalars(
            ctx,
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            kv_dtype == ATTN_KV_DTYPE_FP8,
            _or_zero(desc.sparse_indices_stride),
            kv_form,
            page_size,
            desc.has_flag(ATTN_FLAG_FIXED_TOPK),
            desc.has_flag(ATTN_FLAG_PHYSICAL_INDICES),
            _or_zero(desc.head_dim),
            desc.has_flag(ATTN_FLAG_ROW_INVARIANT_VERIFY),
        )
        return AttnDispatch(
            key=[scalars[2], scalars[3]],
            metadata=[scalars[0], scalars[1], scalars[2]],
        )


def _resolve_msa(batch_size: Int, max_cache_valid_length: Int) -> AttnDispatch:
    # No MSA kernel picks its launch from the cache length; the record holds
    # the batch size and the length for the graph's own use.
    return AttnDispatch(key=[], metadata=[batch_size, max_cache_valid_length])


struct MSADispatchPolicy(AttnDispatchPolicy):
    """MiniMax sparse attention: one launch for every cache length."""

    @staticmethod
    def kind() -> StaticString:
        return "msa"

    @staticmethod
    def resolve(
        ctx: DeviceContext,
        desc: AttnDispatchDesc,
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
    ) raises -> AttnDispatch:
        return _resolve_msa(batch_size, max_cache_valid_length)


struct MSAIndexDispatchPolicy(AttnDispatchPolicy):
    """The MSA indexer's index-K cache: one launch for every cache length."""

    @staticmethod
    def kind() -> StaticString:
        return "msa_index"

    @staticmethod
    def resolve(
        ctx: DeviceContext,
        desc: AttnDispatchDesc,
        batch_size: Int,
        q_max_seq_len: Int,
        max_cache_valid_length: Int,
    ) raises -> AttnDispatch:
        return _resolve_msa(batch_size, max_cache_valid_length)


def _resolve_with[
    Policy: AttnDispatchPolicy
](
    ctx: DeviceContext,
    desc: AttnDispatchDesc,
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
) raises -> AttnDispatch:
    var resolved = Policy.resolve(
        ctx, desc, batch_size, q_max_seq_len, max_cache_valid_length
    )
    if (
        len(resolved.key) > ATTN_DISPATCH_MAX_SCALARS
        or len(resolved.metadata) > ATTN_DISPATCH_MAX_SCALARS
    ):
        raise Error(
            "an attention dispatch holds at most "
            + String(ATTN_DISPATCH_MAX_SCALARS)
            + " scalars"
        )
    return resolved^


def attn_dispatch_resolve(
    ctx: DeviceContext,
    desc: AttnDispatchDesc,
    batch_size: Int,
    q_max_seq_len: Int,
    max_cache_valid_length: Int,
) raises -> AttnDispatch:
    """Resolves the decode dispatch of one KV cache at one step shape.

    A caller that captures at probed cache lengths finds where the key
    changes between two of them by bisecting this function. Nothing promises
    that a key holds over one range of cache lengths only.

    Args:
        ctx: The device the decode runs on.
        desc: The cache's dispatch descriptor.
        batch_size: Requests in the decode batch.
        q_max_seq_len: Query rows per request: 1 for plain decode.
        max_cache_valid_length: The batch's largest valid cache length.

    Returns:
        The capture key and the dispatch record scalars.

    Raises:
        If the kind is unknown, the shape is invalid, a field the kind's
        policy needs is unset, or the kernels do not support the shape.
    """
    if batch_size < 1:
        raise Error("batch_size must be positive.")
    if q_max_seq_len < 1:
        raise Error("q_max_seq_len must be positive.")
    if max_cache_valid_length < 0:
        raise Error("max_cache_valid_length must be non-negative.")
    if (desc.flags & ~_ATTN_FLAGS_KNOWN) != 0:
        raise Error("unknown attention dispatch flags")

    # A new attention family adds its policy here and nothing else.
    if desc.kind == MHADispatchPolicy.kind():
        return _resolve_with[MHADispatchPolicy](
            ctx, desc, batch_size, q_max_seq_len, max_cache_valid_length
        )
    if desc.kind == MLADispatchPolicy.kind():
        return _resolve_with[MLADispatchPolicy](
            ctx, desc, batch_size, q_max_seq_len, max_cache_valid_length
        )
    if desc.kind == MSADispatchPolicy.kind():
        return _resolve_with[MSADispatchPolicy](
            ctx, desc, batch_size, q_max_seq_len, max_cache_valid_length
        )
    if desc.kind == MSAIndexDispatchPolicy.kind():
        return _resolve_with[MSAIndexDispatchPolicy](
            ctx, desc, batch_size, q_max_seq_len, max_cache_valid_length
        )
    raise Error("unknown attention dispatch kind '" + desc.kind + "'")
