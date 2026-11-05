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
"""Single source of truth for the unified spec-decode graph input ordering."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Protocol, overload

from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
)
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
)
from max.nn.legacy.comm import Signals
from max.nn.legacy.graph_groups import (
    BatchContext,
    DecoderGraphInputs,
    SpecDecode,
    SpecDecodeOutputs,
    TokenBitmask,
    TPSignals,
    VisionMerge,
)
from max.tensor import Tensor
from typing_extensions import TypeVar

from ._tensor_compat import as_graph_values, as_tensors

__all__ = [
    "SpecDecodeGraphInputs",
    "SpecDecodeGraphSignature",
    "SpecDecodeInputTypeSpec",
    "build_spec_decode_input_types",
    "declared_spec_decode_inputs",
    "decode_spec_decode_input_values",
    "drop_mixed_step_split",
    "spec_decode_input_tree",
    "spec_decode_outputs",
    "spec_decode_tail_groups",
    "without_mixed_step_split",
]


@dataclass(frozen=True)
class SpecDecodeInputTypeSpec:
    """Structural variation points of a unified spec-decode graph signature."""

    devices: Sequence[DeviceRef]
    """The graph's devices, in signature order. The signature's shape depends
    on ``len(devices)`` for vision, signal buffers and batch context lengths,
    so it is a property of the signature like every other field here."""

    distributed: bool
    data_parallel_degree: int = 1
    enable_vision: bool = False
    vision_hidden_size: int | None = None
    include_in_thinking_phase: bool = False
    enable_structured_output: bool = False
    include_signal_buffers: bool = False
    """Declare signal-buffer inputs even when not distributed, for targets
    whose layers unconditionally use collectives (e.g. Gemma4's
    VocabParallelEmbedding on a single device). Implied by ``distributed``."""
    include_skippable_draft: bool = False
    """Declare the ``draft_slot_ids`` / ``draft_block_offsets`` inputs, which
    let a step run its drafter over a runtime-chosen number of rows. The count
    is zero when the verify-width schedule says the next step will verify
    nothing, so the draft forwards and the draft ``lm_head`` are skipped
    without a second graph.

    Set by the driver, never by an architecture: it declares the inputs when
    :func:`~max.pipelines.speculative.spec_width_policy.declares_skippable_draft`
    says the schedule can skip, and the pipeline binds them by the same rule.
    Only a draft that declares ``supports_zero_draft_rows`` reads them. The
    sequential driver skips on one device, tensor-parallel or data-parallel,
    and the block driver on one device, for argmax and sampled proposals
    alike. A sampled step pads the skipped rows' distributions with zeros."""
    trace_owns_signal_buffers: bool = False
    """Declare no signal-buffer inputs, whatever the flags above say.

    A graph traced by :meth:`~max.nn.Module.compile` appends its
    own after every declared input, so declaring them here too would leave
    two sets."""
    signal_buffer_index: int | None = None
    """Where the trace inserts its signal buffers, under
    ``trace_owns_signal_buffers``, for an engine that binds them there.
    ``None`` appends them."""
    enable_sampled_draft_proposal: bool = False
    """Declare the ``draft_probs_full`` input: the distribution the draft
    sampled its token from, which the acceptance test's residual subtracts and
    reads ``q`` out of. Requires ``vocab_size``. Set by the MiniMax-M3 unified
    pipelines and, through :class:`SequentialDriver`, by any driver built with
    ``draft_proposal="sampled"``."""
    vocab_size: int | None = None
    """Static vocabulary size, required by ``enable_sampled_draft_proposal``."""
    draft_probs_replicas: int = 0
    """Declare ``draft_probs_full`` as one tensor per data-parallel replica,
    each on the replica's leader device and holding that replica's rows, in
    place of the single device-0 tensor. Requires
    ``enable_sampled_draft_proposal`` and a distributed spec."""


def draft_probs_rows_name(replica: int) -> str:
    """The symbolic row count of ``replica``'s proposal distributions."""
    return f"dspark_probs_rows_replica_{replica}"


def replica_leader_devices(spec: SpecDecodeInputTypeSpec) -> list[DeviceRef]:
    """Each data-parallel replica's first device, in replica order."""
    degree = spec.data_parallel_degree
    if not (spec.draft_probs_replicas and spec.distributed and degree > 1):
        raise ValueError(
            "per-replica draft distributions need a distributed spec with"
            " data_parallel_degree > 1"
        )
    if spec.draft_probs_replicas != degree or len(spec.devices) % degree:
        raise ValueError(
            f"draft_probs_replicas={spec.draft_probs_replicas} must equal"
            f" data_parallel_degree={degree}, which divides"
            f" {len(spec.devices)} devices"
        )
    per_replica = len(spec.devices) // degree
    return [spec.devices[r * per_replica] for r in range(degree)]


def _declares_signal_buffers(spec: SpecDecodeInputTypeSpec) -> bool:
    return (
        spec.distributed or spec.include_signal_buffers
    ) and not spec.trace_owns_signal_buffers


_T = TypeVar("_T", TensorValue, Tensor, TensorType, Buffer, default=TensorValue)
"""The inputs' value type: a graph value, an experimental ``Tensor``, a
declared type, or a step's buffer."""

_B = TypeVar("_B", BufferValue, Tensor, BufferType, Buffer, default=BufferValue)
"""The buffer type, alongside :obj:`_T`."""

_V = TypeVar(
    "_V",
    Value[Any],
    Tensor,
    TensorType | BufferType,
    Buffer,
    default=Value[Any],
)
"""The type of an input the decode does not interpret, alongside :obj:`_T`."""


@tree.dataclass(frozen=True)
class SpecDecodeGraphInputs(Generic[_T, _B, _V]):
    """A unified spec-decode graph's inputs, named rather than positional.

    The fields hold the types a signature declares, the graph's values, or a
    step's buffers, and :func:`spec_decode_input_tree` orders them. A field is
    ``None`` (or empty) exactly when the spec flag that gates its group is
    unset.
    """

    tokens: _T
    input_row_offsets: _T
    return_n_logits: _T
    kv_tree: KVCacheInputs[_T, _B]
    draft_tokens: _T
    seed: _T
    temperature: _T
    top_k: _T
    max_k: _T
    top_p: _T
    min_top_p: _T

    host_input_row_offsets: _T | None = None
    data_parallel_splits: _T | None = None
    draft_probs_full: _T | None = None
    draft_probs_replicas: list[_T] | None = None
    in_thinking_phase: _T | None = None
    draft_slot_ids: _T | None = None
    """Which drafter rows this step computes, as the tensor's extent:
    ``arange(batch_size * rows_per_seq)`` to draft, empty to skip. ``None``
    unless the spec set ``include_skippable_draft``."""
    draft_block_offsets: _T | None = None
    """The draft forward's ragged row offsets: ``[0, K, 2K, ...]`` to draft,
    all zeros to skip. ``None`` unless the spec set
    ``include_skippable_draft``."""
    pinned_bitmask: _T | None = None
    wait_payload: _B | None = None
    device_bitmask_scratch: _B | None = None

    vision_embeddings: list[_T] = field(default_factory=list)
    vision_scatter_indices: list[_T] = field(default_factory=list)
    signal_buffers: list[_B] = field(default_factory=list)
    batch_context_lengths: list[_T] = field(default_factory=list)
    ep_inputs: list[_V] = field(default_factory=list)

    leading: list[_V] = field(default_factory=list)
    """Inputs ahead of ``tokens``, for a model that declares its own group
    first. Empty unless the signature declared one."""

    trailing: list[_V] = field(default_factory=list)
    """Inputs past the canonical tail, for a model that appends its own
    group. Empty unless the decode allowed trailing."""

    def kv(self, *path: str) -> list[KVCacheInputsPerDevice[_T, _B]]:
        """Returns one KV leaf's per-device inputs, addressed by name.

        Raises:
            ValueError: If ``path`` names a missing child, stops on an
                interior node, or descends through a leaf.
        """
        node: Any = self.kv_tree
        for i, name in enumerate(path):
            if not isinstance(node, Mapping):
                raise ValueError(
                    f"KV path {'/'.join(path)!r} descends through a leaf at"
                    f" {'/'.join(path[:i]) or '<root>'!r}"
                )
            if name not in node:
                raise ValueError(
                    f"KV path {'/'.join(path)!r} has no child {name!r};"
                    f" available: {sorted(node)}"
                )
            node = node[name]
        if isinstance(node, Mapping):
            raise ValueError(
                f"KV path {'/'.join(path)!r} names an interior node, not a leaf"
            )
        return list(tree.leaves(node, leaf=KVCacheInputsPerDevice))

    @property
    def host_offsets(self) -> _T:
        """:attr:`host_input_row_offsets`, for a model that declares it.

        Declared by every ``distributed=True`` spec and no other.
        """
        assert self.host_input_row_offsets is not None
        return self.host_input_row_offsets

    @property
    def dp_splits(self) -> _T:
        """:attr:`data_parallel_splits`, for a model that declares it."""
        assert self.data_parallel_splits is not None
        return self.data_parallel_splits

    @property
    def thinking_phase(self) -> _T:
        """:attr:`in_thinking_phase`, for a model that declares it.

        Declared whenever the spec sets ``include_in_thinking_phase``.
        """
        assert self.in_thinking_phase is not None
        return self.in_thinking_phase


def declared_spec_decode_inputs(
    spec: SpecDecodeInputTypeSpec,
    *,
    kv_params: KVCacheParamInterface,
    ep_input_types: Sequence[TensorType | BufferType] = (),
    trailing_input_types: Sequence[TensorType | BufferType] = (),
) -> SpecDecodeGraphInputs[TensorType, BufferType, TensorType | BufferType]:
    """Returns the types a graph built from ``spec`` declares, by field.

    :func:`spec_decode_input_tree` orders them.

    Args:
        spec: The spec the signature is built from.
        kv_params: The unified ``{"target", "draft"}`` KV tree. Its inputs,
            the target leaf's then the draft leaf's, carry both caches' blocks
            and dispatch metadata.
        ep_input_types: The target's expert-parallel inputs.
        trailing_input_types: The inputs the target's call takes past the
            canonical tail.

    Raises:
        ValueError: If ``spec`` samples its drafts with no ``vocab_size``.
    """
    devices = spec.devices
    device = devices[0]
    cpu = DeviceRef.CPU()

    def per_row(dtype: DType) -> TensorType:
        return TensorType(dtype, shape=["batch_size"], device=device)

    draft_probs_full: TensorType | None = None
    draft_probs_replicas: list[TensorType] | None = None
    if spec.enable_sampled_draft_proposal:
        if spec.vocab_size is None:
            raise ValueError(
                "vocab_size is required when enable_sampled_draft_proposal is"
                " set"
            )
        if spec.draft_probs_replicas:
            draft_probs_replicas = [
                TensorType(
                    DType.float32,
                    [
                        draft_probs_rows_name(replica),
                        "num_steps",
                        spec.vocab_size,
                    ],
                    device=leader,
                )
                for replica, leader in enumerate(replica_leader_devices(spec))
            ]
        else:
            draft_probs_full = TensorType(
                DType.float32,
                ["batch_size", "num_steps", spec.vocab_size],
                device=device,
            )

    vision_embeddings: list[TensorType] = []
    vision_scatter_indices: list[TensorType] = []
    if spec.enable_vision:
        assert spec.vision_hidden_size is not None
        vision_embeddings = [
            TensorType(
                DType.bfloat16,
                shape=["vision_merged_seq_len", spec.vision_hidden_size],
                device=DeviceRef.from_device(d),
            )
            for d in devices
        ]
        vision_scatter_indices = [
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(d),
            )
            for d in devices
        ]

    # ``draft_slot_ids`` carries the row count as its extent: the host sends
    # ``arange(batch_size * K)`` to draft and an empty tensor to skip. int32
    # (not the uint32 used for row offsets) because ``ops.gather`` requires
    # int32/int64 indices. ``draft_block_offsets`` is the block's ragged
    # offset vector: ``[0, K, 2K, ...]`` to draft and all zeros to skip, which
    # reads as ``batch_size`` sequences of zero query rows and so keeps the KV
    # lookup table aligned 1:1.
    skippable = spec.include_skippable_draft
    # Packed int32 bitmask (1 bit per token, 32 tokens per word): the GPU
    # acceptance sampler unpacks and applies it in one fused pass
    # (apply_packed_bitmask), so the host never unpacks to bool and the
    # in-graph H2D moves 8x less data.
    structured = spec.enable_structured_output
    bitmask = ["batch_size", "num_bitmask_positions", "packed_vocab_size"]
    batch_context_length = TensorType(DType.int32, shape=[1], device=cpu)

    return SpecDecodeGraphInputs(
        tokens=TensorType(DType.int64, shape=["total_seq_len"], device=device),
        input_row_offsets=TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device
        ),
        return_n_logits=TensorType(
            DType.int64, shape=["return_n_logits"], device=cpu
        ),
        kv_tree=kv_params.get_symbolic_inputs(),
        draft_tokens=TensorType(
            DType.int64, ["batch_size", "num_steps"], device=device
        ),
        seed=per_row(DType.uint64),
        temperature=per_row(DType.float32),
        top_k=per_row(DType.int64),
        max_k=TensorType(DType.int64, shape=[], device=cpu),
        top_p=per_row(DType.float32),
        min_top_p=TensorType(DType.float32, shape=[], device=cpu),
        host_input_row_offsets=(
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=cpu
            )
            if spec.distributed
            else None
        ),
        data_parallel_splits=(
            TensorType(
                DType.int64,
                shape=[spec.data_parallel_degree + 1],
                device=cpu,
            )
            if spec.distributed
            else None
        ),
        draft_probs_full=draft_probs_full,
        draft_probs_replicas=draft_probs_replicas,
        in_thinking_phase=(
            per_row(DType.bool) if spec.include_in_thinking_phase else None
        ),
        draft_slot_ids=(
            TensorType(DType.int32, shape=["num_draft_slots"], device=device)
            if skippable
            else None
        ),
        draft_block_offsets=(
            TensorType(DType.uint32, shape=["num_draft_offsets"], device=device)
            if skippable
            else None
        ),
        pinned_bitmask=(
            TensorType(DType.int32, shape=bitmask, device=cpu)
            if structured
            else None
        ),
        wait_payload=(
            BufferType(DType.int64, shape=[2], device=cpu)
            if structured
            else None
        ),
        device_bitmask_scratch=(
            BufferType(DType.int32, shape=bitmask, device=device)
            if structured
            else None
        ),
        vision_embeddings=vision_embeddings,
        vision_scatter_indices=vision_scatter_indices,
        signal_buffers=(
            list(Signals(devices=devices).input_types())
            if _declares_signal_buffers(spec)
            else []
        ),
        batch_context_lengths=(
            [batch_context_length for _ in devices] if spec.distributed else []
        ),
        ep_inputs=list(ep_input_types),
        trailing=list(trailing_input_types),
    )


class _SpecDecodeTail(Protocol):
    """The canonical tail's fields, as a record of inputs holds them."""

    @property
    def draft_tokens(self) -> object: ...
    @property
    def draft_probs_full(self) -> object: ...
    @property
    def draft_probs_replicas(self) -> Sequence[object] | None: ...
    @property
    def seed(self) -> object: ...
    @property
    def temperature(self) -> object: ...
    @property
    def top_k(self) -> object: ...
    @property
    def max_k(self) -> object: ...
    @property
    def top_p(self) -> object: ...
    @property
    def min_top_p(self) -> object: ...
    @property
    def in_thinking_phase(self) -> object: ...
    @property
    def draft_slot_ids(self) -> object: ...
    @property
    def draft_block_offsets(self) -> object: ...
    @property
    def pinned_bitmask(self) -> object: ...
    @property
    def wait_payload(self) -> object: ...
    @property
    def device_bitmask_scratch(self) -> object: ...


def spec_decode_tail_groups(
    inputs: _SpecDecodeTail, *, thinking: bool, bitmask: bool
) -> list[SpecDecode[object] | TokenBitmask[object]]:
    """Groups the canonical tail every unified spec-decode graph ends with.

    The ``spec`` group holds the drafts, the distributions they were sampled
    from, each row's sampling inputs, the thinking-phase flags and the draft
    row inputs; the ``bitmask`` group follows it. A pipeline sets the
    distributions and the draft row inputs exactly when its graph declares
    them, so their presence in ``inputs`` stands for the signature's flags.

    Args:
        inputs: The declared types, the graph's values or a step's buffers,
            by field.
        thinking: Whether the signature takes ``in_thinking_phase``.
        bitmask: Whether the signature takes the structured-output bitmask.

    Returns:
        The ``spec`` group, then the ``bitmask`` group under ``bitmask``.

    Raises:
        ValueError: If ``inputs`` lacks an input the tail takes.
    """

    def required(name: str, value: object) -> object:
        if value is None:
            raise ValueError(
                f"the spec-decode signature takes {name}, which these inputs"
                " lack"
            )
        return value

    tail: SpecDecode[object] = SpecDecode(
        draft_tokens=required("draft_tokens", inputs.draft_tokens)
    )
    if inputs.draft_probs_replicas is not None:
        tail["draft_probs_full"] = tuple(inputs.draft_probs_replicas)
    elif inputs.draft_probs_full is not None:
        tail["draft_probs_full"] = inputs.draft_probs_full
    for name, value in (
        ("seed", inputs.seed),
        ("temperature", inputs.temperature),
        ("top_k", inputs.top_k),
        ("max_k", inputs.max_k),
        ("top_p", inputs.top_p),
        ("min_top_p", inputs.min_top_p),
    ):
        tail[name] = required(name, value)
    if thinking:
        tail["in_thinking_phase"] = required(
            "in_thinking_phase", inputs.in_thinking_phase
        )
    if inputs.draft_slot_ids is not None:
        tail["draft_slot_ids"] = inputs.draft_slot_ids
        tail["draft_block_offsets"] = required(
            "draft_block_offsets", inputs.draft_block_offsets
        )
    groups: list[SpecDecode[object] | TokenBitmask[object]] = [tail]
    if bitmask:
        groups.append(
            TokenBitmask(
                pinned_bitmask=required(
                    "pinned_bitmask", inputs.pinned_bitmask
                ),
                wait_payload=required("wait_payload", inputs.wait_payload),
                device_bitmask_scratch=required(
                    "device_bitmask_scratch", inputs.device_bitmask_scratch
                ),
            )
        )
    return groups


def _laid_out(structure: object, leaves: Sequence[object]) -> Any:
    """Lays ``leaves`` out as ``structure`` lays out its own.

    Without a structure they lie in one tuple.
    """
    if structure is None:
        return tuple(leaves)
    return tree.unflatten(tree.flatten(structure)[1], leaves)


def spec_decode_input_tree(
    spec: SpecDecodeInputTypeSpec,
    inputs: SpecDecodeGraphInputs[_T, _B, _V],
    *,
    ep: object | None = None,
    trailing: Mapping[str, object] | None = None,
) -> DecoderGraphInputs:
    """Names a unified spec-decode graph's inputs by group, in graph order.

    The order lives here alone. A graph's signature is this tree's leaves over
    :func:`declared_spec_decode_inputs`, the decode unflattens the graph's
    inputs by its structure, and a pipeline packs a step's buffers as its
    leaves. The groups are tokens, [vision], input_row_offsets,
    [host_input_row_offsets], return_n_logits, [data_parallel_splits],
    [signals], kv, [batch_context_lengths, ep], spec, [bitmask], [ep] and
    [trailing], a bracketed group gated by ``spec``. The expert-parallel
    inputs sit beside the distributed group, or after the tail for a target
    whose attention isn't data-parallel. ``inputs.leading`` stays outside
    the tree, ahead of it.

    The fields a decoder also takes keep a decoder's names and groups, so a
    serving engine feeds them as it feeds a decoder's; ``spec`` and
    ``bitmask`` hold :func:`spec_decode_tail_groups`.

    Args:
        spec: The spec the signature is built from.
        inputs: The declared types, the graph's values or a step's buffers.
        ep: The tree that names ``inputs.ep_inputs``, such as an
            :class:`~max.nn.legacy.graph_groups.EPPayload`. Without it they
            sit in one unnamed ``ep`` tuple.
        trailing: The groups that name ``inputs.trailing``. Without them the
            inputs sit in one unnamed ``trailing`` tuple.

    Raises:
        ValueError: If ``inputs`` lacks an input ``spec`` declares, or ``ep``
            or ``trailing`` names a different number of inputs.
    """
    named = DecoderGraphInputs(tokens=inputs.tokens)
    if spec.enable_vision:
        named["vision"] = VisionMerge(
            image_embeddings=tuple(inputs.vision_embeddings),
            image_token_indices=tuple(inputs.vision_scatter_indices),
        )
    named["input_row_offsets"] = inputs.input_row_offsets
    if spec.distributed:
        named["host_input_row_offsets"] = inputs.host_offsets
    named["return_n_logits"] = inputs.return_n_logits
    if spec.distributed:
        named["data_parallel_splits"] = inputs.dp_splits
    if _declares_signal_buffers(spec):
        named["signals"] = TPSignals(signal=tuple(inputs.signal_buffers))
    named["kv"] = inputs.kv_tree
    if spec.distributed:
        named["batch_context_lengths"] = BatchContext(
            length=tuple(inputs.batch_context_lengths)
        )
        if inputs.ep_inputs:
            named["ep"] = _laid_out(ep, inputs.ep_inputs)
    tail, *bitmask = spec_decode_tail_groups(
        inputs,
        thinking=spec.include_in_thinking_phase,
        bitmask=spec.enable_structured_output,
    )
    named["spec"] = tail
    if bitmask:
        named["bitmask"] = bitmask[0]
    if not spec.distributed and inputs.ep_inputs:
        named["ep"] = _laid_out(ep, inputs.ep_inputs)
    if inputs.trailing:
        if trailing is None:
            named["trailing"] = tuple(inputs.trailing)
        else:
            named.update(_laid_out(trailing, inputs.trailing))
    return named


def build_spec_decode_input_types(
    spec: SpecDecodeInputTypeSpec,
    *,
    kv_params: KVCacheParamInterface,
    ep_input_types: Sequence[TensorType | BufferType] = (),
    leading_input_types: Sequence[TensorType | BufferType] = (),
    trailing_input_types: Sequence[TensorType | BufferType] = (),
) -> tuple[TensorType | BufferType, ...]:
    """Builds the canonical unified spec-decode graph input signature.

    The leaves of :func:`spec_decode_input_tree` over
    :func:`declared_spec_decode_inputs`, after ``leading_input_types``.

    ``leading_input_types`` sits ahead of ``tokens`` rather than past the
    tail. A model needs it when a non-spec-decode graph for the same target
    already declares that group first and the two signatures have to stay
    interchangeable -- MiniMax-M3's indexer score scratch, which its base
    backbone declares ahead of everything else.
    """
    declared = declared_spec_decode_inputs(
        spec,
        kv_params=kv_params,
        ep_input_types=ep_input_types,
        trailing_input_types=trailing_input_types,
    )
    return (
        *leading_input_types,
        *tree.leaves(spec_decode_input_tree(spec, declared)),
    )


def _tensor(value: Value[Any] | None) -> TensorValue | None:
    return None if value is None else value.tensor


def decode_spec_decode_input_values(
    graph_inputs: Iterable[Value[Any]],
    spec: SpecDecodeInputTypeSpec,
    *,
    kv_params: KVCacheParamInterface,
    ep_input_types: Sequence[TensorType | BufferType] = (),
    leading_input_types: Sequence[TensorType | BufferType] = (),
    trailing_input_types: Sequence[TensorType | BufferType] = (),
) -> SpecDecodeGraphInputs:
    """Decodes a unified spec-decode graph's positional inputs by name.

    The inverse of :func:`build_spec_decode_input_types`: it unflattens the
    inputs past the leading ones by :func:`spec_decode_input_tree`'s
    structure and reads each group back. Pass the arguments the signature was
    built with.

    Args:
        graph_inputs: ``graph.inputs`` of a graph built from this signature.
        spec: The spec the signature was built from.
        kv_params: The unified ``{"target", "draft"}`` KV params.
        ep_input_types: The target's expert-parallel inputs.
        leading_input_types: The inputs the signature takes ahead of
            ``tokens``.
        trailing_input_types: The inputs the signature takes past the
            canonical tail, returned as :attr:`SpecDecodeGraphInputs.trailing`.

    Returns:
        Every input, named. See :class:`SpecDecodeGraphInputs`.

    Raises:
        ValueError: If ``graph_inputs`` holds more or fewer values than the
            signature declares, which means the two have drifted apart.
    """
    declared = declared_spec_decode_inputs(
        spec,
        kv_params=kv_params,
        ep_input_types=ep_input_types,
        trailing_input_types=trailing_input_types,
    )
    structure = tree.flatten(spec_decode_input_tree(spec, declared))[1]
    values = list(graph_inputs)
    lead = len(leading_input_types)
    if len(values) != lead + structure.num_leaves:
        raise ValueError(
            f"unified spec-decode graph has {len(values)} inputs where its"
            f" signature declares {lead + structure.num_leaves}; the graph"
            " and this spec disagree"
        )
    named = tree.unflatten(structure, values[lead:])
    tail = named["spec"]
    bitmask = named.get("bitmask")
    vision = named.get("vision")
    probs = tail.get("draft_probs_full")
    replicas = bool(spec.draft_probs_replicas)
    return SpecDecodeGraphInputs(
        tokens=named["tokens"].tensor,
        input_row_offsets=named["input_row_offsets"].tensor,
        return_n_logits=named["return_n_logits"].tensor,
        kv_tree=kv_params.unflatten_kv_inputs(iter(tree.leaves(named["kv"]))),
        draft_tokens=tail["draft_tokens"].tensor,
        seed=tail["seed"].tensor,
        temperature=tail["temperature"].tensor,
        top_k=tail["top_k"].tensor,
        max_k=tail["max_k"].tensor,
        top_p=tail["top_p"].tensor,
        min_top_p=tail["min_top_p"].tensor,
        host_input_row_offsets=_tensor(named.get("host_input_row_offsets")),
        data_parallel_splits=_tensor(named.get("data_parallel_splits")),
        draft_probs_full=None if replicas else _tensor(probs),
        draft_probs_replicas=(
            [value.tensor for value in probs] if replicas else None
        ),
        in_thinking_phase=_tensor(tail.get("in_thinking_phase")),
        draft_slot_ids=_tensor(tail.get("draft_slot_ids")),
        draft_block_offsets=_tensor(tail.get("draft_block_offsets")),
        pinned_bitmask=(bitmask["pinned_bitmask"].tensor if bitmask else None),
        wait_payload=bitmask["wait_payload"].buffer if bitmask else None,
        device_bitmask_scratch=(
            bitmask["device_bitmask_scratch"].buffer if bitmask else None
        ),
        vision_embeddings=(
            [value.tensor for value in vision["image_embeddings"]]
            if vision
            else []
        ),
        vision_scatter_indices=(
            [value.tensor for value in vision["image_token_indices"]]
            if vision
            else []
        ),
        signal_buffers=[
            value.buffer for value in named.get("signals", {}).get("signal", ())
        ],
        batch_context_lengths=[
            value.tensor
            for value in named.get("batch_context_lengths", {}).get(
                "length", ()
            )
        ],
        ep_inputs=tree.leaves(named.get("ep", ())),
        leading=values[:lead],
        trailing=values[len(values) - len(trailing_input_types) :]
        if trailing_input_types
        else [],
    )


def spec_decode_outputs(
    outputs: Sequence[TensorValue],
) -> OrderedDict[str, SpecDecodeOutputs[TensorValue]]:
    """Names a spec-decode driver's outputs, as a decoder names its groups.

    Args:
        outputs: ``(num_accepted, next_tokens, next_draft_tokens)``, then
            under a sampled draft proposal the draft's distributions, one
            per data-parallel replica when it holds them per replica.

    Returns:
        The outputs in one ``spec`` group, in the same order.
    """
    num_accepted, next_tokens, next_draft_tokens, *probs = outputs
    named = SpecDecodeOutputs(
        num_accepted=num_accepted,
        next_tokens=next_tokens,
        next_draft_tokens=next_draft_tokens,
    )
    if probs:
        named["next_draft_probs"] = (
            probs[0] if len(probs) == 1 else tuple(probs)
        )
    return OrderedDict(spec=named)


if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    class _DriverBatch(DataclassInstance, Protocol):
        """The KV fields that :func:`without_mixed_step_split` reads."""

        @property
        def kv_collections(self) -> object: ...

        @property
        def draft_kv_collections(self) -> object: ...

        @property
        def passthrough_kv(self) -> object: ...


_Batch = TypeVar("_Batch", bound="_DriverBatch")


_KV = TypeVar("_KV")


def drop_mixed_step_split(kv: _KV) -> _KV:
    """Returns the KV tree ``kv`` without the mixed-step split records.

    The record describes the verify batch's rows; a draft forward runs over
    other rows, so every leaf a draft can reach drops it. Leaves of another
    kind, such as recurrent state, pass through.
    """

    def drop(leaf: object) -> object:
        if isinstance(leaf, KVCacheInputsPerDevice):
            return replace(leaf, mixed_step_split=None)
        return leaf

    return tree.map(drop, kv, leaf=KVCacheInputsPerDevice)


def without_mixed_step_split(batch: _Batch) -> _Batch:
    """Returns ``batch`` with no split record on any KV leaf it holds."""
    return replace(
        batch,
        kv_collections=drop_mixed_step_split(batch.kv_collections),
        draft_kv_collections=drop_mixed_step_split(batch.draft_kv_collections),
        passthrough_kv=drop_mixed_step_split(batch.passthrough_kv),
    )


class SpecDecodeGraphSignature:
    """Derives a spec-decode graph's input list and the decode that reads it.

    :meth:`input_types` orders the graph's inputs; :meth:`decode_inputs`
    maps a built graph's positional inputs back to names. A model
    contributes only :attr:`input_spec` (plus :meth:`ep_input_types` under
    expert parallelism), so the order a graph is built with and the decode
    that reads it cannot disagree.
    """

    @property
    def input_spec(self) -> SpecDecodeInputTypeSpec:
        """The spec both directions are derived from."""
        raise NotImplementedError(
            f"{type(self).__name__} must define input_spec"
        )

    @property
    def signature_kv_params(self) -> KVCacheParamInterface | None:
        """The KV params this model owns, for graphs whose caller has none.

        The single-device EAGLE and DFlash graphs build their own tree.
        """
        return None

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        """The target's expert-parallel inputs; empty when EP is off."""
        return ()

    def leading_input_types(self) -> Sequence[TensorType | BufferType]:
        """This model's own group ahead of ``tokens``; empty for most models.

        Both directions read it, so the count the decode skips is always the
        count the signature declared.
        """
        return ()

    def trailing_input_types(self) -> Sequence[TensorType | BufferType]:
        """This model's own group past the canonical tail; empty for most.

        :meth:`decode_inputs` returns it as
        :attr:`SpecDecodeGraphInputs.trailing`.
        """
        return ()

    @property
    def has_trailing_inputs(self) -> bool:
        """Whether this model appends its own group past the canonical tail."""
        return bool(self.trailing_input_types())

    def _signature_kv(
        self, kv_params: KVCacheParamInterface | None
    ) -> KVCacheParamInterface:
        kv = kv_params if kv_params is not None else self.signature_kv_params
        if kv is None:
            raise ValueError(
                f"{type(self).__name__} needs kv_params: pass them in or"
                " override signature_kv_params"
            )
        return kv

    def input_types(
        self, kv_params: KVCacheParamInterface | None = None
    ) -> tuple[TensorType | BufferType, ...]:
        """Builds the unified spec-decode graph signature.

        The leaves of :meth:`input_tree`, after the leading inputs. See
        :func:`spec_decode_input_tree` for the ordering.
        """
        return build_spec_decode_input_types(
            self.input_spec,
            kv_params=self._signature_kv(kv_params),
            ep_input_types=self.ep_input_types(),
            leading_input_types=self.leading_input_types(),
            trailing_input_types=self.trailing_input_types(),
        )

    @overload
    def decode_inputs(
        self,
        graph_inputs: Iterable[Value[Any]],
        kv_params: KVCacheParamInterface | None = None,
    ) -> SpecDecodeGraphInputs: ...

    @overload
    def decode_inputs(
        self,
        graph_inputs: Iterable[Tensor],
        kv_params: KVCacheParamInterface | None = None,
    ) -> SpecDecodeGraphInputs[Tensor, Tensor, Tensor]: ...

    def decode_inputs(
        self,
        graph_inputs: Iterable[Value[Any]] | Iterable[Tensor],
        kv_params: KVCacheParamInterface | None = None,
    ) -> SpecDecodeGraphInputs[Any, Any, Any]:
        """Decodes a graph built from :meth:`input_types` back into names.

        Takes the graph's inputs, or the single-device ``Tensor`` values a
        ModuleV3 ``forward`` receives for them; the result holds the same
        type.
        """
        inputs = list(graph_inputs)
        takes_tensors = any(isinstance(value, Tensor) for value in inputs)
        decoded = decode_spec_decode_input_values(
            as_graph_values(inputs),
            self.input_spec,
            kv_params=self._signature_kv(kv_params),
            ep_input_types=self.ep_input_types(),
            leading_input_types=self.leading_input_types(),
            trailing_input_types=self.trailing_input_types(),
        )
        # The driver declares the draft row inputs from the config, so it
        # takes them back here rather than having every model forward them.
        self._decoded_draft_rows = (
            decoded.draft_slot_ids,
            decoded.draft_block_offsets,
        )
        return as_tensors(decoded) if takes_tensors else decoded

    def input_tree(
        self,
        kv_params: KVCacheParamInterface | None = None,
        *,
        ep: object | None = None,
        trailing: Mapping[str, object] | None = None,
    ) -> DecoderGraphInputs | None:
        """Names the graph's inputs by group.

        :meth:`input_types` is this tree's leaves.

        Args:
            kv_params: The KV params the graph is built with.
            ep: The tree that names the expert-parallel inputs, when the
                signature takes them.
            trailing: The groups that name the trailing inputs, when the
                signature takes them.

        Returns:
            The named inputs, or ``None`` for a signature with inputs no
            group names: leading ones, or trailing or expert-parallel ones
            without their tree.
        """
        if (
            self.leading_input_types()
            or (self.has_trailing_inputs and trailing is None)
            or (self.ep_input_types() and ep is None)
        ):
            return None
        declared = declared_spec_decode_inputs(
            self.input_spec,
            kv_params=self._signature_kv(kv_params),
            ep_input_types=self.ep_input_types(),
            trailing_input_types=self.trailing_input_types(),
        )
        return spec_decode_input_tree(
            self.input_spec, declared, ep=ep, trailing=trailing
        )

    def _draft_rows(
        self,
        draft_slot_ids: TensorValue | None,
        draft_block_offsets: TensorValue | None,
    ) -> tuple[TensorValue | None, TensorValue | None]:
        """The draft row inputs: the caller's, else the decoded graph's."""
        if draft_slot_ids is not None or not (
            self.input_spec.include_skippable_draft
        ):
            return draft_slot_ids, draft_block_offsets
        decoded = getattr(self, "_decoded_draft_rows", None)
        if decoded is None:
            raise ValueError(
                f"{type(self).__name__} declares the draft row inputs, so"
                " build its graph through decode_inputs() or pass them"
            )
        return decoded
