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

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Generic, overload

from max import tree
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
)
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
)
from typing_extensions import TypeVar

from ._tensor_compat import as_graph_values, as_tensors

__all__ = [
    "SpecDecodeGraphInputs",
    "SpecDecodeGraphSignature",
    "SpecDecodeInputTypeSpec",
    "SpecDecodeTailValues",
    "build_spec_decode_input_types",
    "decode_spec_decode_input_values",
    "decode_spec_decode_tail",
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

    A graph traced by :meth:`~max.experimental.nn.Module.compile` appends its
    own after every declared input, so declaring them here too would leave
    two sets."""
    enable_sampled_draft_proposal: bool = False
    """Declare the ``draft_probs_full`` input: the distribution the draft
    sampled its token from, which the acceptance test's residual subtracts and
    reads ``q`` out of. Requires ``vocab_size``. Set by the MiniMax-M3 unified
    pipelines and, through :class:`SequentialDriver`, by any driver built with
    ``draft_proposal="sampled"``."""
    vocab_size: int | None = None
    """Static vocabulary size, required by ``enable_sampled_draft_proposal``."""


def _declares_signal_buffers(spec: SpecDecodeInputTypeSpec) -> bool:
    return (
        spec.distributed or spec.include_signal_buffers
    ) and not spec.trace_owns_signal_buffers


def build_spec_decode_input_types(
    spec: SpecDecodeInputTypeSpec,
    *,
    kv_params: KVCacheParamInterface,
    ep_input_types: Sequence[TensorType | BufferType] = (),
    leading_input_types: Sequence[TensorType | BufferType] = (),
) -> tuple[TensorType | BufferType, ...]:
    """Builds the canonical unified spec-decode graph input signature.

    Order: [leading], tokens, [vision], device_offsets, [host_offsets],
    return_n_logits,
    [data_parallel_splits], [signals], kv_cache_tree,
    [batch_context_lengths, ep], draft_tokens, [draft_probs_full], seed,
    temperature, top_k,
    max_k, top_p, min_top_p, [in_thinking_phase],
    [draft_slot_ids, draft_block_offsets], [bitmask triple]. Bracketed
    groups are gated by the spec flags; the tail mirrors
    ``UnifiedSpecDecodeInputs._spec_decode_tail_buffers``.

    ``leading_input_types`` sits ahead of ``tokens`` rather than past the
    tail. A model needs it when a non-spec-decode graph for the same target
    already declares that group first and the two signatures have to stay
    interchangeable -- MiniMax-M3's indexer score scratch, which its base
    backbone declares ahead of everything else.

    ``kv_params`` is the unified ``{"target", "draft"}`` KV tree; its flattened
    inputs (target leaf then draft leaf) carry both caches' blocks and dispatch
    metadata, so there is no longer a separate draft-KV-blocks group.
    """
    devices = spec.devices
    device_ref = devices[0]

    all_input_types: list[TensorType | BufferType] = [
        *leading_input_types,
        TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
    ]

    if spec.enable_vision:
        assert spec.vision_hidden_size is not None
        all_input_types.extend(
            TensorType(
                DType.bfloat16,
                shape=["vision_merged_seq_len", spec.vision_hidden_size],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        )
        all_input_types.extend(
            TensorType(
                DType.int32,
                shape=["total_image_tokens"],
                device=DeviceRef.from_device(device),
            )
            for device in devices
        )

    all_input_types.append(
        TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
    )
    if spec.distributed:
        all_input_types.append(
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            )
        )
    all_input_types.append(
        TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
    )

    if spec.distributed:
        all_input_types.append(
            TensorType(
                DType.int64,
                shape=[spec.data_parallel_degree + 1],
                device=DeviceRef.CPU(),
            )
        )
    if _declares_signal_buffers(spec):
        all_input_types.extend(Signals(devices=devices).input_types())

    all_input_types.extend(kv_params.flattened_kv_inputs())

    if spec.distributed:
        batch_context_length_type = TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        )
        all_input_types.extend(
            batch_context_length_type for _ in range(len(devices))
        )
        all_input_types.extend(ep_input_types)

    all_input_types.extend(spec_decode_tail_input_types(spec, device_ref))

    return tuple(all_input_types)


def spec_decode_tail_input_types(
    spec: SpecDecodeInputTypeSpec, device_ref: DeviceRef
) -> tuple[TensorType | BufferType, ...]:
    """The input-type tail every unified spec-decode graph ends with.

    Must stay ordered in lockstep with
    ``UnifiedSpecDecodeInputs._spec_decode_tail_buffers``.
    """
    all_input_types: list[TensorType | BufferType] = [
        TensorType(DType.int64, ["batch_size", "num_steps"], device=device_ref)
    ]

    if spec.enable_sampled_draft_proposal:
        if spec.vocab_size is None:
            raise ValueError(
                "vocab_size is required when enable_sampled_draft_proposal is"
                " set"
            )
        all_input_types.append(
            TensorType(
                DType.float32,
                ["batch_size", "num_steps", spec.vocab_size],
                device=device_ref,
            )
        )

    all_input_types.append(
        TensorType(DType.uint64, shape=["batch_size"], device=device_ref)
    )
    all_input_types.extend(
        [
            TensorType(DType.float32, shape=["batch_size"], device=device_ref),
            TensorType(DType.int64, shape=["batch_size"], device=device_ref),
            TensorType(DType.int64, shape=[], device=DeviceRef.CPU()),
            TensorType(DType.float32, shape=["batch_size"], device=device_ref),
            TensorType(DType.float32, shape=[], device=DeviceRef.CPU()),
        ]
    )
    if spec.include_in_thinking_phase:
        all_input_types.append(
            TensorType(DType.bool, shape=["batch_size"], device=device_ref)
        )

    if spec.include_skippable_draft:
        # ``draft_slot_ids`` carries the row count as its extent: the host
        # sends ``arange(batch_size * K)`` to draft and an empty tensor to
        # skip. int32 (not the uint32 used for row offsets) because
        # ``ops.gather`` requires int32/int64 indices. ``draft_block_offsets``
        # is the block's ragged offset vector: ``[0, K, 2K, ...]`` to draft and
        # all zeros to skip, which reads as ``batch_size`` sequences of zero
        # query rows and so keeps the KV lookup table aligned 1:1.
        all_input_types.extend(
            [
                TensorType(
                    DType.int32, shape=["num_draft_slots"], device=device_ref
                ),
                TensorType(
                    DType.uint32,
                    shape=["num_draft_offsets"],
                    device=device_ref,
                ),
            ]
        )

    if spec.enable_structured_output:
        # Packed int32 bitmask (1 bit per token, 32 tokens per word): the GPU
        # acceptance sampler unpacks and applies it in one fused pass
        # (apply_packed_bitmask), so the host never unpacks to bool and the
        # in-graph H2D moves 8x less data.
        all_input_types.extend(
            [
                TensorType(
                    DType.int32,
                    shape=[
                        "batch_size",
                        "num_bitmask_positions",
                        "packed_vocab_size",
                    ],
                    device=DeviceRef.CPU(),
                ),
                BufferType(DType.int64, shape=[2], device=DeviceRef.CPU()),
                BufferType(
                    DType.int32,
                    shape=[
                        "batch_size",
                        "num_bitmask_positions",
                        "packed_vocab_size",
                    ],
                    device=device_ref,
                ),
            ]
        )
    return tuple(all_input_types)


_T = TypeVar("_T", TensorValue, Tensor, default=TensorValue)
"""The inputs' value type: a graph value, or an experimental ``Tensor``."""

_B = TypeVar("_B", BufferValue, Tensor, default=BufferValue)
"""The buffer type, ``Tensor`` alongside a ``Tensor`` :obj:`_T`."""

_V = TypeVar("_V", Value[Any], Tensor, default=Value[Any])
"""The type of an input the decode does not interpret, ``Tensor`` alongside a
``Tensor`` :obj:`_T`."""


@tree.dataclass(frozen=True)
class SpecDecodeGraphInputs(Generic[_T, _B, _V]):
    """A unified spec-decode graph's inputs, named rather than positional.

    Every field corresponds to one group of
    :func:`build_spec_decode_input_types`, and a field is ``None`` (or empty)
    exactly when the spec flag that gates its group is unset.
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


@dataclass(frozen=True)
class SpecDecodeTailValues:
    """The decoded tail every unified spec-decode graph ends with."""

    draft_tokens: TensorValue
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue
    draft_probs_full: TensorValue | None = None
    in_thinking_phase: TensorValue | None = None
    draft_slot_ids: TensorValue | None = None
    """Which drafter rows this step computes, as the tensor's extent:
    ``arange(batch_size * rows_per_seq)`` to draft, empty to skip. ``None``
    unless the spec set ``include_skippable_draft``."""
    draft_block_offsets: TensorValue | None = None
    """The draft forward's ragged row offsets: ``[0, K, 2K, ...]`` to draft,
    all zeros to skip. ``None`` unless the spec set
    ``include_skippable_draft``."""
    pinned_bitmask: TensorValue | None = None
    wait_payload: BufferValue | None = None
    device_bitmask_scratch: BufferValue | None = None


def _next_input(it: Iterator[Value[Any]], *, decoding: str) -> Value[Any]:
    """Returns the next graph input, or says which decode ran dry.

    Bare ``next`` would raise ``StopIteration``, which names no cause and
    which PEP 479 turns into an opaque ``RuntimeError`` inside a generator.

    Args:
        it: The positioned graph-input iterator.
        decoding: The group being decoded, named in the error.

    Returns:
        The next value from ``it``.

    Raises:
        ValueError: If ``it`` is exhausted.
    """
    try:
        return next(it)
    except StopIteration:
        raise ValueError(
            f"unified spec-decode graph ran out of inputs while decoding"
            f" {decoding}; the graph signature and this spec disagree"
        ) from None


def decode_spec_decode_tail(
    it: Iterator[Value[Any]], spec: SpecDecodeInputTypeSpec
) -> SpecDecodeTailValues:
    """Decodes the canonical tail from a positioned graph-input iterator.

    The inverse of :func:`spec_decode_tail_input_types`, for a model whose
    head is its own shape. Leaves ``it`` positioned after the tail.
    """
    take = partial(_next_input, it, decoding="the canonical tail")

    draft_tokens = take().tensor
    draft_probs_full = (
        take().tensor if spec.enable_sampled_draft_proposal else None
    )
    seed = take().tensor
    temperature = take().tensor
    top_k = take().tensor
    max_k = take().tensor
    top_p = take().tensor
    min_top_p = take().tensor
    in_thinking_phase = (
        take().tensor if spec.include_in_thinking_phase else None
    )
    draft_slot_ids = take().tensor if spec.include_skippable_draft else None
    draft_block_offsets = (
        take().tensor if spec.include_skippable_draft else None
    )

    pinned_bitmask: TensorValue | None = None
    wait_payload: BufferValue | None = None
    device_bitmask_scratch: BufferValue | None = None
    if spec.enable_structured_output:
        pinned_bitmask = take().tensor
        wait_payload = take().buffer
        device_bitmask_scratch = take().buffer

    return SpecDecodeTailValues(
        draft_tokens=draft_tokens,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        max_k=max_k,
        top_p=top_p,
        min_top_p=min_top_p,
        draft_probs_full=draft_probs_full,
        in_thinking_phase=in_thinking_phase,
        draft_slot_ids=draft_slot_ids,
        draft_block_offsets=draft_block_offsets,
        pinned_bitmask=pinned_bitmask,
        wait_payload=wait_payload,
        device_bitmask_scratch=device_bitmask_scratch,
    )


def decode_spec_decode_input_values(
    graph_inputs: Iterable[Value[Any]],
    spec: SpecDecodeInputTypeSpec,
    *,
    kv_params: KVCacheParamInterface,
    num_ep_inputs: int = 0,
    num_leading_inputs: int = 0,
    allow_trailing: bool = False,
) -> SpecDecodeGraphInputs:
    """Decodes a unified spec-decode graph's positional inputs by name.

    The exact inverse of :func:`build_spec_decode_input_types`, walking the
    same groups in the same order. Pass both the same ``spec`` object.

    Args:
        graph_inputs: ``graph.inputs`` of a graph built from this spec.
        spec: The spec the graph's signature was built from.
        kv_params: The unified ``{"target", "draft"}`` KV params, used to
            unflatten the KV group.
        num_ep_inputs: Number of expert-parallel inputs the target declared,
            that is, ``len(ep_input_types)`` as passed to the builder.
        num_leading_inputs: Number of inputs the signature declared ahead of
            ``tokens``; that is, ``len(leading_input_types)`` as passed to the
            builder. A count rather than a flag, because these sit before
            every anchor the decode could otherwise resynchronize on.
        allow_trailing: Accept inputs past the canonical tail and return them
            as :attr:`SpecDecodeGraphInputs.trailing`, for a model that
            appends its own group. Off by default so an unexpected leftover
            stays an error.

    Returns:
        Every input, named. See :class:`SpecDecodeGraphInputs`.

    Raises:
        ValueError: If ``graph_inputs`` holds more or fewer values than
            ``spec`` accounts for, which means the graph's signature and this
            decode have drifted apart.
    """
    devices = spec.devices
    it: Iterator[Value[Any]] = iter(graph_inputs)
    take = partial(_next_input, it, decoding="the canonical inputs")

    leading = [take() for _ in range(num_leading_inputs)]
    tokens = take().tensor

    vision_embeddings: list[TensorValue] = []
    vision_scatter_indices: list[TensorValue] = []
    if spec.enable_vision:
        vision_embeddings = [take().tensor for _ in devices]
        vision_scatter_indices = [take().tensor for _ in devices]

    input_row_offsets = take().tensor
    host_input_row_offsets = take().tensor if spec.distributed else None
    return_n_logits = take().tensor
    data_parallel_splits = take().tensor if spec.distributed else None

    signal_buffers: list[BufferValue] = []
    if _declares_signal_buffers(spec):
        signal_buffers = [take().buffer for _ in devices]

    kv_tree = kv_params.unflatten_kv_inputs(it)

    batch_context_lengths: list[TensorValue] = []
    ep_inputs: list[Value[Any]] = []
    if spec.distributed:
        batch_context_lengths = [take().tensor for _ in devices]
        ep_inputs = [take() for _ in range(num_ep_inputs)]

    tail = decode_spec_decode_tail(it, spec)

    # A leftover means the signature declared a group this decode misses,
    # which positional decoding cannot detect on its own.
    trailing = list(it)
    if trailing and not allow_trailing:
        raise ValueError(
            f"unified spec-decode graph has {len(trailing)} input(s) left over"
            " after decoding; the graph signature and this spec disagree."
            " Pass allow_trailing=True if this model appends its own group."
        )

    return SpecDecodeGraphInputs(
        tokens=tokens,
        input_row_offsets=input_row_offsets,
        return_n_logits=return_n_logits,
        kv_tree=kv_tree,
        draft_tokens=tail.draft_tokens,
        seed=tail.seed,
        temperature=tail.temperature,
        top_k=tail.top_k,
        max_k=tail.max_k,
        top_p=tail.top_p,
        min_top_p=tail.min_top_p,
        host_input_row_offsets=host_input_row_offsets,
        data_parallel_splits=data_parallel_splits,
        draft_probs_full=tail.draft_probs_full,
        in_thinking_phase=tail.in_thinking_phase,
        draft_slot_ids=tail.draft_slot_ids,
        draft_block_offsets=tail.draft_block_offsets,
        pinned_bitmask=tail.pinned_bitmask,
        wait_payload=tail.wait_payload,
        device_bitmask_scratch=tail.device_bitmask_scratch,
        vision_embeddings=vision_embeddings,
        vision_scatter_indices=vision_scatter_indices,
        signal_buffers=signal_buffers,
        batch_context_lengths=batch_context_lengths,
        ep_inputs=ep_inputs,
        leading=leading,
        trailing=trailing,
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

    @property
    def has_trailing_inputs(self) -> bool:
        """Whether this model appends its own group past the canonical tail."""
        return False

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

        A model that appends its own inputs overrides this, calls ``super()``
        and extends the result. See :func:`build_spec_decode_input_types`
        for the ordering.
        """
        return build_spec_decode_input_types(
            self.input_spec,
            kv_params=self._signature_kv(kv_params),
            ep_input_types=self.ep_input_types(),
            leading_input_types=self.leading_input_types(),
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
            num_ep_inputs=len(self.ep_input_types()),
            num_leading_inputs=len(self.leading_input_types()),
            allow_trailing=self.has_trailing_inputs,
        )
        # The driver declares the draft row inputs from the config, so it
        # takes them back here rather than having every model forward them.
        self._decoded_draft_rows = (
            decoded.draft_slot_ids,
            decoded.draft_block_offsets,
        )
        return as_tensors(decoded) if takes_tensors else decoded

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
