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
from typing import Any

from max import tree
from max.dtype import DType
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
    enable_sampled_draft_proposal: bool = False
    """Declare the ``draft_probs_full`` input: the distribution the draft
    sampled its token from, which the acceptance test's residual subtracts and
    reads ``q`` out of. Requires ``vocab_size``. Set by the MiniMax-M3 unified
    pipelines and, through :class:`SequentialDriver`, by any driver built with
    ``draft_proposal="sampled"``."""
    vocab_size: int | None = None
    """Static vocabulary size, required by ``enable_sampled_draft_proposal``."""


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
    max_k, top_p, min_top_p, [in_thinking_phase], [bitmask triple]. Bracketed
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
    if spec.distributed or spec.include_signal_buffers:
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


@dataclass(frozen=True)
class SpecDecodeGraphInputs:
    """A unified spec-decode graph's inputs, named rather than positional.

    Every field corresponds to one group of
    :func:`build_spec_decode_input_types`, and a field is ``None`` (or empty)
    exactly when the spec flag that gates its group is unset.
    """

    tokens: TensorValue
    input_row_offsets: TensorValue
    return_n_logits: TensorValue
    kv_tree: KVCacheInputs[TensorValue, BufferValue]
    draft_tokens: TensorValue
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue

    host_input_row_offsets: TensorValue | None = None
    data_parallel_splits: TensorValue | None = None
    draft_probs_full: TensorValue | None = None
    in_thinking_phase: TensorValue | None = None
    pinned_bitmask: TensorValue | None = None
    wait_payload: BufferValue | None = None
    device_bitmask_scratch: BufferValue | None = None

    vision_embeddings: list[TensorValue] = field(default_factory=list)
    vision_scatter_indices: list[TensorValue] = field(default_factory=list)
    signal_buffers: list[BufferValue] = field(default_factory=list)
    batch_context_lengths: list[TensorValue] = field(default_factory=list)
    ep_inputs: list[Value[Any]] = field(default_factory=list)

    leading: list[Value[Any]] = field(default_factory=list)
    """Inputs ahead of ``tokens``, for a model that declares its own group
    first. Empty unless the signature declared one."""

    trailing: list[Value[Any]] = field(default_factory=list)
    """Inputs past the canonical tail, for a model that appends its own
    group. Empty unless the decode allowed trailing."""

    def kv(
        self, *path: str
    ) -> list[KVCacheInputsPerDevice[TensorValue, BufferValue]]:
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
    def host_offsets(self) -> TensorValue:
        """:attr:`host_input_row_offsets`, for a model that declares it.

        Declared by every ``distributed=True`` spec and no other.
        """
        assert self.host_input_row_offsets is not None
        return self.host_input_row_offsets

    @property
    def dp_splits(self) -> TensorValue:
        """:attr:`data_parallel_splits`, for a model that declares it."""
        assert self.data_parallel_splits is not None
        return self.data_parallel_splits

    @property
    def thinking_phase(self) -> TensorValue:
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
            i.e. ``len(ep_input_types)`` as passed to the builder.
        num_leading_inputs: Number of inputs the signature declared ahead of
            ``tokens``, i.e. ``len(leading_input_types)`` as passed to the
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
    if spec.distributed or spec.include_signal_buffers:
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

    def decode_inputs(
        self,
        graph_inputs: Iterable[Value[Any]],
        kv_params: KVCacheParamInterface | None = None,
    ) -> SpecDecodeGraphInputs:
        """Decodes a graph built from :meth:`input_types` back into names."""
        return decode_spec_decode_input_values(
            graph_inputs,
            self.input_spec,
            kv_params=self._signature_kv(kv_params),
            num_ep_inputs=len(self.ep_input_types()),
            num_leading_inputs=len(self.leading_input_types()),
            allow_trailing=self.has_trailing_inputs,
        )
