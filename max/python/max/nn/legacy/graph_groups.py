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
"""Graph groups: the named inputs or outputs one model feature brings.

A graph's input tree is a handful of groups, such as the ragged token batch,
the allreduce signal buffers, or the vision embeddings, and its output tree
likewise, such as the logits. Each group draws its fields from a fixed set,
whatever model it appears in, so the serving engine learns to feed or read a
group once, by its :attr:`GraphGroup.kind`, rather than once per model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from max import tree
from max.graph import BufferType, TensorType, TensorValue
from typing_extensions import Self, TypeVar

__all__ = [
    "BatchContext",
    "ConvStates",
    "DecoderGraphInputs",
    "DecoderOutputs",
    "EPLBProfile",
    "EPPayload",
    "GraphGroup",
    "HiddenStates",
    "ItemPositions",
    "Logits",
    "LogitsName",
    "MRoPEPositions",
    "RaggedBatch",
    "RecurrentStateTail",
    "SparseIndices",
    "SpecDecode",
    "SpecDecodeOutputs",
    "TPSignals",
    "TokenBitmask",
    "VisionEmbeddings",
    "VisionHostMax",
    "VisionMerge",
    "VisionRowIndex",
    "VisionRows",
    "VisionSegments",
    "VisionSelfIndex",
    "group_fields",
    "vision_outputs",
]

_K = TypeVar("_K", bound=str)
_V = TypeVar("_V")
# A ModuleV3 graph's groups hold tensor layouts and tensors in place of
# graph-API types and values, and a step packs some groups with its buffers.
_TensorT = TypeVar("_TensorT", default=TensorValue)
_MergeT = TypeVar(
    "_MergeT",
    default=tuple[TensorType, ...] | tuple[tuple[TensorType, ...], ...],
)
_SignalsT = TypeVar("_SignalsT", default=tuple[BufferType, ...])
_ContextT = TypeVar("_ContextT", default=tuple[TensorType, ...])
_SpecT = TypeVar("_SpecT", default=TensorType | tuple[TensorType, ...])
_BitmaskT = TypeVar("_BitmaskT", default=TensorType | BufferType)


class GraphGroup(OrderedDict[_K, _V]):
    """The named graph inputs or outputs of one feature, in graph order.

    An ``OrderedDict`` keeps insertion order when flattened, where a plain
    ``dict`` would sort its keys. Each subclass fixes the names and values it
    holds.
    """

    kind: ClassVar[str | None] = None
    """Identifies the feature wherever the group sits in a tree."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # A kind names one class, so a subclass that inherits its base's kind
        # leaves the registration to the base.
        if "kind" in cls.__dict__ and cls.kind is not None:
            tree.kind(cls.kind)(cls)

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        """Creates the group, rejecting a subclass with no ``kind``."""
        if cls.kind is None:
            raise TypeError(
                f"{cls.__qualname__} has no kind, so an exported graph could"
                " not name it: subclass GraphGroup with a kind"
            )
        return super().__new__(cls)

    def __tree_flatten__(self) -> tuple[Mapping[_K, _V], None]:
        return self, None

    @classmethod
    def __tree_unflatten__(cls, meta: None, children: Mapping[_K, _V]) -> Self:
        # Not through ``__init__``, which a subclass may make keyword-only.
        group = cls.__new__(cls)
        OrderedDict.update(group, children)
        return group


class RaggedBatch(GraphGroup[str, TensorType | tuple[TensorType, ...]]):
    """The step's ragged token stream: tokens, row offsets, and friends.

    A model whose devices each take their own row offsets holds them as a
    tuple, one per device.
    """

    kind = "ragged_batch"


class TPSignals(GraphGroup[str, _SignalsT]):
    """The allreduce signal buffers, one per device."""

    kind = "tp_signals"


class VisionMerge(GraphGroup[str, _MergeT]):
    """Image embeddings per device and the token positions they replace.

    A model that injects image features into later layers, such as
    Qwen3-VL's deepstack, holds one per-device tuple per layer.
    """

    kind = "vision_merge"


_VisionInput = TensorType | tuple[TensorType, ...]


class VisionRows(GraphGroup[str, _VisionInput]):
    """Vision encoder inputs with a row per patch, grid, or position.

    A serving engine packs several media items into one launch by
    concatenating each field along its first symbolic dim.
    """

    kind = "vision_rows"


class VisionSegments(GraphGroup[str, _VisionInput]):
    """Vision encoder attention segment boundaries, as ``cu_seqlens``.

    Packing concatenates each item's boundaries after the first, offset by
    the rows packed before it.
    """

    kind = "vision_segments"


class VisionRowIndex(GraphGroup[str, _VisionInput]):
    """Vision encoder indices into its own rows, such as a pooling gather.

    Packing concatenates each item's indices offset by the rows packed
    before it.
    """

    kind = "vision_row_index"


class VisionSelfIndex(GraphGroup[str, _VisionInput]):
    """Vision encoder indices into the field's own entries.

    For example, a permutation of merged tokens.

    Packing concatenates each item's indices offset by the entries packed
    before it.
    """

    kind = "vision_self_index"


class VisionHostMax(GraphGroup[str, _VisionInput]):
    """Vision encoder host scalars that bound the launch.

    For example, the longest segment.

    Packing takes the maximum over the items.
    """

    kind = "vision_host_max"


class VisionEmbeddings(GraphGroup[str, tuple[TensorValue, ...]]):
    """A vision encoder's ``image_embeddings``, one per device.

    They feed the decoder's :class:`VisionMerge` group.
    """

    kind = "vision_embeddings"


def vision_outputs(
    image_embeddings: Sequence[TensorValue],
) -> OrderedDict[str, VisionEmbeddings]:
    """Returns a vision encoder graph's named outputs.

    The kinded group sits under a plain root, as a decoder's do, since a
    serving engine reads a group's kind only below the root.

    Args:
        image_embeddings: The embeddings, one per device.
    """
    return OrderedDict(
        embeddings=VisionEmbeddings(image_embeddings=tuple(image_embeddings))
    )


class MRoPEPositions(GraphGroup[str, TensorType]):
    """The 3D position ids of an M-RoPE model."""

    kind = "mrope_positions"


class ItemPositions(GraphGroup[str, TensorValue]):
    """One media item's decoder positions, which its preprocessing graph returns.

    ``position_ids`` holds one column per placeholder token of the item,
    relative to the position its first one starts at, and
    ``position_advance`` how far the item moves the running text position.
    A serving engine offsets each item's positions by the running position
    where it sits in the prompt.
    """

    kind = "item_positions"


class BatchContext(GraphGroup[str, _ContextT]):
    """Per-device host scalars summarizing the batch's context lengths."""

    kind = "batch_context"


class EPPayload(GraphGroup[str, tuple[TensorType | BufferType, ...]]):
    """Expert-parallel communication state, built once at startup."""

    kind = "ep_payload"


class EPLBProfile(GraphGroup[str, tuple[BufferType, ...]]):
    """Per-layer expert counters an EPLB-profiling graph writes in place."""

    kind = "eplb_profile"


LogitsName = Literal["next_token_logits", "logits", "logit_offsets"]
"""The outputs a :class:`Logits` group can hold."""


class Logits(GraphGroup[LogitsName, _TensorT]):
    """The decoder's logits: the next token's, and any others requested.

    Which are present depends on the decoder's ``return_logits``, and a
    caller that set it reads the ones it asked for.
    """

    kind = "logits"

    def __init__(
        self,
        *,
        next_token_logits: _TensorT | None = None,
        logits: _TensorT | None = None,
        logit_offsets: _TensorT | None = None,
    ) -> None:
        super().__init__()
        if next_token_logits is not None:
            self["next_token_logits"] = next_token_logits
        if logits is not None:
            self["logits"] = logits
        if logit_offsets is not None:
            self["logit_offsets"] = logit_offsets


class HiddenStates(GraphGroup[str, tuple[_TensorT, ...]]):
    """Hidden states a decoder returns for one mode.

    The mode is a drafter, a capture, or the next pipeline stage.
    """

    kind = "hidden_states"

    def tensors(self) -> tuple[_TensorT, ...]:
        """Returns the hidden states, whichever mode produced them."""
        (states,) = self.values()
        return states


class ConvStates(GraphGroup[str, tuple[TensorValue, ...]]):
    """The short-convolution states a decoder writes back for the next step."""

    kind = "conv_states"


class SparseIndices(GraphGroup[str, tuple[TensorValue, ...]]):
    """The rows a sparse-attention indexer selected, for a drafter to reuse."""

    kind = "sparse_indices"


class SpecDecode(GraphGroup[str, _SpecT]):
    """A speculative-decoding step's drafts and per-row sampling inputs.

    The drafts to verify, any distributions the draft sampled them from, and
    the seed, temperature, top-k and top-p each row samples with, as a fused
    speculative-decoding graph takes them.
    """

    kind = "spec_decode"


class TokenBitmask(GraphGroup[str, _BitmaskT]):
    """The structured-output token bitmask and the handshake it arrives by.

    The host writes each row's allowed tokens to ``pinned_bitmask`` and sets
    the flag ``wait_payload`` points at; the graph waits on the flag, then
    copies the bitmask to ``device_bitmask_scratch``.
    """

    kind = "token_bitmask"


class RecurrentStateTail(
    GraphGroup[str, tuple[tuple[TensorType | BufferType, ...], ...]]
):
    """A recurrent-state cache's inputs in a fused speculative graph's tail.

    ``pool`` holds each region's pool and ``live_row_ids`` each region's
    ``[num_layers, batch_size]`` rows, every region with one entry per
    device. A serving engine binds them as it binds a decoder's
    ``recurrent_state`` leaves.
    """

    kind = "recurrent_state_tail"


class SpecDecodeOutputs(GraphGroup[str, _TensorT | tuple[_TensorT, ...]]):
    """A speculative-decoding step's verdicts and the next step's drafts.

    How many drafts each row accepted, the token that follows them, the next
    step's drafts, and any distributions the draft sampled those from.
    """

    kind = "spec_decode_outputs"


class DecoderGraphInputs(GraphGroup[str, Any]):
    """A decoder graph's named inputs, keyed by the decoder's input fields.

    A decoder registered by role takes its fields in an order it may choose,
    so a serving engine finds each one by name. A field that is one input is
    a direct leaf, such as ``tokens``; the rest are groups or trees of them,
    such as the ``kv`` tree.
    """

    kind = "decoder_inputs"


_G = TypeVar("_G", Logits, HiddenStates, ConvStates, SparseIndices)


class DecoderOutputs(
    GraphGroup[
        str,
        Logits[_TensorT] | HiddenStates[_TensorT] | ConvStates | SparseIndices,
    ]
):
    """A decoder graph's named outputs.

    Holds its ``logits``, any ``hidden_states``, and any state it writes
    back.
    """

    kind = "decoder_outputs"

    def group(self, name: str, cls: type[_G]) -> _G:
        """Returns the group named ``name``, checked to be a ``cls``."""
        group = self[name]
        assert isinstance(group, cls), f"{name} is a {type(group).__name__}"
        return group

    @property
    def logits(self) -> Logits[_TensorT]:
        """The decoder's logits."""
        logits = self["logits"]
        assert isinstance(logits, Logits), (
            f"logits is a {type(logits).__name__}"
        )
        return logits

    @property
    def hidden_states(self) -> HiddenStates[_TensorT]:
        """The decoder's hidden states."""
        states = self["hidden_states"]
        assert isinstance(states, HiddenStates), (
            f"hidden_states is a {type(states).__name__}"
        )
        return states


def group_fields(groups: Mapping[str, Mapping[str, _V]]) -> dict[str, _V]:
    """Returns the fields of a tree of groups under their own names.

    A graph whose inputs are grouped by kind takes them as arguments named
    by field, so its ``forward`` looks each one up here.
    """
    fields: dict[str, _V] = {}
    for group in groups.values():
        fields.update(group)
    return fields
