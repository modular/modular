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
from collections.abc import Mapping
from typing import ClassVar, Literal, TypeVar

from max import tree
from max.graph import BufferType, TensorType, TensorValue
from typing_extensions import Self

__all__ = [
    "BatchContext",
    "ConvStates",
    "DecoderOutputs",
    "EPLBProfile",
    "EPPayload",
    "GraphGroup",
    "HiddenStates",
    "Logits",
    "LogitsName",
    "MRoPEPositions",
    "RaggedBatch",
    "SparseIndices",
    "TPSignals",
    "VisionMerge",
]

_K = TypeVar("_K", bound=str)
_V = TypeVar("_V")


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


class TPSignals(GraphGroup[str, tuple[BufferType, ...]]):
    """The allreduce signal buffers, one per device."""

    kind = "tp_signals"


class VisionMerge(
    GraphGroup[str, tuple[TensorType, ...] | tuple[tuple[TensorType, ...], ...]]
):
    """Image embeddings per device and the token positions they replace.

    A model that injects image features into later layers, such as
    Qwen3-VL's deepstack, holds one per-device tuple per layer.
    """

    kind = "vision_merge"


class MRoPEPositions(GraphGroup[str, TensorType]):
    """The 3D position ids of an M-RoPE model."""

    kind = "mrope_positions"


class BatchContext(GraphGroup[str, tuple[TensorType, ...]]):
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


class Logits(GraphGroup[LogitsName, TensorValue]):
    """The decoder's logits: the next token's, and any others requested.

    Which are present depends on the decoder's ``return_logits``, and a
    caller that set it reads the ones it asked for.
    """

    kind = "logits"

    def __init__(
        self,
        *,
        next_token_logits: TensorValue | None = None,
        logits: TensorValue | None = None,
        logit_offsets: TensorValue | None = None,
    ) -> None:
        super().__init__()
        if next_token_logits is not None:
            self["next_token_logits"] = next_token_logits
        if logits is not None:
            self["logits"] = logits
        if logit_offsets is not None:
            self["logit_offsets"] = logit_offsets


class HiddenStates(GraphGroup[str, tuple[TensorValue, ...]]):
    """Hidden states a decoder returns for a drafter, a capture, or the next
    pipeline stage, keyed by the one mode it ran in."""

    kind = "hidden_states"

    def tensors(self) -> tuple[TensorValue, ...]:
        """Returns the hidden states, whichever mode produced them."""
        (states,) = self.values()
        return states


class ConvStates(GraphGroup[str, tuple[TensorValue, ...]]):
    """The short-convolution states a decoder writes back for the next step."""

    kind = "conv_states"


class SparseIndices(GraphGroup[str, tuple[TensorValue, ...]]):
    """The rows a sparse-attention indexer selected, for a drafter to reuse."""

    kind = "sparse_indices"


_G = TypeVar("_G", Logits, HiddenStates, ConvStates, SparseIndices)


class DecoderOutputs(
    GraphGroup[str, Logits | HiddenStates | ConvStates | SparseIndices]
):
    """A decoder graph's named outputs: its ``logits``, any
    ``hidden_states``, and any state it writes back."""

    kind = "decoder_outputs"

    def group(self, name: str, cls: type[_G]) -> _G:
        """Returns the group named ``name``, checked to be a ``cls``."""
        group = self[name]
        assert isinstance(group, cls), f"{name} is a {type(group).__name__}"
        return group

    @property
    def logits(self) -> Logits:
        """The decoder's logits."""
        return self.group("logits", Logits)

    @property
    def hidden_states(self) -> HiddenStates:
        """The decoder's hidden states."""
        return self.group("hidden_states", HiddenStates)
