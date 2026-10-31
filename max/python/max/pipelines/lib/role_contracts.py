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
"""The inputs and outputs of each role a registered module fills.

An architecture registers a module class per role in
:attr:`~max.pipelines.lib.arch_lookup.SupportedArchitecture.modules`: a
:class:`~max.nn.Module` (ModuleV3) or a graph-API
:class:`~max.nn.legacy.Module`. The framework builds the role's graph and
batches from the contract here, so the module's ``forward`` (``__call__`` for
a graph-API module) takes the role's inputs and returns its outputs. A
ModuleV3 module sees :class:`~max.tensor.Tensor` fields, and a graph-API
module :class:`~max.graph.TensorValue` fields.
"""

from __future__ import annotations

from typing import Any, Generic

from max import tree
from max.graph import BufferValue
from max.tensor import Tensor
from typing_extensions import TypeVar

from .interfaces.pipeline_model import ModuleV3Outputs

__all__ = [
    "DecoderInputs",
    "DecoderOutputs",
]


TensorT = TypeVar("TensorT", default=Tensor)
"""The tensor type a decoder's fields hold: a ModuleV3
:class:`~max.tensor.Tensor` or a graph-API :class:`~max.graph.TensorValue`.
"""


@tree.dataclass(frozen=True, kw_only=True)
class DecoderInputs(Generic[TensorT]):
    """The inputs of a ``DECODER`` module's ``forward``.

    The decoder graph takes the fields in this order, which is the canonical
    order of the design's role contract. A field that is ``None`` is not a
    graph input: the framework sets each optional field only when the
    deployment needs it.
    """

    tokens: TensorT
    """The batch's ragged tokens, shape ``[total_seq_len]``."""

    input_row_offsets: TensorT
    """Each sequence's start in :attr:`tokens`, shape ``[batch_size + 1]``."""

    host_input_row_offsets: TensorT | None = None
    """A CPU copy of :attr:`input_row_offsets`, under data parallelism.

    It bounds each replica's slice of the batch without a device-to-host
    copy inside the graph.
    """

    return_n_logits: TensorT
    """How many logits per sequence to return, a CPU tensor of shape ``[1]``.
    """

    data_parallel_splits: TensorT | None = None
    """Each replica's first sequence, a CPU ``int64`` tensor of shape
    ``[data_parallel_degree + 1]``, under data parallelism.

    Replica ``i`` serves sequences ``data_parallel_splits[i]`` up to
    ``data_parallel_splits[i + 1]``.
    """

    signals: tuple[BufferValue, ...] | None = None
    """The allreduce signal buffers, one per device, of a graph-API decoder.

    A multi-device graph-API decoder takes them, and so does one that sets
    ``always_signal_buffers``. A ModuleV3 decoder never sees them: its
    compiled graph adds its own.
    """

    kv: Any
    """The KV cache inputs, in the tree the config's KV params declare."""

    lora: tuple[TensorT, ...] | None = None
    """The LoRA routing and adapter inputs, when serving LoRA adapters.

    The framework binds them into the decoder's LoRA layers before it calls
    the decoder, so a decoder doesn't read them.
    """


DecoderOutputs = ModuleV3Outputs
"""The outputs of a ``DECODER`` module's ``forward``.

Generic over the same tensor type as :class:`DecoderInputs`.
"""
