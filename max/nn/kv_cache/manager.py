# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Abstract base class for KVCacheManager for KV Cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, overload

from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from typing_extensions import TypeGuard

from .cache_params import KVCacheParams
from .context import KVCacheAwareContext

_T = TypeVar("_T")


def _is_sequence_of(x: Any, ty: type[_T]) -> TypeGuard[Sequence[_T]]:
    return isinstance(x, Sequence) and all(isinstance(item, ty) for item in x)


@dataclass
class KVCacheInputs:
    """
    A base class that holds KV cache related (Tensor) inputs.

    It is meant to be subclassed by concrete KV cache input types.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class RaggedKVCacheInputs(KVCacheInputs):
            blocks: Tensor
            cache_lengths: Tensor
            lookup_table: Tensor
            max_lengths: Tensor
    """

    def __iter__(self) -> Iterator[Tensor]:
        """Iterates through each Type in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, KVCacheInputs):
                yield from value
            elif _is_sequence_of(value, KVCacheInputs):
                for item in value:
                    yield from item
            else:
                yield cast(Tensor, value)

    @overload
    def __getitem__(self, index: int) -> Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Tensor]: ...

    def __getitem__(self, index: Any) -> Any:
        return list(self)[index]

    def __len__(self) -> int:
        count = 0
        # Iterate over all fields in the dataclass. If we run into a sequence of
        # KVCacheInputs, we expand and recursively call `len` on the KVCacheInputs
        # elements.
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if _is_sequence_of(value, KVCacheInputs):
                count += sum(len(x) for x in value)
            else:
                count += 1
        return count


@dataclass
class PaddedKVCacheInputs(KVCacheInputs):
    """
    ``PaddedKVCacheInputs`` is a class that holds the inputs for
    KV cache when used together with padded tensors.
    """

    k_cache: Tensor
    v_cache: Tensor
    start_pos: Tensor
    null_op: Tensor


@dataclass
class RaggedKVCacheInputs(KVCacheInputs):
    """
    ``RaggedKVCacheInputs`` is a class that holds the inputs for
    KV cache when used together with ragged tensors.
    """

    blocks: Tensor
    cache_lengths: Tensor
    lookup_table: Tensor
    max_lengths: Tensor


@dataclass
class KVCacheInputsSequence(KVCacheInputs):
    """
    ``KVCacheInputsSequence`` is a sequence of :obj:`KVCacheInputs`.

    It is primarily used in our multistep execution to represent batched
    KVCacheInputs.
    """

    kv_cache_inputs: Sequence[KVCacheInputs]


@dataclass
class KVCacheInputSymbols:
    """
    Base class for input symbols for KV cache managers.

    The derived class is responsible for defining the input symbols for the
    specific KV cache manager.
    For example, here's a derived class for a text KV cache manager:

    .. code-block:: python

        @dataclass
        class ContinuousBatchingKVCacheInputSymbols(KVCacheInputSymbols):
            kv_blocks: TensorType
            cache_lengths: TensorType
            lookup_table: TensorType
            max_lengths: TensorType
    """

    def __iter__(self) -> Iterator[Any]:
        """Iterates through each Type in order."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, KVCacheInputSymbols):
                yield from value
            else:
                yield value

    def __getitem__(self, index) -> Any:
        return list(self)[index]


T = TypeVar("T", bound=KVCacheAwareContext)


class KVCacheManager(ABC, Generic[T]):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
        is_ragged: bool = False,
    ) -> None:
        self.params = params
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.devices = devices
        self.session = session

        # Attributes for managing available slots.
        self.available = set(range(self.max_batch_size))
        self.active: set[int] = set()

        self.is_ragged = is_ragged
        increment_cache_lengths_graph = (
            self._create_increment_cache_lengths_graph()
        )
        self.increment_cache_lengths_model = session.load(
            increment_cache_lengths_graph
        )

    @classmethod
    @abstractmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Returns the estimated total memory usage of the kv cache."""
        ...

    @classmethod
    @abstractmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        """Returns the estimated optimal batch size for the kv cache."""
        ...

    @abstractmethod
    def fetch(
        self,
        batch: list[T],
        num_steps: int = 1,
    ) -> list[KVCacheInputs]:
        """Returns blocks and other inputs to kv cache kernel for given
        sequence ids and prompts."""
        ...

    @abstractmethod
    def input_symbols(
        self,
    ) -> Sequence[KVCacheInputSymbols]:
        """Returns the input symbols for the kv cache manager."""
        ...

    def claim(self, n: int) -> list[int]:
        """Claims ``n`` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the :obj:`ContinuousBatchingKVCacheCollection`
        for those sequences.
        """
        # TODO we should remove this interface and just use external_claim.
        seq_ids = []

        for _ in range(n):
            seq_id = self.available.pop()
            self.active.add(seq_id)
            seq_ids.append(seq_id)

        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        for seq_id in seq_ids:
            if seq_id in self.active:
                raise ValueError(
                    f"Attempted to claim {seq_id} but it is already in active set"
                )

            self.available.remove(seq_id)
            self.active.add(seq_id)

    def step(self, batch: list[T]) -> None:
        """Commit the new tokens into the prefix cache.

        This is a no-op if prefix caching is disabled."""
        ...

    def release(self, seq_id: int) -> None:
        """Release :obj:`seq_id` provided, marking this sequence as complete.
        This returns the :obj:`seq_id` back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        if seq_id not in self.active:
            raise ValueError(
                f"Attempted to release {seq_id} but it is not in active set"
            )

        self.active.remove(seq_id)
        self.available.add(seq_id)

    def contains(self, seq_id: int) -> bool:
        return seq_id not in self.slots_remaining

    @property
    def slots_remaining(self) -> set[int]:
        """The outstanding cache slots available."""
        return self.available

    def num_kv_inputs(self) -> int:
        """Returns the default number of KV cache inputs for KV managers.

        Subclasses with a different number of KV cache inputs should override
        this method and :obj:`increment_cache_lengths`.
        """
        return 4

    def increment_cache_lengths(
        self,
        kv_cache_inputs: list[RaggedKVCacheInputs] | list[PaddedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> list[RaggedKVCacheInputs] | list[PaddedKVCacheInputs]:
        """
        Prepare the inputs for a multistep execution, generally by incrementing
        the cache lengths. This should not require a device synchronization,
        as this would defeat the purpose of multistep execution.

        This should also not update the cache lengths in our manager, this batch is
        still considered in-progress.
        """
        if self.is_ragged:
            return self._increment_cache_lengths_ragged(
                kv_cache_inputs=cast(
                    list[RaggedKVCacheInputs], kv_cache_inputs
                ),
                prev_model_inputs=prev_model_inputs,
            )

        return self._increment_cache_lengths_padded(
            kv_cache_inputs=cast(list[PaddedKVCacheInputs], kv_cache_inputs),
            prev_model_inputs=prev_model_inputs,
        )

    def _increment_cache_lengths_ragged(
        self,
        kv_cache_inputs: list[RaggedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> list[RaggedKVCacheInputs]:
        """Prepares cache inputs for the next token in multistep execution.

        Updates the cache lengths for the next inference step without requiring device
        synchronization or memory copies. This is crucial for maintaining performance
        during multi-token generation.

        Args:
            kv_cache_inputs: Current cache state tuples (blocks, lengths, lookup, max_lengths)
            prev_model_inputs: Previous model inputs including row offsets

        Returns:
            Updated cache input tuples with incremented lengths.
        """
        blocks = [kv_cache_inputs[i].blocks for i in range(len(self.devices))]
        cache_lengths = [
            kv_cache_inputs[i].cache_lengths for i in range(len(self.devices))
        ]
        lookup_table = [
            kv_cache_inputs[i].lookup_table for i in range(len(self.devices))
        ]

        # max_lengths is host allocated and the same across all devices.
        max_lengths = kv_cache_inputs[0].max_lengths

        # Update the cache_lengths of our batch by the previous sequence length.
        updated_cache_lengths = self.increment_cache_lengths_model.execute(
            prev_model_inputs.input_row_offsets, *cache_lengths
        )

        # Advance to the next step of the max_lengths tensor.
        updated_max_lengths = max_lengths[1:, :]

        # Return our updated batch.
        for i in range(len(self.devices)):
            updated_cache_length = updated_cache_lengths[i]
            assert isinstance(updated_cache_length, Tensor)
            kv_cache_inputs[i] = RaggedKVCacheInputs(
                blocks=blocks[i],
                cache_lengths=updated_cache_length,
                lookup_table=lookup_table[i],
                max_lengths=updated_max_lengths,
            )
        return kv_cache_inputs

    def _increment_cache_lengths_padded(
        self,
        kv_cache_inputs: list[PaddedKVCacheInputs],
        prev_model_inputs: Any,
    ) -> list[PaddedKVCacheInputs]:
        """
        Prepare the inputs for a multistep execution, generally by incrementing
        the cache lengths. This should not require a device synchronization,
        as this would defeat the purpose of multistep execution.

        This should also not update the cache lengths in our manager, this batch is
        still considered in-progress.
        """
        assert len(kv_cache_inputs) == 1
        curr_kv_cache_inputs = kv_cache_inputs[0]

        new_start_pos = self.increment_cache_lengths_model(
            curr_kv_cache_inputs.start_pos, prev_model_inputs.tokens
        )[0]
        assert isinstance(new_start_pos, Tensor)
        return [
            PaddedKVCacheInputs(
                k_cache=curr_kv_cache_inputs.k_cache,
                v_cache=curr_kv_cache_inputs.v_cache,
                start_pos=new_start_pos,
                null_op=new_start_pos,
            )
        ]

    def _create_increment_cache_lengths_graph(self) -> Graph:
        """Constructs a graph to increment the cache_lengths argument during multi-step inference.

        It's imperative that this operation occurs entirely on GPU,
        otherwise we'll synchronize across devices and incur a latency penalty.
        """
        if self.is_ragged:
            return self._create_ragged_increment_cache_lengths_graph()

        return self._create_padded_increment_cache_lengths_graph()

    def _create_padded_increment_cache_lengths_graph(self) -> Graph:
        start_pos_type = TensorType(
            DType.int64, shape=[], device=DeviceRef.CPU()
        )
        tokens_type = TensorType(
            DType.int64, shape=["batch_size", "seq_len"], device=DeviceRef.CPU()
        )
        with Graph(
            "update_start_pos", input_types=[start_pos_type, tokens_type]
        ) as graph:
            start_pos, tokens = graph.inputs
            assert isinstance(start_pos, TensorValue)
            assert isinstance(tokens, TensorValue)
            graph.output(start_pos + tokens.shape[1])

        return graph

    def _create_ragged_increment_cache_lengths_graph(self) -> Graph:
        input_symbols = self.input_symbols()
        cache_lengths_types = [
            input_symbols[i][1] for i in range(len(self.devices))
        ]

        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef(self.devices[0].label, self.devices[0].id),
        )

        with Graph(
            "update_cache_lengths",
            input_types=[input_row_offsets_type, *cache_lengths_types],
        ) as graph:
            inp_row_offset, *cache_lengths = graph.inputs
            assert isinstance(inp_row_offset, TensorValue)
            # broadcast the inp_row_offset to all devices (naive)
            # get rid of this if statement after #51465 merges
            if len(self.devices) > 1:
                input_row_offsets = [
                    inp_row_offset.to(DeviceRef(d.label, d.id))
                    for d in self.devices
                ]
            else:
                input_row_offsets = [inp_row_offset]
            outputs = []
            for i in range(len(self.devices)):
                cache_length = cache_lengths[i]
                assert isinstance(cache_length, TensorValue)
                right_slice = input_row_offsets[i][1:].rebind(
                    cache_length.shape
                )
                left_slice = input_row_offsets[i][
                    : input_row_offsets[i].shape[0] - 1
                ].rebind(cache_length.shape)
                increment_amount = right_slice - left_slice
                outputs.append(cache_length + increment_amount)
            graph.output(*outputs)

        return graph
