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

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BlockMetadata(_message.Message):
    __slots__ = ("seq_hash", "agent_id", "device_id", "offset", "length")
    SEQ_HASH_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    seq_hash: int
    agent_id: int
    device_id: int
    offset: int
    length: int
    def __init__(self, seq_hash: _Optional[int] = ..., agent_id: _Optional[int] = ..., device_id: _Optional[int] = ..., offset: _Optional[int] = ..., length: _Optional[int] = ...) -> None: ...

class ExchangeMetadataRequest(_message.Message):
    __slots__ = ("agent_metadata", "bytes_per_page")
    AGENT_METADATA_FIELD_NUMBER: _ClassVar[int]
    BYTES_PER_PAGE_FIELD_NUMBER: _ClassVar[int]
    agent_metadata: bytes
    bytes_per_page: int
    def __init__(self, agent_metadata: _Optional[bytes] = ..., bytes_per_page: _Optional[int] = ...) -> None: ...

class ExchangeMetadataResponse(_message.Message):
    __slots__ = ("agent_metadata", "agent_name", "bytes_per_page", "total_num_pages", "base_addr")
    AGENT_METADATA_FIELD_NUMBER: _ClassVar[int]
    AGENT_NAME_FIELD_NUMBER: _ClassVar[int]
    BYTES_PER_PAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_NUM_PAGES_FIELD_NUMBER: _ClassVar[int]
    BASE_ADDR_FIELD_NUMBER: _ClassVar[int]
    agent_metadata: bytes
    agent_name: str
    bytes_per_page: int
    total_num_pages: int
    base_addr: int
    def __init__(self, agent_metadata: _Optional[bytes] = ..., agent_name: _Optional[str] = ..., bytes_per_page: _Optional[int] = ..., total_num_pages: _Optional[int] = ..., base_addr: _Optional[int] = ...) -> None: ...

class BlockSequence(_message.Message):
    __slots__ = ("parent_seq_hash", "seq_hashes")
    PARENT_SEQ_HASH_FIELD_NUMBER: _ClassVar[int]
    SEQ_HASHES_FIELD_NUMBER: _ClassVar[int]
    parent_seq_hash: int
    seq_hashes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, parent_seq_hash: _Optional[int] = ..., seq_hashes: _Optional[_Iterable[int]] = ...) -> None: ...

class AcquireBlocksRequest(_message.Message):
    __slots__ = ("sequences",)
    SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    sequences: _containers.RepeatedCompositeFieldContainer[BlockSequence]
    def __init__(self, sequences: _Optional[_Iterable[_Union[BlockSequence, _Mapping]]] = ...) -> None: ...

class AcquiredBlock(_message.Message):
    __slots__ = ("metadata", "newly_acquired")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NEWLY_ACQUIRED_FIELD_NUMBER: _ClassVar[int]
    metadata: BlockMetadata
    newly_acquired: bool
    def __init__(self, metadata: _Optional[_Union[BlockMetadata, _Mapping]] = ..., newly_acquired: _Optional[bool] = ...) -> None: ...

class AcquireBlocksResponse(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[AcquiredBlock]
    def __init__(self, blocks: _Optional[_Iterable[_Union[AcquiredBlock, _Mapping]]] = ...) -> None: ...

class RegisterBlocksRequest(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[BlockMetadata]
    def __init__(self, blocks: _Optional[_Iterable[_Union[BlockMetadata, _Mapping]]] = ...) -> None: ...

class RegisterBlocksResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReleaseBlocksRequest(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[BlockMetadata]
    def __init__(self, blocks: _Optional[_Iterable[_Union[BlockMetadata, _Mapping]]] = ...) -> None: ...

class ReleaseBlocksResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ReadBlocksRequest(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[BlockMetadata]
    def __init__(self, blocks: _Optional[_Iterable[_Union[BlockMetadata, _Mapping]]] = ...) -> None: ...

class ReadBlocksResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DecrementBlocksRequest(_message.Message):
    __slots__ = ("blocks",)
    BLOCKS_FIELD_NUMBER: _ClassVar[int]
    blocks: _containers.RepeatedCompositeFieldContainer[BlockMetadata]
    def __init__(self, blocks: _Optional[_Iterable[_Union[BlockMetadata, _Mapping]]] = ...) -> None: ...

class DecrementBlocksResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class NotReadyResponse(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class RpcRequest(_message.Message):
    __slots__ = ("acquire_blocks", "register_blocks", "release_blocks", "read_blocks", "decrement_blocks", "exchange_metadata")
    ACQUIRE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REGISTER_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    RELEASE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    READ_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    DECREMENT_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_METADATA_FIELD_NUMBER: _ClassVar[int]
    acquire_blocks: AcquireBlocksRequest
    register_blocks: RegisterBlocksRequest
    release_blocks: ReleaseBlocksRequest
    read_blocks: ReadBlocksRequest
    decrement_blocks: DecrementBlocksRequest
    exchange_metadata: ExchangeMetadataRequest
    def __init__(self, acquire_blocks: _Optional[_Union[AcquireBlocksRequest, _Mapping]] = ..., register_blocks: _Optional[_Union[RegisterBlocksRequest, _Mapping]] = ..., release_blocks: _Optional[_Union[ReleaseBlocksRequest, _Mapping]] = ..., read_blocks: _Optional[_Union[ReadBlocksRequest, _Mapping]] = ..., decrement_blocks: _Optional[_Union[DecrementBlocksRequest, _Mapping]] = ..., exchange_metadata: _Optional[_Union[ExchangeMetadataRequest, _Mapping]] = ...) -> None: ...

class RpcResponse(_message.Message):
    __slots__ = ("acquire_blocks", "register_blocks", "release_blocks", "read_blocks", "decrement_blocks", "exchange_metadata", "error", "not_ready")
    ACQUIRE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REGISTER_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    RELEASE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    READ_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    DECREMENT_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    EXCHANGE_METADATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    NOT_READY_FIELD_NUMBER: _ClassVar[int]
    acquire_blocks: AcquireBlocksResponse
    register_blocks: RegisterBlocksResponse
    release_blocks: ReleaseBlocksResponse
    read_blocks: ReadBlocksResponse
    decrement_blocks: DecrementBlocksResponse
    exchange_metadata: ExchangeMetadataResponse
    error: ErrorResponse
    not_ready: NotReadyResponse
    def __init__(self, acquire_blocks: _Optional[_Union[AcquireBlocksResponse, _Mapping]] = ..., register_blocks: _Optional[_Union[RegisterBlocksResponse, _Mapping]] = ..., release_blocks: _Optional[_Union[ReleaseBlocksResponse, _Mapping]] = ..., read_blocks: _Optional[_Union[ReadBlocksResponse, _Mapping]] = ..., decrement_blocks: _Optional[_Union[DecrementBlocksResponse, _Mapping]] = ..., exchange_metadata: _Optional[_Union[ExchangeMetadataResponse, _Mapping]] = ..., error: _Optional[_Union[ErrorResponse, _Mapping]] = ..., not_ready: _Optional[_Union[NotReadyResponse, _Mapping]] = ...) -> None: ...
