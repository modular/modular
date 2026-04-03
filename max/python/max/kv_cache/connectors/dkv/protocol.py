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

"""Protobuf serialization helpers for the dKV client.

Provides Pythonic types and pre-serialized request/response helpers that
wrap the raw protobuf API.
"""

from __future__ import annotations

import logging
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from enum import Enum

from google.protobuf.message import DecodeError
from typing_extensions import Self

from .client_api_pb2 import (  # type: ignore[attr-defined]
    AcquireBlocksRequest,
    BlockMetadata,
    DecrementBlocksRequest,
    ExchangeMetadataRequest,
    ReadBlocksRequest,
    RegisterBlocksRequest,
    ReleaseBlocksRequest,
    RpcRequest,
    RpcResponse,
)

logger = logging.getLogger(__name__)

_UINT64_MASK = (1 << 64) - 1


def _to_uint64(val: int) -> int:
    """Reinterpret a signed Python int as an unsigned 64-bit value.

    MAX's block hasher returns signed ``Py_ssize_t`` hashes, but the dKV
    protobuf schema uses ``uint64`` for ``seq_hash`` fields.  This
    converts at the protocol boundary so both sides agree on the bit
    pattern.
    """
    return val & _UINT64_MASK


class DKVError(Exception):
    """Base class for dKV client failures."""


class DKVDecodeError(DKVError):
    """Raised when response bytes cannot be parsed as ``RpcResponse`` protobuf."""


class DKVProtocolError(DKVError):
    """Raised when the response envelope is empty or the wrong oneof kind."""

    def __init__(
        self,
        message: str,
        *,
        expected_kind: str | None = None,
        actual_kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_kind = expected_kind
        self.actual_kind = actual_kind


class DKVServerError(DKVError):
    """Raised when the server sets ``RpcResponse.error``."""


class RpcResponseKind(str, Enum):
    """Valid ``RpcResponse.response`` oneof branches (must match ``api.proto``)."""

    ACQUIRE_BLOCKS = "acquire_blocks"
    REGISTER_BLOCKS = "register_blocks"
    RELEASE_BLOCKS = "release_blocks"
    READ_BLOCKS = "read_blocks"
    DECREMENT_BLOCKS = "decrement_blocks"
    EXCHANGE_METADATA = "exchange_metadata"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class BlockDescriptor:
    """Describes the location of a KV cache block in device memory.

    Wraps the protobuf ``BlockMetadata`` message with an immutable, hashable
    Python type.
    """

    seq_hash: int
    """The rolling hash of the prefix up to this block."""

    agent_id: int
    """The NIXL agent identifier."""

    device_id: int
    """The GPU device index (0-indexed)."""

    offset: int
    """The byte offset in device memory."""

    length: int
    """The block size in bytes."""

    @classmethod
    def from_proto(cls, pb: BlockMetadata) -> Self:
        """Creates a descriptor from a protobuf ``BlockMetadata`` message."""
        return cls(
            seq_hash=pb.seq_hash,
            agent_id=pb.agent_id,
            device_id=pb.device_id,
            offset=pb.offset,
            length=pb.length,
        )

    def to_proto(self) -> BlockMetadata:
        """Converts this descriptor to a protobuf ``BlockMetadata`` message."""
        return BlockMetadata(
            seq_hash=_to_uint64(self.seq_hash),
            agent_id=self.agent_id,
            device_id=self.device_id,
            offset=self.offset,
            length=self.length,
        )


def _fill_blocks(
    repeated_field: MutableSequence[BlockMetadata],
    blocks: Sequence[BlockDescriptor],
) -> None:
    """Populates a protobuf repeated ``BlockMetadata`` field from descriptors."""
    for block in blocks:
        repeated_field.append(block.to_proto())


# ---------------------------------------------------------------------------
# Request builders — each returns pre-serialized bytes for socket.send()
# ---------------------------------------------------------------------------


def build_acquire_request(
    seq_hashes: Sequence[int],
    parent_seq_hash: int = 0,
) -> bytes:
    """Builds a serialized ``RpcRequest`` for acquiring blocks.

    Args:
        seq_hashes: The ordered sequence hashes for the blocks to acquire.
        parent_seq_hash: The optional parent sequence hash when extending
            an existing sequence. Defaults to ``""``.

    Returns:
        The serialized request bytes.

    Raises:
        ValueError: If ``seq_hashes`` is empty.
    """
    if not seq_hashes:
        raise ValueError("seq_hashes must not be empty")
    inner = AcquireBlocksRequest(parent_seq_hash=parent_seq_hash)
    inner.seq_hashes.extend(_to_uint64(h) for h in seq_hashes)
    req = RpcRequest(acquire_blocks=inner)
    return req.SerializeToString()


def build_register_request(blocks: Sequence[BlockDescriptor]) -> bytes:
    """Builds a serialized ``RpcRequest`` for registering blocks.

    Args:
        blocks: The blocks to register.

    Returns:
        The serialized request bytes.

    Raises:
        ValueError: If ``blocks`` is empty.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")
    inner = RegisterBlocksRequest()
    _fill_blocks(inner.blocks, blocks)
    req = RpcRequest(register_blocks=inner)
    return req.SerializeToString()


def build_release_request(blocks: Sequence[BlockDescriptor]) -> bytes:
    """Builds a serialized ``RpcRequest`` for releasing blocks.

    Releases blocks from FILLING back to FREE state. Only valid for
    blocks that have been acquired but not yet registered.

    Args:
        blocks: The blocks to release.

    Returns:
        The serialized request bytes.

    Raises:
        ValueError: If ``blocks`` is empty.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")
    inner = ReleaseBlocksRequest()
    _fill_blocks(inner.blocks, blocks)
    req = RpcRequest(release_blocks=inner)
    return req.SerializeToString()


def build_read_request(blocks: Sequence[BlockDescriptor]) -> bytes:
    """Builds a serialized ``RpcRequest`` for reading blocks.

    Args:
        blocks: The blocks to read.

    Returns:
        The serialized request bytes.

    Raises:
        ValueError: If ``blocks`` is empty.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")
    inner = ReadBlocksRequest()
    _fill_blocks(inner.blocks, blocks)
    req = RpcRequest(read_blocks=inner)
    return req.SerializeToString()


def build_decrement_request(blocks: Sequence[BlockDescriptor]) -> bytes:
    """Builds a serialized ``RpcRequest`` for decrementing block ref counts.

    Args:
        blocks: The blocks to decrement.

    Returns:
        The serialized request bytes.

    Raises:
        ValueError: If ``blocks`` is empty.
    """
    if not blocks:
        raise ValueError("blocks must not be empty")
    inner = DecrementBlocksRequest()
    _fill_blocks(inner.blocks, blocks)
    req = RpcRequest(decrement_blocks=inner)
    return req.SerializeToString()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(data: bytes) -> RpcResponse:
    """Parses a serialized ``RpcResponse`` and checks for generic errors.

    Args:
        data: The raw bytes received from the dKV server.

    Returns:
        The parsed response.

    Raises:
        DKVDecodeError: If ``data`` is not valid protobuf.
        DKVServerError: If the server returned ``RpcResponse.error``.
        DKVProtocolError: If the response oneof is unset.
    """
    resp = RpcResponse()
    try:
        resp.ParseFromString(data)
    except DecodeError as e:
        logger.debug(
            "dKV response decode failed: payload_len=%d",
            len(data),
            exc_info=True,
        )
        raise DKVDecodeError(
            "Invalid response from dKV server: failed to decode protobuf"
        ) from e

    oneof = resp.WhichOneof("response")

    if oneof == RpcResponseKind.ERROR:
        logger.debug(
            "dKV server returned ErrorResponse (payload_len=%d)",
            len(data),
        )
        raise DKVServerError(resp.error.message)

    if oneof is None:
        logger.debug(
            "dKV server returned empty RpcResponse (payload_len=%d)",
            len(data),
        )
        raise DKVProtocolError("dKV server returned empty response")

    return resp


def _require_response_kind(
    resp: RpcResponse,
    expected_kind: RpcResponseKind,
    *,
    payload_len: int,
) -> None:
    """Ensures the response oneof matches the expected operation kind."""
    oneof = resp.WhichOneof("response")
    if oneof != expected_kind:
        logger.debug(
            "dKV response kind mismatch: payload_len=%d expected=%s got=%s",
            payload_len,
            expected_kind,
            oneof,
        )
        raise DKVProtocolError(
            "Unexpected response type from dKV server."
            f" Expected: {expected_kind.value}, got: {oneof}",
            expected_kind=expected_kind.value,
            actual_kind=oneof,
        )


def parse_acquire_response(data: bytes) -> list[BlockDescriptor]:
    """Parses and validates an acquire-blocks response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp, RpcResponseKind.ACQUIRE_BLOCKS, payload_len=len(data)
    )
    return [BlockDescriptor.from_proto(pb) for pb in resp.acquire_blocks.blocks]


def parse_register_response(data: bytes) -> None:
    """Parses and validates a register-blocks response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp, RpcResponseKind.REGISTER_BLOCKS, payload_len=len(data)
    )


def parse_release_response(data: bytes) -> None:
    """Parses and validates a release-blocks response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp, RpcResponseKind.RELEASE_BLOCKS, payload_len=len(data)
    )


def parse_read_response(data: bytes) -> None:
    """Parses and validates a read-blocks response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp, RpcResponseKind.READ_BLOCKS, payload_len=len(data)
    )


def parse_decrement_response(data: bytes) -> None:
    """Parses and validates a decrement-blocks response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp, RpcResponseKind.DECREMENT_BLOCKS, payload_len=len(data)
    )


# ---------------------------------------------------------------------------
# Transfer metadata types and helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExchangeMetadataResult:
    """Result of the ExchangeMetadata RPC.

    Contains the remote agent's NIXL metadata and the confirmed slab
    geometry.
    """

    agent_metadata: bytes
    agent_name: str
    bytes_per_page: int
    total_num_pages: int
    base_addr: int


def build_exchange_metadata_request(
    agent_metadata: bytes,
    bytes_per_page: int = 0,
) -> bytes:
    """Builds a serialized ``RpcRequest`` for exchanging agent metadata.

    Args:
        agent_metadata: Opaque NIXL agent metadata blob.
        bytes_per_page: Engine's page size in bytes for slab configuration.
            0 means use dKV's CLI-configured block size (legacy path).
    """
    inner = ExchangeMetadataRequest(
        agent_metadata=agent_metadata,
        bytes_per_page=bytes_per_page,
    )
    req = RpcRequest(exchange_metadata=inner)
    return req.SerializeToString()


def parse_exchange_metadata_response(data: bytes) -> ExchangeMetadataResult:
    """Parses and validates an exchange-metadata response payload."""
    resp = _parse_response(data)
    _require_response_kind(
        resp,
        RpcResponseKind.EXCHANGE_METADATA,
        payload_len=len(data),
    )
    meta = resp.exchange_metadata
    return ExchangeMetadataResult(
        agent_metadata=meta.agent_metadata,
        agent_name=meta.agent_name,
        bytes_per_page=meta.bytes_per_page,
        total_num_pages=meta.total_num_pages,
        base_addr=meta.base_addr,
    )
