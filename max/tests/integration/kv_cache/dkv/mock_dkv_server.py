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

"""Mock dKV server for testing the dKV client."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import zmq
from max.kv_cache.connectors.dkv.client_api_pb2 import (
    AcquireBlocksRequest,
    AcquireBlocksResponse,
    AcquiredBlock,
    BlockMetadata,
    DecrementBlocksResponse,
    ErrorResponse,
    ExchangeMetadataRequest,
    ExchangeMetadataResponse,
    ReadBlocksResponse,
    RegisterBlocksResponse,
    ReleaseBlocksResponse,
    RpcRequest,
    RpcResponse,
)
from max.kv_cache.connectors.dkv.protocol import BlockDescriptor

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_SIZE = 4096


class MockDKVServer:
    """In-process ZMQ REP server simulating a dKV.

    Runs in a daemon thread. Receives protobuf ``RpcRequest`` messages,
    dispatches by oneof field, and returns ``RpcResponse``. Tracks all
    received operations for test assertions.

    Args:
        address: The ZMQ endpoint to bind on.
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self._bound_address: str | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Configurable behavior
        self._error_response: str | None = None
        self._blocks_to_acquire: list[BlockDescriptor] | None = None
        self._existing_seq_hashes: set[int] = set()

        # Stable block map: seq_hash → BlockDescriptor. Ensures re-acquiring
        # the same hash returns identical metadata (same offset/device), just
        # like the real dKV slab allocator.
        self._acquired_blocks: dict[int, BlockDescriptor] = {}
        self._next_offset: int = 0

        # Recorded operations for assertions
        self._registered_blocks: list[list[BlockDescriptor]] = []
        self._released_blocks: list[list[BlockDescriptor]] = []
        self._read_blocks_log: list[list[BlockDescriptor]] = []
        self._decremented_blocks: list[list[BlockDescriptor]] = []
        self._request_count = 0
        self._acquire_parent_hashes: list[int] = []

    def start(self) -> None:
        """Starts the mock server in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._bound_address = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mock-dkv-server"
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 5.0) -> None:
        """Blocks until the REP socket has bound (and ephemeral TCP port is known)."""
        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError(
                "mock dKV server did not become ready within"
                f" {timeout}s (bind or thread failure?)"
            )

    @property
    def bound_address(self) -> str:
        """Endpoint to pass to the client (``LAST_ENDPOINT`` after bind)."""
        if self._bound_address is not None:
            return self._bound_address
        return self._address

    def stop(self) -> None:
        """Stops the mock server and waits for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "mock dKV server thread did not exit within join timeout"
                )
            self._thread = None

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------

    def set_error_response(self, message: str | None) -> None:
        """Configures all subsequent requests to return an error.

        Args:
            message: The error message to return, or ``None`` to clear.
        """
        with self._lock:
            self._error_response = message

    def set_blocks_to_acquire(self, blocks: Sequence[BlockDescriptor]) -> None:
        """Configures the blocks returned by ``acquire_blocks``.

        Args:
            blocks: The blocks to return on acquire requests.
        """
        with self._lock:
            self._blocks_to_acquire = list(blocks)

    def set_existing_seq_hashes(self, hashes: set[int]) -> None:
        """Configures seq_hashes treated as already existing in dKV.

        Blocks whose seq_hash is in this set will have
        ``newly_acquired=False`` in the acquire response.
        """
        with self._lock:
            self._existing_seq_hashes = set(hashes)

    # ---------------------------------------------------------------------------
    # State inspection for assertions
    # ---------------------------------------------------------------------------

    @property
    def registered_blocks(self) -> list[list[BlockDescriptor]]:
        """Returns all block lists received by register operations."""
        with self._lock:
            return list(self._registered_blocks)

    @property
    def released_blocks(self) -> list[list[BlockDescriptor]]:
        """Returns all block lists received by release operations."""
        with self._lock:
            return list(self._released_blocks)

    @property
    def read_blocks_log(self) -> list[list[BlockDescriptor]]:
        """Returns all block lists received by read operations."""
        with self._lock:
            return list(self._read_blocks_log)

    @property
    def decremented_blocks(self) -> list[list[BlockDescriptor]]:
        """Returns all block lists received by decrement operations."""
        with self._lock:
            return list(self._decremented_blocks)

    @property
    def request_count(self) -> int:
        """Returns the total number of requests received."""
        with self._lock:
            return self._request_count

    @property
    def acquire_parent_hashes(self) -> list[int]:
        """Returns all parent_seq_hash values received by acquire operations."""
        with self._lock:
            return list(self._acquire_parent_hashes)

    @property
    def existing_seq_hashes(self) -> set[int]:
        """Returns the set of seq_hashes treated as already existing."""
        with self._lock:
            return set(self._existing_seq_hashes)

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _run(self) -> None:
        """Server loop running in the daemon thread."""
        ctx = zmq.Context.instance()
        sock: zmq.Socket[bytes] = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 100)  # 100ms poll for shutdown
        sock.bind(self._address)
        last_ep = sock.getsockopt(zmq.LAST_ENDPOINT)
        self._bound_address = (
            last_ep.decode() if isinstance(last_ep, bytes) else str(last_ep)
        )
        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    data = sock.recv()
                except zmq.Again:
                    continue  # Poll timeout, check stop_event

                try:
                    response_bytes = self._handle_request(data)
                except Exception:
                    logger.exception(
                        "mock dKV server: exception while handling request"
                    )
                    # Send error response to keep REP socket in valid state.
                    resp = RpcResponse(
                        error=ErrorResponse(message="internal mock error")
                    )
                    response_bytes = resp.SerializeToString()
                sock.send(response_bytes)
        finally:
            sock.close()

    def _handle_request(self, data: bytes) -> bytes:
        """Dispatches a request and returns serialized response bytes."""
        with self._lock:
            self._request_count += 1

            # Check for configured error response
            if self._error_response is not None:
                resp = RpcResponse(
                    error=ErrorResponse(message=self._error_response)
                )
                return resp.SerializeToString()

        req = RpcRequest()
        req.ParseFromString(data)

        oneof = req.WhichOneof("request")

        if oneof == "acquire_blocks":
            return self._handle_acquire(req.acquire_blocks)
        elif oneof == "register_blocks":
            return self._handle_blocks_op(
                req.register_blocks.blocks,
                self._registered_blocks,
                RegisterBlocksResponse,
            )
        elif oneof == "release_blocks":
            return self._handle_blocks_op(
                req.release_blocks.blocks,
                self._released_blocks,
                ReleaseBlocksResponse,
            )
        elif oneof == "read_blocks":
            return self._handle_blocks_op(
                req.read_blocks.blocks,
                self._read_blocks_log,
                ReadBlocksResponse,
            )
        elif oneof == "decrement_blocks":
            return self._handle_blocks_op(
                req.decrement_blocks.blocks,
                self._decremented_blocks,
                DecrementBlocksResponse,
            )
        elif oneof == "exchange_metadata":
            return self._handle_exchange_metadata(req.exchange_metadata)
        else:
            resp = RpcResponse(
                error=ErrorResponse(message=f"Unknown request: {oneof}")
            )
            return resp.SerializeToString()

    def _handle_acquire(self, acquire_req: AcquireBlocksRequest) -> bytes:
        """Handles an acquire_blocks request.

        Flattens all ``BlockSequence`` entries into one response,
        parallel to the concatenation of all ``seq_hashes``.

        Models the real dKV idempotent acquire semantics:
        - First acquire of a hash allocates a stable slab slot and
          returns ``newly_acquired=True``.
        - Subsequent acquires of the same hash return the *same*
          ``BlockDescriptor`` (identical offset/device) with
          ``newly_acquired=False``.
        """
        seq_hashes: list[int] = []
        for seq in acquire_req.sequences:
            seq_hashes.extend(seq.seq_hashes)

        with self._lock:
            for seq in acquire_req.sequences:
                self._acquire_parent_hashes.append(seq.parent_seq_hash)

            blocks: list[BlockDescriptor] = []
            newly_acquired: list[bool] = []

            if self._blocks_to_acquire is not None:
                if len(self._blocks_to_acquire) != len(seq_hashes):
                    msg = (
                        "set_blocks_to_acquire length"
                        f" ({len(self._blocks_to_acquire)}) must match"
                        f" seq_hashes length ({len(seq_hashes)})"
                    )
                    return RpcResponse(
                        error=ErrorResponse(message=msg)
                    ).SerializeToString()
                # Override path: use caller-supplied descriptors but
                # still respect idempotency tracking.
                for h, bd in zip(
                    seq_hashes, self._blocks_to_acquire, strict=False
                ):
                    already = (
                        h in self._existing_seq_hashes
                        or h in self._acquired_blocks
                    )
                    # Always ensure a stable descriptor exists, even
                    # for pre-seeded hashes that were never acquired.
                    if h not in self._acquired_blocks:
                        self._acquired_blocks[h] = bd
                    blocks.append(self._acquired_blocks[h])
                    newly_acquired.append(not already)
            else:
                for h in seq_hashes:
                    already = (
                        h in self._existing_seq_hashes
                        or h in self._acquired_blocks
                    )
                    if h not in self._acquired_blocks:
                        bd = BlockDescriptor(
                            seq_hash=h,
                            agent_id=1,
                            device_id=0,
                            offset=self._next_offset,
                            length=_DEFAULT_BLOCK_SIZE,
                        )
                        self._next_offset += _DEFAULT_BLOCK_SIZE
                        self._acquired_blocks[h] = bd
                    blocks.append(self._acquired_blocks[h])
                    newly_acquired.append(not already)

            self._existing_seq_hashes.update(seq_hashes)

        resp = RpcResponse(
            acquire_blocks=AcquireBlocksResponse(
                blocks=[
                    AcquiredBlock(metadata=b.to_proto(), newly_acquired=n)
                    for b, n in zip(blocks, newly_acquired, strict=False)
                ]
            )
        )
        return resp.SerializeToString()

    def _handle_exchange_metadata(self, req: ExchangeMetadataRequest) -> bytes:
        """Returns mock exchange metadata response."""
        meta = ExchangeMetadataResponse(
            agent_metadata=b"mock-agent-metadata",
            agent_name="mock-dkv-agent-0",
            bytes_per_page=req.bytes_per_page or _DEFAULT_BLOCK_SIZE,
            total_num_pages=64,
            base_addr=0x1000,
        )
        resp = RpcResponse(exchange_metadata=meta)
        return resp.SerializeToString()

    def _handle_blocks_op(
        self,
        pb_blocks: Sequence[BlockMetadata],
        storage: list[list[BlockDescriptor]],
        response_cls: type,
    ) -> bytes:
        """Handles register/release/read/decrement operations."""
        descriptors = [BlockDescriptor.from_proto(pb) for pb in pb_blocks]
        with self._lock:
            storage.append(descriptors)

        resp = RpcResponse()
        if response_cls is RegisterBlocksResponse:
            resp.register_blocks.CopyFrom(RegisterBlocksResponse())
        elif response_cls is ReleaseBlocksResponse:
            resp.release_blocks.CopyFrom(ReleaseBlocksResponse())
        elif response_cls is ReadBlocksResponse:
            resp.read_blocks.CopyFrom(ReadBlocksResponse())
        elif response_cls is DecrementBlocksResponse:
            resp.decrement_blocks.CopyFrom(DecrementBlocksResponse())

        return resp.SerializeToString()
