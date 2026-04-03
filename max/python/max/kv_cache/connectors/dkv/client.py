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

"""Synchronous ZMQ client for the dKV server."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import Enum
from types import TracebackType
from urllib.parse import urlsplit

import zmq
from typing_extensions import Self

from .protocol import (
    BlockDescriptor,
    DKVError,
    ExchangeMetadataResult,
    build_acquire_request,
    build_decrement_request,
    build_exchange_metadata_request,
    build_read_request,
    build_register_request,
    build_release_request,
    parse_acquire_response,
    parse_decrement_response,
    parse_exchange_metadata_response,
    parse_read_response,
    parse_register_response,
    parse_release_response,
)

logger = logging.getLogger(__name__)


class RequestState(str, Enum):
    """What is known about request transmission after a transport failure."""

    NOT_SENT = "not_sent"
    MAYBE_SENT = "maybe_sent"
    SENT = "sent"


class DKVTransportError(DKVError, ConnectionError):
    """Raised when a client transport failure prevents a clean RPC round-trip."""

    def __init__(
        self,
        message: str,
        *,
        request_state: RequestState,
    ) -> None:
        super().__init__(message)
        self.request_state = request_state


class DKVTransportTimeoutError(DKVTransportError, TimeoutError):
    """Raised when the client times out while sending or receiving."""


def _close_socket_on_finalize(sock: zmq.Socket[bytes]) -> None:
    """Best-effort socket cleanup callback used by ``weakref.finalize``."""
    sock.close()


def _validate_address(address: str) -> None:
    """Validates a ZMQ endpoint address.

    Args:
        address: The ZMQ address to validate.

    Raises:
        ValueError: If the address is not a valid ``ipc://`` or ``tcp://``
            endpoint.
    """
    parsed = urlsplit(address)

    if parsed.scheme == "tcp":
        if not parsed.hostname:
            raise ValueError(
                "TCP address must be in the format tcp://host:port."
                f" Found: {address}"
            )
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError(
                "TCP address must not include a path, query, or fragment."
                f" Found: {address}"
            )
        try:
            port = parsed.port
        except ValueError:
            raise ValueError(
                f"TCP address must include a valid port. Found: {address}"
            ) from None
        if port is None:
            raise ValueError(
                "TCP address must be in the format tcp://host:port."
                f" Found: {address}"
            )
        if not (1 <= port <= 65535):
            raise ValueError(
                f"TCP port must be between 1 and 65535. Found: {port}"
            )
    elif parsed.scheme == "ipc":
        if parsed.query or parsed.fragment:
            raise ValueError(
                "IPC address must not include a query or fragment."
                f" Found: {address}"
            )
        ipc_path = address.removeprefix("ipc://")
        if not ipc_path:
            raise ValueError(
                "IPC address requires a path after the protocol."
                f" Found: {address}"
            )
        if len(ipc_path) > zmq.IPC_PATH_MAX_LEN:
            raise ValueError(
                f"IPC path is too long ({len(ipc_path)} chars)."
                f" Maximum is {zmq.IPC_PATH_MAX_LEN}."
            )
    else:
        raise ValueError(
            f"Address must start with tcp:// or ipc://. Found: {address}"
        )


class DKVClient:
    """Synchronous ZMQ client for a single dKV server instance.

    Communicates over ZMQ REQ/REP sockets using Protobuf serialization.
    Recreates the REQ socket after transport failures so later requests can
    proceed on a fresh connection. Callers still need to handle transport
    exceptions and inspect whether the failed request was definitely not sent,
    may have been sent, or was definitely sent.

    Not thread-safe. Each thread must use its own client instance.

    ``close()`` tears down the REQ socket but does not permanently retire the
    client: you may call ``connect()`` again on the same instance to open a
    new connection.

    Note:
        ``ValueError`` is raised for invalid arguments (including empty
        ``seq_hashes`` or ``blocks`` where disallowed, and invalid endpoints
        or non-positive timeouts at construction). Response handling lives in
        ``dkv.client.protocol``, which raises ``DKVError`` subclasses on
        decode, envelope, or server errors. Transport problems use
        ``DKVTransportError`` or ``DKVTransportTimeoutError``; when set,
        inspect ``request_state``.

    Args:
        address: The ZMQ endpoint (for example,
            ``ipc:///var/run/dkv/api.sock`` or ``tcp://host:port``).
            Validated on construction.
        recv_timeout_ms: The maximum time in milliseconds to wait for a
            server response. Defaults to ``1000``.
        send_timeout_ms: The maximum time in milliseconds to wait for
            send buffer space. Defaults to ``1000``.
    """

    def __init__(
        self,
        address: str,
        recv_timeout_ms: int = 1000,
        send_timeout_ms: int = 1000,
    ) -> None:
        if recv_timeout_ms <= 0:
            raise ValueError("recv_timeout_ms must be positive")
        if send_timeout_ms <= 0:
            raise ValueError("send_timeout_ms must be positive")

        _validate_address(address)
        self._address = address
        self._recv_timeout_ms = recv_timeout_ms
        self._send_timeout_ms = send_timeout_ms
        self._socket: zmq.Socket[bytes] | None = None
        self._is_closed = False
        self._finalizer: weakref.finalize | None = None  # type: ignore[type-arg]

    def connect(self) -> None:
        """Connects to the dKV server.

        Creates a ZMQ REQ socket and connects to the configured address.
        Safe to call multiple times. Subsequent calls are no-ops if
        already connected. After :meth:`close`, the next call opens a new
        socket.

        Raises:
            DKVTransportError: If ZeroMQ reports an error while creating or
                configuring the socket or connecting to the endpoint given at
                construction. ``request_state`` is :attr:`RequestState.NOT_SENT`
                (no request bytes were sent on the wire).
        """
        if self._socket is not None:
            return

        self._is_closed = False
        self._socket = self._create_socket()
        self._finalizer = weakref.finalize(
            self, _close_socket_on_finalize, self._socket
        )

    def close(self) -> None:
        """Closes the connection and releases resources.

        Idempotent. Safe to call multiple times. Call :meth:`connect` again
        to use this client with a new REQ socket.
        """
        if not self._is_closed:
            self._is_closed = True
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            if self._socket is not None:
                self._socket.close()
                self._socket = None

    def __enter__(self) -> Self:
        """Context manager entry.

        Raises:
            DKVTransportError: See :meth:`connect`.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """Returns a string representation of the client."""
        connected = self._socket is not None and not self._is_closed
        return f"DKVClient({self._address!r}, connected={connected})"

    # ---------------------------------------------------------------------------
    # Block lifecycle operations
    # ---------------------------------------------------------------------------

    def acquire_blocks(
        self,
        seq_hashes: Sequence[int],
        parent_seq_hash: int = 0,
    ) -> list[BlockDescriptor]:
        """Acquires free blocks from the dKV server.

        Args:
            seq_hashes: The ordered sequence hashes for the blocks to
                acquire. The order determines the chaining order.
            parent_seq_hash: The optional parent sequence hash when
                extending an existing sequence. Defaults to ``""``.

        Returns:
            The list of acquired block descriptors.

        Raises:
            ValueError: If ``seq_hashes`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or another ZMQ
                transport error occurred. Inspect ``request_state`` to
                determine whether the request was sent.
        """
        data = self._send_recv(
            build_acquire_request(seq_hashes, parent_seq_hash)
        )
        blocks = parse_acquire_response(data)
        logger.debug("acquired %d blocks", len(blocks))
        return blocks

    def register_blocks(self, blocks: Sequence[BlockDescriptor]) -> None:
        """Registers blocks as filled with data.

        Transitions blocks from FILLING to REGISTERED state.

        Args:
            blocks: The blocks to register.

        Raises:
            ValueError: If ``blocks`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or another ZMQ
                transport error occurred. Inspect ``request_state`` to
                determine whether the request was sent.
        """
        data = self._send_recv(build_register_request(blocks))
        parse_register_response(data)

    def release_blocks(self, blocks: Sequence[BlockDescriptor]) -> None:
        """Releases blocks from FILLING back to FREE state.

        Only valid for blocks that have been acquired but not yet
        registered. REGISTERED blocks are freed by the dKV server's
        internal eviction policy.

        Args:
            blocks: The blocks to release.

        Raises:
            ValueError: If ``blocks`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or another ZMQ
                transport error occurred. Inspect ``request_state`` to
                determine whether the request was sent.
        """
        data = self._send_recv(build_release_request(blocks))
        parse_release_response(data)

    def read_blocks(self, blocks: Sequence[BlockDescriptor]) -> None:
        """Prepares blocks for transfer by incrementing their read_ref_count.

        The caller must call ``decrement_blocks`` after the transfer
        completes.

        Args:
            blocks: The blocks to read.

        Raises:
            ValueError: If ``blocks`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or another ZMQ
                transport error occurred. Inspect ``request_state`` to
                determine whether the request was sent.

        Note:
            REQ/REP with timeouts means the client can lose certainty: for
            example, a receive timeout may leave ``request_state`` as
            :attr:`RequestState.SENT` while the server already applied the
            increment. The client cannot infer that from a timeout alone, so
            pairing ``decrement_blocks`` with ``read_blocks`` must follow your
            process for ambiguous RPCs (for example server semantics,
            monitoring, or reconciliation).
        """
        data = self._send_recv(build_read_request(blocks))
        parse_read_response(data)

    def decrement_blocks(self, blocks: Sequence[BlockDescriptor]) -> None:
        """Decrements the read_ref_count on blocks after a transfer completes.

        Args:
            blocks: The blocks to decrement.

        Raises:
            ValueError: If ``blocks`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or another ZMQ
                transport error occurred. Inspect ``request_state`` to
                determine whether the request was sent.
        """
        data = self._send_recv(build_decrement_request(blocks))
        parse_decrement_response(data)

    def exchange_metadata(
        self,
        agent_metadata: bytes,
        bytes_per_page: int = 0,
    ) -> ExchangeMetadataResult:
        """Exchanges NIXL agent metadata with the dKV server and
        configures its slab page geometry.

        This is the primary init RPC. Sends the engine's NIXL agent
        metadata and page size, receives dKV's agent metadata and
        confirmed slab geometry in one round-trip.

        Args:
            agent_metadata: Opaque NIXL agent metadata blob from the
                engine's transfer engine.
            bytes_per_page: Engine's KV cache page size in bytes. dKV
                uses this to divide its slab into slots. 0 means use
                dKV's CLI-configured block size (legacy path).

        Returns:
            The dKV agent's metadata, name, and confirmed page geometry.

        Raises:
            DKVError: If the server returned an error (including page
                size mismatch).
            DKVTransportTimeoutError: If the request timed out.
            DKVTransportError: If not connected or transport failure.
        """
        data = self._send_recv(
            build_exchange_metadata_request(agent_metadata, bytes_per_page)
        )
        return parse_exchange_metadata_response(data)

    @contextmanager
    def reading_blocks(
        self, blocks: Sequence[BlockDescriptor]
    ) -> Iterator[None]:
        """Context manager that pins blocks for reading and releases on exit.

        Calls ``read_blocks`` on entry to increment ``read_ref_count`` and
        ``decrement_blocks`` on exit. If ``decrement_blocks`` fails, the
        failure is logged (server-side ref-counts may be leaked). When the
        body raised, the original exception still propagates; when the body
        completed normally, the decrement error propagates.

        Entry is not wrapped by the same exit path as the body: if
        ``read_blocks`` raises (including a receive timeout where
        ``request_state`` is :attr:`RequestState.SENT`), the context body
        does not run and this manager **does not** call ``decrement_blocks``.
        If the server already applied the read increment, that can leave
        server-side read refs elevated until eviction or another recovery
        path. Handle transport errors from ``read_blocks`` like any ambiguous
        RPC; see :meth:`read_blocks`.

        The body is protected with ``except BaseException`` so refcount
        cleanup still runs on :exc:`KeyboardInterrupt` and similar; that may
        perform RPC while handling the interrupt—prefer exiting the context
        cleanly when possible.

        Args:
            blocks: The blocks to read.

        Raises:
            ValueError: If ``blocks`` is empty.
            DKVError: If the server returned an error.
            DKVTransportTimeoutError: If a request timed out.
            DKVTransportError: If a transport error occurred.
        """
        self.read_blocks(blocks)
        try:
            yield
        except BaseException:
            try:
                self.decrement_blocks(blocks)
            except Exception:
                logger.exception(
                    "failed to decrement %d blocks after exception;"
                    " server-side ref-counts may be leaked",
                    len(blocks),
                )
            raise
        else:
            try:
                self.decrement_blocks(blocks)
            except Exception:
                logger.exception(
                    "failed to decrement %d blocks after successful read;"
                    " server-side ref-counts may be leaked",
                    len(blocks),
                )
                raise

    # ---------------------------------------------------------------------------
    # ZMQ socket and RPC
    # ---------------------------------------------------------------------------

    def _create_socket(self) -> zmq.Socket[bytes]:
        """Creates and configures a new ZMQ REQ socket.

        Raises:
            DKVTransportError: If a :exc:`zmq.ZMQError` occurs during setup.
                Chains the original exception and sets ``request_state`` to
                :attr:`RequestState.NOT_SENT`.
        """
        ctx = zmq.Context.instance()
        sock: zmq.Socket[bytes] = ctx.socket(zmq.REQ)
        try:
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
            sock.connect(self._address)
        except Exception as e:
            sock.close()
            raise DKVTransportError(
                f"Failed to create socket: {e}",
                request_state=RequestState.NOT_SENT,
            ) from e
        return sock

    def _reset_socket(self) -> None:
        """Closes the current socket and creates a fresh replacement.

        The old socket is always closed. If creating a new socket fails,
        ``self._socket`` is left as ``None`` and the exception propagates.
        """
        logger.warning("resetting ZMQ socket to %s", self._address)
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        old_socket = self._socket
        self._socket = None
        if old_socket is not None:
            old_socket.close()
        self._socket = self._create_socket()
        self._finalizer = weakref.finalize(
            self, _close_socket_on_finalize, self._socket
        )

    def _try_reset_socket(self) -> None:
        """Best-effort socket reset; logs and continues if recreation fails.

        If recreation fails, ``self._socket`` may be ``None``; the next RPC
        raises "Not connected" until :meth:`connect` succeeds.
        """
        try:
            self._reset_socket()
        except Exception:
            logger.exception(
                "failed to recreate ZMQ socket to %s", self._address
            )

    def _send_recv(self, request_bytes: bytes) -> bytes:
        """Sends a request and receives a response.

        On transport failures, resets the socket so later requests can
        proceed on a fresh REQ socket. Failed requests are not retried
        automatically because the request may already be in flight.

        Args:
            request_bytes: The pre-serialized request.

        Returns:
            The raw response bytes.

        Raises:
            DKVTransportTimeoutError: If the request timed out while sending
                or receiving. Inspect ``request_state`` to determine whether
                the request was sent.
            DKVTransportError: If the client is not connected or a ZMQ error
                occurred. Inspect ``request_state`` to determine whether the
                request was sent.

        Note:
            Internal socket reset after an error is best-effort. If opening a
            replacement socket fails, call :meth:`connect` before the next
            RPC.
        """
        if self._socket is None:
            raise DKVTransportError(
                "Not connected; request was not sent",
                request_state=RequestState.NOT_SENT,
            )

        try:
            self._socket.send(request_bytes)
        except zmq.Again:
            self._try_reset_socket()
            raise DKVTransportTimeoutError(
                "Timed out waiting for send buffer space; request was not sent",
                request_state=RequestState.NOT_SENT,
            ) from None
        except zmq.ZMQError as e:
            self._try_reset_socket()
            raise DKVTransportError(
                f"ZMQ error while sending request: {e}; request may have been sent",
                request_state=RequestState.MAYBE_SENT,
            ) from e

        try:
            return self._socket.recv()
        except zmq.Again:
            self._try_reset_socket()
            raise DKVTransportTimeoutError(
                "dKV server did not respond within"
                f" {self._recv_timeout_ms}ms; request was sent and completion"
                " is unknown",
                request_state=RequestState.SENT,
            ) from None
        except zmq.ZMQError as e:
            self._try_reset_socket()
            raise DKVTransportError(
                f"ZMQ error while receiving response: {e}; request was sent",
                request_state=RequestState.SENT,
            ) from e
