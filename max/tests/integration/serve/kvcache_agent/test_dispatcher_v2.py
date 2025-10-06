# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import TypeVar

import pytest
from max.serve.kvcache_agent import DispatcherClientV2, DispatcherServerV2
from max.serve.queue.zmq_queue import ClientIdentity, generate_zmq_ipc_path

T = TypeVar("T")

TIMEOUT = 1.0


def blocking_recv(fn: Callable[[], T], timeout: float = TIMEOUT) -> T:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return fn()
        except queue.Empty:
            time.sleep(0.001)
    raise queue.Empty()


class BasicDispatcherServer(DispatcherServerV2[int, int]):
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(endpoint=bind_addr, request_type=int, reply_type=int)

    def recv_request_blocking(self) -> tuple[int, ClientIdentity]:
        return blocking_recv(self.recv_request_nowait)


class BasicDispatcherClient(DispatcherClientV2[int, int]):
    def __init__(self, bind_addr: str, default_dest_addr: str | None):
        self.bind_addr = bind_addr
        super().__init__(
            bind_addr=bind_addr,
            default_dest_addr=default_dest_addr,
            request_type=int,
            reply_type=int,
        )

    def recv_reply_blocking(self) -> int:
        return blocking_recv(self.recv_reply_nowait)


def make_servers_and_clients(
    num_servers: int, num_clients: int, use_default_dest_addr: bool = True
) -> tuple[list[BasicDispatcherServer], list[BasicDispatcherClient]]:
    server_addrs = [generate_zmq_ipc_path() for _ in range(num_servers)]
    client_addrs = [generate_zmq_ipc_path() for _ in range(num_clients)]
    servers = [
        BasicDispatcherServer(bind_addr=server_addr)
        for server_addr in server_addrs
    ]
    default_dest_addr = server_addrs[0] if use_default_dest_addr else None
    clients = [
        BasicDispatcherClient(
            bind_addr=client_addr, default_dest_addr=default_dest_addr
        )
        for client_addr in client_addrs
    ]
    return servers, clients


def test_server_client() -> None:
    servers, clients = make_servers_and_clients(1, 1)
    server = servers[0]
    client = clients[0]

    for _ in range(100):
        t0 = time.time()
        client.send_request_nowait(42)
        request, identity = server.recv_request_blocking()
        assert request == 42
        server.send_reply_nowait(99, identity)
        reply = client.recv_reply_blocking()
        assert reply == 99
        t1 = time.time()

        print(f"Time taken: {(t1 - t0) * 1000:.3f} ms")


def test_many_clients_one_server() -> None:
    servers, clients = make_servers_and_clients(1, 10)
    server = servers[0]
    clients = clients

    for i, client in enumerate(clients):
        client.send_request_nowait(i)

    num_requests_received = 0
    while True:
        try:
            request, identity = server.recv_request_blocking()
            num_requests_received += 1
            server.send_reply_nowait(request + 100, identity)
        except queue.Empty:
            break
    assert num_requests_received == len(clients)

    for i, client in enumerate(clients):
        reply = client.recv_reply_blocking()
        assert reply == i + 100


def test_many_servers_one_client() -> None:
    servers, clients = make_servers_and_clients(
        10, 1, use_default_dest_addr=False
    )
    client = clients[0]

    for server in servers:
        client.send_request_nowait(-1, server.bind_addr)

    for i, server in enumerate(servers):
        request, identity = server.recv_request_blocking()
        server.send_reply_nowait(request * i * 100, identity)

    replies = []
    while True:
        try:
            reply = client.recv_reply_blocking()
            replies.append(reply)
        except queue.Empty:
            break

    assert len(replies) == len(servers)
    assert set(replies) == set([-1 * i * 100 for i in range(len(servers))])


def test_spam_server() -> None:
    servers, clients = make_servers_and_clients(1, 1)
    server = servers[0]
    client = clients[0]

    for i in range(100):
        client.send_request_nowait(i)

    while True:
        try:
            request, identity = server.recv_request_blocking()
            server.send_reply_nowait(request + 100, identity)
        except queue.Empty:
            break

    for i in range(100):
        reply = client.recv_reply_blocking()
        assert reply == i + 100


def test_no_dest_addr() -> None:
    _, clients = make_servers_and_clients(1, 1, use_default_dest_addr=False)
    client = clients[0]

    with pytest.raises(ValueError):
        client.send_request_nowait(42)
