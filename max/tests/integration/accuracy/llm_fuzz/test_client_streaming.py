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
"""Exercise streaming cancellation against real buffered HTTP responses."""

import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

from client import FuzzClient, RawResponse, RunConfig


@contextmanager
def streaming_server(
    fragmented: bool = False,
    close_header: bool = False,
) -> Iterator[tuple[str, threading.Event]]:
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            if close_header:
                self.send_header("Connection", "close")
            self.end_headers()

            def send_chunk(data: bytes) -> None:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            send_chunk(b"data: fir" if fragmented else b"data: first\n\n")
            if fragmented:
                time.sleep(0.05)
                send_chunk(b"st\n\ndata: [DONE]\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                # No body bytes arrive before the cancellation/read deadline.
                # Buffered HTTPResponse readers must not be retried after a timeout.
                self.connection.settimeout(2)
                try:
                    if not self.connection.recv(1):
                        disconnected.set()
                except TimeoutError:
                    # The server-side idle timeout is expected: only the
                    # client's own close deadline may be shorter.
                    pass
                except ConnectionResetError:
                    # A reset is how some platforms surface the client's
                    # close(), so it still proves the transport is closed.
                    disconnected.set()
            self.close_connection = True

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


class StreamingClientTests(unittest.TestCase):
    def call(
        self,
        url: str,
        *,
        timeout: float,
        cancel_after_chunks: int | None,
        cancel_after_ms: float | None,
    ) -> RawResponse:
        client = FuzzClient(RunConfig(base_url=url))
        try:
            return client._sync_streaming(
                b"{}",
                {"Content-Type": "application/json"},
                timeout,
                cancel_after_chunks,
                cancel_after_ms,
            )
        finally:
            client._pool.shutdown(wait=True)

    def test_time_cancellation_during_idle_body_closes_connection(self) -> None:
        with streaming_server() as (url, disconnected):
            response = self.call(
                url, timeout=1, cancel_after_chunks=None, cancel_after_ms=250
            )
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.error)
            self.assertTrue(response.cancelled)
            self.assertEqual(response.chunks, ["first"])
            self.assertTrue(disconnected.wait(1))

    def test_chunk_cancellation_does_not_wait_for_1024_bytes(self) -> None:
        with streaming_server() as (url, disconnected):
            response = self.call(
                url, timeout=0.3, cancel_after_chunks=1, cancel_after_ms=None
            )
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.error)
            self.assertTrue(response.cancelled)
            self.assertEqual(response.chunks, ["first"])
            self.assertTrue(disconnected.wait(1))

    def test_idle_body_timeout_is_reported_and_closes_connection(self) -> None:
        with streaming_server() as (url, disconnected):
            response = self.call(
                url,
                timeout=0.15,
                cancel_after_chunks=None,
                cancel_after_ms=None,
            )
            self.assertEqual(response.status, 0)
            self.assertEqual(response.error, "TIMEOUT")
            self.assertFalse(response.cancelled)
            self.assertTrue(disconnected.wait(1))

    def test_fragmented_sse_and_done_are_preserved(self) -> None:
        with streaming_server(fragmented=True) as (url, _):
            response = self.call(
                url, timeout=1, cancel_after_chunks=None, cancel_after_ms=None
            )
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.error)
            self.assertFalse(response.cancelled)
            self.assertEqual(response.chunks, ["first", "[DONE]"])

    def test_connection_close_response_remains_cancellable(self) -> None:
        with streaming_server(close_header=True) as (url, disconnected):
            response = self.call(
                url, timeout=1, cancel_after_chunks=None, cancel_after_ms=250
            )
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.error)
            self.assertTrue(response.cancelled)
            self.assertEqual(response.chunks, ["first"])
            self.assertTrue(disconnected.wait(1))

    def test_read_timeout_before_cancel_deadline_is_not_a_cancellation(
        self,
    ) -> None:
        with streaming_server() as (url, disconnected):
            response = self.call(
                url,
                timeout=0.15,
                cancel_after_chunks=None,
                cancel_after_ms=1000,
            )
            self.assertEqual(response.status, 0)
            self.assertEqual(response.error, "TIMEOUT")
            self.assertFalse(response.cancelled)
            self.assertTrue(disconnected.wait(1))


if __name__ == "__main__":
    unittest.main()
