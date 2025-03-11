# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import contextlib
import datetime
import urllib.parse
from typing import AsyncIterable, Iterator
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from max.serve.recordreplay import interfaces, middleware, schema
from pydantic import BaseModel


class ListRecorder(interfaces.Recorder):
    recorded: list[schema.RecordingItem]

    def __init__(self) -> None:
        self.recorded = []

    def record(self, item: schema.RecordingItem) -> None:
        self.recorded.append(item)


class DepositParams(BaseModel):
    account: str
    amount: int


@pytest.mark.parametrize("include_responses", [False, True])
def test_recording(include_responses: bool) -> None:
    start_time = datetime.datetime.now(datetime.timezone.utc).astimezone()

    recorder = ListRecorder()
    app = FastAPI()
    app.add_middleware(
        middleware.RecorderMiddleware,
        recorder=recorder,
        include_responses=include_responses,
    )

    @app.get("/successor/{n}")
    async def successor(n: int) -> int:
        return n + 1

    @app.get("/file/{path:path}")
    async def file_contents(path: str) -> str:
        return f"Pretend this were the contents of {path}"

    @app.get("/greeting")
    async def greeting(name: str) -> str:
        return f"Good day, {name}"

    @app.post("/deposit")
    async def deposit(params: DepositParams) -> str:
        return f"Deposited {params.amount} into {params.account}"

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def generator() -> AsyncIterable[str]:
            for i in range(3):
                yield f"{i + 1}\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(generator())

    with TestClient(app) as client:
        r = client.get("/successor/3")
        r.raise_for_status()
        assert r.json() == 4
        r = client.get(f"/file/{urllib.parse.quote('My cool file!')}")
        r.raise_for_status()
        assert r.json() == "Pretend this were the contents of My cool file!"
        r = client.get(f"/greeting?name={urllib.parse.quote_plus('Martínez')}")
        r.raise_for_status()
        assert r.json() == "Good day, Martínez"
        r = client.post("/deposit", json={"account": "checking", "amount": 100})
        r.raise_for_status()
        assert r.json() == "Deposited 100 into checking"
        r = client.get("/stream")
        r.raise_for_status()
        assert r.text == "1\n2\n3\n"
        r = client.get("/nothing")
        assert r.status_code == 404

    end_time = datetime.datetime.now(datetime.timezone.utc).astimezone()

    recording = recorder.recorded
    assert len(recording) == 6
    (
        successor_tx,
        file_tx,
        greeting_tx,
        deposit_tx,
        stream_tx,
        nothing_tx,
    ) = recording

    def check_timestamp(d: datetime.datetime | None) -> None:
        assert d is not None
        assert d.tzinfo is not None
        assert d.tzinfo.utcoffset(d) is not None
        assert start_time <= d <= end_time

    def check_request(
        r: schema.Request, method: str, path: str, body: bytes | None = None
    ) -> None:
        check_timestamp(r.start_timestamp)
        check_timestamp(r.end_timestamp)
        assert r.method == method
        assert r.path == path
        assert len(r.headers) > 0
        assert (r.body or None) == (body or None)

    def check_response(r: schema.Response | None, body: bytes) -> None:
        if not include_responses:
            assert r is None
            return
        assert r is not None
        check_timestamp(r.start.timestamp)
        assert r.start.status == 200
        assert len(r.start.headers) > 0
        headers = dict(r.start.headers)
        assert headers[b"content-type"] == b"application/json"
        assert len(r.chunks) == 1
        chunk = r.chunks[0]
        check_timestamp(chunk.timestamp)
        assert r.start.timestamp is not None
        assert chunk.timestamp is not None
        assert r.start.timestamp <= chunk.timestamp
        assert chunk.body == body

    assert isinstance(successor_tx, schema.Transaction)
    check_request(successor_tx.request, "GET", "/successor/3")
    check_response(successor_tx.response, b"4")
    assert isinstance(file_tx, schema.Transaction)
    check_request(
        file_tx.request, "GET", f"/file/{urllib.parse.quote('My cool file!')}"
    )
    check_response(
        file_tx.response, b'"Pretend this were the contents of My cool file!"'
    )
    assert isinstance(greeting_tx, schema.Transaction)
    check_request(
        greeting_tx.request,
        "GET",
        f"/greeting?name={urllib.parse.quote_plus('Martínez')}",
    )
    check_response(greeting_tx.response, '"Good day, Martínez"'.encode("utf-8"))
    assert isinstance(deposit_tx, schema.Transaction)
    check_request(
        deposit_tx.request,
        "POST",
        "/deposit",
        b'{"account": "checking", "amount": 100}',
    )
    check_response(deposit_tx.response, b'"Deposited 100 into checking"')
    assert isinstance(stream_tx, schema.Transaction)
    check_request(stream_tx.request, "GET", "/stream")
    if include_responses:
        assert stream_tx.response is not None
        assert len(stream_tx.response.chunks) == 4
        assert stream_tx.response.chunks[0].body == b"1\n"
        assert stream_tx.response.chunks[1].body == b"2\n"
        assert stream_tx.response.chunks[2].body == b"3\n"
        assert stream_tx.response.chunks[3].body == b""
    else:
        assert stream_tx.response is None
    assert isinstance(nothing_tx, schema.Transaction)
    check_request(nothing_tx.request, "GET", "/nothing")
    r = client.get("/nothing")
    if include_responses:
        assert nothing_tx.response is not None
        assert nothing_tx.response.start.status == 404
    else:
        assert nothing_tx.response is None


def test_lifespan() -> None:
    recorder_mocks = [
        mock.Mock(spec=interfaces.Recorder),
        mock.Mock(spec=interfaces.Recorder),
    ]
    enter_mock = mock.Mock()
    enter_mock.side_effect = recorder_mocks
    exit_mock = mock.Mock()

    @contextlib.contextmanager
    def factory() -> Iterator[interfaces.Recorder]:
        recorder_mock = enter_mock()
        assert recorder_mock in recorder_mocks
        yield recorder_mock
        exit_mock(recorder_mock)

    app = FastAPI()
    app.add_middleware(middleware.RecorderMiddleware, recorder_factory=factory)
    enter_mock.assert_not_called()

    # We manually manage the lifetimes of exit_stack1 and exit_stack2 below,
    # but if an assertion fails, we need this outer meta_exit_stack to make
    # sure we clean up anyway and Pytest does not then hang.
    with contextlib.ExitStack() as meta_exit_stack:
        exit_stack1 = meta_exit_stack.enter_context(contextlib.ExitStack())
        client1 = exit_stack1.enter_context(TestClient(app))
        enter_mock.assert_called_once()
        enter_mock.reset_mock()
        exit_mock.assert_not_called()
        recorder_mocks[0].record.assert_not_called()
        client1.get("/example")
        recorder_mocks[0].record.assert_called_once()
        recorder_mocks[0].record.reset_mock()

        exit_stack2 = meta_exit_stack.enter_context(contextlib.ExitStack())
        client2 = exit_stack2.enter_context(TestClient(app))
        enter_mock.assert_called_once()
        exit_mock.assert_not_called()
        recorder_mocks[1].record.assert_not_called()
        client2.get("/example")
        recorder_mocks[1].record.assert_called_once()
        recorder_mocks[1].record.reset_mock()

        recorder_mocks[0].record.assert_not_called()
        recorder_mocks[1].record.assert_not_called()
        client1.get("/example")
        recorder_mocks[0].record.assert_called_once()
        recorder_mocks[1].record.assert_not_called()
        recorder_mocks[0].record.reset_mock()

        exit_mock.assert_not_called()
        exit_stack1.close()
        exit_mock.assert_called_once_with(recorder_mocks[0])
        exit_mock.reset_mock()

        exit_mock.assert_not_called()
        exit_stack2.close()
        exit_mock.assert_called_once_with(recorder_mocks[1])
        exit_mock.reset_mock()


def test_not_required_lifespan() -> None:
    app = FastAPI()
    app.add_middleware(middleware.RecorderMiddleware, recorder=ListRecorder())

    @app.get("/hello")
    def hello_handler() -> str:
        return "Hello"

    # Here we are NOT using a 'with' statement on the TestClient, so that
    # lifespan callbacks will be missing.
    client = TestClient(app)
    # Since we used "recorder" rather than "recorder_factory", this is OK --
    # lifespan not required.
    r = client.get("/hello")
    r.raise_for_status()


def test_required_lifespan() -> None:
    app = FastAPI()
    app.add_middleware(
        middleware.RecorderMiddleware,
        recorder_factory=ListRecorder,
    )

    @app.get("/hello")
    def hello_handler() -> str:
        return "Hello"

    # Here we are NOT using a 'with' statement on the TestClient, so that
    # lifespan callbacks will be missing.
    client = TestClient(app)
    # Since we used "recorder_factory" rather than "recorder", lifespan is
    # required, so we should fail here.
    with pytest.raises(middleware.LifespanRequiredError):
        client.get("/hello")
