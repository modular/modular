# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from max.serve.recordreplay import jsonl, schema


@pytest.fixture
def example_recording_in() -> schema.Recording:
    return [
        schema.Dummy(),
        schema.Transaction(request=schema.Request(method="GET", path="/")),
        schema.Transaction(
            request=schema.Request(method="GET", path="/"),
            response=schema.Response(
                start=schema.ResponseStart(status=200),
                chunks=[schema.ResponseChunk(body=b"hello")],
            ),
        ),
    ]


def test_read() -> None:
    recording = list(
        jsonl.read_jsonl_recording(
            io.StringIO(
                '{"type": "dummy"}\n'
                '{"type": "transaction", "request": '
                '{"method": "GET", "path": "/", "headers": []}}'
            )
        )
    )
    assert len(recording) == 2
    assert isinstance(recording[0], schema.Dummy)
    assert isinstance(recording[1], schema.Transaction)
    assert recording[1].request.method == "GET"
    assert recording[1].request.path == "/"
    assert recording[1].request.headers == []
    assert recording[1].response is None


def test_write(example_recording_in: schema.Recording) -> None:
    recording_in = example_recording_in
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "example.jsonl"
        with jsonl.JSONLFileRecorder(path) as recorder:
            for item in recording_in:
                recorder.record(item)
        with path.open() as f:
            for line in f:
                # Each line must be an independent JSON item.  Items may not
                # span lines.
                json.loads(line)


def test_write_owning() -> None:
    f = tempfile.TemporaryFile()
    mock_f = mock.Mock(spec=f, wraps=f)
    with jsonl.JSONLFileRecorder(mock_f, owned=True):
        pass
    mock_f.close.assert_called_once()


def test_write_non_owning() -> None:
    f = tempfile.TemporaryFile()
    mock_f = mock.Mock(spec=f, wraps=f)
    with jsonl.JSONLFileRecorder(mock_f, owned=False):
        pass
    mock_f.close.assert_not_called()
    f.close()


def test_roundtrip(example_recording_in: schema.Recording) -> None:
    recording_in = example_recording_in
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "example.jsonl"
        with jsonl.JSONLFileRecorder(path) as recorder:
            for item in recording_in:
                recorder.record(item)
        recording_out = list(jsonl.read_jsonl_recording(path))
    assert recording_in == recording_out
