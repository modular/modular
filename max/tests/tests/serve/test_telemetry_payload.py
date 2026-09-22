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
"""Tests the one-shot startup telemetry payload.

``send_telemetry_log`` assembles an OTLP body from CLI and environment
strings and never inspects the response, so a body the collector rejects
is lost without a trace. These pin the shape rather than the wire format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from max.serve.telemetry import common


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Captures the body instead of sending it."""
    bodies: list[str] = []

    def fake_post(url: str, **kwargs: object) -> None:
        bodies.append(str(kwargs["data"]))

    monkeypatch.setattr(common.requests, "post", fake_post)
    yield bodies


def test_payload_survives_quotes_and_backslashes(posted: list[str]) -> None:
    """A model path may contain either."""
    common.send_telemetry_log('evil"model\\name')
    body = json.loads(posted[0])
    attrs = {
        a["key"]: a["value"]["stringValue"]
        for a in body["resourceLogs"][0]["resource"]["attributes"]
    }
    assert attrs["deployment.model"] == 'evil"model\\name'


def test_payload_stays_ascii(posted: list[str]) -> None:
    """A model path may hold non-ASCII, and that body was not merely invalid
    but unsendable: requests declares Content-Length in UTF-8 bytes while
    http.client encodes a str body as latin-1, so the collector either waits
    for bytes that never arrive or the send raises. json.dumps escapes it."""
    common.send_telemetry_log("mod\u00e8le/\u65e5\u672c\u8a9e")
    assert posted[0].isascii()
    attrs = {
        a["key"]: a["value"]["stringValue"]
        for a in json.loads(posted[0])["resourceLogs"][0]["resource"][
            "attributes"
        ]
    }
    assert attrs["deployment.model"] == "mod\u00e8le/\u65e5\u672c\u8a9e"
