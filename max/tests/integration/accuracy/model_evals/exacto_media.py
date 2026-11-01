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
"""Carries a ``video_url`` content part through Inspect to the wire.

Inspect cannot send video over OpenAI chat completions: its converter raises on
``ContentVideo``. Worse, by default it downloads every sample's media and
inlines it as base64 into the log, so a URL becomes a ~50 MB ``data:`` URI. An
OpenRouter benchmark sends the video *by URL*, and the server under test is
meant to fetch it.

So a task puts the URL in a text part that starts with :data:`MARKER`, which
Inspect passes through untouched, and the ``openai-stream`` / ``openai-chat``
providers in ``exacto_streaming`` rewrite that part into a ``video_url`` part
just before the request is sent. The rewrite happens on the request dict that
Inspect also logs, so the log shows the request as sent.

Stdlib only, so it unit-tests under bazel and imports into the harness venv.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

#: Starts a text part that stands for a video URL. The NUL keeps it from ever
#: matching prompt text.
MARKER = "\x00exacto:video_url\x00"


def video_url_marker(url: str) -> str:
    """Returns the text part that stands for a video URL.

    Raises:
        ValueError: If ``url`` is not an http(s) URL. A ``data:`` URI here
            means the media was inlined, which is exactly what the marker
            exists to prevent.
    """
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"expected an http(s) video URL, got {url[:64]!r}")
    return MARKER + url


def video_url_part(url: str) -> dict[str, object]:
    """Returns the OpenAI chat-completions content part for a video URL."""
    return {"type": "video_url", "video_url": {"url": url}}


def expand_video_url_markers(
    messages: Sequence[MutableMapping[str, object]],
) -> int:
    """Rewrites marker text parts into ``video_url`` parts, in place.

    Args:
        messages: OpenAI chat-completions messages.

    Returns:
        How many parts were rewritten.
    """
    rewritten = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str) and text.startswith(MARKER):
                content[index] = video_url_part(text[len(MARKER) :])
                rewritten += 1
    return rewritten
