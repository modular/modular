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
"""Fakes shared by the benchmark unit tests."""

from __future__ import annotations

from max.benchmark.benchmark_shared.datasets.types import (
    ChatSession,
    SessionMessage,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)


class FakeTokenizer:
    """Picklable stand-in for `PreTrainedTokenizerBase` used by unit tests.

    `fake_loader` constructs one per spawn worker; the parent uses an
    instance of this class directly. Behavior is deterministic so tests
    can assert on outputs (no Mock call-count plumbing).
    """

    name_or_path = "_fake_"
    vocab_size = 1000
    unk_token_id = None
    all_special_ids: frozenset[int] = frozenset({0, 1, 2})

    def __init__(self, model_max_length: int = 4096) -> None:
        self.model_max_length = model_max_length

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: object
    ) -> list[int]:
        # Length-aware so multiturn-fit tests that scale messages to a
        # target token count round-trip through encode→decode→encode and
        # see a meaningful encoded length.
        return list(range(max(4, len(text))))

    def decode(
        self, ids: list[int], skip_special_tokens: bool = False, **_: object
    ) -> str:
        return "Z" * len(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 223


def fake_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> FakeTokenizer:
    return FakeTokenizer(model_max_length=model_max_length or 4096)


class CapturingDriver(RequestDriver):
    """Request driver that records all requests and returns success."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RequestFuncInput] = []

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        assert isinstance(request_func_input, RequestFuncInput)
        self.calls.append(request_func_input)
        return RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.05,
            prompt_len=request_func_input.prompt_len,
            generated_text="ok",
        )


def make_session(session_id: int, num_user_turns: int = 3) -> ChatSession:
    messages: list[SessionMessage] = []
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(source="user", content=f"turn {i}", num_tokens=10)
        )
        messages.append(
            SessionMessage(source="assistant", content="", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)
