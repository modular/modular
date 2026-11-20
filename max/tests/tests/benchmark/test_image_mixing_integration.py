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

"""End-to-end test for MXTOOLS-398 image mixing.

Ties together the pieces added across the stack: SessionMessage carrying
images, dataset-agnostic augmentation, the multi-turn driver attaching
images to outgoing requests, and --dry-run workload stats reporting them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
)
from max.benchmark.benchmark_shared.datasets.image_augmentation import (
    augment_samples_with_images,
)
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    build_fitted_chat_samples,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ImageContentBlock,
    OpenAIImage,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.multi_turn import chat_session_driver
from max.benchmark.benchmark_shared.request import (
    ChatWireHistory,
    OpenAIChatCompletionsRequestDriver,
    RequestCounter,
    RequestFuncInput,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    print_workload_stats,
)
from pytest_mock import MockerFixture
from tests.benchmark.benchmark_fakes import (
    CapturingDriver,
    FakeTokenizer,
    StreamBody,
    fake_loader,
    sent_json,
)


def test_image_mixing_pipeline_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dataset-agnostic multi-turn workload augmented with images: the
    images survive sampling, get attached to the wire request by the
    multi-turn driver, and show up in --dry-run workload stats."""
    tok = FakeTokenizer(model_max_length=50_000)
    pool_texts = [f"turn body {i} " * 20 for i in range(6)]

    with TokenizerPool(tok, loader=fake_loader) as pool:
        samples = build_fitted_chat_samples(
            pool=pool,
            user_text_pool=pool_texts,
            num_sessions=3,
            num_turns="2",
            input_len="80",
            output_len="20",
            delay_between_turns_dist=None,
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
            shuffle_pool=False,
            log_prefix="test-integration",
        )

    augment_samples_with_images(
        samples,
        fraction=1.0,
        image_count=1,
        image_long_side=64,
        image_aspect_ratio=1.0,
        turn="first",
    )

    # The first user turn of every session should carry exactly one image.
    for session in samples.chat_sessions:
        user_messages = [m for m in session.messages if m.source == "user"]
        assert len(user_messages[0].images) == 1
        for later in user_messages[1:]:
            assert len(later.images) == 0

    driver = CapturingDriver()

    async def run_first_session() -> None:
        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=RequestCounter(max_requests=10),
            chat_session=samples.chat_sessions[0],
            max_chat_len=100_000,
            sampling=SamplingConfig(),
        )

    asyncio.run(run_first_session())

    assert len(driver.calls) >= 1
    first_call = driver.calls[0]
    assert isinstance(first_call.prompt, list)
    first_user_content = first_call.prompt[0].content
    assert any(
        isinstance(block, ImageContentBlock) for block in first_user_content
    )

    print_workload_stats(samples)
    out = capsys.readouterr().out
    assert "Image count (per session)" in out


@pytest.mark.asyncio
async def test_serialized_payload_carries_the_image(
    mocker: MockerFixture,
) -> None:
    """Assert the wire payload, not just the in-memory prompt.

    Checking the `ChatMessage` objects alone cannot catch an endpoint that
    drops images during serialization, which is the failure this whole
    feature is exposed to.
    """
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    image: OpenAIImage = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA"},
    }
    wire = ChatWireHistory()
    request_input = RequestFuncInput(
        model="test-model",
        session_id=None,
        sampling=SamplingConfig(),
        prompt=[
            ChatMessage(role="system", content=[TextContentBlock(text="sys")]),
            ChatMessage(role="user", content=[TextContentBlock(text="hi")]),
        ],
        images=[image],
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=10,
        max_tokens=16,
        ignore_eos=False,
        wire_history=wire,
    )

    client = mocker.MagicMock()

    def post(**_: Any) -> Any:
        response = mocker.AsyncMock()
        response.status = 200
        response.content = StreamBody(
            [
                b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        post_ctx = mocker.AsyncMock()
        post_ctx.__aenter__ = mocker.AsyncMock(return_value=response)
        post_ctx.__aexit__ = mocker.AsyncMock(return_value=None)
        return post_ctx

    client.post = mocker.Mock(side_effect=post)

    await OpenAIChatCompletionsRequestDriver(session=client).request(
        request_input
    )

    payload = sent_json(client.post)
    roles = [m["role"] for m in payload["messages"]]
    user_message = payload["messages"][roles.index("user")]
    assert image in user_message["content"], (
        "the serialized payload must carry the image"
    )
    system_message = payload["messages"][roles.index("system")]
    assert image not in system_message["content"], (
        "the image must not land on the system message"
    )

    # This turn's images are spliced into freshly dumped messages, so it cannot
    # use the history's encoding. The next turn, which carries no images, takes
    # the cached path: nothing from the splice may have been cached with it.
    request_input.images = []
    await OpenAIChatCompletionsRequestDriver(session=client).request(
        request_input
    )
    next_payload = sent_json(client.post)
    next_roles = [m["role"] for m in next_payload["messages"]]
    assert next_payload["messages"][next_roles.index("user")]["content"] == [
        {"type": "text", "text": "hi"}
    ], "a turn's image must not leak into the cached history"
