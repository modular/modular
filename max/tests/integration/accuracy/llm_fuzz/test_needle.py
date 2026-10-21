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

import json
import random

import pytest
from scenarios._needle import (
    RECALL_QUESTION,
    CacheHit,
    TurnReply,
    build_haystack,
    classify_cache_hit,
    expected_cached_tokens,
    find_leaked_needles,
    first_turn_content,
    generate_needle,
    haystack_token_target,
    long_haystack_token_target,
    needle_recalled,
    parse_reply,
    split_think_block,
)


def _body(
    content: str = "K7X9Q2ML",
    *,
    prompt_tokens: object = 1500,
    cached_tokens: object = 1408,
    details: bool = True,
    reasoning: str | None = None,
    finish_reason: str | None = "stop",
) -> str:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    usage: dict[str, object] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 4,
        "total_tokens": 1504,
    }
    if details:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return json.dumps(
        {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
    )


def test_needle_mixes_letters_and_digits_without_ambiguous_glyphs() -> None:
    rng = random.Random(0)
    joined = ""
    for _ in range(300):
        needle = generate_needle(rng)
        assert len(needle) == 8
        assert any(c.isalpha() for c in needle)
        assert any(c.isdigit() for c in needle)
        joined += needle
    assert not set("ILO01") & set(joined)


def test_needle_rejects_too_short() -> None:
    with pytest.raises(ValueError):
        generate_needle(random.Random(0), length=1)


def test_haystack_embeds_needle_once_and_is_unique_per_conversation() -> None:
    a = build_haystack(random.Random(1), "K7X9Q2ML", 1024, 0.5)
    b = build_haystack(random.Random(2), "K7X9Q2ML", 1024, 0.5)
    assert a.count("K7X9Q2ML") == 1
    assert a != b
    a_lines = set(a.splitlines()[2:])
    b_lines = set(b.splitlines()[2:])
    # Only the needle line, which carries no conversation id, is shared.
    assert a_lines & b_lines == {
        "The secret access code is K7X9Q2ML. Remember this code exactly."
    }


@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_haystack_places_needle_by_fraction(fraction: float) -> None:
    text = build_haystack(random.Random(3), "K7X9Q2ML", 2048, fraction)
    lines = text.splitlines()[2:]
    pos = next(i for i, line in enumerate(lines) if "K7X9Q2ML" in line)
    assert pos == int(fraction * (len(lines) - 1))


def test_haystack_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        build_haystack(random.Random(0), "K7X9Q2ML", 512, 1.5)


def test_haystack_scales_with_target() -> None:
    small = build_haystack(random.Random(4), "K7X9Q2ML", 512, 0.5)
    large = build_haystack(random.Random(4), "K7X9Q2ML", 2048, 0.5)
    assert 3 < len(large) / len(small) < 5


def test_first_turn_ends_with_the_recall_question() -> None:
    assert first_turn_content("log").endswith(RECALL_QUESTION)


def test_split_think_block_handles_nesting_and_truncation() -> None:
    visible, thinking = split_think_block(
        "<think>outer <think>inner</think> more</think>K7X9Q2ML"
    )
    assert visible == "K7X9Q2ML"
    assert thinking == "outer <think>inner</think> more"

    visible, thinking = split_think_block("<think>the code is K7X9Q2ML")
    assert visible == ""
    assert "K7X9Q2ML" in thinking

    visible, thinking = split_think_block("reading K7X9Q2ML</think>done")
    assert visible == "done"
    assert "K7X9Q2ML" in thinking


def test_recall_tolerates_punctuation_and_case() -> None:
    assert needle_recalled("The code is k7-x9 q2ml.", "K7X9Q2ML")
    assert not needle_recalled("The code is K7X9Q2MK.", "K7X9Q2ML")


def test_leaked_needles_excludes_own() -> None:
    pool = ["K7X9Q2ML", "AB23CD45", "ZZ99YY88"]
    assert find_leaked_needles("K7X9Q2ML and ab23cd45", "K7X9Q2ML", pool) == [
        "AB23CD45"
    ]
    assert find_leaked_needles("K7X9Q2ML", "K7X9Q2ML", pool) == []


def test_parse_reply_reads_fields() -> None:
    reply = parse_reply(200, _body(reasoning="thinking"))
    assert reply.ok
    assert reply.visible == "K7X9Q2ML"
    assert reply.reasoning == "thinking"
    assert reply.finish_reason == "stop"
    assert reply.prompt_tokens == 1500
    assert reply.cached_tokens == 1408
    assert reply.reports_usage


def test_parse_reply_moves_folded_thinking_out_of_visible() -> None:
    reply = parse_reply(200, _body("<think>K7X9Q2ML?</think>K7X9Q2ML"))
    assert reply.visible == "K7X9Q2ML"
    assert reply.reasoning == "K7X9Q2ML?"
    assert reply.content.startswith("<think>")


def test_parse_reply_non_200_and_malformed() -> None:
    assert not parse_reply(503, "", "HTTP 503").ok
    assert parse_reply(0, "", "TIMEOUT").error == "TIMEOUT"
    malformed = parse_reply(200, "{not json")
    assert malformed.status == 200 and not malformed.ok
    assert not parse_reply(200, json.dumps({"choices": []})).ok


@pytest.mark.parametrize(
    "prompt_tokens, cached_tokens, details",
    [
        (None, 5, True),
        (0, 5, True),
        (True, 5, True),
        (1500, None, True),
        (1500, float("nan"), True),
        (1500, 5, False),
    ],
)
def test_parse_reply_rejects_unusable_counts(
    prompt_tokens: object, cached_tokens: object, details: bool
) -> None:
    reply = parse_reply(
        200,
        _body(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            details=details,
        ),
    )
    assert reply.ok
    assert not reply.reports_usage


def test_empty_answer_is_not_scorable() -> None:
    reply = parse_reply(200, _body("  ", finish_reason="length"))
    assert reply.ok and not reply.has_answer


def test_expected_cached_tokens_rounds_down_to_a_page() -> None:
    assert expected_cached_tokens(1500, 128) == 1408
    assert expected_cached_tokens(128, 128) == 128
    assert expected_cached_tokens(100, 128) == 0
    with pytest.raises(ValueError):
        expected_cached_tokens(100, 0)


def test_classify_cache_hit() -> None:
    def reply(cached: int) -> TurnReply:
        return TurnReply(200, prompt_tokens=1600, cached_tokens=cached)

    assert classify_cache_hit(reply(1408), 1500, 128) is CacheHit.FULL
    assert classify_cache_hit(reply(1536), 1500, 128) is CacheHit.FULL
    assert classify_cache_hit(reply(1280), 1500, 128) is CacheHit.PARTIAL
    assert classify_cache_hit(reply(0), 1500, 128) is CacheHit.MISS
    assert classify_cache_hit(reply(1408), None, 128) is CacheHit.UNKNOWN
    unreported = TurnReply(200, prompt_tokens=1600, cached_tokens=None)
    assert classify_cache_hit(unreported, 1500, 128) is CacheHit.UNKNOWN


def test_haystack_token_target_bounds() -> None:
    assert haystack_token_target(4096) == 1024
    assert haystack_token_target(163840) == 2048
    assert haystack_token_target(512) == 256


def test_long_haystack_token_target_is_half_the_window_capped() -> None:
    assert long_haystack_token_target(32768) == 16384
    assert long_haystack_token_target(163840) == 81920
    assert long_haystack_token_target(1048576) == 131072
    assert long_haystack_token_target(256) == 256
