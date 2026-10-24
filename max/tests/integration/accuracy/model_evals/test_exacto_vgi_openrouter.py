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
"""Tests for the port of OpenRouter's VGI-Bench and its MCQ answer extraction.

The expected prompts and extracted letters were printed by the harness's own
``buildVgiBenchPrompt`` and ``extractMcqAnswer`` (commit
``exacto_vgi_openrouter.HARNESS_COMMIT``) under node. The records are made up,
so no benchmark question is committed here.
"""

import copy
import json
from pathlib import Path

import exacto_mcq as mcq
import exacto_media as media
import exacto_vgi_openrouter as vgi
import pytest

PROMPT_HEAD = (
    "Watch the video and answer the multiple choice question about it. The "
    "last line of your response should be of the following format: 'Answer: "
    "$LETTER' (without quotes) where LETTER is one of "
)

HARNESS_PROMPTS = [
    (
        "What color is the car in the first scene?",
        ["red", "blue", "green", "yellow"],
        PROMPT_HEAD + "ABCD.\n\nQuestion: What color is the car in the first "
        "scene?\n\nA) red\nB) blue\nC) green\nD) yellow",
    ),
    (
        "Does the narrator say “stop” before the cut?",
        ["Yes.", "This does not happen in the video."],
        PROMPT_HEAD + "AB.\n\nQuestion: Does the narrator say “stop” "
        "before the cut?\n\nA) Yes.\nB) This does not happen in the video.",
    ),
    (
        "Which tool is used after the $5 bill — the one on the café table?",
        ["A wrench", "Answer: B", "Pliers {option_a}"],
        PROMPT_HEAD + "ABC.\n\nQuestion: Which tool is used after the $5 bill "
        "— the one on the café table?\n\nA) A wrench\nB) Answer: B\n"
        "C) Pliers {option_a}",
    ),
    (
        "How many times does the bell ring?\nCount carefully.",
        ["1", "2", "3", "4", "None of the above"],
        PROMPT_HEAD + "ABCDE.\n\nQuestion: How many times does the bell ring?"
        "\nCount carefully.\n\nA) 1\nB) 2\nC) 3\nD) 4\nE) None of the above",
    ),
]

HARNESS_EXTRACTIONS = [
    ("The man pours the milk first.\n\nAnswer: B", "B"),
    ("Answer: $C", "C"),
    ("**Answer:** E", "E"),
    ("**Answer**: d", "D"),
    ("answer: (b)", "B"),
    ("Final answer - D", "D"),
    ("I would go with Option C here.", "C"),
    ("\\boxed{A}", "A"),
    ("\\boxed{\\text{B}}", "B"),
    ("The correct choice is (C).", "C"),
    ("It is **D**.", "D"),
    ("**B) blue**", "B"),
    ("Reasoning...\nC\n", "C"),
    ("Reasoning...\r\nE.\r\n", "E"),
    ("答案\N{FULLWIDTH COLON}\N{FULLWIDTH LATIN CAPITAL LETTER C}", "C"),
    ("Réponse : B", "B"),
    ("Жавоб: D", "D"),
    ("Answer:\u200b\u200b\u200b\u200b\u200b\u200bB", "B"),
    ("Answer: \N{ZERO WIDTH NO-BREAK SPACE}C", "C"),
    ("Answer:\u001cC", None),
    ("\N{LATIN SMALL LETTER DOTLESS I}dahun: A", None),
    ("Answer: \N{LATIN SMALL LETTER DOTLESS I}", None),
    ("The answer is clear from the second scene", None),
    ("A", "A"),
    ("", None),
    ("Both A and B are plausible, but the video shows B.", None),
    ("Answer: The video shows B", "T"),
    ("line one\N{LINE SEPARATOR}D\N{LINE SEPARATOR}line three", "D"),
]

RECORD = {
    "question_id": 42,
    "video_id": "_-zi-NfJx0M",
    "video_url": "https://cdn.seldon.global/_-zi-NfJx0M.mp4",
    "question": "What color is the car in the first scene?",
    "question_type": "contrastive-hard-negative/transcript_visual",
    "answers": ["red", "blue", "green", "yellow"],
    "correct_answer": 1,
}


@pytest.fixture(scope="module")
def manifest() -> vgi.MediaManifest:
    return vgi.load_media_manifest()


@pytest.mark.parametrize(("question", "answers", "expected"), HARNESS_PROMPTS)
def test_prompt_is_byte_identical_to_the_harness(
    question: str, answers: list[str], expected: str
) -> None:
    assert vgi.build_prompt(question, answers) == expected


@pytest.mark.parametrize(("text", "expected"), HARNESS_EXTRACTIONS)
def test_extraction_matches_the_harness(
    text: str, expected: str | None
) -> None:
    assert mcq.extract_mcq_answer(text) == expected


def test_scoring_counts_a_parse_failure_as_wrong() -> None:
    assert mcq.score_mcq("I could not tell.", "B") == mcq.McqScore(
        correct=False, answer=None, explanation="No answer found"
    )
    assert mcq.score_mcq("Answer: b", " b ").correct
    assert mcq.score_mcq(None, "A").answer is None


def test_the_committed_manifest_is_the_harness_one(
    manifest: vgi.MediaManifest,
) -> None:
    assert manifest.revision == vgi.DATASET_TAG
    assert manifest.manifest_hash == (
        "872691450ddc053ae0b374f554ff8eb6b19fe379aab29e22ec6a0a7ef37bb738"
    )
    assert len(manifest.url_by_video_id) == 323
    assert all(
        url == f"https://vgi-bench-mirror.openrouter.ai/{video_id}.mp4"
        for video_id, url in manifest.url_by_video_id.items()
    )


def test_a_record_resolves_its_video_through_the_manifest(
    manifest: vgi.MediaManifest,
) -> None:
    sample = vgi.record_to_sample(RECORD, manifest)
    assert sample.video_url == (
        "https://vgi-bench-mirror.openrouter.ai/_-zi-NfJx0M.mp4"
    )
    assert sample.id == "vgi_bench-42"
    assert sample.target == "B"
    assert sample.prompt == HARNESS_PROMPTS[0][2]
    assert sample.metadata == {
        "question_id": 42,
        "video_id": "_-zi-NfJx0M",
        "question_type": "contrastive-hard-negative/transcript_visual",
        "family": "contrastive-hard-negative",
        "media_manifest_hash": manifest.manifest_hash,
    }


def test_a_video_missing_from_the_manifest_is_rejected(
    manifest: vgi.MediaManifest,
) -> None:
    with pytest.raises(TypeError, match="missing from media manifest"):
        vgi.record_to_sample({**RECORD, "video_id": "nope"}, manifest)


def test_malformed_records_are_rejected(manifest: vgi.MediaManifest) -> None:
    with pytest.raises(TypeError, match="outside answers range"):
        vgi.record_to_sample({**RECORD, "correct_answer": 4}, manifest)
    with pytest.raises(TypeError, match=r'"answers\[1\]" was not a string'):
        vgi.record_to_sample({**RECORD, "answers": ["a", 2]}, manifest)
    with pytest.raises(TypeError, match='"question_id" was not a number'):
        vgi.record_to_sample({**RECORD, "question_id": "42"}, manifest)


def test_family_is_the_whole_type_without_a_slash(
    manifest: vgi.MediaManifest,
) -> None:
    sample = vgi.record_to_sample(
        {**RECORD, "question_type": "counting"}, manifest
    )
    assert sample.metadata["family"] == "counting"


def test_an_edited_manifest_fails_its_hash(tmp_path: Path) -> None:
    with open(vgi.MANIFEST_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    edited = copy.deepcopy(raw)
    edited["videos"][0]["url"] = "https://example.com/other.mp4"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="hash to"):
        vgi.load_media_manifest(path)
    duplicated = copy.deepcopy(raw)
    duplicated["videos"].append(duplicated["videos"][0])
    with pytest.raises(TypeError, match="duplicate videoId"):
        vgi.build_media_manifest(duplicated)


def test_request_is_one_user_turn_with_the_video_first(
    manifest: vgi.MediaManifest,
) -> None:
    sample = vgi.record_to_sample(RECORD, manifest)
    body = vgi.request_body(sample, "zai-org/GLM-5.3-Flash")
    assert body == {
        "model": "zai-org/GLM-5.3-Flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://vgi-bench-mirror.openrouter.ai/"
                            "_-zi-NfJx0M.mp4"
                        },
                    },
                    {"type": "text", "text": sample.prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "reasoning_effort": "high",
    }
    assert "reasoning_effort" not in vgi.request_body(sample, "m", None)


def test_a_marker_becomes_the_harness_video_part(
    manifest: vgi.MediaManifest,
) -> None:
    sample = vgi.record_to_sample(RECORD, manifest)
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": media.video_url_marker(sample.video_url),
                },
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    assert media.expand_video_url_markers(messages) == 1
    assert messages == [vgi.user_message(sample)]


def test_a_marker_refuses_inlined_media() -> None:
    with pytest.raises(ValueError, match="http"):
        media.video_url_marker("data:video/mp4;base64,AAAA")


def test_plain_text_is_not_rewritten() -> None:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "exacto:video_url https://x"}
    ]
    assert media.expand_video_url_markers(messages) == 0


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (400, vgi.ErrorOutcome.INCORRECT),
        (413, vgi.ErrorOutcome.INCORRECT),
        (422, vgi.ErrorOutcome.INCORRECT),
        (408, vgi.ErrorOutcome.INCORRECT),
        (429, vgi.ErrorOutcome.SKIPPED),
        (500, vgi.ErrorOutcome.SKIPPED),
        (503, vgi.ErrorOutcome.SKIPPED),
        (401, vgi.ErrorOutcome.ABORT),
        (403, vgi.ErrorOutcome.ABORT),
        (404, vgi.ErrorOutcome.ABORT),
        (None, vgi.ErrorOutcome.ABORT),
    ],
)
def test_failed_requests_count_as_the_harness_counts_them(
    status: int | None, outcome: vgi.ErrorOutcome
) -> None:
    assert vgi.model_error_outcome(status) is outcome


def test_status_is_read_from_a_wrapped_sdk_error() -> None:
    text = (
        "BadRequestError(\"Error code: 400 - {'detail': 'video fetch "
        "failed'}\")\n\nRequest:\n{}"
    )
    assert vgi.status_from_error_text(text) == 400
    assert vgi.status_from_error_text("RuntimeError('boom')") is None
