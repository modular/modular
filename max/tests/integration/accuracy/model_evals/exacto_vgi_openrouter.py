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
"""VGI-Bench samples built the way OpenRouter's benchmark harness builds them.

VGI-Bench (Seldon's Video General Intelligence Bench) is a multiple-choice
question-answering benchmark over long videos, and one of the benchmarks
OpenRouter scores provider endpoints on. This module ports the harness's
``src/benchmarks/vgi-bench`` at :data:`HARNESS_COMMIT`: the prompt, the media
manifest that maps each video to OpenRouter's mirror, the request shape and
the run settings, so an Inspect task can ask a local endpoint exactly what the
harness asks a provider.

The manifest ships next to this module, byte for byte as the harness commits
it. Its ``manifestHash`` is recorded on every sample, as the harness records
it, so two runs can be checked for identical inputs.

Stdlib only, so it unit-tests under bazel and imports into the harness venv.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import exacto_media

HARNESS = "OpenRouterTeam/benchmark-harness"
#: The harness commit this module was ported from.
HARNESS_COMMIT = "ccab6a7988c181f396331ae46474a8e7e18eaf80"
#: OpenRouter's ``benchmark_type`` for this benchmark.
BENCHMARK_ID = "vgi_bench"

DATASET = "Seldon-Technologies/VGIBench"
DATASET_CONFIG = "default"
SPLIT = "train"
#: ``VGI_BENCH_DEFAULT_REVISION``: the tag the harness pins.
DATASET_TAG = "v1.0.1"
#: The commit :data:`DATASET_TAG` points at, loaded instead of the tag so a
#: moved tag cannot change the questions under a pinned runner.
DATASET_REVISION = "e769386ef4d165cad8071f51f1628009d75d9430"
NUM_QUESTIONS = 439

#: ``VGI_BENCH_TEMPERATURE``.
TEMPERATURE = 0.0
#: ``VGI_BENCH_META.defaultEpochs``.
EPOCHS = 1
#: ``DEFAULT_REASONING_EFFORT`` in ``src/harness/constants.ts``; the harness
#: CLI sends it on every request unless the run overrides it.
REASONING_EFFORT = "high"
#: ``rateLimitRetrySchedule``'s default: 6 retries of a 429, a 5xx or a
#: response with no status, so 7 attempts in all.
MAX_ATTEMPTS = 7

MANIFEST_PATH = Path(__file__).with_name("exacto_vgi_media_manifest.json")

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class MediaManifest:
    """``VgiBenchMediaManifest``: where each video is fetched from."""

    revision: str
    manifest_hash: str
    url_by_video_id: Mapping[str, str]


def manifest_hash(videos: Sequence[Mapping[str, object]]) -> str:
    """Recomputes a manifest's hash as ``hashManifestEntries`` does.

    Args:
        videos: The manifest's ``videos`` entries, in file order.

    Returns:
        The hex SHA-256 over each entry's id, URL and file digest.
    """
    hasher = hashlib.sha256()
    for video in videos:
        hasher.update(
            f"{video['videoId']}\0{video['url']}\0{video['sha256']}\n".encode()
        )
    return hasher.hexdigest()


def build_media_manifest(raw: Mapping[str, object]) -> MediaManifest:
    """Validates a parsed manifest, as ``buildMediaManifest``.

    Raises:
        TypeError: If a field is missing or malformed, or a video id repeats.
    """
    revision = raw.get("revision")
    recorded_hash = raw.get("manifestHash")
    videos = raw.get("videos")
    if not isinstance(revision, str) or not revision:
        raise TypeError("vgi-bench media manifest has no revision")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise TypeError("vgi-bench media manifest has no 64-char manifestHash")
    if not isinstance(videos, list) or not videos:
        raise TypeError("vgi-bench media manifest has no videos")
    url_by_video_id: dict[str, str] = {}
    for video in videos:
        video_id = video.get("videoId") if isinstance(video, dict) else None
        url = video.get("url") if isinstance(video, dict) else None
        if not isinstance(video_id, str) or not video_id:
            raise TypeError("vgi-bench media manifest entry has no videoId")
        if not isinstance(url, str) or not url.startswith(
            ("https://", "http://")
        ):
            raise TypeError(f'vgi-bench media manifest "{video_id}" has no url')
        if video_id in url_by_video_id:
            raise TypeError(
                f'vgi-bench media manifest has duplicate videoId "{video_id}"'
            )
        url_by_video_id[video_id] = url
    return MediaManifest(revision, recorded_hash, url_by_video_id)


def load_media_manifest(path: Path = MANIFEST_PATH) -> MediaManifest:
    """Loads the committed manifest and checks it against its own hash.

    The harness trusts its manifest as committed. A copy can be edited, so this
    also recomputes the hash: a URL or digest changed here would otherwise
    still be recorded under the harness's hash.

    Raises:
        ValueError: If the recorded hash does not match the entries.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    manifest = build_media_manifest(raw)
    actual = manifest_hash(raw["videos"])
    if actual != manifest.manifest_hash:
        raise ValueError(
            f"{path} records manifestHash {manifest.manifest_hash} but its "
            f"entries hash to {actual}"
        )
    return manifest


def build_prompt(question: str, answers: Sequence[str]) -> str:
    """Builds the user prompt, as ``buildVgiBenchPrompt``.

    Raises:
        ValueError: If there are more answers than letters.
    """
    if len(answers) > len(_LETTERS):
        raise ValueError(f"{len(answers)} answers, at most 26 have letters")
    options = "\n".join(
        f"{_LETTERS[index]}) {answer}" for index, answer in enumerate(answers)
    )
    letters = _LETTERS[: len(answers)]
    return (
        "Watch the video and answer the multiple choice question about it. "
        "The last line of your response should be of the following format: "
        f"'Answer: $LETTER' (without quotes) where LETTER is one of {letters}."
        f"\n\nQuestion: {question}\n\n{options}"
    )


@dataclass(frozen=True)
class VgiSample:
    """One question, ready to send: what ``vgiBenchRecordToSample`` builds."""

    id: str
    prompt: str
    target: str
    video_url: str
    metadata: Mapping[str, object] = field(default_factory=dict)


def _as_str(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str):
        raise TypeError(f'vgi-bench record field "{name}" was not a string')
    return value


def _as_number(record: Mapping[str, object], name: str) -> int:
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TypeError(f'vgi-bench record field "{name}" was not a number')
    # The ids and indices are integers; a float id would print differently
    # from JavaScript's number formatting, so refuse rather than drift.
    if value != int(value):
        raise TypeError(f'vgi-bench record field "{name}" was not an integer')
    return int(value)


def _as_str_list(record: Mapping[str, object], name: str) -> list[str]:
    value = record.get(name)
    if not isinstance(value, list):
        raise TypeError(f'vgi-bench record field "{name}" was not an array')
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(
                f'vgi-bench record field "{name}[{index}]" was not a string'
            )
    return value


def record_to_sample(
    record: Mapping[str, object], manifest: MediaManifest
) -> VgiSample:
    """Builds one question as the harness does under the media manifest.

    Under the pinned revision the harness always resolves the video through
    the manifest; the record's own ``video_url`` points at Seldon's CDN, where
    many files no longer resolve.

    Raises:
        TypeError: If a field is malformed, the video is not in the manifest,
            or ``correct_answer`` is out of range.
    """
    question_id = _as_number(record, "question_id")
    video_id = _as_str(record, "video_id")
    _as_str(record, "video_url")
    video_url = manifest.url_by_video_id.get(video_id)
    if video_url is None:
        raise TypeError(
            f'vgi-bench video_id="{video_id}" is missing from media manifest '
            f"{manifest.manifest_hash}"
        )
    question = _as_str(record, "question")
    question_type = _as_str(record, "question_type")
    answers = _as_str_list(record, "answers")
    correct = _as_number(record, "correct_answer")
    if not 0 <= correct < len(answers):
        raise TypeError(
            f"vgi-bench record question_id={question_id} has "
            f"correct_answer={correct} outside answers range "
            f"[0, {len(answers)})"
        )
    return VgiSample(
        id=f"{BENCHMARK_ID}-{question_id}",
        prompt=build_prompt(question, answers),
        target=_LETTERS[correct],
        video_url=video_url,
        metadata={
            "question_id": question_id,
            "video_id": video_id,
            "question_type": question_type,
            "family": question_type.split("/", 1)[0],
            "media_manifest_hash": manifest.manifest_hash,
        },
    )


def user_message(sample: VgiSample) -> dict[str, object]:
    """Returns the one message the harness sends: the video, then the prompt.

    The harness sends an ``input_video`` part to OpenRouter's Responses API,
    which forwards it to a chat-completions provider as this ``video_url``
    part. There is no system message.
    """
    return {
        "role": "user",
        "content": [
            exacto_media.video_url_part(sample.video_url),
            {"type": "text", "text": sample.prompt},
        ],
    }


def request_body(
    sample: VgiSample,
    model: str,
    reasoning_effort: str | None = REASONING_EFFORT,
) -> dict[str, object]:
    """Returns the chat-completions body a provider receives for ``sample``.

    No ``max_tokens``: the harness sets none, so the server's limit applies.
    """
    body: dict[str, object] = {
        "model": model,
        "messages": [user_message(sample)],
        "temperature": TEMPERATURE,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


class ErrorOutcome(enum.Enum):
    """What the harness does with a request that failed after its retries."""

    #: Scored 0 and kept in the denominator.
    INCORRECT = "incorrect"
    #: Left out of the score, as ``ScoreValue.Skipped``.
    SKIPPED = "skipped"
    #: Stops the run: the endpoint is unusable, not the answer wrong.
    ABORT = "abort"


def model_error_outcome(status: int | None) -> ErrorOutcome:
    """Classifies a failed request as ``evaluateSample`` does in ``run.ts``.

    A 401, 403 or 404, or a failure with no HTTP status, aborts the run; a 429
    or 5xx is skipped; any other status, such as a 400 for a video the server
    could not fetch or a 413 for one it would not take, scores 0.
    """
    if status is None or status in (401, 403, 404):
        return ErrorOutcome.ABORT
    if status == 429 or status >= 500:
        return ErrorOutcome.SKIPPED
    return ErrorOutcome.INCORRECT


_STATUS_IN_MESSAGE = re.compile(r"\bError code: (\d{3})\b")


def status_from_error_text(text: str) -> int | None:
    """Reads the HTTP status the OpenAI SDK writes into an error's text.

    Inspect re-raises a 400, 403 or 422 as a ``RuntimeError`` carrying the
    SDK error's ``repr``, so the status survives only as text.
    """
    match = _STATUS_IN_MESSAGE.search(text)
    return int(match.group(1)) if match else None
