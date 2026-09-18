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
"""Sample data model and JSONL persistence for SpecDecCheck.

A :class:`Sample` is one completion of one prompt on one server. Collection
and analysis are separate stages, so this format is the boundary between them
and depends on nothing but the standard library. A file's first line is a
:class:`SamplesHeader` record carrying what the samples themselves cannot:
the server and model they came from, the prompts as token ids, and the knobs
they were drawn with.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
"""The samples file format this module writes, and the only one it reads."""

TOOL = "spec-dec-check"
"""The tool name every header records, so a stray file is recognizable."""

_HEADER_KIND = "header"


@dataclass(frozen=True)
class Sample:
    """One completion returned by one server for one prompt.

    Args:
        server: Which server produced it, ``"baseline"`` or ``"test"``.
        prompt_index: Index of the prompt within the run's prompt set.
        request_index: Index of this request among the prompt's samples.
        seed: The seed sent with the request, always ``>= 1``, since the
            server treats ``0`` as unset and substitutes a random one.
        prompt_token_ids: The post-template prompt ids the server reported.
        token_ids: The generated ids, EOS included when it ended on one.
        finish_reason: The OpenAI finish reason.
        completion_tokens: The server's own output-token count, or ``None``
            when the response carried no usage block.
        completed_at: A :func:`time.monotonic` stamp from when the response
            arrived, which orders a server's temporal halves.
        text: The decoded completion, reasoning included.
    """

    server: str
    prompt_index: int
    request_index: int
    seed: int
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    finish_reason: str
    completion_tokens: int | None
    completed_at: float
    text: str


@dataclass(frozen=True)
class SamplesHeader:
    """The first record of a samples file: how its samples were collected.

    Args:
        format_version: The file format, :data:`FORMAT_VERSION`.
        tool: The tool that wrote the file, :data:`TOOL`.
        created_at: When the collection finished, ISO-8601 in UTC.
        url: The server the samples were drawn from.
        endpoint: The endpoint the prompts were sent to.
        model: The model name that server reported.
        server: The role they were collected as. A later run may use the
            file in either role.
        num_samples: How many completions were drawn per prompt.
        max_tokens: The token budget each completion was given, which caps
            the depths the file can answer for.
        seed: The base for the per-request seeds.
        max_concurrency: The cap on requests in flight at once.
        request_params: The extra body fields sent with every request,
            empty when the model's defaults were used.
        prompts: The prompts as token ids, in prompt-index order. Ids rather
            than text, so a later run can send a live server the identical
            prefix without going through a tokenizer.
    """

    format_version: int
    tool: str
    created_at: str
    url: str
    endpoint: str
    model: str
    server: str
    num_samples: int
    max_tokens: int
    seed: int
    max_concurrency: int
    request_params: Mapping[str, object]
    prompts: tuple[tuple[int, ...], ...]


def make_header(
    *,
    url: str,
    endpoint: str,
    model: str,
    server: str,
    num_samples: int,
    max_tokens: int,
    seed: int,
    max_concurrency: int,
    request_params: Mapping[str, object],
    prompts: Iterable[Sequence[int]],
) -> SamplesHeader:
    """Builds a header stamped with the current UTC time.

    Args:
        url: The server the samples were drawn from.
        endpoint: The endpoint the prompts were sent to.
        model: The model name that server reported.
        server: The role the samples were collected as.
        num_samples: How many completions were drawn per prompt.
        max_tokens: The output token budget each completion was given.
        seed: The base for the per-request seeds.
        max_concurrency: The cap on requests in flight at once.
        request_params: The extra request body fields sent every time.
        prompts: The prompts as token ids, in prompt-index order.

    Returns:
        The header, with the format version and tool name filled in.
    """
    return SamplesHeader(
        format_version=FORMAT_VERSION,
        tool=TOOL,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        url=url,
        endpoint=endpoint,
        model=model,
        server=server,
        num_samples=num_samples,
        max_tokens=max_tokens,
        seed=seed,
        max_concurrency=max_concurrency,
        request_params=dict(request_params),
        prompts=tuple(tuple(prompt) for prompt in prompts),
    )


def save_samples(
    path: Path, header: SamplesHeader, samples: Iterable[Sample]
) -> None:
    """Writes a header and its samples to ``path`` as JSONL.

    Creates the parent directory and overwrites any existing file.

    Args:
        path: The JSONL file to write.
        header: The header, written as the first line.
        samples: The samples to write, in the order they should appear.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        record = {"kind": _HEADER_KIND, **asdict(header)}
        file.write(json.dumps(record) + "\n")
        for sample in samples:
            file.write(json.dumps(asdict(sample)) + "\n")


def load_samples(path: Path) -> tuple[SamplesHeader, list[Sample]]:
    """Reads a file written by :func:`save_samples`.

    Args:
        path: The JSONL file to read.

    Returns:
        The header and the samples in file order.

    Raises:
        ValueError: If the file is empty or malformed.
    """
    header: SamplesHeader | None = None
    samples: list[Sample] = []
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{where}: expected a JSON object, got"
                    f" {type(record).__name__}"
                )
            if header is None:
                header = _header_from_record(record, where)
                continue
            try:
                samples.append(_sample_from_record(record))
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
    if header is None:
        raise ValueError(
            f"{path} is empty, so it carries no header. A samples file starts"
            f' with a {{"kind": "{_HEADER_KIND}"}} line.'
        )
    return header, samples


def _header_from_record(
    record: Mapping[str, object], where: str
) -> SamplesHeader:
    """Builds the header from the first decoded JSON object of a file."""
    kind = record.get("kind")
    if kind != _HEADER_KIND:
        raise ValueError(
            f"{where}: a samples file must start with a header record, a JSON"
            f' object whose "kind" is "{_HEADER_KIND}", but this line has kind'
            f" {kind!r}. Every file --outdir writes has one; a file"
            " without it was written by something else."
        )
    try:
        version = _as_int(record, "format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"header format_version is {version}, but this build reads"
                f" only version {FORMAT_VERSION}"
            )
        return SamplesHeader(
            format_version=version,
            tool=_as_str(record, "tool"),
            created_at=_as_str(record, "created_at"),
            url=_as_str(record, "url"),
            # Files written before the endpoint was recorded used completions.
            endpoint=(
                _as_str(record, "endpoint")
                if "endpoint" in record
                else "/v1/completions"
            ),
            model=_as_str(record, "model"),
            server=_as_str(record, "server"),
            num_samples=_as_int(record, "num_samples"),
            max_tokens=_as_int(record, "max_tokens"),
            seed=_as_int(record, "seed"),
            max_concurrency=_as_int(record, "max_concurrency"),
            # Files written before extra fields could be sent used none.
            request_params=(
                _as_request_params(record, "request_params")
                if "request_params" in record
                else {}
            ),
            prompts=_as_prompts(record, "prompts"),
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _as_request_params(
    record: Mapping[str, object], key: str
) -> dict[str, object]:
    """Reads a JSON object of request fields, as ``json.loads`` gave it."""
    value = _field(record, key)
    if not isinstance(value, dict):
        raise ValueError(
            f"{key}: expected a JSON object, got {type(value).__name__}"
        )
    return dict(value)


def _sample_from_record(record: Mapping[str, object]) -> Sample:
    """Builds a sample from one decoded JSON object."""
    return Sample(
        server=_as_str(record, "server"),
        prompt_index=_as_int(record, "prompt_index"),
        request_index=_as_int(record, "request_index"),
        seed=_as_int(record, "seed"),
        prompt_token_ids=_as_int_tuple(record, "prompt_token_ids"),
        token_ids=_as_int_tuple(record, "token_ids"),
        finish_reason=_as_str(record, "finish_reason"),
        completion_tokens=_as_optional_int(record, "completion_tokens"),
        completed_at=_as_float(record, "completed_at"),
        text=_as_str(record, "text") if "text" in record else "",
    )


def _field(record: Mapping[str, object], key: str) -> object:
    if key not in record:
        raise ValueError(f"missing field {key!r}")
    return record[key]


def _as_str(record: Mapping[str, object], key: str) -> str:
    value = _field(record, key)
    if not isinstance(value, str):
        raise ValueError(
            f"field {key!r} must be a string, got {type(value).__name__}"
        )
    return value


def _is_int(value: object) -> bool:
    # ``bool`` subclasses ``int``, so a JSON ``true`` would otherwise pass.
    return isinstance(value, int) and not isinstance(value, bool)


def _as_int(record: Mapping[str, object], key: str) -> int:
    value = _field(record, key)
    if not _is_int(value):
        raise ValueError(
            f"field {key!r} must be an integer, got {type(value).__name__}"
        )
    assert isinstance(value, int)
    return value


def _as_optional_int(record: Mapping[str, object], key: str) -> int | None:
    if _field(record, key) is None:
        return None
    return _as_int(record, key)


def _as_float(record: Mapping[str, object], key: str) -> float:
    value = _field(record, key)
    if not isinstance(value, float) and not _is_int(value):
        raise ValueError(
            f"field {key!r} must be a number, got {type(value).__name__}"
        )
    assert isinstance(value, (int, float))
    return float(value)


def _as_int_tuple(record: Mapping[str, object], key: str) -> tuple[int, ...]:
    return _int_tuple(_field(record, key), key)


def _as_prompts(
    record: Mapping[str, object], key: str
) -> tuple[tuple[int, ...], ...]:
    value = _field(record, key)
    if not isinstance(value, list):
        raise ValueError(
            f"field {key!r} must be a list, got {type(value).__name__}"
        )
    return tuple(
        _int_tuple(element, f"{key}[{position}]")
        for position, element in enumerate(value)
    )


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"field {name!r} must be a list, got {type(value).__name__}"
        )
    ids: list[int] = []
    for position, element in enumerate(value):
        if not _is_int(element):
            raise ValueError(
                f"field {name!r} element {position} must be an integer, got"
                f" {type(element).__name__}"
            )
        assert isinstance(element, int)
        ids.append(element)
    return tuple(ids)
