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

"""Shared helpers for benchmark entrypoints."""

from __future__ import annotations

import resource
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def wait_for_server_ready(
    host: str,
    port: int,
    path: str = "health",
    *,
    timeout_s: int = 120 * 60,
    interval_s: float = 5.0,
) -> float:
    """Polls ``http://<host>:<port>/<path>`` until it responds with HTTP 200.

    Always attempts at least one request, even when ``timeout_s == 0``, so
    standalone callers get a fast error on misconfigured ``host``/``port``.
    Returns the elapsed seconds; raises :class:`RuntimeError` on timeout.
    Stdlib-only so both the orchestrator (``benchmark.py``) and the load
    generator (``benchmark_serving.py``) can share one implementation.
    """
    url = f"http://{host}:{port}/{path}"
    start = time.monotonic()
    deadline = start + timeout_s
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Server at {url} not ready after {timeout_s}s")
        time.sleep(interval_s)


def resolve_revision(pretrained_model_name_or_path: str) -> str | None:
    """Resolve the HuggingFace Hub commit SHA for a repo id.

    Returns ``None`` for local paths, private/gated repos without auth, and
    other lookup failures so callers fall back to revision-less loading.
    Used to pin worker tokenizer loads to a specific snapshot so they hit
    the local cache without re-checking the Hub on every spawn.
    """
    try:
        return HfApi().model_info(pretrained_model_name_or_path).sha
    except Exception:
        return None


def get_tokenizer(
    pretrained_model_name_or_path: str,
    *,
    revision: str | None,
    model_max_length: int | None = None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a tokenizer for a benchmark model.

    ``revision`` is explicit; callers should resolve it once via
    :func:`resolve_revision` (or reuse a previously resolved value) so that
    repeated loads across worker processes hit the same cached snapshot.
    """
    tokenizer_kwargs: dict[str, bool | int | str | None] = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    if model_max_length is not None:
        tokenizer_kwargs["model_max_length"] = model_max_length
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, **tokenizer_kwargs
    )
    # Stash the resolved revision so downstream consumers (e.g. the worker
    # tokenizer pool) can pin worker loads to the same snapshot without
    # re-resolving against the Hub. Transformers does not expose this on
    # the tokenizer instance itself.
    tokenizer._resolved_revision = revision
    try:
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        architectures = getattr(config, "architectures", None) or []
    except (ValueError, OSError) as exc:
        print(
            f"Warning: AutoConfig.from_pretrained failed for "
            f"{pretrained_model_name_or_path!r}: {exc}. "
            "Skipping architecture-specific tokenizer overrides."
        )
        architectures = []
    if "KimiK25ForConditionalGeneration" in architectures:
        original_encode = tokenizer.encode

        def encode(text: Any, *args: Any, **kwargs: Any) -> list[int]:
            kwargs.pop("add_special_tokens", None)
            kwargs["allow_special_tokens"] = True
            return original_encode(text, *args, **kwargs)

        tokenizer.encode = encode
    return tokenizer


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    """Raise the open-file soft limit when benchmarks need many sockets."""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as error:
            print(f"Fail to set RLIMIT_NOFILE: {error}")


def print_section(title: str, char: str = "-") -> None:
    """Print a formatted section header for benchmark output."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))


def is_castable_to_int(x: str) -> bool:
    """Return True if *x* can be converted to an ``int`` without error."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def int_or_none(x: str) -> int | None:
    """Parse *x* as an ``int``, returning ``None`` for the literal ``'none'``."""
    if x.lower() == "none":
        return None
    return int(x)


def argmedian(x: np.ndarray) -> int:
    """Return the index of the median value in *x*."""
    return int(np.flatnonzero(x == np.percentile(x, 50, method="nearest"))[0])


_T = TypeVar("_T")


def parse_comma_separated(
    value: str | None,
    convert: Callable[[str], _T],
    *,
    default: _T | None = None,
) -> list[_T]:
    """Split a comma-separated string and convert each element.

    Args:
        value: Comma-separated string (e.g. ``"1,2,4"``), or ``None``.
        convert: Callable applied to each stripped token.
        default: If *value* is ``None``, return ``[default]``.
            When both *value* and *default* are ``None`` the result is
            ``[None]`` (useful for optional concurrency).
    """
    if value is None:
        return [default]  # type: ignore[list-item]
    return [convert(x.strip()) for x in value.split(",")]
