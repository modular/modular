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

"""Per-request latency targets for the serving benchmark's goodput metric.

Goodput counts the requests whose own TTFT, TPOT, and end-to-end latency each
meet the targets given with ``--goodput``. The flag mirrors vLLM's
``benchmark_serving.py``: space-separated ``key:value`` pairs in milliseconds,
with keys ``ttft``, ``tpot``, and ``e2el``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GOODPUT_KEYS: tuple[str, ...] = ("ttft", "tpot", "e2el")


def goodput_tokens(value: object) -> list[str]:
    """Flattens a ``--goodput`` value into ``key:value`` tokens.

    Accepts the forms the value arrives in: CLI tokens, a single
    space-separated string (an environment variable or a YAML string), a YAML
    list, or a YAML mapping of key to milliseconds.

    Args:
        value: The raw ``--goodput`` value.

    Returns:
        One ``key:value`` string per target, in the order given.

    Raises:
        ValueError: If the value is not a string, a list of strings, or a
            mapping.
    """
    if isinstance(value, str):
        return value.split()
    if isinstance(value, Mapping):
        return [f"{k}:{v}" for k, v in value.items()]
    if isinstance(value, Sequence):
        tokens: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"goodput: expected 'key:value' strings, got {item!r}"
                )
            tokens.extend(item.split())
        return tokens
    raise ValueError(
        "goodput: expected space-separated 'key:value' pairs such as"
        f" 'ttft:500 tpot:50', got {value!r}"
    )


@dataclass(frozen=True)
class GoodputTargets:
    """Latency targets a request must meet to count toward goodput.

    Each target is in milliseconds, and ``None`` means no target on that
    metric. At least one is set.
    """

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2el_ms: float | None = None

    @classmethod
    def parse(cls, tokens: Sequence[str]) -> GoodputTargets:
        """Parses ``key:value`` tokens such as ``["ttft:500", "tpot:50"]``.

        Args:
            tokens: The targets, one ``key:value`` string each, with the value
                in milliseconds.

        Returns:
            The parsed targets.

        Raises:
            ValueError: If no target is given, a key is unknown or repeated,
                or a value is not a finite positive number.
        """
        if not tokens:
            raise ValueError(
                "goodput: give at least one target, such as 'ttft:500'"
            )
        targets: dict[str, float] = {}
        for token in tokens:
            key, sep, raw = token.partition(":")
            if not sep:
                raise ValueError(
                    f"goodput: {token!r} is not a 'key:value' pair;"
                    f" valid keys are {', '.join(GOODPUT_KEYS)}"
                )
            if key not in GOODPUT_KEYS:
                raise ValueError(
                    f"goodput: unknown key {key!r} in {token!r};"
                    f" valid keys are {', '.join(GOODPUT_KEYS)}"
                )
            if key in targets:
                raise ValueError(f"goodput: {key!r} is given more than once")
            try:
                ms = float(raw)
            except ValueError:
                raise ValueError(
                    f"goodput: {token!r} needs a number of milliseconds"
                ) from None
            if not math.isfinite(ms) or ms <= 0:
                raise ValueError(
                    f"goodput: {token!r} needs a positive, finite number of"
                    " milliseconds"
                )
            targets[key] = ms
        return cls(
            ttft_ms=targets.get("ttft"),
            tpot_ms=targets.get("tpot"),
            e2el_ms=targets.get("e2el"),
        )

    def is_met(
        self,
        *,
        ttft_s: float | None,
        tpot_s: float | None,
        latency_s: float,
    ) -> bool:
        """Returns whether one request meets every target that is set.

        A request without a value for a targeted metric, such as the TPOT of
        a one-token reply or the TTFT of an endpoint that reports none, does
        not meet that target.

        Args:
            ttft_s: The request's time to first token, in seconds, or
                ``None`` when the endpoint reported none.
            tpot_s: The request's time per output token, in seconds, or
                ``None`` when it is undefined.
            latency_s: The request's end-to-end latency, in seconds.
        """
        for target_ms, value_s in (
            (self.ttft_ms, ttft_s),
            (self.tpot_ms, tpot_s),
            (self.e2el_ms, latency_s),
        ):
            if target_ms is None:
                continue
            if value_s is None or value_s * 1000.0 > target_ms:
                return False
        return True
