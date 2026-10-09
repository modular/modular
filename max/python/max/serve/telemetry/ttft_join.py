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
"""Same-request TTFT leftover join, gated by ``MAX_SERVE_TTFT_JOIN``.

Default is off (unset or ``0``): no stamps, no stderr lines, nightly
unchanged. Set ``MAX_SERVE_TTFT_JOIN=1`` on the decode API to print
one JSON ``ttft_join`` line per first yield and a periodic ``kv_hold``
line. Mammoth opt-in is the
``1p1d_minimax_m3_mxfp8_b200_staging_ttft_join`` overlay, dispatch-only.

Prometheus hop histograms have no request_id, so leftover (server
TTFT minus named hops) cannot be proven on one request. When the
env is set, the decode model worker stamps hop ms onto the first
:class:`~max.pipelines.context.TextGenerationOutput` and the API
process prints at the same instant as ``maxserve.time_to_first_token``.

Stderr print is intentional: Mammoth structured logging already
captures API INFO at startup, but the per-request join line did
not appear in the decode pod log. The write happens after server
TTFT is snapshotted, so leftover on that request is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping

_HOP_KEYS = (
    "admit_ms",
    "disp_ms",
    "span_ms",
    "reply_ms",
    "api_pre_tokenize_ms",
    "api_submit_ms",
    "mw_queue_wait_ms",
)


def is_enabled() -> bool:
    """Returns whether same-request TTFT join logging is on."""
    return os.getenv("MAX_SERVE_TTFT_JOIN", "0") == "1"


def leftover_ms(
    server_ttft_ms: float,
    ipt_ms: float,
    hops: Mapping[str, float] | None,
) -> float:
    """Returns server TTFT minus tokenize time and the named decode hops.

    Does not subtract ``yield_gap_ms``, ``onload_wait_ms``,
    ``transfer_wait_ms``, ``tg_queue_ms``, or ``tg_first_out_ms``. Those
    intervals are printed on the join line so leftover can be compared
    to each piece after the run.

    Args:
        server_ttft_ms: Server-side time to first token, milliseconds.
        ipt_ms: Tokenize / input-prep time on the API process.
        hops: Decode hop durations stamped on the first token, if any.

    Returns:
        The residual milliseconds after subtracting ``ipt_ms`` and each
        named hop that is present.
    """
    leftover = server_ttft_ms - ipt_ms
    if hops is None:
        return leftover
    for key in _HOP_KEYS:
        value = hops.get(key)
        if value is not None:
            leftover -= value
    return leftover


def announce_enabled() -> None:
    """Prints a one-time probe so a Mammoth scrape can see the gate is live."""
    if not is_enabled():
        return
    print("ttft_join enabled", file=sys.stderr, flush=True)


def log_ttft_join(
    *,
    request_id: object,
    server_ttft_ms: float,
    ipt_ms: float,
    hops: Mapping[str, float] | None,
    skipped_n: int = 0,
    yield_gap_ms: float = 0.0,
) -> None:
    """Prints one JSON ``ttft_join`` line to stderr when the env gate is on.

    No-op when ``MAX_SERVE_TTFT_JOIN`` is unset so nightly stays quiet.

    Args:
        request_id: Request identifier written as ``rid`` in the JSON line.
        server_ttft_ms: Server-side time to first token, milliseconds.
        ipt_ms: Tokenize / input-prep time on the API process.
        hops: Decode hop durations stamped on the first token, if any.
            A ``handoff`` key of ``1.0`` means the constrained discard
            path waited for a decode token. ``onload_wait_ms``,
            ``transfer_wait_ms``, ``tg_queue_ms`` and ``tg_first_out_ms``
            split the wait after the PrefillResponse and are not part of
            leftover.
        skipped_n: Worker outputs the API continued past before the first
            yielded chunk (delimiter-only reasoning strips).
        yield_gap_ms: First worker output received to first yield, on the
            same API clock as ``server_ttft_ms``.
    """
    if not is_enabled():
        return
    hop_map = dict(hops) if hops else {}
    payload = {
        "event": "ttft_join",
        "rid": str(request_id),
        "server_ttft_ms": round(server_ttft_ms, 3),
        "ipt_ms": round(ipt_ms, 3),
        "admit_ms": hop_map.get("admit_ms"),
        "disp_ms": hop_map.get("disp_ms"),
        "span_ms": hop_map.get("span_ms"),
        "reply_ms": hop_map.get("reply_ms"),
        "api_pre_tokenize_ms": hop_map.get("api_pre_tokenize_ms"),
        "api_submit_ms": hop_map.get("api_submit_ms"),
        "mw_queue_wait_ms": hop_map.get("mw_queue_wait_ms"),
        "skipped_n": skipped_n,
        "yield_gap_ms": round(yield_gap_ms, 3),
        "onload_wait_ms": hop_map.get("onload_wait_ms"),
        "transfer_wait_ms": hop_map.get("transfer_wait_ms"),
        "tg_queue_ms": hop_map.get("tg_queue_ms"),
        "tg_first_out_ms": hop_map.get("tg_first_out_ms"),
        "handoff": bool(hop_map.get("handoff")),
        "leftover_ms": round(leftover_ms(server_ttft_ms, ipt_ms, hop_map), 3),
    }
    print(
        json.dumps(payload, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def log_kv_hold(**fields: int) -> None:
    """Prints one JSON ``kv_hold`` line to stderr when the env gate is on.

    Args:
        **fields: Per-replica block and request counts from the decode
            scheduler.
    """
    if not is_enabled():
        return
    print(
        json.dumps({"event": "kv_hold", **fields}, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )
