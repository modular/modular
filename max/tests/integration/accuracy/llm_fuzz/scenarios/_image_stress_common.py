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
"""Shared machinery for the image stress scenarios.

Extracted from ``image_stress`` so the KV-saturation and hang-hunt scenarios
reuse one liveness probe rather than each growing their own. Every image
scenario needs the same four things: production-shaped batch sizes, the
window the deployment actually serves, a canary that can tell a wedged engine
from a slow one, and one grader that decides what a batch of failures means.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import time
from typing import TYPE_CHECKING, Any

from client import FuzzClient

from scenarios import ScenarioResult, Verdict
from scenarios._image_fixtures import image_part

if TYPE_CHECKING:
    from client import RunConfig

# Batch sizes taken from the per-pod "fatal encode size" column in
# MXSERV-395 -- the image counts pods were processing when they wedged.
# Deliberately unremarkable; the trigger was concurrency, not size.
PRODUCTION_BATCH_SIZES = [1, 2, 4, 5, 5, 5, 6, 7, 8, 8, 8, 11, 14, 25, 26]

# Source dimensions for a "typical" production image. Under the default
# detail tier this lands around 1.3k tokens, so a handful of them straddle
# the vision encoder's chunk boundary.
TYPICAL_SIDE = 1024

# In-flight requests per simulated node. Production orchestrators cap at 80
# per pod across all traffic; image requests are a fraction of that.
CONCURRENCY_PER_NODE = 8

# Large-payload requests preprocess hundreds of millions of pixels before
# they can answer, well past the 30s default.
HEAVY_TIMEOUT_SEC = 300.0

# Non-image tokens a request carries: chat template, the text prompt, and the
# reserved generation budget. Measured at 305 + max_tokens against
# MiniMax-M3-MXFP4; rounded up so a template change does not push the
# "largest legal payload" case over the window it was sized against.
PAYLOAD_OVERHEAD_TOKENS = 512

# Share of failed requests that stops being incidental. Well above the
# background rate a long run collects from the network path in front of the
# deployment, and far below the near-total loss a wedged pod produces, so
# neither reading is a judgement call. Failures at or under this rate are
# reported, never hidden -- they just do not get to claim the hang.
MASS_FAILURE_RATE = 0.10

# Dropped connections only count toward that rate once there are two of them.
# One is the background rate of the network path in front of the deployment
# (CENG-1050), and in a two-request rung it is 50% -- enough to fail a healthy
# ramp on arithmetic alone. It is still reported, just not charged.
MIN_CHARGED_DROPS = 2


def unique_parts(
    count: int,
    tag: str,
    side: int = TYPICAL_SIDE,
    detail: str | None = None,
) -> list[dict[str, Any]]:
    """Builds ``count`` image parts that are guaranteed cache misses."""
    return [
        image_part(side, side, nonce=f"{tag}-{i}", detail=detail)
        for i in range(count)
    ]


async def served_context_window(
    client: FuzzClient, config: RunConfig
) -> tuple[int, str]:
    """Context window the server enforces, and where the number came from.

    ``model_config`` holds the *architectural* window read from the HF config,
    which is what the model could do rather than what this deployment does:
    MiniMax-M3 reports 1,048,576 there while a recipe routinely serves a
    fraction of it, and the served number is the one that rejects a request.
    ``/v1/models`` reports it as ``max_model_len``, so prefer that and keep
    the architectural value as the fallback for endpoints that omit it.
    """
    architectural = config.model_config.max_position_embeddings
    resp = await client.get_path("/v1/models")
    if resp.status == 200:
        try:
            for entry in json.loads(resp.body).get("data", []):
                served = entry.get("max_model_len")
                if served and entry.get("id") == config.model:
                    return int(served), "served"
        except (ValueError, TypeError, AttributeError):
            pass
    return architectural, "architectural"


class LivenessProbe:
    """Background canary tracking the longest stall in server progress.

    Runs on its own :class:`FuzzClient`, and therefore its own thread pool:
    the shared client's pool is sized to ``--max-concurrency`` and every
    in-flight stress request occupies a slot in it, so a probe sharing that
    pool would queue behind the very traffic it is meant to observe and
    report a stall that is really just saturation.
    """

    def __init__(
        self, config: RunConfig, interval: float = 1.0, timeout: float = 30.0
    ) -> None:
        self._config = dataclasses.replace(
            config, max_concurrency=4, timeout=timeout
        )
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._client: FuzzClient | None = None
        self._last_ok = 0.0
        self.max_gap_sec = 0.0
        self.probes = 0
        self.failures = 0

    async def __aenter__(self) -> LivenessProbe:
        self._client = FuzzClient(self._config)
        self._last_ok = time.monotonic()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close the gap against the wall clock rather than against the last
        # probe that came back. A wedged server answers nothing, so the final
        # probe never returns to record how long the stall had grown -- and a
        # phase that ends mid-stall would otherwise report only the gap as of
        # the last completed probe, understating it by a whole probe timeout.
        self._observe(time.monotonic())
        if self._client is not None:
            await self._client.__aexit__()

    def _observe(self, now: float) -> None:
        self.max_gap_sec = max(self.max_gap_sec, now - self._last_ok)

    async def _loop(self) -> None:
        """Samples at a fixed rate, not a fixed gap between samples.

        Sleeping ``interval`` *after* each probe returns makes the sampling
        period ``interval + latency``, so the probe samples least often exactly
        when the server is slowest -- and a probe that runs into its own
        timeout stretches the period by 30s, thinning coverage right where a
        hang would show. Deducting the elapsed time holds the rate steady and
        lets a stalled probe re-fire immediately.
        """
        assert self._client is not None
        next_tick = time.monotonic()
        while True:
            resp = await self._client.health_check()
            self.probes += 1
            now = time.monotonic()
            if resp.status == 200:
                self._last_ok = now
            else:
                self.failures += 1
            self._observe(now)
            next_tick += self._interval
            # A probe slower than the interval leaves the schedule in the
            # past; resuming from now re-fires once immediately rather than
            # firing a catch-up burst for every tick that was missed.
            next_tick = max(next_tick, now)
            await asyncio.sleep(next_tick - now)

    @property
    def failure_ratio(self) -> float:
        return self.failures / self.probes if self.probes else 0.0

    def summary(self) -> str:
        return (
            f"liveness: {self.probes} probes, {self.failures} failed, "
            f"max stall {self.max_gap_sec:.0f}s"
        )


def stall_threshold(config: RunConfig) -> float:
    """Stall gap that counts as a hang rather than queueing.

    The production hang is permanent -- the pod never recovers without a
    restart -- so this sits well above any plausible queueing delay, to keep
    a slow batch from reading as a deadlock.
    """
    return max(90.0, config.timeout * 2)


def probe_result(
    scenario: str, test: str, probe: LivenessProbe, config: RunConfig
) -> ScenarioResult:
    """Grades the liveness probe. Shared by every image scenario."""
    threshold = stall_threshold(config)
    if probe.probes == 0:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.ERROR,
            detail="liveness probe never ran",
        )
    if probe.max_gap_sec > threshold:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.FAIL,
            detail=(
                f"no forward progress for {probe.max_gap_sec:.0f}s "
                f"(threshold {threshold:.0f}s) -- {probe.summary()}"
            ),
        )
    # A majority of probes failing is conclusive on its own. The gap can stay
    # under the threshold simply because the phase was short -- a probe that
    # never returns is only observed once its timeout expires -- but a trivial
    # request failing more often than not is never healthy at any duration.
    if probe.failure_ratio > 0.5:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.FAIL,
            detail=(
                f"{probe.failures}/{probe.probes} liveness probes failed "
                f"-- {probe.summary()}"
            ),
        )
    if probe.failures or probe.max_gap_sec > threshold / 2:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.INTERESTING,
            detail=f"progress degraded but recovered -- {probe.summary()}",
        )
    return ScenarioResult(
        scenario_name=scenario,
        test_name=test,
        verdict=Verdict.PASS,
        detail=probe.summary(),
    )


@dataclasses.dataclass
class BatchTally:
    """Running counts over one or more batches of responses.

    Accumulates rather than grading each batch alone, because the soak sends
    batches for its whole duration and the verdict is about the run.
    """

    total: int = 0
    ok: int = 0
    client_errors: int = 0
    timeouts: int = 0
    server_errors: int = 0
    dropped: int = 0
    # Statuses no bucket claims -- a 3xx from a proxy, say. Not a failure, but
    # not a pass either, so it is counted rather than dropped on the floor.
    other: int = 0
    # What the failures actually were. A long run collects the occasional
    # connection reset, and a bare count cannot be told apart from the hang
    # without re-running the whole phase.
    causes: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )

    def add(self, responses: list[Any]) -> BatchTally:
        # A timeout and a dropped connection both surface as status 0, so the
        # buckets are kept disjoint -- counting `status == 0` as a server error
        # as well would report every timeout twice and make the tally read as
        # though two separate things went wrong.
        for resp in responses:
            self.total += 1
            if resp.status == 200:
                self.ok += 1
                continue
            if resp.error == "TIMEOUT":
                self.timeouts += 1
            elif resp.status >= 500:
                self.server_errors += 1
            elif resp.status == 0:
                self.dropped += 1
            elif 400 <= resp.status < 500:
                self.client_errors += 1
            else:
                self.other += 1
            self.causes[(resp.error or f"status {resp.status}")[:80]] += 1
        return self

    @property
    def failures(self) -> int:
        """Responses consistent with a hang or a crash. A 4xx is neither."""
        return self.timeouts + self.server_errors + self.dropped

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0

    @property
    def mass_failure(self) -> bool:
        """Whether too much of the batch failed to read as incidental."""
        charged = self.timeouts + self.server_errors
        if self.dropped >= MIN_CHARGED_DROPS:
            charged += self.dropped
        return bool(self.total) and charged / self.total > MASS_FAILURE_RATE

    def summary(self) -> str:
        return (
            f"{self.ok}/{self.total} ok, {self.client_errors} 4xx, "
            f"{self.server_errors} 5xx, {self.timeouts} timeouts, "
            f"{self.dropped} dropped"
            + (f", {self.other} other" if self.other else "")
        )

    def cause_breakdown(self) -> str:
        return ", ".join(
            f"{count}x {cause}" for cause, count in self.causes.most_common(4)
        )


def failure_verdict(
    scenario: str,
    test: str,
    tally: BatchTally,
    config: RunConfig,
    *,
    context: str,
    probe: LivenessProbe | None = None,
) -> ScenarioResult | None:
    """Grades the failure axis of a batch, or ``None`` if nothing failed.

    These scenarios hunt a deadlock: the pod stops completing anything until
    it is restarted. The probe measures that directly, on its own connection
    pool, so it decides the verdict. The request tally cannot -- a long run at
    high concurrency collects the occasional connection reset from the network
    path in front of the deployment, and CENG-1050 saw two of those in 4448
    requests report a hang against a server whose worst stall was 1s.

    So a stall fails on the probe's evidence, a mass failure fails under its
    own cause, and anything smaller is reported with that cause rather than
    dressed up as the deadlock. Whether a *clean* batch passes is the caller's
    call -- only it knows what a 4xx means for its axis -- hence ``None``.

    The probe's other two states (never ran, majority failing) belong to
    ``probe_result``; every caller passing a probe emits one.
    """
    breakdown = tally.cause_breakdown()
    threshold = stall_threshold(config)

    if probe is not None and probe.max_gap_sec > threshold:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.FAIL,
            detail=(
                f"{tally.summary()} -- the server stopped completing "
                f"requests for {probe.max_gap_sec:.0f}s (threshold "
                f"{threshold:.0f}s) ({context})"
                + (f" -- failures: {breakdown}" if breakdown else "")
            ),
        )

    # Reported by cause rather than as the hang, so a run that lost this share
    # of requests without ever wedging still fails -- under what it actually
    # was -- instead of being filed against a bug the probe just cleared.
    stayed_live = (
        f", no stall past {threshold:.0f}s ({probe.summary()})"
        if probe is not None
        else ""
    )
    if tally.mass_failure:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.FAIL,
            detail=(
                f"{tally.failure_rate:.0%} of requests failed{stayed_live}: "
                f"{breakdown} -- {tally.summary()} ({context})"
            ),
        )
    if tally.failures:
        return ScenarioResult(
            scenario_name=scenario,
            test_name=test,
            verdict=Verdict.INTERESTING,
            detail=(
                f"{tally.failures} of {tally.total} requests failed "
                f"({tally.failure_rate:.2%}){stayed_live}, so these read as "
                f"incidental rather than the hang: {breakdown} -- "
                f"{tally.summary()} ({context})"
            ),
        )
    return None
