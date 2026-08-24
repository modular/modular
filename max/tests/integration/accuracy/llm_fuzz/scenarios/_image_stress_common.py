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
scenario needs the same three things: production-shaped batch sizes, the
window the deployment actually serves, and a canary that can tell a wedged
engine from a slow one.
"""

from __future__ import annotations

import asyncio
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
