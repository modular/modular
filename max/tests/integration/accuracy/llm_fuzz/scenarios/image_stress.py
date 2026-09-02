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
"""
Scenario: huge image request batches
Target: the vision preprocessing and encoder path -- deadlocks under
        concurrent cache-miss traffic (MXSERV-395), OOM on large image-token
        counts (CENG-880), and the vendor request limits at their boundaries.

Two failure modes, and they want opposite things from a test.

The OOM is a *size* bug: one big enough request kills the pod, so a single
oversized request finds it. The hang is not. Production pods wedged on
requests carrying 1, 2, 4, 5, 6 and 8 images -- ordinary sizes -- and the
common thread was concurrency and a cache miss, not scale. In 13 of 16
observed cases the last line the server logged was ``MiniMax-M3 vision
encoder: encoding N uncached image(s)``. A size-only stress test walks
straight past it.

So the size axes here (max count, max-size image, the largest image-token
payload the served window allows, skewed aspect ratios) target the OOM and
the validation boundaries, while ``concurrent_uncached_batches`` and
``mixed_cache_hit_miss`` target the hang: modest batches, high concurrency,
every image byte-unique so it misses the server's preprocess cache.

Detecting a hang needs one more thing. If the engine wedges, the requests
holding it open never return -- and a test that only awaits its own traffic
hangs with it, or reports a timeout indistinguishable from ordinary
slowness. The stress phases therefore run under a background liveness
probe on a separate connection pool, which tracks the longest interval in
which the server completed no trivial request at all. That gap is the
signal MXSERV-395 would have tripped, and it is what decides the verdict --
see ``failure_verdict`` for why the request tally cannot.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from client import FuzzClient

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._image_fixtures import (
    IMAGE_MAX_COUNT,
    MAX_IMAGE_SIDE,
    MAX_IMAGE_TOKENS,
    MERGED_PATCH,
    MIN_SHORT_SIDE_PIXEL,
    VISION_CHUNK_TOKENS,
    estimate_tokens,
    image_part,
    image_payload,
)
from scenarios._image_stress_common import (
    CONCURRENCY_PER_NODE,
    HEAVY_TIMEOUT_SEC,
    PAYLOAD_OVERHEAD_TOKENS,
    PRODUCTION_BATCH_SIZES,
    TYPICAL_SIDE,
    BatchTally,
    LivenessProbe,
    failure_verdict,
    probe_result,
    served_context_window,
    unique_parts,
)

if TYPE_CHECKING:
    from client import RunConfig

# Pause before re-sending a request whose connection dropped, and between
# post-stress health probes. Long enough that a retry is not simply the same
# instant of whatever reset the first one, short enough to add nothing
# noticeable to a run that never drops.
RETRY_GAP_SEC = 2.0

# Attempts the post-stress health check gets before it reports the engine
# wedged. A wedged engine fails all of them; a healthy one answers the first.
LIVENESS_ATTEMPTS = 3


@register_scenario
class ImageStress(BaseScenario):
    name = "image_stress"
    description = (
        "Huge image batches: max count/size/tokens, skewed aspect ratios, "
        "and concurrent cache-miss batches under a liveness probe"
    )
    tags = ["vision", "image", "hang", "crash", "oom", "memory", "concurrency"]
    # The limits encoded here are MiniMax-M3 vendor limits, and firing 61
    # max-size images at a text-only model is pure noise. Run it against
    # another vision model with an explicit `--scenarios image_stress`,
    # which bypasses profile filtering.
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # Size and boundary axes first: they are cheap to attribute, and a
        # pod already wedged by them makes the concurrency results
        # meaningless.
        results.append(await self._single_max_image(client, model))
        results.append(await self._max_image_count(client, model))
        results.append(await self._over_max_image_count(client, model))
        results.append(await self._chunk_boundary_straddle(client, config))
        results.append(await self._skewed_aspect_ratios(client, config))
        results.append(await self._max_total_image_tokens(client, config))
        results.append(await self._over_budget_request(client, model))

        # Concurrency axes: the MXSERV-395 shape.
        results.extend(await self._concurrent_uncached_batches(client, config))
        results.extend(await self._mixed_cache_hit_miss(client, config))

        results.append(await self._post_stress_liveness(client))
        return results

    # ------------------------------------------------------------------
    # Size and boundary axes
    # ------------------------------------------------------------------

    async def _single_max_image(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """One image at the pixel cap: 3584x3584 -> 16384 tokens.

        Needs ``detail="high"``; without it the default 2016 tier downscales
        the image to 5184 tokens and the axis is not actually tested.
        """
        payload = image_payload(
            model,
            [image_part(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE, "max-single", "high")],
        )
        resp, dropped = await self._post_once(
            client, payload, timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "single_max_image",
            resp,
            expect_reject=False,
            context=f"1 image, {MAX_IMAGE_TOKENS} tokens (3584x3584, detail=high)",
            dropped_first=dropped,
        )

    async def _max_image_count(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """200 images -- the vendor per-request ceiling, exactly at the limit."""
        parts = unique_parts(IMAGE_MAX_COUNT, "count-200", side=224)
        resp, dropped = await self._post_once(
            client, image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "max_image_count",
            resp,
            expect_reject=False,
            context=f"{IMAGE_MAX_COUNT} images at the vendor limit",
            dropped_first=dropped,
        )

    async def _over_max_image_count(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """201 images -- one past the limit. Must be a clean 4xx."""
        parts = unique_parts(IMAGE_MAX_COUNT + 1, "count-201", side=224)
        resp, dropped = await self._post_once(
            client, image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "over_max_image_count",
            resp,
            expect_reject=True,
            context=f"{IMAGE_MAX_COUNT + 1} images, one past the vendor limit",
            dropped_first=dropped,
        )

    async def _chunk_boundary_straddle(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """Requests landing just under, at, and just over the chunk budget.

        The vision encoder batches images until it would exceed
        ``max_batch_input_tokens`` (8192 by default), then splits. The split
        path is where the chunk loop and its staging buffers live, so the
        interesting inputs are the ones that straddle the edge -- which means
        at least one case has to land *under* the budget and not split.
        """
        per_image = estimate_tokens(TYPICAL_SIDE, TYPICAL_SIDE)
        at_budget = max(1, VISION_CHUNK_TOKENS // per_image)
        payloads = [
            image_payload(config.model, unique_parts(n, f"chunk-{n}"))
            for n in (at_budget - 1, at_budget, at_budget + 1, at_budget * 2)
        ]
        responses = await client.concurrent_requests(payloads, max_concurrent=2)
        return self._classify_many(
            "chunk_boundary_straddle",
            responses,
            config,
            context=(
                f"~{per_image} tokens/image, straddling the "
                f"{VISION_CHUNK_TOKENS}-token chunk budget "
                f"({at_budget - 1}/{at_budget}/{at_budget + 1}/"
                f"{at_budget * 2} images)"
            ),
        )

    async def _skewed_aspect_ratios(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """Extreme aspect ratios and short-side edges.

        200:1 is the headline case: the long-side downscale drops the short
        side to roughly 18 px, and because the resize rules are mutually
        exclusive the 112 px floor never fires, leaving a one-patch-tall
        image. Also covers dimensions that are not multiples of 28, where
        the rounding mode varies.
        """
        cases = [
            (MIN_SHORT_SIDE_PIXEL, MIN_SHORT_SIDE_PIXEL * 200, "skew-tall"),
            (MIN_SHORT_SIDE_PIXEL * 200, MIN_SHORT_SIDE_PIXEL, "skew-wide"),
            (MIN_SHORT_SIDE_PIXEL, MIN_SHORT_SIDE_PIXEL, "floor-exact"),
            (MERGED_PATCH, MERGED_PATCH, "one-patch"),
            (1021, 733, "non-multiple-of-28"),
        ]
        payloads = [
            image_payload(config.model, [image_part(w, h, tag, detail="high")])
            for w, h, tag in cases
        ]
        responses = await client.concurrent_requests(
            payloads, max_concurrent=len(payloads)
        )
        return self._classify_many(
            "skewed_aspect_ratios",
            responses,
            config,
            context=", ".join(f"{w}x{h}" for w, h, _ in cases),
        )

    async def _max_total_image_tokens(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """The largest legal image-token payload the served window allows.

        The CENG-880 shape: the request is valid, and the vision activations
        for it have to fit alongside the KV cache. The count is derived
        rather than fixed -- a deployment serving a fraction of the
        architectural window rejects a fixed count on context length, which
        reveals nothing about the vision path and buries the axis under a
        result that never changes.
        """
        window, source = await served_context_window(client, config)
        count = min(
            IMAGE_MAX_COUNT,
            (window - PAYLOAD_OVERHEAD_TOKENS) // MAX_IMAGE_TOKENS,
        )
        if count < 1:
            return self.make_result(
                self.name,
                "max_total_image_tokens",
                Verdict.INTERESTING,
                detail=(
                    f"{source} context window of {window} cannot hold one "
                    f"max-size image ({MAX_IMAGE_TOKENS} tokens) -- this axis "
                    "cannot run against this deployment"
                ),
            )
        parts = [
            image_part(
                MAX_IMAGE_SIDE, MAX_IMAGE_SIDE, f"max-tokens-{i}", "high"
            )
            for i in range(count)
        ]
        resp, dropped = await self._post_once(
            client,
            image_payload(config.model, parts),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return self._classify(
            "max_total_image_tokens",
            resp,
            expect_reject=False,
            dropped_first=dropped,
            context=(
                f"{count} max-size images, ~{count * MAX_IMAGE_TOKENS} tokens "
                f"-- the most the {source} window of {window} allows"
            ),
        )

    async def _over_budget_request(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """200 max-size images -- ~3.3M tokens, over any context window.

        The rejection is the easy part. What this actually tests is that the
        server survives it: the count check passes, so all 200 images are
        downloaded and preprocessed before anything notices the token
        budget. A 5xx or a timeout here is the OOM, not a validation
        failure.
        """
        parts = [
            image_part(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE, f"over-{i}", "high")
            for i in range(IMAGE_MAX_COUNT)
        ]
        resp, dropped = await self._post_once(
            client, image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "over_budget_request",
            resp,
            expect_reject=True,
            dropped_first=dropped,
            context=(
                f"{IMAGE_MAX_COUNT} max-size images, "
                f"~{IMAGE_MAX_COUNT * MAX_IMAGE_TOKENS} tokens -- "
                "preprocessed in full before the budget check rejects it"
            ),
        )

    # ------------------------------------------------------------------
    # Concurrency axes -- the MXSERV-395 shape
    # ------------------------------------------------------------------

    async def _concurrent_uncached_batches(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        """Production-shaped batches, concurrent, every image a cache miss.

        Scales with ``--image-stress-nodes`` so a multi-node deployment can
        be driven at the same per-node pressure a single pod sees.
        """
        nodes = max(1, config.image_stress_nodes)
        concurrency = nodes * CONCURRENCY_PER_NODE
        payloads = [
            image_payload(
                config.model,
                unique_parts(
                    PRODUCTION_BATCH_SIZES[i % len(PRODUCTION_BATCH_SIZES)],
                    f"conc-{i}",
                ),
            )
            for i in range(concurrency * 3)
        ]

        async with LivenessProbe(config) as probe:
            responses = await client.concurrent_requests(
                payloads, max_concurrent=concurrency
            )

        results = [
            self._classify_many(
                "concurrent_uncached_batches",
                responses,
                config,
                probe=probe,
                context=(
                    f"{len(payloads)} requests at concurrency {concurrency} "
                    f"({nodes} node(s) x {CONCURRENCY_PER_NODE}), batch sizes "
                    "drawn from the MXSERV-395 fatal-encode distribution, "
                    "all images byte-unique"
                ),
            ),
            probe_result(
                self.name, "concurrent_uncached_liveness", probe, config
            ),
        ]
        return results

    async def _mixed_cache_hit_miss(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        """Interleaves repeated and unique images at concurrency.

        A hit and a miss for the same size take different paths through the
        shared preprocess pool and its lock. Running only unique images
        never exercises that interleaving.
        """
        nodes = max(1, config.image_stress_nodes)
        concurrency = nodes * CONCURRENCY_PER_NODE
        shared = unique_parts(4, "shared-hot")
        payloads: list[dict[str, Any]] = []
        for i in range(concurrency * 3):
            if i % 2 == 0:
                parts = list(shared)  # repeat -- hits the preprocess cache
            else:
                parts = unique_parts(
                    PRODUCTION_BATCH_SIZES[i % len(PRODUCTION_BATCH_SIZES)],
                    f"mixed-{i}",
                )
            payloads.append(image_payload(config.model, parts))

        async with LivenessProbe(config) as probe:
            responses = await client.concurrent_requests(
                payloads, max_concurrent=concurrency
            )

        return [
            self._classify_many(
                "mixed_cache_hit_miss",
                responses,
                config,
                probe=probe,
                context=(
                    f"{len(payloads)} requests at concurrency {concurrency}, "
                    f"alternating cached and uncached images"
                ),
            ),
            probe_result(
                self.name, "mixed_cache_hit_miss_liveness", probe, config
            ),
        ]

    async def _post_stress_liveness(self, client: FuzzClient) -> ScenarioResult:
        """Final forward-progress check once all stress traffic has drained.

        Retried for the reason ``_post_once`` gives: one un-retried request
        cannot tell an engine that wedged from a connection the network path
        happened to reset, and this one decides whether the whole scenario
        ends on a hang. An engine that really wedged fails every attempt.
        """
        test = "post_stress_liveness"
        failures: list[str] = []
        last_status = 0
        answered = False
        for attempt in range(LIVENESS_ATTEMPTS):
            if attempt:
                await asyncio.sleep(RETRY_GAP_SEC)
            resp = await client.health_check()
            last_status = resp.status
            if resp.status == 200:
                if not failures:
                    return self.make_result(
                        self.name, test, Verdict.PASS, status_code=200
                    )
                return self.make_result(
                    self.name,
                    test,
                    Verdict.INTERESTING,
                    status_code=200,
                    detail=(
                        f"answered on attempt {attempt + 1} of "
                        f"{LIVENESS_ATTEMPTS}, so the engine was still there "
                        f"-- earlier attempts: {'; '.join(failures)}"
                    ),
                )
            answered = answered or (resp.status > 0 and resp.error != "TIMEOUT")
            failures.append(
                "timeout"
                if resp.error == "TIMEOUT"
                else f"status={resp.status} error={resp.error!r}"
            )
        # A server that returned three 5xx kept answering; only silence says it
        # wedged or died. Reporting both the same way loses the distinction
        # this scenario exists to draw.
        diagnosis = (
            "the server answered but never returned healthy"
            if answered
            else "the server stopped answering, whether it wedged or died"
        )
        return self.make_result(
            self.name,
            test,
            Verdict.FAIL,
            status_code=last_status,
            detail=(
                f"health probe failed {LIVENESS_ATTEMPTS} times running after "
                f"image stress -- {diagnosis}: {'; '.join(failures)}"
            ),
        )

    # ------------------------------------------------------------------
    # Verdict helpers
    # ------------------------------------------------------------------

    async def _post_once(
        self,
        client: FuzzClient,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> tuple[Any, str | None]:
        """Sends one request, re-sending it once if the connection drops.

        A dropped connection is not evidence of the deadlock these axes hunt.
        The network path in front of the deployment resets one occasionally --
        CENG-1050 saw two in 4448 requests against a server that never stopped
        answering. The batched axes tell that apart from a wedge by rate, which
        a single-request test has no way to do, so it re-sends instead: a reset
        does not survive a retry, and a pod that stopped answering fails every
        attempt.

        Returns the response to grade and the first attempt's cause if it
        dropped, so the verdict can name it rather than swallow it. A timeout
        is not retried -- the server took the request and never answered, which
        is the signal here, and re-sending a max-size payload would cost
        another ``HEAVY_TIMEOUT_SEC`` to learn nothing.
        """
        resp = await client.post_json(payload, timeout=timeout)
        if resp.status != 0 or resp.error == "TIMEOUT":
            return resp, None
        await asyncio.sleep(RETRY_GAP_SEC)
        retry = await client.post_json(payload, timeout=timeout)
        return retry, resp.error or "connection dropped"

    def _classify(
        self,
        test: str,
        resp: Any,
        *,
        expect_reject: bool,
        context: str,
        dropped_first: str | None = None,
    ) -> ScenarioResult:
        """Grades one response against the vendor limits.

        A timeout or a 5xx is always a failure: those are the hang and the OOM.
        A dropped connection is one only if it survived the retry in
        ``_post_once``. Everything else depends on whether the request was
        legal.
        """
        retried = f" -- first attempt dropped ({dropped_first})"
        if resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                detail=(
                    f"timeout -- no response ({context})"
                    + (retried if dropped_first else "")
                ),
            )
        if resp.status == 0:
            # Held apart from the 5xx branch below: a drop is a transport
            # failure, not a server error, and reporting one as "server error
            # 0" hides which of the two happened (CENG-1050).
            cause = resp.error or "no error reported"
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                status_code=resp.status,
                detail=(
                    f"connection dropped twice ({dropped_first}, then {cause})"
                    if dropped_first
                    else f"connection dropped ({cause})"
                )
                + f" ({context})",
            )
        if resp.status >= 500:
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                status_code=resp.status,
                detail=f"server error {resp.status} ({context}): {resp.body[:200]!r}"
                + (retried if dropped_first else ""),
            )
        if expect_reject:
            if 400 <= resp.status < 500:
                verdict, detail = Verdict.PASS, f"cleanly rejected ({context})"
            else:
                verdict, detail = (
                    Verdict.INTERESTING,
                    f"accepted a request that exceeds the limits ({context})",
                )
        elif resp.status == 200:
            verdict, detail = Verdict.PASS, context
        else:
            verdict, detail = (
                Verdict.INTERESTING,
                f"rejected a request within the limits ({context}): "
                f"{resp.body[:200]!r}",
            )
        # The drop is reported rather than hidden; it just does not get to
        # claim the hang, since the retry proved the server was still there.
        if dropped_first:
            verdict = (
                Verdict.INTERESTING if verdict is Verdict.PASS else verdict
            )
            detail += retried
        return self.make_result(
            self.name,
            test,
            verdict,
            status_code=resp.status,
            detail=detail,
        )

    def _classify_many(
        self,
        test: str,
        responses: list[Any],
        config: RunConfig,
        *,
        context: str,
        probe: LivenessProbe | None = None,
    ) -> ScenarioResult:
        """Grades a batch: the probe decides the hang, the tally the rest.

        Phases that run without a probe still get a verdict, from the rate
        alone -- see ``BatchTally.mass_failure`` for what a small batch of
        four or five requests does with a single failure.
        """
        tally = BatchTally().add(responses)
        failed = failure_verdict(
            self.name, test, tally, config, context=context, probe=probe
        )
        if failed is not None:
            return failed
        if tally.client_errors or tally.other:
            reason = (
                "valid image requests rejected"
                if tally.client_errors
                else "responses outside any expected status"
            )
            return self.make_result(
                self.name,
                test,
                Verdict.INTERESTING,
                detail=f"{tally.summary()} -- {reason} ({context})",
            )
        return self.make_result(
            self.name,
            test,
            Verdict.PASS,
            detail=f"{tally.summary()} ({context})",
        )


@register_scenario
class ImageStressSoak(BaseScenario):
    """Sustained version of the concurrency axis.

    Production pods took minutes of ordinary traffic to wedge -- the
    reproducer in MXSERV-395 deadlocked at 17 minutes of uptime. The bounded
    matrix above cannot cover that; this runs the same cache-miss batches for
    ``--endurance-duration`` and reports the longest stall observed.
    """

    name = "image_stress_soak"
    description = (
        "Sustained concurrent image batches with a liveness probe "
        "(minutes-scale hang reproduction)"
    )
    tags = ["vision", "image", "hang", "endurance", "concurrency"]
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        nodes = max(1, config.image_stress_nodes)
        concurrency = nodes * CONCURRENCY_PER_NODE
        deadline = time.monotonic() + config.endurance_duration_sec
        sent = 0
        tally = BatchTally()

        async with LivenessProbe(config) as probe:
            while time.monotonic() < deadline:
                payloads = [
                    image_payload(
                        config.model,
                        unique_parts(
                            PRODUCTION_BATCH_SIZES[
                                (sent + i) % len(PRODUCTION_BATCH_SIZES)
                            ],
                            f"soak-{sent + i}",
                        ),
                    )
                    for i in range(concurrency)
                ]
                responses = await client.concurrent_requests(
                    payloads, max_concurrent=concurrency
                )
                sent += len(responses)
                tally.add(responses)

        duration = config.endurance_duration_sec
        elapsed = (
            f"{duration / 60:.0f}min" if duration >= 60 else f"{duration:.0f}s"
        )
        context = f"{sent} requests over {elapsed} at concurrency {concurrency}"
        # The soak is long enough to collect incidental failures, so the probe
        # decides whether the pod wedged -- see ``failure_verdict``.
        failed = failure_verdict(
            self.name,
            "image_soak",
            tally,
            config,
            context=context,
            probe=probe,
        )
        return [
            failed
            or self.make_result(
                self.name,
                "image_soak",
                Verdict.PASS,
                detail=f"{tally.summary()} -- {context} -- {probe.summary()}",
            ),
            probe_result(self.name, "image_soak_liveness", probe, config),
        ]
