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
signal MXSERV-395 would have tripped.
"""

from __future__ import annotations

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
    LivenessProbe,
    probe_result,
    served_context_window,
    stall_threshold,
    unique_parts,
)

if TYPE_CHECKING:
    from client import RunConfig


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
        results.append(await self._chunk_boundary_straddle(client, model))
        results.append(await self._skewed_aspect_ratios(client, model))
        results.append(await self._max_total_image_tokens(client, config))
        results.append(await self._over_budget_request(client, model))

        # Concurrency axes: the MXSERV-395 shape.
        results.extend(await self._concurrent_uncached_batches(client, config))
        results.append(await self._mixed_cache_hit_miss(client, config))

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
        resp = await client.post_json(payload, timeout=HEAVY_TIMEOUT_SEC)
        return self._classify(
            "single_max_image",
            resp,
            expect_reject=False,
            context=f"1 image, {MAX_IMAGE_TOKENS} tokens (3584x3584, detail=high)",
        )

    async def _max_image_count(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """200 images -- the vendor per-request ceiling, exactly at the limit."""
        parts = unique_parts(IMAGE_MAX_COUNT, "count-200", side=224)
        resp = await client.post_json(
            image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "max_image_count",
            resp,
            expect_reject=False,
            context=f"{IMAGE_MAX_COUNT} images at the vendor limit",
        )

    async def _over_max_image_count(
        self, client: FuzzClient, model: str
    ) -> ScenarioResult:
        """201 images -- one past the limit. Must be a clean 4xx."""
        parts = unique_parts(IMAGE_MAX_COUNT + 1, "count-201", side=224)
        resp = await client.post_json(
            image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "over_max_image_count",
            resp,
            expect_reject=True,
            context=f"{IMAGE_MAX_COUNT + 1} images, one past the vendor limit",
        )

    async def _chunk_boundary_straddle(
        self, client: FuzzClient, model: str
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
            image_payload(model, unique_parts(n, f"chunk-{n}"))
            for n in (at_budget - 1, at_budget, at_budget + 1, at_budget * 2)
        ]
        responses = await client.concurrent_requests(payloads, max_concurrent=2)
        return self._classify_many(
            "chunk_boundary_straddle",
            responses,
            context=(
                f"~{per_image} tokens/image, straddling the "
                f"{VISION_CHUNK_TOKENS}-token chunk budget "
                f"({at_budget - 1}/{at_budget}/{at_budget + 1}/"
                f"{at_budget * 2} images)"
            ),
        )

    async def _skewed_aspect_ratios(
        self, client: FuzzClient, model: str
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
            image_payload(model, [image_part(w, h, tag, detail="high")])
            for w, h, tag in cases
        ]
        responses = await client.concurrent_requests(
            payloads, max_concurrent=len(payloads)
        )
        return self._classify_many(
            "skewed_aspect_ratios",
            responses,
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
        resp = await client.post_json(
            image_payload(config.model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "max_total_image_tokens",
            resp,
            expect_reject=False,
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
        resp = await client.post_json(
            image_payload(model, parts), timeout=HEAVY_TIMEOUT_SEC
        )
        return self._classify(
            "over_budget_request",
            resp,
            expect_reject=True,
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
    ) -> ScenarioResult:
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

        result = self._classify_many(
            "mixed_cache_hit_miss",
            responses,
            context=(
                f"{len(payloads)} requests at concurrency {concurrency}, "
                f"alternating cached and uncached images; {probe.summary()}"
            ),
        )
        # A stall during the mixed phase is the same signal as during the
        # uncached phase; fold it into this verdict rather than emitting a
        # second row.
        if (
            result.verdict == Verdict.PASS
            and probe.max_gap_sec > stall_threshold(config)
        ):
            return self.make_result(
                self.name,
                "mixed_cache_hit_miss",
                Verdict.FAIL,
                detail=f"requests completed but {probe.summary()}",
            )
        return result

    async def _post_stress_liveness(self, client: FuzzClient) -> ScenarioResult:
        """Final forward-progress check once all stress traffic has drained."""
        resp = await client.health_check()
        if resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                "post_stress_liveness",
                Verdict.FAIL,
                detail="health probe timed out -- engine wedged after image stress",
            )
        if resp.status != 200:
            return self.make_result(
                self.name,
                "post_stress_liveness",
                Verdict.FAIL,
                status_code=resp.status,
                detail=f"health probe failed: status={resp.status} error={resp.error!r}",
            )
        return self.make_result(
            self.name, "post_stress_liveness", Verdict.PASS, status_code=200
        )

    # ------------------------------------------------------------------
    # Verdict helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        test: str,
        resp: Any,
        *,
        expect_reject: bool,
        context: str,
    ) -> ScenarioResult:
        """Grades one response against the vendor limits.

        A timeout or 5xx is always a failure: those are the hang and the OOM.
        Everything else depends on whether the request was legal.
        """
        if resp.error == "TIMEOUT":
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                detail=f"timeout -- no response ({context})",
            )
        if resp.status >= 500 or resp.status == 0:
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                status_code=resp.status,
                detail=f"server error {resp.status} ({context}): {resp.body[:200]!r}",
            )
        if expect_reject:
            if 400 <= resp.status < 500:
                return self.make_result(
                    self.name,
                    test,
                    Verdict.PASS,
                    status_code=resp.status,
                    detail=f"cleanly rejected ({context})",
                )
            return self.make_result(
                self.name,
                test,
                Verdict.INTERESTING,
                status_code=resp.status,
                detail=f"accepted a request that exceeds the limits ({context})",
            )
        if resp.status == 200:
            return self.make_result(
                self.name,
                test,
                Verdict.PASS,
                status_code=resp.status,
                detail=context,
            )
        return self.make_result(
            self.name,
            test,
            Verdict.INTERESTING,
            status_code=resp.status,
            detail=f"rejected a request within the limits ({context}): {resp.body[:200]!r}",
        )

    def _classify_many(
        self, test: str, responses: list[Any], *, context: str
    ) -> ScenarioResult:
        """Grades a batch. Any timeout or 5xx fails the whole batch."""
        total = len(responses)
        # A timeout and a dropped connection both surface as status 0, so these
        # buckets are kept disjoint -- counting `status == 0` as a server error
        # as well would report every timeout twice and make the tally read as
        # though two separate things went wrong.
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        server_errors = sum(1 for r in responses if r.status >= 500)
        dropped = sum(
            1 for r in responses if r.status == 0 and r.error != "TIMEOUT"
        )
        client_errors = sum(1 for r in responses if 400 <= r.status < 500)
        ok = sum(1 for r in responses if r.status == 200)
        tally = (
            f"{ok}/{total} ok, {client_errors} 4xx, {server_errors} 5xx, "
            f"{timeouts} timeouts, {dropped} dropped"
        )

        if timeouts or server_errors or dropped:
            return self.make_result(
                self.name,
                test,
                Verdict.FAIL,
                detail=f"{tally} -- hang or crash ({context})",
            )
        if client_errors:
            return self.make_result(
                self.name,
                test,
                Verdict.INTERESTING,
                detail=f"{tally} -- valid image requests rejected ({context})",
            )
        return self.make_result(
            self.name, test, Verdict.PASS, detail=f"{tally} ({context})"
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
        timeouts = 0
        server_errors = 0
        dropped = 0

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
                timeouts += sum(1 for r in responses if r.error == "TIMEOUT")
                server_errors += sum(1 for r in responses if r.status >= 500)
                dropped += sum(
                    1
                    for r in responses
                    if r.status == 0 and r.error != "TIMEOUT"
                )

        duration = config.endurance_duration_sec
        elapsed = (
            f"{duration / 60:.0f}min" if duration >= 60 else f"{duration:.0f}s"
        )
        tally = (
            f"{sent} requests over {elapsed} at concurrency "
            f"{concurrency}, {server_errors} 5xx, {timeouts} timeouts, "
            f"{dropped} dropped"
        )
        verdict = (
            Verdict.FAIL
            if (timeouts or server_errors or dropped)
            else Verdict.PASS
        )
        return [
            self.make_result(
                self.name,
                "image_soak",
                verdict,
                detail=f"{tally} -- {probe.summary()}",
            ),
            probe_result(self.name, "image_soak_liveness", probe, config),
        ]
