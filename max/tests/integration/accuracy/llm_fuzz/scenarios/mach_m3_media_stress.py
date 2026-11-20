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
"""Accepted media workloads for MiniMax M3 on Mach's default 16-item limit."""

from __future__ import annotations

import base64
import struct
import uuid
from pathlib import Path

from client import FuzzClient, RunConfig

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._image_fixtures import image_part, image_payload
from scenarios._image_stress_common import (
    HEAVY_TIMEOUT_SEC,
    BatchTally,
    LivenessProbe,
    failure_verdict,
    post_stress_liveness,
    probe_result,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "mach_m3"
VIDEO_FIXTURES = (
    "tiny_16f_8fps_64x64_nobframes.mp4",
    "tiny_32f_g4_8fps_64x64_nobframes.mp4",
    "tiny_32f_g4_bf2_8fps_64x64.mp4",
)


def mp4_part(filename: str, nonce: str) -> dict[str, object]:
    # A trailing ISO-BMFF free box preserves the bitstream and sample offsets
    # while bypassing Mach's byte-keyed media cache on every cold request.
    tag = nonce.encode("ascii")
    data = (
        (DATA / filename).read_bytes()
        + struct.pack(">I4s", len(tag) + 8, b"free")
        + tag
    )
    return {
        "type": "video_url",
        "video_url": {
            "url": "data:video/mp4;base64,"
            + base64.b64encode(data).decode("ascii"),
            "fps": 4,
            "max_long_side_pixel": 224,
        },
    }


class MachM3MediaStress(BaseScenario):
    model_filter = "minimax-m3-mach"
    tags = ["vision", "concurrency", "model:minimax-m3-mach"]

    def parts(self, index: int, nonce: str) -> list[dict[str, object]]:
        raise NotImplementedError

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        nonce = uuid.uuid4().hex
        payloads = [
            image_payload(config.model, self.parts(i, f"{nonce}-{i}"))
            for i in range(8)
        ]
        results = []
        # Replay the same bytes to exercise cache hits as well as cold decode.
        for phase in ("cold", "warm"):
            async with LivenessProbe(config) as probe:
                responses = await client.concurrent_requests(
                    payloads, max_concurrent=4, timeout=HEAVY_TIMEOUT_SEC
                )
            tally = BatchTally().add(responses)
            result = failure_verdict(
                self.name,
                phase,
                tally,
                config,
                context="8 requests, concurrency 4",
                probe=probe,
            )
            if result is None:
                # A healthy canary does not prove that any media was processed.
                # Refusal-only workloads must never receive a stress PASS.
                result = self.make_result(
                    self.name,
                    phase,
                    Verdict.PASS
                    if tally.ok == tally.total
                    else Verdict.INTERESTING,
                    detail=tally.summary(),
                )
            results.extend(
                [
                    result,
                    probe_result(self.name, f"{phase}_liveness", probe, config),
                ]
            )
        results.append(await post_stress_liveness(client, self.name))
        return results


@register_scenario
class MachM3ImageStress(MachM3MediaStress):
    name = "mach_m3_image_stress"
    description = (
        "Mach M3: 1-16 images, large images, chunk boundaries, cold/warm bursts"
    )

    def parts(self, index: int, nonce: str) -> list[dict[str, object]]:
        count, side = ((16, 224), (1, 3584), (3, 1024), (4, 1024))[index % 4]
        return [
            image_part(side, side, f"{nonce}-{i}", "high") for i in range(count)
        ]


@register_scenario
class MachM3VideoStress(MachM3MediaStress):
    name = "mach_m3_video_stress"
    description = "Mach M3: H.264 MP4, B-frames, 1-16 clips, cold/warm bursts"

    def parts(self, index: int, nonce: str) -> list[dict[str, object]]:
        count = (1, 2, 4, 16)[index % 4]
        return [
            mp4_part(
                VIDEO_FIXTURES[(index + i) % len(VIDEO_FIXTURES)],
                f"{nonce}-{i}",
            )
            for i in range(count)
        ]
