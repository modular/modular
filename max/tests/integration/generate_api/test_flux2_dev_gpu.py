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
"""Checks ModuleV3 FLUX.2-dev text to image through AutoPipeline against the
diffusers images stored in S3."""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from _generate_api import (
    FLUX2_DEV,
    FLUX2_GUIDANCE,
    FLUX2_HEIGHT,
    FLUX2_REQUESTS,
    FLUX2_REVISION,
    FLUX2_SEED,
    FLUX2_STEPS,
    FLUX2_WIDTH,
    ImageLimits,
    Report,
    check_images,
    check_module_v3,
    check_revision,
    device_memory_used_gib,
    load_flux2_references,
    save_output,
)
from max.pipelines import AutoPipeline
from max.pipelines.modeling.types import PipelineTask
from PIL import Image
from test_common.verify_utils import compute_lpips, compute_ssim

RUN = "flux2_dev"

_ARGUMENTS: dict[str, Any] = {
    "height": FLUX2_HEIGHT,
    "width": FLUX2_WIDTH,
    "num_inference_steps": FLUX2_STEPS,
    "guidance_scale": FLUX2_GUIDANCE,
    # One prompt per call, since the generate API offsets the seed by row.
    "seed": FLUX2_SEED,
}


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pixels(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def test_flux2_dev() -> None:
    report = Report(RUN)
    try:
        _run(report)
    finally:
        report.save()


def _run(report: Report) -> None:
    revision = FLUX2_REVISION
    check_revision(FLUX2_DEV, revision)
    references = load_flux2_references()
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    with AutoPipeline(
        FLUX2_DEV,
        prefer_module_v3=True,
        huggingface_model_revision=revision,
        huggingface_weight_revision=revision,
    ) as pipe:
        report.time("load_seconds", start)
        report.data["architecture"] = check_module_v3(
            pipe.model, PipelineTask.PIXEL_GENERATION
        )
        report.data["memory_used_gib"] = device_memory_used_gib()
        for request in FLUX2_REQUESTS:
            start = time.monotonic()
            [image] = pipe(request.prompt, **_ARGUMENTS)
            seconds = round(time.monotonic() - start, 2)
            save_output(f"{request.name}.png", _png(image))
            pixels = _pixels(image)
            reference = references[request.name]
            rows.append(
                {
                    "name": request.name,
                    "seconds": seconds,
                    "ssim": round(compute_ssim(pixels, reference), 5),
                    "lpips": round(
                        float(compute_lpips(pixels, reference).max()), 5
                    ),
                }
            )
            report.data["rows"] = rows
    worst: ImageLimits = {
        "ssim": min(row["ssim"] for row in rows),
        "lpips": max(row["lpips"] for row in rows),
    }
    report.data["worst"] = worst
    check_images(worst)
