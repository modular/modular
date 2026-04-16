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
"""Minimal repro: test memory manager behavior with varying resolutions."""

from __future__ import annotations

import argparse
import asyncio
import gc
import time
from typing import Any, cast

import torch
from max.driver import DeviceSpec
from max.interfaces import PipelineTask, PixelGenerationInputs, RequestID
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
    VideoProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import OpenResponsesRequestBody
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.architectures.wan.context import WanContext
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.lib.interfaces import DiffusionPipeline
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.pipeline_variants.pixel_generation import (
    PixelGenerationPipeline,
)

CONFIGS = {
    "480p_49f": (480, 832, 49, 30),
    "480p_81f": (480, 832, 81, 50),
    "720p_49f": (720, 1280, 49, 30),
}


def _load_pipeline(
    model_id: str,
) -> tuple[PixelGenerationPipeline[WanContext], WanTokenizer]:
    manifest = ModelManifest.from_model_path(
        model_id,
        device_specs=[DeviceSpec.accelerator()],
    )
    config = PipelineConfig(
        models=manifest,
        runtime=PipelineRuntimeConfig(),
    )
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        config.models.main_architecture_name,
        task=PipelineTask.PIXEL_GENERATION,
    )
    assert arch is not None
    tokenizer = WanTokenizer(
        model_path=model_id,
        pipeline_config=config,
        subfolder="tokenizer",
        max_length=512,
    )
    pipeline_model = cast(type[DiffusionPipeline], arch.pipeline_model)
    pipeline = PixelGenerationPipeline[WanContext](
        pipeline_config=config,
        pipeline_model=pipeline_model,
    )
    return pipeline, tokenizer


async def _build_inputs(
    model_id: str,
    tokenizer: WanTokenizer,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    seed: int,
) -> tuple[Any, Any]:
    body = OpenResponsesRequestBody(
        model=model_id,
        input=prompt,
        seed=seed,
        provider_options=ProviderOptions(
            image=ImageProviderOptions(
                height=height,
                width=width,
                steps=steps,
                guidance_scale=5.0,
            ),
            video=VideoProviderOptions(
                height=height,
                width=width,
                num_frames=num_frames,
                steps=steps,
            ),
        ),
    )
    request = OpenResponsesRequest(request_id=RequestID(), body=body)
    context = await tokenizer.new_context(request)
    inputs = PixelGenerationInputs[WanContext](
        batch={context.request_id: context}
    )
    return inputs, context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--order",
        default="escalating",
        choices=["escalating", "descending"],
        help=(
            "escalating: 480p/49f -> 480p/81f -> 720p/49f (OOM repro);"
            " descending: 720p/49f -> 480p/81f -> 480p/49f"
        ),
    )
    args = parser.parse_args()

    orders = {
        "escalating": ["480p_49f", "480p_81f", "720p_49f"],
        "descending": ["720p_49f", "480p_81f", "480p_49f"],
    }
    order = orders[args.order]

    print(f"Loading pipeline ({args.model})...", flush=True)
    pipeline, tokenizer = _load_pipeline(args.model)
    print("Pipeline loaded.", flush=True)

    for label in order:
        h, w, nf, steps = CONFIGS[label]
        print(f"\n--- {label}: {w}x{h}, {nf}f, {steps} steps ---", flush=True)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"  GPU before: {free / 1e9:.1f} GB free"
                f" / {total / 1e9:.1f} GB total",
                flush=True,
            )
        try:
            inputs, _ctx = asyncio.run(
                _build_inputs(
                    args.model,
                    tokenizer,
                    "A cat walking",
                    h,
                    w,
                    nf,
                    steps,
                    args.seed,
                )
            )
            t0 = time.perf_counter()
            pipeline.execute(inputs)
            print(f"  OK in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"  GPU after:  {free / 1e9:.1f} GB free"
                f" / {total / 1e9:.1f} GB total",
                flush=True,
            )


if __name__ == "__main__":
    main()
