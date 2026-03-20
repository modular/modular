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

"""Benchmark FLUX2 Transformer (denoising step) with random weights.

Builds a Flux2Transformer2DModel with random weights, compiles it once,
then runs identical inputs through it N times via pytest-benchmark.

Supports first-block caching (step cache) via the ``--step-cache`` flag.

Usage:
    bt //max/tests/integration/architectures/flux2:benchmark_transformer \
        --test_output=streamed --curses=no --noshow_progress

    # With first-block caching enabled:
    bt //max/tests/integration/architectures/flux2:benchmark_transformer \
        --test_output=streamed --curses=no --noshow_progress \
        --test_arg=--step-cache

    # Profile with nsys:
    br-nsys //max/tests/integration/architectures/flux2:benchmark_transformer
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.pipelines.architectures.flux2_modulev3.flux2 import (
    Flux2Transformer2DModel,
)
from max.pipelines.architectures.flux2_modulev3.model_config import Flux2Config
from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig
from max.profiler import Tracer, set_gpu_profiling_state
from pytest_benchmark.fixture import BenchmarkFixture

# Image resolutions to benchmark.
# image_seq_len = (H/16) * (W/16) for FLUX2 (VAE 8x downsample, then /2).
_IMAGE_CONFIGS = [
    (512, 1024),  # image_seq_len = 1024
    (1024, 4096),  # image_seq_len = 4096
]

_TEXT_SEQ_LEN = 256


# --------------------------------------------------------------------------- #
# Session-scoped fixtures & model compilation
# --------------------------------------------------------------------------- #


class CompiledTransformerBundle(NamedTuple):
    compiled_model: Any
    device: Accelerator
    config: Flux2Config
    step_cache_enabled: bool


def _build_compiled_transformer(
    step_cache: bool,
) -> CompiledTransformerBundle:
    device = Accelerator()
    config = Flux2Config(
        patch_size=1,
        in_channels=128,
        out_channels=None,
        num_layers=8,
        num_single_layers=48,
        attention_head_dim=128,
        num_attention_heads=48,
        joint_attention_dim=15360,
        timestep_guidance_channels=256,
        mlp_ratio=3.0,
        axes_dims_rope=(32, 32, 32, 32),
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=True,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
    )

    # Enable GPU profiling so Tracer spans emit NVTX ranges visible in nsys.
    # This is normally done by InferenceSession.gpu_profiling() in the real
    # pipeline; we call the low-level API directly since Module.compile()
    # manages its own session internally.
    set_gpu_profiling_state("detailed")

    cache_config = (
        DenoisingCacheConfig(first_block_caching=True, residual_threshold=0.05)
        if step_cache
        else None
    )

    t0 = time.perf_counter()
    with F.lazy():
        model = Flux2Transformer2DModel(config, cache_config=cache_config)
        model.to(device)

    # Create zero-initialized CPU Buffers and wrap as Tensors to pass to
    # compile() via the weights= parameter.  This mirrors the real pipeline
    # path (which passes checkpoint data from safetensors) and avoids
    # realizing the lazy random.normal() placeholders — which would each
    # compile and execute a separate mini-graph on GPU, exhausting the
    # memory manager cache (see MXF-143).
    weights: dict[str, Tensor] = {}
    for name, param in model.parameters:
        shape = tuple(int(d) for d in param.shape)
        buf = Buffer(param.dtype, shape)
        weights[name] = Tensor(storage=buf)

    compiled = model.compile(*model.input_types(), weights=weights)
    compile_s = time.perf_counter() - t0
    cache_str = "step-cache" if step_cache else "standard"
    print(
        f"\n[flux2-transformer/{cache_str}] Compilation time: {compile_s:.2f}s"
    )

    return CompiledTransformerBundle(
        compiled_model=compiled,
        device=device,
        config=config,
        step_cache_enabled=step_cache,
    )


@pytest.fixture(scope="session")
def compiled_transformer(
    request: pytest.FixtureRequest,
) -> CompiledTransformerBundle:
    step_cache: bool = request.config.getoption("--step-cache")
    return _build_compiled_transformer(step_cache)


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #


def _make_inputs(
    bundle: CompiledTransformerBundle,
    image_seq_len: int,
    text_seq_len: int,
) -> tuple[Any, ...]:
    """Build random input tensors for the transformer."""
    rng = np.random.default_rng(42)
    cfg = bundle.config
    dev = bundle.device
    dt = np.float32  # numpy doesn't support bfloat16

    def _bf16(arr: np.ndarray) -> Tensor:
        """Create a bfloat16 tensor on device from a float32 numpy array."""
        return Tensor(arr, dtype=DType.float32, device=dev).cast(DType.bfloat16)

    batch = 1
    hidden_states = _bf16(
        rng.standard_normal((batch, image_seq_len, cfg.in_channels)).astype(dt),
    )
    encoder_hidden_states = _bf16(
        rng.standard_normal(
            (batch, text_seq_len, cfg.joint_attention_dim)
        ).astype(dt),
    )
    timestep = _bf16(
        rng.random((batch,)).astype(dt),
    )
    img_ids = Tensor(
        rng.integers(0, 128, size=(batch, image_seq_len, 4)).astype(np.int64),
        device=dev,
    )
    txt_ids = Tensor(
        rng.integers(0, 128, size=(batch, text_seq_len, 4)).astype(np.int64),
        device=dev,
    )
    guidance = _bf16(
        np.full((batch,), 3.5, dtype=dt),
    )

    base_args: tuple[Any, ...] = (
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    )

    if not bundle.step_cache_enabled:
        return base_args

    inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
    out_channels = cfg.out_channels or cfg.in_channels
    out_dim = cfg.patch_size * cfg.patch_size * out_channels

    prev_residual = _bf16(
        rng.standard_normal((batch, image_seq_len, inner_dim)).astype(dt),
    )
    prev_output = _bf16(
        rng.standard_normal((batch, image_seq_len, out_dim)).astype(dt),
    )
    return (*base_args, prev_residual, prev_output)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "resolution,image_seq_len",
    _IMAGE_CONFIGS,
    ids=[f"{res}px_seq{seq}" for res, seq in _IMAGE_CONFIGS],
)
def test_benchmark_transformer(
    compiled_transformer: CompiledTransformerBundle,
    benchmark: BenchmarkFixture,
    resolution: int,
    image_seq_len: int,
) -> None:
    bundle = compiled_transformer
    inputs = _make_inputs(bundle, image_seq_len, _TEXT_SEQ_LEN)
    bundle.device.synchronize()

    cfg = bundle.config
    cache_tag = "_stepcache" if bundle.step_cache_enabled else ""
    span_name = (
        f"transformer_img{image_seq_len}_txt{_TEXT_SEQ_LEN}"
        f"_h{cfg.num_attention_heads}x{cfg.attention_head_dim}"
        f"_d{cfg.num_layers}+{cfg.num_single_layers}{cache_tag}"
    )

    def _run() -> None:
        with Tracer(span_name):
            bundle.compiled_model(*inputs)

    benchmark.pedantic(_run, rounds=20, warmup_rounds=3, iterations=1)
