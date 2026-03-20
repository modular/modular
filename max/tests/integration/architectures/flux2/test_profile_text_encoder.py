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

"""Benchmark FLUX2 TextEncoder for kernel launch determinism.

Builds a TextEncoder with random weights, compiles it once, then runs
identical input through it N times via pytest-benchmark.

Supports two encoder variants via the ``--encoder`` flag:

    mistral3  (default) — Mistral3 encoder used by FLUX.2 Dev
    qwen3               — Qwen3 encoder used by FLUX.2 Klein

Usage:
    bt //max/tests/integration/architectures/flux2:benchmark_text_encoder \
        --test_output=streamed --curses=no --noshow_progress

    # Benchmark the Klein (Qwen3) encoder:
    bt //max/tests/integration/architectures/flux2:benchmark_text_encoder \
        --test_output=streamed --curses=no --noshow_progress \
        --test_arg=--encoder --test_arg=qwen3

    # Profile with nsys:
    br-nsys //max/tests/integration/architectures/flux2:benchmark_text_encoder
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.pipelines.architectures.mistral3.text_encoder.mistral3 import (
    Mistral3TextEncoderTransformer,
)
from max.pipelines.architectures.mistral3.text_encoder.model_config import (
    Mistral3TextEncoderConfig,
)
from max.pipelines.architectures.qwen3.text_encoder.model_config import (
    Qwen3TextEncoderConfig,
)
from max.pipelines.architectures.qwen3.text_encoder.qwen3 import (
    Qwen3TextEncoderTransformer,
)
from max.profiler import Tracer
from pytest_benchmark.fixture import BenchmarkFixture

# --------------------------------------------------------------------------- #
# Encoder variant registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EncoderVariant:
    """All the knobs that differ between encoder backends."""

    name: str
    config_factory: Callable[..., Any]
    model_factory: Callable[..., Any]


def _mistral3_config(*, max_seq_len: int, device: DeviceRef) -> Any:
    return Mistral3TextEncoderConfig(
        hidden_size=5120,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=31,
        head_dim=128,
        vocab_size=131072,
        intermediate_size=14336,
        rope_theta=1000000.0,
        max_seq_len=max_seq_len,
        rms_norm_eps=1e-5,
        dtype=DType.bfloat16,
        device=device,
        hidden_state_layers=[10, 20, 30],
    )


def _qwen3_config(*, max_seq_len: int, device: DeviceRef) -> Any:
    return Qwen3TextEncoderConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=28,
        head_dim=128,
        vocab_size=151936,
        intermediate_size=12288,
        rope_theta=1000000.0,
        max_seq_len=max_seq_len,
        rms_norm_eps=1e-6,
        dtype=DType.bfloat16,
        device=device,
        hidden_state_layers=[9, 18, 27],
    )


_ENCODER_VARIANTS: dict[str, EncoderVariant] = {
    "mistral3": EncoderVariant(
        name="mistral3",
        config_factory=_mistral3_config,
        model_factory=Mistral3TextEncoderTransformer,
    ),
    "qwen3": EncoderVariant(
        name="qwen3",
        config_factory=_qwen3_config,
        model_factory=Qwen3TextEncoderTransformer,
    ),
}

_DEFAULT_SEQ_LENS = [128, 256, 512]


# --------------------------------------------------------------------------- #
# Session-scoped fixtures & model compilation
# --------------------------------------------------------------------------- #


class CompiledTextEncoderBundle(NamedTuple):
    compiled_model: Any
    device: Accelerator
    config: Any


def _build_compiled_text_encoder(
    variant: EncoderVariant,
    seq_lens: list[int],
) -> CompiledTextEncoderBundle:
    device = Accelerator()
    max_seq_len = max(s * 2 for s in seq_lens)
    config = variant.config_factory(
        max_seq_len=max(max_seq_len, 1024),
        device=DeviceRef.GPU(),
    )

    t0 = time.perf_counter()
    with F.lazy():
        model = variant.model_factory(config)
        model.to(device)

    compiled = model.compile(*model.input_types())
    compile_s = time.perf_counter() - t0
    print(f"\n[{variant.name}] Compilation time: {compile_s:.2f}s")

    return CompiledTextEncoderBundle(
        compiled_model=compiled,
        device=device,
        config=config,
    )


@pytest.fixture(scope="session")
def compiled_text_encoder(
    request: pytest.FixtureRequest,
) -> CompiledTextEncoderBundle:
    variant_name: str = request.config.getoption("--encoder")
    variant = _ENCODER_VARIANTS[variant_name]
    return _build_compiled_text_encoder(variant, _DEFAULT_SEQ_LENS)


# --------------------------------------------------------------------------- #
# Test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "seq_len",
    _DEFAULT_SEQ_LENS,
    ids=[f"seq{s}" for s in _DEFAULT_SEQ_LENS],
)
def test_benchmark_text_encoder(
    compiled_text_encoder: CompiledTextEncoderBundle,
    benchmark: BenchmarkFixture,
    seq_len: int,
) -> None:
    bundle = compiled_text_encoder

    # Fixed input — identical across all iterations.
    rng = np.random.default_rng(123)
    tokens_np = rng.integers(
        0, bundle.config.vocab_size, size=(seq_len,), dtype=np.int64
    )
    tokens = Tensor(tokens_np, device=bundle.device)
    bundle.device.synchronize()

    def _run() -> None:
        with Tracer("text_encoder_iteration") as tracer:
            tracer.push("forward")
            bundle.compiled_model(tokens)
            bundle.device.synchronize()
            tracer.pop()

    benchmark.pedantic(_run, rounds=50, warmup_rounds=5, iterations=1)
