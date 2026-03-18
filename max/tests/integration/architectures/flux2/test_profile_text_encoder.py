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

Builds the Mistral3 TextEncoder (used by FLUX2) with random weights, compiles
it once, then runs identical input through it N times via pytest-benchmark.

Usage:
    bt //max/tests/integration/architectures/flux2:benchmark_text_encoder \
        --test_output=streamed --curses=no --noshow_progress

    # Profile with nsys:
    br-nsys //max/tests/integration/architectures/flux2:benchmark_text_encoder
"""

from __future__ import annotations

import time
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
from max.profiler import Tracer
from pytest_benchmark.fixture import BenchmarkFixture

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_HIDDEN_SIZE = 5120
_NUM_HEADS = 32
_NUM_KV_HEADS = 8
_NUM_LAYERS = 31  # forward stops at layer 30
_HEAD_DIM = 128
_VOCAB_SIZE = 131072
_INTERMEDIATE_SIZE = 14336
_HIDDEN_STATE_LAYERS = [10, 20, 30]

_DEFAULT_SEQ_LENS = [128, 256, 512]


# --------------------------------------------------------------------------- #
# Session-scoped fixtures & model compilation
# --------------------------------------------------------------------------- #


class CompiledTextEncoderBundle(NamedTuple):
    compiled_model: Any
    device: Accelerator
    config: Mistral3TextEncoderConfig


def _make_config(max_seq_len: int) -> Mistral3TextEncoderConfig:
    return Mistral3TextEncoderConfig(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_hidden_layers=_NUM_LAYERS,
        head_dim=_HEAD_DIM,
        vocab_size=_VOCAB_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        rope_theta=1000000.0,
        max_seq_len=max_seq_len,
        rms_norm_eps=1e-5,
        dtype=DType.bfloat16,
        device=DeviceRef.GPU(),
        hidden_state_layers=_HIDDEN_STATE_LAYERS,
    )


def _build_compiled_text_encoder(
    seq_lens: list[int],
) -> CompiledTextEncoderBundle:
    device = Accelerator()
    max_seq_len = max(s * 2 for s in seq_lens)
    config = _make_config(max_seq_len=max(max_seq_len, 1024))

    t0 = time.perf_counter()
    with F.lazy():
        model = Mistral3TextEncoderTransformer(config)
        model.to(device)

    # weights=None lets compile() use the module's own lazy parameters.
    compiled = model.compile(*model.input_types())
    compile_s = time.perf_counter() - t0
    print(f"\nCompilation time: {compile_s:.2f}s")

    return CompiledTextEncoderBundle(
        compiled_model=compiled,
        device=device,
        config=config,
    )


@pytest.fixture(scope="session")
def compiled_text_encoder() -> CompiledTextEncoderBundle:
    return _build_compiled_text_encoder(_DEFAULT_SEQ_LENS)


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
