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
"""Consumer: load the eager-GC warm store on a REAL GPU.

The warm is built with no physical GPU: ``warm_cpu`` fragments for the CPU slot
and ``warm_accelerators`` fragments for ``WARM_DEVICE_COUNT`` virtual
accelerators of this lane's arch. This test runs on a real multi-GPU box with
them named as its warm store (see ``eager_py_test.bzl``), and checks that CPU and
GPU dispatches use models loaded from it, with no cold recompile. A box with
fewer GPUs than were warmed loads only slots ``0..k-1``.
"""

from __future__ import annotations

import pytest
import torch
from max._interpreter_ops import gc_compile
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.executor import InterpreterExecutor, set_default_executor
from max.experimental.tensor import Tensor, realization_context


def test_matmul_cpu_and_gpu_load_from_the_store(
    capfd: pytest.CaptureFixture[str],
) -> None:
    a_cpu = torch.randn(8, 16, dtype=torch.float32)
    w_cpu = torch.randn(16, 32, dtype=torch.float32)
    a_gpu = torch.randn(8, 16, dtype=torch.float32, device="cuda")
    w_gpu = torch.randn(16, 32, dtype=torch.float32, device="cuda")
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y_cpu = Tensor.from_dlpack(a_cpu) @ Tensor.from_dlpack(w_cpu)
            y_gpu = Tensor.from_dlpack(a_gpu) @ Tensor.from_dlpack(w_gpu)
        out_cpu = torch.from_dlpack(y_cpu)
        out_gpu = torch.from_dlpack(y_gpu)
    torch.testing.assert_close(out_cpu, a_cpu @ w_cpu, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_gpu, a_gpu @ w_gpu, rtol=1e-2, atol=1e-2)

    assert gc_compile.adopted_from_store("matmul")
    err = capfd.readouterr().err
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )


def test_unary_cpu_and_gpu_load_from_the_store(
    capfd: pytest.CaptureFixture[str],
) -> None:
    a_cpu = torch.randn(64, dtype=torch.float32).abs()
    a_gpu = torch.randn(64, dtype=torch.float32, device="cuda").abs()
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y_cpu = F.sqrt(Tensor.from_dlpack(a_cpu))
            y_gpu = F.sqrt(Tensor.from_dlpack(a_gpu))
        out_cpu = torch.from_dlpack(y_cpu)
        out_gpu = torch.from_dlpack(y_gpu)
    torch.testing.assert_close(out_cpu, a_cpu.sqrt(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_gpu, a_gpu.sqrt(), rtol=1e-2, atol=1e-2)

    assert gc_compile.adopted_from_store("unary")
    err = capfd.readouterr().err
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )
