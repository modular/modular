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
"""Consumer: load the CPU-only eager-GC warm store on a GPU-less box.

``warm_cpu`` fragments hold only the CPU slot. This test runs with its GPUs
hidden, so ``accelerator_count() == 0``, and under ``MAX_EAGER_OP_PRECOMPILE=1``
(see ``eager_py_test.bzl``), so every family loads from the warm store at the
first ``_interpreter_ops`` import instead of compiling.
"""

from __future__ import annotations

import pytest
import torch
from max._interpreter_ops import gc_compile
from max.driver import accelerator_count
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.executor import InterpreterExecutor, set_default_executor
from max.experimental.tensor import Tensor, realization_context


def test_cpu_only_box_has_no_accelerator() -> None:
    # The CPU-only warm has no accelerator slot, so a box that showed one would
    # need slots this warm never built.
    assert accelerator_count() == 0, (
        f"expected a CPU-only box but accelerator_count()={accelerator_count()};"
        " CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES did not hide the GPUs"
    )


def test_every_family_loaded_from_the_store(
    capfd: pytest.CaptureFixture[str],
) -> None:
    a = torch.randn(8, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    s = torch.randn(64, dtype=torch.float32).abs()
    with set_default_executor(InterpreterExecutor()):
        with rc.EagerRealizationContext() as ctx, realization_context(ctx):
            y = Tensor.from_dlpack(a) @ Tensor.from_dlpack(w)
            z = F.sqrt(Tensor.from_dlpack(s))
        out = torch.from_dlpack(y)
        out_z = torch.from_dlpack(z)
    torch.testing.assert_close(out, a @ w, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_z, s.sqrt(), rtol=1e-2, atol=1e-2)

    not_loaded = [
        family.name
        for family in gc_compile.registered_families()
        if not gc_compile.adopted_from_store(family.name)
    ]
    assert not not_loaded, f"not loaded from the warm store: {not_loaded}"
    err = capfd.readouterr().err
    assert "compileMojoToBinary" not in err, (
        f"cold Mojo kernel recompile on the consumer:\n{err[-3000:]}"
    )
