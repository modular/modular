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
"""Reproduction of https://github.com/modular/modular/issues/6307

Verifies that repeated F.custom calls with the same custom_extensions
are fast after the initial warm-up (i.e., the eager model cache works).
"""

import os
import time
from pathlib import Path

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType


def run_my_add(extension_path: Path) -> Tensor:
    x = Tensor.ones([64], dtype=DType.float32, device=CPU())
    y = Tensor.ones([64], dtype=DType.float32, device=CPU())
    out_type = TensorType(dtype=x.dtype, shape=x.shape, device=x.device)

    result = F.custom(
        name="my_add",
        device=x.device,
        values=[x, y],
        out_types=[out_type],
        custom_extensions=[extension_path],
    )[0]
    return result


def test_custom_cache_timing() -> None:
    """Issue #6307: repeated F.custom calls should be fast after warm-up."""
    kernel_ops_path = Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])

    # Warm-up call (compilation expected)
    run_my_add(kernel_ops_path)

    # Timed calls — should hit the cache and be fast
    times = []
    for _ in range(5):
        start = time.monotonic()
        result = run_my_add(kernel_ops_path)
        elapsed = time.monotonic() - start
        times.append(elapsed)
        assert result.real
        print(f"execution time {elapsed:.4f} s")

    avg = sum(times) / len(times)
    print(f"average execution time: {avg:.4f} s")

    # With caching working, each call should be well under 1 second.
    # Without caching, each call takes ~2-4+ seconds (full compilation).
    assert avg < 1.0, (
        f"Average execution time {avg:.2f}s is too slow — "
        f"eager F.custom cache is likely not working. Times: {times}"
    )
