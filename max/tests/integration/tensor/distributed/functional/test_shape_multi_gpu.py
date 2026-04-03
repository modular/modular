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
"""Shape tests on real 4 GPUs — small representative subset.

Full coverage (including error paths, all op families, and edge cases)
lives in simulated_cpu and simulated_gpu targets.  Multi-GPU is limited
to a handful of tests because the EagerRealizationContext signal-buffer
lifecycle currently leaks GPU state after ~5 context teardowns on H100,
causing CUDA_ERROR_MISALIGNED_ADDRESS.  This is a known runtime issue,
not a dispatch logic bug — tracked for a proper fix.
"""

import numpy as np
from _test_helpers import from_np, shard, to_np
from max.driver import Accelerator
from max.experimental.distributed_functional.shape import (
    permute,
    reshape,
    transpose,
)
from max.experimental.sharding import DeviceMesh, Sharded

MESH = DeviceMesh(
    devices=(Accelerator(0), Accelerator(1), Accelerator(2), Accelerator(3)),
    mesh_shape=(4,),
    axis_names=("tp",),
)


class TestShapeMultiGPU:
    """One test per core shape op on real multi-GPU."""

    def test_transpose(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = shard(from_np(t_np), MESH, [Sharded(0)])
        out = transpose(t, 0, 1)
        assert out.placements == (Sharded(1),)
        np.testing.assert_allclose(to_np(out), t_np.T, rtol=1e-5)

    def test_reshape(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = shard(from_np(t_np), MESH, [Sharded(0)])
        out = reshape(t, (8, 2, 2))
        assert out.placements == (Sharded(0),)
        np.testing.assert_allclose(to_np(out), t_np.reshape(8, 2, 2), rtol=1e-5)

    def test_permute(self) -> None:
        t_np = np.arange(96, dtype=np.float32).reshape(4, 8, 3)
        t = shard(from_np(t_np), MESH, [Sharded(0)])
        out = permute(t, [2, 0, 1])
        assert out.placements == (Sharded(1),)
        np.testing.assert_allclose(
            to_np(out), t_np.transpose(2, 0, 1), rtol=1e-5
        )
