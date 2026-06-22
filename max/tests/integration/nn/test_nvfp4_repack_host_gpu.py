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
"""Byte-exact validation of the host NVFP4 repack against the GPU kernel.

Runs the ACTUAL ``repack_nvfp4_g16`` custom op (via ``nvfp4_skeleton_repack``)
on the GPU and asserts the produced combined buffer matches the pure-numpy
host replica ``nvfp4_repack_host`` BYTE-FOR-BYTE. This is the foundation of a
load-time repack: the host can produce the skeleton's combined weight buffer
so the original packed weight is never uploaded.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn._nvfp4_repack_host import nvfp4_repack_host
from max.nn.kernels import nvfp4_skeleton_repack


def _skip_if_unsupported() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available")
    if accelerator_api() == "hip":
        pytest.skip("NVFP4 skeleton path is Nvidia-only")


@pytest.mark.parametrize(
    "n,k",
    [
        (64, 128),  # smallest tile: one 64-row block, one WGROUP of K
        (128, 256),
        (256, 128),
        (512, 256),  # gemma4-ish shape
    ],
)
def test_nvfp4_repack_host_matches_gpu(n: int, k: int) -> None:
    _skip_if_unsupported()

    rng = np.random.default_rng(1234 + n * 100003 + k)
    packed_k = k // 2
    scale_k = k // 16

    weight_np = rng.integers(0, 256, size=(n, packed_k), dtype=np.uint8)
    # Small positive block scales on representable e4m3 values.
    scale_f32 = (rng.random((n, scale_k), dtype=np.float32) * 0.1).astype(
        np.float32
    )
    scale_fp8_t = torch.from_numpy(scale_f32).to(torch.float8_e4m3fn)
    # raw e4m3 bytes (uint8) as the host replica consumes them.
    scale_fp8_bytes = (
        scale_fp8_t.view(torch.uint8).cpu().numpy().reshape(n, scale_k)
    )
    global_scale = 0.5

    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)

    session = InferenceSession(devices=[device])
    with Graph(
        "nvfp4_repack_host_validate",
        input_types=[
            TensorType(DType.uint8, (n, packed_k), device=device_ref),
            TensorType(DType.float8_e4m3fn, (n, scale_k), device=device_ref),
            TensorType(DType.float32, (1,), device=device_ref),
        ],
    ) as graph:
        gw, gws, gws2 = graph.inputs
        combined = nvfp4_skeleton_repack(gw.tensor, gws.tensor, gws2.tensor)
        graph.output(combined)

    compiled = session.load(graph)
    out = compiled.execute(
        torch.from_numpy(weight_np).cuda(),
        scale_fp8_t.cuda(),
        torch.tensor([global_scale], dtype=torch.float32).cuda(),
    )
    gpu_combined = (
        torch.from_dlpack(out[0]).view(torch.uint8).cpu().numpy()
    )

    host_combined = nvfp4_repack_host(weight_np, scale_fp8_bytes, global_scale)

    assert gpu_combined.shape == host_combined.shape, (
        f"shape mismatch: gpu {gpu_combined.shape} host {host_combined.shape}"
    )

    gpu_flat = gpu_combined.reshape(-1)
    host_flat = host_combined.reshape(-1)
    if not np.array_equal(gpu_flat, host_flat):
        # Report the EXACT first mismatch for debugging signal.
        diff = np.flatnonzero(gpu_flat != host_flat)
        off = int(diff[0])
        weight_bytes = n * k // 2
        region = "weights" if off < weight_bytes else "scales"
        raise AssertionError(
            f"N={n} K={k}: first mismatch at flat byte offset {off} "
            f"(region={region}, weight_region_bytes={weight_bytes}): "
            f"gpu=0x{int(gpu_flat[off]):02x} host=0x{int(host_flat[off]):02x}; "
            f"total mismatches={diff.size}/{gpu_flat.size}"
        )

    np.testing.assert_array_equal(gpu_combined, host_combined)
    print(
        f"[BYTE-EXACT] N={n} K={k}: {gpu_combined.size} bytes match "
        f"(weights={n * k // 2}, scales={(k // 16) * n * 2})"
    )
