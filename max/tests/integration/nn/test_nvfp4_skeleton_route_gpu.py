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
"""Equivalence test for the pre-Blackwell NVFP4 multistage-skeleton GEMM.

Runs both the bespoke `nvfp4_gemm` and the skeleton path (`nvfp4_skeleton_repack`
+ `nvfp4_skeleton_gemm`) in one graph on IDENTICAL canonical NVFP4 weights and
asserts the outputs match. This validates the full graph wiring of the skeleton
path (custom-op registration, the one-time global-scale host read in the repack
op, and the Python wrappers) end to end through the engine -- the piece the
Mojo-level `test_nvfp4_repack_e2e` cannot cover.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    nvfp4_gemm,
    nvfp4_skeleton_gemm,
    nvfp4_skeleton_repack,
)


def _skip_if_unsupported() -> None:
    if accelerator_count() == 0:
        pytest.skip("No GPU available")
    if accelerator_api() == "hip":
        pytest.skip("NVFP4 skeleton path is Nvidia-only")


@pytest.mark.parametrize(
    "m,n,k",
    [
        (482, 4096, 4096),
        (64, 4096, 4096),
        (482, 15360, 3840),
    ],
)
def test_nvfp4_skeleton_matches_bespoke(m: int, n: int, k: int) -> None:
    _skip_if_unsupported()

    torch.manual_seed(0)
    device = Accelerator(0)
    device_ref = DeviceRef(device.label, device.id)

    packed_k = k // 2
    scale_k = k // 16
    weight = torch.randint(0, 256, (n, packed_k), dtype=torch.uint8)
    # Small positive block scales on representable e4m3 values.
    weight_scale = (torch.rand(n, scale_k, dtype=torch.float32) * 0.1).to(
        torch.float8_e4m3fn
    )
    weight_scale_2 = torch.tensor([0.5], dtype=torch.float32)
    x = torch.randn((m, k), dtype=torch.bfloat16) * 0.1

    session = InferenceSession(devices=[device])
    with Graph(
        "nvfp4_skeleton_vs_bespoke",
        input_types=[
            TensorType(DType.bfloat16, (m, k), device=device_ref),
            TensorType(DType.uint8, (n, packed_k), device=device_ref),
            TensorType(DType.float8_e4m3fn, (n, scale_k), device=device_ref),
            TensorType(DType.float32, (1,), device=device_ref),
        ],
    ) as graph:
        gx, gw, gws, gws2 = graph.inputs
        gx = gx.tensor
        gw = gw.tensor
        gws = gws.tensor
        gws2 = gws2.tensor

        scales_f32 = gws.cast(DType.float32) * gws2
        out_bespoke = nvfp4_gemm(gx, gw, scales_f32)

        combined = nvfp4_skeleton_repack(gw, gws, gws2)
        out_skeleton = nvfp4_skeleton_gemm(gx, combined)

        graph.output(out_bespoke, out_skeleton)

    compiled = session.load(graph)
    out = compiled.execute(
        x.cuda(),
        weight.cuda(),
        weight_scale.cuda(),
        weight_scale_2.cuda(),
    )
    bespoke = torch.from_dlpack(out[0]).float().cpu()
    skeleton = torch.from_dlpack(out[1]).float().cpu()

    assert bespoke.shape == (m, n)
    assert torch.isfinite(skeleton).all(), "skeleton output has NaN/Inf"
    # Both decode the same E2M1 codes and scales; the only divergence is the
    # bf16 GEMM accumulation order, so a loose relative tolerance is expected.
    torch.testing.assert_close(skeleton, bespoke, rtol=2e-2, atol=1e-2)
