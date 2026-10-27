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
"""GPU execution tests (TP2) for the V4.1 mHC stream chains.

Every test in this target executes over a two-way GPU mesh and fails --
never skips -- when fewer than two devices are visible: the target
requests two GPUs through ``test.resources:gpu-memory`` = 2, so a green
test.log proves real TP2 execution.

The chains exercise the placement contract the gate claims fix: the
ruleless gate outputs are claimed onto the stream's mapping before the
collapse's rule math consumes them, and the identity mix is built from
the mapping, never from a single-device read.
"""

from __future__ import annotations

import numpy as np
import torch
from max.dtype import DType
from max.experimental.sharding import DeviceMapping, DeviceMesh, Replicated
from max.experimental.tensor import Tensor, TensorLayout
from max.nn.module import Module
from max.pipelines.architectures.deepseekV41_modulev3.layers.mhc import (
    mhc_initial,
)
from mhc_chain_reference import (
    HC,
    HC_EPS,
    HIDDEN,
    NORM_EPS,
    SINKHORN_ITERS,
    mesh,
    param,
    require_two_devices,
    round_bf16,
    run_chain,
    run_prod_chain,
    site_params,
)


def test_shifted_two_site_chain_tp2() -> None:
    """The shifted chain over a two-way mesh, replicated end to end."""
    require_two_devices("shifted chain TP2")
    got, want = run_chain(mesh(), HIDDEN)
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)


def test_initial_identity_collapse_tp2() -> None:
    """``mhc_initial`` over the mesh copies stream 0 unweighted."""

    class _Initial(Module[[Tensor], Tensor]):
        fn: Tensor
        base: Tensor
        scale: Tensor

        def __init__(self, mesh: DeviceMesh) -> None:
            super().__init__()
            rng = np.random.default_rng(43)
            fn, base, scale = site_params(rng, HIDDEN)
            self.fn = param(fn, mesh)
            self.base = param(base, mesh)
            self.scale = param(scale, mesh)
            self.host = {"fn": fn, "base": base, "scale": scale}

        def forward(self, stream: Tensor) -> Tensor:
            _, _, _, collapsed = mhc_initial(
                stream,
                self.fn,
                self.scale,
                self.base,
                HC,
                norm_eps=NORM_EPS,
                hc_eps=HC_EPS,
                sinkhorn_iters=SINKHORN_ITERS,
            )
            return collapsed

    require_two_devices("initial identity TP2")
    devices = mesh()
    rng = np.random.default_rng(44)
    stream = round_bf16(
        (rng.standard_normal((5, HC, HIDDEN)) * 0.3).astype(np.float32)
    )
    module = _Initial(devices)
    run = module.compile(
        TensorLayout(
            DType.bfloat16,
            [5, HC, HIDDEN],
            DeviceMapping(devices, (Replicated(),)),
        ),
        weights=dict(module.host),
    )
    out = run(
        Tensor.from_dlpack(torch.from_numpy(stream).to(torch.bfloat16)).to(
            DeviceMapping(devices, (Replicated(),))
        )
    )
    np.testing.assert_array_equal(
        out.cast(DType.float32).to_numpy(), stream[:, 0]
    )


def test_production_chain_collapses_are_live_tp2() -> None:
    """The live-collapse production chain over the two-way mesh."""
    require_two_devices("production chain TP2")
    got, want = run_prod_chain(mesh(), HIDDEN)
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)
