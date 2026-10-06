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
"""Runs EP MoE forward passes under a kernel tracer.

``test_ep_side_stream.py`` launches this under ``rocprofv3`` once per arm of
its A/B, reading ``MODULAR_OVERLAP_SHARED_EXPERT`` from the environment. It is
a separate process because the tracer attributes every kernel in the process
it wraps: running pytest itself under the tracer would mix in collection and
any other test's kernels.

The model comes from the same ``conftest`` builder as the BF16 EP MoE tests,
which sets ``has_shared_experts=True`` and leaves ``EPConfig``'s
``fused_shared_expert`` and ``use_allreduce`` at their ``False`` defaults --
exactly the condition under which ``forward_moe_sharded_layers`` routes the
shared expert through ``ops.side_stream``. It spans two devices rather than
the fixture's four: two already give the cross-GPU dispatch/combine the shared
expert is meant to overlap, and they fit the remote multi-GPU pools.

Exit codes: 0 ran, 2 skipped (not enough accelerators, or EP init declined).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from max.driver import Buffer, accelerator_count

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    HIDDEN_DIM,
    _build_compiled_ep_models,
    generate_moe_weights,
)

N_DEVICES = 2
SKIP_EXIT_CODE = 2
TOKENS_PER_DEVICE = 128


def main() -> int:
    if accelerator_count() < N_DEVICES:
        print(
            f"skip: need {N_DEVICES} accelerators, found {accelerator_count()}",
            file=sys.stderr,
        )
        return SKIP_EXIT_CODE

    models = _build_compiled_ep_models(
        generate_moe_weights(), n_devices=N_DEVICES
    )
    if models is None:
        print("skip: EP model build declined", file=sys.stderr)
        return SKIP_EXIT_CODE

    inputs_torch = [
        torch.randn(
            TOKENS_PER_DEVICE, HIDDEN_DIM, dtype=torch.bfloat16, device="cpu"
        )
        for _ in range(N_DEVICES)
    ]
    inputs = [
        Buffer.from_dlpack(t).to(models.devices[i])
        for i, t in enumerate(inputs_torch)
    ]
    extra_inputs = models.ep_comm_init.model_inputs()

    # Whether an iteration overlaps varies run to run, and early iterations
    # tend not to; one MI355X run overlapped in only 2 of 5 on one device.
    # Enough iterations that "never overlapped" means serialized, not unlucky.
    iters = int(os.environ.get("EP_SIDE_STREAM_ITERS", "20"))
    for _ in range(iters):
        results = models.moe_model.execute(*inputs, *extra_inputs)
        # Pull every output back to host: the execute is asynchronous, and
        # without a consumer the process can exit before the kernels the
        # tracer is meant to see have run.
        for result in results:
            torch.from_dlpack(result).to("cpu")

    print(
        "ran"
        f" iters={iters}"
        f" devices={N_DEVICES}"
        f" overlap={os.environ.get('MODULAR_OVERLAP_SHARED_EXPERT', '1')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
