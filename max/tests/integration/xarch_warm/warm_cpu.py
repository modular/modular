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
"""Producer: warm ONLY the CPU slot of the eager-interpreter GC sweep.

Sets no virtual accelerator, so the builder's real (GPU-less, ``cfg=exec``)
device set is just the CPU slot, and exports each family's CPU artifact
(``matmul_cpu.mef``, ...) into a :class:`~max.engine.MefStore` at ``--out``.

This action is config-independent per host arch: it takes only ``--cpu-target``
(no GPU ``--target``), so its Bazel action is byte-identical across every x86
GPU/CPU lane and the remote cache shares it, the ~20-min CPU sweep runs once
per host arch instead of once per lane. aarch64 is a valid CPU target here; the
x86 guard is GPU-mode only and lives in ``warm_accelerators``.
"""

from __future__ import annotations

import click
from max.driver import set_virtual_cpu_target
from max.engine import MefStore
from max.tests.integration.xarch_warm import warm_lib


@click.command()
@click.option(
    "--out",
    required=True,
    help="Directory to write the warm store into.",
)
@click.option(
    "--cpu-target",
    required=True,
    help="Virtual host-CPU target descriptor, e.g. "
    "'triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3'.",
)
@click.option(
    "--family",
    default=None,
    help="Warm only this GC family (a GC_FAMILIES name); default warms all.",
)
def main(out: str, cpu_target: str, family: str | None) -> None:
    # Pin the host-CPU target (the warmed CPU kernels' compile key includes it).
    # No virtual accelerator: the real GPU-less worker presents only the CPU
    # slot, so the sweep is CPU-only.
    set_virtual_cpu_target(cpu_target)

    warm_lib.export_slots(
        MefStore.for_export(out),
        include_cpu=True,
        include_accelerators=False,
        only_family=family,
    )


if __name__ == "__main__":
    main()
