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
"""Shared producer loop for the split eager-interpreter warm.

The warm is split into Bazel actions, one per (op family, device class), so the
~20-min CPU sweep runs once per host arch instead of once per GPU lane (see
``BUILD.bazel``). Each action writes a :class:`~max.engine.MefStore` directory
holding one artifact per (family, device slot); consumers union the
directories (see ``gc_compile.warm_store``).

Kept deliberately import-light: it must NOT import ``max._interpreter_ops`` at
module top. Those modules freeze their device set from ``accelerator_count()``
at import, so the producers must set the virtual-device knobs FIRST; the
deferred import below therefore runs inside the function, after the caller has
set the knobs.
"""

from __future__ import annotations

import importlib

from max.engine import InferenceSession, MefStore


def export_slots(
    store: MefStore,
    *,
    include_cpu: bool,
    include_accelerators: bool,
    only_family: str | None = None,
) -> None:
    """Compiles and exports one artifact per (op family, kept device).

    For each op family, and each device that family sweeps, filtered to the
    requested device classes, build that slot's per-device module, compile it
    on a single-device session, and export it into *store* under the name
    adoption looks it up by.

    Per-slot artifacts (rather than one module spanning every device) keep the
    warm device-count-independent, and let ``warm_cpu`` and
    ``warm_accelerators`` each compile only their own device class.

    Args:
        store: The exporting store the artifacts go into.
        include_cpu: Warm the CPU slot.
        include_accelerators: Warm the accelerator slots.
        only_family: Warm only this GC family (a ``GC_FAMILIES`` name); ``None``
            warms all.

    Raises:
        ValueError: If ``only_family`` names no family, or a family sweeps a
            device it builds nothing for. Adoption loads an artifact for every
            device a family sweeps, so such a slot would fail every consumer.
    """
    # Deferred import: freezes each family's device set from the (already-set)
    # virtual-device knobs, then reads the shared GC_FAMILIES registry.
    interpreter_ops = importlib.import_module("max._interpreter_ops")
    gc_compile = interpreter_ops.gc_compile

    families = interpreter_ops.GC_FAMILIES
    if only_family is not None:
        known = {f.name for f in families}
        if only_family not in known:
            raise ValueError(
                f"unknown GC family {only_family!r}; known: {sorted(known)}"
            )
        families = tuple(f for f in families if f.name == only_family)

    for family in families:
        for device in family.sweep_devices():
            is_cpu = device.label == "cpu"
            if is_cpu and not include_cpu:
                continue
            if not is_cpu and not include_accelerators:
                continue
            module = family.build_module_for_device(device)
            if not module.top_level_graph_names():
                raise ValueError(
                    f"{family.name} sweeps {device.label}:{device.id} but builds"
                    " no graph for it; exclude it in sweep_devices()"
                )
            name = gc_compile.warm_artifact_name(family.name, device)
            InferenceSession(devices=[device]).compile(module).export_mef(
                store.claim_export_artifact(name)
            )
    store.write_manifest()
