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
"""Shared producer helpers for the split eager-interpreter warm cache.

The warm producer is split into single-purpose Bazel actions so the ~20-min CPU
sweep runs once per host arch instead of once per GPU lane (see ``BUILD.bazel``
and ``interp_cache.bzl``): ``warm_cpu`` warms only the CPU slot,
``warm_accelerators`` only the accelerator slots, and ``compose`` merges them.
This module holds the parts those two per-slot producers share, the
per-(op family, device) compile+export loop and the stamp/manifest writers --
so each script stays thin (it only chooses *which* devices and *which*
envelope).

Kept deliberately import-light: it must NOT import ``max._interpreter_ops`` at
module top. Those modules freeze their device set from ``accelerator_count()``
at import, so the producers must set the virtual-device knobs FIRST; the
deferred imports below therefore run inside the functions, after the caller has
set the knobs.
"""

from __future__ import annotations

import importlib
import os

from max.engine import InferenceSession


def export_slots(
    derived: str, *, include_cpu: bool, include_accelerators: bool
) -> list[dict[str, str]]:
    """Compile and export one single-device MEF per (op family, kept device).

    For each op family, and each device that family sweeps, filtered to the
    requested device classes, build that slot's per-device module, compile it on
    a single-device session, and export one MEF into *derived*. Returns the
    manifest entries indexing the exported MEFs.

    Per-slot MEFs (rather than one batched module spanning every device) keep
    the warm device-count-independent, and let ``warm_cpu`` and
    ``warm_accelerators`` each compile only their own device class, so the
    costly CPU sweep runs in exactly one Bazel action.

    Args:
        derived: The cache dir the MEFs are exported into (``MODULAR_DERIVED_PATH``).
        include_cpu: Warm the CPU slot.
        include_accelerators: Warm the accelerator slots.

    Returns:
        One ``{family, device_class, mef}`` entry per exported MEF.
    """
    # Deferred: importing these freezes their device set from the (already-set)
    # virtual-device knobs; see the module docstring.
    matmul_gc = importlib.import_module("max._interpreter_ops.matmul_gc")
    unary_gc = importlib.import_module(
        "max._interpreter_ops.unary_elementwise_gc"
    )

    entries: list[dict[str, str]] = []
    for family, devices, build_for_device in (
        (
            "matmul",
            matmul_gc._sweep_devices(),
            matmul_gc.build_matmul_module_for_device,
        ),
        (
            "unary",
            list(unary_gc._DEVICES),
            unary_gc.build_unary_module_for_device,
        ),
    ):
        for device in devices:
            is_cpu = device.label == "cpu"
            if is_cpu and not include_cpu:
                continue
            if not is_cpu and not include_accelerators:
                continue
            # The CPU slot is device_class "cpu"; each accelerator is
            # "gpu:{id}", keyed on the same id its graphs embed.
            if is_cpu:
                device_class, mef_name = "cpu", f"{family}_cpu.mef"
            else:
                device_class = f"gpu:{device.id}"
                mef_name = f"{family}_slot_{device.id}.mef"
            InferenceSession(devices=[device]).compile(
                build_for_device(device)
            ).export_mef(os.path.join(derived, mef_name))
            entries.append(
                {
                    "family": family,
                    "device_class": device_class,
                    "mef": mef_name,
                }
            )
    return entries


def write_stamp() -> bool:
    """Write the warm stamp for the current (virtual-device) context.

    Wraps :func:`gc_compile.write_warm_stamp` (deferred import, same reason as
    :func:`export_slots`). The stamp's context signature reflects whatever
    device set the caller's virtual knobs presented, accelerator count + arch
    in ``warm_accelerators``, or ``accelerators=0`` in ``warm_cpu``. Returns
    False if the cache dir can't be located (``MODULAR_DERIVED_PATH`` unset), so
    the caller can hard-fail the build action.
    """
    gc_compile = importlib.import_module("max._interpreter_ops.gc_compile")
    return gc_compile.write_warm_stamp()


def write_manifest(
    envelope: dict[str, object], entries: list[dict[str, str]]
) -> bool:
    """Write the force-load manifest. Wraps :func:`gc_compile.write_manifest`."""
    gc_compile = importlib.import_module("max._interpreter_ops.gc_compile")
    return gc_compile.write_manifest(envelope=envelope, entries=entries)
