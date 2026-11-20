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

"""Defaults for the UCX (NIXL) inter-node transport env.

UCX needs to be told which transports it may use (``UCX_TLS``) and which
network device to bind (``UCX_NET_DEVICES``). Left to its own devices on a
multi-NIC InfiniBand box, UCX auto-selects across every transport and NIC,
which hangs during metadata exchange or spreads traffic onto a non-local NIC.
Historically the transfer engine required the operator to set both by hand;
this module derives sensible defaults so a correctly-provisioned host works out
of the box, while any value the operator sets explicitly still wins.

The two knobs differ: ``UCX_TLS`` is essentially static, but the right
``UCX_NET_DEVICES`` is the IB device PCIe-local to the GPU this process drives,
which we resolve from the host topology.

An IB device on a different PCIe root complex from the GPU is not used. B200
hosts carry extra 2X HDR ConnectX-7 ports (apparently for NVLink fabric
management) that report an active InfiniBand port, but GPUDirect RDMA through
them fails with a remote access error. When IB is present but none of it is
GPU-local, UCX is kept off IB entirely.

That makes one host able to produce two configs: a worker on a GPU with a local
IB device pins it, and a worker on a GPU without one runs ``UCX_TLS=^ib``. NIXL
creates endpoints with peer-failure handling, which the shared-memory
transports lack, so the two need a common network transport for active
messages. On such a mixed host the pinned worker also enables ``tcp`` on the
host's Ethernet interfaces, which the ``^ib`` worker uses too.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("max.pipelines")


class _UcxDevice(Protocol):
    """The subset of ``max.driver.Device`` this module reads."""

    @property
    def api(self) -> str: ...
    @property
    def id(self) -> int: ...


_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_NET_CLASS_ROOT = Path("/sys/class/net")
_IB_CLASS_ROOT = Path("/sys/class/infiniband")
_VERBS_DEV_ROOT = Path("/dev/infiniband")
_VERBS_CLASS_ROOT = Path("/sys/class/infiniband_verbs")
_GDRDRV_NODE = Path("/dev/gdrdrv")


def _verbs_accessible_devices(
    *,
    verbs_dev_root: Path = _VERBS_DEV_ROOT,
    verbs_class_root: Path = _VERBS_CLASS_ROOT,
) -> set[str]:
    """Returns the IB device names this namespace can actually open.

    ``/sys/class/infiniband`` lists every device on the HOST, but a container
    exposes only a subset as verbs char nodes in ``/dev/infiniband``. Map each
    exposed ``uverbsN`` node back to its device via
    ``/sys/class/infiniband_verbs/uverbsN/ibdev`` — the same set
    ``ibv_devinfo -l`` prints. An empty result means the mapping is unavailable
    (older kernels, odd sysfs), in which case callers should not filter.
    """
    names: set[str] = set()
    try:
        nodes = list(verbs_dev_root.iterdir())
    except OSError:
        return names
    for node in nodes:
        if not node.name.startswith("uverbs"):
            continue
        try:
            names.add(
                (verbs_class_root / node.name / "ibdev").read_text().strip()
            )
        except OSError:
            continue
    return names


def _verbs_usable() -> bool:
    """Returns whether IB verbs are actually usable on this host.

    ``/sys/class/infiniband`` can list devices whose verbs char nodes
    (``/dev/infiniband/uverbsN``) are absent — for instance a test sandbox that
    bind-mounts host sysfs but not the device nodes. UCX cannot open such a
    device, so pinning it (and dropping the tcp fallback) would leave the worker
    with no transport. Gate NIC selection on a real uverbs node.
    """
    try:
        return any(
            p.name.startswith("uverbs") for p in _VERBS_DEV_ROOT.iterdir()
        )
    except OSError:
        return False


def _plugin_lacks_ib() -> bool:
    """Returns whether NIXL selected the plain ``cuda`` UCX plugin flavor.

    That flavor has no IB transports; ``Support/lib/NixlPluginDir.cpp`` picks
    it whenever ``cuda-verbs`` cannot load, so pinning ``rc`` would leave none.
    """
    plugin_dir = os.environ.get("NIXL_PLUGIN_DIR")
    return plugin_dir is not None and Path(plugin_dir).name == "cuda"


def default_ucx_tls(*, gdr_copy: bool | None = None, tcp: bool = False) -> str:
    """Returns the default ``UCX_TLS`` transport list for GPU transfers.

    Covers every good path and omits the ones that hang on a multi-NIC box
    (notably ``tcp``/``ud``): ``cuda_ipc`` for same-node NVLink peers, ``rc``
    for inter-node InfiniBand RDMA, ``sm``/``self`` for the local worker's
    loopback, and ``cuda_copy`` to stage through host memory. ``gdr_copy``
    (direct GPU-NIC copy) is added when ``/dev/gdrdrv`` is present.

    Args:
        gdr_copy: Whether to include the ``gdr_copy`` transport. Defaults to
            probing ``/dev/gdrdrv``.
        tcp: Whether to include ``tcp``, for reaching same-host peers that
            run without InfiniBand. Callers must also restrict
            ``UCX_NET_DEVICES`` so it does not span every NIC.
    """
    if gdr_copy is None:
        gdr_copy = _GDRDRV_NODE.exists()
    transports = ["cuda_ipc", "cuda_copy", "rc", "sm", "self"]
    if gdr_copy:
        transports.append("gdr_copy")
    if tcp:
        transports.append("tcp")
    return ",".join(transports)


def _gpu_pci_bus_id(device: _UcxDevice) -> str | None:
    """Returns the PCIe bus id (``0000:65:00.0``) of a CUDA device, or None.

    Uses the CUDA driver's ``cuDeviceGetPCIBusId`` on the device's own ordinal,
    so it agrees with the ordinal MAX uses (avoiding the NVML-vs-CUDA device
    ordering hazard). Best-effort: returns None for non-CUDA devices or if the
    driver is unavailable.
    """
    if device.api != "cuda":
        return None
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    libcuda.cuInit(0)  # idempotent; MAX has already initialized the driver
    handle = ctypes.c_int()
    if libcuda.cuDeviceGet(ctypes.byref(handle), ctypes.c_int(device.id)) != 0:
        return None
    buf = ctypes.create_string_buffer(32)
    if libcuda.cuDeviceGetPCIBusId(buf, ctypes.c_int(len(buf)), handle) != 0:
        return None
    return buf.value.decode().lower()


def _first_active_ib_port(ports_dir: Path) -> str | None:
    """Returns the first InfiniBand-link-layer port that is ACTIVE, else None.

    Filters out non-IB RDMA devices (for example, AWS EFA reports a
    non-InfiniBand link layer) and down ports.
    """
    if not ports_dir.is_dir():
        return None
    for port in sorted(ports_dir.iterdir()):
        try:
            link_layer = (port / "link_layer").read_text().strip()
            state = (port / "state").read_text().strip()
        except OSError:
            continue
        if link_layer == "InfiniBand" and "ACTIVE" in state:
            return port.name
    return None


def _pci_root(path: str) -> str | None:
    """Returns the PCIe root complex (``pci0000:55``) a sysfs device sits under."""
    return next(
        (part for part in path.split("/") if part.startswith("pci")), None
    )


def _ib_devices(
    *,
    ib_class_root: Path = _IB_CLASS_ROOT,
    verbs_dev_root: Path = _VERBS_DEV_ROOT,
    verbs_class_root: Path = _VERBS_CLASS_ROOT,
) -> list[tuple[str, str, str]]:
    """Returns ``(device, port, pci_path)`` for each usable IB device.

    Only devices with an active InfiniBand-link-layer port that this namespace
    can actually open are listed. ``/sys/class/infiniband`` lists every host
    NIC, but a container exposes only a subset, and pinning a non-exposed one
    leaves UCX with no usable device.
    """
    if not ib_class_root.is_dir():
        return []

    # An empty set means the mapping is unavailable, so don't filter.
    accessible = _verbs_accessible_devices(
        verbs_dev_root=verbs_dev_root, verbs_class_root=verbs_class_root
    )

    devices = []
    # Sorted so ties resolve deterministically to the lowest-numbered device.
    for entry in sorted(ib_class_root.iterdir()):
        if accessible and entry.name not in accessible:
            continue
        port = _first_active_ib_port(entry / "ports")
        if port is None:
            continue
        devices.append((entry.name, port, os.path.realpath(entry / "device")))
    return devices


def local_ib_device(
    gpu_bus_id: str,
    *,
    pci_devices_root: Path = _PCI_DEVICES_ROOT,
    ib_class_root: Path = _IB_CLASS_ROOT,
    verbs_dev_root: Path = _VERBS_DEV_ROOT,
    verbs_class_root: Path = _VERBS_CLASS_ROOT,
) -> str | None:
    """Returns the ``<device>:<port>`` of the IB HCA PCIe-closest to a GPU.

    Ranks each device from :func:`_ib_devices` by the length of the
    ``/sys/devices`` PCIe path it shares with the GPU — more shared bridges
    means a closer (lower latency, GPUDirect-capable) device. A device under a
    different PCIe root complex is never returned. Returns None when no such
    device exists.

    Args:
        gpu_bus_id: PCIe bus id of the GPU, for example, ``0000:65:00.0``.
        pci_devices_root: Override of ``/sys/bus/pci/devices`` (for testing).
        ib_class_root: Override of ``/sys/class/infiniband`` (for testing).
        verbs_dev_root: Override of ``/dev/infiniband`` (for testing).
        verbs_class_root: Override of ``/sys/class/infiniband_verbs``.
    """
    gpu_path = os.path.realpath(pci_devices_root / gpu_bus_id)
    # Unknown when the GPU has no sysfs entry; rank by path alone then.
    gpu_root = _pci_root(gpu_path)

    best_shared = -1
    best_device: str | None = None
    for name, port, ib_path in _ib_devices(
        ib_class_root=ib_class_root,
        verbs_dev_root=verbs_dev_root,
        verbs_class_root=verbs_class_root,
    ):
        if gpu_root is not None and _pci_root(ib_path) != gpu_root:
            continue
        shared = _shared_path_len(gpu_path, ib_path)
        if shared > best_shared:
            best_shared = shared
            best_device = f"{name}:{port}"
    return best_device


_NVIDIA_VENDOR_ID = "0x10de"
# PCI class codes for a VGA controller (0x0300) and a 3D controller (0x0302);
# the latter is how datacenter GPUs enumerate. NVSwitches are bridges (0x0680).
_GPU_CLASS_PREFIXES = ("0x0300", "0x0302")


def _host_gpu_bus_ids(
    *, pci_devices_root: Path = _PCI_DEVICES_ROOT
) -> list[str]:
    """Returns the PCIe bus ids of every NVIDIA GPU on the host.

    Reads sysfs rather than CUDA, which only enumerates the GPUs in
    ``CUDA_VISIBLE_DEVICES``. Same-host peers can drive the others.
    """
    try:
        entries = sorted(pci_devices_root.iterdir())
    except OSError:
        return []
    bus_ids = []
    for entry in entries:
        try:
            vendor = (entry / "vendor").read_text().strip()
            pci_class = (entry / "class").read_text().strip()
        except OSError:
            continue
        if vendor == _NVIDIA_VENDOR_ID and pci_class.startswith(
            _GPU_CLASS_PREFIXES
        ):
            bus_ids.append(entry.name)
    return bus_ids


def _host_has_gpu_without_local_ib() -> bool:
    """Returns whether a same-host peer may run with ``UCX_TLS=^ib``."""
    return any(local_ib_device(bus) is None for bus in _host_gpu_bus_ids())


def _ethernet_netdevs(*, net_class_root: Path = _NET_CLASS_ROOT) -> list[str]:
    """Returns the names of the host's up Ethernet interfaces.

    Skips loopback and IPoIB. A pod's address may live on any of these (an
    IPv6 pod IP on ``eth0`` next to an IPv4 egress link), so all are kept.
    """
    try:
        entries = sorted(net_class_root.iterdir())
    except OSError:
        return []
    names = []
    for entry in entries:
        try:
            if (entry / "type").read_text().strip() != "1":  # ARPHRD_ETHER
                continue
            operstate = (entry / "operstate").read_text().strip()
        except OSError:
            continue
        if entry.name != "lo" and operstate in ("up", "unknown"):
            names.append(entry.name)
    return names


def _shared_path_len(a: str, b: str) -> int:
    """Number of leading ``/``-separated components two paths share."""
    shared = 0
    for x, y in zip(a.split("/"), b.split("/"), strict=False):
        if x != y:
            break
        shared += 1
    return shared


def configure_ucx_env(device: _UcxDevice) -> None:
    """Sets UCX transport defaults for a GPU NIXL/UCX agent, if unset.

    Applies :func:`default_ucx_tls` and, for CUDA devices, the GPU-local IB
    device from :func:`local_ib_device`. If IB devices exist but none is local
    to the GPU, sets ``UCX_TLS=^ib`` instead so UCX keeps its other transports
    but stays off those devices. When the host also has a GPU without a local
    IB device, a pinned worker adds ``tcp`` on the host's Ethernet interfaces
    so a ``^ib`` peer on that GPU can still connect to it. All of these use
    ``setdefault`` semantics, so a value the operator exported explicitly is
    never overridden. Best-effort: any failure to resolve the NIC leaves
    ``UCX_NET_DEVICES`` unset, which the transfer engine's inter-node check
    then reports.
    """
    # These defaults name CUDA and InfiniBand transports. AMD (ROCm) UCX
    # exposes rocm_* transports instead, so forcing this UCX_TLS there would
    # leave it with no usable transport; leave non-CUDA devices (and hosts) to
    # configure themselves.
    if device.api != "cuda":
        return

    # An operator-pinned device wins; still give it a matching TLS default.
    if "UCX_NET_DEVICES" in os.environ:
        os.environ.setdefault("UCX_TLS", default_ucx_tls())
        return

    # Only touch the env when IB verbs are actually usable here. Otherwise —
    # a non-IB host or a sandbox with a sysfs-only device — leave UCX to its
    # defaults so its tcp/cuda_ipc transports keep intra-node transfers working.
    if not _verbs_usable():
        return
    if _plugin_lacks_ib():
        logger.warning(
            "InfiniBand devices are present but NIXL loaded the UCX plugin "
            "without IB transports (%s); leaving UCX_TLS/UCX_NET_DEVICES "
            "unset so UCX falls back to its defaults, including tcp.",
            os.environ["NIXL_PLUGIN_DIR"],
        )
        return
    try:
        bus_id = _gpu_pci_bus_id(device)
        if bus_id is None:
            return
        net_device = local_ib_device(bus_id)
        remote_ib = net_device is None and bool(_ib_devices())
        tcp_netdevs = (
            _ethernet_netdevs()
            if net_device is not None and _host_has_gpu_without_local_ib()
            else []
        )
    except Exception:
        # Never let topology probing break agent creation; the operator can
        # always set UCX_NET_DEVICES explicitly.
        logger.debug("UCX_NET_DEVICES auto-derivation failed", exc_info=True)
        return
    if remote_ib:
        # UCX would otherwise pick rc on these devices and GPUDirect into the
        # GPU across root complexes, which fails on the first transfer.
        if "UCX_TLS" not in os.environ:
            os.environ["UCX_TLS"] = "^ib"
            logger.info(
                "No InfiniBand device shares a PCIe root complex with GPU %s "
                "at %s; set UCX_TLS=^ib to keep UCX off InfiniBand.",
                device.id,
                bus_id,
            )
        return
    if net_device is None:
        return
    os.environ["UCX_NET_DEVICES"] = ",".join([net_device, *tcp_netdevs])
    os.environ.setdefault("UCX_TLS", default_ucx_tls(tcp=bool(tcp_netdevs)))
    logger.info(
        "Auto-selected UCX_NET_DEVICES=%s (InfiniBand device local to GPU %s "
        "at %s)",
        os.environ["UCX_NET_DEVICES"],
        device.id,
        bus_id,
    )
    if tcp_netdevs:
        logger.info(
            "A GPU on this host has no local InfiniBand device; enabled tcp "
            "so its UCX_TLS=^ib workers can connect to this one."
        )
