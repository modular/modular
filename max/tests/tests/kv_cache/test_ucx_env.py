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

"""Unit tests for UCX transport-env defaulting (ucx_env)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pytest
from max.pipelines.kv_cache.paged_kv_cache import _ucx_env
from max.pipelines.kv_cache.paged_kv_cache._ucx_env import (
    configure_ucx_env,
    default_ucx_tls,
    local_ib_device,
)


class _FakeDevice:
    """Duck-typed stand-in for max.driver.Device (api + id only)."""

    def __init__(self, api: str, id: int) -> None:
        self.api = api
        self.id = id


def test_default_ucx_tls_keeps_nvlink_and_ib() -> None:
    tls = default_ucx_tls(gdr_copy=False).split(",")
    # NVLink (cuda_ipc) and IB RDMA (rc) must both survive, plus the local
    # loopback transports; tcp/ud must not appear (they cause the hang).
    assert {"cuda_ipc", "cuda_copy", "rc", "sm", "self"} == set(tls)
    assert "tcp" not in tls and "ud" not in tls


def test_default_ucx_tls_adds_gdr_copy_when_present() -> None:
    assert "gdr_copy" in default_ucx_tls(gdr_copy=True).split(",")
    assert "gdr_copy" not in default_ucx_tls(gdr_copy=False).split(",")


def _make_pci_leaf(devices_root: Path, *segments: str) -> Path:
    """Creates a nested PCIe device path and returns the leaf dir."""
    leaf = devices_root.joinpath(*segments)
    leaf.mkdir(parents=True, exist_ok=True)
    return leaf


def _make_ib_device(
    ib_root: Path, name: str, pci_leaf: Path, link_layer: str, state: str
) -> None:
    dev = ib_root / name
    (dev).mkdir(parents=True)
    os.symlink(pci_leaf, dev / "device")
    port = dev / "ports" / "1"
    port.mkdir(parents=True)
    (port / "link_layer").write_text(link_layer + "\n")
    (port / "state").write_text(state + "\n")


def _make_gpu(pci_devices: Path, bus_id: str, leaf: Path) -> None:
    (leaf / "vendor").write_text("0x10de\n")
    (leaf / "class").write_text("0x030200\n")
    os.symlink(leaf, pci_devices / bus_id)


@pytest.fixture
def fake_sysfs(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic /sys with a GPU and three RDMA devices.

    Topology (shared bridge = closer):
      pci0000:60/0000:60:01.0/0000:65:00.0   GPU
      pci0000:60/0000:60:01.0/0000:67:00.0   mlx5_same_bridge  (closest, IB)
      pci0000:60/0000:60:02.0/0000:66:00.0   mlx5_same_root    (IB)
      pci0000:00/0000:00:01.0/0000:03:00.0   mlx5_far          (IB, far)
      pci0000:60/0000:60:02.0/0000:66:00.1   efa_0             (non-IB)
    """
    devices = tmp_path / "devices"
    gpu = _make_pci_leaf(devices, "pci0000:60", "0000:60:01.0", "0000:65:00.0")
    near = _make_pci_leaf(devices, "pci0000:60", "0000:60:01.0", "0000:67:00.0")
    mid = _make_pci_leaf(devices, "pci0000:60", "0000:60:02.0", "0000:66:00.0")
    far = _make_pci_leaf(devices, "pci0000:00", "0000:00:01.0", "0000:03:00.0")
    efa = _make_pci_leaf(devices, "pci0000:60", "0000:60:02.0", "0000:66:00.1")

    pci_devices = tmp_path / "pci_devices"
    pci_devices.mkdir()
    _make_gpu(pci_devices, "0000:65:00.0", gpu)

    ib = tmp_path / "infiniband"
    _make_ib_device(ib, "mlx5_same_bridge", near, "InfiniBand", "4: ACTIVE")
    _make_ib_device(ib, "mlx5_same_root", mid, "InfiniBand", "4: ACTIVE")
    _make_ib_device(ib, "mlx5_far", far, "InfiniBand", "4: ACTIVE")
    _make_ib_device(ib, "efa_0", efa, "Ethernet", "4: ACTIVE")
    return pci_devices, ib


def test_local_ib_device_picks_pcie_closest(
    fake_sysfs: tuple[Path, Path],
) -> None:
    pci_devices, ib = fake_sysfs
    got = local_ib_device(
        "0000:65:00.0", pci_devices_root=pci_devices, ib_class_root=ib
    )
    # Shares the GPU's own bridge (0000:60:01.0) — the closest of the three.
    assert got == "mlx5_same_bridge:1"


def _remove_ib_device(ib: Path, name: str) -> None:
    for child in sorted((ib / name).rglob("*"), reverse=True):
        child.unlink() if child.is_symlink() or child.is_file() else child.rmdir()
    (ib / name).rmdir()


def test_local_ib_device_skips_non_infiniband(
    fake_sysfs: tuple[Path, Path], tmp_path: Path
) -> None:
    # Without mlx5_same_bridge, efa_0 (Ethernet) ties mlx5_same_root on PCIe
    # distance and sorts first; the Ethernet device must never be chosen.
    pci_devices, ib = fake_sysfs
    _remove_ib_device(ib, "mlx5_same_bridge")
    got = local_ib_device(
        "0000:65:00.0", pci_devices_root=pci_devices, ib_class_root=ib
    )
    assert got == "mlx5_same_root:1"


def test_local_ib_device_skips_other_root_complex(
    fake_sysfs: tuple[Path, Path], tmp_path: Path
) -> None:
    # Only mlx5_far (under pci0000:00, not the GPU's pci0000:60) is left. HGX
    # B200 NVLink-management ConnectX-7 ports look like this: active IB, but
    # GPUDirect into the GPU through them fails, so they must not be pinned.
    pci_devices, ib = fake_sysfs
    _remove_ib_device(ib, "mlx5_same_bridge")
    _remove_ib_device(ib, "mlx5_same_root")
    got = local_ib_device(
        "0000:65:00.0", pci_devices_root=pci_devices, ib_class_root=ib
    )
    assert got is None


def test_local_ib_device_none_when_no_ib(tmp_path: Path) -> None:
    pci_devices = tmp_path / "pci_devices"
    ib = tmp_path / "infiniband"
    pci_devices.mkdir()
    ib.mkdir()
    assert (
        local_ib_device(
            "0000:65:00.0", pci_devices_root=pci_devices, ib_class_root=ib
        )
        is None
    )


def _make_verbs_namespace(
    tmp_path: Path, exposed: list[str]
) -> tuple[Path, Path]:
    """Fake /dev/infiniband + /sys/class/infiniband_verbs exposing `exposed`."""
    dev = tmp_path / "dev_infiniband"
    cls = tmp_path / "infiniband_verbs"
    dev.mkdir()
    cls.mkdir()
    for i, name in enumerate(exposed):
        (dev / f"uverbs{i}").write_text("")  # char-node stand-in
        (cls / f"uverbs{i}").mkdir()
        (cls / f"uverbs{i}" / "ibdev").write_text(name + "\n")
    return dev, cls


def test_local_ib_device_restricts_to_accessible(
    fake_sysfs: tuple[Path, Path], tmp_path: Path
) -> None:
    # sysfs lists all NICs, but the namespace exposes only mlx5_same_root (not
    # the PCIe-closest mlx5_same_bridge). The exposed one must win — pinning a
    # non-exposed device leaves UCX with no usable NIC (the Hydra-pod bug).
    pci_devices, ib = fake_sysfs
    vdev, vcls = _make_verbs_namespace(tmp_path, ["mlx5_same_root"])
    got = local_ib_device(
        "0000:65:00.0",
        pci_devices_root=pci_devices,
        ib_class_root=ib,
        verbs_dev_root=vdev,
        verbs_class_root=vcls,
    )
    assert got == "mlx5_same_root:1"


def test_local_ib_device_no_filter_when_mapping_absent(
    fake_sysfs: tuple[Path, Path], tmp_path: Path
) -> None:
    # No /dev/infiniband mapping (empty accessible set) => don't filter; fall
    # back to the PCIe-closest device.
    pci_devices, ib = fake_sysfs
    empty = tmp_path / "empty_dev"
    empty.mkdir()
    got = local_ib_device(
        "0000:65:00.0",
        pci_devices_root=pci_devices,
        ib_class_root=ib,
        verbs_dev_root=empty,
        verbs_class_root=tmp_path / "nonexistent",
    )
    assert got == "mlx5_same_bridge:1"


def test_configure_ucx_env_does_not_override_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCX_TLS", "operator-tls")
    monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_7:1")
    # api != "cuda" short-circuits the (GPU-only) NIC probe; the early return
    # on a preset UCX_NET_DEVICES means no /sys access happens here anyway.
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert os.environ["UCX_TLS"] == "operator-tls"
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_7:1"


def test_configure_ucx_env_sets_tls_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UCX_TLS", raising=False)
    # Preset UCX_NET_DEVICES so the NIC probe is skipped (no GPU in this test).
    monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert "rc" in os.environ["UCX_TLS"].split(",")


@pytest.mark.parametrize("api", ["hip", "cpu"])
def test_configure_ucx_env_skips_non_cuda(
    api: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The defaults name CUDA/IB transports; on ROCm (or a host) they must not
    # be applied — doing so left AMD UCX with no usable transport.
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    configure_ucx_env(_FakeDevice(api=api, id=0))
    assert "UCX_TLS" not in os.environ
    assert "UCX_NET_DEVICES" not in os.environ


def test_configure_ucx_env_skips_without_verbs_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # sysfs may list an IB device whose /dev/infiniband char node is absent
    # (e.g. a test sandbox); pinning it would strip UCX's tcp fallback and leave
    # no transport. With no uverbs node, leave the env untouched.
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.setattr(
        _ucx_env, "_VERBS_DEV_ROOT", tmp_path
    )  # empty: no uverbs
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert "UCX_TLS" not in os.environ
    assert "UCX_NET_DEVICES" not in os.environ


@pytest.mark.parametrize(
    ("flavor", "pinned"), [("cuda", False), ("cuda-verbs", True)]
)
def test_configure_ucx_env_gates_on_plugin_flavor(
    flavor: str, pinned: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # uverbs nodes exist either way; only the verbs flavor carries rc, so the
    # plain flavor must leave UCX its tcp fallback. configure_ucx_env writes
    # os.environ directly, so give it a copy that monkeypatch will restore.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    (tmp_path / "uverbs0").write_text("")
    monkeypatch.setattr(_ucx_env, "_VERBS_DEV_ROOT", tmp_path)
    monkeypatch.setattr(_ucx_env, "_gpu_pci_bus_id", lambda _: "0000:65:00.0")
    monkeypatch.setattr(_ucx_env, "local_ib_device", lambda _: "mlx5_0:1")
    monkeypatch.setenv("NIXL_PLUGIN_DIR", str(tmp_path / "nixl" / flavor))
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert ("UCX_NET_DEVICES" in os.environ) == pinned
    assert ("UCX_TLS" in os.environ) == pinned


def _patch_topology(
    monkeypatch: pytest.MonkeyPatch,
    pci_devices: Path,
    ib: Path,
    tmp_path: Path,
    bus_id: str = "0000:65:00.0",
) -> None:
    """Points configure_ucx_env's NIC probe at a fake sysfs and a usable GPU."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for var in ("UCX_TLS", "UCX_NET_DEVICES", "NIXL_PLUGIN_DIR"):
        monkeypatch.delenv(var, raising=False)
    # Empty verbs mapping: don't filter by exposure, whatever the host has.
    no_verbs = tmp_path / "no_verbs"
    monkeypatch.setattr(_ucx_env, "_verbs_usable", lambda: True)
    monkeypatch.setattr(_ucx_env, "_gpu_pci_bus_id", lambda _: bus_id)
    monkeypatch.setattr(
        _ucx_env,
        "_host_gpu_bus_ids",
        functools.partial(
            _ucx_env._host_gpu_bus_ids, pci_devices_root=pci_devices
        ),
    )
    monkeypatch.setattr(_ucx_env, "_ethernet_netdevs", lambda: ["eth0"])
    monkeypatch.setattr(
        _ucx_env,
        "local_ib_device",
        functools.partial(
            local_ib_device,
            pci_devices_root=pci_devices,
            ib_class_root=ib,
            verbs_dev_root=no_verbs,
            verbs_class_root=no_verbs,
        ),
    )
    monkeypatch.setattr(
        _ucx_env,
        "_ib_devices",
        functools.partial(
            _ucx_env._ib_devices,
            ib_class_root=ib,
            verbs_dev_root=no_verbs,
            verbs_class_root=no_verbs,
        ),
    )


def test_configure_ucx_env_avoids_ib_on_other_root_complex(
    monkeypatch: pytest.MonkeyPatch,
    fake_sysfs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    # The AWS B200 pod layout: both IB devices sit under a different root
    # complex than the GPU. Pinning one made UCX GPUDirect into the GPU over it
    # and fail intranode DI with NIXL_ERR_REMOTE_DISCONNECT; keep UCX off IB.
    pci_devices, ib = fake_sysfs
    _remove_ib_device(ib, "mlx5_same_bridge")
    _remove_ib_device(ib, "mlx5_same_root")
    _patch_topology(monkeypatch, pci_devices, ib, tmp_path)
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert os.environ["UCX_TLS"] == "^ib"
    assert "UCX_NET_DEVICES" not in os.environ


def test_configure_ucx_env_pins_same_root_complex_ib(
    monkeypatch: pytest.MonkeyPatch,
    fake_sysfs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    # Internode IB hosts (e.g. Hydra) have a NIC under the GPU's root complex;
    # that one is still pinned alongside the rc TLS default.
    pci_devices, ib = fake_sysfs
    _patch_topology(monkeypatch, pci_devices, ib, tmp_path)
    configure_ucx_env(_FakeDevice(api="cuda", id=0))
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_same_bridge:1"
    assert "rc" in os.environ["UCX_TLS"].split(",")
    assert "tcp" not in os.environ["UCX_TLS"].split(",")


def _add_gpu_without_local_ib(pci_devices: Path, tmp_path: Path) -> None:
    """Adds GPU 0000:85:00.0 under pci0000:80, a root complex with no IB."""
    leaf = _make_pci_leaf(
        tmp_path / "devices", "pci0000:80", "0000:80:01.0", "0000:85:00.0"
    )
    _make_gpu(pci_devices, "0000:85:00.0", leaf)


def test_host_gpu_bus_ids_skips_non_gpus(
    fake_sysfs: tuple[Path, Path], tmp_path: Path
) -> None:
    pci_devices, _ = fake_sysfs
    _add_gpu_without_local_ib(pci_devices, tmp_path)
    nvswitch = _make_pci_leaf(
        tmp_path / "devices", "pci0000:90", "0000:90:01.0", "0000:91:00.0"
    )
    (nvswitch / "vendor").write_text("0x10de\n")
    (nvswitch / "class").write_text("0x068000\n")
    os.symlink(nvswitch, pci_devices / "0000:91:00.0")
    assert _ucx_env._host_gpu_bus_ids(pci_devices_root=pci_devices) == [
        "0000:65:00.0",
        "0000:85:00.0",
    ]


def test_ethernet_netdevs_skips_loopback_ipoib_and_down(
    tmp_path: Path,
) -> None:
    for name, arp_type, state in [
        ("eth0", "1", "up"),
        ("v4if0", "1", "up"),
        ("lo", "772", "unknown"),
        ("ibp115s0f0", "32", "up"),
        ("eth1", "1", "down"),
    ]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "type").write_text(arp_type + "\n")
        (tmp_path / name / "operstate").write_text(state + "\n")
    assert _ucx_env._ethernet_netdevs(net_class_root=tmp_path) == [
        "eth0",
        "v4if0",
    ]


def _am_transports(env: dict[str, str]) -> set[str]:
    """Network transports a worker with this env can carry NIXL AM on.

    NIXL endpoints need peer-failure handling, so the shared-memory transports
    (posix, sysv, cma) never qualify, nor do the cuda ones (no am bcopy); only
    rc and tcp remain. Assumes the fake host's IB device is ``mlx5_*`` and its
    Ethernet interface is ``eth0``.
    """
    tls = env.get("UCX_TLS", "all")
    if tls == "all":
        enabled = {"rc", "tcp"}
    elif tls.startswith("^"):
        excluded = set(tls[1:].split(","))
        enabled = {"tcp"} if "ib" in excluded else {"rc", "tcp"}
    else:
        enabled = set(tls.split(",")) & {"rc", "tcp"}
    devices = env.get("UCX_NET_DEVICES", "all").split(",")
    usable = set()
    if "rc" in enabled and (
        "all" in devices or any(d.startswith("mlx5") for d in devices)
    ):
        usable.add("rc")
    if "tcp" in enabled and ("all" in devices or "eth0" in devices):
        usable.add("tcp")
    return usable


def test_configure_ucx_env_mixed_locality_workers_can_connect(
    monkeypatch: pytest.MonkeyPatch,
    fake_sysfs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    # B200 Coder pods have IB local to GPUs 4-7 only. A worker there pins the
    # NIC while one on GPU 0 runs UCX_TLS=^ib; without a shared AM-capable
    # transport, connect() fails with "no active messages transport".
    pci_devices, ib = fake_sysfs
    _add_gpu_without_local_ib(pci_devices, tmp_path)

    envs = {}
    for bus_id in ("0000:65:00.0", "0000:85:00.0"):
        _patch_topology(monkeypatch, pci_devices, ib, tmp_path, bus_id)
        configure_ucx_env(_FakeDevice(api="cuda", id=0))
        envs[bus_id] = {
            k: os.environ[k]
            for k in ("UCX_TLS", "UCX_NET_DEVICES")
            if k in os.environ
        }

    pinned, no_ib = envs["0000:65:00.0"], envs["0000:85:00.0"]
    assert pinned["UCX_NET_DEVICES"] == "mlx5_same_bridge:1,eth0"
    assert {"rc", "tcp"} <= set(pinned["UCX_TLS"].split(","))
    assert no_ib == {"UCX_TLS": "^ib"}
    assert _am_transports(pinned) & _am_transports(no_ib)


def test_am_transports_finds_no_overlap_without_tcp() -> None:
    # Guards the model against passing the mixed-locality test vacuously.
    pinned = {
        "UCX_TLS": default_ucx_tls(gdr_copy=False),
        "UCX_NET_DEVICES": "mlx5_0:1",
    }
    assert not _am_transports(pinned) & _am_transports({"UCX_TLS": "^ib"})
