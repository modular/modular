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
    os.symlink(gpu, pci_devices / "0000:65:00.0")

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
) -> None:
    """Points configure_ucx_env's NIC probe at a fake sysfs and a usable GPU."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for var in ("UCX_TLS", "UCX_NET_DEVICES", "NIXL_PLUGIN_DIR"):
        monkeypatch.delenv(var, raising=False)
    # Empty verbs mapping: don't filter by exposure, whatever the host has.
    no_verbs = tmp_path / "no_verbs"
    monkeypatch.setattr(_ucx_env, "_verbs_usable", lambda: True)
    monkeypatch.setattr(_ucx_env, "_gpu_pci_bus_id", lambda _: "0000:65:00.0")
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
