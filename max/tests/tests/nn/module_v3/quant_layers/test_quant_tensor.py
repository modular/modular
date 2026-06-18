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

"""CPU-only checks for the quantized tensor wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, Sharded
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    QTensor,
)


def test_init_stores_tensors_and_block_size() -> None:
    """``__init__`` keeps the passed tensors and block size verbatim."""
    with F.lazy():
        data = Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn)
        scale_inv = Tensor.zeros((1, 1), dtype=DType.float32)
        qt = FP8BlockTensor(
            data=data, scale_inv=scale_inv, block_size=(128, 128)
        )

        assert qt.data is data
        assert qt.scale_inv is scale_inv
        assert qt.block_size == (128, 128)


def test_init_default_block_size() -> None:
    """The default block size is ``(128, 128)``."""
    with F.lazy():
        qt = FP8BlockTensor(
            data=Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn),
            scale_inv=Tensor.zeros((1, 1), dtype=DType.float32),
        )
        assert qt.block_size == (128, 128)


def test_zeros_dtypes_are_kernel_fixed() -> None:
    """``zeros()`` pins the FP8 data and float32 scale dtypes."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert qt.data.dtype == DType.float8_e4m3fn
        assert qt.scale_inv.dtype == DType.float32


def test_zeros_data_shape_matches_request() -> None:
    """``zeros()`` data buffer has exactly the requested shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert list(qt.data.shape) == [256, 512]


def test_zeros_scale_shape_divisible() -> None:
    """Scale shape is ``shape / block_size`` when dims divide evenly."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(128, 128))
        # 256 / 128 == 2, 512 / 128 == 4
        assert list(qt.scale_inv.shape) == [2, 4]
        assert qt.block_size == (128, 128)


def test_zeros_scale_shape_uses_ceildiv() -> None:
    """Non-divisible dims round up: ``ceil(rows / m)``, ``ceil(cols / k)``."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((130, 100), block_size=(128, 128))
        # ceil(130 / 128) == 2, ceil(100 / 128) == 1
        assert list(qt.scale_inv.shape) == [2, 1]


def test_zeros_custom_block_size() -> None:
    """A non-default block size flows into both block_size and scale shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(64, 256))
        assert qt.block_size == (64, 256)
        # 256 / 64 == 4, 512 / 256 == 2
        assert list(qt.scale_inv.shape) == [4, 2]


def test_zeros_single_block() -> None:
    """A shape smaller than one block still yields a 1x1 scale grid."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((16, 32), block_size=(128, 128))
        assert list(qt.data.shape) == [16, 32]
        assert list(qt.scale_inv.shape) == [1, 1]


def test_is_module_subclass() -> None:
    """QTensor / FP8BlockTensor are Modules for parameter discovery."""
    assert issubclass(QTensor, Module)
    assert issubclass(FP8BlockTensor, QTensor)


def test_local_parameters_discovered() -> None:
    """``data`` and ``scale_inv`` are discoverable as local parameters."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        local = dict(qt.local_parameters)

        assert set(local) == {"data", "scale_inv"}
        assert local["data"] is qt.data
        assert local["scale_inv"] is qt.scale_inv


def test_block_size_is_not_a_parameter() -> None:
    """``block_size`` is metadata, not a discovered tensor parameter."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        names = {name for name, _ in qt.parameters}
        assert names == {"data", "scale_inv"}


def test_parameters_discovered_through_parent_module() -> None:
    """Nested in a Module, the inner tensors get qualified ``weight.*`` names."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.weight = FP8BlockTensor.zeros((256, 512))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        names = {name for name, _ in wrapper.parameters}
        assert names == {"weight.data", "weight.scale_inv"}


def test_forward_raises() -> None:
    """QTensors are data wrappers, not callable layers."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt()


def test_forward_method_raises_directly() -> None:
    """Calling ``forward()`` directly raises the same guard."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt.forward()


# --------------------------------------------------------------------------- #
# FP8BlockTensor.shard() -- Slice 3
# --------------------------------------------------------------------------- #


def _fake_gpu(id: int = 0) -> Device:
    """Minimal mock GPU device matching the conftest helper."""

    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


def test_fp8_shard_row_axis_data_shape() -> None:
    """Sharding on axis 0 (row axis) halves the data row extent per device.

    Stacked weight ``[E, 2*moe_dim, hidden]`` sharded on axis 1 (the moe_dim
    row axis, which is axis 1 of the 3-D stacked tensor).  Here we test a
    simpler 2-D weight, axis 0.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        # 2-D FP8 weight [256, 256] with (128, 128) blocks.
        qt = FP8BlockTensor.zeros((256, 256), block_size=(128, 128))
        sharded = qt.shard(axis=0, mesh=mesh)

        # data: [256, 256] → each device [128, 256]
        assert sharded.data.placements == (Sharded(axis=0),)
        for shard in sharded.data.local_shards:
            assert list(shard.shape) == [256 // num_devices, 256]

        # scale_inv: [2, 2] → each device [1, 2]
        assert sharded.scale_inv.placements == (Sharded(axis=0),)
        for shard in sharded.scale_inv.local_shards:
            assert list(shard.shape) == [2 // num_devices, 2]

        # block_size preserved
        assert sharded.block_size == (128, 128)


def test_fp8_shard_col_axis_data_shape() -> None:
    """Sharding on axis -1 (col axis) halves the data col extent per device.

    This mirrors the TP ``down_proj`` shard: ``[E, hidden, moe_dim]`` sharded
    on axis -1.  We test a 2-D proxy ``[256, 256]``.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        qt = FP8BlockTensor.zeros((256, 256), block_size=(128, 128))
        sharded = qt.shard(axis=-1, mesh=mesh)

        # data last dim: 256 → 128 per device
        for shard in sharded.data.local_shards:
            assert list(shard.shape) == [256, 256 // num_devices]

        # scale_inv last dim: 2 → 1 per device
        for shard in sharded.scale_inv.local_shards:
            assert list(shard.shape) == [2, 2 // num_devices]


def test_fp8_shard_3d_down_proj_shape() -> None:
    """Sharding a 3-D ``[E, H, moe_dim]`` weight on axis -1 (TP down_proj).

    Uses 2 devices and (128, 128) blocks.  ``hidden=256``, ``moe_dim=512``.
    After shard on axis=-1 each device should hold ``[E, H, moe_dim//2]``.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        num_experts = 4
        hidden = 256
        moe_dim = 512

        # FP8BlockTensor.zeros only accepts 2-D shapes; build directly.
        data = Tensor.zeros(
            (num_experts, hidden, moe_dim), dtype=DType.float8_e4m3fn
        )
        # scale_inv shape: [E, ceil(H/128), ceil(moe_dim/128)] = [4, 2, 4]
        scale_inv = Tensor.zeros(
            (num_experts, hidden // 128, moe_dim // 128), dtype=DType.float32
        )
        qt = FP8BlockTensor(
            data=data, scale_inv=scale_inv, block_size=(128, 128)
        )
        sharded = qt.shard(axis=-1, mesh=mesh)

        for shard in sharded.local_shards:
            assert list(shard.data.shape) == [
                num_experts,
                hidden,
                moe_dim // num_devices,
            ]
            assert list(shard.scale_inv.shape) == [
                num_experts,
                hidden // 128,
                moe_dim // 128 // num_devices,
            ]
