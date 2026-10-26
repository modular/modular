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
from max import tree
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
    MXFP4Quantizer,
    MXFP4Tensor,
    MXFP8Quantizer,
    MXFP8Tensor,
    NVFP4Activation,
    NVFP4Tensor,
    NVFP4TensorQuantizer,
    QTensor,
    all_fp8_block,
    all_mxfp4,
    all_mxfp8,
    all_nvfp4,
)
from max.experimental.sharding import DeviceMesh, Sharded
from max.experimental.tensor import Tensor


def test_init_stores_tensors_and_block_size() -> None:
    """``__init__`` keeps the passed tensors and block size verbatim."""
    with F.lazy():
        data = Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn)
        weight_scale_inv = Tensor.zeros((1, 1), dtype=DType.float32)
        qt = FP8BlockTensor(
            weight=data,
            weight_scale_inv=weight_scale_inv,
            block_size=(128, 128),
        )

        assert qt.weight is data
        assert qt.weight_scale_inv is weight_scale_inv
        assert qt.block_size == (128, 128)


def test_init_default_block_size() -> None:
    """The default block size is ``(128, 128)``."""
    with F.lazy():
        qt = FP8BlockTensor(
            weight=Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn),
            weight_scale_inv=Tensor.zeros((1, 1), dtype=DType.float32),
        )
        assert qt.block_size == (128, 128)


def test_zeros_dtypes_are_kernel_fixed() -> None:
    """``zeros()`` pins the FP8 data and float32 scale dtypes."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert qt.weight.dtype == DType.float8_e4m3fn
        assert qt.weight_scale_inv.dtype == DType.float32


def test_zeros_data_shape_matches_request() -> None:
    """``zeros()`` data buffer has exactly the requested shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert list(qt.weight.shape) == [256, 512]


def test_zeros_scale_shape_divisible() -> None:
    """Scale shape is ``shape / block_size`` when dims divide evenly."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(128, 128))
        # 256 / 128 == 2, 512 / 128 == 4
        assert list(qt.weight_scale_inv.shape) == [2, 4]
        assert qt.block_size == (128, 128)


def test_zeros_scale_shape_uses_ceildiv() -> None:
    """Non-divisible dims round up: ``ceil(rows / m)``, ``ceil(cols / k)``."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((130, 100), block_size=(128, 128))
        # ceil(130 / 128) == 2, ceil(100 / 128) == 1
        assert list(qt.weight_scale_inv.shape) == [2, 1]


def test_zeros_custom_block_size() -> None:
    """A non-default block size flows into both block_size and scale shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(64, 256))
        assert qt.block_size == (64, 256)
        # 256 / 64 == 4, 512 / 256 == 2
        assert list(qt.weight_scale_inv.shape) == [4, 2]


def test_zeros_single_block() -> None:
    """A shape smaller than one block still yields a 1x1 scale grid."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((16, 32), block_size=(128, 128))
        assert list(qt.weight.shape) == [16, 32]
        assert list(qt.weight_scale_inv.shape) == [1, 1]


def test_is_module_subclass() -> None:
    """QTensor / FP8BlockTensor are Modules for parameter discovery."""
    assert issubclass(QTensor, Module)
    assert issubclass(FP8BlockTensor, QTensor)


def test_local_parameters_discovered() -> None:
    """The leaves are local parameters, keyed as the checkpoint names them."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        local = dict(qt.local_parameters)

        assert set(local) == {"weight", "weight_scale_inv"}
        assert local["weight"] is qt.weight
        assert local["weight_scale_inv"] is qt.weight_scale_inv


def test_block_size_is_not_a_parameter() -> None:
    """``block_size`` is metadata, not a discovered tensor parameter."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        names = {name for name, _ in qt.parameters}
        assert names == {"weight", "weight_scale_inv"}


def test_parameters_discovered_through_parent_module() -> None:
    """Held raw as ``proj``, the leaves take the checkpoint's ``proj.*`` names."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.proj = FP8BlockTensor.zeros((256, 512))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        names = {name for name, _ in wrapper.parameters}
        assert names == {"proj.weight", "proj.weight_scale_inv"}


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
        assert sharded.weight.placements == (Sharded(axis=0),)
        for shard in sharded.weight.local_shards:
            assert list(shard.shape) == [256 // num_devices, 256]

        # weight_scale_inv: [2, 2] → each device [1, 2]
        assert sharded.weight_scale_inv.placements == (Sharded(axis=0),)
        for shard in sharded.weight_scale_inv.local_shards:
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
        for shard in sharded.weight.local_shards:
            assert list(shard.shape) == [256, 256 // num_devices]

        # weight_scale_inv last dim: 2 → 1 per device
        for shard in sharded.weight_scale_inv.local_shards:
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
        # weight_scale_inv shape: [E, ceil(H/128), ceil(moe_dim/128)] = [4, 2, 4]
        weight_scale_inv = Tensor.zeros(
            (num_experts, hidden // 128, moe_dim // 128), dtype=DType.float32
        )
        qt = FP8BlockTensor(
            weight=data,
            weight_scale_inv=weight_scale_inv,
            block_size=(128, 128),
        )
        sharded = qt.shard(axis=-1, mesh=mesh)

        for shard in sharded.local_shards:
            assert list(shard.weight.shape) == [
                num_experts,
                hidden,
                moe_dim // num_devices,
            ]
            assert list(shard.weight_scale_inv.shape) == [
                num_experts,
                hidden // 128,
                moe_dim // 128 // num_devices,
            ]


# --------------------------------------------------------------------------- #
# NVFP4Tensor
# --------------------------------------------------------------------------- #


def test_nvfp4_zeros_dtypes_are_kernel_fixed() -> None:
    """``zeros()`` pins the packed-uint8 data and the two scale dtypes."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert qt.weight.dtype == DType.uint8
        assert qt.weight_scale.dtype == DType.float8_e4m3fn
        assert qt.weight_scale_2.dtype == DType.float32
        assert qt.input_scale.dtype == DType.float32


def test_nvfp4_zeros_packs_two_values_per_byte() -> None:
    """``zeros()`` takes the logical shape; ``data`` halves the K extent."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert list(qt.weight.shape) == [256, 256]


def test_nvfp4_zeros_scale_grid_uses_logical_columns() -> None:
    """The block-scale grid is derived from the *unpacked* column count."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        # 256 / 1 == 256 rows, 512 / 16 == 32 blocks along K.
        assert list(qt.weight_scale.shape) == [256, 32]
        assert qt.block_size == (1, 16)


def test_nvfp4_zeros_scale_shape_uses_ceildiv() -> None:
    """A partial trailing block still gets a scale."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((8, 24))
        # ceil(24 / 16) == 2
        assert list(qt.weight_scale.shape) == [8, 2]


def test_nvfp4_zeros_global_scales_are_host_scalars() -> None:
    """``weight_scale_2`` / ``input_scale`` are rank-0 and stay on the host."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert list(qt.weight_scale_2.shape) == []
        assert list(qt.input_scale.shape) == []
        assert qt.weight_scale_2.device.is_host
        assert qt.input_scale.device.is_host


def test_nvfp4_init_stores_tensors_and_block_size() -> None:
    """``__init__`` keeps the passed tensors and block size verbatim."""
    with F.lazy():
        data = Tensor.zeros((4, 8), dtype=DType.uint8)
        weight_scale = Tensor.zeros((4, 1), dtype=DType.float8_e4m3fn)
        weight_scale_2 = Tensor.zeros((), dtype=DType.float32)
        input_scale = Tensor.zeros((), dtype=DType.float32)
        qt = NVFP4Tensor(
            weight=data,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
            input_scale=input_scale,
            block_size=(1, 16),
        )

        assert qt.weight is data
        assert qt.weight_scale is weight_scale
        assert qt.weight_scale_2 is weight_scale_2
        assert qt.input_scale is input_scale
        assert qt.block_size == (1, 16)


def test_nvfp4_local_parameters_discovered() -> None:
    """All four leaves are discoverable as local parameters."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        local = dict(qt.local_parameters)

        assert set(local) == {
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale",
        }
        assert local["weight"] is qt.weight


def test_nvfp4_parameters_discovered_through_parent_module() -> None:
    """Held raw as ``proj``, the leaves take a modelopt checkpoint's names."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.proj = NVFP4Tensor.zeros((256, 512))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        names = {name for name, _ in wrapper.parameters}
        assert names == {
            "proj.weight",
            "proj.weight_scale",
            "proj.weight_scale_2",
            "proj.input_scale",
        }


def test_nvfp4_is_qtensor_subclass() -> None:
    assert issubclass(NVFP4Tensor, QTensor)


def test_nvfp4_forward_raises() -> None:
    """NVFP4Tensors are data wrappers, not callable layers."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt()


def test_nvfp4_shard_row_axis() -> None:
    """Sharding on axis 0 co-shards data and the block-scale grid."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        qt = NVFP4Tensor.zeros((256, 512))
        sharded = qt.shard(axis=0, mesh=mesh)

        assert sharded.weight.placements == (Sharded(axis=0),)
        assert sharded.weight_scale.placements == (Sharded(axis=0),)
        for shard in sharded.local_shards:
            assert list(shard.weight.shape) == [256 // num_devices, 256]
            assert list(shard.weight_scale.shape) == [256 // num_devices, 32]
        assert sharded.block_size == (1, 16)


def test_nvfp4_shard_keeps_global_scales_whole() -> None:
    """The per-tensor scales are not sharded -- every shard sees them all."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        mesh = DeviceMesh((mock(0), mock(1)), (2,), (TP,))

        qt = NVFP4Tensor.zeros((256, 512))
        sharded = qt.shard(axis=0, mesh=mesh)

        assert sharded.weight_scale_2 is qt.weight_scale_2
        assert sharded.input_scale is qt.input_scale
        for shard in sharded.local_shards:
            assert shard.weight_scale_2 is qt.weight_scale_2
            assert shard.input_scale is qt.input_scale


def test_nvfp4_shard_3d_down_proj_shape() -> None:
    """A stacked ``[E, N, K/2]`` down-proj weight shards on the packed K axis."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        num_experts = 4
        hidden = 256
        moe_dim = 512

        qt = NVFP4Tensor(
            weight=Tensor.zeros(
                (num_experts, hidden, moe_dim // 2), dtype=DType.uint8
            ),
            weight_scale=Tensor.zeros(
                (num_experts, hidden, moe_dim // 16),
                dtype=DType.float8_e4m3fn,
            ),
            weight_scale_2=Tensor.zeros((num_experts,), dtype=DType.float32),
            input_scale=Tensor.zeros((num_experts,), dtype=DType.float32),
        )
        sharded = qt.shard(axis=-1, mesh=mesh)

        for shard in sharded.local_shards:
            assert list(shard.weight.shape) == [
                num_experts,
                hidden,
                moe_dim // 2 // num_devices,
            ]
            assert list(shard.weight_scale.shape) == [
                num_experts,
                hidden,
                moe_dim // 16 // num_devices,
            ]


def test_nvfp4_to_device_moves_only_the_block_leaves() -> None:
    """``to()`` transfers data/weight_scale; the global scales stay on host."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        device = mock(0)

        qt = NVFP4Tensor.zeros((256, 512))
        moved = qt.to(device)

        assert moved.weight.device == device
        assert moved.weight_scale.device == device
        assert moved.weight_scale_2 is qt.weight_scale_2
        assert moved.input_scale is qt.input_scale
        assert moved.block_size == qt.block_size


def test_qtensor_base_rejects_sharding_helpers() -> None:
    """The base class refuses the placement helpers subclasses must implement."""
    with F.lazy():
        qt = QTensor()
        with pytest.raises(NotImplementedError, match="local_shards"):
            _ = qt.local_shards
        with pytest.raises(NotImplementedError, match="to\\(\\)"):
            qt.to(CPU())


# --------------------------------------------------------------------------- #
# Homogeneity type guards
# --------------------------------------------------------------------------- #


def test_nvfp4_quantizer_stores_the_raw_tensor() -> None:
    """The quantizer keeps the bf16 tensor and the block size it was given."""
    with F.lazy():
        raw = Tensor.zeros((8, 32), dtype=DType.bfloat16)
        qt = NVFP4TensorQuantizer(raw, block_size=(1, 16))

        assert qt._tensor is raw
        assert qt.block_size == (1, 16)
        assert list(qt.shape) == [8, 32]


def test_nvfp4_quantizer_rejects_k_not_multiple_of_block() -> None:
    """K must divide the NVFP4 block so the packed layout is exact."""
    with F.lazy():
        raw = Tensor.zeros((4, 15), dtype=DType.bfloat16)
        with pytest.raises(ValueError, match="multiple of 16"):
            NVFP4TensorQuantizer(raw)


def test_nvfp4_quantizer_parameter_path_is_the_raw_weight() -> None:
    """The bf16 tensor loads as the checkpoint's ``proj.weight``."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.proj = NVFP4TensorQuantizer(
                Tensor.zeros((8, 32), dtype=DType.bfloat16)
            )

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        parameters = dict(wrapper.parameters)

        assert set(parameters) == {"proj.weight"}
        assert parameters["proj.weight"] is wrapper.proj._tensor


def test_nvfp4_quantizer_tree_roundtrip_keeps_block_size() -> None:
    """Rebuilding the module tree restores the block size, not only the tensor."""
    with F.lazy():
        raw = Tensor.zeros((8, 32), dtype=DType.bfloat16)
        qt = NVFP4TensorQuantizer(raw, block_size=(1, 16))
        leaves, treedef = tree.flatten(qt)
        rebuilt = tree.unflatten(treedef, leaves)

        assert isinstance(rebuilt, NVFP4TensorQuantizer)
        assert rebuilt.block_size == (1, 16)
        assert rebuilt._tensor is raw


def test_nvfp4_quantizer_to_moves_the_raw_tensor() -> None:
    """``to()`` places the wrapped tensor and leaves the block size alone."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        device = mock(0)
        raw = Tensor.zeros((8, 32), dtype=DType.bfloat16)
        qt = NVFP4TensorQuantizer(raw)

        moved = qt.to(device)

        assert moved is qt
        assert qt._tensor.device == device
        assert qt.block_size == (1, 16)


def test_quantizer_to_drops_the_cached_quantization() -> None:
    """Moving a quantizer rebuilds its scales where the weight now lives.

    The scales are a function of ``weight``, so ``to()`` moves the one tensor
    it holds and drops the cached container rather than moving a stale copy.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        device = mock(0)
        qt = MXFP4Quantizer(Tensor.zeros((8, 32), dtype=DType.bfloat16))
        before = qt.quant_tensor

        qt.to(device)

        assert "quant_tensor" not in vars(qt)
        assert qt.quant_tensor is not before


def test_nvfp4_quantizer_packs_nvfp4_on_first_use(sm100_arch: None) -> None:
    """First use quantizes the bf16 tensor into packed NVFP4 leaves."""
    del sm100_arch
    with F.lazy():
        # 128 and 64 match the SM100 scale-atom grid, so the packed scales
        # reshape onto ``[rows, K / 16]``.
        qt = NVFP4TensorQuantizer(Tensor.zeros((128, 64), dtype=DType.bfloat16))

        assert qt.weight.dtype == DType.uint8
        assert list(qt.weight.shape) == [128, 32]
        assert qt.weight_scale.dtype == DType.float8_e4m3fn
        assert list(qt.weight_scale.shape) == [128, 4]
        assert qt.weight_scale_2.dtype == DType.float32
        assert list(qt.weight_scale_2.shape) == []
        assert qt.input_scale.dtype == DType.float32
        assert list(qt.input_scale.shape) == []


def test_type_guards_narrow_homogeneous_bundles() -> None:
    """``all_nvfp4`` / ``all_fp8_block`` only accept a uniform bundle."""
    with F.lazy():
        nvfp4 = NVFP4Tensor.zeros((256, 512))
        fp8 = FP8BlockTensor.zeros((256, 512))
        plain = Tensor.zeros((256, 512), dtype=DType.bfloat16)

        quantizer = NVFP4TensorQuantizer(
            Tensor.zeros((8, 32), dtype=DType.bfloat16)
        )

        assert all_nvfp4([nvfp4, nvfp4])
        assert all_nvfp4([nvfp4, quantizer])
        assert not all_nvfp4([nvfp4, fp8])
        assert not all_nvfp4([nvfp4, plain])
        assert all_fp8_block([fp8, fp8])
        assert not all_fp8_block([fp8, nvfp4])


# --------------------------------------------------------------------------- #
# NVFP4Activation
# --------------------------------------------------------------------------- #


def test_nvfp4_activation_leaves() -> None:
    """The dispatched-token wrapper exposes its four leaves as parameters."""
    with F.lazy():
        act = NVFP4Activation(
            data=Tensor.zeros((8, 128), dtype=DType.uint8),
            scales=Tensor.zeros((1, 1, 32, 4, 4), dtype=DType.float8_e4m3fn),
            input_scale=Tensor.zeros((), dtype=DType.float32),
            scales_offset=Tensor.zeros((4,), dtype=DType.uint32),
        )

        assert {name for name, _ in act.parameters} == {
            "data",
            "scales",
            "input_scale",
            "scales_offset",
        }
        assert list(act.data.shape) == [8, 128]


# --------------------------------------------------------------------------- #
# MXFP8Tensor
# --------------------------------------------------------------------------- #


def test_mxfp8_zeros_dtypes_and_scale_grid() -> None:
    """One E4M3 byte per value and one E8M0 scale per 32 values along K."""
    with F.lazy():
        qt = MXFP8Tensor.zeros((256, 512))
        assert qt.weight.dtype == DType.float8_e4m3fn
        assert qt.weight_scale.dtype == DType.float8_e8m0fnu
        assert list(qt.weight.shape) == [256, 512]
        assert list(qt.weight_scale.shape) == [256, 16]
        assert qt.block_size == (1, 32)


def test_mxfp8_zeros_scale_shape_uses_ceildiv() -> None:
    """A partial trailing block still gets a scale."""
    with F.lazy():
        qt = MXFP8Tensor.zeros((8, 40))
        assert list(qt.weight_scale.shape) == [8, 2]


def test_mxfp8_local_parameters_discovered() -> None:
    """Both leaves are discoverable, under the checkpoint's leaf names."""
    with F.lazy():
        qt = MXFP8Tensor.zeros((256, 512))
        assert set(dict(qt.local_parameters)) == {"weight", "weight_scale"}


@pytest.mark.parametrize(
    ("axis", "data_shape", "scale_shape"),
    [(0, [128, 512], [128, 16]), (-1, [256, 256], [256, 8])],
)
def test_mxfp8_shard_co_shards_scales(
    axis: int, data_shape: list[int], scale_shape: list[int]
) -> None:
    """Sharding places the scales with their data rows or K blocks."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        sharded = MXFP8Tensor.zeros((256, 512)).shard(axis=axis, mesh=mesh)

        assert isinstance(sharded, MXFP8Tensor)
        assert sharded.weight.placements == (Sharded(axis=axis),)
        assert sharded.weight_scale.placements == (Sharded(axis=axis),)
        for shard in sharded.local_shards:
            assert list(shard.weight.shape) == data_shape
            assert list(shard.weight_scale.shape) == scale_shape


def test_all_mxfp8_narrows_only_homogeneous_sequences() -> None:
    with F.lazy():
        mx = MXFP8Tensor.zeros((256, 512))
        assert all_mxfp8([mx, mx])
        assert not all_mxfp8([mx, NVFP4Tensor.zeros((256, 512))])


# --------------------------------------------------------------------------- #
# MXFP4Tensor
# --------------------------------------------------------------------------- #


def test_mxfp4_zeros_dtypes_and_scale_grid() -> None:
    """Two e2m1 codes per byte and one E8M0 scale per 32 values along K."""
    with F.lazy():
        qt = MXFP4Tensor.zeros((256, 512))
        assert qt.weight.dtype == DType.uint8
        assert qt.weight_scale.dtype == DType.float8_e8m0fnu
        assert list(qt.weight.shape) == [256, 256]
        assert list(qt.weight_scale.shape) == [256, 16]
        assert qt.block_size == (1, 32)


def test_mxfp4_zeros_scale_shape_uses_ceildiv() -> None:
    """A partial trailing block still gets a scale."""
    with F.lazy():
        qt = MXFP4Tensor.zeros((8, 40))
        assert list(qt.weight.shape) == [8, 20]
        assert list(qt.weight_scale.shape) == [8, 2]


def test_mxfp4_local_parameters_discovered() -> None:
    """Both leaves are discoverable, under the checkpoint's leaf names."""
    with F.lazy():
        qt = MXFP4Tensor.zeros((256, 512))
        assert set(dict(qt.local_parameters)) == {"weight", "weight_scale"}


@pytest.mark.parametrize(
    ("axis", "data_shape", "scale_shape"),
    [(0, [128, 256], [128, 16]), (-1, [256, 128], [256, 8])],
)
def test_mxfp4_shard_co_shards_scales(
    axis: int, data_shape: list[int], scale_shape: list[int]
) -> None:
    """Sharding places the scales with their packed rows or K blocks."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        sharded = MXFP4Tensor.zeros((256, 512)).shard(axis=axis, mesh=mesh)

        assert isinstance(sharded, MXFP4Tensor)
        assert sharded.weight.placements == (Sharded(axis=axis),)
        assert sharded.weight_scale.placements == (Sharded(axis=axis),)
        for shard in sharded.local_shards:
            assert list(shard.weight.shape) == data_shape
            assert list(shard.weight_scale.shape) == scale_shape


def test_all_mxfp4_narrows_only_homogeneous_sequences() -> None:
    """The two MX guards reject each other's tensors, not just NVFP4's."""
    with F.lazy():
        mx4 = MXFP4Tensor.zeros((256, 512))
        mx8 = MXFP8Tensor.zeros((256, 512))
        assert all_mxfp4([mx4, mx4])
        assert not all_mxfp4([mx4, mx8])
        assert not all_mxfp4([mx4, NVFP4Tensor.zeros((256, 512))])
        assert not all_mxfp8([mx8, mx4])


# --------------------------------------------------------------------------- #
# MX containers, both formats
# --------------------------------------------------------------------------- #

_MX_CONTAINERS = pytest.mark.parametrize(
    ("cls", "storage_dtype", "values_per_byte"),
    [
        (MXFP8Tensor, DType.float8_e4m3fn, 1),
        (MXFP4Tensor, DType.uint8, 2),
    ],
)


@_MX_CONTAINERS
def test_mx_storage_is_named_on_the_class(
    cls: type[MXFP8Tensor | MXFP4Tensor],
    storage_dtype: DType,
    values_per_byte: int,
) -> None:
    """Each MX format names its stored dtype and packing."""
    assert cls.storage_dtype == storage_dtype
    assert cls.values_per_byte == values_per_byte
    assert cls.scale_dtype == DType.float8_e8m0fnu


@_MX_CONTAINERS
def test_mx_zeros_keeps_leading_dims(
    cls: type[MXFP8Tensor | MXFP4Tensor],
    storage_dtype: DType,
    values_per_byte: int,
) -> None:
    """A stacked-experts shape keeps its leading axis on both leaves."""
    with F.lazy():
        qt = cls.zeros((4, 256, 512))
        assert list(qt.weight.shape) == [4, 256, 512 // values_per_byte]
        assert list(qt.weight_scale.shape) == [4, 256, 16]


@_MX_CONTAINERS
def test_mx_to_moves_both_leaves(
    cls: type[MXFP8Tensor | MXFP4Tensor],
    storage_dtype: DType,
    values_per_byte: int,
) -> None:
    """MX has no host-side scalars, so ``to()`` moves every leaf."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        device = mock(0)
        moved = cls.zeros((256, 512)).to(device)

        assert moved.weight.device == device
        assert moved.weight_scale.device == device


# --------------------------------------------------------------------------- #
# MXFP8Quantizer / MXFP4Quantizer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [MXFP8Quantizer, MXFP4Quantizer])
def test_mx_quantizer_rejects_k_not_multiple_of_block(
    cls: type[MXFP8Quantizer | MXFP4Quantizer],
) -> None:
    """K must divide the 32-wide MX block."""
    with F.lazy():
        with pytest.raises(ValueError, match="multiple of 32"):
            cls(Tensor.zeros((4, 48), dtype=DType.bfloat16))


_MX_QUANTIZERS = pytest.mark.parametrize(
    ("cls", "container"),
    [(MXFP8Quantizer, MXFP8Tensor), (MXFP4Quantizer, MXFP4Tensor)],
)


@_MX_QUANTIZERS
def test_mx_quantizer_is_its_container(
    cls: type[MXFP8Quantizer | MXFP4Quantizer],
    container: type[MXFP8Tensor | MXFP4Tensor],
) -> None:
    """A quantizer passes wherever its format's container does."""
    with F.lazy():
        raw = Tensor.zeros((8, 32), dtype=DType.bfloat16)
        qt = cls(raw)

        assert isinstance(qt, container)
        assert qt._tensor is raw
        assert qt.block_size == (1, 32)
        assert list(qt.shape) == [8, 32]
        assert all_mxfp8([MXFP8Tensor.zeros((8, 32)), qt]) == (
            container is MXFP8Tensor
        )
        assert all_mxfp4([MXFP4Tensor.zeros((8, 32)), qt]) == (
            container is MXFP4Tensor
        )


@_MX_QUANTIZERS
def test_mx_quantizer_parameter_path_is_the_raw_weight(
    cls: type[MXFP8Quantizer | MXFP4Quantizer],
    container: type[MXFP8Tensor | MXFP4Tensor],
) -> None:
    """The bf16 tensor loads as the checkpoint's ``proj.weight``."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.proj = cls(Tensor.zeros((8, 32), dtype=DType.bfloat16))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        parameters = dict(wrapper.parameters)

        assert set(parameters) == {"proj.weight"}
        assert parameters["proj.weight"] is wrapper.proj._tensor


@_MX_QUANTIZERS
def test_mx_quantizer_tree_roundtrip_keeps_the_raw_tensor(
    cls: type[MXFP8Quantizer | MXFP4Quantizer],
    container: type[MXFP8Tensor | MXFP4Tensor],
) -> None:
    """Rebuilding the module tree puts the bf16 tensor back as ``_tensor``."""
    with F.lazy():
        raw = Tensor.zeros((8, 32), dtype=DType.bfloat16)
        leaves, treedef = tree.flatten(cls(raw))
        rebuilt = tree.unflatten(treedef, leaves)

        assert isinstance(rebuilt, cls)
        assert rebuilt.block_size == (1, 32)
        assert rebuilt._tensor is raw


def test_mxfp8_quantizer_packs_on_first_use(sm100_arch: None) -> None:
    """Its fields read the MXFP8 quantization, with rank-2 checkpoint scales."""
    del sm100_arch
    with F.lazy():
        qt = MXFP8Quantizer(Tensor.zeros((128, 64), dtype=DType.bfloat16))

        assert qt.weight.dtype == DType.float8_e4m3fn
        assert list(qt.weight.shape) == [128, 64]
        assert qt.weight_scale.dtype == DType.float8_e8m0fnu
        assert list(qt.weight_scale.shape) == [128, 2]


def test_mxfp4_quantizer_packs_on_first_use() -> None:
    """Its fields read the MXFP4 quantization; the op is rank-2 everywhere."""
    with F.lazy():
        qt = MXFP4Quantizer(Tensor.zeros((128, 64), dtype=DType.bfloat16))

        assert qt.weight.dtype == DType.uint8
        assert list(qt.weight.shape) == [128, 32]
        assert qt.weight_scale.dtype == DType.float8_e8m0fnu
        assert list(qt.weight_scale.shape) == [128, 2]
