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
"""GPU tests for `scatter_elements` (the `mo.scatter*` kernel).

Each case runs the GPU and CPU paths on the same inputs and checks both
against a serial host reference. Shapes are prime so the elementwise launch
has a ragged tail; outputs start as poison with a guard band after them.
Reduction cases send many updates to a few output elements, with update
values chosen so the result is exact in any application order.
"""

from std.collections import OptionalReg
from std.math import isnan
from std.testing import assert_equal, assert_true
from std.utils import IndexList
from std.utils.numerics import inf, max_finite, nan

from extensibility import DynamicTensor
from max.gpu.host import DeviceContext
from nn.gather_scatter import (
    _AtomicUpdateFn,
    _atomic_add,
    _atomic_reduce,
    scatter_elements,
)

comptime ReduceFn = def[dtype: DType, width: SIMDLength](
    SIMD[dtype, width], SIMD[dtype, width]
) thin -> SIMD[dtype, width]

# Elements after the output that the kernel must leave untouched.
comptime GUARD = 17


@inline(.always)
def _add[
    ty: DType, width: SIMDLength
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return lhs + rhs


@inline(.always)
def _mul[
    ty: DType, width: SIMDLength
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return lhs * rhs


@inline(.always)
def _max[
    ty: DType, width: SIMDLength
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return max(lhs, rhs)


@inline(.always)
def _min[
    ty: DType, width: SIMDLength
](lhs: SIMD[ty, width], rhs: SIMD[ty, width]) -> SIMD[ty, width]:
    return min(lhs, rhs)


@inline(.always)
def _upd_distinct[dt: DType](k: Int) -> Scalar[dt]:
    return Scalar[dt](k % 97 + 20)


@inline(.always)
def _upd_small[dt: DType](k: Int) -> Scalar[dt]:
    return Scalar[dt](k % 3 + 1)


@inline(.always)
def _upd_sparse_twos[dt: DType](k: Int) -> Scalar[dt]:
    """Every 256 updates, 4 consecutive ones are twos. Along an axis with 4
    duplicate targets those land on distinct targets, so each target gets at
    most 17 factors of two: products stay exact and in range even for int32,
    and a lost update changes the result."""
    return Scalar[dt](2 if k % 256 < 4 else 1)


@inline(.always)
def _upd_signed[dt: DType](k: Int) -> Scalar[dt]:
    return Scalar[dt]((k * 37) % 251 - 125)


@inline(.always)
def _poison[dt: DType]() -> Scalar[dt]:
    comptime if dt.is_floating_point():
        return nan[dt]()
    else:
        return max_finite[dt]()


@inline(.always)
def _is_poison[dt: DType](v: Scalar[dt]) -> Bool:
    comptime if dt.is_floating_point():
        return isnan(v)
    else:
        return v == max_finite[dt]()


@inline(.always)
def _upd_with_inf[dt: DType](k: Int) -> Scalar[dt]:
    return inf[dt]() if k % 7 == 0 else Scalar[dt](k % 3 + 1)


@inline(.always)
def _upd_with_nan[dt: DType](k: Int) -> Scalar[dt]:
    return nan[dt]() if k % 7 == 0 else Scalar[dt](k % 3 + 1)


def _unravel[rank: Int](flat: Int, shape: IndexList[rank]) -> IndexList[rank]:
    var coords = IndexList[rank]()
    var rem = flat
    comptime for i in reversed(range(rank)):
        coords[i] = rem % shape[i]
        rem //= shape[i]
    return coords


def _ravel[rank: Int](coords: IndexList[rank], shape: IndexList[rank]) -> Int:
    var flat = 0
    comptime for i in range(rank):
        flat = flat * shape[i] + coords[i]
    return flat


def _poisoned_output[dt: DType](num_elements: Int) -> List[Scalar[dt]]:
    return List[Scalar[dt]](length=num_elements + GUARD, fill=_poison[dt]())


def _check_output[
    dt: DType
](got: List[Scalar[dt]], expected: List[Scalar[dt]], label: String) raises:
    for i in range(len(expected)):
        comptime if dt.is_floating_point():
            if isnan(expected[i]):
                assert_true(
                    isnan(got[i]), String(label, " expected NaN, i=", i)
                )
                continue
        assert_equal(got[i], expected[i], String(label, " i=", i))
    for i in range(len(expected), len(got)):
        assert_true(
            _is_poison(got[i]), String(label, " wrote guard element ", i)
        )


def run_case[
    rank: Int,
    //,
    dt: DType,
    itype: DType,
    update_fn: def(Int) thin -> Scalar[dt],
    reduce_fn: OptionalReg[ReduceFn] = None,
    atomic_update_fn: OptionalReg[_AtomicUpdateFn] = (
        OptionalReg[_AtomicUpdateFn](
            _atomic_reduce[reduce_fn.value()]
        ) if reduce_fn else None
    ),
](
    gpu: DeviceContext,
    cpu: DeviceContext,
    in_shape: IndexList[rank],
    upd_shape: IndexList[rank],
    axis: Int,
    dup_targets: Int = 0,
    neg_indices: Bool = False,
) raises:
    """Scatters `upd_shape` updates into `in_shape` along `axis`.

    With `dup_targets == 0`, indices along the axis are distinct for each
    slice (so overwrite has a single winner); otherwise every update lands on
    one of `dup_targets` positions. `neg_indices` writes every other index in
    its negative form.
    """
    var label = String(
        dt,
        " ",
        in_shape,
        " <- ",
        upd_shape,
        " axis=",
        axis,
        " dup=",
        dup_targets,
        " neg=",
        neg_indices,
    )
    var norm_axis = axis + rank if axis < 0 else axis
    var dim = in_shape[norm_axis]
    var n_in = in_shape.flattened_length()
    var n_upd = upd_shape.flattened_length()

    var input = List[Scalar[dt]](capacity=n_in)
    for i in range(n_in):
        input.append(Scalar[dt](i % 13))

    var updates = List[Scalar[dt]](capacity=n_upd)
    var indices = List[Scalar[itype]](capacity=n_upd)
    for k in range(n_upd):
        updates.append(update_fn(k))
        var coords = _unravel(k, upd_shape)
        var along = coords[norm_axis]
        var other = 0
        comptime for i in range(rank):
            if i != norm_axis:
                other += coords[i]
        var idx: Int
        if dup_targets == 0:
            # 7 is coprime with every axis size used below, so this is a
            # bijection on [0, dim) for each fixed set of other coordinates.
            idx = (along * 7 + other) % dim
        else:
            idx = (along + other) % dup_targets
        if neg_indices and k % 2 == 1:
            idx -= dim
        indices.append(Scalar[itype](idx))

    var expected = input.copy()
    for k in range(n_upd):
        var coords = _unravel(k, upd_shape)
        var idx = Int(indices[k])
        coords[norm_axis] = idx + dim if idx < 0 else idx
        var o = _ravel(coords, in_shape)
        comptime if reduce_fn:
            comptime reduce = reduce_fn.value()
            expected[o] = reduce[dt, 1](expected[o], updates[k])
        else:
            expected[o] = updates[k]

    var cpu_out = _poisoned_output[dt](n_in)
    scatter_elements[target="cpu", atomic_update_fn=atomic_update_fn](
        DynamicTensor[dt, rank](input.unsafe_ptr(), in_shape),
        DynamicTensor[itype, rank](indices.unsafe_ptr(), upd_shape),
        DynamicTensor[dt, rank](updates.unsafe_ptr(), upd_shape),
        axis,
        DynamicTensor[dt, rank](cpu_out.unsafe_ptr(), in_shape),
        cpu,
    )
    _check_output(cpu_out, expected, String("cpu ", label))

    var in_dev = gpu.enqueue_create_buffer[dt](n_in)
    var upd_dev = gpu.enqueue_create_buffer[dt](n_upd)
    var idx_dev = gpu.enqueue_create_buffer[itype](n_upd)
    var out_dev = gpu.enqueue_create_buffer[dt](n_in + GUARD)
    var gpu_out = _poisoned_output[dt](n_in)
    gpu.enqueue_copy(in_dev, input.unsafe_ptr())
    gpu.enqueue_copy(upd_dev, updates.unsafe_ptr())
    gpu.enqueue_copy(idx_dev, indices.unsafe_ptr())
    gpu.enqueue_copy(out_dev, gpu_out.unsafe_ptr())

    scatter_elements[target="gpu", atomic_update_fn=atomic_update_fn](
        DynamicTensor[dt, rank](in_dev.unsafe_ptr(), in_shape),
        DynamicTensor[itype, rank](idx_dev.unsafe_ptr(), upd_shape),
        DynamicTensor[dt, rank](upd_dev.unsafe_ptr(), upd_shape),
        axis,
        DynamicTensor[dt, rank](out_dev.unsafe_ptr(), in_shape),
        gpu,
    )
    gpu.enqueue_copy(gpu_out.unsafe_ptr(), out_dev)
    gpu.synchronize()
    _check_output(gpu_out, expected, String("gpu ", label))

    _ = in_dev^
    _ = upd_dev^
    _ = idx_dev^
    _ = out_dev^


def test_overwrite(gpu: DeviceContext, cpu: DeviceContext) raises:
    comptime f32 = DType.float32
    comptime upd = _upd_distinct[f32]
    run_case[f32, DType.int64, upd](gpu, cpu, IndexList[1](13), (11,), 0)
    run_case[f32, DType.int64, upd](
        gpu, cpu, IndexList[1](13), (11,), -1, neg_indices=True
    )
    run_case[f32, DType.int32, upd](gpu, cpu, IndexList[2](17, 13), (11, 13), 0)
    run_case[f32, DType.int32, upd](
        gpu,
        cpu,
        IndexList[2](17, 13),
        (17, 5),
        -1,
        neg_indices=True,
    )
    run_case[f32, DType.int64, upd](
        gpu, cpu, IndexList[3](3, 11, 5), (2, 11, 3), 1
    )
    run_case[f32, DType.int64, upd](
        gpu,
        cpu,
        IndexList[3](3, 11, 5),
        (3, 4, 5),
        -2,
        neg_indices=True,
    )
    # 37 * 257 updates straddle many thread blocks.
    run_case[f32, DType.int64, upd](
        gpu, cpu, IndexList[2](37, 263), (37, 257), 1
    )
    run_case[DType.bfloat16, DType.int64, _upd_distinct[DType.bfloat16]](
        gpu, cpu, IndexList[2](17, 13), (11, 13), 0
    )
    run_case[DType.int32, DType.int32, _upd_distinct[DType.int32]](
        gpu, cpu, IndexList[2](17, 13), (17, 5), 1, neg_indices=True
    )
    run_case[DType.int8, DType.int64, _upd_distinct[DType.int8]](
        gpu, cpu, IndexList[3](3, 11, 5), (3, 4, 5), -2
    )


def test_reductions[dt: DType](gpu: DeviceContext, cpu: DeviceContext) raises:
    # Each case sends 5 * 4099 updates to 4 positions per row, so thousands
    # of threads reduce into the same elements concurrently.
    comptime heavy_in = IndexList[2](5, 4099)
    comptime heavy_upd = IndexList[2](5, 4099)
    run_case[dt, DType.int64, _upd_small[dt], _add, _atomic_add](
        gpu, cpu, heavy_in, heavy_upd, 1, dup_targets=4
    )
    run_case[dt, DType.int32, _upd_sparse_twos[dt], _mul](
        gpu, cpu, heavy_in, heavy_upd, -1, dup_targets=4
    )
    run_case[dt, DType.int64, _upd_signed[dt], _max](
        gpu, cpu, heavy_in, heavy_upd, 1, dup_targets=4, neg_indices=True
    )
    run_case[dt, DType.int64, _upd_signed[dt], _min](
        gpu, cpu, heavy_in, heavy_upd, -1, dup_targets=4
    )
    run_case[dt, DType.int64, _upd_small[dt], _add, _atomic_add](
        gpu,
        cpu,
        IndexList[3](7, 3, 11),
        (7, 3, 11),
        0,
        dup_targets=3,
        neg_indices=True,
    )
    run_case[dt, DType.int64, _upd_signed[dt], _max](
        gpu, cpu, IndexList[3](7, 3, 11), (5, 2, 11), -3
    )


def test_native_add_dtypes(gpu: DeviceContext, cpu: DeviceContext) raises:
    """64-bit adds, and inf and NaN, which an atomic add carries through."""
    comptime heavy = IndexList[2](5, 4099)
    comptime f64 = DType.float64
    comptime f32 = DType.float32
    comptime if not gpu.target.is_apple_gpu():
        run_case[f64, DType.int64, _upd_small[f64], _add, _atomic_add](
            gpu, cpu, heavy, (5, 4099), 1, dup_targets=4
        )
    run_case[f32, DType.int64, _upd_with_inf[f32], _add, _atomic_add](
        gpu, cpu, heavy, (5, 4099), 1, dup_targets=4
    )
    run_case[f32, DType.int64, _upd_with_nan[f32], _add, _atomic_add](
        gpu, cpu, heavy, (5, 4099), 1, dup_targets=4
    )
    # Every update of a row hits one element.
    run_case[f32, DType.int32, _upd_small[f32], _add, _atomic_add](
        gpu, cpu, IndexList[2](3, 8191), (3, 8191), 1, dup_targets=1
    )


def test_narrow_reductions(gpu: DeviceContext, cpu: DeviceContext) raises:
    """16- and 8-bit dtypes go through sub-word compare-exchange atomics.
    Few enough collisions that bf16 sums stay exact."""
    comptime bf16 = DType.bfloat16
    comptime i8 = DType.int8
    run_case[bf16, DType.int64, _upd_small[bf16], _add, _atomic_add](
        gpu, cpu, IndexList[2](3, 31), (3, 31), 1, dup_targets=2
    )
    run_case[bf16, DType.int64, _upd_signed[bf16], _max](
        gpu,
        cpu,
        IndexList[2](5, 4099),
        (5, 4099),
        1,
        dup_targets=4,
    )
    run_case[i8, DType.int64, _upd_signed[i8], _min](
        gpu,
        cpu,
        IndexList[2](5, 4099),
        (5, 4099),
        -1,
        dup_targets=4,
    )


def test_empty_updates(gpu: DeviceContext, cpu: DeviceContext) raises:
    """No updates: the output is a plain copy of the input."""
    run_case[
        DType.float32, DType.int64, _upd_small[DType.float32], _add, _atomic_add
    ](gpu, cpu, IndexList[2](7, 5), (0, 5), 0)


def main() raises:
    var cpu = DeviceContext(api="cpu")
    with DeviceContext() as gpu:
        test_overwrite(gpu, cpu)
        test_reductions[DType.float32](gpu, cpu)
        test_reductions[DType.int32](gpu, cpu)
        test_native_add_dtypes(gpu, cpu)
        # Metal has no atomic compare-exchange at any width but 32 bits.
        comptime if not gpu.target.is_apple_gpu():
            test_reductions[DType.int64](gpu, cpu)
            test_narrow_reductions(gpu, cpu)
        test_empty_updates(gpu, cpu)
