# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_utils import UnsafeTensorSlice
from buffer.dimlist import DimList
import algorithm.functional


fn register(name: StringLiteral):
    return


fn elementwise():
    return


@value
@register_passable
struct KernelSpec:
    var target: StringLiteral
    var single_thread_blocking_override: Bool

    fn __init__(inout self):
        self.target = "cpu"
        self.single_thread_blocking_override = False


@no_inline
fn shapeof(tensor_name: StringLiteral) -> Optional[DimList]:
    return None


@no_inline
fn stridesof(tensor_name: StringLiteral) -> Optional[DimList]:
    return None


@no_inline
fn output_lambda[
    type: DType = DType.invalid,
    rank: Int = 0,
](tensor_name: StringLiteral) -> Optional[
    fn[width: Int] (StaticIntTuple[rank], SIMD[type, width]) capturing -> None
]:
    return None


@no_inline
fn foreach[
    type: DType,
    rank: Int, //,
    func: fn[width: Int] (StaticIntTuple[rank]) capturing -> SIMD[type, width],
](inout tensor: UnsafeTensorSlice[type, rank]):
    alias simd_width = simdwidthof[tensor.type]()

    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](index: StaticIntTuple[rank]) capturing:
        constrained[rank == tensor.rank]()
        var val = func[width](rebind[StaticIntTuple[tensor.rank]](index))
        tensor.store(index, val)

    algorithm.functional.elementwise[elementwise_fn_wrapper, simd_width](
        tensor.get_static_spec().shape
    )
