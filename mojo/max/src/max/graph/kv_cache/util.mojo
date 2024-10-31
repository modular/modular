# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import NDBuffer
from max.driver import Tensor
from max.tensor import TensorShape
from utils import IndexList


def ndbuffer_view_from_tensor[
    type: DType, rank: Int
](tensor: Tensor[type, rank],) -> NDBuffer[type, rank]:
    return ndbuffer_view_from_tensor_with_tensor_shape[type, rank, rank](
        tensor, 0, tensor.spec().shape
    )


def ndbuffer_view_from_tensor_with_tensor_shape[
    type: DType, in_rank: Int, out_rank: Int
](
    tensor: Tensor[type, in_rank],
    offset: Int,
    out_shape: TensorShape,
) -> NDBuffer[type, out_rank]:
    var cvt_shape = IndexList[out_rank]()

    for i in range(out_rank):
        cvt_shape[i] = out_shape[i]

    return ndbuffer_view_from_tensor(tensor, offset, cvt_shape)


def ndbuffer_view_from_tensor[
    type: DType, in_rank: Int, out_rank: Int
](
    tensor: Tensor[type, in_rank],
    offset: Int,
    out_shape: IndexList[out_rank],
) -> NDBuffer[type, out_rank]:
    var ptr = tensor._ptr + offset
    return NDBuffer[type, out_rank](ptr, out_shape)
