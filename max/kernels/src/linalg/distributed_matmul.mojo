# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from gpu.comm.allreduce import (
    MAX_GPUS,
    Signal,
    allreduce,
    elementwise_epilogue_type,
)
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from memory import UnsafePointer
from linalg.matmul_gpu import _matmul_gpu
from sys import sizeof
from utils import IndexList, StaticTuple
from gpu.grid_controls import PDLLevel


@parameter
fn _matmul_allreduce[
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
    type: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
](
    a_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    This function is used as a reference to check correctness and benchmark other versions that split the matrices and overlap the computation.
    """

    @parameter
    for i in range(ngpus):
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_temp_buffers[i], a_buffers[i], b_buffers[i], ctxs[i]
        )

    allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
        rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
            c_temp_buffers
        ),
        rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
            output_buffers
        ),
        rank_sigs,
        ctxs,
    )


@parameter
fn _matmul_allreduce_split_m[
    ngpus: Int,
    num_partitions: Int,
    outputs_lambda: elementwise_epilogue_type,
    type: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    Split the A matrices into `num_partitions` parts of size (M // num_partitions, K)
    This way we can perform `num_partitions` independent matmul + allreduce kernels, and overlap some of the computation.
    """

    var m = c_temp_buffers[0].dim[0]()
    var n = c_temp_buffers[0].dim[1]()
    var k = b_buffers[0].dim[1]()
    var m_part = m // num_partitions
    var length = m_part * n

    alias m_part_dim = Dim() if a_static_shape.at[
        0
    ]().is_dynamic() else a_static_shape.at[0]() // num_partitions
    alias a_part_static_shape = DimList(m_part_dim, a_static_shape.get[1]())
    alias c_part_static_shape = DimList(m_part_dim, c_static_shape.get[1]())

    # Create list of partial A and C NDBuffers for matmul.
    var A_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, a_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, a_part_static_shape]())
    var C_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape]())
    var Out_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape]())

    # Overlap matmul with previous partition's allreduce
    @parameter
    for stage in range(num_partitions):
        alias pdl_matmul = PDLLevel.OVERLAP_AT_END if stage == 0 else PDLLevel.NO_WAIT_OVERLAP_AT_END

        @parameter
        for i in range(ngpus):
            A_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, a_part_static_shape
            ](
                a_buffers[i].data + stage * m_part * k,
                DimList(m_part, k),
            )
            C_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, c_part_static_shape
            ](
                c_temp_buffers[i].data + stage * length,
                DimList(m_part, n),
            )
            Out_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, c_part_static_shape
            ](
                output_buffers[i].data + stage * length,
                DimList(m_part, n),
            )
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
                pdl_level = pdl_matmul if overlap_with_dpl else PDLLevel(),
            ](C_parts[i], A_parts[i], b_buffers[i], ctxs[i])

        @always_inline
        @parameter
        @__copy_capture(stage, m_part)
        fn outputs_lambda_wrapper[
            input_index: Int,
            _type: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_type, _width]) -> None:
            # Convert coords in the split buffer to the global coords.
            var i = coords[0]
            var j = coords[1]
            var global_coords = rebind[IndexList[_rank]](
                IndexList[2](i + stage * m_part, j)
            )
            outputs_lambda[input_index, alignment=_alignment](
                global_coords, val
            )

        allreduce[
            ngpus=ngpus,
            outputs_lambda=outputs_lambda_wrapper,
            pdl_level = PDLLevel.OVERLAP_AT_BEGINNING if overlap_with_dpl else PDLLevel(),
        ](
            rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
                C_parts
            ),
            rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
                Out_parts
            ),
            rank_sigs,
            ctxs,
        )


@parameter
fn _matmul_allreduce_split_n[
    ngpus: Int,
    num_partitions: Int,
    outputs_lambda: elementwise_epilogue_type,
    type: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    Split the B matrices into `num_partitions` parts of size (N // num_partitions, K)
    This way we can perform `num_partitions` independent matmul + allreduce kernels, and overlap some of the computation.
    """

    constrained[
        not b_static_shape.at[0]().is_dynamic(), "N dimension must be static"
    ]()
    alias n = b_static_shape.get[0]()
    constrained[
        n % num_partitions == 0, "num_partitions doesn't split evenly N"
    ]()
    alias n_part = n // num_partitions
    var m = c_temp_buffers[0].dim[0]()
    var k = b_buffers[0].dim[1]()
    var length = m * n_part

    alias b_part_static_shape = DimList(n_part, b_static_shape.get[1]())
    alias c_part_static_shape = DimList(c_static_shape.get[0](), n_part)

    # Create list of partial B and C NDBuffers for matmul.
    var B_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, b_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, b_part_static_shape]())
    var C_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape]())
    var Out_parts = InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](NDBuffer[type, 2, MutableAnyOrigin, c_part_static_shape]())

    # Overlap matmul with previous partition's allreduce
    @parameter
    for stage in range(num_partitions):
        alias pdl_matmul = PDLLevel.OVERLAP_AT_END if stage == 0 else PDLLevel.NO_WAIT_OVERLAP_AT_END

        @parameter
        for i in range(ngpus):
            B_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, b_part_static_shape
            ](
                b_buffers[i].data + stage * n_part * k,
                DimList(n_part, k),
            )
            C_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, c_part_static_shape
            ](
                c_temp_buffers[i].data + stage * length,
                DimList(m, n_part),
            )
            Out_parts[i] = NDBuffer[
                type, 2, MutableAnyOrigin, c_part_static_shape
            ](
                output_buffers[i].data + stage * length,
                DimList(m, n_part),
            )
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
                pdl_level = pdl_matmul if overlap_with_dpl else PDLLevel(),
            ](C_parts[i], a_buffers[i], B_parts[i], ctxs[i])

        @always_inline
        @parameter
        @__copy_capture(stage, n_part)
        fn outputs_lambda_wrapper[
            input_index: Int,
            _type: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_type, _width]) -> None:
            # Convert coords in the split buffer to the global coords.
            var i = coords[0]
            var j = coords[1]
            var global_coords = rebind[IndexList[_rank]](
                IndexList[2](i, j + stage * n_part)
            )
            outputs_lambda[input_index, alignment=_alignment](
                global_coords, val
            )

        allreduce[
            ngpus=ngpus,
            outputs_lambda=outputs_lambda_wrapper,
            pdl_level = PDLLevel.OVERLAP_AT_BEGINNING if overlap_with_dpl else PDLLevel(),
        ](
            rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
                C_parts
            ),
            rebind[InlineArray[NDBuffer[type, 2, MutableAnyOrigin], ngpus]](
                Out_parts
            ),
            rank_sigs,
            ctxs,
        )


@parameter
fn matmul_allreduce[
    ngpus: Int,
    partition_dim: Int,
    num_partitions: Int,
    outputs_lambda: elementwise_epilogue_type,
    type: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[type, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    Split the A or B and C matrices into `num_partitions` submatrices at dimension `partition_dim`.
    This way we can perform `num_partitions` independent matmul + allreduce kernels, and overlap some of the computation.
    """

    constrained[partition_dim == 0 or partition_dim == 1]()

    @parameter
    if num_partitions == 1:
        _matmul_allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
            a_buffers,
            b_buffers,
            c_temp_buffers,
            output_buffers,
            rank_sigs,
            ctxs,
        )
    elif partition_dim == 0:
        _matmul_allreduce_split_m[
            ngpus=ngpus,
            num_partitions=num_partitions,
            outputs_lambda=outputs_lambda,
            overlap_with_dpl=overlap_with_dpl,
        ](a_buffers, b_buffers, c_temp_buffers, output_buffers, rank_sigs, ctxs)
    else:
        _matmul_allreduce_split_n[
            ngpus=ngpus,
            num_partitions=num_partitions,
            outputs_lambda=outputs_lambda,
            overlap_with_dpl=overlap_with_dpl,
        ](a_buffers, b_buffers, c_temp_buffers, output_buffers, rank_sigs, ctxs)
