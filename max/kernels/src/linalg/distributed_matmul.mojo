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


from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from comm.allreduce import (
    MAX_GPUS,
    Signal,
    elementwise_epilogue_type,
    allreduce,
)
from gpu.host import DeviceContext
from linalg.matmul_gpu import _matmul_gpu
from utils import IndexList
from gpu.grid_controls import PDLLevel, _SUPPORT_PDL_LAUNCH
from internal_utils._utils import ValOrDim, dynamic, static


@parameter
fn _matmul_allreduce[
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
](
    a_buffers: InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, out_static_shape], ngpus
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
        rebind[InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]](
            c_temp_buffers
        ),
        rebind[InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]](
            output_buffers
        ),
        rank_sigs,
        ctxs,
    )


@parameter
fn _matmul_allreduce_split_m[
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
    num_partitions: Int,
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

    alias a_part_static_shape = DimList(Dim(), a_static_shape.get[1]())
    alias c_part_static_shape = DimList(Dim(), c_static_shape.get[1]())

    # Create list of partial A and C NDBuffers for matmul.
    var A_parts = InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_part_static_shape], ngpus
    ](fill={})
    var C_parts = InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](fill={})
    var Out_parts = InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](fill={})

    # Overlap matmul with previous partition's allreduce
    for stage in range(num_partitions):

        @parameter
        for i in range(ngpus):
            A_parts[i] = NDBuffer[
                a_dtype, 2, MutableAnyOrigin, a_part_static_shape
            ](
                a_buffers[i].data + stage * m_part * k,
                DimList(m_part, k),
            )
            C_parts[i] = NDBuffer[
                out_dtype, 2, MutableAnyOrigin, c_part_static_shape
            ](
                c_temp_buffers[i].data + stage * length,
                DimList(m_part, n),
            )
            Out_parts[i] = NDBuffer[
                out_dtype, 2, MutableAnyOrigin, c_part_static_shape
            ](
                output_buffers[i].data + stage * length,
                DimList(m_part, n),
            )
            if stage == 0:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=True,
                    pdl_level = PDLLevel.OVERLAP_AT_END if overlap_with_dpl else PDLLevel(),
                ](C_parts[i], A_parts[i], b_buffers[i], ctxs[i])
            else:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=True,
                    pdl_level = PDLLevel.NO_WAIT_OVERLAP_AT_END if overlap_with_dpl else PDLLevel(),
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
            rebind[
                InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]
            ](C_parts),
            rebind[
                InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]
            ](Out_parts),
            rank_sigs,
            ctxs,
        )


@parameter
fn _matmul_allreduce_split_n[
    ngpus: Int,
    num_partitions: Int,
    outputs_lambda: elementwise_epilogue_type,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, out_static_shape], ngpus
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
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_part_static_shape], ngpus
    ](fill={})
    var C_parts = InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](fill={})
    var Out_parts = InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_part_static_shape], ngpus
    ](fill={})

    # Overlap matmul with previous partition's allreduce
    @parameter
    for stage in range(num_partitions):
        alias pdl_matmul = PDLLevel.OVERLAP_AT_END if stage == 0 else PDLLevel.NO_WAIT_OVERLAP_AT_END

        @parameter
        for i in range(ngpus):
            B_parts[i] = NDBuffer[
                b_dtype, 2, MutableAnyOrigin, b_part_static_shape
            ](
                b_buffers[i].data + stage * n_part * k,
                DimList(n_part, k),
            )
            C_parts[i] = NDBuffer[
                out_dtype, 2, MutableAnyOrigin, c_part_static_shape
            ](
                c_temp_buffers[i].data + stage * length,
                DimList(m, n_part),
            )
            Out_parts[i] = NDBuffer[
                out_dtype, 2, MutableAnyOrigin, c_part_static_shape
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
            _dtype: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
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
            rebind[
                InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]
            ](C_parts),
            rebind[
                InlineArray[NDBuffer[out_dtype, 2, MutableAnyOrigin], ngpus]
            ](Out_parts),
            rank_sigs,
            ctxs,
        )


@parameter
fn matmul_allreduce[
    ngpus: Int,
    partition_dim: Int,
    outputs_lambda: elementwise_epilogue_type,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
    overlap_with_dpl: Bool = True,
](
    a_buffers: InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
    num_partitions: ValOrDim,
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    Split the A or B and C matrices into `num_partitions` submatrices at dimension `partition_dim`.
    This way we can perform `num_partitions` independent matmul + allreduce kernels, and overlap some of the computation.
    """

    constrained[partition_dim == 0 or partition_dim == 1]()

    @parameter
    if not num_partitions.dim.is_dynamic() and num_partitions.dim.get() == 1:
        _matmul_allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
            a_buffers,
            b_buffers,
            c_temp_buffers,
            output_buffers,
            rank_sigs,
            ctxs,
        )
    elif partition_dim == 0:
        # Split on the M axis if there is more than one partition.
        if num_partitions.value == 1:
            _matmul_allreduce[ngpus=ngpus, outputs_lambda=outputs_lambda](
                a_buffers,
                b_buffers,
                c_temp_buffers,
                output_buffers,
                rank_sigs,
                ctxs,
            )
        else:
            _matmul_allreduce_split_m[
                ngpus=ngpus,
                outputs_lambda=outputs_lambda,
                overlap_with_dpl=overlap_with_dpl,
            ](
                a_buffers,
                b_buffers,
                c_temp_buffers,
                output_buffers,
                rank_sigs,
                ctxs,
                num_partitions.value,
            )

    else:
        constrained[
            not num_partitions.dim.is_dynamic(),
            "for split_n num_partitions must be a constant",
        ]()
        _matmul_allreduce_split_n[
            ngpus=ngpus,
            num_partitions = num_partitions.dim.get(),
            outputs_lambda=outputs_lambda,
            overlap_with_dpl=overlap_with_dpl,
        ](a_buffers, b_buffers, c_temp_buffers, output_buffers, rank_sigs, ctxs)


@parameter
fn matmul_allreduce[
    ngpus: Int,
    outputs_lambda: elementwise_epilogue_type,
    a_dtype: DType,
    b_dtype: DType,
    out_dtype: DType,
    a_static_shape: DimList,
    b_static_shape: DimList,
    c_static_shape: DimList,
    out_static_shape: DimList,
](
    a_buffers: InlineArray[
        NDBuffer[a_dtype, 2, MutableAnyOrigin, a_static_shape], ngpus
    ],
    b_buffers: InlineArray[
        NDBuffer[b_dtype, 2, MutableAnyOrigin, b_static_shape], ngpus
    ],
    c_temp_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, c_static_shape], ngpus
    ],
    output_buffers: InlineArray[
        NDBuffer[out_dtype, 2, MutableAnyOrigin, out_static_shape], ngpus
    ],
    rank_sigs: InlineArray[UnsafePointer[Signal], MAX_GPUS],
    ctxs: List[DeviceContext],
) raises:
    """Performs C = matmul(A, B^T) followed with Out = allreduce(C) operation across multiple GPUs.
    The implementation might potentially split A / B / C matrices and overlap computation to speedup performance.
    """

    # If we don't support PDL, we can't split the computation.
    @parameter
    if not _SUPPORT_PDL_LAUNCH:
        return matmul_allreduce[
            ngpus=ngpus,
            partition_dim=0,
            outputs_lambda=outputs_lambda,
            a_static_shape=a_static_shape,
            b_static_shape=b_static_shape,
            c_static_shape=c_static_shape,
            out_static_shape=out_static_shape,
            overlap_with_dpl=True,
        ](
            a_buffers,
            b_buffers,
            c_temp_buffers,
            output_buffers,
            rank_sigs,
            ctxs,
            num_partitions=static[1](),
        )

    # TODO: Improve logic to chose the split dim / size
    var m = c_temp_buffers[0].dim[0]()
    alias partition_dim = 0
    var num_partitions = max(1, m // 2048)

    matmul_allreduce[
        ngpus=ngpus,
        partition_dim=partition_dim,
        outputs_lambda=outputs_lambda,
        a_static_shape=a_static_shape,
        b_static_shape=b_static_shape,
        c_static_shape=c_static_shape,
        out_static_shape=out_static_shape,
        overlap_with_dpl=True,
    ](
        a_buffers,
        b_buffers,
        c_temp_buffers,
        output_buffers,
        rank_sigs,
        ctxs,
        dynamic(num_partitions),
    )
