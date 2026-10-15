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

"""
The code below computes the attention score for a tile of size BN x BD.
It follows the exact arithmetic as described in the FlashAttention-2 paper
(https://arxiv.org/pdf/2307.08691). The variable names in this program
reflect the variable names in the algorithms from the paper.

Here, the following tensors are Q for the query,
K for the key, V for the value, and O for the output.

       Q            K               V
  +----D----+  +----D----+   +--+--BD--+---+
  |         |  |.........|   |  |......|   |
  |         |  |.........|   |  |......|   |
  +---------+  |.........|   |  |......|   |
  |.........|  |.........|   |  |......|   |
  BN........|  N.........|   N  |......|   |
  |.........|  |.........|   |  |......|   |
  +---------+  |.........|   |  |......|   |
  |         |  |.........|   |  |......|   |
  |         |  |.........|   |  |......|   |
  |         |  |.........|   |  |......|   |
  +---------+  +---------+   +--+------+---+

The main trick is in the softmax computation.
As the paper says, S and P are intermediate values.

Let S = Q * K^T  ∈ R^{N, D}
    P = Softmax(S) ∈ R^{N, D}
The attention score is O = P * V ∈ R^{N, D}.

One way to think about this is to consider what happens if we
split the dimensions N in  K, and Q into two tiles: K_1 and K_2, V_1 and V_2.
Then we can incrementally compute the output as follows:
  S_1 = Q * K_1, S_2 = Q * K_2
  O_i = O_{i-1} * renormalization_factor + softmax(S_i) * V_i

This allows for the incremental computation of softmax(S_i) * V_i,
leading to the final output.
"""


from max.gpu.host import DeviceContext
from max.gpu import block_idx
from max.gpu.sync import barrier
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from layout.math import max, sum
from layout.tensor_core import TensorCore

from extensibility import register, InputTensor, OutputTensor

from std.utils import Index


@register("modular_ops::fused_attention_custom")
struct FusedAttention:
    """Registers the `fused_attention_custom` op, allowing python to use it from the `max`
    package.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,  # Forces the previous two params to be inferred from the args
        BN: Int,  # Dimension of blocks to split Q into
        BD: Int,  # Dimension of blocks to split K, V into
        target: StaticString,  # "cpu" or "gpu"
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        query: InputTensor[dtype=dtype, rank=rank, ...],
        key: InputTensor[dtype=dtype, rank=rank, ...],
        value: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert rank == 2, "rank must be 2"

        # Query tensor
        var Q = query.to_tile_tensor().as_unsafe_any_origin()
        # Key tensor
        var K = key.to_tile_tensor().as_unsafe_any_origin()
        # Value tensor
        var V = value.to_tile_tensor().as_unsafe_any_origin()
        # Attention output tensor
        var O = output.to_tile_tensor().as_unsafe_any_origin()

        comptime if target == "cpu":
            print("Running on CPU")
            fused_attention_cpu[BN, BD](Q, K, V, O)
        else:
            var dev_ctx = ctx
            print("Running on GPU")
            fused_attention_gpu[BN, BD](dev_ctx, Q, K, V, O)


# CustomOpLibrary does not support names with '::' in them currently. Create
# a simple alias so the whisper example works.
@register("fused_attention_custom")
struct FusedAttentionAlias:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,  # Forces the previous two params to be inferred from the args
        BN: Int,  # Dimension of blocks to split Q into
        BD: Int,  # Dimension of blocks to split K, V into
        target: StaticString,  # "cpu" or "gpu"
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        query: InputTensor[dtype=dtype, rank=rank, ...],
        key: InputTensor[dtype=dtype, rank=rank, ...],
        value: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        FusedAttention.execute[BN=BN, BD=BD, target=target](
            output, query, key, value, ctx
        )


@inline(.always)
def matmul_b_transpose(
    lhs: TileTensor,
    rhs: TileTensor,
    out res: TileTensor[
        lhs.dtype,
        type_of(row_major[lhs.static_shape[0], rhs.static_shape[0]]()),
        MutUntrackedOrigin,
    ],
):
    res = stack_allocation[lhs.dtype](
        row_major[lhs.static_shape[0], rhs.static_shape[0]]()
    )

    comptime for m in range(lhs.static_shape[0]):
        comptime for n in range(rhs.static_shape[0]):
            res[m, n] = 0.0

            comptime for k in range(lhs.static_shape[1]):
                res[m, n] += rebind[res.ElementType](
                    lhs[m, k].cast[res.dtype]()
                ) * rebind[res.ElementType](rhs[n, k].cast[res.dtype]())


# The bulk of the code below implements what the papers calls
# an "online softmax", which is local to each block.
# The algorithm is described as:
#
# $$$
# m_1 = rowmax(S_1)
# l_1 = rowsum(e^(S_1-m_1))
# P_1 = diag(l_1)^-1 * e^(S_1-m_1)
# O_1 = P_1*V_1 = diag(l_1)^-1 * e^(S_1-m_1) * V_1
# m_2 = max(m_1, rowmax(S_2)) = m
# l_2 = e^(m_1-m_2) * l_1 _ rowsum(e^(S_2-m_2))
#     = rowsum(e^(S_1-m)) + rowsum(e^(S_2-m)) = ls
# P_2 = diag(l_2)^-1 * e^(S_2-m_2)
# O_2 = diag(l_1/l_2)^-1 * O_1 + (P_2 * V_2)
#     = diag(l_2)^-1 * e^(S_2-m) * V
# $$$


@inline(.always)
def fused_attention_cpu[
    BN: Int, BD: Int
](Q: TileTensor, K: TileTensor, V: TileTensor, O: TileTensor[mut=True, ...],):
    comptime N = K.static_shape[0]
    comptime D = K.static_shape[1]

    comptime for tile_n in range(N // BN):
        var Q_tile = Q.tile[BN, D](tile_n, 0)

        comptime for tile_d in range(D // BD):
            var m_1 = stack_allocation[Q.dtype](row_major[BN]()).fill(
                Scalar[Q.dtype].MIN
            )

            var l_1 = stack_allocation[Q.dtype](row_major[BN]()).fill(0)

            var O_i = stack_allocation[Q.dtype](row_major[BN, BD]()).fill(0)

            comptime for tile_n_idx in range(N // BN):
                var K_tile = K.tile[BN, D](tile_n_idx, 0)
                var V_tile = V.tile[BN, BD](tile_n_idx, tile_d)

                var S = matmul_b_transpose(Q_tile, K_tile)
                var m_2 = stack_allocation[Q.dtype](row_major[BN]())
                max[axis=1](S, m_2)
                m_2.max(m_1)

                # l_2 = exp(m_1 - m_2) * l_1 + sum(axis=1, exp(S - m_2))
                var l_2 = stack_allocation[Q.dtype](row_major[BN]())
                l_2.copy_from(m_1)
                l_2 -= m_2
                l_2.exp()
                l_2 *= l_1
                var E = stack_allocation[Q.dtype](row_major[BN, BN]())
                E.copy_from(S)
                E -= m_2
                E.exp()
                var l_2_sum = stack_allocation[Q.dtype](row_major[BN]())
                sum[axis=1](E, l_2_sum)
                l_2 += l_2_sum

                # P = exp(S - m_2) / l_2
                E /= l_2
                O_i *= l_1
                O_i /= l_2
                var m_1_delta = stack_allocation[Q.dtype](row_major[BN]())
                m_1_delta.copy_from(m_1)
                m_1_delta -= m_2
                m_1_delta.exp()
                O_i *= m_1_delta
                var PV = matmul["cpu"](E, V_tile)
                O_i += PV
                m_1.copy_from(m_2)
                l_1.copy_from(l_2)

            O.tile[BN, BD](tile_n, tile_d).copy_from(O_i)


@inline(.always)
def matmul[
    dtype: DType,
    rhs_dtype: DType,
    //,
    target: StaticString,
    transpose_b: Bool = False,
](
    lhs: TileTensor[dtype, ...],
    rhs: TileTensor[rhs_dtype, ...],
    out res: TileTensor[
        dtype,
        type_of(row_major[lhs.static_shape[0], rhs.static_shape[0]]()),
        MutUntrackedOrigin,
        address_space=lhs.address_space,
    ],
):
    res = stack_allocation[dtype, address_space=lhs.address_space](
        row_major[lhs.static_shape[0], rhs.static_shape[0]]()
    )

    comptime if target == "cpu":
        comptime for m in range(lhs.static_shape[0]):
            comptime for n in range(rhs.static_shape[1]):
                res[m, n] = 0.0

                comptime for k in range(lhs.static_shape[1]):
                    res[m, n] += rebind[res.ElementType](
                        lhs[m, k].cast[res.dtype]()
                    ) * rebind[res.ElementType](rhs[k, n].cast[res.dtype]())
    else:
        comptime M = lhs.static_shape[0]
        comptime N = rhs.static_shape[0]
        comptime K = lhs.static_shape[1]

        var out_sram = stack_allocation[res.dtype, address_space=.SHARED](
            row_major[M, N]()
        )

        comptime BK = 8

        comptime assert K % 8 == 0, "K needs to be a multiple of 8"

        var mma_b_t = TensorCore[dtype, dtype, Index(M, N, BK), transpose_b]()

        var c_reg = stack_allocation[dtype, address_space=.LOCAL](
            mma_b_t.c_fragment_layout
        ).fill(0)

        # `TensorCore`'s native loads require matching operand dtypes;
        # rebind the B view once when `rhs` carries a different dtype.
        var rhs_typed = rebind[
            TileTensor[
                dtype,
                rhs.LayoutType,
                rhs.origin,
                Engine=rhs.Engine,
                address_space=rhs.address_space,
                linear_idx_type=rhs.linear_idx_type,
            ]
        ](rhs)

        comptime for k_i in range(K // BK):
            var a_reg = mma_b_t.load_a(lhs.tile[M, BK](0, k_i))

            var b_reg = mma_b_t.load_b(rhs_typed.tile[BK, N](k_i, 0))

            comptime if transpose_b:
                b_reg = rebind[type_of(b_reg)](
                    mma_b_t.load_b(rhs_typed.tile[N, BK](0, k_i))
                )

            var d_reg = mma_b_t.mma_op(a_reg, b_reg, c_reg)
            c_reg.copy_from(d_reg)
        mma_b_t.store_d(out_sram, c_reg)

        barrier()
        res.copy_from(out_sram)


def fused_attention_kernel[
    q_dtype: DType,
    q_layout: TensorLayout,
    k_dtype: DType,
    k_layout: TensorLayout,
    v_dtype: DType,
    v_layout: TensorLayout,
    o_dtype: DType,
    o_layout: TensorLayout,
    BN: Int,
    BD: Int,
](
    Q: TileTensor[q_dtype, q_layout, ImmutAnyOrigin],
    K: TileTensor[k_dtype, k_layout, ImmutAnyOrigin],
    V: TileTensor[v_dtype, v_layout, ImmutAnyOrigin],
    O: TileTensor[o_dtype, o_layout, MutAnyOrigin],
):
    comptime N = Q.static_shape[0]
    comptime D = Q.static_shape[1]

    var Q_tile = Q.tile[BN, D](block_idx.y, 0)

    var m_1 = stack_allocation[q_dtype](row_major[BN]()).fill(
        Scalar[q_dtype].MIN
    )
    var l_1 = stack_allocation[q_dtype](row_major[BN]()).fill(0)
    var O_i = stack_allocation[q_dtype](row_major[BN, BD]()).fill(0)

    comptime BN_1 = 8

    for tile_n_idx in range(N // BN_1):
        var K_tile = K.tile[BN_1, D](tile_n_idx, 0)
        var V_tile = V.tile[BN_1, BD](tile_n_idx, block_idx.x)
        var S = matmul["gpu", transpose_b=True](Q_tile, K_tile)
        var m_2 = stack_allocation[q_dtype](row_major[BN]())
        max[axis=1](S, m_2)
        m_2.max(m_1)

        # l_2 = exp(m_1 - m_2) * l_1 + sum(axis=1, exp(S - m_2))
        var l_2 = stack_allocation[q_dtype](row_major[BN]())
        l_2.copy_from(m_1)
        l_2 -= m_2
        l_2.exp()
        l_2 *= l_1
        var E = stack_allocation[q_dtype](row_major[BN, BN_1]())
        E.copy_from(S)
        E -= m_2
        E.exp()
        var l_2_sum = stack_allocation[q_dtype](row_major[BN]())
        sum[axis=1](E, l_2_sum)
        l_2 += l_2_sum

        # P = exp(S - m_2) / l_2
        E /= l_2
        O_i *= l_1
        O_i /= l_2
        var m_1_delta = stack_allocation[q_dtype](row_major[BN]())
        m_1_delta.copy_from(m_1)
        m_1_delta -= m_2
        m_1_delta.exp()
        O_i *= m_1_delta
        var PV = matmul["gpu"](E, V_tile)
        O_i += PV
        m_1.copy_from(m_2)
        l_1.copy_from(l_2)
    O.tile[BN, BD](block_idx.y, block_idx.x).copy_from(O_i)


def fused_attention_gpu[
    BN: Int,
    BD: Int,
](
    ctx: DeviceContext,
    Q: TileTensor,
    K: TileTensor,
    V: TileTensor,
    mut O: TileTensor,
) raises:
    comptime kernel_func = fused_attention_kernel[
        Q.dtype,
        Q.LayoutType,
        K.dtype,
        K.LayoutType,
        V.dtype,
        V.LayoutType,
        O.dtype,
        O.LayoutType,
        BN,
        BD,
    ]
    ctx.enqueue_function[kernel_func](
        Q,
        K,
        V,
        O,
        grid_dim=(Int(Q.dim[1]()) // BD, Int(Q.dim[0]()) // BN),
        block_dim=(32),
    )
