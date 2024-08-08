# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from layout import LayoutTensor


# Updates res with the outer product of lhs, rhs vectors, res += outer(lhs, rhs).
#
@always_inline
fn outer_product_acc(
    res: LayoutTensor,
    lhs: LayoutTensor,
    rhs: LayoutTensor,
):
    constrained[
        res.layout.known_shape()
        and lhs.layout.known_shape()
        and rhs.layout.known_shape(),
        "outer_product_acc expects inputs with statically know shapes",
    ]()
    constrained[res.rank == 2, "Only rank 2 res is allowed."]()
    constrained[lhs.rank == 1, "Only rank 1 lhs is allowed."]()
    constrained[rhs.rank == 1, "Only rank 1 rhs is allowed."]()

    alias dtype = res.dtype

    alias M = res.shape[0]()
    alias N = res.shape[1]()

    constrained[lhs.shape[0]() == M, "lhs shape mismatch"]()
    constrained[rhs.shape[0]() == N, "rhs shape mismatch"]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            res[i, j] += rebind[res.element_type](
                lhs[i].cast[dtype]()
            ) * rebind[res.element_type](rhs[j].cast[dtype]())
