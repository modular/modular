# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import sqrt, copysign
from layout.layout_tensor import LayoutTensor
from os import abort


fn qr_factorization[
    dtype: DType
](sigma: LayoutTensor[dtype], A: LayoutTensor[dtype]):
    """Performs QR factorization of a matrix `A` using the Householder reflector
    method.

    This function computes the QR factorization of matrix `A` in-place using
    Householder reflections. The result is stored directly in the input matrix
    `A`, with scaling factors in `sigma`. The implementation follows the LAPACK
    algorithm for generating Householder reflectors in-place.

    Algorithm:
        The Householder reflector is defined as:
            U = I - σww^H
        where:
            w = (x + νe₁)/ξ
            σ = ξ/ν
            ξ = x₀ + ν
            ν = sign(x₀)‖x‖₂

        This ensures that U^H x = -νe₁ and U^H U = I.

    References:
        [1] Lehoucq, R. B. (1996). The computation of elementary unitary matrices.
            ACM Transactions on Mathematical Software, 22(4), 393-400.
            https://www.netlib.org/lapack/lawnspdf/lawn72.pdf
            https://library.eecs.utk.edu/files/ut-cs-94-233.pdf

    Note:
        There is a typo in reference [lawn72]. The correct result is U^H x =
        -νe₁.
    """
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for k in range(n):
        x_0 = A[k, k]
        x_norm = SIMD[dtype, A.element_layout.size()](0.0)
        for i in range(m - k):
            x_norm += A[k + i, k] * A[k + i, k]
        x_norm = sqrt(x_norm)
        nu = copysign(x_norm, x_0)
        A[k, k] = -nu
        xi = x_0 + nu
        inv_xi = 1.0 / xi
        for i in range(m - k - 1):
            A[k + i + 1, k] *= inv_xi
        sigma[k] = xi / nu
        # apply reflector to A[k + 1:m, k + 1:n] for each column vector v in A[k
        # :m, k + 1:n], we compute:
        #   (I - \sigma [1; w] [1; w]^T) v = v - \sigma [1; w] ([1; w]^T v)
        # = v - \sigma ([1; w]^T v) [1; w]
        # = v - s [1; w]            where  s = \sigma * (v[0] + w^T v[1:])
        # v[0] -= s
        # v[1:] -= s * w
        for j in range(n - k - 1):
            dot = A[k, k + j + 1]  # v[0]
            for i in range(m - k - 1):
                wi = A[k + i + 1, k]  # w[i]
                vi = A[k + i + 1, k + j + 1]  # v[i + 1]
                dot += wi * vi
            s = sigma[k] * dot
            A[k, k + j + 1] -= s  # v[0] -= s
            for i in range(m - k - 1):
                A[k + i + 1, k + j + 1] -= (
                    s * A[k + i + 1, k]
                )  # v[i + 1] -= s * w


fn apply_q[
    dtype: DType
](sigma: LayoutTensor[dtype], A: LayoutTensor[dtype], X: LayoutTensor[dtype]):
    """Applies the implicit Q factor stored in `A` and `sigma` after calling
    `qr_factorization` to the `X` matrix.

    See `qr_factorization` for more details on the construction of the
    Householder reflector.
    """
    m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    q_m, q_n = Int(X.runtime_layout.shape[0]), Int(X.runtime_layout.shape[1])
    if q_m != m:
        abort("apply_q: X must have the same number of rows as A")
    for k in range(n - 1, -1, -1):
        for j in range(q_n):
            dot = X[k, j]  # v[0]
            for i in range(m - k - 1):
                wi = A[k + i + 1, k]  # w[i]
                vi = X[k + i + 1, j]  # v[i + 1]
                dot += wi * vi
            s = sigma[k] * dot
            X[k, j] -= s  # v[0] -= s
            for i in range(m - k - 1):
                X[k + i + 1, j] -= s * A[k + i + 1, k]  # v[i + 1] -= s * w


fn form_q[
    dtype: DType
](sigma: LayoutTensor[dtype], A: LayoutTensor[dtype], Q: LayoutTensor[dtype]):
    """Forms the Q factor from the implicit Q factor stored in `A` and `sigma`
    after calling `qr_factorization` and stores the result in `Q`.
    """
    q_m, q_n = Int(Q.runtime_layout.shape[0]), Int(Q.runtime_layout.shape[1])
    min_mn = min(q_m, q_n)

    # Q.fill(0.0) doesn't work
    for i in range(q_m):
        for j in range(q_n):
            Q[i, j] = 0.0

    # Set diagonal to 1.0
    for i in range(min_mn):
        Q[i, i] = 1.0

    apply_q[dtype](sigma, A, Q)
