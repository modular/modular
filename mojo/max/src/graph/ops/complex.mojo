# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# These operations assume that all tensors have last dim == 2, representing real
# and imaginary parts.


from tensor import TensorShape


def as_complex(real: Symbol, imag: Symbol) -> Symbol:
    return stack[axis= -1](real, imag)


def as_interleaved_complex(interleaved: Symbol) -> Symbol:
    """Reshape the input tensor as complex, interpreting the last dimension
    as being alternating (real, imag) pairs."""
    var g = interleaved.graph()
    let rank = interleaved.tensor_type().rank()
    let shape = shape_of(interleaved)
    let back_dims = g.constant(Tensor[DType.int64](TensorShape(2), -1, 2))
    let new_shape = concat[axis=0](shape[: rank - 1], back_dims)
    return reshape(interleaved, new_shape)


def as_real(complex: Symbol) -> (Symbol, Symbol):
    let real: Symbol
    let imag: Symbol
    real, imag = split[axis= -1](complex, (1, 1))
    return (squeeze(real, axis=-1), squeeze(imag, axis=-1))


def mul_complex(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Multiply complex-like real valued tensors."""
    let l_real: Symbol
    let l_imag: Symbol
    let r_real: Symbol
    let r_imag: Symbol
    l_real, l_imag = as_real(lhs)
    r_real, r_imag = as_real(rhs)

    let out_real = (l_real * r_real) - (l_imag * r_imag)
    let out_imag = (l_real * r_imag) + (l_imag * r_real)
    return as_complex(out_real, out_imag)
