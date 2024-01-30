# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# These operations assume that all tensors have last dim == 2, representing real
# and imaginary parts.


from tensor import Tensor, TensorShape


# ===----------------------------------------------------------------------=== #
# Converters
# ===----------------------------------------------------------------------=== #


def as_complex(real: Symbol, imag: Symbol) -> Symbol:
    return stack((real, imag), axis=-1)


def as_interleaved_complex(interleaved: Symbol) -> Symbol:
    """Reshape the input tensor as complex, interpreting the last dimension
    as being alternating (real, imag) pairs."""
    var g = interleaved.graph()
    let interleaved_t = interleaved.tensor_type()
    let last_d = interleaved_t.rank() - 1

    let shape = shape_of(interleaved)
    let back_dims = g.constant(Tensor[DType.int64](TensorShape(2), -1, 2))
    let new_shape = concat((shape[:last_d], back_dims))

    var new_dims = interleaved_t.dims
    new_dims[last_d] = dyn() if (new_dims[last_d] == dyn()) else (
        new_dims[last_d] // 2
    )
    new_dims.append(2)

    return reshape(interleaved, new_shape, new_dims)


def as_real(complex: Symbol) -> SymbolTuple:
    let splits = split[2](complex, (1, 1), axis=-1)
    let real = splits[0]
    let imag = splits[1]
    return (squeeze(real, axis=-1), squeeze(imag, axis=-1))


# ===----------------------------------------------------------------------=== #
# Ops
# ===----------------------------------------------------------------------=== #


def mul_complex(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Multiply complex-like real valued tensors."""
    let lhs_pair = as_real(lhs)
    let rhs_pair = as_real(lhs)
    let l_real = lhs_pair.get[0, Symbol]()
    let l_imag = lhs_pair.get[1, Symbol]()
    let r_real = rhs_pair.get[0, Symbol]()
    let r_imag = rhs_pair.get[1, Symbol]()

    let out_real = (l_real * r_real) - (l_imag * r_imag)
    let out_imag = (l_real * r_imag) + (l_imag * r_real)
    return as_complex(out_real, out_imag)
