# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# These operations assume that all tensors have last dim == 2, representing real
# and imaginary parts.


from tensor import Tensor, TensorShape
from max.graph.type import Dim


# ===----------------------------------------------------------------------=== #
# Converters
# ===----------------------------------------------------------------------=== #


def as_complex(real: Symbol, imag: Symbol) -> Symbol:
    return stack((real, imag), axis=-1)


def as_interleaved_complex(interleaved: Symbol) -> Symbol:
    """Reshape the input tensor as complex, interpreting the last dimension
    as being alternating (real, imag) pairs."""
    var g = interleaved.graph()
    var interleaved_t = interleaved.tensor_type()
    var last_d = interleaved_t.rank() - 1

    var shape = shape_of(interleaved)
    var back_dims = g.constant(Tensor[DType.int64](TensorShape(2), -1, 2))
    var new_shape = concat((shape[:last_d], back_dims))

    var new_dims = interleaved_t.dims
    var last_dim = new_dims[last_d]
    new_dims[last_d] = Dim.static(
        last_dim.num_elements() // 2
    ) if last_dim.is_static() else Dim.dynamic()
    new_dims.append(2)

    return reshape(interleaved, new_shape, new_dims)


def as_real(complex: Symbol) -> SymbolTuple:
    var splits = split[2](complex, (1, 1), axis=-1)
    var real = splits[0]
    var imag = splits[1]
    return (squeeze(real, axis=-1), squeeze(imag, axis=-1))


# ===----------------------------------------------------------------------=== #
# Ops
# ===----------------------------------------------------------------------=== #


def mul_complex(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Multiply complex-like real valued tensors."""
    var lhs_pair = as_real(lhs)
    var rhs_pair = as_real(lhs)
    var l_real = lhs_pair[0]
    var l_imag = lhs_pair[1]
    var r_real = rhs_pair[0]
    var r_imag = rhs_pair[1]

    var out_real = (l_real * r_real) - (l_imag * r_imag)
    var out_imag = (l_real * r_imag) + (l_imag * r_real)
    return as_complex(out_real, out_imag)
