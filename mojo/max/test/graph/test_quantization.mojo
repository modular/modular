# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo "%s"

import sys
from memory import memset_zero

from max.graph import Graph, TensorType, Type
from max.graph.quantization import BlockQ40, BFloat16Encoding, Q4_0Encoding
from max.graph._testing import (
    assert_tensors_almost_equal,
    assert_tensors_equal,
    execute_unary,
)
from max.tensor import Tensor, TensorShape


def test_quantize_bfloat16():
    @parameter
    if sys.has_neon():
        # TODO(KERN-228): Enable once LLVM bfloat16 emulation matures on ARM.
        return

    # Define hyperparameters.
    alias num_tokens = 32
    alias channels = 16

    # Initialize graph.
    g = Graph(
        "bfloat16_quantization",
        in_types=List[Type](TensorType(DType.uint8, num_tokens, 2 * channels)),
        out_types=List[Type](TensorType(DType.uint8, num_tokens, 2 * channels)),
    )

    # Generate normally-distributed random tensor to quantize.
    f32_tensor = Tensor[DType.float32].randn(TensorShape(num_tokens, channels))

    # Quantize the float32 token embeddings to bfloat16.
    bfloat16_symbol = g.quantize[BFloat16Encoding](f32_tensor)

    # Add zeros with the weights as a "pseudo identity" op.
    # Otherwise the API/runtime returns an invalid stack address as output.
    # TODO(GRA-498): Fix issue returning mgp.buffer.constant directly.
    g.output(bfloat16_symbol + g[0])

    zeros = Tensor[DType.uint8](TensorShape(num_tokens, 2 * channels))
    memset_zero(zeros.unsafe_ptr(), zeros.num_elements())

    output = execute_unary(g, zeros)

    # Bitcast output to bfloat16 then cast to float32 to verify.
    output = Tensor(
        TensorShape(num_tokens, channels),
        output._steal_ptr().bitcast[DType.bfloat16](),
    ).astype[DType.float32]()

    # Set rtol based on bfloat16 resolution: 0.01.
    assert_tensors_almost_equal(f32_tensor, output, rtol=1e-2)


def test_quantize_q4_0():
    # Define hyperparameters.
    alias num_rows = 1

    # Initialize graph.
    g = Graph(
        "q4_0_quantization",
        in_types=List[Type](
            TensorType(DType.uint8, num_rows, 2 * sizeof[BlockQ40]())
        ),
        out_types=List[Type](
            TensorType(DType.uint8, num_rows, 2 * sizeof[BlockQ40]())
        ),
    )

    # fmt: off
    f32_tensor = Tensor[DType.float32](
        # Two blocks of row 0 of Llama's `token_embd` parameter.
        TensorShape(num_rows, 2 * BlockQ40.elements_per_block()),
        # Block 0 float32 weights.
        1.2293457984924316e-06, -1.8179416656494141e-06, -4.3511390686035156e-06, 8.0466270446777344e-06,
        1.9222497940063477e-06, -5.6028366088867188e-06, 3.0845403671264648e-06, 1.1995434761047363e-06,
        -6.8247318267822266e-06, -1.6763806343078613e-06, -4.4703483581542969e-06, -4.3809413909912109e-06,
        -7.3388218879699707e-07, -8.4638595581054688e-06, 2.1457672119140625e-06, 1.0251998901367188e-05,
        -4.8801302909851074e-07, -1.5273690223693848e-06, 1.6242265701293945e-06, 1.1324882507324219e-06,
        2.8759241104125977e-06, 9.4771385192871094e-06, 3.3080577850341797e-06, -2.8014183044433594e-06,
        -1.2874603271484375e-05, -2.816319465637207e-06, 5.6326389312744141e-06, -1.1175870895385742e-06,
        -3.3676624298095703e-06, -3.0100345611572266e-06, -2.1886080503463745e-07, 1.4156103134155273e-06,
        # Block 1 float32 weights.
        9.1791152954101562e-06, 2.5779008865356445e-06, 1.9669532775878906e-06, 9.8347663879394531e-07,
        -1.1086463928222656e-05, -5.7220458984375e-06, 3.9637088775634766e-06, -1.1026859283447266e-05,
        7.2121620178222656e-06, 1.8477439880371094e-06, 4.5895576477050781e-06, 2.1904706954956055e-06,
        1.3113021850585938e-06, -2.86102294921875e-06, -1.4841556549072266e-05, -6.4671039581298828e-06,
        2.6524066925048828e-06, 6.7055225372314453e-06, -2.6524066925048828e-06, 8.3446502685546875e-06,
        1.5869736671447754e-06, -1.3053417205810547e-05, 4.6193599700927734e-06, 7.8082084655761719e-06,
        -5.5730342864990234e-06, -6.3180923461914062e-06, -3.7811696529388428e-07, 9.2387199401855469e-06,
        5.3644180297851562e-06, -3.9637088775634766e-06, -2.4437904357910156e-06, -4.6193599700927734e-06,
    )
    expected_output = Tensor[DType.uint8](
        TensorShape(num_rows, 2 * sizeof[BlockQ40]()),
        # Block 0 float16 scale.
        27, 0,
        # Block 0 packed nibble weights.
        137, 119, 149, 157, 169, 229, 170, 105, 4, 103, 197, 117, 104, 99, 137, 158,
        # Block 1 float16 scale.
        31, 0,
        # Block 1 packed nibble weights.
        157, 201, 121, 201, 146, 21, 170, 194, 92, 89, 138, 217, 185, 102, 112, 101,
    )
    # fmt: on

    # Quantize the float32 token embeddings to q4_0.
    q4_0_symbol = g.quantize[Q4_0Encoding](f32_tensor)

    # TODO(GRA-498): Remove pseudo-identity once GRA-498 is fixed.
    g.output(q4_0_symbol + g[0])

    zeros = Tensor[DType.uint8](TensorShape(num_rows, 2 * sizeof[BlockQ40]()))
    memset_zero(zeros.unsafe_ptr(), zeros.num_elements())

    actual_output = execute_unary[outtype = DType.uint8](g, zeros)

    assert_tensors_equal(actual_output, expected_output)


def main():
    test_quantize_bfloat16()
    test_quantize_q4_0()
