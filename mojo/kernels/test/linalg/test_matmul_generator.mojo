# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from Matmul import GemmShape, MatmulOperandLayout
from MatmulGenerator import (
    GemmIdentifiers,
    MatmulAction,
    MatmulActionKind,
    MatmulDataType,
    MatmulDynamicState,
    MatmulGenerator,
    MatmulStaticState,
)


# CHECK-LABEL: test_tile_gen
fn test_tile_gen():
    print("== test_tile_gen")

    alias data_type = MatmulDataType {
        value_type: DType.float32, accum_type: DType.float32
    }

    var dynamic_state = MatmulDynamicState[data_type]()
    dynamic_state.global_offset = GemmShape(0, 0, 0)
    dynamic_state.valid_tile_bound = GemmShape(10, 11, 12)

    # CHECK: global offset: (0, 0, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 0, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 0, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 3, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 3, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 3, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 6, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 6, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 6, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (0, 9, 0)
    # CHECK: dynamic tile bound: (4, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (0, 9, 5)
    # CHECK: dynamic tile bound: (4, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (0, 9, 10)
    # CHECK: dynamic tile bound: (4, 2, 2)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (4, 0, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 0, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 0, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 3, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 3, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 3, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 6, 0)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 6, 5)
    # CHECK: dynamic tile bound: (4, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 6, 10)
    # CHECK: dynamic tile bound: (4, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (4, 9, 0)
    # CHECK: dynamic tile bound: (4, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (4, 9, 5)
    # CHECK: dynamic tile bound: (4, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (4, 9, 10)
    # CHECK: dynamic tile bound: (4, 2, 2)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (8, 0, 0)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 0, 5)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 0, 10)
    # CHECK: dynamic tile bound: (2, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 3, 0)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 3, 5)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 3, 10)
    # CHECK: dynamic tile bound: (2, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 6, 0)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 6, 5)
    # CHECK: dynamic tile bound: (2, 3, 5)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 6, 10)
    # CHECK: dynamic tile bound: (2, 3, 2)
    # CHECK: static tile bound: (-1, 3, -1)

    # CHECK: global offset: (8, 9, 0)
    # CHECK: dynamic tile bound: (2, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (8, 9, 5)
    # CHECK: dynamic tile bound: (2, 2, 5)
    # CHECK: static tile bound: (-1, 2, -1)

    # CHECK: global offset: (8, 9, 10)
    # CHECK: dynamic tile bound: (2, 2, 2)
    # CHECK: static tile bound: (-1, 2, -1)

    # Define an no-op epilog function.
    @always_inline
    @parameter
    fn epilog_no_op(dynamic_state: MatmulDynamicState[data_type]):
        return

    MatmulGenerator[data_type, epilog_no_op].generate[
        MatmulStaticState.initialize[
            data_type,
            MatmulOperandLayout {
                transpose_a: False,
                transpose_b: False,
                a_packed: False,
                b_packed: False,
                pack_a_inner_size: 1,
                pack_b_inner_size: 1,
            },
        ](),
        VariadicList[MatmulAction](
            # Tile on M Dimension.
            MatmulAction {
                kind: MatmulActionKind.TileDynamic,
                tile_sizes: VariadicList[Int](),
                tiled_dimension: GemmIdentifiers.DimM,
                do_epilog: False,
            },
            # Tile on N Dimension.
            MatmulAction {
                kind: MatmulActionKind.TileStatic,
                tile_sizes: VariadicList[Int](3, 2, 1),
                tiled_dimension: GemmIdentifiers.DimN,
                do_epilog: False,
            },
            # Tile on K Dimension.
            MatmulAction {
                kind: MatmulActionKind.TileAndUnswitch,
                tile_sizes: VariadicList[Int](),
                tiled_dimension: GemmIdentifiers.DimK,
                do_epilog: False,
            },
            # Print the tile state.
            MatmulAction {
                kind: MatmulActionKind.PrintTileState,
                tile_sizes: VariadicList[Int](),  # unused
                tiled_dimension: 0,  # unused
                do_epilog: False,
            },
        ),
    ](
        dynamic_state,
        VariadicList[VariadicList[Int]](
            VariadicList[Int](4, 3, 2),  # M - dynamic
            VariadicList[Int](),  # N - static
            VariadicList[Int](5, 3),  # K - dynamic
            VariadicList[Int](),
        ),
    )


fn main():
    test_tile_gen()
