# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.tile import tile

from utils import StaticIntTuple


# CHECK-LABEL: test_tile_eg1
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg1() raises:
    print("== test_tile_eg1")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 0
    input[StaticIntTuple[rank](0, 1)] = 1
    input[StaticIntTuple[rank](1, 0)] = 2
    input[StaticIntTuple[rank](1, 1)] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 4),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg2
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg2() raises:
    print("== test_tile_eg2")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 0
    input[StaticIntTuple[rank](0, 1)] = 1
    input[StaticIntTuple[rank](1, 0)] = 2
    input[StaticIntTuple[rank](1, 1)] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 3
    repeats[StaticIntTuple[rank_repeats](1)] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(6, 4),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(6):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg3
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg3() raises:
    print("== test_tile_eg3")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 0
    input[StaticIntTuple[rank](0, 1)] = 1
    input[StaticIntTuple[rank](1, 0)] = 2
    input[StaticIntTuple[rank](1, 1)] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 6),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(6):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg4
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
fn test_tile_eg4() raises:
    print("== test_tile_eg4")
    alias rank = 3
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 1, 1)] = 3

    input[StaticIntTuple[rank](1, 0, 0)] = 4
    input[StaticIntTuple[rank](1, 0, 1)] = 5
    input[StaticIntTuple[rank](1, 1, 0)] = 6
    input[StaticIntTuple[rank](1, 1, 1)] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(3),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 1
    repeats[StaticIntTuple[rank_repeats](2)] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 2, 2),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg5
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,
fn test_tile_eg5() raises:
    print("== test_tile_eg5")
    alias rank = 3
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 1, 1)] = 3

    input[StaticIntTuple[rank](1, 0, 0)] = 4
    input[StaticIntTuple[rank](1, 0, 1)] = 5
    input[StaticIntTuple[rank](1, 1, 0)] = 6
    input[StaticIntTuple[rank](1, 1, 1)] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(3),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 1
    repeats[StaticIntTuple[rank_repeats](2)] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 2, 4),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(2):
            for k in range(4):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg6
# CHECK: 1.0 ,2.0 ,1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,3.0 ,4.0 ,
fn test_tile_eg6() raises:
    print("== test_tile_eg6")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 1
    input[StaticIntTuple[rank](0, 1)] = 2
    input[StaticIntTuple[rank](1, 0)] = 3
    input[StaticIntTuple[rank](1, 1)] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 1
    repeats[StaticIntTuple[rank_repeats](1)] = 2

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(2, 4),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(2):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg7
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,
# CHECK: 3.0 ,4.0 ,
fn test_tile_eg7() raises:
    print("== test_tile_eg7")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 1
    input[StaticIntTuple[rank](0, 1)] = 2
    input[StaticIntTuple[rank](1, 0)] = 3
    input[StaticIntTuple[rank](1, 1)] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 2),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(2):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg8
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
# CHECK: 1.0 ,2.0 ,3.0 ,4.0 ,
fn test_tile_eg8() raises:
    print("== test_tile_eg8")
    alias rank = 2
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(1, 4),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0)] = 1
    input[StaticIntTuple[rank](0, 1)] = 2
    input[StaticIntTuple[rank](0, 2)] = 3
    input[StaticIntTuple[rank](0, 3)] = 4

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(2),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 4
    repeats[StaticIntTuple[rank_repeats](1)] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 4),
    ].stack_allocation()

    for i in range(4):
        for j in range(4):
            output[StaticIntTuple[rank](i, j)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(4):
            print(output[i, j], ",", end="")
        print()
    print()


# CHECK-LABEL: test_tile_eg9
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
fn test_tile_eg9() raises:
    print("== test_tile_eg9")
    alias rank = 3
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 1, 1)] = 3

    input[StaticIntTuple[rank](1, 0, 0)] = 4
    input[StaticIntTuple[rank](1, 0, 1)] = 5
    input[StaticIntTuple[rank](1, 1, 0)] = 6
    input[StaticIntTuple[rank](1, 1, 1)] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(3),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 2
    repeats[StaticIntTuple[rank_repeats](2)] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 4, 2),
    ].stack_allocation()

    for i in range(4):
        for j in range(4):
            for k in range(2):
                output[StaticIntTuple[rank](i, j, k)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg10
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
fn test_tile_eg10() raises:
    print("== test_tile_eg10")
    alias rank = 3
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 1, 1)] = 3

    input[StaticIntTuple[rank](1, 0, 0)] = 4
    input[StaticIntTuple[rank](1, 0, 1)] = 5
    input[StaticIntTuple[rank](1, 1, 0)] = 6
    input[StaticIntTuple[rank](1, 1, 1)] = 7

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(3),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 3
    repeats[StaticIntTuple[rank_repeats](1)] = 2
    repeats[StaticIntTuple[rank_repeats](2)] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(6, 4, 6),
    ].stack_allocation()

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(6):
        for j in range(4):
            for k in range(6):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg11
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,
fn test_tile_eg11() raises:
    print("== test_tile_eg11")
    alias rank = 3
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(3, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 1, 1)] = 3

    input[StaticIntTuple[rank](1, 0, 0)] = 4
    input[StaticIntTuple[rank](1, 0, 1)] = 5
    input[StaticIntTuple[rank](1, 1, 0)] = 6
    input[StaticIntTuple[rank](1, 1, 1)] = 7

    input[StaticIntTuple[rank](2, 0, 0)] = 8
    input[StaticIntTuple[rank](2, 0, 1)] = 9
    input[StaticIntTuple[rank](2, 1, 0)] = 10
    input[StaticIntTuple[rank](2, 1, 1)] = 11

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(3),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 3
    repeats[StaticIntTuple[rank_repeats](2)] = 1

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(6, 6, 2),
    ].stack_allocation()

    for i in range(6):
        for j in range(6):
            for k in range(2):
                output[StaticIntTuple[rank](i, j, k)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(6):
        for j in range(6):
            for k in range(2):
                print(output[i, j, k], ",", end="")
            print()
        print()
    print()


# CHECK-LABEL: test_tile_eg12
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
fn test_tile_eg12() raises:
    print("== test_tile_eg12")
    alias rank = 4
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(1, 1, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 3

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(4),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 1
    repeats[StaticIntTuple[rank_repeats](1)] = 1
    repeats[StaticIntTuple[rank_repeats](2)] = 2
    repeats[StaticIntTuple[rank_repeats](3)] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(1, 1, 4, 6),
    ].stack_allocation()

    for i in range(1):
        for j in range(1):
            for k in range(4):
                for l in range(6):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(1):
        for j in range(1):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABE: test_tile_eg13
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
fn test_tile_eg13() raises:
    print("== test_tile_eg13")
    alias rank = 4
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 3

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 4
    input[StaticIntTuple[rank](0, 1, 0, 1)] = 5
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 6
    input[StaticIntTuple[rank](0, 1, 1, 1)] = 7

    input[StaticIntTuple[rank](1, 0, 0, 0)] = 8
    input[StaticIntTuple[rank](1, 0, 0, 1)] = 9
    input[StaticIntTuple[rank](1, 0, 1, 0)] = 10
    input[StaticIntTuple[rank](1, 0, 1, 1)] = 11

    input[StaticIntTuple[rank](1, 1, 0, 0)] = 12
    input[StaticIntTuple[rank](1, 1, 0, 1)] = 13
    input[StaticIntTuple[rank](1, 1, 1, 0)] = 14
    input[StaticIntTuple[rank](1, 1, 1, 1)] = 15

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(4),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 1
    repeats[StaticIntTuple[rank_repeats](1)] = 2
    repeats[StaticIntTuple[rank_repeats](2)] = 2
    repeats[StaticIntTuple[rank_repeats](3)] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(2, 4, 4, 6),
    ].stack_allocation()

    for i in range(2):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(2):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


# CHECK-LABE: test_tile_eg14
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,1.0 ,
# CHECK: 2.0 ,3.0 ,2.0 ,3.0 ,2.0 ,3.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 4.0 ,5.0 ,4.0 ,5.0 ,4.0 ,5.0 ,
# CHECK: 6.0 ,7.0 ,6.0 ,7.0 ,6.0 ,7.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 8.0 ,9.0 ,8.0 ,9.0 ,8.0 ,9.0 ,
# CHECK: 10.0 ,11.0 ,10.0 ,11.0 ,10.0 ,11.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
# CHECK: 12.0 ,13.0 ,12.0 ,13.0 ,12.0 ,13.0 ,
# CHECK: 14.0 ,15.0 ,14.0 ,15.0 ,14.0 ,15.0 ,
fn test_tile_eg14() raises:
    print("== test_tile_eg14")
    alias rank = 4
    alias type = DType.float32

    var input = NDBuffer[
        type,
        rank,
        DimList(2, 2, 2, 2),
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 3

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 4
    input[StaticIntTuple[rank](0, 1, 0, 1)] = 5
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 6
    input[StaticIntTuple[rank](0, 1, 1, 1)] = 7

    input[StaticIntTuple[rank](1, 0, 0, 0)] = 8
    input[StaticIntTuple[rank](1, 0, 0, 1)] = 9
    input[StaticIntTuple[rank](1, 0, 1, 0)] = 10
    input[StaticIntTuple[rank](1, 0, 1, 1)] = 11

    input[StaticIntTuple[rank](1, 1, 0, 0)] = 12
    input[StaticIntTuple[rank](1, 1, 0, 1)] = 13
    input[StaticIntTuple[rank](1, 1, 1, 0)] = 14
    input[StaticIntTuple[rank](1, 1, 1, 1)] = 15

    # rank_repeats is always 1
    alias rank_repeats = 1
    # type_repeats is always DType.int64
    alias type_repeats = DType.int64

    var repeats = NDBuffer[
        type_repeats,
        rank_repeats,
        DimList(4),
    ].stack_allocation()

    repeats[StaticIntTuple[rank_repeats](0)] = 2
    repeats[StaticIntTuple[rank_repeats](1)] = 2
    repeats[StaticIntTuple[rank_repeats](2)] = 2
    repeats[StaticIntTuple[rank_repeats](3)] = 3

    # Output rank = input rank
    # output_dim[i] = input_dim[i] * repeats[i]
    var output = NDBuffer[
        type,
        rank,
        DimList(4, 4, 4, 6),
    ].stack_allocation()

    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    tile[rank, type, rank_repeats, type_repeats](
        input.make_dims_unknown(),
        repeats.make_dims_unknown(),
        output.make_dims_unknown(),
    )

    print()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(6):
                    print(output[i, j, k, l], ",", end="")
                print()
            print()
        print()
    print()


fn main() raises:
    test_tile_eg1()
    test_tile_eg2()
    test_tile_eg3()
    test_tile_eg4()
    test_tile_eg5()
    test_tile_eg6()
    test_tile_eg7()
    test_tile_eg8()
    test_tile_eg9()
    test_tile_eg10()
    test_tile_eg11()
    test_tile_eg12()
    test_tile_eg13()
    test_tile_eg14()
