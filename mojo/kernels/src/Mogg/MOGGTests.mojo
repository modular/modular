# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import vectorize
from algorithm.functional import elementwise
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from register import *
from runtime.asyncrt import MojoCallContextPtr

from utils import StaticTuple, unroll
from utils.index import IndexList

# ===----------------------------------------------------------------------===#
# Special test targets just for generation tests
# ===----------------------------------------------------------------------===#


@register_internal_override("mo.concat", 1)
@always_inline
fn test_concat_dim_0_inputs_lambda_tuple[
    type: DType,
    rank: Int,
    # TODO: Do we need two flags? one for input, one for output?
    lambdas_have_fusion: Bool,
    # A tuple of callbacks, one per input.
    input_1_fn_tuple: StaticTuple[
        fn[
            width: Int, rank: Int
        ] (IndexList[rank]) capturing -> SIMD[type, width], *_
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
](
    # TODO: We should switch the variadic inputs to
    # `StaticTuple[IndexList[rank], inputs_fns.size]`. Graph compiler does
    # not know how to handle a tuple of shapes currently.
    axis: Scalar,
    input_shapes: StaticTuple[IndexList[rank], input_1_fn_tuple.size],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr,
) raises:
    print("In override concat.")
    var offset = 0

    @parameter
    for i in range(input_1_fn_tuple.size):
        alias input_i_fn = input_1_fn_tuple[i]
        var input_shape = input_shapes[i]

        @parameter
        @always_inline
        fn elementwise_wrapper[width: Int, rank: Int](indices: IndexList[rank]):
            # get data from input through lambda.
            var value = input_i_fn[width, rank](indices)
            var c = IndexList[rank]()

            # Transform the indices
            @parameter
            for i in range(rank):
                c[i] = indices[i] + (offset if i == 0 else 0)

            # call the output lambda
            output_0_fn[width, rank, 1](c, value)

        elementwise[elementwise_wrapper, 1](input_shape, ctx)
        offset = offset + input_shape[0]


@register_internal("test_many_ranks_and_types")
fn test_many_ranks_and_types[
    type1: DType,
    rank1: Int,
    type2: DType,
    rank2: Int,
    type3: DType,
    rank3: Int,
    type4: DType,
    rank4: Int,
    type5: DType,
    rank5: Int,
](
    tensor1: NDBuffer[type1, rank1],
    tensor2: NDBuffer[type2, rank2],
    tensor3: NDBuffer[type3, rank3],
    tensor4: NDBuffer[type4, rank4],
    tensor5: NDBuffer[type5, rank5],
) -> NDBuffer[type1, rank1]:
    """
    Used as a test target to ensure parameter deduction works when there are
    many to deduce and also used to check errors.
    """
    return tensor1


@register_internal("test_one_rank_many_tensor")
fn test_one_rank_many_tensor[
    type: DType, rank: Int
](
    tensor1: NDBuffer[type, rank],
    tensor2: NDBuffer[type, rank],
    tensor3: NDBuffer[type, rank],
    tensor4: NDBuffer[type, rank],
    tensor5: NDBuffer[type, rank],
) -> NDBuffer[type, rank]:
    """
    Used as a test target to ensure we can deduce type and rank when used by
    many arguments.
    """
    return tensor1


@register_internal("mutable_test_op")
fn mutable_test_op[
    type: DType,
    rank: Int,
](output: NDBuffer[type, rank], ctx: MojoCallContextPtr):
    """
    Used to test having a custom op that directly writes to a tensor in-place.
    """
    output.data.store(0, 0)


@register_internal("test_3D_in_out_lambda")
fn test_3D_in_out_lambda[
    type: DType,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
](tensor1: NDBuffer[type, 3], output: NDBuffer[type, 3],) -> NDBuffer[type, 3]:
    """
    Used as a target to test passing input and output lambdas.
    """

    for x in range(0, tensor1.dim[0]()):
        for y in range(0, tensor1.dim[1]()):

            @always_inline
            @parameter
            fn func_wrapper[simd_width: Int](idx: Int):
                var indices = IndexList[3](x, y, idx)
                var result = input_0_fn[simd_width, 3](indices)
                output_0_fn[simd_width, 3, element_alignment=1](indices, result)

            vectorize[func_wrapper, simd_width](tensor1.dim[2]())

    return output


@register_internal_override("mo.sqrt", 1)
@mogg_elementwise
fn sqrt_wrapped[
    type: DType, simd_width: Int
](value: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    print("In override sqrt")
    return value


@register_internal("test_static_shape_deduction")
fn test_static_shape_deduction[
    type: DType, rank: Int, input_0_static_shape: DimList
](tensor: NDBuffer[type, rank, input_0_static_shape],):
    print("Printing shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = input_0_static_shape.at[idx]()

        @parameter
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[body, rank]()


@register_internal("test_static_stride_deduction")
fn test_static_stride_deduction[
    type: DType,
    rank: Int,
    input_0_static_shape: DimList,
    input_0_static_stride: DimList,
](tensor: NDBuffer[type, rank, input_0_static_shape, input_0_static_stride],):
    print("Printing stride: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = input_0_static_stride.at[idx]()

        @parameter
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[body, rank]()


@register_internal("test_address_space_deduction")
fn test_address_space_deduction(tensor: NDBuffer):
    print("Printing address space: ")
    print("Address Space: " + str(tensor.address_space._value))


@register_internal("test_ndbuffer_exclusive_deduction")
fn test_ndbuffer_exclusive_deduction(tensor: NDBuffer):
    print("Printing exclusive flag: " + str(tensor.exclusive))


@register_internal("test_static_shape_output")
fn test_static_shape_output[
    type: DType, rank: Int, output_0_static_shape: DimList
]() -> NDBuffer[type, rank, output_0_static_shape]:
    print("Printing output shape: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = output_0_static_shape.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[body, rank]()
    return NDBuffer[type, rank, output_0_static_shape](
        UnsafePointer[Scalar[type]](),
        IndexList[rank](),
        IndexList[rank](),
    )


@register_internal("test_int_list_param")
fn test_int_list_param[length: Int, int_list: DimList]():
    print("Printing parameter: ")

    @always_inline
    @parameter
    fn body[idx: Int]():
        alias dim = int_list.at[idx]()
        if dim.is_dynamic():
            print("unknown")
        else:
            print(dim.get())

    unroll[body, length]()


@register_internal("test_custom_op")
@always_inline
fn test_unary_kernel[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](input_shape: IndexList[rank], output_shape: IndexList[rank],):
    print("World!")


@register_internal_shape_func("test_custom_op")
@always_inline
fn test_unary_kernel_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) -> IndexList[rank]:
    print("Hello")

    return data.get_shape()


@register_internal("test_custom_op_params")
@always_inline
fn test_unary_kernel_params[
    type: DType,
    rank: Int,
    extra_param: Int,
    extra_param2: StringLiteral,
    target: StringLiteral = "cpu",
](tensor1: NDBuffer[type, rank], output: NDBuffer[type, rank],):
    print(extra_param)
    print(extra_param2)


@register_internal("tf.Identity")
@register_internal("torch.aten.abs")
@register_internal("monnx.abs_v13")
@always_inline
fn test_custom_identity[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](input_shape: IndexList[rank], output_shape: IndexList[rank],):
    print("The custom identity op is running!")

    @parameter
    @always_inline
    fn identity[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var x = input_0_fn[simd_width, rank](idx)
        output_0_fn[simd_width, rank, element_alignment=1](idx, x)

    elementwise[
        identity,
        simd_width=simd_width,
        use_blocking_impl=single_thread_blocking_override,
        target="cpu",
    ](
        input_shape,
    )


@register_internal_shape_func("tf.Identity")
@register_internal_shape_func("torch.aten.abs")
@register_internal_shape_func("monnx.abs_v13")
@always_inline
fn test_custom_identity_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank]) -> IndexList[rank]:
    return data.get_shape()


@register_internal("test_variadic")
@always_inline
fn concat(
    ctx: MojoCallContextPtr,
    *variadic_ins: NDBuffer[DType.float32, 1],
):
    pass


@register_internal_shape_func("test_custom_op_inline")
fn reduce_shape_no_explicit_inline[
    input_rank: Int,
    input_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    axis_buf: NDBuffer[axis_type, 1],
) -> IndexList[input_rank]:
    # extract hyper parameter
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank

    # compute and return the output shape
    var output_shape = input_buf.get_shape()
    output_shape[axis] = 1
    return output_shape


@register_internal("custom_op_that_raises")
fn custom_op_that_raises[
    type: DType,
    rank: Int,
    simd_width: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_0_fn: fn[width: Int, rank: Int, element_alignment: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    single_thread_blocking_override: Bool,
](input_shape: IndexList[rank], output_shape: IndexList[rank]) raises:
    if input_shape[0] == 10:
        raise ("input_shape[0] == 10")

    @parameter
    @always_inline
    fn identity[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var x = input_0_fn[simd_width, rank](idx)
        output_0_fn[simd_width, rank, element_alignment=1](idx, x)

    elementwise[
        identity,
        simd_width=simd_width,
        use_blocking_impl=single_thread_blocking_override,
        target="cpu",
    ](
        input_shape,
    )


@register_internal_shape_func("custom_op_that_raises")
@always_inline
fn custom_shape_func_that_raises[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](data: NDBuffer[type, rank],) raises -> IndexList[rank]:
    # This print ensures we won't symbolicize this shape function call, so we
    # can test its runtime execution.
    print("Hello")

    var out_shape = data.get_shape()
    if out_shape[0] == 20:
        raise ("data.get_shape()[0] == 20")
    return out_shape


@register_internal("relative_rank_deduction")
@always_inline
fn relative_rank_deduction[
    type: DType,
    rank: Int,
](data: NDBuffer[type, rank], data2: NDBuffer[type, rank + 1]):
    pass


@register_internal("get_index")
fn get_index() -> Int:
    return 1


@register_internal("print_index")
fn print_index(i: Int):
    print("index = ", i)


@register_internal("print_indices")
fn print_indices[rank: Int](i: IndexList[rank]):
    print(i)


@register_internal("test_type_parameter_deduction")
fn test_type_parameter_deduction[
    A: AnyTrivialRegType, B: AnyTrivialRegType
](arg0: A, arg1: B) -> A:
    return arg0


@register_internal("print_tensor_test")
fn print_tensor_test[type: DType, rank: Int](buffer: NDBuffer[type, rank]):
    print("Rank:", rank)
    print("Shape:", buffer.get_shape())
    for i in range(buffer.num_elements()):
        print(buffer.data.load(i))


@register_internal("print_tensor_shape")
fn print_tensor_shape[type: DType, rank: Int](buffer: NDBuffer[type, rank]):
    print("Rank:", rank)
    print("Shape:", buffer.get_shape())
    print("Stride:", buffer.get_strides())


struct MyCustomInt(Movable):
    var val: Int

    @implicit
    fn __init__(out self, val: Int):
        self.val = val

    fn __moveinit__(out self, owned other: MyCustomInt):
        self.val = other.val


@register_internal("test_make_custom_int")
@no_inline
fn test_make_custom_int() -> MyCustomInt:
    return MyCustomInt(42)


@register_internal("basic_target")
fn basic_target[
    type: DType, rank: Int, target: StringLiteral = "cpu"
](x: NDBuffer[type, rank], out: NDBuffer[type, rank]):
    print("hello from kernel on", target)


struct MyCustomSIMD[type: DType, len: Int](Movable):
    var val: SIMD[type, len]

    @implicit
    fn __init__(out self, val: Int):
        self.val = val

    fn __moveinit__(out self, owned other: Self):
        self.val = other.val


# For testing support for Scalar[...] in Mojo
@register_internal("supports_scalar_kernel")
fn supports_scalar_kernel[
    type: DType, target: StringLiteral
](x: NDBuffer[type, 1], y: Scalar[type], out: NDBuffer[type, 1]):
    print("datatype is", type)
    out[0] = y


struct MyCustomScalar[type: DType](Movable):
    var val: Scalar[type]

    @implicit
    fn __init__(out self, val: Scalar[type]):
        print("MyCustomScalar.__init__", val)
        self.val = val

    fn __moveinit__(out self, owned other: Self):
        print("MyCustomScalar.__moveinit__", other.val)
        self.val = other.val

    fn __del__(owned self):
        print("MyCustomScalar.__del__", self.val)


@register_internal("tensor_to_my_custom_scalar_ndbuff")
fn tensor_to_my_custom_scalar_ndbuff[
    type: DType
](x: NDBuffer[type, 1]) -> MyCustomScalar[type]:
    var val = x.data[0]
    return MyCustomScalar(val)


@register_internal("scale_with_my_custom_scalar_ndbuff")
fn scale_with_my_custom_scalar_ndbuff[
    type: DType, rank: Int
](
    x: NDBuffer[type, rank],
    scale: MyCustomScalar[type],
    out: NDBuffer[type, rank],
):
    @parameter
    @always_inline
    fn func[simd_width: Int, fn_rank: Int](idx: IndexList[fn_rank]):
        var val = x.load[width=simd_width](
            rebind[IndexList[rank]](idx)
        ) * scale.val
        out.store[width=simd_width](rebind[IndexList[rank]](idx), val)

    elementwise[func, simd_width=1, use_blocking_impl=True, target="cpu"](
        x.get_shape(),
    )


@register_internal_shape_func("scale_with_my_custom_scalar_ndbuff")
fn scale_with_my_custom_scalar_ndbuff_shape_func[
    type: DType, rank: Int, single_thread_blocking_override: Bool
](x: NDBuffer[type, rank], scale: MyCustomScalar[type],) -> IndexList[rank]:
    return x.get_shape()


# Invalid kernel: owned custom types not supported
@register_internal("invalid_kernel_owned_arg")
fn invalid_kernel_owned_arg(
    owned x: MyCustomScalar[DType.int64],
) -> MyCustomScalar[DType.int64]:
    return MyCustomScalar(x.val)


@value
@register_passable
struct MyCustomScalarReg[type: DType]:
    var val: Scalar[type]

    @implicit
    fn __init__(out self, val: Scalar[type]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarReg.__del__", self.val)


@register_internal("buff_to_my_custom_scalar_reg")
fn buff_to_my_custom_scalar_reg[
    type: DType
](x: NDBuffer[type, 1]) -> MyCustomScalarReg[type]:
    return MyCustomScalarReg(x.data[0])


@register_internal("my_custom_scalar_reg_to_buff")
fn my_custom_scalar_reg_to_buff[
    type: DType
](x: MyCustomScalar[type], out: NDBuffer[type, 1]):
    out.data[0] = x.val


@register_internal("kernel_with_no_target")
fn kernel_with_no_target[
    type: DType, rank: Int
](x: NDBuffer[type, rank], out: NDBuffer[type, rank]):
    print("hello from kernel with no target")
