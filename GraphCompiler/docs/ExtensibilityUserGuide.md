# Kernel Extensibility API User Guide

This document covers the Extensibility APIs for two main types of users: kernel
writers who want to plug their kernels into the graph compiler and graph-api
users who want to invoke the resulting functions.

## Registering a New Kernel

An example can be found in the `BottomK` struct in
`max/kernels/src/graph_compiler/builtin_kernels/reductions.mojo`.

```mojo
@extensibility.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=DType.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContext,
    ) raises:
        top_k[largest=False, target=target](
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[DType.int64](),
            indices.to_tile_tensor[DType.int64](),
            sorted,
            ctx,
        )


@extensibility.register_shape_function("mo.bottom_k")
def bottom_k_shape(
    input: Some[TileTensorable],
    k: Scalar,
    axis: Scalar,
    sorted: Scalar[.bool],
) raises -> IndexList[type_of(input).rank]:
    return rebind[IndexList[type_of(input).rank]](
        top_k_shape_impl(input.to_tile_tensor(), Int(k), Int(axis))
    )
```

We go over the most important components and what they mean:

### Registering a Function

The key is one operation is organized under a singular struct:

```mojo
@extensibility.register("mo.bottom_k")
struct BottomK:
```

This indicates the resulting methods in the struct correspond to the
mo-operation “mo.bottom_k”. If we instead we did
`@extensibility.register("my_cool_op")` , then this would correspond to call
to a custom op with symbol “my_cool_op.”

### Required Methods

Right now we have two supported functions (plus one special thing for PT):

```mojo
def execute()
@extensibility.register_shape_function("mo.bottom_k")
def bottom_k_shape() raises -> IndexList[rank]
```

Execute implements the kernel body, while the shape function is used by the
compiler to calculate the dynamic shape of outputs. The shape function is a
standalone function, not a method of the struct.

Note the **shape function** is only needed if the compiler cannot otherwise
figure out the parametric shapes.

E.g. for the graph-api folks if at the end of the compilation pipeline for MO
you have:

```python
mo.foo<() -> A, B, C>(...) ... -> !mo.tensor<[A, B, C], f32>
```

This needs a registered shape function to run to define “A”, “B”, and
“C”. Note:

```python
mo.foo(...) ... -> !mo.tensor<[A, B, C], f32>
```

does not require a shape function as in this notation, “A”, “B”, and “C” are
defined elsewhere.

### **Operand Types Supported**

- We support **ManagedTensorSlice** as a first class citizen.
  - Since both output and input tensors are passed as arguments to the execute
    function we distinguish which is which with parameters on the
    ManagedTensorSlice struct
    - These are encoded with the **IOSpec** parameter
    - Rather than setting these directly users are encouraged to use three
      convenient aliases: **OutputTensor**, **InputTensor**, and
      **MutableInputTensor**

- Variadic tensors can be represented via
**VariadicTensors[type, rank, size]**.
  - As with **ManagedTensorSlice** users must provide a specialization of its
    io_spec parameter, but are encouraged to use the **InputVariadicTensors**
    and **OutputVariadicTensors** aliases instead
  - We don’t support other variadic types besides Tensor
- Dynamic lists of tensors are supported via
**List[ManagedTensorSlice[dtype, rank]]** though support is
limited at this time.
  - We don’t support lists of other types besides Tensor
  - We only support lists of inputs at this time

Some special operand structs exist which should go at the end of the operand
list:

- `DeviceContext`
  - If specified, the graph compiler will give you access to the device
    context, which is needed for GPU support at the moment
  - If you don’t need this, you don’t have to include this
- `DeviceContextArray[N]`
  - For multi-device ops, declare the context argument as
    `DeviceContextArray[N]`. The graph compiler synthesizes the array from
    the per-device contexts attached to the operation, with one
    `DeviceContext` per device in launch order. Index with
    `dev_ctxs[i]` at runtime or `dev_ctxs.__getitem_param__[i]()` at
    compile time.
- `Scalar`
  - This is just syntax sugar over **ManagedTensorSlice’s** of one value
  - At the moment, this just dereferences the associated pointer, other
    implementations are open in the future (e.g. don't make a whole tensor for
    one value).
  - The tensor must be on host

**Any other structs will be done via opaque-type support** which should be as
fully featured as the old API. We support both register and memory-only types.

**Example:**

```python
@extensibility.register("myOp")
struct MyOp:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
      # Variadic output support.
      # It is assumed the size of the tuple can be inferred from the call site
        output: OutputVariadicTensors[dtype, rank, *_],

        # First class citizen
        input: InputTensor[dtype=output.dtype, rank=output.rank],

        mutable_input: MutableInputTensor[dtype=output.dtype, rank=output.rank]

        # Handled via opaque type support
        cache: MyCacheStruct,

        # Syntax sugar over index_tensor[0]
        index: Scalar,

        # Transparently added by runtime to the call as needed
        # (Optional if you don't need it)
        ctx: DeviceContext,
    ) raises:
     ...
```

### Shape function operands

Shape functions run on the host and must declare tensor arguments with tensor
traits rather than `ManagedTensorSlice` or its aliases:

- `Some[Tensor]` when the function reads only shapes.
- `Some[TileTensorable]` when the function reads element values or calls a
  helper that takes a tile tensor, as `bottom_k_shape` does above.
- `TensorTuple` for variadic inputs.

These arguments carry no IO-role, fusion, access, or stable marker. For
example, `convert_e4m3fn_to_e4m3fnuz_shape` in
`max/kernels/src/graph_compiler/builtin_kernels/conv.mojo` reads only the input
shape:

```mojo
@extensibility.register_shape_function("mo.convert_e4m3fn_to_e4m3fnuz")
def convert_e4m3fn_to_e4m3fnuz_shape(
    input: Some[Tensor],
) -> IndexList[type_of(input).rank]:
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )
```

### **Parameters Supported**

- We assume all included parameters can be inferred from the input arguments.
- If a parameter cannot be inferred from the input arguments solely, there are
many workarounds so please contact us.
- We do not support parameterized opaque types at the moment (both in the API
and in our stack).

We have a few “special parameters” the compiler always knows how to handle:

- **target: StaticString** which might be “cpu” or “gpu” and can be used for
specializing code for devices. This corresponds to the **device annotated on
the associated MO op.** `mo.add(...) {device = **gpu:0**}`
- **synchronous: Bool** if the graph compiler chooses to run this kernel
synchronously, this will be true. The heuristic meaning behind this being true
is “there is not a lot of computation.” This is often used to dispatch to a
simpler implementation of the kernel (e.g. not parallelized).
- These special parameters are optional if not needed.

Any additional parameters are specially handled by the graph compiler.

### **Compile-Time Tensor Information**

In order to access static information about tensors (e.g. shape) from the graph
compiler we use the `static_spec` parameter of the `ManagedTensorSlice`.
The `ManagedTensorSlice` provides aliases for the commonly accessed compile-time
information of each tensor.

```python
def execute[
    target: StaticString
](
    output: OutputTensor,
    input: InputTensor[dtype=output.dtype, rank=output.rank],
    ctx: DeviceContext,
) raises:
    alias static_shape = input._static_shape
    alias static_strides = input._static_strides
    alias alignment = input.alignment
    alias exclusive = input.exclusive
```

The `static_spec` parameter has the type `StaticTensorSpec[dtype, rank]`. In
addition to the information above, the `StaticTensorSpec` also contains the
fusion lambdas for input/output fusion, which should not be used directly.

In general, the methods on `ManagedTensorSlice` should be preferred when
possible. For shape and strides, `ManagedTensorSlice::shape` and
`ManagedTensorSlice::strides` will do the correct thing. The `_static_shape` and
`_static_strides` aliases should only be used when in a parameter context.

The `_fused_load` and `_fused_store` methods should be preferred over directly
accessing the fusion lambdas when loading or storing to a tensor.

### DPS Operands

```python
@extensibility.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        # An "output"
        values: OutputTensor[dtype=dtype, rank=rank],
        # Another "output" but specifying io_spec directly
        indices: ManagedTensorSlice[dtype = DType.int64, rank=rank, io_spec=Output],

        # Inputs below!
        input: InputTensor[dtype=dtype, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Bool,
    ):
```

All **OutputTensor** arguments must come
before any **InputTensor** or **MutableInputTensor** arguments

If an operation models side-effects on one of its input tensors then
the **MutableInputTensor** alias of **ManagedTensorSlice** should be used.

```python
@extensibility.register("mutating_operation")
struct TensorMutatingOperation:
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType, target: StaticString
    ](
        #These are all "inputs" to the graph-api operation
        mutable_input: MutableInputTensor[dtype=dtype, rank=6],
        input: InputTensor[dtype=dtype, rank=4]
    ):
     # Since this kernel takes a MutableInputTensor as an input
     # we want chains on the graph-api side to reason about side-effects
     # correctly.
     pass
```

### **Value Outputs**

We do not need to return things in a DPS style. We can do regular returns too
and things should work as you expect:

```python
@extensibility.register(
    "paged_kv_cache_collection_h8_d128_bshd"
)
struct Struct_paged_kv_cache_collection_h8_d128_bshd:
    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType, target: StaticString
    ](
        blocks: InputTensor[dtype, 6],
        cache_lengths: InputTensor[DType.uint32, 1],
        lookup_table: InputTensor[DType.uint32, 2],
        max_prompt_length: InputTensor[DType.uint32, 1],
        max_cache_length: InputTensor[DType.uint32, 1],
    **) -> PagedKVCacheCollection[
        dtype,
        kv_params_h8_d128_bshd,
        page_size_128,
    ]:**
        return generic_get_paged_cache(
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
        )

@extensibility.register("returning_transpose")
struct ReturningTranspose:
    @inline(.always)
    @staticmethod
    def transpose_in_place(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) -> **ManagedTensorSlice[dtype = input.dtype, rank = input.rank]:**
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        comptime for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return ManagedTensorSlice[dtype = input.dtype, rank = input.rank](
            input._ptr, new_shape, new_stride
        )
```

Note, one thing we are lacking is the ability to create **ManagedTensorSlice**
which point to **newly allocated memory**. **You can do it, but it won’t be
properly freed at this time and leak** (probably, i haven’t tried). Support for
this will come later if needed.

### Special Decorators

There are some special decorators like `@compiler.elementwise` which are used to
control optimizations by the graph compiler. These are undocumented for now,
though if you think you need one of these, let us know.

### Fusion

You can access the input/output lambdas via `static_spec` parameter.

### Converting Between Tensor Types

A `ManagedTensorSlice` can be converted to the tensor types used by the
underlying kernels via its `to_tile_tensor` method,
defined in
`max/kernels/src/graph_compiler/extensibility/managed_tensor_slice.mojo`.
Plenty of examples of calling these can be found throughout the kernel
registrations in `max/kernels/src/graph_compiler/builtin_kernels/`.

## Calling a New Kernel in the Graph API

### Mapping MO Calls To The Registered Kernel

The new API kernels can be called via custom ops. E.g.

```python
@extensibility.register("MyOp")
struct MyOp:
 pass
```

Can be called via `mo.custom [symbol = "MyOp"]` .

The remaining useful knowledge is how to properly invoke this custom op to
match with the registered kernel. Here an example is prudent:

```llvm
# 3 SSA outputs, 3 SSA Inputs
%dps0, %dps1, %out_value = mo.custom [symbol = "MyOp"](%in1, %in2, %in3)
 : (!mo.tensor, !mo.tensor, !mo.opaque<"OutputTy">)
   -> (!mo.tensor, !mo.tensor, !mo.tensor)
```

This maps to:

```python
@extensibility.register("MyOp")
struct MyOp:
    @staticmethod
    def execute(
     dps0: OutputTensor,
     dps1: OutputTensor,
     in1: InputTensor,
     in2: InputTensor,
     in3: InputTensor,
   ) -> OutputTy:
    ...
   var out_value = ...
   return out_value
```

AKA: the dps outputs come “first”, followed by the return by value outputs.

To see support for variadic returns (of which we only support
ManagedTensorSlice): see “mo.split” and “mo.concat” implementations.

### Ordering And Side Effects

Say we want to mutate the 2nd input of MyOp the signature would now look like:

```python
@extensibility.register("MyOp")
struct MyOp:
    @staticmethod
    def execute(
     dps0: OutputTensor,
     dps1: OutputTensor,
     in1: InputTensor,
     in2: MutableInputTensor,
     in3: InputTensor,
   ) -> OutputTy:
    ...
   var out_value = ...
   return out_value
```

Chains should be used in the graph api level to order side-effects in places
where existing data dependencies are not sufficient (e.g. an op with no
“outputs” or it actually mutates something).

In terms of mapping chains to the kernel registration, just pretend they don’t
exist. Chains should go at the end of the operand/result lists in the graph api
however.

```llvm
%chain_in: !mo.chain
%chain_out: !mo.chain
%dps0, %dps1, %out_value, **%chain_out** =
 mo.custom [symbol = "MyOp"](%in1, %in2, %in3, **%chain_in**)
 : (!mo.tensor, !mo.tensor, !mo.opaque<"OutputTy">, !mo.chain)
   -> (!mo.tensor, !mo.buffer, !mo.tensor, !mo.chain)
```
