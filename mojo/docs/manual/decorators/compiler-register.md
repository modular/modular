---
title: '@compiler.register'
description: Registers a custom operation for use with the MAX Graph API.
codeTitle: true

---

The `@compiler.register` decorator registers a custom operation for use with the
Graph API. For more information on custom operations, see
[Intro to custom ops](/max/custom-ops).

To define a custom operation:

- Import the `compiler` package.

- Create a struct that implements the `execute()` and (optional) `shape()`
  static methods.

- Register it using the `@compiler.register` decorator.

The following snippet shows the outline of a custom operation:

```mojo
@compiler.register("add_vectors_custom")
struct AddVectorsCustom:

    @staticmethod
    fn execute[...](...) raises:
        pass

    @staticmethod
    fn shape(...) raises -> IndexList:
        pass
```

The `@compiler.register` decorator takes a single arguments, the name of the
custom operation, as a string. This name is used to load the custom op into your
graph.

Output from the `execute()` method is usually returned using one or more
destination-passing style (DPS) output tensors. Destination-passing style (DPS)
means that the calling function passes in pre-allocated storage space for the
output value(s). This allows for more efficient memory management. For example,
the graph compiler can optimize memory use by allocating output tensors on the
stack, instead of requiring custom ops to allocate heap storage for return
values.

Destination passing style requires the graph compiler to determine the
dimensions of the output tensor(s) before executing the operation. It uses the
operation's `shape()` function to determine the dimensions if they can't be
determined statically.

The following sections describe the `execute()` and `shape()` functions.

### `execute()` function

The `execute()` function performs the actual work of the custom op. It takes the
following parameter:

- `target` (`StaticString`): Indicates the device the operation is running on:
  currently takes the values `"cpu"` or `"gpu"`.

Graph output and input tensors are passed to the `execute()` function as
instances of
[`OutputTensor`](/max/api/mojo/tensor/managed_tensor_slice/#aliases) and
[`InputTensor`](/max/api/mojo/tensor/managed_tensor_slice/#aliases),
respectively. These are both aliases for specific configurations of
[`ManagedTensorSlice`](/max/api/mojo/tensor/managed_tensor_slice/ManagedTensorSlice),
so they both have the same API.

In addition to input and output tensors, the function can take the following
arguments:

- Any arguments of type [`Scalar`](/mojo/manual/types#scalar-values).

- A single argument of type `DeviceContextPtr`. This opaque pointer is
  currently required for GPU support.

```mojo
import compiler
from utils.index import IndexList
from max.tensor import OutputTensor, InputTensor, foreach, ManagedTensorSlice
from runtime.asyncrt import DeviceContextPtr

@compiler.register("add_vectors_custom")
struct AddVectorsCustom:
    @staticmethod
    fn execute[
        # "gpu" or "cpu"
        target: StaticString,
    ](
        # the first argument is the output
        out: OutputTensor,
        # starting here is the list of inputs
        x: InputTensor[type = out.type, rank = out.rank],
        y: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return x.load[width](idx) + y.load[width](idx)

        foreach[func, target=target](out, ctx)
```

### `shape()` function

The `shape()` function returns the dimensions of the output tensor(s).

The `shape()` function is required only if the graph compiler can't statically
determine the shape of the output tensor(s), and you don't manually annotate the
output shapes when building a graph.

The function takes the same arguments as the `execute()` function, minus the
output tensors and `DeviceContextPtr`. It must return an
[`IndexList`](/mojo/stdlib/utils/index_/IndexList/) specifying the dimensions of
the output tensor.

For example, if the operation takes two input tensors, and the shape of the
output tensor matches the first input tensor, you could use the following
`shape()` function:

```mojo
    @staticmethod
    fn shape(
        in1: InputTensor,
        in2: InputTensor,
    ) raises -> IndexList[in1.rank]:
        return in1.spec.shape
```
