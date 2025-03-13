---
title: GPU Basics
sidebar_position: 1
description: The basics of GPU programming with Mojo.
---

This documentation aims to build your GPU programming knowledge from the ground
up, starting with the lowest levels of the stack before progressing to
higher-level functionality. It’s designed for a diverse audience, from
experienced GPU developers to programmers new to GPU coding. Mojo allows you to
program NVIDIA and AMD GPUs, with direct access to low-level GPU primitives,
while sharing types and functions that can also run on CPUs where applicable.
If you're experienced with [NVIDIA Compute Unified Device
Architecture](https://developer.nvidia.com/cuda-toolkit) (CUDA) or [AMD Radeon
Open Compute](https://www.amd.com/en/products/software/rocm.html) (ROCm), what
you'll learn here will enable you to expand your reach to more hardware.

## Introduction to massively parallel programming

We can no longer rely on new generations of CPUs to increase application
performance through improved clock speeds. Power demands and heat dissipation
limits have stalled that trend, pushing the hardware industry toward increasing
the number of physical cores. Modern consumer CPUs now boast 16 cores or more,
capable of running in parallel, which forces programmers to rethink how they
maximize performance. This shift is especially evident in AI applications, where
performance scales remarkably well with additional cores.

NVIDIA’s breakthrough came with CUDA, a general programming model that allows
developers to target both server and consumer GPUs for any application domain.
This vision sparked an AI revolution when Alex Krizhevsky, Ilya Sutskever, and
Geoffrey Hinton trained AlexNet on consumer GPUs, significantly outperforming
traditional computer vision methods. GPUs pack thousands of cores, the NVIDIA
H100 can run 16,896 threads in parallel in a single clock cycle, with over
270,000 threads queued and ready to go. They're also engineered in a way where
the cost of scheduling threads is much lower compared to a traditional CPU.

Harnessing this hardware requires a new programming mindset. Mojo represents a
chance to rethink GPU programming and make it more approachable. C/C++ is at the
core of GPU programming, but we’ve seen leaps in ergonomics and memory safety
from systems programming languages in recent years. Mojo expands on Python’s
familiar syntax, adds direct access to low-level CPU and GPU intrinsics for
systems programming, and introduces ergonomic and safety improvements from
modern languages. This course aims to empower programmers with minimal
specialized knowledge to build high-performance, GPU-enabled applications. By
lowering the barrier to entry, we aim to fuel more breakthroughs and accelerate
innovation.

:::tip Setup

All of these notebook cells are runnable through a VS Code extension. You can
install
[Markdown Lab](https://marketplace.visualstudio.com/items?itemName=jackos.mdlab),
then clone the repo that contains the markdown that generated this website:

```sh
git clone git@github.com:modular/max
cd max/mojo/docs/manual/gpu
```

And open e.g. `gpu-basics.md` to run the code cells interactively.

If you prefer the traditional approach, create a file such as `main.mojo` and
put everything except the imports into a `def main`:

```mojo :once
from gpu import thread_idx, DeviceContext

def main():
    fn printing_kernel():
        print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")

    var ctx = DeviceContext()

    ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
    ctx.synchronize()
```

Then run the file, if you haven't setup Mojo yet, check out the [Getting
Started](../get-started.mdx) guide.

:::

## Imports

These are all the imports required to run the examples, put this at the top of
your file if you're running from `mojo main.mojo`:

```mojo
from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace, external_memory
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof
```

## Your first kernel

In the context of GPU programming, a kernel is a program that runs on each
thread that you launch:

```mojo
fn printing_kernel():
    print("GPU thread: [", thread_idx.x, thread_idx.y, thread_idx.z, "]")
```

:::note

The term `kernel` in this context originated in the 1980s with the introduction
of the
[Single Program, Multiple Data](https://en.wikipedia.org/wiki/Single_program,_multiple_data)
(SPMD) parallel programming technique, which underpins ROCm and CUDA. In this
approach, a kernel executes concurrently across distinct elements of large data
structures.

:::

We can pass this function as a parameter to `enqueue_function()` to compile it
for your attached GPU and launch it. First we need to get the
[`DeviceContext`](/mojo/stdlib/gpu/host/device_context/DeviceContext) for your
GPU:

```mojo
var ctx = DeviceContext()
```

Now we have the `DeviceContext` we can compile and launch the kernel:

```mojo :once
ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)

# Wait for the kernel to finish executing before handing back to CPU
ctx.synchronize()
```

```text
GPU thread: [ 0 0 0 ]
GPU thread: [ 1 0 0 ]
GPU thread: [ 2 0 0 ]
GPU thread: [ 3 0 0 ]
```

## Threads

Because we passed `block_dim=4`, we launched 4 threads on the x dimension, the
kernel code we wrote is executed on each thread. The printing can be out of
order depending on which thread reaches that `print()` call first.

Now add the y and z dimensions with `block_dim=(2, 2, 2)`:

```mojo :once
ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=(2, 2, 2))
ctx.synchronize()
```

```text
GPU thread: [ 0 0 0 ]
GPU thread: [ 1 0 0 ]
GPU thread: [ 0 1 0 ]
GPU thread: [ 1 1 0 ]
GPU thread: [ 0 0 1 ]
GPU thread: [ 1 0 1 ]
GPU thread: [ 0 1 1 ]
GPU thread: [ 1 1 1 ]
```

We're now launching 8 (2x2x2) threads in total.

## Host vs device and enqueue

You'll see the word host which refers to the CPU that schedules work for the
device, device refers to the accelerator which in this case is a GPU.

When you encounter the term `enqueue` in a method or function call, it means
that the host is scheduling the operation to execute asynchronously on the
device. If your host-side code relies on the outcome of these device-enqueued
operations, you need to call `ctx.synchronize()`. For instance, printing from
the CPU without first synchronizing might result in out-of-order output:

```mojo :once
ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
print("This might finish before the GPU has completed its work")
```

```text
This might finish before the GPU has completed its work
GPU thread: [ 0 0 0 ]
GPU thread: [ 1 0 0 ]
GPU thread: [ 2 0 0 ]
GPU thread: [ 3 0 0 ]
```

In the above example we failed to call `synchronize()` before printing on the
host, the device could be slightly slower to finish its work, so you might
see that output after the host output. Let's add a `synchronize()` call:

```mojo :once
ctx.enqueue_function[printing_kernel](grid_dim=1, block_dim=4)
ctx.synchronize()
print("This will finish after the GPU has completed its work")
```

```text
GPU thread: [ 0 0 0 ]
GPU thread: [ 1 0 0 ]
GPU thread: [ 2 0 0 ]
GPU thread: [ 3 0 0 ]
This will finish after the GPU has completed its work
```

Any method or function you `enqueue` to run on the device, will run in the order
that you enqueued them. It's only when you're doing something from the host
which is dependent on the results of enqueued calls that you have to
synchronize. More on this later when we introduce device buffers.

In GPU programming with Mojo, when there's a tradeoff between GPU performance
and safety or ergonomics, performance takes priority, aligning with the
expectations of kernel engineers. For instance, while we could eliminate the
`enqueue` prefix from method calls and force synchronization for each of them,
this would come at a performance cost. Take note to remember anything from these
warning text blocks for potential safety violations:

:::warning Synchronization

For any methods or functions prefixed with `enqueue`, you must synchronize
before running CPU code that is dependent on what you're enqueuing. Enqueueing
multiple method or function calls for a single GPU is safe, as they are
scheduled to run in the order you call them.

:::

Mojo enhances the safety and ergonomics of C++ GPU programming where it doesn't
sacrifice performance. For example, ASAP destruction automatically frees buffer
memory on last use of the object, eliminating memory leaks and ensuring memory
is released as early as possible. This is an evolution on modern memory
management solutions such as C++ RAII, which is scope based and may hold onto
memory for longer than expected, which is a precious resource in AI
applications.

## Blocks

This kernel demonstrates how blocks work:

```mojo :once
fn block_kernel():
    print(
        "block: [",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        "]",
        "thread: [",
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        "]"
    )

ctx.enqueue_function[block_kernel](grid_dim=(2, 2), block_dim=2)
ctx.synchronize()
```

```text
block: [ 1 1 0 ] thread: [ 0 0 0 ]
block: [ 1 1 0 ] thread: [ 1 0 0 ]
block: [ 0 0 0 ] thread: [ 0 0 0 ]
block: [ 0 0 0 ] thread: [ 1 0 0 ]
block: [ 1 0 0 ] thread: [ 0 0 0 ]
block: [ 1 0 0 ] thread: [ 1 0 0 ]
block: [ 0 1 0 ] thread: [ 0 0 0 ]
block: [ 0 1 0 ] thread: [ 1 0 0 ]
```

We're still launching 8 (2x2x2) threads, where there are 4 blocks, each with 2
threads. In GPU programming this grouping of blocks and threads is important,
each block can have its own fast SRAM (Static Random Access Memory) which allows
threads to communicate. The threads within a block can also communicate through
registers, we'll cover this concept when we get to warps. For now the
important information to internalize is:

- `grid_dim` defines how many blocks are launched.
- `block_dim` defines how many threads are launched in each block.

## Tiles

The x, y, z dimensions of blocks are important for splitting up large jobs into
tiles, so each thread can work on its own subset of the problem. Below is a
visualization for how a contiguous array of data can be split up into tiles, if
we have an array of UInt32 (Unsigned Integer 32bit) data like:

```plaintext
[ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ]
```

We could split work up between threads and blocks, we're only going to use the x
dimension for threads and blocks to get started:

```plaintext
Thread  |    0  1  2  3
-------------------------
block 0 | [  0  1  2  3 ]
block 1 | [  4  5  6  7 ]
block 2 | [  8  9 10 11 ]
block 3 | [ 12 13 14 15 ]
```

If you had a much larger data array you could further split it up in into tiles,
e.g. tile with widths [2, 2] at index (0, 0) would be:

```plaintext
[ 0 1 ]
[ 4 5 ]
```

And index (1, 0) would be:

```plaintext
[ 2 3 ]
[ 6 7 ]
```

This is where you'd introduce the y dimension. For now we're going to focus on
how blocks and threads interact, splitting up an array into 1 row per block, and
1 value per thread.

## Host buffer

First we'll initialize a contiguous array on CPU and fill in its values:

```mojo
alias dtype = DType.uint32
alias blocks = 4
alias threads = 4
# One value per thread
alias in_els = blocks * threads

# Allocate data on the host and return a buffer which owns that data
var in_host = ctx.enqueue_create_host_buffer[dtype](in_els)

# Ensure the host buffer has finished being created
ctx.synchronize()

# Fill in the buffer with values from 0 to 15 and print it
iota(in_host.unsafe_ptr(), in_els)
print(in_host)
```

```text
DeviceBuffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
```

## Device buffer

We now have a host buffer that we can copy to the GPU:

```mojo
# Allocate a buffer for the GPU
var in_dev = ctx.enqueue_create_buffer[dtype](in_els)

# Copy the data from the CPU to the GPU buffer
in_host.enqueue_copy_to(in_dev)
```

Creating the GPU buffer is allocating _global memory_ which can be accessed from
any block and thread, this memory is relatively slow compared to _shared memory_
which is shared between all of the threads in a block, more on that later.

## Enqueue scheduling

As previously stated, when you `enqueue` multiple method/function calls to run
on a device they will be scheduled to run in the order that you enqueue them:

```mojo
alias size = 4
alias type = DType.uint8

fn dummy_kernel(buffer: DeviceBuffer[type]):
    buffer[thread_idx.x] = thread_idx.x

# All of these method calls run in the order that they were enqueued
var host_buffer = ctx.enqueue_create_host_buffer[type](size)
var dev_buffer = ctx.enqueue_create_buffer[type](size)
ctx.enqueue_function[dummy_kernel](dev_buffer, grid_dim=1, block_dim=size)
dev_buffer.enqueue_copy_to(host_buffer)

# Have to synchronize here before printing on CPU, or else the kernel may
# not have finished executing.
ctx.synchronize()
print(host_buffer)
```

```text
DeviceBuffer([0, 1, 2, 3])
```

You can schedule multiple independent kernels to run at the same time using
streams, which is a concept we'll introduce later.

## Tensor indexing from threads

Now that we have the data set up, we can wrap the data in a `LayoutTensor` so
that we can reason about how to index into the array, allowing each thread to
get its corresponding value:

```mojo
# Row major: elements are stored sequentially in memory [0, 1, 2, 3, 4, 5, ...]
# Column major: used in some GPU optimizations, stored as [0, 4, 8, 12, 1, ...]
alias layout = Layout.row_major(blocks, threads)

var tensor = LayoutTensor[dtype, layout](in_dev)
```

This `LayoutTensor` is a wrapper over the data stored inside `in_dev`, it
doesn't own its memory but allows us to index using block and thread ids. We'll
create an alias so that we don't have to repeat the type information, arguments
on GPU have a [static origin](/mojo/manual/values/lifetimes) which means it
lives for the duration of the kernel:

```mojo
alias InputLayoutTensor = LayoutTensor[dtype, layout, StaticConstantOrigin]
```

Initially we'll just print the values to confirm it's indexing as we expect:

```mojo :once
fn print_values_kernel(tensor: InputLayoutTensor):
    var bid = block_idx.x
    var tid = thread_idx.x
    print("block:", bid, "thread:", tid, "val:", tensor[bid, tid])

ctx.enqueue_function[print_values_kernel](
    tensor, grid_dim=blocks, block_dim=threads,
)
ctx.synchronize()
```

```text
block: 0 thread: 0 val: 0
block: 0 thread: 1 val: 1
block: 0 thread: 2 val: 2
block: 0 thread: 3 val: 3
block: 3 thread: 0 val: 12
block: 3 thread: 1 val: 13
block: 3 thread: 2 val: 14
block: 3 thread: 3 val: 15
block: 2 thread: 0 val: 8
block: 2 thread: 1 val: 9
block: 2 thread: 2 val: 10
block: 2 thread: 3 val: 11
block: 1 thread: 0 val: 4
block: 1 thread: 1 val: 5
block: 1 thread: 2 val: 6
block: 1 thread: 3 val: 7
```

As in the visualization above, the block/thread is getting the corresponding
value that we expect. You can see `block: 3 thread: 3` has the last value 15.

## Multiply kernel

Now that we've verified we're getting the correct values when indexing, we'll
launch a kernel to multiply each value:

```mojo :once
fn multiply_kernel[multiplier: Int](tensor: InputLayoutTensor):
    tensor[block_idx.x, thread_idx.x] *= multiplier

ctx.enqueue_function[multiply_kernel[2]](
    tensor,
    grid_dim=blocks,
    block_dim=threads,
)

# Copy data back to host and print as 2D array
in_dev.enqueue_copy_to(in_host)
ctx.synchronize()

var host_tensor = LayoutTensor[dtype, layout](in_host)
print(host_tensor)
```

```text
0 2 4 6
8 10 12 14
16 18 20 22
24 26 28 30
```

Congratulations! You've successfully run a kernel that modifies values from your
GPU, and printed the result on your CPU. You can see above that each thread
multiplied a single value by 2 in parallel on the GPU, and copied the result
back to the CPU.

## Sum reduce output tensor

We're going to set up a new buffer which will have all the reduced values with
the sum of each thread in the block:

```plaintext
Output: [ block[0] block[1] block[2] block[3] ]
```

Set up the output buffer for the host and device:

```mojo
var out_host = ctx.enqueue_create_host_buffer[dtype](blocks)
var out_dev = ctx.enqueue_create_buffer[dtype](blocks)

# Zero the values on the device as they'll be used to accumulate results
ctx.enqueue_memset(out_dev, 0)
```

The problem here is that we can't have all the threads summing their values into
the same index in the output buffer as that will introduce race conditions.
We're going to introduce new concepts to deal with this.

## Shared memory

This kernel uses shared memory to accumulate values. Shared memory is much
faster than global memory because it resides on-chip, closer to the processing
cores, reducing latency and increasing bandwidth. It's not an optimal solution
for this kind of reduction operation, but it's a good way to introduce shared
memory in a simple example.  We'll cover better solutions in the next sections.

```mojo :once
fn sum_reduce_kernel(
    tensor: InputLayoutTensor, out_buffer: DeviceBuffer[dtype]
):
    # Allocates memory to be shared between threads once, prior to the kernel launch
    var shared = stack_allocation[
        blocks * sizeof[dtype](),
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    # Place the corresponding value into shared memory
    shared[thread_idx.x] = tensor[block_idx.x, thread_idx.x][0]

    # Await all the threads to finish loading their values into shared memory
    barrier()

    # If this is the first thread, sum and write the result to global memory
    if thread_idx.x == 0:
        for i in range(threads):
            out_buffer[block_idx.x] += shared[i]

ctx.enqueue_function[sum_reduce_kernel](
    tensor,
    out_dev,
    grid_dim=blocks,
    block_dim=threads,
)

# Copy the data back to the host and print out the SIMD vector
out_dev.enqueue_copy_to(out_host)
ctx.synchronize()

print(out_host)
```

```text
DeviceBuffer([6, 22, 38, 54])
```

For our first block/tile we summed the values:

```plaintext
sum([ 0 1 2 3 ]) == 6
```

And the reduction resulted in the output having the sum of 6 in the first
position. Every tile/block has been reduced to:

```plaintext
[ 6 22 38 54]
```

## Thread level SIMD

We could skip using shared memory altogether using [Single Instruction, Multiple
Data](/mojo/manual/operators#simd-values) (SIMD), this is a faster option to consider
if it suits your problem. Each thread has access to SIMD registers which can
perform operations on a vector in a single instruction. Here we'll be launching
one thread per block, loading the 4 corresponding values from that block as a
SIMD vector, and summing them together in a single instruction:

```mojo :once

fn simd_reduce_kernel(
    tensor: InputLayoutTensor, out_buffer: DeviceBuffer[dtype]
):
    out_buffer[block_idx.x] = tensor.load[4](block_idx.x, 0).reduce_add()

# Reset the output values first
ctx.enqueue_memset(out_dev, 0)

ctx.enqueue_function[simd_reduce_kernel](
    tensor,
    out_dev,
    grid_dim=blocks,
    block_dim=1, # one thread per block
)

# Ensure we have the same result
out_dev.enqueue_copy_to(out_host)
ctx.synchronize()

print(out_host)
```

```text
DeviceBuffer([6, 22, 38, 54])
```

This is much cleaner and faster, instead of 4 threads writing to shared memory,
we're using 1 thread per block to do a single SIMD reduction.  Shared memory has
many uses, but as you learn more tools you'll be able decipher which is the most
performant for your particular problem.

## Warps

:::note Warps

Warp level instructions are an advanced concept, this section is just to
demonstrate that these low-level primitives are available from Mojo. We'll go
into more depth on warps later, so don't worry if it doesn't make sense yet.

:::

A _warp_ is a group of threads (32 on NVIDIA, 64 on AMD) within a block. Threads
within the same warp can synchronize their execution, and at a particular step
perform SIMD instructions using values from the other threads in lockstep. We
have only 4 threads within each block, well under the 32 limit, if this wasn't
the case you'd have to do two reductions, one from each warp to shared memory,
then another from shared memory to the output buffer/tensor.

Here is a warp reduction kernel:

```mojo :once
fn warp_reduce_kernel(
    tensor: InputLayoutTensor, out_buffer: DeviceBuffer[dtype]
):
    var value = tensor.load[1](block_idx.x, thread_idx.x)

    # Each thread gets the value from one thread higher, summing them as they go
    value = warp.sum(value)

    # Print each reduction step in the first block
    if block_idx.x == 0:
        print("thread:", thread_idx.x, "value:", value)

    barrier()

    # Thread 0 has the reduced sum of the values from all the other threads
    if thread_idx.x == 0:
        out_buffer[block_idx.x] = value

ctx.enqueue_memset(out_dev, 0)

ctx.enqueue_function[warp_reduce_kernel](
    tensor,
    out_dev,
    grid_dim=blocks,
    block_dim=threads,
)
ctx.synchronize()

# Ensure we have the same result
out_dev.enqueue_copy_to(out_host)
ctx.synchronize()

print(out_host)
```

```text
thread: 0 value: 6
thread: 1 value: 6
thread: 2 value: 5
thread: 3 value: 3
DeviceBuffer([6, 22, 38, 54])
```

You can see in the output that the first block had the values [0 1 2 3] and was
reduced from top to bottom (shuffle down) in this way, where the result of one
thread is passed to the next thread down:

```plaintext
thread 3: value=3   next_value=N/A   result=3
thread 2: value=2   next_value=3     result=5
thread 1: value=1   next_value=5     result=6
thread 0: value=0   next_value=6     result=6
```

## Exercises

Now that you've learnt some of the core primitives for GPU programming, here are
some exercise to cement some of the knowledge. Feel free to go back and look at
the examples.

1. Create a host buffer for the input of `DType` `Int64`, with 32 elements, and
initialize the numbers ordered sequentially. Copy the host buffer to the device.
2. Create a tensor that wraps the host buffer, with the dimensions (8, 4)
3. Create an host and device buffer for the output of `DType` `Int64`, with 8
elements.
4. Launch a GPU kernel with 8 blocks and 4 threads that takes every value and
raises it to the power of 2 using x**2, then does a reduction using your
preferred method to write to the output buffer.
5. Copy the device buffer to the host buffer, and print it out on the CPU.

The next chapter is coming soon, in the meantime you can check out some more
advanced [GPU programming examples
here](https://github.com/modular/max/tree/main/examples/gpu_functions).
