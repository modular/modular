# Using the MAX C API to execute a precompiled graph

The [MAX graph API](https://docs.modular.com/max/graph/) provides a powerful
framework for constructing computational graphs to be run on GPUs, CPUs, and
more. The process of compilation and execution can be separated. This example
demonstrates how to construct and compile a graph using the MAX Python API,
serialize the compiled graph to disk, and then load and execute the graph from
a C program.

> [!NOTE]
> The C interfaces shown here are undergoing rapid development and may change
> significantly over time. Additionally, the MEF file format is not a general
> serialization solution. It is device-specific and can only be transferred
> between two similar systems.

Within `test_capi.py`, a simple graph is constructed that performs the addition
of two vectors. A symbolic dimension of `vector_width` is specified, which
later allows variable sizes of vectors to be input to the graph, as long as
the two input vectors are the same size. The graph itself is configured to be
run on a GPU, and the input and output tensors are specified to reside on the
GPU. The graph is then compiled and serialized to a MEF file on disk.

Inside `example.c`, host tensors for the two vectors are initialized with
values and then transferred to the accelerator. The compiled graph is loaded
from the previously-saved MEF file, placed on the accelerator, and run using
the input tensors. The resulting tensor is moved from device to host, and the
results printed and compared against a reference.

A [Pixi](https://pixi.sh/latest/) command is available to run the entire
process of building the graph, compiling it, saving it to disk, and then
finally executing it on the GPU:

```sh
pixi run test
```

## Loading models with external weights

The `weights_example.c` file demonstrates how to provide model weights at
runtime using the C API's weights registry. This is the mechanism used when
a model's weights are stored separately from the compiled graph (for example,
in safetensors or GGUF files).

The Python script `test_weights_capi.py` builds a graph that references an
external weight via `ops.constant_external()`, compiles it to a MEF file, and
then runs the C program. The C program creates a `M_WeightsRegistry` with the
weight data and passes it to `M_initModel()`.

This example runs on CPU and does not require a GPU.

## Device graph capture and replay

`graph_capture.c` demonstrates using device graph capture (e.g. CUDA graphs) to
reduce kernel launch overhead for repeated model execution. This technique
records a model execution into a replayable graph, then replays it with
near-zero launch overhead on subsequent runs.

The example shows the three key operations:

1. **Capture** (`M_captureModelSync`): Run the model once and record the
   execution as a device graph. Returns output tensors that are updated
   in-place on each replay.
2. **Replay** (`M_replayModelSync`): Re-execute the captured graph using the
   same input buffers. Results appear in the output tensors from capture.
3. **Debug verify** (`M_debugVerifyReplayModelSync`): Run eagerly and compare
   the kernel launch trace against the captured graph to verify correctness.

> [!NOTE]
> Device graph capture requires a CUDA or HIP GPU. It is not available on
> Apple GPUs or CPU-only systems.
