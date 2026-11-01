# Debugging the Device Context

The device context manages interactions with the different kinds of devices
supported by our runtime. By default, the `DeviceContext` launches user kernels
asynchronously, which makes debugging code launched through the `DeviceContext`
difficult. That is further compounded by the entire process sitting on top of
the `AsyncRT` runtime, which is itself asynchronous.

In Mojo, the `DeviceContext` attempts to turn failures into Mojo
errors, but this only occurs at synchronization points which loses the context
of where the error originally occurred.

The code below launches a kernel which writes to a memory location on the host,
and attempts to print the resulting error from the `DeviceContext`.
The launched kernel writes to a host pointer and will always result in a
`CUDA_ERROR_ILLEGAL_ADDRESS` when run on a CUDA device.

```mojo
# Always errors with a CUDA_ERROR_ILLEGAL_ADDRESS
def kernel(res: UnsafePointer[UInt32]):
    res[] = 0

def test_function_error(ctx: DeviceContext):
    try:
      var res_host = UnsafePointer[UInt32].alloc(1)
      ctx.enqueue_function[kernel](res_host, block_dim=(1), grid_dim=(1))
      ctx.synchronize()
      res_host.free()
    except e:
      # Prints 'max/kernels/test/gpu/device_context/test_function_error.mojo:30:24'
      # pointing to the ctx.synchronize() call.
      print(e)
```

By default, the asynchronous execution of `ctx.enqueue_function` results
in the `CUDA_ERROR_ILLEGAL_ADDRESS` being detected and raised from the call to
`ctx.synchronize()`.
At that point, Mojo has lost all context as to the provenance of the error and
will print a trace pointing to `ctx.synchronize()`.
This makes identifying the offending kernel difficult when multiple kernels are
launched, or when the kernels are being executed from a MAX Graph.

## Enabling Synchronous Mode

> [!NOTE]
> There are two ways to enable synchronous execution in the device context:
>
> - set the configuration option `device-context.sync-mode = true` in your
>   `modular.cfg`
> - set the environment variable `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1`

The `DeviceContext` can be configured to launch kernels synchronously in order
to provide better error tracking.
To do so, set the environment variable `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1`.
Setting that environment variable will cause all `DeviceContext` methods which
launch device operations to implicitly synchronize.

When the example code is run with `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1`, a Mojo
error is raised from `ctx.enqueue_function[kernel](...)` rather than
`ctx.synchronize()`.

The full error generated looks like the following:

```text
max/kernels/test/gpu/device_context/test_function_error_sync_mode.mojo:29:37 failed calling 'test_function_error_sync_mode::kernel' on device cuda:0 with error 'CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)'
```

The `MODULAR_DEVICE_CONTEXT_SYNC_MODE` works in both standalone Mojo code as
well as code run from MAX Graphs.

## CUDA Driver API Logging

Set `MLRT_CUDA_DEBUG=1` to log all CUDA driver API calls:

```bash
MLRT_CUDA_DEBUG=1 ./your_program
```

Example output:

```text
[CUDA] cuMemcpyHtoDAsync_v2
[CUDA] cuMemcpyHtoDAsync_v2 -> 0
[CUDA] cuLaunchKernel
[CUDA] cuLaunchKernel -> 0
```

## Limitations

- When used to debug MAX Graphs, errors are reported in terms of the Mojo source
  code rather than the MAX Graph operation they correspond to.

- Launching kernels synchronously will slow down the execution of the graph, so
  `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1` is not suitable as the default mode of
  execution.
