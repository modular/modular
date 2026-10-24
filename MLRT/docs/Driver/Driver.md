# AsyncRT Driver Overview

The Driver layer of the Modular stack consists of the **Device** module -
low-level access to devices (CPU, GPU etc.) - as well the communication module -
**NIXL**.

## Device

The Device module provides direct access to CPU and GPU devices and associated
constructs like contexts and streams. This layer can be thought of as the
equivalent of the combination of CUDA Driver and Runtime API/libraries.

- A `Device` object is a singleton per available GPU device or the CPUs (all
  CPUs are represented as a single device).
- Multiple `DeviceContext` objects can be created on a `Device`.
- Each `DeviceContext` represents a **stream** of execution on its device. Since
  multiple `DeviceContext`s can be associated with a `Device` this allows for
  parallel execution of different streams of execution (e.g. kernel execution
  running **concurrently** with a device->host copy).
- `DeviceContext` provides synchronization primitives across different streams
  of executions, aka `DeviceContext`s: `enqueueEvent()` and `enqueueWaitFor()`.
- `DeviceStream` is a minimal API purely exposing these synchronization
  primitives. It is in spirit the base class of `DeviceContext`, but the C++
  implementation does not treat it that way. Maybe it should.
- In order to interact and synchronize with external systems (e.g. PyTorch) we
  have an API to create an externally managed `DeviceStream` object based on the
  raw GPU based driver value: `DeviceContext::createExternalStream(void*)`. This
  API-call simply creates a shared reference to the externally-managed stream
  for internal use.

In short:

- `Device <-> DeviceContext` ratio 1:N
- `DeviceContext <-> stream-of-execution` ratio 1:1
- `DeviceStream <-> external-stream-of-execution` ratio 1:1
- Note: To simplify the APIs interacting with external code, it is possible to
  get a `DeviceStream` object from a `DeviceContext`, meaning that the last
  statement is not exactly true.

## NIXL

The NIXL module is our hack-to-own adaptation of the [eponymous open-source
project](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) for inter-GPU
communication. We modify the source code to be aware of our own `DeviceContext`
implementation. Currently,

- we only support inter-NVIDIA-GPU communication on a single node. Support for
AMD GPUs and for inter-node communication are future features.
- we only employ one of the many compatible back-ends -
[UCX](https://github.com/openucx/ucx/tree/master).
