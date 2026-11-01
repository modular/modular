# Driver: Cross-vendor kernel execution runtime

## Introduction

Driver is a low-level cross-vendor runtime which allows device discovery,
memory management and kernel execution on the CPU and various different
accelerators and GPUs, and NIXL; a cross-vendor multi-node communication layer.

## Interfaces

C++ and Mojo consumers use the headers under `MLRT/include/MLRT/Driver/`.
Consumers that cannot include C++ headers use the C interface, which mirrors the
Python bindings in `max/python/max/_core/driver.pyi`.

### The C interface

`MLRT/include/MLRT/Driver/CAPI/DriverCAPI.h` declares opaque, refcounted
handles for devices (a device plus its process-wide context), streams, events,
buffers and registered host mappings, and the operations on them: stream
creation and synchronization, `wait_for` ordering, event recording, timed
events, pointer-level asynchronous copies on a chosen stream, registration of
foreign host memory for DMA, and peer-access queries and peer copies.

It is a translation layer over the same `M::Driver` objects the Python bindings
use, so a handle obtained through it executes on the contexts MAX itself uses,
and memory allocated through one is addressable through the other. A consumer
loads it from `libmax.so`, or from the shim `//MLRT:Driver/CAPIShim` when it
runs without MAX, and resolves the entry points by name, starting with
`M_DriverCApiVersion`.
