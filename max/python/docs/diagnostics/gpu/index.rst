:description: The MAX Python GPU Diagnostics API reference.
:title: gpu
:type: package
:lang: python
:wrapper_class: rst-index
:source: max/diagnostics/gpu/

MAX GPU Diagnostics API
-----------------------

Real-time GPU monitoring and diagnostic capabilities for NVIDIA and AMD graphics
hardware.

The GPU diagnostics module provides comprehensive tools for monitoring graphics
processing unit performance, memory usage, and utilization metrics. It supports
both NVIDIA GPUs through NVML and AMD GPUs through ROCm SMI, offering unified
access to hardware statistics regardless of vendor. The API enables both
synchronous queries for immediate metrics and asynchronous background collection
for continuous monitoring during long-running inference sessions.

Classes
=======

*
  :code_link:`/max/api/python/diagnostics/gpu/BackgroundRecorder|BackgroundRecorder`:
  Asynchronous GPU metrics collection.
* :code_link:`/max/api/python/diagnostics/gpu/GPUDiagContext|GPUDiagContext`:
  Context manager providing unified access to GPU diagnostic information across
  NVIDIA and AMD hardware.
* :code_link:`/max/api/python/diagnostics/gpu/GPUStats|GPUStats`: Comprehensive
  GPU state snapshot containing memory and utilization statistics.
* :code_link:`/max/api/python/diagnostics/gpu/MemoryStats|MemoryStats`: Detailed
  GPU memory usage statistics including total, free, used, and reserved memory.
*
  :code_link:`/max/api/python/diagnostics/gpu/UtilizationStats|UtilizationStats`:
  GPU compute and memory activity utilization percentages.


.. toctree::
   :hidden:

   BackgroundRecorder
   GPUDiagContext
   GPUStats
   MemoryStats
   UtilizationStats