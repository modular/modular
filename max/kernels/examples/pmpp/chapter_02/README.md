# Chapter 2: Heterogeneous data parallel computing

Vector addition is the "hello world" of GPU programming. The examples here build
from a CPU baseline to a complete GPU implementation with memory management and
kernel launch, introducing the core programming model along the way.

## Files

| File           | Description                                                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `fig2_4.mojo`  | CPU-only vector addition, the sequential baseline before any GPU code                                                                         |
| `fig2_8.mojo`  | GPU memory management: allocate device buffers, copy host-to-device and device-to-host (deallocation is automatic via Mojo's ownership model) |
| `fig2_10.mojo` | Vector addition GPU kernel; each thread computes one element of C = A + B                                                                     |
| `fig2_12.mojo` | Launch configuration: calculating grid and block dimensions with `ceildiv()`                                                                  |
| `fig2_13.mojo` | Complete host function combining memory management and kernel launch                                                                          |
| `vec_add.mojo` | Full working example: kernel + host function + correctness verification                                                                       |
