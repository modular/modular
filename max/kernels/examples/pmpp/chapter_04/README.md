# Chapter 4: Compute architecture and scheduling

Warps, scheduling, and execution hazards become visible when barrier
synchronization goes wrong. The single example here shows what happens when
threads in the same block diverge at a barrier.

## Files

| File | Description |
|------|-------------|
| `fig4_4.mojo` | **Incorrect** barrier usage: a conditional `barrier()` that causes deadlock when threads in the same block take different branches |

## Note

This file is an intentional example of broken code. The book uses it to explain
why all threads in a block must reach the same barrier. Do not use this pattern.
This file has no `main` function and is not runnable.
