# Chapter 4: Compute architecture and scheduling

This chapter covers how the GPU hardware schedules warps and executes threads, including the hazards that arise when barrier synchronization is used incorrectly.

## Files

| File | Description |
|------|-------------|
| `fig4_4.mojo` | **Incorrect** barrier usage: a conditional `barrier()` that causes deadlock when threads in the same block take different branches |

## Note

This file is an intentional example of broken code. The book uses it to explain why all threads in a block must reach the same barrier. Do not use this pattern.
