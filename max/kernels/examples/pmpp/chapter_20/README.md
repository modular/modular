# Chapter 20: Softmax and attention

This chapter covers two operations central to transformer models: softmax and scaled dot-product attention. Both require reductions across sequences and draw on the same techniques from earlier chapters — shared memory, warp shuffles, and register blocking.

## Files

| File | Description |
|------|-------------|
| `fig20_4.mojo` | Softmax kernel; numerically stable softmax over a sequence, using block reduce for the max and sum passes |
| `fig20_9.mojo` | Flash Attention forward kernel; tiled attention following the Flash Attention algorithm (Dao et al., 2022), processes Q, K, V in blocks to avoid materializing the full attention matrix |

## Notes

`fig20_9.mojo` is the most complex kernel in the PMPP examples. It combines tiling, warp shuffles (`shuffle_idx()`, `lane_group_max()`, `lane_group_sum()`), and careful tracking of the running max and sum across K/V tiles. The tile dimensions (`B_r`, `B_c`) and model dimension (`D_MODEL`) are compile-time constants.

These examples connect directly to MAX's built-in attention and softmax operations, which handle batching, masking, and multi-head projection on top of the same underlying GPU kernels.
