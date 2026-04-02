# Chapter 20: Softmax and attention

Softmax and scaled dot-product attention are the operations at the heart of
transformer models. Both require reductions across sequences and draw on
techniques from earlier chapters (shared memory, warp shuffles, and register
blocking).

## Files

| File           | Description                                                                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `fig20_4.mojo` | Softmax kernel; numerically stable softmax over a sequence, using block reduce for the max and sum passes                                                                                |
| `fig20_9.mojo` | Flash Attention forward kernel; tiled attention following the Flash Attention algorithm (Dao et al., 2022), processes Q, K, V in blocks to avoid materializing the full attention matrix |

## Notes

`fig20_9.mojo` is the most complex kernel in the PMPP examples. It combines
tiling, warp shuffles (`shuffle_idx()`, `lane_group_max()`,
`lane_group_sum()`), and careful tracking of the running max and sum across K/V
tiles. The tile dimensions (`B_r`, `B_c`) and model dimension (`D_MODEL`) are
compile-time constants.
