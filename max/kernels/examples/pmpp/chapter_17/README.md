# Chapter 17: Sparse matrix-vector multiplication

Sparse matrix-vector multiplication (SpMV) appears throughout scientific
computing, graph algorithms, and machine learning. Performance varies
significantly depending on storage format, so these examples compare four: COO,
CSR, ELL, and CSC.

## Files

| File              | Description                                                                                                                                             |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `spmv_utils.mojo` | Shared utilities; defines `COOMatrix`, `CSRMatrix`, `ELLMatrix`, and `CSCMatrix` structs, plus `generate_sparse_matrix()`, `spmv_cpu()`, and `verify()` |
| `fig17_5.mojo`    | SpMV with COO (Coordinate) format; each nonzero stored as a (row, col, value) triple, uses atomic adds to accumulate row results                        |
| `fig17_9.mojo`    | SpMV with CSR (Compressed Sparse Row) format; each thread processes one row, iterating over its nonzeros                                                |
| `fig17_12.mojo`   | SpMV with ELL (ELLPACK) format; padded rows of fixed length enable coalesced access, one thread per row                                                 |
| `fig17_18.mojo`   | SpMV with CSC (Compressed Sparse Column) format; each thread processes one column and scatters contributions to output rows via atomics                 |

## Notes

CSR is the general-purpose format. ELL works well for matrices with regular
sparsity patterns. COO and CSC require atomic operations.

`spmv_utils.mojo` is a dependency for all figure files in this directory.
