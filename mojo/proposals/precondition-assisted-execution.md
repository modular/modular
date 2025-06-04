## Proposal: Implement Precondition Assisted Execution Pattern for mojo standard library 

**Author:** Anish Kanthamneni (akneni)
**Date:** June 3, 2025
**Status:** Proposed

### Abstract

This proposal suggests introducing a pattern of "unsafe assumption parameters" for functions in the Mojo standard library and for general use. These compile-time parameters would allow developers to assert specific preconditions, enabling functions to bypass certain checks or utilize more specialized, higher-performance code paths. This approach aims to provide significant performance gains in critical scenarios, reduce API fragmentation compared to numerous distinct `unsafe_` prefixed functions, and potentially improve code maintainability.

### 1. Motivation for this Proposal
Mojo standard library functions prioritize safety and generality, which is essential. However, this can lead to performance overhead in situations where developers *know* specific preconditions about their data or environment are met. For instance, a string might be known to be ASCII, or a memory buffer might be guaranteed to have a certain alignment.

Currently, one way to offer optimized paths is to create separate `unsafe_` prefixed functions (e.g., `List.unsafe_get()`). While clear, this approach can lead to significant API fragmentation if applied broadly, potentially doubling the number of relevant functions. It can also lead to boilerplate or near-duplicate code across the safe and unsafe variants.

We need a mechanism that:
1.  Allows opt-in to unsafe, specialized behavior for performance.
2.  Clearly signals the associated risks and responsibilities to the developer.
3.  Minimizes API surface bloat.


### 2. Proposed Solution: Unsafe Assumption Parameters

I propose the use of parameters in function signatures to specify assumptions. These parameters would be prefixed with `unsafe_` to clearly indicate that enabling them by overriding the safe default value shifts responsibility to the caller for upholding the asserted precondition.

**General Form:**

```mojo
fn some_function[
    unsafe_assume_X: Bool = False, # Default to safe behavior
](...) -> OutputType:
    @parameter
    if unsafe_assume_X:
        # Path optimized assuming X is true
        # ...
    else:
        # Default safe and general path
        # ...
```

The `unsafe_` prefix serves as a contract: if the developer sets `unsafe_assume_X`, they guarantee that condition X holds. If the assumption is violated, the behavior is undefined. The default values ensure that calls without these explicit parameters retain current safe behavior.

### 3. Benefits

* **Significant Performance Gains**: Allows bypassing checks (e.g., UTF-8 validation) or using hardware-specific optimizations (e.g., aligned SIMD instructions) that are not possible in general-purpose code.
* **Reduced API Fragmentation**: Instead of `foo()`, `unsafe_foo_ascii()`, and `unsafe_foo_aligned()`, we can have a single `foo()` with `unsafe_assume_ascii` and `unsafe_assume_alignment` parameters. This keeps the API surface cleaner and more focused.
* **Improved Code Maintainability**:
    * Consolidates related logic within a single function definition, making it easier to understand the relationship between the general case and optimized variants.
    * Reduces boilerplate that might arise from creating many separate, largely similar `unsafe_` functions. Common setup, teardown, or non-optimized parts of the logic can be shared, with only the assumption-specific sections varying based on compile-time parameters.

### 4. Concrete Examples

#### Example 1: String Processing - `unsafe_assume_ascii`

Many text processing tasks, such as parsing structured data like JSON or specific log formats, may operate on inputs known to be ASCII.

* **Proposed Parameterization**:
    ```mojo
    fn parse_json_ish_text[
        unsafe_assume_ascii: Bool = False
        # ... other parameters ...
    ](text: String) -> ParsedData:
        @parameter
        if unsafe_assume_ascii:
            # Optimized path for ASCII:
            # - Skip UTF-8 validation/decoding steps.
            # - Use direct byte comparisons/manipulations.
            # - Potentially use SIMD for certain ASCII operations 
            #   (e.g., finding special characters, case conversion).
            # ...
        else:
            # Default path with full UTF-8 support.
            # ...
    ```
* **Benefit**: For ASCII-guaranteed inputs, this can dramatically speed up parsing and processing by removing complex UTF-8 handling. Mojo's string type also allows for other string assumptions like `unsafe_assume_static`, `unsafe_assume_inline`, or `unsafe_assume_heap` could similarly allow optimizations by bypassing checks and code branches. 

#### Example 2: Memory Alignment - `unsafe_assume_alignment`

Numerical algorithms, tensor operations, and low-level memory routines are heavily impacted by memory alignment, especially when using SIMD instructions.

* **Proposed Parameterization**:
    ```mojo
    fn sum_tensor_elements[
        DType: SIMDable, # Example constraint
        Shape: AnyType,  # Example constraint
        unsafe_assume_alignment: UInt = 1 # 1 (or 0) implies default system alignment
    ](tensor: Tensor[DType, Shape]) -> DType:
        let data_ptr = tensor.data_ptr()

        @parameter
        if unsafe_assume_alignment >= 64 and target_has_avx512():
            # Highly optimized path using AVX512 intrinsics for 64-byte aligned data
            # ...
        elif unsafe_assume_alignment >= 32 and target_has_avx2():
            # Optimized path using AVX2 intrinsics for 32-byte aligned data
            # ...
        elif unsafe_assume_alignment >= 16: # For SSE or other 16-byte SIMD
            # Optimized path for 16-byte aligned data
            # ...
        else:
            # Default path: scalar operations or SIMD with unaligned loads/stores
            # ...
    ```
* **Benefit**: Allows direct use of the most powerful SIMD instructions available for the guaranteed alignment (e.g., `_mm256_load_ps` for AVX2 with 32-byte alignment, `_mm512_load_ps` for AVX512 with 64-byte alignment). This avoids unaligned access penalties, potential crashes on some platforms, or the overhead of manual alignment checks and workarounds. This parameter is broadly applicable to many functions processing contiguous memory, not just tensor operations.

### 5. Safety Considerations

It's important that the `unsafe_` nature of these parameters is understood:
* **User Responsibility**: If a developer provides an assumption (e.g., `unsafe_assume_alignment = 32`) but the underlying condition is not met (e.g., data is not 32-byte aligned), the behavior is **undefined**. This can lead to incorrect results, crashes, or security vulnerabilities.
* **Clear Signal**: The `unsafe_` prefix on the parameter name is the explicit warning. Documentation must clearly state the invariants the caller must uphold for each such parameter.
* **Opt-In Complexity**: This feature is intended for performance-critical situations where developers can rigorously verify and guarantee the assumptions.

### 6. Conclusion

Introducing "unsafe assumption parameters" offers a powerful mechanism to unlock significant performance improvements in Mojo by leveraging developer-guaranteed preconditions. This approach aligns with Mojo's performance goals, provides a cleaner API than proliferating `unsafe_` function variants, and can improve code maintainability. I believe this would be a valuable addition to the language and standard library. I've proposed this informally in the [modular forum](https://forum.modular.com/t/precondition-based-optimization-library-design-proposal/1568/3), and it seemed to get some support from @lattner and @owenhilyard among others, so I'd love to see what everyone else thinks about it!



