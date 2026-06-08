# Fix for Issue #5733: Division by Zero Returns NaN Instead of Inf

## Problem Summary

Mojo incorrectly returns `nan` when dividing a non-zero number by zero, instead of 
returning `inf` (infinity) as specified by the IEEE 754 floating-point standard.

## Current Behavior (Bug)

```mojo
fn main():
    print("inf:", 1.0 / 0.0)  # Prints: inf: nan (WRONG!)
```

**Output:** `inf: nan` ❌

## Expected Behavior (Fix)

```mojo
fn main():
    print("inf:", 1.0 / 0.0)  # Should print: inf: inf
```

**Output:** `inf: inf` ✅

## IEEE 754 Standard

According to the IEEE 754 floating-point standard:

| Expression | Expected Result |
|------------|----------------|
| `positive / 0.0` | `+inf` (positive infinity) |
| `negative / 0.0` | `-inf` (negative infinity) |
| `0.0 / 0.0` | `nan` (not a number) |
| `positive / -0.0` | `-inf` |
| `negative / -0.0` | `+inf` |

## Root Cause

The bug appears to be in the **constant folding optimization** phase of the Mojo compiler.
When the compiler evaluates `1.0 / 0.0` at compile time, it incorrectly produces `nan`
instead of following IEEE 754 rules.

Possible causes:
1. Incorrect constant evaluation in the compiler's optimizer
2. Wrong handling of division by zero in the constant folder
3. Confusion between `0.0 / 0.0` (which should be nan) and `non-zero / 0.0` (which should be inf)

## Implementation Notes

The fix should be implemented in the Mojo compiler's constant folding logic:

1. **Check the numerator before folding:**
   - If numerator is 0.0 and denominator is 0.0 → produce `nan`
   - If numerator is positive and denominator is 0.0 → produce `+inf`
   - If numerator is negative and denominator is 0.0 → produce `-inf`

2. **Handle negative zero:**
   - IEEE 754 distinguishes between `+0.0` and `-0.0`
   - `1.0 / -0.0` should produce `-inf` (not `+inf`)

3. **Apply to all floating-point types:**
   - Float16, Float32, Float64, BFloat16
   - Both compile-time constants and runtime operations

## Test Coverage

The test file `test_division_by_zero.mojo` verifies:
- ✅ Positive number / 0.0 = +inf
- ✅ Negative number / 0.0 = -inf
- ✅ 0.0 / 0.0 = nan
- ✅ Division by negative zero
- ✅ Operations with infinity values
- ✅ Comparisons with infinity

## Benefits of This Fix

1. ✅ **IEEE 754 Compliance**: Correct behavior according to the standard
2. ✅ **Interoperability**: Matches behavior of C, C++, Python, Rust, etc.
3. ✅ **Mathematical Correctness**: Properly represents mathematical infinity
4. ✅ **Numerical Stability**: Allows algorithms to handle edge cases correctly

## Example Use Cases

Many numerical algorithms rely on correct infinity handling:

```mojo
# Computing limits
var limit = 1.0 / (x - x0)  # Should approach inf as x→x0

# Numerical ranges
var max_value = 1.0 / 0.0  # Use inf as sentinel value

# Algorithm stability
if result > 1.0 / 0.0:  # Never true for finite values
    print("Impossible!")
```

## Related Standards

- **IEEE 754-2008**: Binary floating-point arithmetic
- **ISO C11**: Specifies division by zero behavior
- **Python**: `1.0 / 0.0` raises ZeroDivisionError, but `float('inf')` exists
- **C/C++**: `1.0 / 0.0` produces `INFINITY` (with appropriate compilation flags)
