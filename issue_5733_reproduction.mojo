# Reproduction of Issue #5733
# Division by zero produces nan instead of inf

fn main():
    print("=== Issue #5733: Division by Zero Bug ===")
    print()

    print("Testing: 1.0 / 0.0")
    var result1 = 1.0 / 0.0
    print("Result:", result1)
    print("Expected: inf")
    print("Actual: This currently prints 'nan' (BUG!)")
    print()

    print("Testing: -1.0 / 0.0")
    var result2 = -1.0 / 0.0
    print("Result:", result2)
    print("Expected: -inf")
    print()

    print("Testing: 0.0 / 0.0")
    var result3 = 0.0 / 0.0
    print("Result:", result3)
    print("Expected: nan (this one should be nan)")
    print()

    print("Testing: 5.0 / 0.0")
    var result4 = 5.0 / 0.0
    print("Result:", result4)
    print("Expected: inf")
    print()

    print("=== IEEE 754 Standard Behavior ===")
    print("According to IEEE 754:")
    print("  positive_number / 0.0  = +inf")
    print("  negative_number / 0.0  = -inf")
    print("  0.0 / 0.0              = nan")
    print()
    print("The bug is that Mojo's compiler incorrectly optimizes")
    print("1.0 / 0.0 to nan instead of inf during constant folding.")
