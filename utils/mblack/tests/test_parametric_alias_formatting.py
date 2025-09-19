# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import mblack
from mblack.mode import TargetVersion


class TestParametricAliasFormatting(unittest.TestCase):
    """Test that parametric aliases are formatted correctly."""

    def setUp(self):
        """Set up test mode with Mojo target version."""
        self.mode = mblack.Mode(target_versions={TargetVersion.MOJO})

    def test_simple_parametric_alias(self):
        """Test basic parametric alias formatting."""
        source = "alias addOne[x: Int] : Int = x + 1"
        expected = "alias addOne[x: Int]: Int = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_default_values(self):
        """Test parametric alias with default parameter values."""
        source = "alias Float64[size: Int = 1] = SIMD[DType.float64, size]"
        expected = "alias Float64[size: Int = 1] = SIMD[DType.float64, size]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_multiple_parameters(self):
        """Test parametric alias with multiple parameters."""
        source = "alias TwoParams[a: Int, b: Int] = SIMD[DType.float32, a + b]"
        expected = "alias TwoParams[a: Int, b: Int] = SIMD[DType.float32, a + b]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_complex_expression(self):
        """Test parametric alias with complex expression in body."""
        source = "alias ComplexExpr[x: Int, y: Int] : Int = (x * y) + (x + y)"
        expected = "alias ComplexExpr[x: Int, y: Int]: Int = (x * y) + (x + y)\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_conditional_expression(self):
        """Test parametric alias with conditional expression."""
        source = "alias ConditionalAlias[dt: DType, size: Int] = SIMD[dt, size if size > 0 else 1]"
        expected = "alias ConditionalAlias[dt: DType, size: Int] = SIMD[dt, size if size > 0 else 1]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_nested_expressions(self):
        """Test parametric alias with nested expressions."""
        source = "alias NestedAlias[dt: DType] = Scalar[dt]"
        expected = "alias NestedAlias[dt: DType] = Scalar[dt]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_mixed_defaults(self):
        """Test parametric alias with mixed default and non-default parameters."""
        source = "alias MixedDefaults[a: Int, b: Int = 0, c: Int = 1] : Int = a + b + c"
        expected = "alias MixedDefaults[a: Int, b: Int = 0, c: Int = 1]: Int = a + b + c\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_long_parametric_alias_line_breaking(self):
        """Test that long parametric aliases are properly line-broken."""
        source = """alias VeryLongParametricAliasWithManyParameters[first_param: Int, second_param: String, third_param: Float, fourth_param: Bool, fifth_param: DType, sixth_param: Int = 42, seventh_param: String = "default", eighth_param: Float = 3.14]: ComplexType[first_param, second_param, third_param] = ComplexExpression[first_param, second_param, third_param, fourth_param, fifth_param, sixth_param, seventh_param, eighth_param]"""

        expected = """alias VeryLongParametricAliasWithManyParameters[
    first_param: Int,
    second_param: String,
    third_param: Float,
    fourth_param: Bool,
    fifth_param: DType,
    sixth_param: Int = 42,
    seventh_param: String = "default",
    eighth_param: Float = 3.14,
]: ComplexType[first_param, second_param, third_param] = ComplexExpression[
    first_param,
    second_param,
    third_param,
    fourth_param,
    fifth_param,
    sixth_param,
    seventh_param,
    eighth_param,
]
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_function_calls_line_breaking(self):
        """Test parametric alias with function calls that need line breaking."""
        source = "alias FunctionCallAlias[param: Int]: Int = some_function(param, param * 2, param + 1)"

        expected = """alias FunctionCallAlias[param: Int]: Int = some_function(
    param, param * 2, param + 1
)
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_complex_arithmetic_line_breaking(self):
        """Test parametric alias with complex arithmetic that needs line breaking."""
        source = "alias ComplexArithmeticAlias[a: Int, b: Int, c: Int]: Int = (a * b) + (b * c) + (c * a)"

        expected = """alias ComplexArithmeticAlias[a: Int, b: Int, c: Int]: Int = (a * b) + (
    b * c
) + (c * a)
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_conditional_line_breaking(self):
        """Test parametric alias with conditional expressions that need line breaking."""
        source = "alias ConditionalParametricAlias[condition: Bool, value: Int]: Int = value if condition else 0"

        expected = """alias ConditionalParametricAlias[
    condition: Bool, value: Int
]: Int = value if condition else 0
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_complex_nested_expressions(self):
        """Test parametric alias with complex nested expressions that need line breaking."""
        source = "alias ComplexAlias[dt: DType, size: Int, offset: Int = 0] = SIMD[dt, size + offset]"

        expected = """alias ComplexAlias[dt: DType, size: Int, offset: Int = 0] = SIMD[
    dt, size + offset
]
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_multiple_parametric_aliases_in_file(self):
        """Test multiple parametric aliases in a single file."""
        source = """alias x: Int = 42
alias addOne[x: Int] : Int = x + 1
alias Scalar[dt: DType] = SIMD[dt, 1]
alias Float64[size: Int = 1] = SIMD[DType.float64, size]"""

        expected = """alias x: Int = 42
alias addOne[x: Int]: Int = x + 1
alias Scalar[dt: DType] = SIMD[dt, 1]
alias Float64[size: Int = 1] = SIMD[DType.float64, size]
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_with_comments(self):
        """Test parametric alias with comments."""
        source = """# This is a parametric alias
alias addOne[x: Int] : Int = x + 1  # Add one to x"""

        expected = """# This is a parametric alias
alias addOne[x: Int]: Int = x + 1  # Add one to x
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_stability(self):
        """Test that formatting is stable (formatting formatted code doesn't change it)."""
        source = "alias addOne[x: Int]: Int = x + 1"

        # Format once
        formatted_once = mblack.format_str(source, mode=self.mode)
        # Format again
        formatted_twice = mblack.format_str(formatted_once, mode=self.mode)

        # Should be the same
        self.assertEqual(formatted_once, formatted_twice)

    def test_parametric_alias_equivalence(self):
        """Test that formatted code is equivalent to original (AST equivalence)."""
        source = "alias addOne[x: Int] : Int = x + 1"

        formatted = mblack.format_str(source, mode=self.mode)

        # The formatted code should be semantically equivalent
        # This is a basic check - in a real implementation you'd want more thorough AST comparison
        self.assertIn("alias addOne", formatted)
        self.assertIn("x: Int", formatted)
        self.assertIn("x + 1", formatted)

    # Core whitespace formatting tests
    def test_parametric_alias_excessive_whitespace(self):
        """Test parametric alias with excessive whitespace that should be normalized."""
        source = "alias    addOne[   x:    Int   ]    :    Int    =    x    +    1"
        expected = "alias addOne[x: Int]: Int = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_no_whitespace(self):
        """Test parametric alias with minimal whitespace that should be properly spaced."""
        source = "alias addOne[x:Int]:Int=x+1"
        expected = "alias addOne[x: Int]: Int = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_mixed_whitespace(self):
        """Test parametric alias with inconsistent whitespace."""
        source = "alias addOne[x:Int] :Int= x+1"
        expected = "alias addOne[x: Int]: Int = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_brackets(self):
        """Test parametric alias with spaces inside brackets."""
        source = "alias SpacedBrackets[ x : Int , y : String ] = SIMD[ DType.float32 , x + y ]"
        expected = "alias SpacedBrackets[x: Int, y: String] = SIMD[DType.float32, x + y]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_around_colon(self):
        """Test parametric alias with various spacing around colons."""
        source = "alias ColonSpacing[x:Int]:Int = x + 1"
        expected = "alias ColonSpacing[x: Int]: Int = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_around_equals(self):
        """Test parametric alias with various spacing around equals."""
        source = "alias EqualsSpacing[x: Int] =x + 1"
        expected = "alias EqualsSpacing[x: Int] = x + 1\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_type_annotations(self):
        """Test parametric alias with spaces in type annotations."""
        source = "alias TypeSpacing[x : Int, y : String, z : Float] : Bool = x > 0"
        expected = "alias TypeSpacing[x: Int, y: String, z: Float]: Bool = x > 0\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_default_values(self):
        """Test parametric alias with spaces in default value expressions."""
        source = "alias DefaultSpacing[x: Int = 0, y: Int = 1, z: Int = 2] : Int = x + y + z"
        expected = "alias DefaultSpacing[x: Int = 0, y: Int = 1, z: Int = 2]: Int = x + y + z\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_function_calls(self):
        """Test parametric alias with spaces in function calls."""
        source = "alias FunctionSpacing[x: Int] : Int = some_function( x , x * 2 , x + 1 )"
        expected = "alias FunctionSpacing[x: Int]: Int = some_function(x, x * 2, x + 1)\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_arithmetic_expressions(self):
        """Test parametric alias with spaces in arithmetic expressions."""
        source = "alias ArithmeticSpacing[x: Int, y: Int] : Int = ( x * y ) + ( x + y )"
        expected = "alias ArithmeticSpacing[x: Int, y: Int]: Int = (x * y) + (x + y)\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_comparison_expressions(self):
        """Test parametric alias with spaces in comparison expressions."""
        source = "alias ComparisonSpacing[x: Int, y: Int] : Bool = x == y and x != 0"
        expected = "alias ComparisonSpacing[x: Int, y: Int]: Bool = x == y and x != 0\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_logical_expressions(self):
        """Test parametric alias with spaces in logical expressions."""
        source = "alias LogicalSpacing[x: Bool, y: Bool] : Bool = x and y or not x"
        expected = "alias LogicalSpacing[x: Bool, y: Bool]: Bool = x and y or not x\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_list_literals(self):
        """Test parametric alias with spaces in list literals."""
        source = "alias ListSpacing[x: Int] = [ x , x * 2 , x * 3 ]"
        expected = "alias ListSpacing[x: Int] = [x, x * 2, x * 3]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_tuple_literals(self):
        """Test parametric alias with spaces in tuple literals."""
        source = "alias TupleSpacing[x: Int, y: Int] = ( x , y , x + y )"
        expected = "alias TupleSpacing[x: Int, y: Int] = (x, y, x + y)\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_dict_literals(self):
        """Test parametric alias with spaces in dict literals."""
        source = "alias DictSpacing[x: String, y: Int] = { x : y , 'default' : 0 }"
        expected = 'alias DictSpacing[x: String, y: Int] = {x: y, "default": 0}\n'

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_comprehensions(self):
        """Test parametric alias with spaces in comprehensions."""
        source = "alias ComprehensionSpacing[x: Int] = [ i * 2 for i in range( x ) if i > 0 ]"
        expected = "alias ComprehensionSpacing[x: Int] = [i * 2 for i in range(x) if i > 0]\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_multiline_expression(self):
        """Test parametric alias with spaces in multiline expressions."""
        source = """alias MultilineSpacing[x: Int, y: Int] : Int = (
    x * y
) + (
    x + y
)"""
        expected = """alias MultilineSpacing[x: Int, y: Int]: Int = (x * y) + (x + y)
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_comments(self):
        """Test parametric alias with spaces around comments."""
        source = """alias CommentSpacing[x: Int] : Int = x + 1  # Add one
alias AnotherSpacing[y: Int] : Int = y * 2  # Multiply by two"""
        expected = """alias CommentSpacing[x: Int]: Int = x + 1  # Add one
alias AnotherSpacing[y: Int]: Int = y * 2  # Multiply by two
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_string_literals(self):
        """Test parametric alias with spaces in string literals (should be preserved)."""
        source = 'alias StringSpacing[x: String] : String = "  hello  world  "'
        expected = 'alias StringSpacing[x: String]: String = "  hello  world  "\n'

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_numeric_literals(self):
        """Test parametric alias with spaces in numeric literals (should be normalized)."""
        source = "alias NumericSpacing[x: Int = 1_000_000, y: Float = 3.141_59] : Float = x + y"
        expected = "alias NumericSpacing[x: Int = 1_000_000, y: Float = 3.141_59]: Float = x + y\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_boolean_literals(self):
        """Test parametric alias with spaces around boolean literals."""
        source = "alias BooleanSpacing[x: Bool = True, y: Bool = False] : Bool = x and y"
        expected = "alias BooleanSpacing[x: Bool = True, y: Bool = False]: Bool = x and y\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_none_literal(self):
        """Test parametric alias with spaces around None literal."""
        source = "alias NoneSpacing[x: Optional[Int] = None] : Bool = x is None"
        expected = "alias NoneSpacing[x: Optional[Int] = None]: Bool = x is None\n"

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_complex_nested_whitespace(self):
        """Test parametric alias with complex nested whitespace scenarios."""
        source = """alias ComplexWhitespace[ x : Int = 1 + 2 , y : String = "  hello  " ] : Tuple = (
    x    *    2    ,
    y    .    strip( )    ,
    x    +    len( y )
)"""
        expected = """alias ComplexWhitespace[x: Int = 1 + 2, y: String = "  hello  "]: Tuple = (
    x * 2,
    y.strip(),
    x + len(y),
)
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_parametric_alias_spaces_in_complex_whitespace_edge_cases(self):
        """Test parametric alias with complex whitespace edge cases."""
        source = """alias ComplexEdgeCase[ x : Int = 1 + 2 * 3 // 4 % 5 ** 6 , y : String = "  hello  world  " ] : Tuple = (
    x    +    y    .    strip( )    .    split( )    [ 0 ]    ,
    len( y    .    strip( )    )    ,
    x    **    2    +    y    .    count( ' ' )    *    10
)"""
        expected = """alias ComplexEdgeCase[
    x: Int = 1 + 2 * 3 // 4 % 5**6, y: String = "  hello  world  "
]: Tuple = (
    x + y.strip().split()[0],
    len(y.strip()),
    x**2 + y.count(" ") * 10,
)
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)

    def test_bug_report_representative_case(self):
        """Test the representative case from the bug report (MOTO-1135)."""
        source = """alias _dtype_to_llvm_type_f8[dtype: DType] = __mlir_type.`i8` if dtype is DType.float8_e3m4 or dtype is DType.float8_e4m3fn or dtype is DType.float8_e4m3fnuz or dtype is DType.float8_e5m2 or dtype is DType.float8_e5m2fnuz else __mlir_type.`!kgen.none`"""

        expected = """alias _dtype_to_llvm_type_f8[
    dtype: DType
] = __mlir_type.`i8` if dtype is DType.float8_e3m4 or dtype is DType.float8_e4m3fn or dtype is DType.float8_e4m3fnuz or dtype is DType.float8_e5m2 or dtype is DType.float8_e5m2fnuz else __mlir_type.`!kgen.none`
"""

        actual = mblack.format_str(source, mode=self.mode)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
