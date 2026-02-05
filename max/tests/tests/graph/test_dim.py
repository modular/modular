# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Tests for Dim algebra operations."""

import pytest
from max.graph import AlgebraicDim, Dim, Graph, StaticDim, SymbolicDim


class TestStaticDimAlgebraNoContext:
    """Tests for static Dim algebra outside of a Graph context.

    Static dim arithmetic should work without an MLIR context by computing
    results in pure Python.
    """

    def test_static_dim_add(self) -> None:
        """Static dims can be added outside Graph context."""
        result = Dim(5) + Dim(10)
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_add_int(self) -> None:
        """Static dims can be added to ints outside Graph context."""
        result = Dim(5) + 10
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_radd_int(self) -> None:
        """Ints can be added to static dims outside Graph context."""
        result = 10 + Dim(5)
        assert isinstance(result, StaticDim)
        assert int(result) == 15

    def test_static_dim_sub(self) -> None:
        """Static dims can be subtracted outside Graph context."""
        result = Dim(10) - Dim(3)
        assert isinstance(result, StaticDim)
        assert int(result) == 7

    def test_static_dim_mul(self) -> None:
        """Static dims can be multiplied outside Graph context."""
        result = Dim(5) * Dim(4)
        assert isinstance(result, StaticDim)
        assert int(result) == 20

    def test_static_dim_mul_int(self) -> None:
        """Static dims can be multiplied by ints outside Graph context."""
        result = Dim(5) * 4
        assert isinstance(result, StaticDim)
        assert int(result) == 20

    def test_static_dim_floordiv(self) -> None:
        """Static dims can be floor divided outside Graph context."""
        result = Dim(10) // Dim(3)
        assert isinstance(result, StaticDim)
        assert int(result) == 3

    def test_static_dim_floordiv_int(self) -> None:
        """Static dims can be floor divided by ints outside Graph context."""
        result = Dim(10) // 3
        assert isinstance(result, StaticDim)
        assert int(result) == 3

    def test_static_dim_neg(self) -> None:
        """Static dims can be negated outside Graph context."""
        result = -Dim(5)
        assert isinstance(result, StaticDim)
        assert int(result) == -5

    def test_static_dim_complex_expression(self) -> None:
        """Complex static dim expressions work outside Graph context."""
        # (5 + 3) * 2 - 4 // 2 = 16 - 2 = 14
        result = (Dim(5) + Dim(3)) * 2 - Dim(4) // 2
        assert isinstance(result, StaticDim)
        assert int(result) == 14

    def test_static_dim_zero_division_raises(self) -> None:
        """Floor division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            Dim(10) // 0

        with pytest.raises(ZeroDivisionError):
            Dim(10) // Dim(0)


class TestSymbolicDimAlgebraNoContext:
    """Tests for symbolic Dim algebra outside of a Graph context.

    Symbolic dim arithmetic requires an MLIR context and should raise
    a helpful error message when attempted outside a Graph context.
    """

    def test_symbolic_dim_add_raises_helpful_error(self) -> None:
        """Symbolic dim addition raises TypeError with workaround hint."""
        with pytest.raises(TypeError, match="graph context"):
            Dim("batch") + Dim(10)

    def test_symbolic_dim_add_raises_mentions_lazy(self) -> None:
        """Error message mentions F.lazy() workaround."""
        with pytest.raises(TypeError, match=r"F\.lazy"):
            Dim("batch") + 1

    def test_symbolic_dim_add_raises_mentions_graph(self) -> None:
        """Error message mentions Graph context workaround."""
        with pytest.raises(TypeError, match="Graph"):
            Dim("seq_len") * 2

    def test_symbolic_dim_mul_raises(self) -> None:
        """Symbolic dim multiplication raises outside Graph context."""
        with pytest.raises(TypeError, match="graph context"):
            Dim("dim") * Dim(4)

    def test_symbolic_dim_floordiv_raises(self) -> None:
        """Symbolic dim floor division raises outside Graph context."""
        with pytest.raises(TypeError, match="graph context"):
            Dim("dim") // 4


class TestDimAlgebraInsideGraphContext:
    """Tests for Dim algebra inside a Graph context.

    All dim arithmetic should work inside a Graph context.
    """

    def test_static_dim_add_inside_graph(self) -> None:
        """Static dims can be added inside Graph context."""
        with Graph("test"):
            result = Dim(5) + Dim(10)
            assert isinstance(result, StaticDim)
            assert int(result) == 15

    def test_symbolic_dim_add_inside_graph(self) -> None:
        """Symbolic dims can be added inside Graph context."""
        with Graph("test"):
            result = Dim("batch") + Dim(10)
            assert isinstance(result, AlgebraicDim)
            assert "batch" in str(result)

    def test_symbolic_dim_mul_inside_graph(self) -> None:
        """Symbolic dims can be multiplied inside Graph context."""
        with Graph("test"):
            result = Dim("dim") * 4
            assert isinstance(result, AlgebraicDim)
            assert "dim" in str(result)

    def test_symbolic_dim_floordiv_inside_graph(self) -> None:
        """Symbolic dims can be floor divided inside Graph context."""
        with Graph("test"):
            result = Dim("dim") // 4
            assert isinstance(result, AlgebraicDim)
            assert "dim" in str(result)

    def test_mixed_symbolic_static_inside_graph(self) -> None:
        """Mixed symbolic and static dims work inside Graph context."""
        with Graph("test"):
            result = (Dim("batch") + 1) * 2
            assert isinstance(result, AlgebraicDim)


class TestDimTypes:
    """Tests for Dim type construction and properties."""

    def test_dim_from_int_is_static(self) -> None:
        """Dim from int creates StaticDim."""
        d = Dim(5)
        assert isinstance(d, StaticDim)
        assert int(d) == 5

    def test_dim_from_str_is_symbolic(self) -> None:
        """Dim from str creates SymbolicDim."""
        d = Dim("batch")
        assert isinstance(d, SymbolicDim)
        assert d.name == "batch"

    def test_dim_from_dim_returns_same(self) -> None:
        """Dim from Dim returns the same object."""
        original = Dim(5)
        result = Dim(original)
        assert result is original

    def test_static_dim_int_conversion(self) -> None:
        """StaticDim can be converted to int."""
        d = StaticDim(42)
        assert int(d) == 42

    def test_symbolic_dim_int_conversion_raises(self) -> None:
        """SymbolicDim cannot be converted to int."""
        d = SymbolicDim("batch")
        with pytest.raises(TypeError, match="static dims"):
            int(d)
