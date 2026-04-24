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

"""Integration test for the columnar DataFrame example.

Run via:
    pixi run test-example
"""

import mojo.importer  # noqa: F401, I001

import columnar_mojo  # type: ignore


def test_mojo_columnar() -> None:
    print("Hello from Basic Columnar Example!")

    # Mismatched column lengths should raise.
    try:
        _ = columnar_mojo.DataFrame.with_columns([1.0, 2.0, 3.0], [0.1, 0.2])
        raise Exception("ValueError expected due to unbalanced columns.")
    except Exception as ex:
        assert "not match" in str(ex)

    df = columnar_mojo.DataFrame.with_columns([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert "DataFrame" in str(df)

    # __len__ (mp_length)
    assert len(df) == 3

    # __getitem__ (mp_subscript)
    assert df[0] == (1.0, 0.1)
    assert df[1] == (2.0, 0.2)

    # __setitem__ (mp_ass_subscript — assignment)
    df[1] = (5.0, 6.0)
    assert df[1] == (5.0, 6.0)
    assert df[0] == (1.0, 0.1)  # neighbours unchanged
    assert df[2] == (3.0, 0.3)

    # __delitem__ (mp_ass_subscript — deletion)
    for_delete = columnar_mojo.DataFrame.with_columns(
        [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]
    )
    del for_delete[0]
    assert for_delete[0] == (2.0, 0.2)

    big_df = columnar_mojo.DataFrame.with_columns(
        [1.0, 2.0, 30000.0], [0.1, 0.2, 1.0]
    )

    def rich_compare_counts(
        d: columnar_mojo.DataFrame,
    ) -> tuple[int, int, int]:
        return tuple(
            d.get_call_count(f"rich_compare[{op}]") for op in (0, 2, 4)
        )

    # LT and EQ are implemented directly.
    assert df < big_df
    assert rich_compare_counts(df) == (1, 0, 0)
    assert rich_compare_counts(big_df) == (0, 0, 0)

    # GT is not implemented, so Python retries as LT on the other operand.
    assert big_df > df
    assert rich_compare_counts(df) == (2, 0, 0)
    assert rich_compare_counts(big_df) == (0, 0, 1)

    # ------------------------------------------------------------------
    # Number protocol
    # ------------------------------------------------------------------

    num_df = columnar_mojo.DataFrame.with_columns(
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    )

    # __neg__ (nb_negative)
    neg_df = -num_df
    assert neg_df[0] == (-1.0, -4.0)
    assert neg_df[2] == (-3.0, -6.0)

    # __abs__ (nb_absolute)
    abs_df = abs(neg_df)
    assert abs_df[0] == (1.0, 4.0)
    assert abs_df[2] == (3.0, 6.0)

    # __bool__ (nb_bool)
    assert bool(num_df)
    assert not bool(columnar_mojo.DataFrame())

    # __add__ (nb_add) — concatenate two DataFrames
    df_a = columnar_mojo.DataFrame.with_columns([1.0, 2.0], [3.0, 4.0])
    df_b = columnar_mojo.DataFrame.with_columns([5.0, 6.0], [7.0, 8.0])
    combined = df_a + df_b
    assert len(combined) == 4
    assert combined[0] == (1.0, 3.0)
    assert combined[2] == (5.0, 7.0)

    # __add__ with a non-DataFrame returns NotImplemented, so Python raises TypeError
    try:
        _ = df_a + 42
        raise Exception("TypeError expected")
    except TypeError:
        pass

    # __mul__ (nb_multiply) — scale coordinates by a scalar
    scaled = df_a * 2.0
    assert scaled[0] == (2.0, 6.0)
    assert scaled[1] == (4.0, 8.0)

    # __mul__ with a non-numeric returns NotImplemented → TypeError
    try:
        _ = df_a * "oops"
        raise Exception("TypeError expected")
    except TypeError:
        pass

    # __pow__ (nb_power) — raise coordinates to a power
    squared = df_a**2
    assert squared[0] == (1.0, 9.0)
    assert squared[1] == (4.0, 16.0)

    print("🎉🎉🎉 Mission Success! 🎉🎉🎉")


if __name__ == "__main__":
    test_mojo_columnar()
