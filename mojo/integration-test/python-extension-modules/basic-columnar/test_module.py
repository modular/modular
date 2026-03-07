# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

# Imports from 'mojo_module.so'
import mojo_module


def test_mojo_columnar() -> None:
    print("Hello from Basic Columnar Example!")

    try:
        _ = mojo_module.DataFrame.with_columns([1.0, 2.0, 3.0], [0.1, 0.2])
        raise Exception("ValueError expected due to unbalanced columns.")
    except Exception as ex:
        assert "not match" in str(ex)

    df = mojo_module.DataFrame.with_columns([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert "DataFrame" in str(df)

    # Test __len__ (mapping protocol mp_length)
    assert len(df) == 3

    # Access rows in the DataFrame
    assert df[0] == (1.0, 0.1)
    assert df[1] == (2.0, 0.2)

    # Test __setitem__ (mapping protocol mp_ass_subscript)
    df[1] = (5.0, 6.0)
    assert df[1] == (5.0, 6.0)
    # Verify other values unchanged
    assert df[0] == (1.0, 0.1)
    assert df[2] == (3.0, 0.3)
    # Test the delete operation in mp_ass_subscript.
    for_delete = mojo_module.DataFrame.with_columns(
        [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]
    )
    del for_delete[0]
    assert for_delete[0] == (2.0, 0.2)

    big_df = mojo_module.DataFrame.with_columns(
        [1.0, 2.0, 30000.0], [0.1, 0.2, 1.0]
    )
    # Test rich compare, LT (0) and EQ(2) are implemented, GT (4) is not.

    def get_rich_compare_counts(
        df: mojo_module.DataFrame,
    ) -> tuple[int, int, int]:
        """Helper function to get the call counts for the rich compare method."""
        return tuple(
            df.get_call_count(f"rich_compare[{op}]") for op in (0, 2, 4)
        )

    assert df < big_df
    assert get_rich_compare_counts(df) == (1, 0, 0)
    assert get_rich_compare_counts(big_df) == (0, 0, 0)

    # Rich_compare GT is not directly implemented so we expect the Python
    # interpreter to call us twice:
    #   - call GT on `big_df` -> NotImplemented.
    #   - call LT on `df` (in addition to the one above).
    assert big_df > df
    assert get_rich_compare_counts(df) == (2, 0, 0)
    assert get_rich_compare_counts(big_df) == (0, 0, 1)

    print("🎉🎉🎉 Mission Success! 🎉🎉🎉")
