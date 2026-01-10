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
        pass

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

    big_df = mojo_module.DataFrame.with_columns(
        [1.0, 2.0, 30000.0], [0.1, 0.2, 1.0]
    )
    assert big_df > df

    print("🎉🎉🎉 Mission Success! 🎉🎉🎉")
