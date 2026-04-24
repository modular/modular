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

"""Integration test for MappingProtocolBuilder (mp_length, mp_subscript, mp_ass_subscript)."""

import mapping_mojo_module as mojo_module  # type: ignore[import-not-found]


def _run_mapping_assertions(cls) -> None:
    obj = cls.from_list([10, 20, 30])

    # __len__ (mp_length)
    assert len(obj) == 3

    # __getitem__ (mp_subscript)
    assert obj[0] == 10
    assert obj[1] == 20
    assert obj[2] == 30

    # __getitem__ out-of-range raises
    try:
        _ = obj[5]
        raise Exception("Expected an error for out-of-range index")
    except Exception as ex:
        assert "range" in str(ex)

    # __setitem__ (mp_ass_subscript — assignment)
    obj[1] = 99
    assert obj[1] == 99
    assert obj[0] == 10  # neighbours unchanged
    assert obj[2] == 30

    # __delitem__ (mp_ass_subscript — deletion)
    d = cls.from_list([1, 2, 3])
    del d[0]
    assert len(d) == 2
    assert d[0] == 2
    assert d[1] == 3

    # Empty list has length 0
    empty = cls()
    assert len(empty) == 0


def test_mapping_protocol() -> None:
    print("Testing mapping protocol...")
    _run_mapping_assertions(mojo_module.SimpleList)
    print("  ptr-receiver: ok")
    _run_mapping_assertions(mojo_module.SimpleListV)
    print("  value-receiver: ok")
    print("Mapping protocol tests passed!")


if __name__ == "__main__":
    test_mapping_protocol()
