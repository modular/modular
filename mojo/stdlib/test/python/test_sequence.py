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

"""Integration test for SequenceProtocolBuilder (sq_length, sq_item, sq_ass_item, sq_contains, sq_concat, sq_repeat)."""

import sequence_mojo_module as mojo_module  # type: ignore[import-not-found]


def _run_sequence_assertions(cls) -> None:
    obj = cls.from_list([10, 20, 30])

    # __len__ (sq_length)
    assert len(obj) == 3

    # __getitem__ (sq_item — integer index)
    assert obj[0] == 10
    assert obj[1] == 20
    assert obj[2] == 30

    # __getitem__ out-of-range raises
    try:
        _ = obj[5]
        raise Exception("Expected an error for out-of-range index")
    except Exception as ex:
        assert "range" in str(ex)

    # __setitem__ (sq_ass_item — assignment)
    obj[1] = 99
    assert obj[1] == 99
    assert obj[0] == 10  # neighbours unchanged
    assert obj[2] == 30

    # __delitem__ (sq_ass_item — deletion)
    d = cls.from_list([1, 2, 3])
    del d[0]
    assert len(d) == 2
    assert d[0] == 2
    assert d[1] == 3

    # __contains__ (sq_contains)
    s = cls.from_list([1, 2, 3])
    assert 1 in s
    assert 3 in s
    assert 4 not in s

    # __add__ / sq_concat
    a = cls.from_list([1, 2])
    b = cls.from_list([3, 4, 5])
    c = a + b
    assert len(c) == 5
    assert c[0] == 1
    assert c[2] == 3
    assert c[4] == 5

    # __mul__ / sq_repeat
    r = cls.from_list([7, 8]) * 3
    assert len(r) == 6
    assert r[0] == 7
    assert r[1] == 8
    assert r[2] == 7  # second repetition
    assert r[4] == 7  # third repetition

    # Empty sequence
    empty = cls()
    assert len(empty) == 0
    assert 0 not in empty


def test_sequence_protocol() -> None:
    print("Testing sequence protocol...")
    _run_sequence_assertions(mojo_module.Seq)
    print("  ptr-receiver: ok")
    _run_sequence_assertions(mojo_module.SeqV)
    print("  value-receiver: ok")
    print("Sequence protocol tests passed!")


if __name__ == "__main__":
    test_sequence_protocol()
