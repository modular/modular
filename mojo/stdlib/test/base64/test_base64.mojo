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

from std.base64 import b16decode, b16encode, b64decode, b64encode


from std.testing import assert_equal, assert_raises
from std.testing import TestSuite


def _bytes_of(str: StringSlice[mut=False, _]) -> List[Byte]:
    return List[Byte](str.as_bytes())


def _random_bytes() -> List[Byte]:
    # fmt: off
    return [
        0xC5, 0x66, 0xFF, 0x7D, 0xC3, 0x1A, 0xC7, 0xFE, 0x5D, 0x2B,
        0x4D, 0x02, 0x4F, 0xE9, 0xD6, 0x34, 0x35, 0xB8, 0x7D, 0xBC,
        0x4E, 0xCC, 0x13, 0x3A, 0x57, 0x0F, 0x3F, 0x0A, 0x7D, 0x0A,
        0xCC, 0xE1, 0xD9, 0x31, 0x97, 0x8B, 0x42, 0xD1, 0x8E, 0x6A,
        0xFF, 0x08, 0x3A,
    ]
    # fmt: on


def test_b64encode() raises:
    assert_equal(b64encode("a"), "YQ==")

    assert_equal(b64encode("fo"), "Zm8=")

    assert_equal(b64encode("Hello Mojo!!!"), "SGVsbG8gTW9qbyEhIQ==")

    assert_equal(b64encode("Hello 🔥!!!"), "SGVsbG8g8J+UpSEhIQ==")

    assert_equal(
        b64encode("the quick brown fox jumps over the lazy dog"),
        "dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==",
    )

    assert_equal(b64encode("ABCDEFabcdef"), "QUJDREVGYWJjZGVm")
    assert_equal(b64encode("\x00\n\x14\x1e(2<FPZdn"), "AAoUHigyPEZQWmRu")
    # 43 random bytes — constructed from a byte list because bytes >= 0x80
    # are not valid standalone UTF-8 characters and can't be expressed in a
    # string literal via `\xNN` (which denotes a codepoint, not a raw byte).
    var random_bytes = _random_bytes()
    assert_equal(
        b64encode(random_bytes),
        "xWb/fcMax/5dK00CT+nWNDW4fbxOzBM6Vw8/Cn0KzOHZMZeLQtGOav8IOg==",
    )


def test_b64decode() raises:
    assert_equal(b64decode("YQ=="), _bytes_of("a"))

    assert_equal(b64decode("Zm8="), _bytes_of("fo"))

    assert_equal(b64decode("SGVsbG8gTW9qbyEhIQ=="), _bytes_of("Hello Mojo!!!"))

    assert_equal(b64decode("SGVsbG8g8J+UpSEhIQ=="), _bytes_of("Hello 🔥!!!"))

    assert_equal(
        b64decode(
            "dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw=="
        ),
        _bytes_of("the quick brown fox jumps over the lazy dog"),
    )

    assert_equal(b64decode("QUJDREVGYWJjZGVm"), _bytes_of("ABCDEFabcdef"))
    assert_equal(
        b64decode(
            "xWb/fcMax/5dK00CT+nWNDW4fbxOzBM6Vw8/Cn0KzOHZMZeLQtGOav8IOg=="
        ),
        _random_bytes(),
    )

    with assert_raises(
        contains="ValueError: Input length '21' must be divisible by 4"
    ):
        _ = b64decode[validate=True]("invalid base64 string")

    with assert_raises(
        contains="ValueError: Unexpected character ' ' encountered"
    ):
        _ = b64decode[validate=True]("invalid base64 string!!!")


def test_b16encode() raises:
    assert_equal(b16encode("a"), "61")

    assert_equal(b16encode("fo"), "666F")

    assert_equal(b16encode("Hello Mojo!!!"), "48656C6C6F204D6F6A6F212121")

    assert_equal(b16encode("Hello 🔥!!!"), "48656C6C6F20F09F94A5212121")

    assert_equal(
        b16encode("the quick brown fox jumps over the lazy dog"),
        "74686520717569636B2062726F776E20666F78206A756D7073206F76657220746865206C617A7920646F67",
    )

    assert_equal(b16encode("ABCDEFabcdef"), "414243444546616263646566")


def test_b16decode() raises:
    assert_equal(b16decode("61"), _bytes_of("a"))

    assert_equal(b16decode("666F"), _bytes_of("fo"))

    assert_equal(
        b16decode("48656C6C6F204D6F6A6F212121"), _bytes_of("Hello Mojo!!!")
    )

    assert_equal(
        b16decode("48656C6C6F20F09F94A5212121"), _bytes_of("Hello 🔥!!!")
    )

    assert_equal(
        b16encode("the quick brown fox jumps over the lazy dog"),
        "74686520717569636B2062726F776E20666F78206A756D7073206F76657220746865206C617A7920646F67",
    )

    assert_equal(
        b16decode("414243444546616263646566"), _bytes_of("ABCDEFabcdef")
    )
    var b16_expected: List[Byte] = [0xC5, 0x66, 0xFF, 0x80]
    assert_equal(b16decode("C566FF80"), b16_expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
