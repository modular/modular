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


@fieldwise_init
struct Sentiment(EqualityComparable, ImplicitlyCopyable):
    var _value: Int

    alias NEGATIVE = Sentiment(0)
    alias NEUTRAL = Sentiment(1)
    alias POSITIVE = Sentiment(2)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)


fn is_happy(s: Sentiment):
    if s == Sentiment.POSITIVE:
        print("Yes. 😀")
    else:
        print("No. ☹️")


def main():
    var mood = Sentiment.POSITIVE
    is_happy(mood)
