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

from max.serve.pipelines.stop_detection import StopDetector


def test_stop_none() -> None:
    sd = StopDetector(None)

    assert sd.step("abc") is None


def test_stop_list() -> None:
    sd = StopDetector(["abc", "abcdef"])

    assert sd.step("a") is None
    assert sd.step("b") is None
    assert sd.step("c") == "abc"


def test_stop_str() -> None:
    sd = StopDetector("abc")

    assert sd.step("all good here") is None
    assert sd.step("ab") is None
    assert sd.step("c") == "abc"


def test_long_continuation() -> None:
    sd = StopDetector("abc")

    for x in "long continuation" * 1024:
        assert sd.step(x) is None

    assert sd.step("abc") == "abc"
