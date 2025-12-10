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

from max.pipelines.dataprocessing import max_tokens_to_generate


def test_max_tokens_to_generate() -> None:
    assert max_tokens_to_generate(10, 12, -1) == 2
    assert max_tokens_to_generate(3, 25, 8) == 8
    assert max_tokens_to_generate(5, 3, -1) == 0
    assert max_tokens_to_generate(10, 10, -1) == 0
    assert max_tokens_to_generate(10, 10, 3) == 0
