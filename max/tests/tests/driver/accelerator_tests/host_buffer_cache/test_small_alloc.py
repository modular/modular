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


from conftest import MiB, alloc_pinned


def test_small_alloc(buffer_cache_config: None) -> None:
    # The cache has 100MiB so we try to alloc / free 200 buffers of 1MiB each.
    for _ in range(200):
        _ = alloc_pinned(1 * MiB)
