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
"""Tests for tensor lifecycle and garbage collection."""

import gc
import weakref

from max.experimental.tensor import Tensor


def test_anonymous_tensors() -> None:
    def build_graph() -> Tensor:
        return Tensor.ones((2, 2)) + 3.0

    z = build_graph()
    gc.collect()

    val = z[0, 0].item()
    assert val == 4.0


def test_tensor_lifecycle_garbage_collection() -> None:
    t = Tensor.ones((10, 10))
    t_ref = weakref.ref(t)

    # Trigger execution. 't' is migrated to the fresh graph.
    str(t + 1.0)
    
    t = t * 2
    # Old 't' is currently held by the idle graph's safety list.
    # Trigger one more execution to flush the list.
    val = t[0, 0].item()
    assert val == 2.0

    gc.collect()
    assert t_ref() is None
