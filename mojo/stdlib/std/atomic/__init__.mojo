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
"""Atomic operations and consistency models.

The `atomic` package provides the `Atomic` type for performing atomic
read-modify-write operations on scalar values, along with the `Consistency`
type for specifying the memory consistency model of those operations. It also
exposes the `fence` function to create standalone memory barriers.

Use this package when implementing lock-free data structures, reference
counting, or any other synchronization primitive that requires fine-grained
control over memory ordering between threads.
"""

from .atomic import Atomic, Consistency, fence
