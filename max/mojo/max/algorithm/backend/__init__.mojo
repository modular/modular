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
"""Implements the per-target halves of the data-parallel algorithms.

The entry points in the parent package select CPU or GPU at compile time and
forward into the matching subpackage here. Reach for those entry points rather
than these modules, whose split tracks the dispatch and shifts with it.

Host thread-pool parallelism has no GPU counterpart, so `parallelize()` and its
siblings reach callers only through the CPU side; the GPU side additionally
carries block- and row-level reductions, for kernel authors composing a
reduction inside a larger kernel.
"""
