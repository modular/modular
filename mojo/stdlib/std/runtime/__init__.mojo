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
"""Runtime services: async execution and program tracing.

The `runtime` package provides infrastructure for asynchronous task execution
and program profiling. It includes low-level concurrency primitives for
managing async coroutines, task groups, and parallel execution. The package
also offers tracing utilities for performance analysis and debugging through
various profiling backends.

Use this package for implementing async patterns, managing concurrent
execution, or when you need detailed profiling and performance analysis of your
Mojo programs.
"""
