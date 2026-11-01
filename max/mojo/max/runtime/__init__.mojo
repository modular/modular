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
"""Provides Mojo access to the runtime that schedules and instruments kernels.

MAX kernels execute under the C++ AsyncRT runtime, and these modules are the
thin Mojo surface onto it: tracing spans and profiling levels that compile away
when profiling is disabled, queries for the parallelism and task assignment of
the device context a kernel was scheduled on, and an owning handle that keeps a
device allocation alive for as long as the runtime still refers to it.

Reach for `tracing` when instrumenting a kernel, and `asyncrt` when work
partitioning depends on the worker count of the context it runs on. Despite the
`asyncrt` name, nothing here is an async programming interface: there is no
coroutine, task-spawning, or future API.
"""
