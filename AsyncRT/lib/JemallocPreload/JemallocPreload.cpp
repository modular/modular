//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

// The library this file belongs to exists only so that jemalloc's allocator
// can be interposed into an already-linked process through LD_PRELOAD, which
// is the only way to replace the allocator of a CPython interpreter that
// someone else started. It has no API of its own.
//
// @jemalloc is not alwayslink and a linker keeps an archive member only if
// some symbol needs it, so an empty translation unit here produces a library
// with no symbols at all. Naming malloc and free pulls in the member that
// defines them, which is the same one that defines the rest of jemalloc's C
// entry points. Taking addresses rather than calling keeps the cost purely at
// link time, and the pointers are non-const so that they have external
// linkage and cannot be elided.
//
// C++ allocation is not interposed here and does not need to be: the
// toolchain puts -lstdc++ ahead of this library's inputs, so a reference to
// operator new binds there rather than to jemalloc, and libstdc++'s operator
// new reaches jemalloc through malloc anyway.

#include <cstddef>
#include <cstdlib>

void *(*modularJemallocMallocAnchor)(std::size_t) = &std::malloc;
void (*modularJemallocFreeAnchor)(void *) = &std::free;
