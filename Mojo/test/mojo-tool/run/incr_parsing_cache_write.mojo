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
# Cache-write test for the incremental parsing cache
# (--experimental-incr-parsing): a cold run compiles normally and serializes
# the parsed aggregate (with the module dependency graph embedded) to the
# cache path. Serving from the cache lands separately; this checks the write
# side alone. The printed value is the negative control: the run's output
# must reflect the sources.
#
# The module graph exercises the interesting shapes: a three-level import
# chain (main -> a -> b -> c), a diamond (main -> a -> b and
# main -> d -> b), and an import cycle (a -> b -> c -> a).

# RUN: rm -rf %t && mkdir -p %t
# RUN: printf 'import b\n\ndef fa() -> Int:\n    return b.fb() + 1\n' > %t/a.mojo
# RUN: printf 'import c\n\ndef fb() -> Int:\n    return c.fc() + 1\n' > %t/b.mojo
# RUN: printf 'import a\n\ndef fc() -> Int:\n    return 1\n' > %t/c.mojo
# RUN: printf 'import b\n\ndef fd() -> Int:\n    return b.fb()\n' > %t/d.mojo
# RUN: printf 'import a\nimport d\n\ndef main():\n    print(a.fa() + d.fd())\n' > %t/main.mojo

## Cold run: parses everything, writes the cache.
# RUN: %mojo-no-debug --experimental-incr-parsing %t/cache.mojoc %t/main.mojo | FileCheck %s --check-prefix=COLD-OUT
# COLD-OUT: 5

## The cache file exists and is non-empty.
# RUN: test -s %t/cache.mojoc

## The cache embeds the module dependency graph: a node per source module
## with its xxh3-64 content hash (of the exact bytes written above) and its
## path, every dep edge (including the diamond on b and the c -> a cycle
## edge), and the root marker. (kgen-opt verifies what it reads, so it can
## only dump a cache whose decls all fully resolved - hence the
## fully-referenced sources above.)
# RUN: kgen-opt %t/cache.mojoc | FileCheck %s -DTMP=%t --check-prefix=CACHE
# CACHE: depgraph.graph root @main {
# CACHE-NEXT: depgraph.module @main(%[[A:[0-9]+]], %[[D:[0-9]+]]) <path = "{{.*}}[[TMP]]/main.mojo", content_hash = "fb23cf78bb113aa9">
# CACHE-NEXT: %[[A]] = depgraph.module @a(%[[B:[0-9]+]]) <path = "{{.*}}[[TMP]]/a.mojo", content_hash = "cc86407def3a4aa9">
# CACHE-NEXT: %[[D]] = depgraph.module @d(%[[B]]) <path = "{{.*}}[[TMP]]/d.mojo", content_hash = "37c53bd607b3c7eb">
# CACHE-NEXT: %[[B]] = depgraph.module @b(%[[C:[0-9]+]]) <path = "{{.*}}[[TMP]]/b.mojo", content_hash = "2f548f55c53947fb">
# CACHE-NEXT: %[[C]] = depgraph.module @c(%[[A]]) <path = "{{.*}}[[TMP]]/c.mojo", content_hash = "ac4971d4f37935ef">
# CACHE-NEXT: }

## A declaration the lazy parse never resolved (helper is unreferenced) is
## retained in the cache with a source-resume mark instead of being swept.
## Such a cache is intentionally pre-verification IR - the cache reader opens
## it with verification off - so the dump disables the parse verifier and
## prints in generic form without verifying (an unnamed placeholder `lit.fn`
## asserts in the symbol verifier rather than failing it). The payload is
## [resumeOffset, resumeIndentation, endOffset, declIndentation]: helper's
## `def` starts at byte 50 of a.mojo at top level (indentation 0), and its
## end cursor sits one past the 85-byte buffer.
# RUN: printf 'import b\n\ndef fa() -> Int:\n    return b.fb() + 1\n\ndef helper() -> Int:\n    return 42\n' > %t/a.mojo
# RUN: %mojo-no-debug --experimental-incr-parsing %t/cache_unparsed.mojoc %t/main.mojo | FileCheck %s --check-prefix=UNPARSED-OUT
# UNPARSED-OUT: 5
# RUN: test -s %t/cache_unparsed.mojoc
# RUN: kgen-opt --mlir-very-unsafe-disable-verifier-on-parsing --mlir-print-op-generic --mlir-print-assume-verified %t/cache_unparsed.mojoc | grep -A2 'sourceName = "helper"' | FileCheck %s --check-prefix=UNPARSED-CACHE
# UNPARSED-CACHE: sourceName = "helper"
# UNPARSED-CACHE: lit.incr_parsing_cache_resume_info = array<i64: 50, 0, 86, 0>
