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
//
// KGEN_CompilerRT_AlignedAlloc(alignment, size). Argument order is the
// opposite of POSIX aligned_alloc. alignment <= 0 uses 16.
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#define DEFAULT_ALIGN ((size_t)16)

void *KGEN_CompilerRT_AlignedAlloc(ssize_t alignment, ssize_t size) {
  if (size <= 0)
    return NULL;

  size_t a = (alignment <= 0) ? DEFAULT_ALIGN : (size_t)alignment;
  /* aligned_alloc requires a power of two; punt non-PoT to the default. */
  if ((a & (a - 1)) != 0)
    a = DEFAULT_ALIGN;
  if (a < sizeof(void *))
    a = sizeof(void *);

  /* C11 aligned_alloc requires size to be a multiple of alignment. */
  size_t s = ((size_t)size + (a - 1)) & ~(a - 1);
  return aligned_alloc(a, s);
}
