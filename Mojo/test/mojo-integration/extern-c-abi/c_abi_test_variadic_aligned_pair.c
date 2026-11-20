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
// Mojo test: test_variadic_aligned_pair.mojo

#include <stdarg.h>

typedef struct {
  long long a, b;
} __attribute__((aligned(16))) AlignedPair;

static long long read_scalar_then_pair(va_list args) {
  long long first = va_arg(args, long long);
  AlignedPair pair = va_arg(args, AlignedPair);
  return first * 10000 + pair.a * 100 + pair.b;
}

long long variadic_aligned_pair_early(int marker, ...) {
  va_list args;
  va_start(args, marker);
  long long result = read_scalar_then_pair(args);
  va_end(args);
  return result;
}

long long variadic_aligned_pair_late(long long a, long long b, long long c,
                                     long long d, long long e, long long f,
                                     long long g, ...) {
  va_list args;
  va_start(args, g);
  long long result = read_scalar_then_pair(args);
  va_end(args);
  return result;
}

typedef struct {
  long long a, b, c;
} __attribute__((aligned(32))) Aligned32;

long long aligned32_after_seven(long long a, long long b, long long c,
                                long long d, long long e, long long f,
                                long long g, Aligned32 s) {
  if ((unsigned long)&s % 32 != 0)
    return -1;
  return g * 1000 + s.a * 100 + s.b * 10 + s.c;
}
