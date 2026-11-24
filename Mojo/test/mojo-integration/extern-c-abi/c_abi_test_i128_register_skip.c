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

typedef unsigned __int128 SkipScalar;
typedef struct {
  SkipScalar x;
} SkipWide;
typedef long long SkipVector __attribute__((vector_size(16)));
typedef struct {
  SkipVector x;
} SkipSIMD;
typedef struct {
  long long a, b;
} SkipPair;

long long i128_scalar_even(SkipScalar u, long long b, long long c, long long d,
                           SkipPair pair, long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}
long long i128_scalar_odd(long long a, SkipScalar u, long long b, long long c,
                          long long d, SkipPair pair, long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}
long long i128_wide_even(SkipWide u, long long b, long long c, long long d,
                         SkipPair pair, long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}
long long i128_wide_odd(long long a, SkipWide u, long long b, long long c,
                        long long d, SkipPair pair, long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}
long long i128_simd_even(SkipSIMD u, long long b, long long c, long long d,
                         long long e, long long f, SkipPair pair,
                         long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}
long long i128_simd_odd(long long a, SkipSIMD u, long long b, long long c,
                        long long d, long long e, long long f, SkipPair pair,
                        long long tail) {
  return pair.a + 2 * pair.b + 4 * tail;
}

long long invoke_i128_scalar_even(long long (*fn)(SkipScalar, long long,
                                                  long long, long long,
                                                  SkipPair, long long)) {
  return fn(27, 1, 2, 3, (SkipPair){11, 22}, 33);
}
long long invoke_i128_scalar_odd(long long (*fn)(long long, SkipScalar,
                                                 long long, long long,
                                                 long long, SkipPair,
                                                 long long)) {
  return fn(0, 27, 1, 2, 3, (SkipPair){11, 22}, 33);
}
long long invoke_i128_wide_even(long long (*fn)(SkipWide, long long, long long,
                                                long long, SkipPair,
                                                long long)) {
  return fn((SkipWide){27}, 1, 2, 3, (SkipPair){11, 22}, 33);
}
long long invoke_i128_wide_odd(long long (*fn)(long long, SkipWide, long long,
                                               long long, long long, SkipPair,
                                               long long)) {
  return fn(0, (SkipWide){27}, 1, 2, 3, (SkipPair){11, 22}, 33);
}
long long invoke_i128_simd_even(long long (*fn)(SkipSIMD, long long, long long,
                                                long long, long long, long long,
                                                SkipPair, long long)) {
  return fn((SkipSIMD){{27, 28}}, 1, 2, 3, 4, 5, (SkipPair){11, 22}, 33);
}
long long invoke_i128_simd_odd(long long (*fn)(long long, SkipSIMD, long long,
                                               long long, long long, long long,
                                               long long, SkipPair,
                                               long long)) {
  return fn(0, (SkipSIMD){{27, 28}}, 1, 2, 3, 4, 5, (SkipPair){11, 22}, 33);
}
