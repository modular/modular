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

// Structs with an alignment of 16 bytes and a size of 16 bytes. On AAPCS64,
// the alignment controls the register pair and the stack slot of the
// argument. Linux uses the natural alignment, which ignores an alignment
// attribute on the struct itself but includes the alignment of its fields.
// Darwin uses the full alignment.

typedef struct {
  long long a, b;
} A16Pair;
typedef struct {
  unsigned __int128 x;
} A16Wide;
typedef struct __attribute__((aligned(16))) {
  long long a, b;
} A16P16;
typedef struct __attribute__((aligned(16))) {
  long long a;
} A16S16;
typedef struct {
  A16P16 inner;
} A16NestP16;

// The struct argument goes in registers. On Linux, an argument with a natural
// alignment of 16 starts at an even register, so the pair can go to the stack.
long long align16_reg_p16(long long a, A16P16 u, long long b, long long c,
                          long long d, A16Pair p, long long t) {
  return u.a + 2 * u.b + 4 * p.a + 8 * p.b + 16 * t;
}
long long align16_reg_s16(long long a, A16S16 u, long long b, long long c,
                          long long d, A16Pair p, long long t) {
  return u.a + 4 * p.a + 8 * p.b + 16 * t;
}
long long align16_reg_nestp16(long long a, A16NestP16 u, long long b,
                              long long c, long long d, A16Pair p,
                              long long t) {
  return u.inner.a + 2 * u.inner.b + 4 * p.a + 8 * p.b + 16 * t;
}

// The registers are full, so the struct argument goes on the stack after `s`.
// The alignment of the struct sets its stack offset.
long long align16_stack_wide(long long a0, long long a1, long long a2,
                             long long a3, long long a4, long long a5,
                             long long a6, long long a7, long long s, A16Wide u,
                             long long t) {
  return s + 2 * (long long)u.x + 8 * t;
}
long long align16_stack_p16(long long a0, long long a1, long long a2,
                            long long a3, long long a4, long long a5,
                            long long a6, long long a7, long long s, A16P16 u,
                            long long t) {
  return s + 2 * u.a + 4 * u.b + 8 * t;
}
long long align16_stack_s16(long long a0, long long a1, long long a2,
                            long long a3, long long a4, long long a5,
                            long long a6, long long a7, long long s, A16S16 u,
                            long long t) {
  return s + 2 * u.a + 8 * t;
}
long long align16_stack_nestp16(long long a0, long long a1, long long a2,
                                long long a3, long long a4, long long a5,
                                long long a6, long long a7, long long s,
                                A16NestP16 u, long long t) {
  return s + 2 * u.inner.a + 4 * u.inner.b + 8 * t;
}

long long invoke_align16_reg_p16(long long (*fn)(long long, A16P16, long long,
                                                 long long, long long, A16Pair,
                                                 long long)) {
  return fn(0, (A16P16){3, 5}, 1, 2, 3, (A16Pair){11, 22}, 33);
}
long long invoke_align16_reg_s16(long long (*fn)(long long, A16S16, long long,
                                                 long long, long long, A16Pair,
                                                 long long)) {
  return fn(0, (A16S16){3}, 1, 2, 3, (A16Pair){11, 22}, 33);
}
long long invoke_align16_reg_nestp16(long long (*fn)(long long, A16NestP16,
                                                     long long, long long,
                                                     long long, A16Pair,
                                                     long long)) {
  return fn(0, (A16NestP16){{3, 5}}, 1, 2, 3, (A16Pair){11, 22}, 33);
}
long long invoke_align16_stack_wide(long long (*fn)(
    long long, long long, long long, long long, long long, long long, long long,
    long long, long long, A16Wide, long long)) {
  return fn(0, 1, 2, 3, 4, 5, 6, 7, 9, (A16Wide){27}, 33);
}
long long invoke_align16_stack_p16(long long (*fn)(
    long long, long long, long long, long long, long long, long long, long long,
    long long, long long, A16P16, long long)) {
  return fn(0, 1, 2, 3, 4, 5, 6, 7, 9, (A16P16){3, 5}, 33);
}
long long invoke_align16_stack_s16(long long (*fn)(
    long long, long long, long long, long long, long long, long long, long long,
    long long, long long, A16S16, long long)) {
  return fn(0, 1, 2, 3, 4, 5, 6, 7, 9, (A16S16){3}, 33);
}
long long invoke_align16_stack_nestp16(long long (*fn)(
    long long, long long, long long, long long, long long, long long, long long,
    long long, long long, A16NestP16, long long)) {
  return fn(0, 1, 2, 3, 4, 5, 6, 7, 9, (A16NestP16){{3, 5}}, 33);
}
