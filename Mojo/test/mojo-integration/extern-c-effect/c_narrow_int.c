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
// C reference for test_narrow_int_extension.mojo. Built at -O2, where clang
// trusts the producer of a narrow integer to have extended it, so
// each function hands back the register unchanged and exposes a missing
// extension. At -O0 clang reloads the value from a stack slot, which
// re-extends it and hides the bug.

#include <stdint.h>

uint32_t c_widen_u8(uint8_t x) { return x; }
int32_t c_widen_i8(int8_t x) { return x; }
uint32_t c_widen_u16(uint16_t x) { return x; }
int32_t c_widen_i16(int16_t x) { return x; }

typedef uint32_t (*WidenU8Fn)(uint8_t);
WidenU8Fn c_get_widen_u8(void) { return c_widen_u8; }

uint32_t c_call_u8_fn(uint8_t (*f)(uint32_t), uint32_t x) { return f(x); }
int32_t c_call_i8_fn(int8_t (*f)(uint32_t), uint32_t x) { return f(x); }

// A struct of one narrow integer is coerced to an integer of its size, which
// clang neither extends where it produces it nor trusts where it receives it.
// Mojo can declare these as bare integers, so Mojo must not trust the upper
// bits of one it receives either.
struct U8Box {
  uint8_t x;
};
struct I8Box {
  int8_t x;
};

struct U8Box c_box_u8(uint32_t x) {
  struct U8Box b = {(uint8_t)(x * 7)};
  return b;
}
struct I8Box c_box_i8(uint32_t x) {
  struct I8Box b = {(int8_t)(x * 7)};
  return b;
}

// The callee receives a struct of one byte in a register whose upper bits are
// unspecified, but clang clears them on arm64. So these pass the register as
// a uint64_t, which keeps the struct in its low byte and sets the upper bits.
uint32_t c_call_box_u8_fn(uint32_t (*f)(uint64_t), uint32_t x) {
  return f((uint64_t)x * 7);
}
int32_t c_call_box_i8_fn(int32_t (*f)(uint64_t), uint32_t x) {
  return f((uint64_t)x * 7);
}

struct U8Pair {
  uint8_t a, b;
};

uint32_t c_call_pair_box_u8_fn(uint32_t (*f)(struct U8Pair, uint64_t),
                               uint32_t x) {
  struct U8Pair p = {(uint8_t)(x * 3), (uint8_t)(x * 5)};
  return f(p, (uint64_t)x * 7);
}
