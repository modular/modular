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
// C reference for nested struct ABI tests.
// Mojo test: test_struct_arguments_nested.mojo
//
// Tests that Outer { Inner {f32, f32}, f32 } (12 bytes) is passed and returned
// correctly via the System V x86-64 ABI: SSEPair(f64, f32) — eightbyte 0
// holds the two f32s of Inner, eightbyte 1 holds z.

#include <stdint.h>

struct Inner {
  float x;
  float y;
};

// 12-byte struct: Inner (8 bytes, eightbyte 0) + z (4 bytes, eightbyte 1).
struct Outer {
  struct Inner inner;
  float z;
};

struct Outer c_func_nested_float_12byte(struct Outer s) {
  s.inner.x += 1.0f;
  s.inner.y += 1.0f;
  s.z += 1.0f;
  return s;
}

typedef int int2 __attribute__((vector_size(8)));

// 16 bytes: v (eightbyte 0) + Inner (eightbyte 1), both SSE on x86-64.
struct VecInner {
  int2 v;
  struct Inner inner;
};

struct VecInner c_func_vec_inner_16byte(struct VecInner s) {
  s.v += 1;
  s.inner.x += 1.0f;
  s.inner.y += 1.0f;
  return s;
}

struct Vec3 {
  float x, y, z;
};

// 24 bytes, 6 float members: not an HFA, so passed by pointer and returned via
// sret on every target.
struct Box {
  struct Vec3 min, max;
};

struct Box c_func_nested_float_24byte(struct Box b) {
  b.min.x += 1.0f;
  b.min.y += 1.0f;
  b.min.z += 1.0f;
  b.max.x += 1.0f;
  b.max.y += 1.0f;
  b.max.z += 1.0f;
  return b;
}

// 12 bytes: SSEPair on x86-64, the array classified per element.
// An HFA of 3 floats on arm64.
struct FloatArray3 {
  float v[3];
};

struct FloatArray3 c_func_float_array_12byte(struct FloatArray3 s) {
  s.v[0] += 1.0f;
  s.v[1] += 1.0f;
  s.v[2] += 1.0f;
  return s;
}

// An empty struct (a GNU C extension) has size 0 and is skipped: TaggedPair is
// 8 bytes and 2 floats, an HFA on arm64 and one SSE eightbyte on x86-64.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-empty-struct"
struct EmptyTag {};
#pragma clang diagnostic pop

struct TaggedPair {
  float x;
  struct EmptyTag tag;
  float y;
};

struct TaggedPair c_func_empty_member_8byte(struct TaggedPair p) {
  p.x += 1.0f;
  p.y += 1.0f;
  return p;
}
