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
// C ABI reference for HFA member-counting tests
// Mojo test: test_struct_arguments_hfa_counting.mojo
//
// AAPCS64 counts HFA members after flattening nested structs: a struct of
// two 3-float structs has six members, exceeds the four-member limit, and is
// passed by pointer to a caller copy (returned through x8). A lowering that
// counts top-level fields instead keeps such a struct an HFA and passes it in
// v0-v5, which a C caller never matches.

typedef struct vec2_t {
  float x;
  float y;
} vec2_t;

typedef struct vec3_t {
  float x;
  float y;
  float z;
} vec3_t;

// Six flattened members: not an HFA. 24 bytes: indirect arg + sret return.
typedef struct box_t {
  vec3_t min;
  vec3_t max;
} box_t;

// Four flattened members: still an HFA (v0-v3).
typedef struct hfa4_t {
  vec2_t a;
  float b;
  float c;
} hfa4_t;

// Five flattened members: not an HFA. 20 bytes: indirect arg + sret return.
typedef struct hfa5_t {
  vec2_t a;
  vec2_t b;
  float c;
} hfa5_t;

box_t c_func_box(box_t b) {
  b.min.x += 1.0f;
  b.min.y += 1.0f;
  b.min.z += 1.0f;
  b.max.x += 1.0f;
  b.max.y += 1.0f;
  b.max.z += 1.0f;
  return b;
}

hfa4_t c_func_hfa4(hfa4_t s) {
  s.a.x += 1.0f;
  s.a.y += 1.0f;
  s.b += 1.0f;
  s.c += 1.0f;
  return s;
}

hfa5_t c_func_hfa5(hfa5_t s) {
  s.a.x += 1.0f;
  s.a.y += 1.0f;
  s.b.x += 1.0f;
  s.b.y += 1.0f;
  s.c += 1.0f;
  return s;
}
