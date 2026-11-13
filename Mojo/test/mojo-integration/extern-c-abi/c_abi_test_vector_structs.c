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
// C reference for vector struct ABI tests.
// Mojo test: test_struct_arguments_vectors.mojo

// GCC vector extension: 2-element float vector (8 bytes, naturally 8-byte
// aligned). Matches the ABI layout of Mojo's SIMD[DType.float32, 2].
typedef float float2 __attribute__((vector_size(8)));

struct VectorStruct8 {
  float2 v;
};

struct VectorStruct8 c_func_vec_8byte(struct VectorStruct8 s) {
  s.v[0] += 1.0f;
  s.v[1] += 1.0f;
  return s;
}

// GCC vector extension: 4-element int vector (16 bytes). Matches the ABI
// layout of Mojo's SIMD[DType.int32, 4].
typedef int int4 __attribute__((vector_size(16)));

// 32 bytes: an HVA of two 128-bit vectors on arm64 (q0-q1), MEMORY on x86-64.
struct IntVectorPair {
  int4 a;
  int4 b;
};

struct IntVectorPair c_func_vec_int_32byte(struct IntVectorPair s) {
  s.a += 1;
  s.b += 1;
  return s;
}
