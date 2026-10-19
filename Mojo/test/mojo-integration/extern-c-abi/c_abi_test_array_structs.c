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
// C reference for array member ABI tests.
// Mojo test: test_struct_arguments_arrays.mojo

// 24-byte struct with one array member of three doubles. AAPCS64 counts the
// array's elements as members, so it is an HFA passed and returned in d0-d2.
// x86-64 passes it in memory (>16 bytes).
struct Vec3d {
  double v[3];
};

struct Vec3d c_func_array_double_24byte(struct Vec3d s) {
  s.v[0] += 1.0;
  s.v[1] += 1.0;
  s.v[2] += 1.0;
  return s;
}

// Calls back into Mojo with a Vec3d built in C.
double c_call_array_double_24byte(double (*fn)(struct Vec3d)) {
  struct Vec3d s = {{1.0, 2.0, 3.0}};
  return fn(s);
}
