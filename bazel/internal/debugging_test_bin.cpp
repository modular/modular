//===----------------------------------------------------------------------===//
// Copyright (c) 2025, Modular Inc. All rights reserved.
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

// This binary + library exist to verify that source-level debugging works.
// Notably it is possible that debug info works for the main source file of a
// binary, but not for the dependent libraries or vice versa.
//
// Steps:
// 1. Start debugging: bd //bazel/internal:debugging_test_bin
// 2. Set breakpoints in the library and binary:
//   a: b main
//   b: b test_lib_function
//   c: b test_binary_function
// 3. Run the program: r
// 4. See that it stops, and you can see the source annotations in lldb
// 5. Continue to the next breakpoint: c
// 6. Repeat until the program exits.

#include <iostream>

#include "debugging_test_lib.h"

void test_binary_function() {
  std::cout << "Debugging test binary function called." << std::endl;
}

int main() {
  test_lib_function();
  test_binary_function();
  return 0;
}
