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

// NOTE: This test is intentionally C to validate building a C file importing
// the C API works

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/symbol_export.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

int main() {
  M_Dtype dtype = M_UNKNOWN;
  return dtype;
}
