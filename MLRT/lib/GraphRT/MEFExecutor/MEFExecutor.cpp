//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file implements various small utilities shared by subsystems of the MEF
// executor.
//
//===----------------------------------------------------------------------===//

#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveDescriptor.h"
using namespace M;

//===----------------------------------------------------------------------===//
// RunnablePrimitiveDescriptor implementation
//===----------------------------------------------------------------------===//

/// Create an AsyncValue with the specified cause, and our current location.
/// This can be set into a result or used for other purposes.
AnyAsyncValueRef RunnablePrimitiveDescriptor::getError(Error cause) {
  return AnyAsyncValueRef::createError(
      cpuDevice, EncodedDiagnostic(std::move(cause), copyLocation()));
}

/// Mark that execution of the primitive failed with the specified reason.
/// This fills in all remaining unspecified results with that error value.
void RunnablePrimitiveDescriptor::fail(Error cause) {
  AnyAsyncValueRef *errorOperand = nullptr;
  for (auto &result : getResults()) {
    // Ignore results already specified.
    if (result)
      continue;

    // If this is our first error operand, allocate the error.
    if (!errorOperand) {
      result = getError(std::move(cause));
      errorOperand = &result;
      continue;
    }

    // Otherwise, reuse the previous error.
    result = errorOperand->copy();
  }
}
