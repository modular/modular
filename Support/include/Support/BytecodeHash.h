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

#ifndef SUPPORT_BYTECODEHASH_H
#define SUPPORT_BYTECODEHASH_H

#include "Support/LogicalResult.h"

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace M {

/// Compute the xxhash for the given MLIR operation.
/// The hash ignores source location information.
FailureOr<std::string> getBytecodeHash(mlir::Operation *op);

} // namespace M
#endif // SUPPORT_BYTECODEHASH_H
