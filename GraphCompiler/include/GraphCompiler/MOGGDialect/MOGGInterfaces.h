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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/LogicalResult.h"

// Needed by the inline `cloneWithMut` default implementation in the generated
// header. Neither includes `MOGGInterfaces.h`, so there is no cycle.
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "mlir/IR/Builders.h"

namespace M::MOGG {
// Enforces `!mogg.buffer => mut` for `MOGG_TensorCreateInterface`. Defined in
// MOGGOpsVerifiers.cpp because the generated `verifyTrait` cannot name the MOGG
// types.
::llvm::LogicalResult
verifyTensorCreateInterfaceMutability(::mlir::Operation *op);

// What `RowwisePhaseOpInterface::getReadCache` returns.
class RowwiseCacheOp;

/// Where a graph blob launch can build a value, as `LaunchOperandInterface`
/// reports it.
enum class LaunchPlacement {
  /// The kernel rebuilds the value from the op's operands.
  Kernel,
  /// The host builds the value, and it crosses the launch as its device type.
  Argument,
  /// The host builds the value, and it can't cross the launch.
  HostOnly,
};
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
