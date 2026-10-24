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

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H
#define MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

namespace M::GRT {

/// Indicates whether a control flow op is 'loop' like or 'condition' like.
enum ControlFlowFlavor { kLoopLike, kBranchLike };

// Verifies correctness of op implementing GRTControlFlowInterface.
LogicalResult verifyControlFlowInterfaceOp(Operation *op);

} // namespace M::GRT

#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h.inc"

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_H
