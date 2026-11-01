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
// This file declares the operation classes for the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPS_H
#define GENERICML_MGPDIALECT_MGPOPS_H

#include "GraphCompiler/Interfaces/EmitsMojoInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPAttrs.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTExtendsLifetimeInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTMayAliasInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CommonAttributes.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class IRRewriter;
class ModuleOp;
} // namespace mlir

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGP.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

namespace M::MGP {

/// Returns the alignment constraint for the given attribute, or none if it
/// has no such constraint.
ErrorOr<std::optional<uint64_t>> getImpliedAlignment(ElementsAttr value);

/// Verifies all the types and attributes of op uses devices drawn from
/// expectedDevices.
LogicalResult
verifyConsistentDevices(Operation *op,
                        std::initializer_list<DeviceRefAttr> expectedDevices);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPOPS_H
