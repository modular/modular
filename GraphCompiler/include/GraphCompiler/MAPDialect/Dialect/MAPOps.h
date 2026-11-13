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
// This file declares the operation classes for the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H

#include "GraphCompiler/Interfaces/LowerableToMOGGInterface.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAP.h.inc"

namespace M::MAP {

/// The single `map.store` in a single-output kernel or closure `body`,
/// asserting there is exactly one and that it sits directly in `body`, so it
/// stores every element. A multi-output kernel, which horizontal fusion builds,
/// takes `getOutputStores`. Prefer this over assuming the store is the body's
/// terminator.
StoreOp getUniqueStore(mlir::Block &body);

/// Every `map.store` in a kernel or closure `body`, nested ones included, in
/// program order -- one per output, so a multi-output kernel yields several.
/// Asserts there is at least one.
llvm::SmallVector<StoreOp> getOutputStores(mlir::Block &body);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_H
