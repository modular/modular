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
// This file declares the operation classes for the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H

#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class RewriterBase;
} // namespace mlir

namespace M::MOSH {
class ShapeAttr;

/// Try folding MOSH ops.
ErrorOr<TypedAttr> tryFoldOp(Operation *op);
} // namespace M::MOSH

namespace M::KGEN {
class FuncOp;
} // namespace M::KGEN

// Needed because the ODS definition of some KGEN ops don't have the namespace.
using M::KGEN::FuncTypeGeneratorType;
using M::KGEN::ParamDeclArrayAttr;
using M::KGEN::ParamDeclAttr;
using M::KGEN::ParamDefValue;
using M::KGEN::ParameterExprArrayAttr;
using M::KGEN::printStringParam;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MOSHDialect/MOSH.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_H
