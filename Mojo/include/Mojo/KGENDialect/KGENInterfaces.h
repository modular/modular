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

#ifndef KGEN_KGENDIALECT_KGENINTERFACES_H
#define KGEN_KGENDIALECT_KGENINTERFACES_H

#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
class IRRewriter;
class RewriterBase;
} // namespace mlir

namespace M::KGEN {
class DeclInterface;
class FunctionLike;
class GeneratorUserOpInterface;
class KGENCallOpInterface;

using InlineResult = std::pair<Operation *, std::function<void(Operation *)>>;

/// This class describes how a parameter value can be defined. A parameter
/// definition can depend on a number of parameter expressions.
///
/// Example:
///
/// ```mlir
/// kgen.param.declare A = <add(B, 1)>
/// ```
///
/// The definition of `A` depends on the parameter expression `add(B, 1)`. The
/// parameter use-def graph will scan the expression to determine a use edge
/// from the definition of `A` to the parameter declaration `B`.
struct ParamDefValue {
  ParamDefValue() {}
  ParamDefValue(Attribute expr) : exprs(1, expr) {}

  SmallVector<Attribute, 1> exprs;
};

namespace impl {
void scanAllAttrsAndTypes(Operation *op, function_ref<void(Attribute)> scanAttr,
                          function_ref<void(Type)> scanType);
LogicalResult verifyFunctionLike(FunctionLike op);
LogicalResult verifyGeneratorUser(GeneratorUserOpInterface op);
LogicalResult verifyIfTopLevel(DeclInterface decl,
                               SymbolTableCollection &symtab);
LogicalResult verifyExportInterface(Operation *op);
Operation *lookupConformance(Operation *op, ParameterEvaluator &evaluator,
                             TraitSymbolAttr traitSymbol);
} // namespace impl
} // namespace M::KGEN

#include "Mojo/KGENDialect/KGENInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENINTERFACES_H
