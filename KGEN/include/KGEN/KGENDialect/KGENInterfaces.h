//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENINTERFACES_H
#define KGEN_KGENDIALECT_KGENINTERFACES_H

#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

namespace mlir {
class IRRewriter;
} // namespace mlir

namespace M::KGEN {
class DeclInterface;
class FunctionLike;
class GeneratorUserOpInterface;
class KGENCallOpInterface;

using InlineResult = std::pair<Operation *, std::function<void(Operation *)>>;

/// This class describes how a parameter value can be defined. A parameter
/// definition can depend on a number of parameter expressions and regions.
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
///
/// Example:
///
/// ```mlir
/// kgen.param.declare.region Fn = <A>() -> index {
///   %0 = kgen.param.constant = <add(A, B)>
///   kgen.return %0 : index
/// }
/// ```
///
/// The definition of `Fn` depends on one region: the body of the lambda
/// function. It therefore depends on any parameters captured by the function.
/// Here, the parameter use-def graph will determine that the definition of `Fn`
/// depends on the parameter declaration `B`.
///
/// Example:
///
/// ```mlir
/// kgen.param.if <lt(C, 1) -> output> {
///   kgen.param.yield<A>
/// } else {
///   kgen.param.yield<B>
/// }
///
/// The definition of the result parameter `output` depends on the parameter
/// expression `lt(C, 1)` and both regions of the `kgen.param.if`. The parameter
/// use-def graph will determine that the definition of `output` depends
/// directly on the parameters `A`, `B`, and `C`.
struct ParamDefValue {
  ParamDefValue() {}
  ParamDefValue(Attribute expr) : exprs(1, expr) {}
  ParamDefValue(Region *region) : regions(1, region) {}
  ParamDefValue(ArrayRef<Attribute> exprs, ArrayRef<Region *> regions)
      : exprs(exprs), regions(regions) {}

  SmallVector<Attribute, 1> exprs;
  SmallVector<Region *, 0> regions;
};

namespace impl {
void scanAllAttrsAndTypes(Operation *op, function_ref<void(Attribute)> scanAttr,
                          function_ref<void(Type)> scanType);
LogicalResult verifyFunctionLike(FunctionLike op);
LogicalResult verifyGeneratorUser(GeneratorUserOpInterface op);
LogicalResult verifyIfTopLevel(DeclInterface decl,
                               SymbolTableCollection &symtab);
LogicalResult verifyExportInterface(Operation *op);
} // namespace impl
} // namespace M::KGEN

#include "KGEN/KGENDialect/KGENInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENINTERFACES_H
