//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "UserLibraryChecker.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "MOGGDecorators.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <list>

namespace M::KGEN::MOGGPreElab {

struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, GeneratorOp, CallOp> {
  using CallGraphNodeBase::CallGraphNodeBase;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  bool shouldAddToGraph(KGENCallOpInterface call, CallGraphNode *node) {
    return true;
  }

  const SymbolTable symtab;
};

bool isDecorator(TypedAttr decorator, StringLiteral annotation) {
  if (auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator))
    if (auto sym = dyn_cast<KGEN::SymbolConstantAttr>(apply.getOperand(0))) {
      StringRef decoratorName = sym.getSymbol().getLeafReference().strref();
      if (decoratorName.starts_with(annotation))
        return true;
    }
  return false;
}

bool hasDecorator(GeneratorOp gen, StringLiteral annotation) {
  llvm::ArrayRef<TypedAttr> decorators = gen.getDecorators();
  return std::any_of(
      decorators.begin(), decorators.end(),
      [&](TypedAttr decorator) { return isDecorator(decorator, annotation); });
}

bool hasAnyDecorator(GeneratorOp gen,
                     llvm::ArrayRef<StringLiteral> annotations) {
  llvm::ArrayRef<TypedAttr> decorators = gen.getDecorators();
  return std::any_of(
      decorators.begin(), decorators.end(), [&](TypedAttr decorator) {
        return std::any_of(annotations.begin(), annotations.end(),
                           [&](const StringLiteral &annot) {
                             return isDecorator(decorator, annot);
                           });
      });
}

} // namespace M::KGEN::MOGGPreElab
