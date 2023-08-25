//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// KGENDialectFoldInterface
//===----------------------------------------------------------------------===//

struct KGENDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Never hoist a constant out of a declaration scope. We could scan the
  /// parameters declarations to find the highest scope a constant could be
  /// hoisted into, but that is expensive to do. We also do not hoist constants
  /// out of ops that define a subprogram location scope, since the hoisted
  /// constant would carry incorrect scope information into their new scope.
  bool shouldMaterializeInto(Region *region) const override {
    if (DebugInfo::shouldMaterializeConstantsInto(*region))
      return true;
    return isa<DeclInterface>(region->getParentOp());
  }
};

//===----------------------------------------------------------------------===//
// KGENDialectOpAsmDialectInterface
//===----------------------------------------------------------------------===//

struct KGENDialectOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  //===--------------------------------------------------------------------===//
  // Aliases

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    // Alias DeclRefType if required.
    if (auto ref = dyn_cast<DeclRefType>(type)) {
      if (std::optional<StringRef> aliasName = ref.getAliasName()) {
        os << *aliasName;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    return AliasResult::NoAlias;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void KGENDialect::initialize() {
  registerAttributes();
  registerTypes();
  addInterfaces<KGENDialectFoldInterface, KGENDialectOpAsmDialectInterface>();
  registerBytecodeInterface();
  injectAttrInterfaces();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/KGENDialect/KGEN.cpp.inc"
      >();
}

void KGENDialect::registerKeywordParser(StringRef keyword, TypeParseFn parse) {
  if (!typeParseFns.try_emplace(keyword, parse).second)
    llvm::report_fatal_error("duplicate pretty type keyword: " + keyword);
}

void KGENDialect::registerPrettyType(StringRef keyword, TypeParseFn parse,
                                     TypeID id, TypePrintFn print) {
  registerKeywordParser(keyword, parse);
  if (!typePrintFns.try_emplace(id, print).second)
    llvm::report_fatal_error("duplicate printer for: " + keyword);
  typeNames.try_emplace(id, keyword);
}

std::optional<StringRef> KGENDialect::getTypeName(TypeID id) {
  auto it = typeNames.find(id);
  if (it == typeNames.end())
    return {};
  return it->second;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/KGENDialect/KGENDialect.cpp.inc"
