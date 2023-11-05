//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::DebugInfo;

//===----------------------------------------------------------------------===//
// OpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    // Always alias source name attributes. They tend to be long.
    if (auto sourceName = dyn_cast<SourceNameAttr>(attr)) {
      if (sourceName.getParamTypes().empty() &&
          sourceName.getArgTypes().empty() &&
          sourceName.getParamValues().empty() && !sourceName.getParent())
        return AliasResult::NoAlias;
      if (llvm::all_of(sourceName.getName(),
                       [](char c) { return std::isalnum(c) || c == '_'; })) {
        os << sourceName.getName().getValue() << "_name";
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    // Essentially all of the debug info attributes are heavy syntax-wise, so
    // just print them all as aliases whenever we can.
    return TypeSwitch<Attribute, AliasResult>(attr)
        .Case<
#define GET_ATTRDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.cpp.inc"
            >([&](auto attr) {
          os << decltype(attr)::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    // Essentially all of the debug info types are heavy syntax-wise, so
    // just print them all as aliases whenever we can.
    return TypeSwitch<Type, AliasResult>(type)
        .Case<
#define GET_TYPEDEF_LIST
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.cpp.inc"
            >([&](auto attr) {
          os << decltype(attr)::getMnemonic();
          return AliasResult::OverridableAlias;
        })
        .Default([](Type) { return AliasResult::NoAlias; });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// DebugInfoDialect
//===----------------------------------------------------------------------===//

void DebugInfoDialect::initialize() {
  registerAttributes();
  registerOperations();
  registerTypes();
  addInterfaces<DebugInfoOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoDialect.cpp.inc"
