//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/CODialect/CODialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/Compiler/Bytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN::LIT;
using KGEN::ArgConvention;
using KGEN::DeclInterface;

//===----------------------------------------------------------------------===//
// LITDialectFoldInterface
//===----------------------------------------------------------------------===//

namespace {
struct LITDialectFoldInterface : public mlir::DialectFoldInterface {
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
// LITOpAsmDialectInterface
//===----------------------------------------------------------------------===//

struct LITOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    // Alias StructType if required.
    if (auto ref = dyn_cast<StructType>(type)) {
      if (std::optional<StringRef> aliasName = ref.getAliasName()) {
        os << *aliasName;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    if (auto trait = dyn_cast<TraitType>(type)) {
      if (std::optional<StringRef> name =
              StructType::getAliasName(trait.getSymbol())) {
        os << *name;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    if (auto meta = dyn_cast<AnyStructType>(type)) {
      if (!meta.getParamValues().empty())
        return AliasResult::NoAlias;
      if (std::optional<StringRef> name =
              StructType::getAliasName(meta.getSymbol())) {
        os << "mt_" << *name;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    if (isa<DocStringAttr>(attr)) {
      // Doc strings are nearly always long, so make sure to print them as
      // aliases.
      os << "doc_string";
      return AliasResult::OverridableAlias;
    }

    if (auto symbol = dyn_cast<SymbolAttr>(attr)) {
      if (std::optional<StringRef> alias =
              StructType::getAliasName(symbol.getValue())) {
        os << *alias;
        return AliasResult::OverridableAlias;
      }
      return AliasResult::NoAlias;
    }

    return AliasResult::NoAlias;
  }
};

//===----------------------------------------------------------------------===//
// LITDialectBytecodeInterface
//===----------------------------------------------------------------------===//

using WrappedBindTypeAttr = WrappedAttrType<BindTypeAttr>;
using WrappedStructExtractAttr = WrappedAttrType<StructExtractAttr>;
using WrappedLifetimeUnionAttr = WrappedAttrType<LifetimeUnionAttr>;
using WrappedLifetimeMutCastAttr = WrappedAttrType<LifetimeMutCastAttr>;
using WrappedLifetimeSetAttr = WrappedAttrType<LifetimeSetAttr>;
using WrappedLifetimeSetUnionAttr = WrappedAttrType<LifetimeSetUnionAttr>;

//===----------------------------------------------------------------------===//
// Utilities

using mlir::DialectBytecodeReader;
using mlir::DialectBytecodeWriter;
using mlir::get;

static LogicalResult
readStructValues(DialectBytecodeReader &reader,
                 SmallVectorImpl<std::tuple<StringAttr, TypedAttr>> &values) {
  return reader.readList(values, [&](std::tuple<StringAttr, TypedAttr> &value) {
    if (failed(reader.readAttribute(std::get<0>(value))) ||
        failed(reader.readAttribute(std::get<1>(value))))
      return failure();
    return LogicalResult::success();
  });
}

static void
writeStructValues(DialectBytecodeWriter &writer,
                  ArrayRef<std::tuple<StringAttr, TypedAttr>> values) {
  writer.writeList(values, [&](auto &value) {
    writer.writeAttribute(std::get<0>(value));
    writer.writeAttribute(std::get<1>(value));
  });
}

#include "KGEN/LITDialect/LITDialectBytecode.cpp.inc"

struct LITDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  LITDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/LITDialect/LITDialect.cpp.inc"

void LITDialect::initialize() {
  // Register attributes.
  registerAttributes();
  addInterfaces<LITDialectFoldInterface, LITOpAsmDialectInterface>();

  // Register types.
  registerTypes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/LITDialect/LIT.cpp.inc"
      >();

  addInterface<LITDialectBytecodeInterface>();
}

Operation *LITDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return b.create<ParamConstantOp>(loc, type, cast<TypedAttr>(value));
}
