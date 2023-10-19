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
using namespace KGEN;
using namespace LIT;

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

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    return TypeSwitch<Attribute, AliasResult>(attr)
        .Case([&](DocStringAttr attr) {
          // Doc strings are nearly always long, so make sure to print them as
          // aliases.
          os << "doc_string";
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
  }
};

//===----------------------------------------------------------------------===//
// LITDialectBytecodeInterface
//===----------------------------------------------------------------------===//

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

static TypedAttr readStructExtractAttr(DialectBytecodeReader &reader) {
  TypedAttr structValue;
  StringAttr field;
  Type type;
  if (failed(reader.readAttribute(structValue)) ||
      failed(reader.readAttribute(field)) || failed(reader.readType(type)))
    return {};
  return LIT::StructExtractAttr::get(structValue, field, type);
}

static LogicalResult writeStructExtractAttr(LIT::StructExtractAttr attr,
                                            DialectBytecodeWriter &writer) {
  writer.writeAttribute(attr.getStructValue());
  writer.writeAttribute(attr.getField());
  writer.writeType(attr.getType());
  return success();
}

struct LITDialectBytecodeInterface : public mlir::BytecodeDialectInterface {
  LITDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    FailureOr<APInt> isStructExtract = reader.readAPIntWithKnownWidth(1);
    if (failed(isStructExtract))
      return {};
    if (isStructExtract->isOne())
      return readStructExtractAttr(reader);
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    auto structExtract = dyn_cast<LIT::StructExtractAttr>(attr);
    writer.writeAPIntWithKnownWidth(APInt(1, static_cast<bool>(structExtract)));
    if (structExtract)
      return writeStructExtractAttr(structExtract, writer);
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

  // Give the lifetime type a pretty kgen type.
  auto *kgenDialect = getContext()->getOrLoadDialect<KGENDialect>();
  kgenDialect->registerPrettyType(
      "lifetime",
      [](AsmParser &p) -> Type { return LifetimeType::get(p.getContext()); },
      TypeID::get<LifetimeType>(),
      +[](AsmPrinter &p, Type type) { p << "lifetime"; });

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
