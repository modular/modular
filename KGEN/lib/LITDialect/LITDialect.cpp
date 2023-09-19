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
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/Compiler/Bytecode.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
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

using LIT::NoneType;
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
  return StructExtractAttr::get(structValue, field, type);
}

static LogicalResult writeStructExtractAttr(StructExtractAttr attr,
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
    auto structExtract = dyn_cast<StructExtractAttr>(attr);
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
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/LITDialect/LITTypes.cpp.inc"
      >();

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

//===----------------------------------------------------------------------===//
// Type implementations.
//===----------------------------------------------------------------------===//

RefType RefType::get(bool isMutable, TypedAttr elementType,
                     TypedAttr lifetime) {
  auto *ctx = elementType.getContext();
  return get(ctx, isMutable, elementType, lifetime);
}

RefType RefType::get(bool isMutable, Type elementType, TypedAttr lifetime) {
  return get(isMutable, TypeConstantAttr::get(elementType), lifetime);
}

Type RefType::getElementAsType() {
  TypedAttr elemType = getElementType();
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(elemType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(elemType.getType()) &&
         "parameter expr must have metatype type");
  return ParamRefType::get(elemType);
}

/// Return the pointer type that corresponds to this reference type, ignoring
/// the lifetime and the mutability.
PointerType RefType::getAsPointerType() {
  return PointerType::get(getElementType());
}

REPLResultRefType REPLResultRefType::get(Type elementType) {
  auto *ctx = elementType.getContext();
  return get(ctx, elementType);
}

/// Print/Parse a parameter value that is known to have `lifetime` type.
static void printLifetimeParamValue(AsmPrinter &p, TypedAttr value) {
  printParamValue(p, value);
}
static ParseResult parseLifetimeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value,
                         LifetimeType::get(p.getBuilder().getContext()));
}

/// Print/Parse the 'mut' keyword as 1, and its absence as 0.
static void printMutFlag(AsmPrinter &p, bool value) {
  if (value)
    p << "mut ";
}
static ParseResult parseMutFlag(AsmParser &p, bool &value) {
  value = succeeded(p.parseOptionalKeyword("mut"));
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

static ParseResult parseLITSignature(AsmParser &p, Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<Type> inputParamTypes, resultParamTypes;
  if (succeeded(p.parseOptionalLess())) {
    if (p.parseOptionalGreater()) {
      if (succeeded(p.parseOptionalLSquare())) {
        if (p.parseRSquare())
          return failure();
      } else if (p.parseCommaSeparatedList([&] {
                   return parseKGENType(p, inputParamTypes.emplace_back());
                 })) {
        return failure();
      }
      if (succeeded(p.parseOptionalArrow())) {
        if (p.parseCommaSeparatedList([&] {
              return parseKGENType(p, resultParamTypes.emplace_back());
            }))
          return failure();
      }
      if (p.parseGreater())
        return failure();
    }
  }

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaults;
  SmallVector<ValueInputConvention> inputConventions;
  auto parseElt = [&]() -> Type {
    // Parse an optional argument name.
    std::string argName;
    if (succeeded(p.parseOptionalString(&argName)))
      if (failed(p.parseColon()))
        return {};
    argNames.push_back(p.getBuilder().getStringAttr(argName));

    // Parse the argument type and its input convention.
    Type type;
    if (p.parseType(type) ||
        parseInputConvention(p, inputConventions.emplace_back()))
      return {};

    // Parse an optional default value.
    if (succeeded(p.parseOptionalEqual())) {
      TypedAttr value;
      if (parseParamValue(p, value, type))
        return {};
      defaults.push_back(value);
    }

    return type;
  };

  FunctionType functionType;
  FnEffects effects;
  if (parseSignatureValues(p, parseElt, functionType, effects,
                           /*optionalResultList=*/false))
    return failure();
  signature = SignatureType::getChecked(
      [&] { return p.emitError(loc); }, functionType,
      TypeArrayAttr::get(p.getContext(), inputParamTypes),
      TypeArrayAttr::get(p.getContext(), resultParamTypes), inputConventions,
      effects, FnMetadataAttr::get(p.getContext(), argNames, defaults));
  return success(!!signature);
}

Type LITDialect::parseType(DialectAsmParser &p) const {
  llvm::SMLoc typeLoc = p.getCurrentLocation();
  StringRef mnemonic;
  Type genType;
  OptionalParseResult parseResult = generatedTypeParser(p, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Special alias for `!lit.signature` type.
  if (mnemonic == "signature") {
    if (p.parseLess() || parseLITSignature(p, genType) || p.parseGreater())
      return {};
    return genType;
  }

  p.emitError(typeLoc) << "unknown  type `" << mnemonic << "` in dialect `"
                       << getNamespace() << "`";
  return {};
}

void LITDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (succeeded(generatedTypePrinter(type, p)))
    return;
}

// FIXME: Split out a LITTypes.cpp from this file.
LITSignatureType::LITSignatureType(SignatureType sig) : SignatureType(sig) {
  assert(::isa_and_nonnull<FnMetadataAttr>(sig.getMetadata()) &&
         "expected LIT function metadata");
}

FnMetadataAttr LITSignatureType::getMetadata() {
  return ::cast<FnMetadataAttr>(SignatureType::getMetadata());
}

ArrayRef<StringAttr> LITSignatureType::getArgNames() {
  return getMetadata().getArgNames();
}

StringAttr LITSignatureType::getArgName(size_t inputNo) {
  return getArgNames()[inputNo];
}

ArrayRef<TypedAttr> LITSignatureType::getDefaultArguments() {
  return getMetadata().getDefaultArguments();
}

bool LITSignatureType::classof(SignatureType type) {
  return ::isa_and_nonnull<FnMetadataAttr>(type.getMetadata());
}

bool LITSignatureType::classof(Type type) {
  if (auto sig = ::dyn_cast<SignatureType>(type))
    return classof(sig);
  return false;
}

LITSignatureType LITSignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                       TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

LITSignatureType LITSignatureType::get(FunctionType values,
                                       TypeArrayAttr inputParams,
                                       TypeArrayAttr resultParams,
                                       ArrayRef<ValueInputConvention> convs,
                                       FnEffects effects, Attribute metadata) {
  if (!metadata)
    metadata = FnMetadataAttr::get(values.getContext(), values.getNumInputs());
  return SignatureType::get(values, inputParams, resultParams, convs, effects,
                            metadata);
}
