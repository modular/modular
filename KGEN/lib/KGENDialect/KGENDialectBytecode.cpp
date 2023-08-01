//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;

using BytecodeReader = mlir::DialectBytecodeReader;
using BytecodeWriter = mlir::DialectBytecodeWriter;

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace Encoding {
/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.
enum AttributeCode {
  ///
  ///   ParameterExprArrayAttr {
  ///     value: Attribute[]
  ///   }
  kParameterExprArrayAttr = 0,
  ///
  ///   StringArrayAttr {
  ///     value: Attribute[]
  ///   }
  kStringArrayAttr = 1,
  ///
  ///   TypeArrayAttr {
  ///     value: Attribute[]
  ///   }
  kTypeArrayAttr = 2,
  ///
  ///   ParamDeclAttr {
  ///     name: StringAttr
  ///     type: Type
  ///   }
  kParamDeclAttr = 3,
  ///
  ///   ParamDeclArrayAttr {
  ///     value: Attribute[]
  ///   }
  kParamDeclArrayAttr = 4,
  ///
  ///   ParamBindAttr {
  ///     name: StringAttr
  ///     value: TypedAttr
  ///   }
  kParamBindAttr = 5,
  ///
  ///   ParamBindArrayAttr {
  ///     value: Attribute[]
  ///   }
  kParamBindArrayAttr = 6,
  ///
  ///   ConstraintAttr {
  ///     expr: TypedAttr
  ///     message: StringAttr
  ///     loc: LocationAttr
  ///   }
  kConstraintAttr = 7,
  ///
  ///   ConstraintArrayAttr {
  ///     value: Attribute[]
  ///   }
  kConstraintArrayAttr = 8,
  ///
  ///   FnMetadataAttr {
  ///     inputConventions: varint[]
  ///     defaultArguments: TypedAttr[]
  ///     fnEffects: varint
  ///   }
  kFnMetadataAttr = 9,
  ///
  ///   VariadicAttr {
  ///     values: TypedAttr[]
  ///     type: VariadicType
  ///   }
  kVariadicAttr = 10,
  ///
  ///   UnknownAttr {
  ///     type: Type
  ///   }
  kUnknownAttr = 11,
  ///
  ///   UnboundAttr {
  ///     type: Type
  ///   }
  kUnboundAttr = 12,
  ///
  ///   ParamDeclRefAttr {
  ///     name: StringAttr
  ///     type: Type
  ///   }
  kParamDeclRefAttr = 13,
  ///
  ///   ParamIndexRefAttr {
  ///     depth: varint
  ///     isResult: varint
  ///     index: varint
  ///     type: Type
  ///   }
  kParamIndexRefAttr = 14,
  ///
  ///   ConcreteTypeConstantAttr {
  ///     value: Type
  ///   }
  kConcreteTypeConstantAttr = 15,
  ///
  ///   ParameterizedTypeConstantAttr {
  ///     value: Type
  ///   }
  kParameterizedTypeConstantAttr = 16,
  ///
  ///   DTypeConstantAttr {
  ///     dtype: varint
  ///   }
  kDTypeConstantAttr = 17,
  ///
  ///   SymbolConstantAttr {
  ///     symbol: SymbolRefAttr
  ///     paramValues: ParamBindArrayAttr
  ///     type: SignatureType
  ///   }
  kSymbolConstantAttr = 18,
  ///
  ///   TargetParamAttr {
  ///     target: TargetInfoAttr
  ///   }
  kTargetParamAttr = 19,
  ///
  ///   BuildInfoParamAttr {
  ///     info: BuildInfoAttr
  ///   }
  kBuildInfoParamAttr = 20,
  ///
  ///   ParamOperatorAttr {
  ///     opcode: varint
  ///     operands: TypedAttr[]
  ///     type: Type
  ///   }
  kParamOperatorAttr = 21,
  ///
  ///   MLIROpAttr {
  ///     name: StringAttr
  ///     attrs: DictionaryAttr
  ///     type: SignatureType
  ///   }
  kMLIROpAttr = 22,
};

/// This enum contains marker codes used to indicate which type is currently
/// being decoded, and how it should be decoded. The order of these codes should
/// generally be unchanged, as any changes will inevitably break compatibility
/// with older bytecode.
enum TypeCode {
  ///
  ///   ParamRefType {
  ///     param: TypedAttr
  ///   }
  kParamRefType = 0,
  ///
  ///   MLIRTypeType {
  ///   }
  kMLIRTypeType = 1,
  ///
  ///   DTypeType {
  ///   }
  kDTypeType = 2,
  ///
  ///   StringType {
  ///   }
  kStringType = 3,
  ///
  ///   SignatureType {
  ///     inputParams: ParamDeclArrayAttr
  ///     resultParams: ParamDeclArrayAttr
  ///     values: FunctionType
  ///     metadata: FnMetadataAttr
  ///   }
  kSignatureType = 4,
  ///
  ///   DeclRefType {
  ///     symbol: SymbolRefAttr
  ///     paramValues: ParamBindArrayAttr
  ///   }
  kDeclRefType = 5,
  ///
  ///   TargetType {
  ///   }
  kTargetType = 6,
  ///
  ///   BuildInfoType {
  ///   }
  kBuildInfoType = 7,
  ///
  ///   VariadicType {
  ///     elementType: TypedAttr
  ///   }
  kVariadicType = 8,
};

} // namespace Encoding
} // namespace

//===----------------------------------------------------------------------===//
// KGENBytecodeInterface
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the bytecode interface for the KGEN dialect.
struct KGENBytecodeInterface : public mlir::BytecodeDialectInterface {
  KGENBytecodeInterface(Dialect *dialect) : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(BytecodeReader &reader) const override;
  template <typename T>
  T readArrayOfAttrs(BytecodeReader &reader) const;
  template <typename T>
  T readArrayOfTypes(BytecodeReader &reader) const;

  BuildInfoParamAttr readBuildInfoParamAttr(BytecodeReader &reader) const;
  Attribute readConcreteTypeConstantAttr(BytecodeReader &reader) const;
  ConstraintAttr readConstraintAttr(BytecodeReader &reader) const;
  DTypeConstantAttr readDTypeConstantAttr(BytecodeReader &reader) const;
  FnMetadataAttr readFnMetadataAttr(BytecodeReader &reader) const;
  Attribute readMLIROpAttr(BytecodeReader &reader) const;
  ParamBindAttr readParamBindAttr(BytecodeReader &reader) const;
  ParamDeclAttr readParamDeclAttr(BytecodeReader &reader) const;
  Attribute readParamOperatorAttr(BytecodeReader &reader) const;
  Attribute readParameterizedTypeConstantAttr(BytecodeReader &reader) const;
  ParamDeclRefAttr readParamDeclRefAttr(BytecodeReader &reader) const;
  ParamIndexRefAttr readParamIndexRefAttr(BytecodeReader &reader) const;
  SymbolConstantAttr readSymbolConstantAttr(BytecodeReader &reader) const;
  TargetParamAttr readTargetParamAttr(BytecodeReader &reader) const;
  UnboundAttr readUnboundAttr(BytecodeReader &reader) const;
  UnknownAttr readUnknownAttr(BytecodeReader &reader) const;
  VariadicAttr readVariadicAttr(BytecodeReader &reader) const;

  LogicalResult writeAttribute(Attribute attr,
                               BytecodeWriter &writer) const override;
  template <typename T>
  LogicalResult writeArrayOfAttrs(T attr, uint64_t attrCode,
                                  BytecodeWriter &writer) const;
  template <typename T>
  LogicalResult writeArrayOfTypes(T attr, uint64_t attrCode,
                                  BytecodeWriter &writer) const;
  void write(BuildInfoParamAttr attr, BytecodeWriter &writer) const;
  void write(ConcreteTypeConstantAttr attr, BytecodeWriter &writer) const;
  void write(ConstraintAttr attr, BytecodeWriter &writer) const;
  void write(DTypeConstantAttr attr, BytecodeWriter &writer) const;
  void write(FnMetadataAttr attr, BytecodeWriter &writer) const;
  void write(MLIROpAttr attr, BytecodeWriter &writer) const;
  void write(ParamBindAttr attr, BytecodeWriter &writer) const;
  void write(ParamDeclAttr attr, BytecodeWriter &writer) const;
  void write(ParamDeclRefAttr attr, BytecodeWriter &writer) const;
  void write(ParamIndexRefAttr attr, BytecodeWriter &writer) const;
  void write(ParamOperatorAttr attr, BytecodeWriter &writer) const;
  void write(ParameterizedTypeConstantAttr attr, BytecodeWriter &writer) const;
  void write(SymbolConstantAttr attr, BytecodeWriter &writer) const;
  void write(TargetParamAttr attr, BytecodeWriter &writer) const;
  void write(UnboundAttr attr, BytecodeWriter &writer) const;
  void write(UnknownAttr attr, BytecodeWriter &writer) const;
  void write(VariadicAttr attr, BytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(BytecodeReader &reader) const override;
  Type readDeclRefType(BytecodeReader &reader) const;
  Type readParamRefType(BytecodeReader &reader) const;
  Type readSignatureType(BytecodeReader &reader) const;
  Type readVariadicType(BytecodeReader &reader) const;

  LogicalResult writeType(Type type, BytecodeWriter &writer) const override;
  void write(DeclRefType type, BytecodeWriter &writer) const;
  void write(ParamRefType type, BytecodeWriter &writer) const;
  void write(SignatureType type, BytecodeWriter &writer) const;
  void write(VariadicType type, BytecodeWriter &writer) const;
};
} // namespace

void KGENDialect::registerBytecodeInterface() {
  addInterfaces<KGENBytecodeInterface>();
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

Attribute KGENBytecodeInterface::readAttribute(BytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case Encoding::kParameterExprArrayAttr:
    return readArrayOfAttrs<ParameterExprArrayAttr>(reader);
  case Encoding::kStringArrayAttr:
    return readArrayOfAttrs<StringArrayAttr>(reader);
  case Encoding::kTypeArrayAttr:
    return readArrayOfTypes<TypeArrayAttr>(reader);
  case Encoding::kParamDeclAttr:
    return readParamDeclAttr(reader);
  case Encoding::kParamDeclArrayAttr:
    return readArrayOfAttrs<ParamDeclArrayAttr>(reader);
  case Encoding::kParamBindAttr:
    return readParamBindAttr(reader);
  case Encoding::kParamBindArrayAttr:
    return readArrayOfAttrs<ParamBindArrayAttr>(reader);
  case Encoding::kConstraintAttr:
    return readConstraintAttr(reader);
  case Encoding::kConstraintArrayAttr:
    return readArrayOfAttrs<ConstraintArrayAttr>(reader);
  case Encoding::kFnMetadataAttr:
    return readFnMetadataAttr(reader);
  case Encoding::kUnknownAttr:
    return readUnknownAttr(reader);
  case Encoding::kUnboundAttr:
    return readUnboundAttr(reader);
  case Encoding::kParamDeclRefAttr:
    return readParamDeclRefAttr(reader);
  case Encoding::kParamIndexRefAttr:
    return readParamIndexRefAttr(reader);
  case Encoding::kConcreteTypeConstantAttr:
    return readConcreteTypeConstantAttr(reader);
  case Encoding::kParameterizedTypeConstantAttr:
    return readParameterizedTypeConstantAttr(reader);
  case Encoding::kDTypeConstantAttr:
    return readDTypeConstantAttr(reader);
  case Encoding::kSymbolConstantAttr:
    return readSymbolConstantAttr(reader);
  case Encoding::kTargetParamAttr:
    return readTargetParamAttr(reader);
  case Encoding::kBuildInfoParamAttr:
    return readBuildInfoParamAttr(reader);
  case Encoding::kParamOperatorAttr:
    return readParamOperatorAttr(reader);
  case Encoding::kVariadicAttr:
    return readVariadicAttr(reader);
  case Encoding::kMLIROpAttr:
    return readMLIROpAttr(reader);
  default:
    reader.emitError() << "unknown kgen attribute code: " << code;
    return Attribute();
  }
}

template <typename T>
T KGENBytecodeInterface::readArrayOfAttrs(BytecodeReader &reader) const {
  SmallVector<std::decay_t<decltype(std::declval<T>().getValue()[0])>> elements;
  if (failed(reader.readAttributes(elements)))
    return T();
  return T::get(getContext(), elements);
}

template <typename T>
T KGENBytecodeInterface::readArrayOfTypes(BytecodeReader &reader) const {
  SmallVector<std::decay_t<decltype(std::declval<T>().getValue()[0])>> elements;
  if (failed(reader.readTypes(elements)))
    return T();
  return T::get(getContext(), elements);
}

LogicalResult
KGENBytecodeInterface::writeAttribute(Attribute attr,
                                      BytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<BuildInfoParamAttr, ConcreteTypeConstantAttr, ConstraintAttr,
            DTypeConstantAttr, FnMetadataAttr, MLIROpAttr, ParamBindAttr,
            ParamDeclAttr, ParamDeclRefAttr, ParamIndexRefAttr,
            ParamOperatorAttr, ParameterizedTypeConstantAttr,
            SymbolConstantAttr, TargetParamAttr, UnboundAttr, UnknownAttr,
            VariadicAttr>([&](auto attr) {
        write(attr, writer);
        return success();
      })
      .Case([&](ConstraintArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kConstraintArrayAttr, writer);
      })
      .Case([&](ParamBindArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kParamBindArrayAttr, writer);
      })
      .Case([&](ParamDeclArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kParamDeclArrayAttr, writer);
      })
      .Case([&](ParameterExprArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kParameterExprArrayAttr,
                                 writer);
      })
      .Case([&](StringArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kStringArrayAttr, writer);
      })
      .Case([&](TypeArrayAttr attr) {
        return writeArrayOfTypes(attr, Encoding::kTypeArrayAttr, writer);
      })
      .Default([&](Attribute) { return failure(); });
}

template <typename T>
LogicalResult
KGENBytecodeInterface::writeArrayOfAttrs(T attr, uint64_t attrCode,
                                         BytecodeWriter &writer) const {
  writer.writeVarInt(attrCode);
  writer.writeAttributes(attr.getValue());
  return success();
}

template <typename T>
LogicalResult
KGENBytecodeInterface::writeArrayOfTypes(T attr, uint64_t attrCode,
                                         BytecodeWriter &writer) const {
  writer.writeVarInt(attrCode);
  writer.writeTypes(attr.getValue());
  return success();
}

//===----------------------------------------------------------------------===//
// BuildInfoParamAttr

BuildInfoParamAttr
KGENBytecodeInterface::readBuildInfoParamAttr(BytecodeReader &reader) const {
  BuildInfoAttr info;
  if (failed(reader.readAttribute(info)))
    return BuildInfoParamAttr();
  return BuildInfoParamAttr::get(info, BuildInfoType::get(getContext()));
}

void KGENBytecodeInterface::write(BuildInfoParamAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kBuildInfoParamAttr);
  writer.writeAttribute(attr.getBuildInfo());
}

//===----------------------------------------------------------------------===//
// ConcreteTypeConstantAttr

Attribute KGENBytecodeInterface::readConcreteTypeConstantAttr(
    BytecodeReader &reader) const {
  Type value;
  if (failed(reader.readType(value)))
    return Attribute();
  return ConcreteTypeConstantAttr::get(value);
}

void KGENBytecodeInterface::write(ConcreteTypeConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kConcreteTypeConstantAttr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// ConstraintAttr

ConstraintAttr
KGENBytecodeInterface::readConstraintAttr(BytecodeReader &reader) const {
  TypedAttr expr;
  StringAttr message;
  mlir::LocationAttr loc;
  if (failed(reader.readAttribute(expr)) ||
      failed(reader.readAttribute(message)) ||
      failed(reader.readAttribute(loc)))
    return ConstraintAttr();
  return ConstraintAttr::get(expr, message, loc);
}

void KGENBytecodeInterface::write(ConstraintAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kConstraintAttr);
  writer.writeAttribute(attr.getExpr());
  writer.writeAttribute(attr.getMessage());
  writer.writeAttribute(attr.getLoc());
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr

DTypeConstantAttr
KGENBytecodeInterface::readDTypeConstantAttr(BytecodeReader &reader) const {
  uint64_t dtype;
  if (failed(reader.readVarInt(dtype)))
    return DTypeConstantAttr();
  return DTypeConstantAttr::get(getContext(), KGENDType(dtype));
}

void KGENBytecodeInterface::write(DTypeConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kDTypeConstantAttr);
  writer.writeVarInt(attr.getDType().getValue());
}

//===----------------------------------------------------------------------===//
// FnMetadataAttr
//===----------------------------------------------------------------------===//

FnMetadataAttr
KGENBytecodeInterface::readFnMetadataAttr(BytecodeReader &reader) const {
  SmallVector<ValueInputConvention> inputConventions;
  auto parseConvention = [&](ValueInputConvention &convention) {
    uint64_t value;
    if (failed(reader.readVarInt(value)))
      return failure();
    convention = static_cast<ValueInputConvention>(value);
    return mlir::success();
  };
  if (failed(reader.readList(inputConventions, parseConvention)))
    return FnMetadataAttr();

  SmallVector<TypedAttr> defaultArguments;
  if (failed(reader.readAttributes(defaultArguments)))
    return FnMetadataAttr();

  uint64_t fnEffects;
  if (failed(reader.readVarInt(fnEffects)))
    return FnMetadataAttr();

  return FnMetadataAttr::get(getContext(), inputConventions, defaultArguments,
                             static_cast<FnEffects>(fnEffects));
}

void KGENBytecodeInterface::write(FnMetadataAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kFnMetadataAttr);
  writer.writeList(attr.getInputConventions(), [&](ValueInputConvention value) {
    writer.writeVarInt(static_cast<uint64_t>(value));
  });
  writer.writeAttributes(attr.getDefaultArguments());
  writer.writeVarInt(static_cast<uint64_t>(attr.getFnEffects()));
}

//===----------------------------------------------------------------------===//
// MLIROpAttr

Attribute KGENBytecodeInterface::readMLIROpAttr(BytecodeReader &reader) const {
  StringAttr name;
  DictionaryAttr attrs;
  SignatureType type;
  if (failed(reader.readAttribute(name)) ||
      failed(reader.readAttribute(attrs)) || failed(reader.readType(type)))
    return Attribute();
  return MLIROpAttr::get(name, attrs, type);
}

void KGENBytecodeInterface::write(MLIROpAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kMLIROpAttr);
  writer.writeAttribute(attr.getName());
  writer.writeAttribute(attr.getAttrs());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// ParamBindAttr

ParamBindAttr
KGENBytecodeInterface::readParamBindAttr(BytecodeReader &reader) const {
  StringAttr name;
  TypedAttr value;
  if (failed(reader.readAttribute(name)) || failed(reader.readAttribute(value)))
    return ParamBindAttr();
  return ParamBindAttr::get(name, value);
}

void KGENBytecodeInterface::write(ParamBindAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamBindAttr);
  writer.writeAttribute(attr.getName());
  writer.writeAttribute(attr.getValue());
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr

ParamDeclAttr
KGENBytecodeInterface::readParamDeclAttr(BytecodeReader &reader) const {
  StringAttr name;
  Type type;
  if (failed(reader.readAttribute(name)) || failed(reader.readType(type)))
    return ParamDeclAttr();
  return ParamDeclAttr::get(name, type);
}

void KGENBytecodeInterface::write(ParamDeclAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamDeclAttr);
  writer.writeAttribute(attr.getName());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr

ParamDeclRefAttr
KGENBytecodeInterface::readParamDeclRefAttr(BytecodeReader &reader) const {
  StringAttr name;
  Type type;
  if (failed(reader.readAttribute(name)) || failed(reader.readType(type)))
    return ParamDeclRefAttr();
  return ParamDeclRefAttr::get(name, type);
}

void KGENBytecodeInterface::write(ParamDeclRefAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamDeclRefAttr);
  writer.writeAttribute(attr.getName());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// ParamIndexRefAttr

ParamIndexRefAttr
KGENBytecodeInterface::readParamIndexRefAttr(BytecodeReader &reader) const {
  uint64_t depth, isResult, index;
  Type type;
  if (failed(reader.readVarInt(depth)) || failed(reader.readVarInt(isResult)) ||
      failed(reader.readVarInt(index)) || failed(reader.readType(type)))
    return ParamIndexRefAttr();
  return ParamIndexRefAttr::get(depth, static_cast<bool>(isResult), index,
                                type);
}

void KGENBytecodeInterface::write(ParamIndexRefAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamIndexRefAttr);
  writer.writeVarInt(attr.getDepth());
  writer.writeVarInt(attr.getIsResult());
  writer.writeVarInt(attr.getIndex());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr

Attribute
KGENBytecodeInterface::readParamOperatorAttr(BytecodeReader &reader) const {
  uint64_t opcode;
  SmallVector<TypedAttr> operands;
  Type type;
  if (failed(reader.readVarInt(opcode)) ||
      failed(reader.readAttributes(operands)) || failed(reader.readType(type)))
    return ParamOperatorAttr();
  return ParamOperatorAttr::get(getContext(), static_cast<POC>(opcode),
                                operands, type);
}

void KGENBytecodeInterface::write(ParamOperatorAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamOperatorAttr);
  writer.writeVarInt(static_cast<uint64_t>(attr.getOpcode()));
  writer.writeAttributes(attr.getOperands());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// ParameterizedTypeConstantAttr

Attribute KGENBytecodeInterface::readParameterizedTypeConstantAttr(
    BytecodeReader &reader) const {
  Type value;
  if (failed(reader.readType(value)))
    return Attribute();
  return ParameterizedTypeConstantAttr::get(value);
}

void KGENBytecodeInterface::write(ParameterizedTypeConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParameterizedTypeConstantAttr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr

SymbolConstantAttr
KGENBytecodeInterface::readSymbolConstantAttr(BytecodeReader &reader) const {
  SymbolRefAttr symbol;
  SmallVector<TypedAttr> paramValues;
  SignatureType type;
  if (failed(reader.readAttribute(symbol)) ||
      failed(reader.readAttributes(paramValues)) ||
      failed(reader.readType(type)))
    return SymbolConstantAttr();
  return SymbolConstantAttr::get(symbol, paramValues, type);
}

void KGENBytecodeInterface::write(SymbolConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kSymbolConstantAttr);
  writer.writeAttribute(attr.getSymbol());
  writer.writeAttributes(attr.getParamValues());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// TargetParamAttr

TargetParamAttr
KGENBytecodeInterface::readTargetParamAttr(BytecodeReader &reader) const {
  TargetInfoAttr target;
  if (failed(reader.readAttribute(target)))
    return TargetParamAttr();
  return TargetParamAttr::get(target, TargetType::get(getContext()));
}

void KGENBytecodeInterface::write(TargetParamAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kTargetParamAttr);
  writer.writeAttribute(attr.getTarget());
}

//===----------------------------------------------------------------------===//
// UnboundAttr

UnboundAttr
KGENBytecodeInterface::readUnboundAttr(BytecodeReader &reader) const {
  Type type;
  if (failed(reader.readType(type)))
    return UnboundAttr();
  return UnboundAttr::get(type);
}

void KGENBytecodeInterface::write(UnboundAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kUnboundAttr);
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// UnknownAttr

UnknownAttr
KGENBytecodeInterface::readUnknownAttr(BytecodeReader &reader) const {
  Type type;
  if (failed(reader.readType(type)))
    return UnknownAttr();
  return UnknownAttr::get(type);
}

void KGENBytecodeInterface::write(UnknownAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kUnknownAttr);
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// VariadicAttr

VariadicAttr
KGENBytecodeInterface::readVariadicAttr(BytecodeReader &reader) const {
  SmallVector<TypedAttr> operands;
  VariadicType type;
  if (failed(reader.readAttributes(operands)) || failed(reader.readType(type)))
    return VariadicAttr();
  return VariadicAttr::get(operands, type);
}

void KGENBytecodeInterface::write(VariadicAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kVariadicAttr);
  writer.writeAttributes(attr.getValues());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

Type KGENBytecodeInterface::readType(BytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();
  switch (code) {
  case Encoding::kParamRefType:
    return readParamRefType(reader);
  case Encoding::kMLIRTypeType:
    return MLIRTypeType::get(getContext());
  case Encoding::kDTypeType:
    return DTypeType::get(getContext());
  case Encoding::kStringType:
    return StringType::get(getContext());
  case Encoding::kSignatureType:
    return readSignatureType(reader);
  case Encoding::kDeclRefType:
    return readDeclRefType(reader);
  case Encoding::kTargetType:
    return TargetType::get(getContext());
  case Encoding::kBuildInfoType:
    return BuildInfoType::get(getContext());
  case Encoding::kVariadicType:
    return readVariadicType(reader);

  default:
    reader.emitError() << "unknown kgen type code: " << code;
    return Type();
  }
}

LogicalResult KGENBytecodeInterface::writeType(Type type,
                                               BytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<DeclRefType, ParamRefType, SignatureType, VariadicType>(
          [&](auto type) {
            write(type, writer);
            return success();
          })
      .Case([&](BuildInfoType) {
        writer.writeVarInt(Encoding::kBuildInfoType);
        return success();
      })
      .Case([&](DTypeType) {
        writer.writeVarInt(Encoding::kDTypeType);
        return success();
      })
      .Case([&](MLIRTypeType) {
        writer.writeVarInt(Encoding::kMLIRTypeType);
        return success();
      })
      .Case([&](StringType) {
        writer.writeVarInt(Encoding::kStringType);
        return success();
      })
      .Case([&](TargetType) {
        writer.writeVarInt(Encoding::kTargetType);
        return success();
      })
      .Default([&](Type) { return failure(); });
}

//===----------------------------------------------------------------------===//
// DeclRefType

Type KGENBytecodeInterface::readDeclRefType(BytecodeReader &reader) const {
  SymbolRefAttr symbol;
  ParamBindArrayAttr paramValues;
  if (failed(reader.readAttribute(symbol)) ||
      failed(reader.readAttribute(paramValues)))
    return Type();
  return DeclRefType::get(symbol, paramValues);
}

void KGENBytecodeInterface::write(DeclRefType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kDeclRefType);
  writer.writeAttribute(type.getSymbol());
  writer.writeAttribute(type.getParamValues());
}

//===----------------------------------------------------------------------===//
// ParamRefType

Type KGENBytecodeInterface::readParamRefType(BytecodeReader &reader) const {
  TypedAttr param;
  if (failed(reader.readAttribute(param)))
    return Type();
  return ParamRefType::get(param);
}

void KGENBytecodeInterface::write(ParamRefType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParamRefType);
  writer.writeAttribute(type.getParam());
}

//===----------------------------------------------------------------------===//
// SignatureType

Type KGENBytecodeInterface::readSignatureType(BytecodeReader &reader) const {
  TypeArrayAttr inputParamTypes, resultParamTypes;
  FunctionType values;
  FnMetadataAttr metadata;
  if (failed(reader.readAttribute(inputParamTypes)) ||
      failed(reader.readAttribute(resultParamTypes)) ||
      failed(reader.readType(values)) || failed(reader.readAttribute(metadata)))
    return Type();
  return SignatureType::get(getContext(), inputParamTypes, resultParamTypes,
                            values, metadata);
}

void KGENBytecodeInterface::write(SignatureType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kSignatureType);
  writer.writeAttribute(type.getInputParamTypes());
  writer.writeAttribute(type.getResultParamTypes());
  writer.writeType(type.getValues());
  writer.writeAttribute(type.getMetadata());
}

//===----------------------------------------------------------------------===//
// VariadicType

Type KGENBytecodeInterface::readVariadicType(BytecodeReader &reader) const {
  TypedAttr elementType;
  if (failed(reader.readAttribute(elementType)))
    return Type();
  return VariadicType::get(elementType);
}

void KGENBytecodeInterface::write(VariadicType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kVariadicType);
  writer.writeAttribute(type.getElementType());
}
