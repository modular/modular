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
  ///   ParamDeclAttr {
  ///     name: StringAttr
  ///     type: Type
  ///   }
  kParamDeclAttr = 1,
  ///
  ///   ParamDeclArrayAttr {
  ///     value: Attribute[]
  ///   }
  kParamDeclArrayAttr = 2,
  ///
  ///   ConstraintAttr {
  ///     expr: TypedAttr
  ///     message: StringAttr
  ///     loc: LocationAttr
  ///   }
  kConstraintAttr = 5,
  ///
  ///   ConstraintArrayAttr {
  ///     value: Attribute[]
  ///   }
  kConstraintArrayAttr = 6,
  ///
  ///   VariadicAttr {
  ///     values: TypedAttr[]
  ///     type: VariadicType
  ///   }
  kVariadicAttr = 7,
  ///
  ///   UnknownAttr {
  ///     type: Type
  ///   }
  kUnknownAttr = 8,
  ///
  ///   UnboundAttr {
  ///     type: Type
  ///   }
  kUnboundAttr = 9,
  ///
  ///   NoneAttr {
  ///   }
  kNoneAttr = 10,
  ///
  ///   ParamDeclRefAttr {
  ///     name: StringAttr
  ///     type: Type
  ///   }
  kParamDeclRefAttr = 11,
  ///
  ///   ParamIndexRefAttr {
  ///     depth: varint
  ///     isResult: varint
  ///     index: varint
  ///     type: Type
  ///   }
  kParamIndexRefAttr = 12,
  ///
  ///   ConcreteTypeConstantAttr {
  ///     value: Type
  ///   }
  kConcreteTypeConstantAttr = 13,
  ///
  ///   ParameterizedTypeConstantAttr {
  ///     value: Type
  ///   }
  kParameterizedTypeConstantAttr = 14,
  ///
  ///   DTypeConstantAttr {
  ///     dtype: varint
  ///   }
  kDTypeConstantAttr = 15,
  ///
  ///   IntLiteralAttr {
  ///     value: varint
  ///   }
  kIntLiteralAttr = 16,
  ///
  ///   SymbolConstantAttr {
  ///     symbol: SymbolRefAttr
  ///     paramValues: TypedAttr[]
  ///     type: SignatureType
  ///   }
  kSymbolConstantAttr = 17,
  ///
  ///   TargetParamAttr {
  ///     target: TargetInfoAttr
  ///   }
  kTargetParamAttr = 18,
  ///
  ///   BuildInfoParamAttr {
  ///     info: BuildInfoAttr
  ///   }
  kBuildInfoParamAttr = 19,
  ///
  ///  EnvAttr {
  ///    values: DictionaryAttr
  ///  }
  kEnvAttr = 20,
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
  ///
  ///  DecoratorsAttr {
  ///    value: TypedAttr[]
  ///  }
  kDecoratorsAttr = 23,
  ///
  ///  ExportKindAttr {
  ///    value: varint
  ///  }
  kExportKindAttr = 24,
  ///
  ///  PackageArchiveAttr {
  ///    target: TargetInfoAttr
  ///    elaboratedModule: DenseResourceElementsAttr
  ///    archive: DenseResourceElementsAttr
  ///  }
  kPackageArchiveAttr = 25,
  ///
  ///   PackageArchiveArrayAttr {
  ///     value: Attribute[]
  ///   }
  kPackageArchiveArrayAttr = 26,
  ///
  ///   StructAttr {
  ///     values: TypedAttr[]
  ///     type: Type
  ///   }
  kStructAttr = 27,
  ///
  ///   StructExtractAttr {
  ///     structValue: TypedAttr
  ///     fieldNo: varint
  ///     type: Type
  ///   }
  kStructExtractAttr = 28,
  ///
  ///   PackAttr {
  ///     values: TypedAttr[]
  ///     type: Type
  ///   }
  kPackAttr = 29,
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
  ///   IntLiteralType {
  ///   }
  kIntLiteralType = 4,
  ///
  ///   SignatureType {
  ///     inputParams: ParamDeclArrayAttr
  ///     resultParams: ParamDeclArrayAttr
  ///     values: FunctionType
  ///     metadata: Attribute
  ///   }
  kSignatureType = 5,
  ///
  ///   DeclRefType {
  ///     symbol: SymbolRefAttr
  ///     paramValues: TypedAttr[]
  ///   }
  kDeclRefType = 6,
  ///
  ///   TargetType {
  ///   }
  kTargetType = 7,
  ///
  ///   BuildInfoType {
  ///   }
  kBuildInfoType = 8,
  ///
  ///   VariadicType {
  ///     elementType: TypedAttr
  ///   }
  kVariadicType = 9,
  ///
  ///   NoneType {
  ///   }
  kNoneType = 10,
  ///
  ///   StructType {
  ///     values: TypedAttr[]
  ///   }
  kStructType = 11,
  ///
  ///   PackType {
  ///     variadic: TypedAttr
  ///   }
  kPackType = 12,
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

  BuildInfoParamAttr readBuildInfoParamAttr(BytecodeReader &reader) const;
  Attribute readConcreteTypeConstantAttr(BytecodeReader &reader) const;
  ConstraintAttr readConstraintAttr(BytecodeReader &reader) const;
  DTypeConstantAttr readDTypeConstantAttr(BytecodeReader &reader) const;
  EnvAttr readEnvAttr(BytecodeReader &reader) const;
  ExportKindAttr readExportKindAttr(BytecodeReader &reader) const;
  IntLiteralAttr readIntLiteralAttr(BytecodeReader &reader) const;
  Attribute readMLIROpAttr(BytecodeReader &reader) const;
  PackageArchiveAttr readPackageArchiveAttr(BytecodeReader &reader) const;
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
  StructAttr readStructAttr(BytecodeReader &reader) const;
  TypedAttr readStructExtractAttr(BytecodeReader &reader) const;
  PackAttr readPackAttr(BytecodeReader &reader) const;

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
  void write(EnvAttr attr, BytecodeWriter &writer) const;
  void write(ExportKindAttr attr, BytecodeWriter &writer) const;
  void write(IntLiteralAttr attr, BytecodeWriter &writer) const;
  void write(MLIROpAttr attr, BytecodeWriter &writer) const;
  void write(PackageArchiveAttr attr, BytecodeWriter &writer) const;
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
  void write(StructAttr attr, BytecodeWriter &writer) const;
  void write(StructExtractAttr attr, BytecodeWriter &writer) const;
  void write(PackAttr attr, BytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(BytecodeReader &reader) const override;
  Type readDeclRefType(BytecodeReader &reader) const;
  Type readParamRefType(BytecodeReader &reader) const;
  Type readSignatureType(BytecodeReader &reader) const;
  Type readVariadicType(BytecodeReader &reader) const;
  Type readStructType(BytecodeReader &reader) const;
  Type readPackType(BytecodeReader &reader) const;

  LogicalResult writeType(Type type, BytecodeWriter &writer) const override;
  void write(DeclRefType type, BytecodeWriter &writer) const;
  void write(ParamRefType type, BytecodeWriter &writer) const;
  void write(SignatureType type, BytecodeWriter &writer) const;
  void write(VariadicType type, BytecodeWriter &writer) const;
  void write(StructType type, BytecodeWriter &writer) const;
  void write(PackType type, BytecodeWriter &writer) const;
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
  case Encoding::kParamDeclAttr:
    return readParamDeclAttr(reader);
  case Encoding::kParamDeclArrayAttr:
    return readArrayOfAttrs<ParamDeclArrayAttr>(reader);
  case Encoding::kConstraintAttr:
    return readConstraintAttr(reader);
  case Encoding::kConstraintArrayAttr:
    return readArrayOfAttrs<ConstraintArrayAttr>(reader);
  case Encoding::kUnknownAttr:
    return readUnknownAttr(reader);
  case Encoding::kUnboundAttr:
    return readUnboundAttr(reader);
  case Encoding::kNoneAttr:
    return NoneAttr::get(reader.getContext());
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
  case Encoding::kIntLiteralAttr:
    return readIntLiteralAttr(reader);
  case Encoding::kSymbolConstantAttr:
    return readSymbolConstantAttr(reader);
  case Encoding::kTargetParamAttr:
    return readTargetParamAttr(reader);
  case Encoding::kBuildInfoParamAttr:
    return readBuildInfoParamAttr(reader);
  case Encoding::kEnvAttr:
    return readEnvAttr(reader);
  case Encoding::kParamOperatorAttr:
    return readParamOperatorAttr(reader);
  case Encoding::kVariadicAttr:
    return readVariadicAttr(reader);
  case Encoding::kMLIROpAttr:
    return readMLIROpAttr(reader);
  case Encoding::kDecoratorsAttr:
    return readArrayOfAttrs<DecoratorsAttr>(reader);
  case Encoding::kExportKindAttr:
    return readExportKindAttr(reader);
  case Encoding::kPackageArchiveAttr:
    return readPackageArchiveAttr(reader);
  case Encoding::kPackageArchiveArrayAttr:
    return readArrayOfAttrs<PackageArchiveArrayAttr>(reader);
  case Encoding::kStructAttr:
    return readStructAttr(reader);
  case Encoding::kStructExtractAttr:
    return readStructExtractAttr(reader);
  case Encoding::kPackAttr:
    return readPackAttr(reader);
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

LogicalResult
KGENBytecodeInterface::writeAttribute(Attribute attr,
                                      BytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<BuildInfoParamAttr, ConcreteTypeConstantAttr, ConstraintAttr,
            DTypeConstantAttr, EnvAttr, ExportKindAttr, IntLiteralAttr,
            MLIROpAttr, PackageArchiveAttr, ParamDeclAttr, ParamDeclRefAttr,
            ParamIndexRefAttr, ParamOperatorAttr, ParameterizedTypeConstantAttr,
            SymbolConstantAttr, TargetParamAttr, UnboundAttr, UnknownAttr,
            VariadicAttr, StructAttr, StructExtractAttr, PackAttr>(
          [&](auto attr) {
            write(attr, writer);
            return success();
          })
      .Case([&](ConstraintArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kConstraintArrayAttr, writer);
      })
      .Case([&](ParamDeclArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kParamDeclArrayAttr, writer);
      })
      .Case([&](ParameterExprArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kParameterExprArrayAttr,
                                 writer);
      })
      .Case([&](PackageArchiveArrayAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kPackageArchiveArrayAttr,
                                 writer);
      })
      .Case([&](DecoratorsAttr attr) {
        return writeArrayOfAttrs(attr, Encoding::kDecoratorsAttr, writer);
      })
      .Case([&](NoneAttr attr) {
        writer.writeVarInt(Encoding::kNoneAttr);
        return success();
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
  return BuildInfoParamAttr::get(info);
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
  MLIRTypeType type;
  if (failed(reader.readType(value)) || failed(reader.readType(type)))
    return Attribute();
  return ConcreteTypeConstantAttr::get(value, type);
}

void KGENBytecodeInterface::write(ConcreteTypeConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kConcreteTypeConstantAttr);
  writer.writeType(attr.getValue());
  writer.writeType(attr.getType());
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
// EnvAttr

EnvAttr KGENBytecodeInterface::readEnvAttr(BytecodeReader &reader) const {
  DictionaryAttr values;
  if (failed(reader.readAttribute(values)))
    return {};
  return EnvAttr::get(values);
}

void KGENBytecodeInterface::write(EnvAttr attr, BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kEnvAttr);
  writer.writeAttribute(attr.getValues());
}

//===----------------------------------------------------------------------===//
// ExportKindAttr

ExportKindAttr
KGENBytecodeInterface::readExportKindAttr(BytecodeReader &reader) const {
  uint64_t kind;
  if (failed(reader.readVarInt(kind)))
    return {};
  return ExportKindAttr::get(getContext(), static_cast<ExportKind>(kind));
}

void KGENBytecodeInterface::write(ExportKindAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kExportKindAttr);
  writer.writeVarInt(static_cast<uint64_t>(attr.getValue()));
}

//===----------------------------------------------------------------------===//
// IntLiteralAttr

IntLiteralAttr
KGENBytecodeInterface::readIntLiteralAttr(BytecodeReader &reader) const {
  uint64_t width;
  if (failed(reader.readVarInt(width)))
    return {};
  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(width);
  if (failed(value))
    return {};
  return IntLiteralAttr::get(getContext(), IPInt(std::move(*value)));
}

void KGENBytecodeInterface::write(IntLiteralAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kIntLiteralAttr);
  uint64_t width = attr.getValue().getAPInt().getSignificantBits();
  writer.writeVarInt(width);
  writer.writeAPIntWithKnownWidth(attr.getValue().getAPInt().trunc(width));
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
// PackageArchiveAttr

PackageArchiveAttr
KGENBytecodeInterface::readPackageArchiveAttr(BytecodeReader &reader) const {
  TargetInfoAttr target;
  DenseResourceElementsAttr elaboratedModule, archive;
  if (failed(reader.readAttribute(target)) ||
      failed(reader.readAttribute(elaboratedModule)) ||
      failed(reader.readAttribute(archive)))
    return {};
  return PackageArchiveAttr::get(target, elaboratedModule, archive);
}

void KGENBytecodeInterface::write(PackageArchiveAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kPackageArchiveAttr);
  writer.writeAttribute(attr.getTarget());
  writer.writeAttribute(attr.getElaboratedModule());
  writer.writeAttribute(attr.getArchive());
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
  MLIRTypeType type;
  if (failed(reader.readType(value)) || failed(reader.readType(type)))
    return Attribute();
  return ParameterizedTypeConstantAttr::get(value, type);
}

void KGENBytecodeInterface::write(ParameterizedTypeConstantAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kParameterizedTypeConstantAttr);
  writer.writeType(attr.getValue());
  writer.writeType(attr.getType());
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
  return TargetParamAttr::get(target);
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
// StructAttr

StructAttr KGENBytecodeInterface::readStructAttr(BytecodeReader &reader) const {
  SmallVector<TypedAttr> values;
  StructType type;
  if (failed(reader.readAttributes(values)) || failed(reader.readType(type)))
    return StructAttr();
  return StructAttr::get(getContext(), values, type);
}

void KGENBytecodeInterface::write(StructAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kStructAttr);
  writer.writeAttributes(attr.getValues());
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// StructExtractAttr

TypedAttr
KGENBytecodeInterface::readStructExtractAttr(BytecodeReader &reader) const {
  uint64_t fieldNo;
  TypedAttr structValue;
  Type type;
  if (failed(reader.readAttribute(structValue)) ||
      failed(reader.readVarInt(fieldNo)) || failed(reader.readType(type)))
    return StructExtractAttr();
  return StructExtractAttr::get(getContext(), structValue,
                                static_cast<unsigned>(fieldNo), type);
}

void KGENBytecodeInterface::write(StructExtractAttr attr,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kStructExtractAttr);
  writer.writeAttribute(attr.getStructValue());
  writer.writeVarInt(static_cast<uint64_t>(attr.getFieldNo()));
  writer.writeType(attr.getType());
}

//===----------------------------------------------------------------------===//
// PackAttr

PackAttr KGENBytecodeInterface::readPackAttr(BytecodeReader &reader) const {
  SmallVector<TypedAttr> values;
  PackType type;
  if (failed(reader.readAttributes(values)) || failed(reader.readType(type)))
    return PackAttr();
  return PackAttr::get(getContext(), values, type);
}

void KGENBytecodeInterface::write(PackAttr attr, BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kPackAttr);
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
  case Encoding::kIntLiteralType:
    return IntLiteralType::get(getContext());
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
  case Encoding::kNoneType:
    return KGEN::NoneType::get(reader.getContext());
  case Encoding::kStructType:
    return readStructType(reader);
  case Encoding::kPackType:
    return readPackType(reader);

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
      .Case([&](IntLiteralType) {
        writer.writeVarInt(Encoding::kIntLiteralType);
        return success();
      })
      .Case([&](TargetType) {
        writer.writeVarInt(Encoding::kTargetType);
        return success();
      })
      .Case([&](KGEN::NoneType) {
        writer.writeVarInt(Encoding::kNoneType);
        return success();
      })
      .Default([&](Type) { return failure(); });
}

//===----------------------------------------------------------------------===//
// DeclRefType

Type KGENBytecodeInterface::readDeclRefType(BytecodeReader &reader) const {
  SymbolRefAttr symbol;
  SmallVector<TypedAttr> paramValues;
  if (failed(reader.readAttribute(symbol)) ||
      failed(reader.readAttributes(paramValues)))
    return Type();
  return DeclRefType::get(symbol, paramValues);
}

void KGENBytecodeInterface::write(DeclRefType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kDeclRefType);
  writer.writeAttribute(type.getSymbol());
  writer.writeAttributes(type.getParamValues());
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
  FnMetadataAttrInterface metadata;
  if (failed(reader.readAttribute(inputParamTypes)) ||
      failed(reader.readAttribute(resultParamTypes)) ||
      failed(reader.readType(values)) ||
      failed(reader.readOptionalAttribute(metadata)))
    return Type();

  SmallVector<ValueInputConvention> inputConventions;
  auto parseConvention = [&](ValueInputConvention &convention) {
    uint64_t value;
    if (failed(reader.readVarInt(value)))
      return failure();
    convention = static_cast<ValueInputConvention>(value);
    return mlir::success();
  };
  if (failed(reader.readList(inputConventions, parseConvention)))
    return Type();
  FailureOr<APInt> fnEffectsValue =
      reader.readAPIntWithKnownWidth(/*bitWidth=*/16);
  if (failed(fnEffectsValue))
    return Type();
  auto fnEffects =
      static_cast<impl::FnEffects>(fnEffectsValue->getLimitedValue());

  return SignatureType::get(getContext(), inputParamTypes, resultParamTypes,
                            values, inputConventions, fnEffects, metadata);
}

void KGENBytecodeInterface::write(SignatureType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kSignatureType);
  writer.writeAttribute(type.getInputParamTypes());
  writer.writeAttribute(type.getResultParamTypes());
  writer.writeType(type.getValues());
  writer.writeOptionalAttribute(type.getMetadata());
  writer.writeList(type.getInputConventions(), [&](ValueInputConvention value) {
    writer.writeVarInt(static_cast<uint64_t>(value));
  });
  APInt fnEffectsValue(/*numBits=*/16,
                       static_cast<uint64_t>(type.getFnEffects().getImpl()));
  writer.writeAPIntWithKnownWidth(fnEffectsValue);
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

//===----------------------------------------------------------------------===//
// StructType

Type KGENBytecodeInterface::readStructType(BytecodeReader &reader) const {
  SmallVector<TypedAttr> elementTypes;
  if (failed(reader.readAttributes(elementTypes)))
    return Type();
  return StructType::get(getContext(), elementTypes);
}

void KGENBytecodeInterface::write(StructType type,
                                  BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kStructType);
  writer.writeAttributes(type.getElementTypes());
}

//===----------------------------------------------------------------------===//
// PackType

Type KGENBytecodeInterface::readPackType(BytecodeReader &reader) const {
  TypedAttr variadic;
  if (failed(reader.readAttribute(variadic)))
    return Type();
  return PackType::get(getContext(), variadic);
}

void KGENBytecodeInterface::write(PackType type, BytecodeWriter &writer) const {
  writer.writeVarInt(Encoding::kPackType);
  writer.writeAttribute(type.getVariadicAttr());
}
