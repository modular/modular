//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_H
#define KGEN_KGENDIALECT_KGENTYPES_H

#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENEnums.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"

namespace M::KGEN {
class FnMetadataAttrInterface;
class FuncInterface;
class GeneratorMetadataAttrInterface;
class ParamDeclAttr;
class ParamDeclArrayAttr;
class SignatureGeneratorType;
class SignatureType;
class StructDefFieldAttr;
class VariadicType;
class VariadicAttr;

/// Create an uninitialized TypedAttr instance of the type.
TypedAttr createUninitializedValueOf(Type type);
} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.h.inc"

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// SignatureGeneratorType
//===----------------------------------------------------------------------===//
class SignatureGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  SignatureGeneratorType(GeneratorType sig);

  static SignatureGeneratorType
  get(ArrayRef<Type> inputParamTypes, FunctionType values,
      ArrayRef<ArgConvention> argConvs = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {});

  SignatureGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> inputParamValues,
                          function_ref<InFlightDiagnostic()> emitErrorFn = {});
  SignatureGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> inputParamValues,
                          Location location);

  /// Construct a signature from named parameter declarations, a function
  /// type, and metadata. This helper is used to convert between a named
  /// signature structure to a nameless `SignatureGeneratorType`
  /// representation.
  static SignatureGeneratorType remapToSignatureGenerator(
      ArrayRef<ParamDeclAttr> inputParams, FunctionType functionType,
      ArrayRef<ArgConvention> argConventions = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      function_ref<InFlightDiagnostic()> emitError = {});

  /// Temporary back-compat with existing signature type.
  static SignatureGeneratorType get(SignatureType sig);
  SignatureType asOldSignature();

  NewSignatureType getBody();
  NewSignatureType getInstantiatedBody();

  /// A SignatureGeneratorType is a GeneratorType containing a NewSignatureType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENTYPES_H
