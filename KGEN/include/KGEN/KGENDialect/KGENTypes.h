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
class ParameterEvaluationContext;
class FnMetadataAttrInterface;
class FuncInterface;
class GeneratorMetadataAttrInterface;
class ParamDeclAttr;
class ParamDeclArrayAttr;
class FuncTypeGeneratorType;
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
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//
class FuncTypeGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  FuncTypeGeneratorType(GeneratorType sig);

  static FuncTypeGeneratorType
  get(ArrayRef<Type> inputParamTypes, FunctionType values,
      ArrayRef<ArgConvention> argConvs = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {});

  /// Get this GeneratorType with some parameters bound.
  FuncTypeGeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings,
      function_ref<InFlightDiagnostic()> emitErrorFn = {},
      ParameterEvaluationContext *evaluationContext = nullptr);
  FuncTypeGeneratorType getSpecializedGenerator(
      ArrayRef<TypedAttr> paramBindings, Location location,
      ParameterEvaluationContext *evaluationContext = nullptr);

  /// Construct a signature from named parameter declarations, a function
  /// type, and metadata. This helper is used to convert between a named
  /// signature structure to a nameless `FuncTypeGeneratorType`
  /// representation.
  static FuncTypeGeneratorType remapToFuncTypeGenerator(
      ArrayRef<ParamDeclAttr> inputParams, FunctionType functionType,
      ArrayRef<ArgConvention> argConventions = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      function_ref<InFlightDiagnostic()> emitError = {});

  FuncType getBody();
  FuncType getInstantiatedBody();

  /// A FuncTypeGeneratorType is a GeneratorType containing a FuncType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENTYPES_H
