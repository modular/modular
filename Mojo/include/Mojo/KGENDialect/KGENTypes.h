//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file declares types for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_H
#define KGEN_KGENDIALECT_KGENTYPES_H

#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/SymbolExport.h"

namespace M::KGEN {
/// Return the size a struct field occupies before the next field. This is the
/// alloc size, except for an untagged union, which occupies only its store
/// size: the largest member's alloc size, not rounded up to the union's
/// alignment. The next field can then start in the union's tail padding, so
/// an enum's tag goes directly after its largest payload. This is safe because
/// every write of a union stops at its store size: initialization, a
/// whole-value copy, a member store and an interpreter write. A union's stride
/// (`size_of`, array elements) stays its alloc size.
std::optional<int64_t> getStructFieldSize(TargetInfoAttr target, Type type);

class ParameterEvaluationContext;
class FnMetadataAttrInterface;
class FuncInterface;
class ParamDeclAttr;
class ParamDeclArrayAttr;
class ParamsBindingArrayAttr;
struct PartiallySpecializedInputParams;
class FuncSymbolAttr;
class FuncOp;
class GeneratorAttr;
class FuncTypeGeneratorType;
class FuncLiteralTypeGeneratorType;
class ConstraintAttr;
class FnMetadataAttr;
class PogListAttr;
class StructDefFieldAttr;
class ParamListType;
class ParamListAttr;
} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#pragma GCC visibility push(default)
#include "Mojo/KGENDialect/KGENTypes.h.inc"
#pragma GCC visibility pop

namespace M::KGEN {
/// Return the layout offsets for a struct, cached on the interpreter's target.
std::optional<ArrayRef<int64_t>>
getInterpreterStructElementOffsets(StructType type, InterpreterState &state);

//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//
class MODULAR_CXX_EXPORT FuncTypeGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  FuncTypeGeneratorType(GeneratorType sig);

  static FuncTypeGeneratorType
  get(ArrayRef<Type> inputParamTypes, FunctionType values,
      ArrayRef<ArgConvention> argConvs = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      Attribute argListAttrs = {});

  static FuncTypeGeneratorType get(ArrayRef<Type> inputParamTypes, FuncType sig,
                                   Attribute genMetadata);

  /// Get this GeneratorType with some parameters bound.
  FuncTypeGeneratorType
  getSpecializedGenerator(ParamsBindingArrayAttr paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          function_ref<InFlightDiagnostic()> emitErrorFn = {});
  FuncTypeGeneratorType
  getSpecializedGenerator(ParamsBindingArrayAttr paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          Location location);

  /// Construct a signature from named parameter declarations, a function
  /// type, and metadata. This helper is used to convert between a named
  /// signature structure to a nameless `FuncTypeGeneratorType`
  /// representation.
  static FuncTypeGeneratorType remapToFuncTypeGenerator(
      ArrayRef<ParamDeclAttr> inputParams, FunctionType functionType,
      ArrayRef<ArgConvention> argConventions = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      function_ref<InFlightDiagnostic()> emitError = {},
      Attribute argListAttrs = {});

  FuncType getBody();
  FuncType getInstantiatedBody();

  /// A FuncTypeGeneratorType is a GeneratorType containing a FuncType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// FuncLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

class FuncLiteralTypeGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  FuncLiteralTypeGeneratorType(GeneratorType gen);

  static FuncLiteralTypeGeneratorType get(ArrayRef<Type> inputParamTypes,
                                          TypedAttr funcLiteral,
                                          Attribute genMetadata = {});

  /// Get this GeneratorType with some parameters bound.
  FuncLiteralTypeGeneratorType
  getSpecializedGenerator(ParamsBindingArrayAttr paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          function_ref<InFlightDiagnostic()> emitErrorFn = {});
  FuncLiteralTypeGeneratorType
  getSpecializedGenerator(ParamsBindingArrayAttr paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          Location location);

  // Unwrap the target literal from the literal type.
  TypedAttr getFuncSymbolAttr();
  /// Return the signature of the target literal, i.e. this generator with the
  /// FuncLiteralType body replaced by its FuncType.
  FuncTypeGeneratorType getNonLiteralGeneratorType();

  FuncLiteralType getBody();
  FuncLiteralType getInstantiatedBody();

  /// A FuncLiteralTypeGeneratorType is a GeneratorType containing a
  /// FuncLiteralType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENTYPES_H
