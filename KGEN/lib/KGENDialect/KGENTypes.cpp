//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KGENDialect
//===----------------------------------------------------------------------===//

void KGENDialect::registerTypes() {
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ParamRefType
//===----------------------------------------------------------------------===//

Type ParamRefType::get(TypedAttr param) {
  // If the parameter is already resolved to a constant, fold this to the
  // indicated type.
  if (auto constant = param.dyn_cast<TypeConstantAttr>())
    return constant.getValue();

  // Otherwise, form the ParamRefType like normal.
  return Base::get(param.getContext(), param);
}

void ParamRefType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getParam());
}

Type ParamRefType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return ParamRefType::get(replAttrs[0]);
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

void SignatureType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getInputParams());
  walkAttrsFn(getResultParamTypes());
  walkTypesFn(getValues());
}

Type SignatureType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.size() == 1);
  return SignatureType::get(replAttrs[0].cast<ParamDeclArrayAttr>(),
                            replAttrs[1].cast<TypeArrayAttr>(),
                            replTypes[0].cast<FunctionType>());
}

SignatureType SignatureType::get(ParamBindArrayAttr inputParams,
                                 ParamDeclArrayAttr resultParams,
                                 FunctionType values) {
  SmallVector<ParamDeclAttr> inputParamDecls;
  inputParamDecls.reserve(inputParams.size());
  for (ParamBindAttr inputParam : inputParams)
    inputParamDecls.push_back(inputParam.getDecl());

  SmallVector<Type> resultParamTypes;
  resultParamTypes.reserve(resultParams.size());
  for (ParamDeclAttr resultParam : resultParams)
    resultParamTypes.push_back(resultParam.getType());

  return get(ParamDeclArrayAttr::get(values.getContext(), inputParamDecls),
             TypeArrayAttr::get(values.getContext(), resultParamTypes), values);
}

SignatureType SignatureType::get(FunctionType values) {
  return get(ParamDeclArrayAttr::get(values.getContext(), {}),
             TypeArrayAttr::get(values.getContext(), {}), values);
}

SignatureType SignatureType::get(MLIRContext *ctx, TypeRange inputs,
                                 TypeRange results) {
  return get(FunctionType::get(ctx, inputs, results));
}

/// Return a signature with the specified parameter bindings substituted
/// into it as happens in a call.  The types specified in the parameter
/// bindings affects the type signature of the value input and outputs, and
/// also can remap the signature in the parameter list itself.
///
/// If an error occurs making the substitution, report it with emitErrorFn
/// and return null.
SignatureType SignatureType::getSpecializedSignature(
    ArrayRef<ParamBindAttr> inputParamValues,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn) {
  // We need to substitute and simplify expressions that occur in the argument
  // list and parameter types, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !pop.scalar<type>)
  //     kgen.generator @callee2<size>(%x: !pop.simd<size, f32>)
  // ... call @callee1<type: dtype = f32>(%arg1) : (!pop.scalar<f32>) -> ()
  // ... call @callee2<size=4>(%arg2) : (!pop.simd<4, f32>) -> ()
  //
  // This can also occur in parameter types, e.g. for region types (dt vs f32):
  //     kgen.generator @g<dt: dtype, region: () -> !pop.scalar<dt>>(...
  //     call @g<dt: dtype = f32, region: () -> !pop.scalar<f32>(...
  if (inputParamValues.size() != getInputParams().size()) {
    emitErrorFn() << "caller has " << inputParamValues.size()
                  << " input parameters but callee expects "
                  << getInputParams().size();
    return SignatureType();
  }

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  if (inputParamValues.empty())
    return *this;

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  unsigned paramNo = 0;
  for (auto [bind, decl] : llvm::zip(inputParamValues, getInputParams())) {
    if (bind.getName() != decl.getName()) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has name "
                    << bind.getName() << " but callee expected name "
                    << decl.getName();
      return SignatureType();
    }
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());
    ++paramNo;
  }

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };
  auto remapParamDeclType = [&](ParamDeclAttr attr) -> ParamDeclAttr {
    auto newTy = remapType(attr.getType());
    return newTy == attr.getType() ? attr
                                   : ParamDeclAttr::get(attr.getName(), newTy);
  };

  // Remap the parameter decls and result types.
  SmallVector<ParamDeclAttr> newInputParams;
  SmallVector<Type> newParamResultTypes;
  llvm::append_range(newInputParams,
                     llvm::map_range(getInputParams(), remapParamDeclType));
  llvm::append_range(newParamResultTypes,
                     llvm::map_range(getResultParamTypes(), remapType));

  // Remap the value types.
  SmallVector<Type> inputTypes, resultTypes;
  llvm::append_range(inputTypes,
                     llvm::map_range(getValues().getInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(getValues().getResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), newInputParams),
      TypeArrayAttr::get(getContext(), newParamResultTypes),
      FunctionType::get(getContext(), inputTypes, resultTypes));
}

SignatureType SignatureType::getSpecializedSignature(
    ParamBindArrayAttr inputParams,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitErrorFn) {
  return getSpecializedSignature(inputParams.getValue(), emitErrorFn);
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

Optional<int64_t> DTypeType::getTypeSize(TargetInfoAttr target) const {
  return sizeof(uint8_t);
}

Optional<int64_t> DTypeType::getTypeAlign(TargetInfoAttr target) const {
  return 1;
}

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

void RefType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrs,
    function_ref<void(Type)> walkTypes) const {
  walkAttrs(getSymbol());
  walkAttrs(getParamValues());
}

Type RefType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                          ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return get(replAttrs[0].cast<FlatSymbolRefAttr>(),
             replAttrs[1].cast<ParamBindArrayAttr>());
}

RefType RefType::get(FlatSymbolRefAttr name, ParamBindArrayAttr paramValues) {
  return get(name.getContext(), name, paramValues);
}

RefType RefType::get(FlatSymbolRefAttr name) {
  return get(name, ParamBindArrayAttr::get(name.getContext(), {}));
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
