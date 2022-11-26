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

Type ParamRefType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return ParamRefType::get(replAttrs[0]);
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

ListType ListType::get(TypedAttr elementType, TypedAttr length) {
  return get(elementType.getContext(), elementType, length);
}

ListType ListType::get(Type elementType, int64_t length) {
  return get(TypeConstantAttr::get(elementType),
             Builder(elementType.getContext()).getIndexAttr(length));
}

LogicalResult ListType::verify(function_ref<InFlightDiagnostic()> emitError,
                               TypedAttr elementType, TypedAttr length) {
  if (!llvm::isa<MLIRTypeType>(elementType.getType()))
    return emitError()
           << "expected element type expression to be a '!kgen.mlirtype'";
  if (!llvm::isa<IndexType>(length.getType()))
    return emitError() << "expected length expression to be an 'index'";
  return success();
}

Optional<int64_t> ListType::getResolvedLength() const {
  if (auto length = llvm::dyn_cast<IntegerAttr>(getLength()))
    return length.getInt();
  return {};
}

Type ListType::getResolvedElementType() const {
  if (auto type = llvm::dyn_cast<ConcreteTypeConstantAttr>(getElementType()))
    return type.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

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
                  << getInputParams().size() << "; signature is " << *this;
    return SignatureType();
  }

  // If the signature isn't parameterized, then there are no substitutions to
  // perform.
  if (inputParamValues.empty())
    return *this;

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  unsigned paramNo = 0;
  for (auto [bind, decl] : llvm::zip(inputParamValues, getInputParams())) {
    if (bind.getName() != decl.getName()) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has name "
                    << bind.getName() << " but callee expected name "
                    << decl.getName();
      return SignatureType();
    }

    // Bound parameters are allowed to refine the type of subsequent parameters,
    // e.g. in `<ty: type, fn: () -> !kgen.paramref<ty>>`, the expected type of
    // the second parameter will be refined when the first parameter is bound.
    auto remappedDeclType = evaluator.getReboundType(decl.getType());
    if (bind.getType() != remappedDeclType) {
      emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                    << bind.getType() << " but callee expected name "
                    << decl.getType();
      return SignatureType();
    }

    evaluator.setParameterValue(bind.getDecl(), bind.getValue());
    ++paramNo;
  }

  // Remap the parameter decls and result types.
  SmallVector<Type> newParamResultTypes;
  llvm::append_range(newParamResultTypes,
                     llvm::map_range(getResultParamTypes(), remapType));

  // Remap the value types.
  SmallVector<Type> inputTypes, resultTypes;
  llvm::append_range(inputTypes,
                     llvm::map_range(getValues().getInputs(), remapType));
  llvm::append_range(resultTypes,
                     llvm::map_range(getValues().getResults(), remapType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), {}),
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
// DeclRefType
//===----------------------------------------------------------------------===//

DeclRefType DeclRefType::get(FlatSymbolRefAttr name,
                             ParamBindArrayAttr paramValues) {
  return get(name.getContext(), name, paramValues);
}

DeclRefType DeclRefType::get(FlatSymbolRefAttr name,
                             ArrayRef<ParamBindAttr> paramValues) {
  return get(name.getContext(), name,
             ParamBindArrayAttr::get(name.getContext(), paramValues));
}

DeclRefType DeclRefType::get(FlatSymbolRefAttr name) {
  return get(name, ArrayRef<ParamBindAttr>());
}

//===----------------------------------------------------------------------===//
// LITDeclRefType
//===----------------------------------------------------------------------===//

LITDeclRefType LITDeclRefType::get(SymbolRefAttr name,
                                   ParamBindArrayAttr paramValues) {
  return get(name.getContext(), name, paramValues);
}

LITDeclRefType LITDeclRefType::get(SymbolRefAttr name,
                                   ArrayRef<ParamBindAttr> paramValues) {
  return get(name.getContext(), name,
             ParamBindArrayAttr::get(name.getContext(), paramValues));
}

LITDeclRefType LITDeclRefType::get(SymbolRefAttr name) {
  return get(name, ArrayRef<ParamBindAttr>());
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
