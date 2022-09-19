//===- KGENDialect.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect Types
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ParamRefType

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
  return SignatureType::get(
      getContext(), replAttrs[0].cast<ParamDeclArrayAttr>(),
      replAttrs[1].cast<TypeArrayAttr>(), replTypes[0].cast<FunctionType>());
}

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/KGENDialect/KGENDialect.cpp.inc"

void KGENDialect::initialize() {
  registerAttributes();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/KGENDialect/KGENTypes.cpp.inc"
      >();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/KGENDialect/KGEN.cpp.inc"
      >();
}
