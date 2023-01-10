//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace mlir {
/// Parse a dtype.
template <>
struct FieldParser<KGENDType> {
  static FailureOr<KGENDType> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    return KGENDType::getFromString(value);
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// KGENDialect attribute support
//===----------------------------------------------------------------------===//

void KGENDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parameter Helper Functions
//===----------------------------------------------------------------------===//

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> KGEN::getParamDecls(Operation *op) {
  if (auto declItf = dyn_cast<DeclInterface>(op))
    return declItf.getInputParamDeclsAttr();
  if (auto paramDeclsArray =
          op->getAttrOfType<ParamDeclArrayAttr>("paramDecls"))
    return paramDeclsArray;
  return {};
}

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

LogicalResult
ParamBindAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      ParamDeclAttr decl, TypedAttr value) {
  if (decl.getType() == value.getType())
    return success();

  return emitError() << "value has incorrect type, expected " << decl.getType()
                     << " but got " << value.getType();
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

/// Parse an optional location or use the current location of the parser.
static ParseResult parseConstraintLoc(AsmParser &parser,
                                      FailureOr<Location> &loc) {
  if (succeeded(parser.parseOptionalComma())) {
    mlir::LocationAttr locAttr;
    if (parser.parseAttribute(locAttr))
      return failure();
    loc.emplace(locAttr);
  } else {
    loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  }
  return success();
}

/// Always print the location.
static void printConstraintLoc(AsmPrinter &printer, Location loc) {
  printer << ", ";
  printer.printAttribute(loc);
}

//===----------------------------------------------------------------------===//
// ConventionsAttr
//===----------------------------------------------------------------------===//

ConventionsAttr ConventionsAttr::get(MLIRContext *ctx, unsigned numInputs) {
  return get(ctx, SmallVector<ValueInputConvention>(numInputs),
             FnEffects::None);
}

bool ConventionsAttr::isDefault() {
  return getFnEffects() == FnEffects::None &&
         llvm::all_of(getInputConventions(),
                      [](ValueInputConvention inputConv) {
                        return inputConv == ValueInputConvention::ByVal;
                      });
}

//===----------------------------------------------------------------------===//
// ListAttr
//===----------------------------------------------------------------------===//

static ParseResult parseListValue(AsmParser &p,
                                  FailureOr<SmallVector<TypedAttr>> &values,
                                  ListType type) {
  auto elementType = ParamRefType::get(type.getElementType());
  values.emplace();
  return p.parseCommaSeparatedList(
      [&] { return parseParamValue(p, values->emplace_back(), elementType); });
}

static void printListValue(AsmPrinter &p, ArrayRef<TypedAttr> values,
                           ListType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

LogicalResult ListAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<TypedAttr> values, ListType type) {
  auto elementType = ParamRefType::get(type.getElementType());
  for (TypedAttr value : values) {
    if (value.getType() != elementType)
      return emitError() << "expected all list elements to have type "
                         << elementType;
  }
  return success();
}

/// A list constant is simple if its type and elements are fully-resolved.
bool ListAttr::isConstant() const {
  return !isParameterizedType(getType()) &&
         llvm::all_of(getValues(), [](TypedAttr attr) {
           return ParameterAttr::isSimpleConstant(attr);
         });
}

//===----------------------------------------------------------------------===//
// UnknownAttr
//===----------------------------------------------------------------------===//

bool UnknownAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// UnboundAttr
//===----------------------------------------------------------------------===//

bool UnboundAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

/// A parameter reference forms the basis of a non-constant parameter attribute.
bool ParamDeclRefAttr::isConstant() const { return false; }

/// Sort the parameter references by name.
bool ParamDeclRefAttr::isLessThan(Attribute rhs) const {
  if (auto ref = llvm::dyn_cast<ParamDeclRefAttr>(rhs))
    return getName().getValue() < ref.getName().getValue();
  // Otherwise, named parameters are always to the right.
  return false;
}

//===----------------------------------------------------------------------===//
// TypeConstantAttr
//===----------------------------------------------------------------------===//

TypedAttr TypeConstantAttr::get(Type value) {
  return ParameterizedTypeConstantAttr::get(value);
}

bool TypeConstantAttr::classof(Attribute attr) {
  return attr.isa<ConcreteTypeConstantAttr, ParameterizedTypeConstantAttr>();
}

//===----------------------------------------------------------------------===//
// ConcreteTypeConstantAttr
//===----------------------------------------------------------------------===//

TypedAttr ConcreteTypeConstantAttr::get(Type type) {
  auto *ctx = type.getContext();
  assert(!isParameterizedType(type) &&
         "Cannot create a ConcreteTypeConstantAttr with parameterized type");

  // If this is a ParamRefType, then we're unwrapping a wrapper.  Remove this to
  // keep the types canonical.
  if (auto refType = ::dyn_cast<ParamRefType>(type))
    return refType.getParam();

  return Base::get(ctx, type, MLIRTypeType::get(ctx));
}

LogicalResult
ConcreteTypeConstantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Type type, Type attrType) {
  if (!attrType.isa<MLIRTypeType>())
    return emitError() << "expected type to be !kgen.mlirtype";
  return success();
}

/// Always a constant by definition.
bool ConcreteTypeConstantAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ParameterizedTypeConstantAttr
//===----------------------------------------------------------------------===//

Attribute ParameterizedTypeConstantAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  // NOTE: This will automatically convert to ConcreteTypeConstantAttr if the
  // subtype is non-parametric.
  return get(replTypes[0]);
}

TypedAttr ParameterizedTypeConstantAttr::getChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *ctx, mlir::Type argType, mlir::Type resultType) {
  auto result = get(argType);
  if (result.getType() == resultType)
    return result;
  emitError() << "expected type to be !kgen.mlirtype";
  return {};
}

TypedAttr ParameterizedTypeConstantAttr::get(MLIRContext *ctx, Type type) {
  return get(type);
}

TypedAttr ParameterizedTypeConstantAttr::get(MLIRContext *ctx, Type type,
                                             Type resultType) {
  return get(type);
}

TypedAttr ParameterizedTypeConstantAttr::get(Type type) {
  // If this is a ParamRefType, then we're unwrapping a wrapper.  Remove this to
  // keep the types canonical.
  if (auto refType = ::dyn_cast<ParamRefType>(type))
    return refType.getParam();

  auto *ctx = type.getContext();
  auto typeType = MLIRTypeType::get(ctx);
  if (isParameterizedType(type))
    return Base::get(ctx, type, typeType);
  return ConcreteTypeConstantAttr::Base::get(ctx, type, typeType);
}

LogicalResult ParameterizedTypeConstantAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Type type, Type attrType) {
  if (!attrType.isa<MLIRTypeType>())
    return emitError() << "expected type to be !kgen.mlirtype";
  return success();
}

/// Always not a constant by definition.
bool ParameterizedTypeConstantAttr::isConstant() const { return false; }

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

DTypeConstantAttr DTypeConstantAttr::get(MLIRContext *ctx, KGENDType dtype) {
  return get(ctx, dtype, DTypeType::get(ctx));
}

LogicalResult
DTypeConstantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          KGENDType dtype, Type type) {
  if (!type || !type.isa<DTypeType>())
    return emitError() << "kgen.dtype.constant requires !kgen.dtype type";
  return success();
}

bool DTypeConstantAttr::isConvertibleTo(Type type) {
  KGENDType dtype = getDType();

  // Bool can only be `i1`.
  if (dtype.isBool())
    return type.isSignlessInteger(1);

  // Index DType can only be the mlir `index` type.
  if (dtype.isIndex())
    return type.isIndex();

  // Integer dtypes can be converted to MLIR integers of the same width and
  // un-opposing signedness; signed integer dtypes can be converted to signless
  // and signed MLIR integer types but not unsigned.
  if (dtype.isInt()) {
    auto intType = llvm::dyn_cast<IntegerType>(type);
    if (!intType || intType.getWidth() != dtype.getWidthInBits())
      return false;
    return intType.isSignless() || intType.isSigned() == dtype.isSInt();
  }

  // Floating point dtypes can be converted to equivalent MLIR float types.
  if (dtype.isFloat()) {
    if (auto fpType = llvm::dyn_cast<FloatType>(type))
      return areEquivalentFloatTypes(dtype, fpType);
    return false;
  }

  return false;
}

bool DTypeConstantAttr::isConvertibleFrom(Type type) {
  KGENDType dtype = getDType();

  if (dtype.isBool())
    return llvm::isa<IntegerType>(type);

  // Signless integers cannot be converted.
  if (type.isSignlessInteger() && !dtype.isIndex())
    return false;

  // Index dtypes can be converted if the type is an IndexType.
  if (dtype.isIndex() && type.isa<IndexType>())
    return true;

  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    if (dtype.isIndex())
      return true;
    // Integers can be converted to dtypes of the same width and signedness.
    if (dtype.isInt() && dtype.getWidthInBits() == intType.getWidth() &&
        dtype.isSInt() == intType.isSigned())
      return true;
    // Otherwise, we risk loosing bits, so we conservatively disallow.
    return false;
  }

  // Floating point types can be converted to equivalent dtypes.
  if (auto fpType = llvm::dyn_cast<FloatType>(type))
    return dtype.isFloat() && areEquivalentFloatTypes(dtype, fpType);

  return false;
}

/// Always a constant by definition.
bool DTypeConstantAttr::isConstant() const { return true; }

/// Sort by dtype value.
bool DTypeConstantAttr::isLessThan(Attribute rhs) const {
  if (auto dtype = llvm::dyn_cast<DTypeConstantAttr>(rhs))
    return getDType().getValue() < dtype.getDType().getValue();
  return true;
}

//===----------------------------------------------------------------------===//
// ExprFuncAttr
//===----------------------------------------------------------------------===//

static ParseResult parseExprFunc(AsmParser &p,
                                 FailureOr<ParamDeclArrayAttr> &paramDecls,
                                 FailureOr<ParamDeclArrayAttr> &inputs,
                                 FailureOr<ParameterExprArrayAttr> &exprs,
                                 SignatureType type) {
  if (!type.getResultParamTypes().empty())
    return p.emitError(p.getCurrentLocation())
           << "cannot have result parameters";

  // We can infer the input parameters from the signature.
  paramDecls = type.getInputParams();

  // Parse the inputs and expression using the types from the signature type.
  SmallVector<ParamDeclAttr> inputDecls;
  ArrayRef<Type> inputTypes = type.getValueInputs();
  auto typeIt = inputTypes.begin(), typeE = inputTypes.end();
  auto parseInput = [&]() -> ParseResult {
    if (typeIt == typeE)
      return p.emitError(p.getCurrentLocation(), "too many input declarations");
    StringAttr name;
    if (parseParamName(p, name))
      return failure();
    inputDecls.push_back(ParamDeclAttr::get(name, *typeIt++));
    return success();
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseInput))
    return failure();
  if (typeIt != typeE)
    return p.emitError(p.getCurrentLocation(), "not enough input declarations");
  if (p.parseArrow())
    return failure();

  SmallVector<TypedAttr> exprVals;
  ArrayRef<Type> resultTypes = type.getValueResults();
  auto resultIt = resultTypes.begin(), resultE = resultTypes.end();
  auto parseExpr = [&]() -> ParseResult {
    if (resultIt == resultE)
      return p.emitError(p.getCurrentLocation(), "too many result expressions");
    return parseParamValue(p, exprVals.emplace_back(), *resultIt++);
  };
  if (p.parseOptionalLParen()) {
    if (failed(parseExpr()))
      return failure();
  } else {
    if (p.parseOptionalRParen())
      if (p.parseCommaSeparatedList(parseExpr) || p.parseRParen())
        return failure();
  }
  if (resultIt != resultE)
    return p.emitError(p.getCurrentLocation(), "not enough result expressions");

  inputs = ParamDeclArrayAttr::get(p.getContext(), inputDecls);
  exprs = ParameterExprArrayAttr::get(p.getContext(), exprVals);
  return success();
}

static void printExprFunc(AsmPrinter &p, ParamDeclArrayAttr paramDecls,
                          ParamDeclArrayAttr inputs,
                          ParameterExprArrayAttr exprs, SignatureType type) {
  p << '(';
  llvm::interleaveComma(
      inputs, p, [&](ParamDeclAttr input) { p << input.getName().getValue(); });
  p << ") -> ";
  if (exprs.size() != 1)
    p << '(';
  llvm::interleaveComma(
      exprs, p, [&](TypedAttr expr) { printParamValue(expr, p.getStream()); });
  if (exprs.size() != 1)
    p << ')';
}

LogicalResult ExprFuncAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ParamDeclArrayAttr paramDecls,
                                   ParamDeclArrayAttr inputs,
                                   ParameterExprArrayAttr exprs,
                                   SignatureType type) {
  if (!type.getResultParamTypes().empty())
    return emitError() << "cannot have result parameters";
  if (exprs.size() != type.getValues().getNumResults())
    return emitError() << "has " << exprs.size() << " expressions but expected "
                       << type.getValues().getNumResults();
  for (auto [idx, expr, type] : llvm::zip(llvm::seq<unsigned>(0, exprs.size()),
                                          exprs, type.getValueResults())) {
    if (expr.getType() != type) {
      return emitError() << "expected expression result #" << idx
                         << " type to be " << type << " but got "
                         << expr.getType();
    }
  }
  if (paramDecls != type.getInputParams())
    return emitError() << "input parameters do not match signature";
  if (inputs.size() != type.getValues().getNumInputs())
    return emitError() << "wrong number of inputs";
  for (auto [input, sigInputType] : llvm::zip(inputs, type.getValueInputs()))
    if (input.getType() != sigInputType)
      return emitError() << "input types do not match";

  return checkSelfContained(emitError, paramDecls, inputs, exprs);
}

/// The expression function is a simple constant if it has no parameters.
bool ExprFuncAttr::isConstant() const {
  return !isParameterizedType(getType()) && getParamDecls().empty();
}

//===----------------------------------------------------------------------===//
// SymbolConstantAttr
//===----------------------------------------------------------------------===//

/// Always a constant by definition.
bool SymbolConstantAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// TargetParamAttr
//===----------------------------------------------------------------------===//

Attribute TargetParamAttr::parse(AsmParser &p, Type type) {
  auto targetType = llvm::dyn_cast_or_null<TargetType>(type);
  if (!targetType) {
    p.emitError(p.getCurrentLocation(),
                "target parameter expected a target type");
    return {};
  }
  // Check for a special host attribute.
  if (succeeded(p.parseOptionalKeyword("host")))
    return TargetParamAttr::get(TargetInfoAttr::getForHost(p.getContext()),
                                targetType);

  // Otherwise, parse the whole target info attribute.
  TargetInfoAttr target;
  if (p.parseCustomAttributeWithFallback(target))
    return {};
  return TargetParamAttr::get(target, targetType);
}

void TargetParamAttr::print(AsmPrinter &p) const { getTarget().print(p); }

/// Always a constant.
bool TargetParamAttr::isConstant() const { return true; }

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

/// Given a list of decls and values, produce an array of ParamBindAttrs.
static SmallVector<ParamBindAttr>
getBindAttrsForDeclsAndValues(ParamDeclArrayAttr decls,
                              ArrayRef<TypedAttr> values) {
  SmallVector<ParamBindAttr> binds;
  for (auto [decl, value] : llvm::zip(decls, values))
    binds.push_back(ParamBindAttr::get(decl.getName(), value));
  return binds;
}

static FailureOr<SignatureType>
verifyBindSignature(ArrayRef<TypedAttr> operands,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty())
    return emitError() << "'bind_signature' requires a function parameter";
  auto signature = dyn_cast<SignatureType>(operands[0].getType());
  if (!signature)
    return emitError()
           << "first operand of 'bind_signature' must have signature type";

  // Convert the input operands into a ParamBindAttr's for
  // getSpecializedSignature.
  SmallVector<ParamBindAttr> inputParams = getBindAttrsForDeclsAndValues(
      signature.getInputParams(), operands.drop_front());

  // Get the specialized version of the signature with all the known parameters
  // substituted in.
  auto result = signature.getSpecializedSignature(inputParams, emitError);
  if (!result)
    return failure();

  return result;
}

static LogicalResult verifyApply(ArrayRef<TypedAttr> operands, Type type,
                                 function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty())
    return emitError() << "'apply' expected a function parameter";

  auto signature = cast<SignatureType>(operands.front().getType());
  if (!signature.getResultParamTypes().empty() ||
      !signature.getInputParams().empty())
    return emitError() << "'apply' function cannot be parametric";

  FunctionType func = signature.getValues();
  if (func.getNumResults() != 1)
    return emitError() << "'apply' function must return one result";
  if (func.getResult(0) != type)
    return emitError() << "'apply' function result type must be " << type
                       << " but got " << func.getResult(0);

  return success();
}

LogicalResult ParamOperatorAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, POC opcode,
    ArrayRef<TypedAttr> operands, Type type) {
  // All the operand types must match except for 'bind_signature' and 'apply'.
  if (!llvm::is_contained({POC::GetListElement, POC::BindSignature, POC::Apply,
                           POC::TargetHasFeature, POC::TargetIsArch,
                           POC::TargetGetField, POC::GetSizeOf,
                           POC::GetAlignOf},
                          opcode) &&
      !llvm::all_of(operands, [&](auto operand) {
        return operand.getType() == operands.front().getType();
      }))
    return emitError() << "operand type mismatch";

  // Check invariants on the expression.
  switch (opcode) {
  case POC::Add:
  case POC::Mul:
  case POC::And:
  case POC::Or:
  case POC::Xor:
  case POC::Max:
  case POC::Min:
    if (operands.empty())
      return emitError() << stringifyEnum(opcode)
                         << " operator must have at least one operand";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    // Check the types that are supported.
    if (type.isIntOrIndex())
      break; // Index and fixed-width integer types supported for all of these.
    return emitError() << "operator requires an index or integer type";
    break;
  // Binary expressions.
  case POC::Shl:
  case POC::Shr:
  case POC::Div:
  case POC::Mod:
    if (operands.size() != 2)
      return emitError() << stringifyEnum(opcode) << " must have two operands";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "operator requires an index or integer type";
    break;
  case POC::EQ:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::LT:
  case POC::LE:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";

    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";

    // Relational operations only work on index types.
    if (!operands[0].getType().isIntOrIndex())
      return emitError() << "relational comparisons only allowed on index or "
                            "integer values";
    break;
  case POC::TargetEq:
    if (operands.size() != 2)
      return emitError() << "target_eq must have two operands";
    if (!type.isInteger(1))
      return emitError() << "target_eq returns i1";
    // TargetEq only work on target types.
    if (!operands[0].getType().isa<TargetType>() ||
        !operands[1].getType().isa<TargetType>())
      return emitError() << "target_eq only allowed on target types";
    break;
  case POC::TargetHasFeature:
  case POC::TargetIsArch:
    if (operands.size() != 2)
      return emitError() << "target comparisons must have two operands";
    if (!type.isInteger(1))
      return emitError() << "target comparison returns i1";
    if (!operands[0].getType().isa<TargetType>())
      return emitError() << "target comparison operand 0 must be a target type";
    if (!operands[1].getType().isa<StringType>())
      return emitError() << "target comparison operand 1 must be a string type";
    break;
  case POC::TargetGetField:
    if (operands.size() != 2)
      return emitError() << "target_get_field must have two operands";
    if (!operands[0].getType().isa<TargetType>())
      return emitError() << "target_get_field operand 0 must be a target type";
    if (!operands[1].getType().isa<StringType>())
      return emitError() << "target_get_field operand 1 must be a string type";
    break;
  case POC::In:
    if (operands.empty())
      return emitError() << "operator requires at least one operand";
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  case POC::GetSizeOf:
    if (operands.size() != 2)
      return emitError() << "'get_sizeof' operator requires two operands";
    if (!operands.front().getType().isa<MLIRTypeType>())
      return emitError() << "'get_sizeof' operand 0 should be a !kgen.mlirtype";
    if (!operands[1].getType().isa<TargetType>())
      return emitError() << "'get_sizeof' operand 1 should be a !kgen.target";
    if (!type.isa<IndexType>())
      return emitError() << "'get_sizeof' should return an index";
    break;
  case POC::GetAlignOf:
    if (operands.size() != 2)
      return emitError() << "'get_alignof' operator requires two operands";
    if (!llvm::isa<MLIRTypeType>(operands.front().getType()))
      return emitError()
             << "'get_alignof' operand 0 should be a !kgen.mlirtype";
    if (!operands[1].getType().isa<TargetType>())
      return emitError() << "'get_alignof' operand 1 should be a !kgen.target";
    if (!type.isa<IndexType>())
      return emitError() << "'get_alignof' should return an index";
    break;
  case POC::GetListElement: {
    if (operands.size() != 2)
      return emitError() << "'get_list_element' requires 2 operands";
    auto list = llvm::dyn_cast<ListType>(operands[0].getType());
    if (!list)
      return emitError() << "'get_list_element' first operand should be a list";
    if (!llvm::isa<IndexType>(operands[1].getType()))
      return emitError()
             << "'get_list_element' second operand should be an index";
    if (type != ParamRefType::get(list.getElementType()))
      return emitError()
             << "'get_list_element' result should match list element type";
    return success();
  }
  case POC::BindSignature: {
    FailureOr<SignatureType> actualType =
        verifyBindSignature(operands, emitError);
    if (failed(actualType))
      return failure();
    if (*actualType != type)
      return emitError() << "bind_signature expected to return " << type
                         << " but actually returns " << *actualType;
    break;
  }
  case POC::Apply:
    if (failed(verifyApply(operands, type, emitError)))
      return failure();
    break;
  }
  return success();
}

/// If the specified attribute is a ParamOperatorAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamOperatorAttr dyn_castPE(POC opcode, Attribute value) {
  if (auto expr = dyn_cast<ParamOperatorAttr>(value))
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
}

/// Treat `index` and signed integers as signed. Treat signless and unsigned
/// integers as unsigned.
static bool isSignedIntType(Type type) {
  return type.isIndex() || type.isSignedInteger();
}

/// Given a function_ref from `(APInt,APInt)->T` and two APInt's, compute the
/// result value T and return it.
///
/// Note that this function has special behavior when 'valueTy' (the MLIR type
/// of the two operand values) is 'index' type. In this case, it does extra work
/// to make sure that a 32-bit and 64-bit target will compute the same result
/// using the same approach as the index dialect.  If they differ, this refuses
/// to fold the operation, returning a null IntegerAttr.
template <typename ResultTy>
static IntegerAttr foldBinaryValues(
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &unsignedCalculateFn,
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &signedCalculateFn,
    const APInt &lhs, const APInt &rhs, Type valueTy, Type resultTy = {}) {
  const auto &calculateFn =
      isSignedIntType(valueTy) ? signedCalculateFn : unsignedCalculateFn;

  // Clients can specify resultTy if it differs from valueTy (e.g. for
  // compares), but not specifying it defaults to the result being the same type
  // as the operands.
  if (!resultTy)
    resultTy = valueTy;

  auto result1 = calculateFn(lhs, rhs);
  if (!valueTy.isa<IndexType>())
    return IntegerAttr::get(resultTy, result1);

  // If this is an index computation, then we just did the 64-bit computation,
  // see what would happen on a 32-bit host.
  assert(lhs.getBitWidth() == 64);

  // We require that the computation satisfy the invariant that:
  //   trunc(f(a, b)) = f(trunc(a), trunc(b))
  auto result2 = calculateFn(lhs.trunc(32), rhs.trunc(32));

  // If not bool result (e.g. a compare), truncate the LHS for our check.
  auto result1test = result1;
  if constexpr (!std::is_same_v<bool, ResultTy>) {
    result1test = result1.trunc(result2.getBitWidth());
  }

  // We can use the full 64-bit folded result if they match, otherwise leave
  // unfolded.
  return result1test == result2 ? IntegerAttr::get(resultTy, result1)
                                : IntegerAttr();
}

/// Given a fully associative variadic integer operation, constant fold any
/// constant operands and move them to the right.  If the whole expression is
/// constant, then return that, otherwise update the operands list.
static Attribute simplifyAssocOp(
    POC opcode, SmallVectorImpl<TypedAttr> &operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
    llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn = {},
    llvm::function_ref<bool(const APInt &)> identityConstantFn = {},
    llvm::function_ref<bool(const APInt &)> destructiveConstantFn = {}) {
  auto type = operands[0].getType();
  if (operands.size() == 1)
    return operands[0];

  // Flatten any of the same operation into the operand list:
  // `(add x, (add y, z))` => `(add x, y, z)`.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(opcode, operands[i])) {
      std::swap(operands[i], operands.back());
      operands.pop_back();
      --e;
      --i;
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
    }
  }

  // Impose an ordering on the operands, pushing subexpressions to the left and
  // constants to the right, with ParamRefs in the middle - but predictably
  // ordered w.r.t. each other.
  llvm::stable_sort(operands, ParameterAttr::compare);

  // Merge any constants, they will appear at the back of the operand list now.
  if (operands.back().isa<IntegerAttr>()) {
    while (operands.size() >= 2 &&
           operands[operands.size() - 2].isa<IntegerAttr>()) {
      APInt c1 = operands[operands.size() - 2].cast<IntegerAttr>().getValue();
      APInt c2 = operands.back().cast<IntegerAttr>().getValue();
      if (auto resultConstant = foldBinaryValues(
              unsignedFn, signedFn ? signedFn : unsignedFn, c1, c2, type)) {
        operands.pop_back();
        operands.pop_back();
        operands.push_back(resultConstant);
      } else {
        // If we couldn't fold the two values, bail.
        break;
      }
    }

    auto resultCst = operands.back().cast<IntegerAttr>();

    // If the resulting constant is the destructive constant (e.g. `x*0`), then
    // return it.
    if (destructiveConstantFn && destructiveConstantFn(resultCst.getValue()))
      return resultCst;

    // Remove the constant back to our operand list if it is the identity
    // constant for this operator (e.g. `x*1`) and there are other operands.
    if (identityConstantFn && identityConstantFn(resultCst.getValue()) &&
        operands.size() != 1)
      operands.pop_back();
  }

  return operands.size() == 1 ? operands[0] : Attribute();
}

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (standin for "multiplication by 1").
static std::pair<TypedAttr, TypedAttr> decomposeAddend(TypedAttr operand) {
  if (auto mul = dyn_castPE(POC::Mul, operand))
    if (auto cst = dyn_cast<IntegerAttr>(mul.getOperands().back())) {
      auto nonCst =
          ParamOperatorAttr::get(POC::Mul, mul.getOperands().drop_back());
      return {nonCst, cst};
    }
  return {operand, TypedAttr()};
}

static Attribute getOneOfType(Type type) {
  size_t width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return IntegerAttr::get(type, APInt(width, 1));
}

static Attribute simplifyAdd(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Add, operands, [](auto a, auto b) { return a + b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<std::pair<TypedAttr, TypedAttr>> decomposedOperands;
  llvm::SmallDenseSet<TypedAttr> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().first).second) {
      // The thing we multiply will be the common expression.
      TypedAttr mulOperand = decomposedOperands.back().first;

      // Find the index of the first occurrence.
      size_t i = 0;
      while (decomposedOperands[i].first != mulOperand)
        ++i;
      // Remove both occurrences from the operand list.
      operands.erase(operands.begin() + (&op - &operands[0]));
      operands.erase(operands.begin() + i);

      auto type = mulOperand.getType();
      auto c1 = decomposedOperands[i].second,
           c2 = decomposedOperands.back().second;
      // Fill in missing constant multiplicands with 1.
      if (!c1)
        c1 = getOneOfType(type);
      if (!c2)
        c2 = getOneOfType(type);
      // Re-add the "a"*(c1+c2) expression to the operand list and
      // re-canonicalize.
      auto constant = ParamOperatorAttr::get(POC::Add, c1, c2);
      auto mulCst = ParamOperatorAttr::get(POC::Mul, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamOperatorAttr::get(POC::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyMul(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Mul, operands, [](auto a, auto b) { return a * b; }, {},
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto addSubExpr = dyn_castPE(POC::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      operands.erase(operands.begin() + i);

      // Build each add operand.
      SmallVector<TypedAttr> addOperands;
      for (auto addOperand : addSubExpr.getOperands()) {
        operands.push_back(addOperand);
        addOperands.push_back(ParamOperatorAttr::get(POC::Mul, operands));
        operands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamOperatorAttr::get(POC::Add, addOperands);
    }
  }

  return {};
}

static Attribute simplifyAnd(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::And, operands, [](auto a, auto b) { return a & b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyOr(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Or, operands, [](auto a, auto b) { return a | b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); });
}

static Attribute simplifyXor(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Xor, operands, [](auto a, auto b) { return a ^ b; }, {},
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

/// Duplicate the operands in-place for ops like `min` and `max`.
static void deduplicateOperands(SmallVectorImpl<TypedAttr> &operands) {
  llvm::SetVector<TypedAttr, SmallVector<TypedAttr>, SmallPtrSet<Attribute, 4>>
      uniqueOperands(operands.begin(), operands.end());
  operands = uniqueOperands.takeVector();
}

/// Returns true if the integer is at its max value.
static bool intIsMaxValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMaxSignedValue() : value.isMaxValue();
}

/// Returns true if the integer is at its min value.
static bool intIsMinValue(Type type, const APInt &value) {
  return isSignedIntType(type) ? value.isMinSignedValue() : value.isMinValue();
}

static Attribute simplifyMax(SmallVectorImpl<TypedAttr> &operands) {
  deduplicateOperands(operands);
  Type type = operands.front().getType();
  return simplifyAssocOp(
      POC::Max, operands, llvm::APIntOps::umax, llvm::APIntOps::smax,
      [&](auto cst) { return intIsMinValue(type, cst); },
      [&](auto cst) { return intIsMaxValue(type, cst); });
}

static Attribute simplifyMin(SmallVectorImpl<TypedAttr> &operands) {
  deduplicateOperands(operands);
  Type type = operands.front().getType();
  return simplifyAssocOp(
      POC::Min, operands, llvm::APIntOps::umin, llvm::APIntOps::smin,
      [&](auto cst) { return intIsMaxValue(type, cst); },
      [&](auto cst) { return intIsMinValue(type, cst); });
}

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold functions to compute the result.
static Attribute
foldBinaryOp(ArrayRef<TypedAttr> operands,
             llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedFn,
             llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = dyn_cast<IntegerAttr>(operands[0]))
    if (auto rhs = dyn_cast<IntegerAttr>(operands[1])) {
      if (auto resultConstant =
              foldBinaryValues(unsignedFn, signedFn, lhs.getValue(),
                               rhs.getValue(), lhs.getType()))
        return resultConstant;
    }
  return {};
}

/// Folds constants given a comparison function that returns bool.  The client
/// must handle signedness etc.
static IntegerAttr foldCompareOp(
    TypedAttr lhs, TypedAttr rhs,
    llvm::function_ref<bool(const APInt &, const APInt &)> unsignedCompareFn,
    llvm::function_ref<bool(const APInt &, const APInt &)> signedCompareFn =
        {}) {
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs))
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      if (auto resultConstant = foldBinaryValues(
              unsignedCompareFn,
              signedCompareFn ? signedCompareFn : unsignedCompareFn,
              lhsInt.getValue(), rhsInt.getValue(), lhsInt.getType(),
              IntegerType::get(rhs.getContext(), 1)))
        return resultConstant;
    }
  return {};
}

/// Compute the result of == for the two specified attributes, handling the
/// index truncation issue but otherwise relying on MLIR's canonicalization of
/// attributes to do the job for us.  Both operands may be null, and this
/// returns null if no folding is possible.
static IntegerAttr foldEquality(TypedAttr lhs, TypedAttr rhs) {
  // foldCompareOp handles 32-bit truncation of input values correctly.
  if (lhs.getType().isIndex() && isa<IntegerAttr>(lhs) && isa<IntegerAttr>(rhs))
    return foldCompareOp(lhs, rhs, [](auto a, auto b) { return a == b; });

  // If the values have pointer equality, we know they are equal.
  if (lhs == rhs)
    return BoolAttr::get(rhs.getContext(), true);

  // Otherwise, we can use pointer equality for the attributes we support that
  // are known to have agreeable widths.
  if (ParameterAttr::isSimpleConstant(lhs) &&
      ParameterAttr::isSimpleConstant(rhs))
    return BoolAttr::get(rhs.getContext(), lhs == rhs);

  // Otherwise can't fold something like "x == y".
  return {};
}

static Attribute simplifyShl(SmallVectorImpl<TypedAttr> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1])) {
    // NOTE: This is correct even for index types because an overlong shift will
    // turn the result to zero.
    // FIXME: getOneBitSet asserts the shift amount should be in-range.  We need
    // to check this.
    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamOperatorAttr::get(POC::Mul, operands[0],
                                  IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShr(SmallVectorImpl<TypedAttr> &operands) {
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isZero())
      return operands[0]; // `x >> 0 = x`.
  // TODO: 0 >> x, -1 >>> x

  // FIXME: Must care about high bits.
  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.lshr(b); },
      [](auto a, auto b) { return a.ashr(b); });
}

static Attribute simplifyDiv(SmallVectorImpl<TypedAttr> &operands) {
  // Implement support for identities like `x/1 = x`.
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isOne())
      return operands[0];

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.udiv(b); },
      [](auto a, auto b) { return a.sdiv(b); });
}

static Attribute simplifyMod(SmallVectorImpl<TypedAttr> &operands) {
  // Implement support for identities like `x%1 = 0`.
  if (auto rhs = dyn_cast<IntegerAttr>(operands[1]))
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.urem(b); },
      [](auto a, auto b) { return a.srem(b); });
}

static Attribute simplifyEQ(SmallVectorImpl<TypedAttr> &operands) {
  // Make sure parameters are ordered correctly, which also matters if they
  // don't fold.
  llvm::stable_sort(operands, ParameterAttr::compare);

  return foldEquality(operands[0], operands[1]);
}

/// Simplify the < and <= operations.
static Attribute
simplifyRelationalCompare(POC opcode, SmallVectorImpl<TypedAttr> &operands) {
  auto rhs = dyn_cast<IntegerAttr>(operands[1]);
  auto lhs = dyn_cast<IntegerAttr>(operands[0]);

  if (rhs && !lhs) {
    // If this is a `(le x, RHS)` and RHS is a constant, canonicalize to `lt`.
    if (opcode == POC::LE) {
      if (intIsMaxValue(rhs.getType(), rhs.getValue())) // x <= 127 --> TRUE.
        return BoolAttr::get(rhs.getContext(), true);
      return ParamOperatorAttr::get(
          POC::LT, operands[0],
          IntegerAttr::get(rhs.getType(), rhs.getValue() + 1));
    }
    // If this is (x < MAXCST) canonicalize to (x != MAXCST).
    if (intIsMaxValue(rhs.getType(), rhs.getValue()))
      return ParamOperatorAttr::getNE(operands[0], rhs);
  }

  if (lhs && !rhs) {
    // (le cst, x) -> !(lt x, cst)
    if (opcode == POC::LE)
      return ParamOperatorAttr::getNot(
          ParamOperatorAttr::get(POC::LT, operands[1], operands[0]));
    // (lt cst, x) -> !(le x, cst)
    return ParamOperatorAttr::getNot(
        ParamOperatorAttr::get(POC::LE, operands[1], operands[0]));
  }

  if (opcode == POC::LT)
    return foldCompareOp(
        operands[0], operands[1], [](auto a, auto b) { return a.ult(b); },
        [](auto a, auto b) { return a.slt(b); });
  assert(opcode == POC::LE);
  return foldCompareOp(
      operands[0], operands[1], [](auto a, auto b) { return a.ule(b); },
      [](auto a, auto b) { return a.sle(b); });
}

static Attribute simplifyTargetEq(SmallVectorImpl<TypedAttr> &operands) {
  // TODO: Make simplifyTargetEq more granular than just checking that the
  // TargetAttrInfos are the same.
  if (operands[0].isa<TargetParamAttr>() &&
      operands[1].isa<TargetParamAttr>()) {
    bool targetsAreSame = (operands[0] == operands[1]);
    return IntegerAttr::get(IntegerType::get(operands[0].getContext(), 1),
                            targetsAreSame);
  }
  return {};
}

static Attribute simplifyHasFeature(SmallVectorImpl<TypedAttr> &operands) {
  auto target = dyn_cast<TargetParamAttr>(operands[0]);
  auto feature = dyn_cast<StringAttr>(operands[1]);
  if (!target || !feature)
    return {};
  return Builder(target.getContext())
      .getBoolAttr(target.getTarget().hasFeature(feature));
}

static Attribute simplifyIsArch(SmallVectorImpl<TypedAttr> &operands) {
  auto target = dyn_cast<TargetParamAttr>(operands[0]);
  auto arch = dyn_cast<StringAttr>(operands[1]);
  if (!target || !arch)
    return {};
  return Builder(target.getContext())
      .getBoolAttr(target.getTarget().isArch(arch));
}

static Attribute simplifyTargetGetField(SmallVectorImpl<TypedAttr> &operands) {
  auto target = dyn_cast<TargetParamAttr>(operands[0]);
  auto field = dyn_cast<StringAttr>(operands[1]);
  if (!target || !field)
    return {};
  std::optional<ssize_t> result =
      llvm::StringSwitch<std::optional<ssize_t>>(field.getValue())
          .Case("pointer_size", target.getTarget().getPointerSize())
          .Case("simd_bit_width", target.getTarget().getSimdBitWidth())
          .Default(std::nullopt);
  if (!result)
    return {};
  return Builder(target.getContext()).getIndexAttr(*result);
}

/// Simplifies an `in` (also `in(:dtype`) operator.  We know the all the
/// operands have the same type.
static Attribute simplifyIn(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  MutableArrayRef<TypedAttr> trailing =
      llvm::makeMutableArrayRef(operands).drop_front();

  Builder b(lhs.getContext());

  // If there are no trailing operands, fold to false.
  if (trailing.empty())
    return b.getBoolAttr(false);

  // If there is only one trailing operand, canonicalize to an `eq` operator.
  if (trailing.size() == 1)
    return ParamOperatorAttr::get(POC::EQ, operands);

  bool allKnownFalse = true;
  for (TypedAttr operand : trailing) {
    // Fold to true if a match was found by value.
    if (auto knownEq = foldEquality(lhs, operand)) {
      if (knownEq.getValue().isOne())
        return knownEq;
    } else if (lhs == operand) {
      // Fold to true if they match symbolically, like "x+1" and "x+1".
      return b.getBoolAttr(true);
    } else {
      // If this is a symbolic comparison like "x == 5", then we cannot fold the
      // non-containment case.
      allKnownFalse = false;
    }
  }

  // Ok, we know that LHS isn't known to equal any member of the set, but it or
  // they might be symbolic.  If we know for sure that LHS *isn't* equal to any
  // of the elements in the set then we can fold to false.
  if (allKnownFalse)
    return b.getBoolAttr(false);

  // Sort and unique the trailing operands.
  llvm::stable_sort(trailing, ParameterAttr::compare);
  SmallVector<TypedAttr> newOperands;
  newOperands.reserve(operands.size());
  newOperands.push_back(lhs);
  SmallPtrSet<Attribute, 4> seenTrailing;
  for (TypedAttr operand : trailing)
    if (seenTrailing.insert(operand).second)
      newOperands.push_back(operand);
  if (newOperands == operands)
    return {};
  return ParamOperatorAttr::get(POC::In, newOperands);
}

/// Simplifies a `get_sizeof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetSizeOf(SmallVectorImpl<TypedAttr> &operands) {
  auto typeCst = dyn_cast<ConcreteTypeConstantAttr>(operands[0]);
  auto target = dyn_cast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeSizeInBytes(
      target.getTarget(), typeCst.getValue());
  if (!size)
    return {};
  return Builder(typeCst.getContext()).getIndexAttr(*size);
}

/// Simplifies a `get_alignof` operator. Try to narrow the operand to a type
/// constant. If it does, query its data layout.
static Attribute simplifyGetAlignOf(SmallVectorImpl<TypedAttr> &operands) {
  auto typeCst = dyn_cast<ConcreteTypeConstantAttr>(operands[0]);
  auto target = dyn_cast<TargetParamAttr>(operands[1]);
  if (!typeCst || !target)
    return {};
  std::optional<int64_t> size = DataLayoutInterface::getTypeAlignInBytes(
      target.getTarget(), typeCst.getValue());
  if (!size)
    return {};
  return Builder(typeCst.getContext()).getIndexAttr(*size);
}

static Attribute simplifyBindSignature(ArrayRef<TypedAttr> operands,
                                       Type &resultType) {
  // If there is only a single operand, then nothing is bound.
  if (operands.size() == 1)
    return operands[0];

  // Otherwise, compute the result type. If an error is producted, just abort.
  SignatureType resultSig =
      *verifyBindSignature(operands, []() -> mlir::InFlightDiagnostic {
        llvm_unreachable("invalid bind_signature operator");
      });
  resultType = resultSig;

  // If the actual operand is a SymbolConstantAttr operand, then we can simplify
  // the bind_signature by folding the parameter values into it directly.
  if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(operands.front())) {
    bool hasUnboundParameters = symbolConstant.getParamValues().empty();
    hasUnboundParameters |=
        llvm::any_of(symbolConstant.getParamValues(), [](ParamBindAttr bind) {
          return bind.getValue().isa<UnboundAttr>();
        });
    assert(hasUnboundParameters &&
           "cannot have already bound all the input parameters, because we'd "
           "end up with a nongeneric signature that would fail verification");

    SignatureType signature = symbolConstant.getType();
    SmallVector<ParamBindAttr> paramBinds;
    if (symbolConstant.getParamValues().empty()) {
      paramBinds = getBindAttrsForDeclsAndValues(signature.getInputParams(),
                                                 operands.drop_front());
    } else {
      // We have to interleave the new values wherever there's an unbound thing
      // so we preserve the order. Drop the first operand because it's the
      // signature itself.
      auto operandIter = operands.begin() + 1;
      for (auto param : symbolConstant.getParamValues()) {
        // If we have this parameter already, we're good.
        if (!param.getValue().isa<UnboundAttr>()) {
          paramBinds.push_back(param);
          continue;
        }
        // Otherwise, bind it to the operand provided.
        paramBinds.push_back(
            ParamBindAttr::get(param.getName(), *operandIter++));
      }
      assert(operandIter == operands.end() && "Didn't use all the operands?");
    }

    return SymbolConstantAttr::get(
        symbolConstant.getSymbol(),
        ParamBindArrayAttr::get(resultType.getContext(), paramBinds),
        resultSig);
  }

  // If the operand is an expression function, substitute the parameter
  // declarations. If any parameter value is not a simple constant, we cannot
  // simplify the bound function.
  if (auto exprFunc = dyn_cast<ExprFuncAttr>(operands.front())) {
    ParameterEvaluator evaluator;
    for (auto [param, value] :
         llvm::zip(exprFunc.getParamDecls(), operands.drop_front())) {
      if (!ParameterAttr::isSimpleConstant(value))
        return {};
      evaluator.setParameterValue(param, value);
    }
    // Bind inputs to themselves.
    for (ParamDeclAttr input : exprFunc.getInputs())
      evaluator.setParameterValue(
          input, ParamDeclRefAttr::get(input.getName(), input.getType()));

    SmallVector<TypedAttr> exprs;
    exprs.reserve(exprFunc.getExprs().size());
    for (TypedAttr expr : exprFunc.getExprs())
      exprs.push_back(evaluator.getReboundAttribute(expr));
    return ExprFuncAttr::get(
        exprFunc.getInputs(),
        ParameterExprArrayAttr::get(exprFunc.getContext(), exprs), resultSig);
  }

  return {};
}

static Attribute simplifyGetListElement(ArrayRef<TypedAttr> operands,
                                        Type &resultType) {
  resultType =
      ParamRefType::get(cast<ListType>(operands[0].getType()).getElementType());
  auto list = dyn_cast<ListAttr>(operands[0]);
  auto index = dyn_cast<IntegerAttr>(operands[1]);
  if (!list || !index)
    return {};
  if (index.getInt() < 0 ||
      index.getInt() >= static_cast<int64_t>(list.getValues().size()))
    return {};
  return list.getValues()[index.getInt()];
}

static Attribute simplifyApply(ArrayRef<TypedAttr> operands, Type &resultType) {
  TypedAttr func = operands.front();
  operands = operands.drop_front();
  resultType = cast<SignatureType>(func.getType()).getValues().getResult(0);

  if (auto exprFunc = dyn_cast<ExprFuncAttr>(func)) {
    // Evalute the function expression. Map the operands to the input
    // parameters.
    ParameterEvaluator evaluator;
    for (auto [param, value] : llvm::zip(exprFunc.getInputs(), operands))
      evaluator.setParameterValue(param, value);

    return evaluator.getReboundAttribute(exprFunc.getExprs().front());
  }

  return {};
}

TypedAttr ParamOperatorAttr::get(MLIRContext *context, POC opcode,
                                 ArrayRef<TypedAttr> operandsIn, Type type) {
  auto result = get(opcode, operandsIn);
  assert((!type || type == result.getType()) && "unexpected type");
  return result;
}

TypedAttr
ParamOperatorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, POC opcode,
                              ArrayRef<TypedAttr> operandsIn, Type type) {
  if (failed(verify(emitError, opcode, operandsIn, type)))
    return {};
  return get(context, opcode, operandsIn, type);
}

/// Return (not x) which is the same as (xor x, true).  The `operand` value
/// must have type `i1`.
TypedAttr ParamOperatorAttr::getNot(TypedAttr operand) {
  TypedAttr one = BoolAttr::get(operand.getContext(), true);
  return ParamOperatorAttr::get(POC::Xor, {operand, one});
}

TypedAttr ParamOperatorAttr::get(POC opcode, ArrayRef<TypedAttr> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  // All operands must have the same type.  The result type is usually the
  // same as the operands, but is i1 for comparisons (overridden below).
  auto resultType = operandsIn.front().getType();
  assert(
      llvm::is_contained({POC::GetListElement, POC::BindSignature, POC::Apply,
                          POC::TargetHasFeature, POC::TargetIsArch,
                          POC::TargetGetField, POC::GetSizeOf, POC::GetAlignOf},
                         opcode) ||
      llvm::all_of(operandsIn.drop_front(),
                   [&](auto op) { return op.getType() == resultType; }));

  SmallVector<TypedAttr, 4> operands(operandsIn.begin(), operandsIn.end());

  auto *context = operandsIn[0].getContext();

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case POC::Add:
    result = simplifyAdd(operands);
    break;
  case POC::Mul:
    result = simplifyMul(operands);
    break;
  case POC::And:
    result = simplifyAnd(operands);
    break;
  case POC::Or:
    result = simplifyOr(operands);
    break;
  case POC::Xor:
    result = simplifyXor(operands);
    break;
  case POC::Max:
    result = simplifyMax(operands);
    break;
  case POC::Min:
    result = simplifyMin(operands);
    break;
  case POC::Shl:
    result = simplifyShl(operands);
    break;
  case POC::Shr:
    result = simplifyShr(operands);
    break;
  case POC::Div:
    result = simplifyDiv(operands);
    break;
  case POC::Mod:
    result = simplifyMod(operands);
    break;
  case POC::EQ:
    result = simplifyEQ(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::LT:
  case POC::LE:
    result = simplifyRelationalCompare(opcode, operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::TargetEq:
    result = simplifyTargetEq(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::TargetHasFeature:
    result = simplifyHasFeature(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::TargetIsArch:
    result = simplifyIsArch(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::TargetGetField:
    result = simplifyTargetGetField(operands);
    resultType = IndexType::get(context);
    break;
  case POC::In:
    result = simplifyIn(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::GetSizeOf:
    result = simplifyGetSizeOf(operands);
    resultType = IndexType::get(context);
    break;
  case POC::GetAlignOf:
    result = simplifyGetAlignOf(operands);
    resultType = IndexType::get(context);
    break;
  case POC::GetListElement:
    result = simplifyGetListElement(operands, resultType);
    break;
  case POC::BindSignature:
    result = simplifyBindSignature(operands, resultType);
    break;
  case POC::Apply:
    result = simplifyApply(operands, resultType);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return result;

  return Base::get(context, opcode, operands, resultType);
}

void ParamOperatorAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (auto operand : getOperands())
    walkAttrsFn(operand);
}

Attribute
ParamOperatorAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(!replAttrs.empty() && replTypes.empty());
  SmallVector<TypedAttr> castedAttrs;
  for (auto attr : replAttrs) {
    castedAttrs.push_back(llvm::dyn_cast<TypedAttr>(attr));
    // Reject attempts to change an operand to something that isn't a TypedAttr.
    if (!castedAttrs.back())
      return {};
  }
  return ParamOperatorAttr::get(getOpcode(), castedAttrs);
}

/// Parameter operators are the basis of parameter expressions and are never
/// simple constants.
bool ParamOperatorAttr::isConstant() const { return false; }

/// Sort operators by opcode, then number of operands, then recursively sort by
/// operand values.
bool ParamOperatorAttr::isLessThan(Attribute rhs) const {
  auto op = llvm::dyn_cast<ParamOperatorAttr>(rhs);
  // Expressions are always to the left of non-expressions.
  if (!op)
    return true;

  // Sort by string value of the opcode.
  if (getOpcode() != op.getOpcode())
    return stringifyPOC(getOpcode()) < stringifyPOC(op.getOpcode());

  // If they are the same opcode, sort by arity. More complex expressions are to
  // the left.
  if (getNumOperands() != op.getNumOperands())
    return getNumOperands() > op.getNumOperands();

  // We know the two subexpressions are different (they'd otherwise be pointer
  // equivalent) so just go compare all of the elements.
  for (auto [lhs, rhs] : llvm::zip(getOperands(), op.getOperands())) {
    if (ParameterAttr::compare(lhs, rhs))
      return true;
    if (ParameterAttr::compare(rhs, lhs))
      return false;
  }

  llvm_unreachable("unexpected equal parameter expressions");
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Implementation
//===----------------------------------------------------------------------===//

Type TypeConstantAttr::getValue() const {
  return static_cast<detail::ConcreteTypeConstantAttrStorage *>(impl)->value;
}

Type ParameterizedTypeConstantAttr::getType() const { return getImpl()->type; }

Type ParameterizedTypeConstantAttr::getValue() const {
  return getImpl()->value;
}
