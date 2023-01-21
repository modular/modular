//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

SymbolRefAttr LIT::getFullyResolvedSymbolRef(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  do {
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
  } while ((op = dyn_cast<mlir::SymbolOpInterface>(op->getParentOp())));

  // Form a reference from the symbols we collected.
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &odsBuilder, OperationState &state,
                         StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addRegion()->push_back(new Block());
}

LogicalResult
FileModuleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyIfTopLevel(symbolTable);
}

/// Modules don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> FileModuleOp::getInputParamDecls() { return {}; }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Return a SymbolConstantAttr for this function, optionally bound to a set
/// of parameter bindings.
SymbolConstantAttr LIT::FuncOp::getBoundReference(ParamBindArrayAttr bindings) {
  if (!bindings) // We allow null for convenience.
    bindings = ParamBindArrayAttr::get(getContext(), {});

  // SymbolConstantAttr provides a type for the SymbolRefAttr with the
  // parameters substituted in.  The function reference binds any parameter
  // bindings present on the access (in bindings), which typically concretizes
  // the signature.
  SignatureType resultType = getFullSignature();
  if (!bindings.empty()) {
    resultType = resultType.getSpecializedSignature(
        bindings, [&]() -> InFlightDiagnostic {
          llvm_unreachable("bad bindings specified for getBoundReference");
        });
    assert(resultType && "bad bindings specified for getBoundReference");
  }

  return SymbolConstantAttr::get(getFullyResolvedSymbolRef(*this), bindings,
                                 resultType);
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "constraints",     "implements", "signature",  "sym_name",
    "valueParamNames", "evaluator",  "defaultImpl"};

/// Parses a LIT Generator.
ParseResult LIT::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;
  Builder &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();

  // Parse the function signature.
  ParamDeclArrayAttr inputParamDecls;
  TypeArrayAttr resultParamTypes;
  ConventionsAttr conventions;
  llvm::SMLoc sigLoc;
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParamTypes) ||
      parser.getCurrentLocation(&sigLoc) ||
      parseFunctionSignature(parser, entryArgs, resultTypes, conventions))
    return failure();

  ConstraintArrayAttr constraints;
  if (parseOptionalConstraints(parser, constraints))
    return failure();
  result.addAttribute(getConstraintsAttrName(result.name), constraints);

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  FunctionType type = builder.getFunctionType(argTypes, resultTypes);
  auto signature =
      parser.getChecked<SignatureType>(parser.getContext(), inputParamDecls,
                                       resultParamTypes, type, conventions);
  if (!signature)
    return failure();

  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signature));

  // Handle keyword argument names.
  SmallVector<StringAttr> names;
  for (OpAsmParser::Argument &arg : entryArgs) {
    StringRef spelling;
    if (arg.ssaName.name.size() < 2)
      return parser.emitError(sigLoc, "arguments requires SSA names");
    if (isdigit(arg.ssaName.name[1])) // %42 -> no name.
      spelling = "";
    else
      spelling = arg.ssaName.name.drop_front();
    names.push_back(builder.getStringAttr(spelling));
  }

  result.addAttribute(getValueParamNamesAttrName(result.name),
                      StringArrayAttr::get(builder.getContext(), names));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this function implements an interface.
  if (succeeded(parser.parseOptionalKeyword("implements"))) {
    SymbolRefAttr implementsAttr;
    if (parser.parseAttribute(implementsAttr, {},
                              getImplementsAttrName(result.name),
                              result.attributes))
      return failure();
  }

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : disallowedAttrNames) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Parse the required function body.
  Region *region = result.addRegion();

  mlir::OptionalParseResult regionResult =
      parser.parseOptionalRegion(*region, entryArgs);
  if (regionResult.has_value()) {
    if (failed(*regionResult))
      return failure();
  } else {
    if (!result.attributes.get(getIsInterfaceAttrName(result.name)))
      return parser.emitError(sigLoc, "expected a function body");
    auto *body = new Block;
    body->addArguments(
        argTypes, SmallVector<Location>(argTypes.size(),
                                        parser.getEncodedSourceLoc(sigLoc)));
    region->push_back(body);
  }

  // Parse an optional evaluator.
  if (parser.parseOptionalKeyword("evaluator"))
    return success();

  Type sigType;
  TypedAttr evaluator;
  if (parseKGENType(parser, sigType) || parser.parseEqual() ||
      parseParamValue(parser, evaluator, sigType))
    return failure();
  result.addAttribute(getEvaluatorAttrName(result.name), evaluator);
  return success();
}

// Print the LIT::FuncOp using the shared printing logic.
void LIT::FuncOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  FuncInterface op = cast<FuncInterface>(getOperation());
  auto func = cast<mlir::FunctionOpInterface>(*op);
  // Print the operation and the function name.
  p << ' ';

  p.printSymbolName(func.getName());
  printOptionalParameterSpec(p, op.getInputParamDeclsAttr(),
                             op.getResultParamTypesAttr());

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  printFunctionSignature(p, func.getFunctionBody(), argTypes,
                         op.getResultTypes(), op.getConventions(),
                         getValueParamNamesAttr());

  // Don't print the following in lit.func.
  SmallVector<StringRef> ignoredAttrNames(
      (ArrayRef<StringRef>(disallowedAttrNames)));

  printFunctionAttributes(p, op, ignoredAttrNames);
  printOptionalConstraints(p, func, cast<DeclInterface>(*op).getConstraints());

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (SymbolRefAttr implementsAttr = getImplementsAttr()) {
    p.printNewline();
    p << "  implements " << implementsAttr;
  }

  p << ' ';
  if (!func.getFunctionBody().front().empty())
    p.printRegion(func.getFunctionBody(), /*printEntryBlockArgs=*/false);
  if (SymbolConstantAttr evaluator = getEvaluatorAttr()) {
    p << " evaluator ";
    printKGENType(p, evaluator.getType());
    p << " = ";
    printParamValue(p, evaluator);
  }
}

// Name the arguments of the region with the valueParamNames.
void LIT::FuncOp::getAsmBlockArgumentNames(
    Region &body, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Set a name for each argument.
  if (body.empty())
    return;
  Block *bodyBlock = getBody();
  for (auto [arg, name] :
       llvm::zip(bodyBlock->getArguments(), getValueParamNames()))
    setNameFn(arg, name);
}

Region *LIT::FuncOp::getCallableRegion() {
  // If the body is empty, return null to indicate that this is an "external"
  // callable.
  if (getBody()->empty())
    return nullptr;
  return &getBodyRegion();
}

ArrayRef<Type> LIT::FuncOp::getCallableResults() { return getResultTypes(); }

/// Reject functions with default arguments whose type does not match the type
/// of the argument they're specifying a default for.
LogicalResult LIT::FuncOp::verify() {
  std::optional<ArrayRef<DefaultArgumentAttr>> defaults = getDefaults();
  if (!defaults)
    return success();

  SmallDenseMap<int64_t, TypedAttr> defaultAttrs;
  for (const DefaultArgumentAttr &def : *defaults)
    defaultAttrs.insert({def.getIndex().getInt(), def.getValue()});

  for (auto [idx, arg] : llvm::enumerate(getArguments())) {
    auto def = defaultAttrs.find(idx);
    if (def != defaultAttrs.end() && arg.getType() != def->second.getType())
      return emitError() << "argument #" << idx << " has type " << arg.getType()
                         << " but default argument has type "
                         << def->second.getType();
  }

  return success();
}

LogicalResult LIT::FuncOp::verifyRegions() {
  // Check that the number of argument labels matches the number of argument
  // types.
  if (getValueParamNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  // Interfaces must have empty bodies and cannot have an implements attribute.
  if (getIsInterface()) {
    if (getImplementsAttr())
      return emitOpError("@interface and @implements decorators "
                         "cannot be set at the same time");
    return success();
  }

  // Generators must have non-empty bodies terminated by a return.
  if (getFunctionBody().empty())
    return emitOpError("expected non-empty function body");
  if (!getBody()->back().hasTrait<OpTrait::IsTerminator>())
    return emitOpError("expected a terminator");

  return success();
}

LogicalResult
LIT::FuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If this function is top-level, see if the parameter definitions and uses
  // within the generator are structured correctly.
  if (failed(verifyIfTopLevel(symbolTable)))
    return failure();

  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  SymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  auto module = KGENModule::from(*this, symbolTable);
  auto interface = module.lookup<GeneratorInterfaceOp>(interfaceSym);
  auto funcInterface = module.lookup<LIT::FuncOp>(interfaceSym);
  if (!interface && (!funcInterface || !funcInterface.getIsInterface()))
    return emitError() << interfaceSym
                       << " does not reference a generator interface";
  TypeArrayAttr interfaceResultParamTypesAttr;
  if (funcInterface)
    interfaceResultParamTypesAttr = funcInterface.getResultParamTypesAttr();
  else
    interfaceResultParamTypesAttr = interface.getResultParamTypesAttr();
  // Result parameters need to match, but input parameters may be inferred.
  if (getResultParamTypesAttr() != interfaceResultParamTypesAttr)
    return emitError() << "lit.func result parameter types must match "
                          "interface types";

  return success();
}

void LIT::FuncOp::build(OpBuilder &builder, OperationState &result) {
  auto context = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto signatureType =
      SignatureType::get(context, ArrayRef<Type>(), {errorType});

  auto emptyParamNames = StringArrayAttr::get(context, {});

  // NOTE: We set an attribute named 'sym_namex' here instead of setting
  // 'sym_name' because we don't /know/ the symbol name on construction and need
  // to set it during signature resolution phase of the parser.
  //
  // Unfortunately, we cannot set it to null because that causes the SymbolTable
  // logic to be extremely cranky and breaks other MLIR invariants.
  //
  // We also cannot completely omit the symbol, because ODS is doing some clever
  // stuff to speed up attribute lookup.  That clever stuff requires that a slot
  // is filled in the attr dict, so we set this thing and remove it when the
  // real name is set.
  result.addAttribute("sym_namex", emptyParamNames);

  result.addAttribute(getValueParamNamesAttrName(result.name), emptyParamNames);
  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signatureType));
  result.addAttribute(getConstraintsAttrName(result.name),
                      ConstraintArrayAttr::get(context, {}));
  result.addRegion()->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

/// Verify that the body has no arguments.
LogicalResult StructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");
  return success();
}

/// Verify that there are no duplicate field names.
LogicalResult StructDeclOp::verifyRegions() {
  SmallDenseMap<StringAttr, StructFieldOp, 8> seenFields;
  for (Operation &op : getFields().front()) {
    auto field = dyn_cast<StructFieldOp>(&op);
    if (!field)
      continue;
    auto [it, inserted] = seenFields.try_emplace(field.getNameAttr(), field);
    if (!inserted) {
      return (field.emitError("duplicate struct field ") << field.getNameAttr())
                 .attachNote(it->second.getLoc())
             << "see previous declaration here";
    }
  }
  return success();
}

/// Verify parameter uses.
LogicalResult
StructDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyIfTopLevel(symbolTable);
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  auto context = builder.getContext();
  build(builder, result, name, ParamDeclArrayAttr::get(context, {}));
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

/// Parse the struct field name as a keyword literal.
static ParseResult parseKeywordAsString(OpAsmParser &p, StringAttr &name) {
  StringRef value;
  if (p.parseKeyword(&value))
    return failure();
  name = p.getBuilder().getStringAttr(value);
  return success();
}

/// Print the struct field name as a keyword literal.
static void printKeywordAsString(OpAsmPrinter &p, Operation *op,
                                 StringAttr name) {
  p << name.getValue();
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static ParameterEvaluator getEvaluatorForBoundStructType(DeclRefType refType) {
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : refType.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());
  return evaluator;
}

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static StructDeclOp lookupStructDecl(SymbolTableCollection &symbolTable,
                                     Operation *user, DeclRefType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto structDecl = module.lookup<StructDeclOp>(ref.getSymbol());
  // Currently, this is impossible to fail because the symbol use was verified
  // by the parameter verifier.
  assert(structDecl && "expected a struct declaration");
  return structDecl;
}

/// Verify the reference struct type.
LogicalResult
StructCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the types of the fields in the operands match those in the
  // struct declaration.
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(getType());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());
  auto fields = structDecl.getFieldDecls();
  unsigned numFields = std::distance(fields.begin(), fields.end());
  if (numFields != getNumOperands())
    return emitOpError("expected ")
           << numFields << " operands but got " << getNumOperands();
  if (getFieldsAttr().size() != numFields)
    return emitOpError("expected ")
           << numFields << " based on the declaration, but got "
           << getFieldsAttr().size();

  for (auto [fieldDecl, fieldAttrInOp, operand, i] :
       llvm::zip(fields, getFieldsAttr(), getOperands(),
                 llvm::seq<unsigned>(0, numFields))) {
    auto nameInDecl = fieldDecl.getNameAttr();
    auto nameInOp = fieldAttrInOp;
    if (nameInDecl != nameInOp) {
      return emitOpError("the field name '")
             << nameInOp << "' at the position #" << i
             << " did not match the name '" << nameInDecl
             << "' in the op declaration.";
    }

    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != operand.getType()) {
      return emitOpError("operand #")
             << i << " has type " << operand.getType()
             << " but corresponding struct field " << fieldDecl.getNameAttr()
             << " expected " << fieldDecl.getType();
    }
  }
  return success();
}

ParseResult LIT::StructCreateOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  auto ctx = parser.getContext();

  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  SmallVector<StringAttr> fieldNames;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            std::string fieldNameStr;
            OpAsmParser::UnresolvedOperand opnd;
            if (parser.parseKeywordOrString(&fieldNameStr) ||
                parser.parseEqual() || parser.parseOperand(opnd))
              return failure();
            allOperands.push_back(opnd);
            fieldNames.push_back(StringAttr::get(ctx, fieldNameStr));
            return success();
          }))
    return failure();

  result.addAttribute("fields", StringArrayAttr::get(ctx, fieldNames));

  FunctionType functionType;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(functionType))
    return failure();

  result.addTypes(functionType.getResults());
  return parser.resolveOperands(allOperands, functionType.getInputs(),
                                allOperandLoc, result.operands);
}

void StructCreateOp::print(OpAsmPrinter &p) {
  p << "(";
  auto op = getOperation();
  llvm::interleaveComma(
      llvm::zip(getFieldsAttr().getValue(), op->getOperands()), p,
      [&](const auto &val) {
        p << std::get<0>(val).str() << "=" << std::get<1>(val);
      });

  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"fields"});
  p << " : ";
  p.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

LogicalResult
StructInsertOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(getType());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != getFieldAttr())
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != getValue().getType())
      return emitOpError("cannot insert value of type ")
             << getValue().getType() << " into struct field " << getFieldAttr()
             << " which expected " << reboundType;
    return success();
  }

  return emitOpError("struct ")
         << getType().getSymbol() << " has no field named " << getFieldAttr();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         DeclRefType ref, StringAttr fieldName, Type type) {
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(ref);
  StructDeclOp structDecl = lookupStructDecl(symbolTable, op, ref);

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != type)
      return op->emitOpError("cannot extract value of type ")
             << type << " from struct field " << fieldName << " which has type "
             << reboundType;
    return success();
  }

  return op->emitOpError("struct ")
         << ref.getSymbol() << " has no field named " << fieldName;
}

LogicalResult
StructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyStructFieldAndType(symbolTable, *this, getContainer().getType(),
                                  getFieldAttr(), getValue().getType());
}

void StructExtractOp::build(OpBuilder &builder, OperationState &result,
                            Value structBase, StructFieldOp field) {
  auto structType = cast<DeclRefType>(structBase.getType());
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(structType);
  build(builder, result, evaluator.getReboundType(field.getType()),
        field.getNameAttr(), structBase);
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  // Fold
  //  %S = lit.struct.create(a=%a, b=%b)
  //  %x = lit.struct.extract %S[a]
  // into %a.
  if (auto create = getContainer().getDefiningOp<StructCreateOp>()) {
    for (size_t i = 0, e = create->getNumOperands(); i < e; i++) {
      if (create.getFieldsAttr()[i] == getFieldAttr())
        return create.getOperand(i);
    }
    // A field referred to in the struct.extract op must appear in the previous
    // struct.create op that we're scanning in the loop above.
    llvm_unreachable("Didn't find the field to extract in struct.create op.");
  }
  // Fold
  //    %S = lit.struct.insert %x, %S0[a]
  //    %y = lit.struct.extract %S[a]
  // into %x
  if (auto insert = getContainer().getDefiningOp<StructInsertOp>()) {
    if (insert.getFieldAttr() == getFieldAttr())
      return insert.getOperand(0);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult
StructGEPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TypedAttr refExpr = getContainer().getType().getElementType();
  return verifyStructFieldAndType(
      symbolTable, *this,
      cast<DeclRefType>(cast<TypeConstantAttr>(refExpr).getValue()),
      getFieldAttr(),
      ParamRefType::get(getResult().getType().getElementType()));
}

void StructGEPOp::build(OpBuilder &builder, OperationState &result,
                        Value structBasePtr, StructFieldOp field) {
  TypedAttr refExpr =
      cast<POP::PointerType>(structBasePtr.getType()).getElementType();
  auto structType =
      cast<DeclRefType>(cast<TypeConstantAttr>(refExpr).getValue());

  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(structType);
  build(builder, result,
        POP::PointerType::get(evaluator.getReboundType(field.getType())),
        field.getNameAttr(), structBasePtr);
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

static ParseResult parseExceptRegion(OpAsmParser &p, Region &region) {
  OpAsmParser::Argument arg;
  if (p.parseLParen() || p.parseArgument(arg, /*allowType=*/true) ||
      p.parseRParen() || p.parseRegion(region, arg))
    return failure();
  return success();
}

static void printExceptRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << '(';
  p.printRegionArgument(region.getArgument(0));
  p << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

LogicalResult TryOp::verify() {
  if (getTryRegion().getNumArguments() != 0)
    return emitOpError("expected try region to have zero arguments");
  if (getExceptRegion().getNumArguments() != 1)
    return emitOpError("expected except region to have one arguments");
  if (getElseRegion().getNumArguments() != 0)
    return emitOpError("expected else region to have zero arguments");
  return success();
}

void TryOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0);
}

ValueRange TryOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return {};
  return getRegion(*target).getArguments();
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

bool TryYieldOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  Region *region = (*this)->getParentRegion();
  // Figure out which region this yield is in.
  if (!isa<TryOp>(region->getParentOp()))
    region = region->getParentRegion();

  // The region indices of the try operation.
  enum { TRY, EXCEPT, ELSE };
  switch (region->getRegionNumber()) {
  case TRY:
    // Yield from the 'try' region branches to the 'else' region.
    targets.emplace_back(ELSE);
    break;
  case EXCEPT:
  case ELSE:
    // Yield from either the 'except' or 'else' regions branches back to the
    // parent operation.
    targets.emplace_back(std::nullopt);
    break;
  default:
    llvm_unreachable("unknown lit.try region");
  }
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

bool TryRaiseOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryRaiseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(1, (*this)->getOperands());
}

//===----------------------------------------------------------------------===//
// LetDeclOp / VarDeclOp
//===----------------------------------------------------------------------===//

void LetDeclOp::build(OpBuilder &builder, OperationState &state,
                      Type resultType, StringAttr name) {
  state.addAttribute(getNameAttrName(state.name), name);
  state.addTypes(resultType);
}

void LetDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

void VarDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

static POP::CoroutineType getCoroutineOfResultTypes(Type type) {
  return POP::CoroutineType::get(cast<SignatureType>(type));
}

LogicalResult AsyncCallOp::verify() {
  if (cast<SignatureType>(getCallee().getType()).isAsync())
    return success();
  return emitOpError("callable must be 'async'");
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::ReturnOp::verify() {
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.func` operation");
  return checkResultArgumentTypes(*this, getParameters(),
                                  func.getResultParamTypes(),
                                  func.getResultTypes());
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

LogicalResult RaiseOp::verify() {
  Operation *op = *this;
  auto func = op->getParentOfType<LIT::FuncOp>();
  if (func && func.isThrows())
    return success();

  auto tryOp = op->getParentOfType<TryOp>();
  if (tryOp && tryOp.getTryRegion().isAncestor(op->getBlock()->getParent()))
    return success();

  return emitOpError("must be nested inside the 'try' region of a `lit.try` "
                     "operation or within a `lit.func` that throws");
}

//===----------------------------------------------------------------------===//
// BreakOp / ContinueOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyBreakOrContinueOp(Operation *op) {
  if (op->getParentOfType<HLCF::LoopOp>())
    return success();
  return op->emitOpError("must be nested within an `hlcf.loop` operation");
}

LogicalResult BreakOp::verify() { return verifyBreakOrContinueOp(*this); }
LogicalResult ContinueOp::verify() { return verifyBreakOrContinueOp(*this); }

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.cpp.inc"
