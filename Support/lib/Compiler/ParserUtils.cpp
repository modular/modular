//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/ParserUtils.h"

#include "llvm/ADT/StringSet.h"

using namespace M;

ParseResult M::parseParenOperandListWithShadowing(
    OpAsmParser &parser, OperationState &state,
    SmallVectorImpl<OpAsmParser::Argument> &argumentInfo) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  auto parseOperandFn = [&]() -> ParseResult {
    // The operand to capture as part of the ops inputs.
    OpAsmParser::UnresolvedOperand unresolvedOperand;
    // The corresponding argument to use for the op's nested blocks.
    OpAsmParser::Argument arg;

    // Parse the input operand.
    if (parser.parseOperand(unresolvedOperand,
                            /*allowResultNumber=*/false))
      return failure();

    // Parse an optional 'as' clause to name the nested block
    // argument. This can be used to given nested blocks a unique argument name
    // even if the same SSA value is repeated as on operand.
    if (succeeded(parser.parseOptionalKeyword("as"))) {
      if (parser.parseOperand(arg.ssaName, /*allowResultNumber=*/false))
        return failure();
    } else {
      // The nested blocks will 'pun' the operand name, presumably without
      // ambiguity.
      arg.ssaName = unresolvedOperand;
    }

    // Parse type annotation
    if (parser.parseColon() || parser.parseType(arg.type))
      return failure();

    // No block argument attributes.
    NamedAttrList attrs;
    arg.attrs = attrs.getDictionary(parser.getContext());

    argumentInfo.push_back(arg);

    // Resolve the input operand into the operation state.
    if (parser.resolveOperand(unresolvedOperand, arg.type, state.operands))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperandFn, "in operand list"))
    return failure();

  StringSet<> argumentNames;
  for (auto arg : argumentInfo)
    argumentNames.insert(arg.ssaName.name);
  if (argumentNames.size() != argumentInfo.size())
    return parser.emitError(
        loc, "has duplicate SSA values in its operand list which have not been "
             "renamed apart by 'as' clauses.");

  return success();
}

ParseResult M::parseParenOperandListWithDefaultType(OpAsmParser &parser,
                                                    OperationState &state,
                                                    Type defaultType) {
  auto parseOperandFn = [&]() -> ParseResult {
    // The operand to capture as part of the ops inputs.
    OpAsmParser::UnresolvedOperand unresolvedOperand;

    // Parse the input operand.
    if (parser.parseOperand(unresolvedOperand,
                            /*allowResultNumber=*/false))
      return failure();

    // Parse optional type annotation
    Type type = defaultType;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseType(type))
        return failure();
    }

    // Resolve the input operand into the operation state.
    if (parser.resolveOperand(unresolvedOperand, type, state.operands))
      return failure();

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
}

/// Returns true if all values are distinct.
static bool allDistinct(const SmallVector<Value> &values) {
  DenseSet<Value> unique;
  for (auto value : values)
    unique.insert(value);
  return unique.size() == values.size();
}

ParseResult M::parseRegionWithShadowing(
    OpAsmParser &parser, const SmallVector<OpAsmParser::Argument> &argumentInfo,
    Region &region) {
  return parser.parseRegion(region, argumentInfo,
                            /*enableNameShadowing=*/true);
}

void M::printParenOperandListWithShadowing(
    OpAsmPrinter &printer, const OperandRange &operands,
    const Block::BlockArgListType &arguments) {
  assert(operands.size() == arguments.size());
  bool needAs = !allDistinct(operands);
  printer << "(";
  bool first = true;
  for (auto [op, arg] : llvm::zip(operands, arguments)) {
    if (first)
      first = false;
    else
      printer << ", ";
    printer << op;
    if (needAs)
      printer << " as " << arg;
    printer << ": " << op.getType();
  }
  printer << ")";
}

void M::printParenOperandListWithDefaultType(OpAsmPrinter &printer,
                                             const OperandRange &operands,
                                             Type defaultType) {
  auto printArg = [&](Value arg) {
    printer << arg;
    if (arg.getType() != defaultType)
      printer << ": " << arg.getType();
  };
  printer << '(';
  llvm::interleaveComma(operands, printer, printArg);
  printer << ')';
}

void M::printRegionWithShadowing(OpAsmPrinter &printer,
                                 const OperandRange &operands, Region &region) {
  assert(operands.size() == region.getNumArguments());
  bool canShadow = allDistinct(operands);
  if (canShadow) {
    printer.shadowRegionArgs(region, operands);
    printer.printRegion(region, /*printEntryBlockArgs=*/false);
  } else {
    printer.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

ParseResult M::parseBufferSignature(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &buffers,
    SmallVectorImpl<Type> &bufferTypes,
    InOutSignatureAttr &inOutSignatureAttr) {
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics;
  auto parseOperandFn = [&]() -> ParseResult {
    llvm::SMLoc loc;

    // Parse the in/out/mut keyword.
    StringRef inOutMut;
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&inOutMut))
      return failure();
    if (inOutMut == "in")
      semantics.emplace_back(InOutSignatureAttr::kIn);
    else if (inOutMut == "out")
      semantics.emplace_back(InOutSignatureAttr::kOut);
    else if (inOutMut == "mut")
      semantics.emplace_back(InOutSignatureAttr::kMut);
    else
      return parser.emitError(loc) << "expecting 'in', 'out' or 'mut' keyword";

    // Parse the operand proper.
    OpAsmParser::UnresolvedOperand &unresolvedOperand = buffers.emplace_back();
    if (parser.parseOperand(unresolvedOperand))
      return failure();

    // Parse the type annotation.
    Type &bufferType = bufferTypes.emplace_back();
    if (parser.parseColon() || parser.parseType(bufferType))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperandFn, "in buffer operand list"))
    return failure();

  inOutSignatureAttr = InOutSignatureAttr::get(parser.getContext(), semantics);

  return success();
}

void M::printBufferSignature(OpAsmPrinter &printer, const Operation *opIgnored,
                             ValueRange buffers, TypeRange bufferTypes,
                             InOutSignatureAttr inOutSignatureAttr) {
  size_t arity = inOutSignatureAttr.size();
  assert(buffers.size() == arity);
  assert(bufferTypes.size() == arity);
  printer << "(";
  for (size_t i = 0; i < arity; ++i) {
    if (i > 0)
      printer << ", ";
    switch (inOutSignatureAttr[i]) {
    case InOutSignatureAttr::kNone:
      llvm::llvm_unreachable_internal(
          "unexpected kNone buffer semantics in signature");
      break;
    case InOutSignatureAttr::kIn:
      printer << "in ";
      break;
    case InOutSignatureAttr::kOut:
      printer << "out ";
      break;
    case InOutSignatureAttr::kMut:
      printer << "mut ";
      break;
    }
    printer << buffers[i];
    printer << " : ";
    printer << bufferTypes[i];
  }
  printer << ")";
}

/// Returns true if the given string can be represented as a bare identifier
/// compatible with the MLIR lexer.
static bool isBareIdentifier(StringRef name) {
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

void StreamAsmPrinter::printString(StringRef string) {
  os << "\"";
  llvm::printEscapedString(string, os);
  os << '"';
}

void StreamAsmPrinter::printKeywordOrString(StringRef keyword) {
  if (isBareIdentifier(keyword)) {
    os << keyword;
    return;
  }
  os << "\"";
  llvm::printEscapedString(keyword, os);
  os << '"';
}

void StreamAsmPrinter::printSymbolName(StringRef symbolRef) {
  os << '@';
  printKeywordOrString(symbolRef);
}

void StreamAsmPrinter::printResourceHandle(
    const mlir::AsmDialectResourceHandle &resource) {
  auto *interface = cast<OpAsmDialectInterface>(resource.getDialect());
  os << interface->getResourceKey(resource);
}

void StreamAsmPrinter::printFloat(const APFloat &value) {
  if (!value.isInfinity() && !value.isNaN()) {
    SmallString<128> strValue;
    value.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                   /*TruncateZero=*/false);
    if (APFloat(value.getSemantics(), strValue).bitwiseIsEqual(value)) {
      os << strValue;
      return;
    }
    strValue.clear();
    value.toString(strValue);
    if (strValue.str().contains('.')) {
      os << strValue;
      return;
    }
  }
  SmallVector<char, 16> str;
  APInt apInt = value.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
}
