//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"

using namespace mlir;

ParseResult
M::MOGG::parseBindings(OpAsmParser &parser, StringRef keyword,
                       SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand> &outers,
                       SmallVectorImpl<Type> &outerTypes, Type impliedType,
                       bool required) {
  if (required) {
    if (parser.parseKeyword(keyword))
      return failure();
  } else if (failed(parser.parseOptionalKeyword(keyword))) {
    return success();
  }
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Braces, [&]() -> ParseResult {
        OpAsmParser::Argument inner;
        OpAsmParser::UnresolvedOperand outer;
        if (parser.parseArgument(inner) || parser.parseEqual() ||
            parser.parseOperand(outer))
          return failure();
        inner.type = impliedType;
        if (succeeded(parser.parseOptionalColon())) {
          if (parser.parseType(inner.type))
            return failure();
        } else if (!impliedType) {
          return parser.emitError(parser.getCurrentLocation(),
                                  "expected ':' and a type");
        }
        entryArgs.push_back(inner);
        outers.push_back(outer);
        outerTypes.push_back(inner.type);
        return success();
      });
}

void M::MOGG::printBindings(OpAsmPrinter &p, StringRef keyword,
                            ValueRange outers, ArrayRef<BlockArgument> inners,
                            Type impliedType, bool required) {
  if (outers.empty() && !required)
    return;
  p << keyword << " {";
  llvm::interleaveComma(llvm::zip_equal(inners, outers), p, [&](auto pair) {
    auto [inner, outer] = pair;
    p << inner << " = " << outer;
    if (inner.getType() != impliedType)
      p << " : " << inner.getType();
  });
  p << "}";
}

ParseResult
M::MOGG::parseLeadingArgs(OpAsmParser &parser,
                          SmallVectorImpl<OpAsmParser::Argument> &args) {
  return parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                                  /*allowType=*/true);
}

void M::MOGG::printLeadingArgs(OpAsmPrinter &p, ArrayRef<BlockArgument> args) {
  p << "(";
  llvm::interleaveComma(
      args, p, [&](BlockArgument arg) { p << arg << " : " << arg.getType(); });
  p << ")";
}

ParseResult M::MOGG::parseKernelParamsBody(OpAsmParser &parser,
                                           DictionaryAttr &params) {
  SmallVector<NamedAttribute> entries;
  StringRef key;
  while (succeeded(parser.parseOptionalKeyword(&key))) {
    Attribute value;
    if (parser.parseEqual() || parser.parseAttribute(value))
      return failure();
    entries.emplace_back(StringAttr::get(parser.getContext(), key), value);
    if (failed(parser.parseOptionalComma()))
      break;
  }
  params = DictionaryAttr::get(parser.getContext(), entries);
  return success();
}

void M::MOGG::printKernelParamsBody(OpAsmPrinter &p, Operation *,
                                    DictionaryAttr params) {
  llvm::interleaveComma(params, p, [&](NamedAttribute attr) {
    p.printKeywordOrString(attr.getName().strref());
    p << " = ";
    p.printAttribute(attr.getValue());
  });
}

ParseResult M::MOGG::parseOptionalKernelParams(OpAsmParser &parser,
                                               DictionaryAttr &params) {
  if (failed(parser.parseOptionalLSquare()))
    return success();
  return failure(parseKernelParamsBody(parser, params) ||
                 parser.parseRSquare());
}

void M::MOGG::printOptionalKernelParams(OpAsmPrinter &p, Operation *op,
                                        DictionaryAttr params) {
  if (!params || params.empty())
    return;
  p << "[";
  printKernelParamsBody(p, op, params);
  p << "]";
}

ParseResult M::MOGG::parseOptionalDispatch(OpAsmParser &parser,
                                           DispatchKindAttr &kind) {
  if (failed(parser.parseOptionalKeyword("dispatch")))
    return success();
  StringRef keyword;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen())
    return failure();
  std::optional<DispatchKind> value = symbolizeDispatchKind(keyword);
  if (!value)
    return parser.emitError(loc, "expected 'deferred' or 'immediate'");
  kind = DispatchKindAttr::get(parser.getContext(), *value);
  return success();
}

bool M::MOGG::printOptionalDispatch(OpAsmPrinter &p, DispatchKindAttr kind,
                                    DispatchKind defaultKind) {
  if (!kind || kind.getValue() == defaultKind)
    return false;
  p << "dispatch(" << stringifyDispatchKind(kind.getValue()) << ")";
  return true;
}

ParseResult M::MOGG::parseTileFusionLane(OpAsmParser &parser,
                                         TileFusionLaneAttr &lane) {
  StringRef keyword;
  if (parser.parseKeyword("lane") || parser.parseEqual())
    return failure();
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();
  std::optional<TileFusionLane> value = symbolizeTileFusionLane(keyword);
  if (!value)
    return parser.emitError(loc, "expected 'compute' or 'store'");
  lane = TileFusionLaneAttr::get(parser.getContext(), *value);
  return success();
}

void M::MOGG::printTileFusionLane(OpAsmPrinter &p, TileFusionLaneAttr lane) {
  p << "lane = " << stringifyTileFusionLane(lane.getValue());
}

ParseResult M::MOGG::parseImmediateDispatch(OpAsmParser &parser,
                                            DispatchKindAttr &kind) {
  return parseOptionalDispatch(parser, kind);
}

void M::MOGG::printImmediateDispatch(OpAsmPrinter &p, Operation *,
                                     DispatchKindAttr kind) {
  printOptionalDispatch(p, kind, DispatchKind::Immediate);
}
