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

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace M;
using namespace M::MOGG;

//===----------------------------------------------------------------------===//
// MOGGDialect attribute registration
//===----------------------------------------------------------------------===//

void MOGGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr verification
//===----------------------------------------------------------------------===//

LogicalResult
FusionInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       Attribute root) {
  if (!isa<FusionLeafAttr, FusionSequenceAttr, FusionHorizontalAttr,
           FusionElementwiseAttr, FusionEpilogueAttr, FusionPrologueAttr,
           FusionViewAttr, FusionMutableLoadAttr, FusionRowwiseAttr>(root))
    return emitError() << "root must be a valid fusion info attribute";
  return success();
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr::toString
//===----------------------------------------------------------------------===//

namespace {

// Strip the dialect prefix (e.g., "mo.add" -> "add") from an operation name
// to make trace descriptions more readable.
std::string stripDialectPrefix(StringRef opName) {
  auto dotPos = opName.find('.');
  if (dotPos != StringRef::npos)
    return opName.substr(dotPos + 1).str();
  return opName.str();
}

void nodeToString(Attribute node, llvm::raw_string_ostream &os,
                  bool includeLineNumbers, bool quote,
                  bool stripDialectPrefixes) {
  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        if (quote)
          os << '"';
        std::string name = stripDialectPrefixes
                               ? stripDialectPrefix(leaf.getOpName().getValue())
                               : leaf.getOpName().getValue().str();
        llvm::printEscapedString(name, os);
        if (quote)
          os << '"';
        if (includeLineNumbers)
          if (auto fileLoc = dyn_cast_or_null<FileLineColLoc>(leaf.getLoc()))
            if (fileLoc.getLine() > 0)
              os << ':' << fileLoc.getLine();
      })
      .Case<FusionSequenceAttr, FusionHorizontalAttr, FusionElementwiseAttr,
            FusionViewAttr, FusionMutableLoadAttr, FusionRowwiseAttr>(
          [&](auto attr) {
            os << decltype(attr)::getFusionName() << '(';
            llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
              nodeToString(child, os, includeLineNumbers, quote,
                           stripDialectPrefixes);
            });
            os << ')';
          })
      .Case<FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        os << decltype(attr)::getFusionName() << '(';
        llvm::interleaveComma(attr.getChildren(), os, [&](Attribute child) {
          nodeToString(child, os, includeLineNumbers, quote,
                       stripDialectPrefixes);
        });
        os << ", operandIndices=[";
        llvm::interleaveComma(attr.getOperandIndices(), os);
        os << "])";
      });
}

/// Pretty-print a fusion info tree node with tree-drawing connectors.
/// If \p highlightLeaves is non-empty, leaf nodes whose op name appears in
/// the set are annotated with " <-- error" in the output.
void nodeToPrettyTree(Attribute node, llvm::raw_string_ostream &os,
                      const llvm::DenseSet<Attribute> &highlightLeaves,
                      StringRef prefix = "", bool isLast = true) {
  auto emitChildren = [&](auto attr, ArrayRef<Attribute> children,
                          ArrayRef<uint32_t> operandIndices = {}) {
    os << decltype(attr)::getFusionName();
    if (!operandIndices.empty()) {
      os << " (operandIndices=[";
      llvm::interleaveComma(operandIndices, os);
      os << "])";
    }
    os << '\n';
    StringRef connector = isLast ? "    " : "│   ";
    std::string childPrefix = (prefix + connector).str();
    for (size_t i = 0; i < children.size(); ++i) {
      bool childIsLast = (i == children.size() - 1);
      os << childPrefix << (childIsLast ? "└── " : "├── ");
      nodeToPrettyTree(children[i], os, highlightLeaves, childPrefix,
                       childIsLast);
    }
  };

  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        os << leaf.getOpName().getValue();
        if (auto fileLoc = dyn_cast_or_null<FileLineColLoc>(leaf.getLoc())) {
          StringRef fn = fileLoc.getFilename().getValue();
          uint32_t line = fileLoc.getLine();
          if (!fn.empty() && line)
            os << " (" << fn << ':' << line << ')';
          else if (line)
            os << ':' << line;
        }
        if (auto sig = leaf.getSignature()) {
          if (auto funcTy = dyn_cast<FunctionType>(sig.getValue())) {
            os << " : (";
            llvm::interleaveComma(funcTy.getInputs(), os);
            os << ") -> ";
            if (funcTy.getResults().size() == 1)
              os << funcTy.getResults()[0];
            else {
              os << '(';
              llvm::interleaveComma(funcTy.getResults(), os);
              os << ')';
            }
          }
        }
        if (highlightLeaves.count(leaf))
          os << "  <-- error";
        os << '\n';
      })
      .Case<FusionSequenceAttr, FusionHorizontalAttr, FusionElementwiseAttr,
            FusionViewAttr, FusionMutableLoadAttr, FusionRowwiseAttr>(
          [&](auto attr) { emitChildren(attr, attr.getChildren()); })
      .Case<FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        emitChildren(attr, attr.getChildren(), attr.getOperandIndices());
      });
}

} // namespace

std::string FusionInfoAttr::toString(bool includeLineNumbers, bool quote,
                                     bool stripDialectPrefixes) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToString(getRoot(), os, includeLineNumbers, quote, stripDialectPrefixes);
  return result;
}

/// Drops the dialect prefix of every dotted op name in `name`, including ones
/// nested in a wrapper: `mo.reduce.rms_norm` -> `reduce.rms_norm`,
/// `shapeof(mo.range)` -> `shapeof(range)`.
static std::string stripNestedDialectPrefixes(StringRef name) {
  std::string out;
  bool atOpStart = true;
  for (size_t i = 0; i < name.size();) {
    if (atOpStart) {
      size_t end = i;
      while (end < name.size() &&
             (llvm::isAlnum(name[end]) || name[end] == '_'))
        ++end;
      if (end > i && end < name.size() && name[end] == '.')
        i = end + 1;
    }
    char c = name[i++];
    out.push_back(c);
    atOpStart = c == '(' || c == ',' || c == ' ';
  }
  return out;
}

static void collectLeafNames(Attribute node,
                             SmallVectorImpl<std::string> &names) {
  llvm::TypeSwitch<Attribute>(node)
      .Case<FusionLeafAttr>([&](FusionLeafAttr leaf) {
        names.push_back(
            stripNestedDialectPrefixes(leaf.getOpName().getValue()));
      })
      .Case<FusionSequenceAttr, FusionHorizontalAttr, FusionElementwiseAttr,
            FusionViewAttr, FusionMutableLoadAttr, FusionRowwiseAttr,
            FusionEpilogueAttr, FusionPrologueAttr>([&](auto attr) {
        for (Attribute child : attr.getChildren())
          collectLeafNames(child, names);
      });
}

SmallVector<std::string> FusionInfoAttr::getLeafNames() const {
  SmallVector<std::string> names;
  collectLeafNames(getRoot(), names);
  return names;
}

std::string
FusionInfoAttr::toPrettyTree(const llvm::DenseSet<Attribute> &highlightLeaves,
                             bool indent) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  nodeToPrettyTree(getRoot(), os, highlightLeaves);
  while (!result.empty() && result.back() == '\n')
    result.pop_back();
  if (!indent)
    return result;
  std::string indented;
  llvm::raw_string_ostream ios(indented);
  for (auto line : llvm::split(result, '\n'))
    ios << "  " << line << '\n';
  return indented;
}

//===----------------------------------------------------------------------===//
// FusionInfo custom printing/parsing for ops
//===----------------------------------------------------------------------===//

void M::MOGG::printFusionInfo(AsmPrinter &p, Operation *op,
                              FusionInfoAttr fusionInfo) {
  if (!fusionInfo)
    return;
  p << " fusionInfo = ";
  p.printAttribute(fusionInfo);
}

ParseResult M::MOGG::parseFusionInfo(AsmParser &p, FusionInfoAttr &fusionInfo) {
  if (failed(p.parseOptionalKeyword("fusionInfo")))
    return success();
  return failure(p.parseEqual() || p.parseAttribute(fusionInfo));
}

//===----------------------------------------------------------------------===//
// TableGen-generated code
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MOGGDialect/MOGGAttrs.cpp.inc"
