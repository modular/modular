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

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/MojoParser/StabilityMarkers.h"
#include "Mojo/MojoTooling/DeclRendering.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "MojoToolingInternals.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <optional>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

namespace {

/// Stability information extracted from the @stable decorator on a decl.
struct StableInfo {
  bool isStable = false;
  bool isStabilityTracked = false;
  std::string sinceVersion;
};

/// Rendered `where` conditions of a decl's conditional trait conformances,
/// keyed by the conformed trait's symbol.
using ConformanceConditionMap =
    llvm::DenseMap<KGEN::TraitSymbolAttr, std::string>;

static std::string extractSymbolLeafName(SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().empty())
    return symbol.getRootReference().getValue().str();
  return symbol.getNestedReferences().back().getAttr().getValue().str();
}

static std::string extractBaseTypeName(ASTType astType, StringRef fullTypeStr) {
  if (astType) {
    Type type = astType.mlirType;
    if (auto structType = sugarDynCast<LIT::StructType>(type)) {
      if (auto symbol = structType.getSymbol())
        return extractSymbolLeafName(symbol);
    } else if (sugarIsa<LIT::RefType>(type)) {
      if (ASTType elementType = astType.getReferenceElementType())
        return extractBaseTypeName(elementType, fullTypeStr);
    } else if (auto traitType = sugarDynCast<LIT::TraitType>(type)) {
      if (!traitType.getSymbols().empty())
        return extractSymbolLeafName(
            traitType.getSymbols().front().getSymbol());
    } else if (sugarIsa<LIT::OriginSetType>(type)) {
      return "OriginSet";
    }
  }

  StringRef baseType = fullTypeStr;
  if (baseType.starts_with("fn(") || baseType.starts_with("fn[") ||
      baseType.starts_with("def(") || baseType.starts_with("def["))
    return baseType.trim().str();
  if (size_t bracket = baseType.find('['); bracket != StringRef::npos)
    baseType = baseType.substr(0, bracket);
  return baseType.trim().str();
}

static std::string extractBaseTypeName(StringRef fullTypeStr) {
  return extractBaseTypeName(ASTType{}, fullTypeStr);
}

static std::string extractModulePathFromDecl(ASTDecl *decl) {
  SmallVector<StringRef> components;
  for (ASTDecl *current = decl; current; current = current->getParentDecl())
    if (isa<FileModuleASTDecl, PackageASTDecl>(current))
      if (auto name = current->getName())
        components.push_back(*name);
  std::reverse(components.begin(), components.end());
  return llvm::join(components, ".");
}

static ASTDecl *tryResolveTypeToDecl(StringRef typeName,
                                     SharedState &sharedState,
                                     const ASTDecl *contextDecl) {
  for (const ASTDecl *current = contextDecl; current;
       current = current->getParentDecl()) {
    for (auto [name, children] : current->getDeclsInScope()) {
      if (name.getValue() != typeName)
        continue;
      for (ASTDecl *child : children)
        if (isa_and_nonnull<StructASTDecl, TraitASTDecl, AliasASTDecl>(child))
          return child;
    }
  }
  if (contextDecl)
    return sharedState.lookupBuiltinTrait(typeName, SMLoc());
  return nullptr;
}

static std::string generateDocPath(StringRef module, StringRef typeName,
                                   StringRef docsBasePath,
                                   bool isAlias = false) {
  if (typeName.empty())
    return {};

  SmallVector<StringRef> components;
  module.split(components, '.');
  std::string moduleStr;
  for (StringRef component : components) {
    if (component == "__init__")
      continue;
    if (!moduleStr.empty())
      moduleStr += ".";
    moduleStr += component.str();
  }

  std::string path;
  if (StringRef(moduleStr).starts_with("std.")) {
    path = moduleStr;
    if (StringRef(path).ends_with(".index"))
      path = StringRef(path).drop_back(6).str() + ".index_";
  } else if (!docsBasePath.empty()) {
    path = docsBasePath.str();
    if (!moduleStr.empty())
      path += "/" + moduleStr;
  } else {
    path = moduleStr;
  }
  std::replace(path.begin(), path.end(), '.', '/');
  path = "/" + path;

  if (!isAlias)
    return path.empty() ? typeName.str() : path + "/" + typeName.str();

  std::string anchor = typeName.lower();
  return path.empty() ? "#" + anchor : path + "/#" + anchor;
}

struct ResolvedPath {
  std::string docPath;
};

static llvm::json::Object
extractLibraryInfo(StringRef typeStr,
                   const ASTDecl *currentDeclContext = nullptr,
                   SharedState *sharedState = nullptr) {
  static std::map<std::string, ResolvedPath> pathCache;
  static std::mutex cacheMutex;

  auto makeMetadata = [&](const ResolvedPath &resolved) {
    llvm::json::Object metadata{{"type", typeStr.str()}};
    if (!resolved.docPath.empty())
      metadata["path"] = resolved.docPath;
    return metadata;
  };

  StringRef baseType = typeStr;
  if (size_t bracket = baseType.find('['); bracket != StringRef::npos)
    baseType = baseType.substr(0, bracket);
  baseType = baseType.trim();

  std::string cacheKey = baseType.str();
  {
    std::lock_guard<std::mutex> lock(cacheMutex);
    if (auto cached = pathCache.find(cacheKey); cached != pathCache.end())
      return makeMetadata(cached->second);
  }

  if (sharedState) {
    if (ASTDecl *decl =
            tryResolveTypeToDecl(baseType, *sharedState, currentDeclContext)) {
      std::string modulePath = extractModulePathFromDecl(decl);
      if (!modulePath.empty()) {
        bool isAlias = isa<AliasASTDecl>(decl);
        std::string typeName =
            isAlias ? baseType.str() : extractBaseTypeName(typeStr);
        ResolvedPath resolved{generateDocPath(
            modulePath, typeName, sharedState->getDocsBasePath(), isAlias)};
        {
          std::lock_guard<std::mutex> lock(cacheMutex);
          pathCache[cacheKey] = resolved;
        }
        return makeMetadata(resolved);
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(cacheMutex);
    pathCache[cacheKey] = ResolvedPath{};
  }
  return makeMetadata(ResolvedPath{});
}

} // namespace

static std::string getPogMarkdownDocString(ASTDecl &decl);

/// Returns true if the operation is nested inside a package that has opted into
/// stability tracking.
static bool isInStabilityOptedPackage(mlir::Operation *op) {
  for (mlir::Operation *cur = op->getParentOp(); cur;
       cur = cur->getParentOp()) {
    if (auto pkgOp = dyn_cast<PackageOp>(cur))
      if (isPackageOptedIntoStabilityMarkers(pkgOp.getSymName()))
        return true;
  }
  return false;
}

/// Extract @stable decorator information from an op into a StableInfo struct.
static StableInfo computeStableInfo(StabilityDecoratorInterface itf) {
  bool tracked = isInStabilityOptedPackage(itf.getOperation());
  if (!itf.getHasStableDecorator())
    return {/*isStable=*/false, /*isStabilityTracked=*/tracked,
            /*sinceVersion=*/{}};
  return {/*isStable=*/true, /*isStabilityTracked=*/tracked,
          itf.getSinceVersion().str()};
}

/// Parses compound trait types like "Representable & Copyable & Movable" into
/// individual components. Returns a vector of individual trait names, or a
/// single-element vector for non-compound types.
static SmallVector<std::string> parseCompoundTraitType(StringRef typeStr) {
  SmallVector<std::string> traits;

  // Check if this is a compound type (contains &)
  if (!typeStr.contains(" & ")) {
    // Not a compound type, return as-is
    traits.push_back(typeStr.str());
    return traits;
  }

  // Split by " & " and trim each component
  SmallVector<StringRef> components;
  typeStr.split(components, " & ");

  for (StringRef component : components) {
    StringRef trimmed = component.trim();
    if (!trimmed.empty()) {
      traits.push_back(trimmed.str());
    }
  }

  return traits;
}

/// Two spaces that are forcefully added to markdown lines that can be used for
/// indentation.
static constexpr const char *kMarkdownIndent = "&nbsp;&nbsp;";

/// Return an ordering priority number for the given decl name. Lower numbers
/// are ordered first.
static unsigned getDeclNamePriority(StringRef name) {
  // If the name is a special function, use that as the priority.
  SpecialFunctionKind specialFnKind = SpecialFunctionInfo::lookupKind(name);
  if (specialFnKind != SpecialFunctionKind::kNormal)
    return static_cast<unsigned>(specialFnKind);

  // Otherwise, we can't discern any priority from the name.
  return std::numeric_limits<unsigned>::max();
}

/// Given the names of two decls, returns if `lhs` should be ordered before
/// `rhs`.
static bool compareDeclNames(StringRef lhs, StringRef rhs) {
  // If the names are the same, we don't need to do anything.
  if (lhs == rhs)
    return false;

  // First compare the priority of the names.
  unsigned lhsPriority = getDeclNamePriority(lhs);
  unsigned rhsPriority = getDeclNamePriority(rhs);
  if (lhsPriority != rhsPriority)
    return lhsPriority < rhsPriority;

  // If there is no name priority, then leave in the original source order.
  return false;
}

/// Return the indentation level of the first line of the string.
static size_t getIndentationLevel(StringRef str) {
  return str.size() - str.ltrim().size();
}

/// Skip a structured documentation section whose entries are computed on
/// demand instead of being retained in the current rendering data.
static void skipDocumentationSection(ArrayRef<StringRef> lines, size_t &line,
                                     size_t lineEnd) {
  for (++line; line < lineEnd && !lines[line].empty(); ++line)
    ;
}

// Generate a string attribute from the given paragraph form:
///
/// Header:
///   Element1...
static std::string parseDocStringSection(ArrayRef<StringRef> lines,
                                         size_t &line, size_t lineEnd) {
  // A doc string may end with "Header:". This is diagnosed by the validator,
  // but invalid doc strings may still be emitted as JSON.
  if (line >= lines.size())
    return {};

  const size_t headerIndent = getIndentationLevel(lines[line]);
  size_t cur = line + 1;

  // Skip blank lines after the section header.
  while (cur < lineEnd && lines[cur].trim().empty())
    ++cur;

  // If the next non-blank line is not more indented, section is empty.
  if (cur >= lineEnd || getIndentationLevel(lines[cur]) <= headerIndent) {
    line = cur - 1; // so the main loop can process the next section header
    return {};
  }

  // Otherwise, collect all lines more indented than the header.
  std::string paragraph;
  llvm::raw_string_ostream paragraphOS(paragraph);

  paragraphOS << lines[cur].trim();
  line = cur;
  while (++line < lineEnd) {
    if (lines[line].trim().empty()) {
      paragraphOS << "\n";
      continue;
    }
    if (getIndentationLevel(lines[line]) <= headerIndent)
      break;
    paragraphOS << "\n" << lines[line].trim();
  }
  --line; // so the main loop can process the next section header
  return paragraphOS.str();
}

/// Parse "fake" sections to ensure they don't have unnecessary
/// indentation. Don't trim lines because they'll be merged back into the
/// (unprocessed) descriptionLines. Called after checking for the
/// defined section headings, so we know the current line is either
/// an ad-hoc heading or a regular line of text.
/// TODO: We could eliminate this whole function if we had
/// docstring linting that prevented this class of errors.
static void
maybeParseDocStringAdHocSection(SmallVector<std::string> &pureDescriptionLines,
                                ArrayRef<StringRef> lines, size_t &line,
                                size_t lineEnd) {
  if (line >= lines.size())
    return;

  static const SmallVector<StringLiteral> adHocSections = {
      DocString::kAdHocSectionExample,     DocString::kAdHocSectionExamples,
      DocString::kAdHocSectionNote,        DocString::kAdHocSectionNotes,
      DocString::kAdHocSectionPerformance, DocString::kAdHocSectionSafety,
      DocString::kAdHocSectionWarning,
  };

  bool isAdHoc = false;
  StringRef section = lines[line];
  if (section.consume_back(":")) {
    auto it = std::find(adHocSections.begin(), adHocSections.end(), section);
    isAdHoc = (it != adHocSections.end());
  }
  // Whether or not the current line is an ad-hoc heading, add it to the
  // output.
  pureDescriptionLines.push_back(lines[line].str());
  if (isAdHoc) {
    size_t sectionIndent = getIndentationLevel(lines[line]);

    // Don't set indent based on an empty line.
    while (++line < lineEnd && lines[line].empty())
      pureDescriptionLines.push_back(lines[line].str());

    StringRef currentLine = lines[line];
    size_t contentIndent = getIndentationLevel(currentLine);
    if (contentIndent == sectionIndent) {
      // Content is formatted appropriately, with no extra indent.
      // This could be a new section heading, so back up and return
      // control to the caller.
      --line;
      return;
    } else {
      // Over-indented content, fix it.
      size_t dedent = contentIndent - sectionIndent;
      pureDescriptionLines.push_back(lines[line].drop_front(dedent).str());
      // Merge in additional description lines that have equal or larger
      // indentation.
      while (++line < lineEnd) {
        currentLine = lines[line];
        if (currentLine.empty()) {
          // Don't dedent empty lines
          pureDescriptionLines.push_back(currentLine.str());
        } else if (getIndentationLevel(currentLine) < contentIndent) {
          // End of the indented section. This line could be another
          // section heading, so back up and return control to the caller.
          --line;
          return;
        } else {
          // Merge in additional description lines that have equal or larger
          // indentation.
          pureDescriptionLines.push_back(currentLine.drop_front(dedent).str());
        }
      }
      return;
    }
  }
}

/// Extract a list of direct children decls from a given decl. It omits
/// children whose name start with _, except for special functions that start
/// and end with __. `shouldHideFn` allows for additional filtering of decls to
/// hide.
template <typename ASTDeclType, typename OpType>
static SmallVector<ASTDeclType *, 2>
extractChildDecls(const ASTDecl &decl,
                  function_ref<bool(OpType, StringRef)> shouldHideFn = {}) {
  DenseSet<Operation *> seenOps;
  SmallVector<ASTDeclType *, 2> children;

  for (const auto &[name, decls] : decl.getDeclsInScope()) {
    if (decls.empty() || !decls.front() ||
        !isa_and_nonnull<OpType>(decls.front()->getIfOperation()))
      continue;

    for (ASTDecl *child : decls) {
      OpType childOp = dyn_cast_or_null<OpType>(child->getIfOperation());
      if (!childOp || shouldHideDeclInDocGen(*child, name))
        continue;

      // Skip declarations that were imported from other scopes, unless the
      // import asked for the target's documentation to appear here.
      if ((child->getParentDecl() != &decl && !decl.isDocInlineName(name)) ||
          !seenOps.insert(childOp).second)
        continue;
      // Skip synthetic declarations that don't have accompanying documentation
      // generated with them.
      if (childOp.isSynthetic() && !childOp.getDocStringAttr())
        continue;
      if (shouldHideFn && shouldHideFn(childOp, name))
        continue;
      children.push_back(cast<ASTDeclType>(child));
    }
  }

  llvm::stable_sort(children, [](ASTDeclType *lhs, ASTDeclType *rhs) {
    return compareDeclNames(lhs->getName().value_or(StringRef{}),
                            rhs->getName().value_or(StringRef{}));
  });

  return children;
}

template <typename OpType>
static SmallVector<DefASTDecl *, 2>
extractChildFunctions(const ASTDecl &decl,
                      function_ref<bool(OpType, StringRef)> shouldHideFn = {}) {
  DenseSet<Operation *> seenOps;
  SmallVector<DefASTDecl *, 2> children;
  for (const auto &[name, decls] : decl.getDeclsInScope()) {
    if (decls.empty() || !decls.front() ||
        !isa_and_nonnull<OpType>(decls.front()->getIfOperation()))
      continue;
    for (ASTDecl *child : decls) {
      OpType childOp = dyn_cast_or_null<OpType>(child->getIfOperation());
      if (!childOp || shouldHideDeclInDocGen(*child, name) ||
          (shouldHideFn && shouldHideFn(childOp, name)) ||
          (child->getParentDecl() != &decl && !decl.isDocInlineName(name)) ||
          !seenOps.insert(childOp).second)
        continue;
      if (childOp.isSynthetic() && !childOp.getDocStringAttr())
        continue;
      children.push_back(cast<DefASTDecl>(child));
    }
  }
  llvm::stable_sort(children, [](DefASTDecl *lhs, DefASTDecl *rhs) {
    return compareDeclNames(lhs->getName().value_or(StringRef{}),
                            rhs->getName().value_or(StringRef{}));
  });
  return children;
}

template <typename ASTDeclRange>
static llvm::json::Array toPublicDeclJSONArray(MojoParserContext &ctx,
                                               const ASTDeclRange &decls) {
  llvm::json::Array result;
  for (ASTDecl *decl : decls) {
    auto serialized = serializeDocDecl(*decl, ctx);
    assert(serialized && "expected a serializable public declaration");
    result.push_back(std::move(*serialized));
  }
  return result;
}

/// Dump the markdown header common to all decls that support docstring
/// documentation. Optionally dump the `description` after the `summary`,
/// skipping any sections.
static void dumpMarkdownDocumentationHeader(llvm::raw_ostream &os,
                                            StringRef summary,
                                            StringRef description = {}) {
  if (!summary.empty())
    os << summary << "\n";

  if (!description.empty())
    os << "\n" << description << "\n";
}

/// Dump the markdown description common to all decls that support docstring
/// documentation.
static void dumpMarkdownDocumentationDescription(llvm::raw_ostream &os,
                                                 StringRef description) {
  if (!description.empty())
    os << "\n" << description << "\n";
}

static void dumpMarkdownSectionTitle(llvm::raw_ostream &os, StringRef title) {
  os << "\n#### " << title << ":\n";
}

/// Dump a markdown section with a list of argument or parameter declarations.
static void dumpMarkdownDeclListSection(llvm::raw_ostream &os,
                                        StringRef sectionTitle,
                                        ArrayRef<ASTDecl *> decls) {
  bool isFirst = true;
  for (ASTDecl *decl : decls) {
    std::string documentation = getDeclMarkdownDocString(*decl);
    if (documentation.empty())
      continue;
    StringRef description = StringRef(documentation).drop_back();

    if (isFirst) {
      isFirst = false;
      /// We only show the section title if there's at least one item to show.
      dumpMarkdownSectionTitle(os, sectionTitle);
    } else {
      /// This is a special separator for unbulleted lists.
      os << "\\\n";
    }
    os << kMarkdownIndent << decl->getPogMetadata().getName().getValue() << ": "
       << description << "\n";
  }
}

/// Dump a markdown section with plain text as content and a section title. The
/// section is only dumped if the text is not empty.
static void dumpMarkdownTextSection(llvm::raw_ostream &os,
                                    StringRef sectionTitle, StringRef text) {
  if (!text.empty()) {
    dumpMarkdownSectionTitle(os, sectionTitle);
    os << kMarkdownIndent << text << "\n";
  }
}

static std::string getVariableType(ASTDecl &decl) {
  ASTType type = decl.getToolingType().getReferenceElementType();
  return type.getAsString(ASTTypePrinterContext(&decl.getShared()));
}

static std::string getStructFieldType(StructFieldASTDecl &decl) {
  std::string type;
  llvm::raw_string_ostream os(type);
  ASTType(decl.getOp().getType()).print(os, /*ctx=*/{&decl.getShared()});
  return type;
}

static std::string getAliasMarkdownDocString(AliasASTDecl &decl);

//===----------------------------------------------------------------------===//
// Public declaration rendering
//===----------------------------------------------------------------------===//

std::string M::KGEN::LIT::getDeclDeclarationSnippet(ASTDecl &decl) {
  if (auto *structDecl = dyn_cast<StructASTDecl>(&decl))
    return getStructDeclarationSnippet(*structDecl);
  return getDeclarationSnippet(decl);
}

std::string M::KGEN::LIT::getDeclFullMarkdownString(ASTDecl &decl) {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  // A code snippet used when rendering the documentation string.
  const char *docStringSnippet = R"(
---

###
{0}
)";

  // A code snippet used when rendering the declaration snippet.
  const char *declarationSnippet = R"(
---

###
```mojo
{0}
```)";

  StringRef kindName = decl.getToolingKindName();
  if (kindName.empty())
    return {};
  StringRef name = decl.getName().value_or(StringRef());
  if (decl.isUserParameterDecl())
    name = decl.getPogMetadata().getName();
  os << formatv("### {0} `{1}`\n", kindName, name);
  if (auto docString = getDeclMarkdownDocString(decl); !docString.empty())
    os << llvm::formatv(docStringSnippet, docString);

  os << llvm::formatv(declarationSnippet, getDeclDeclarationSnippet(decl));
  return buff;
}

static llvm::json::Object serializeParameterDecl(ASTDecl &decl,
                                                 MojoParserContext &ctx) {
  PogMetadataAttr pog = decl.getPogMetadata();
  std::string type = getParameterTypeString(decl);
  std::string description = getPogMarkdownDocString(decl);
  if (!description.empty())
    description.pop_back();
  llvm::json::Object typeMetadata =
      extractLibraryInfo(type, &decl, &decl.getShared());
  SmallVector<std::string> traitNames = parseCompoundTraitType(type);

  llvm::json::Object object{
      {"kind", "parameter"},
      {"name", prependVariadicIdentifiers(pog.getName().getValue(),
                                          decl.getPogVariadicKind())
                   .str()},
      {"passingKind", stringifyPassingKind(pog.getPassingKind())},
      {"description", std::move(description)},
  };

  if (traitNames.size() == 1) {
    for (auto &pair : typeMetadata) {
      object[pair.first] = std::move(pair.second);
    }
  } else {
    object["type"] = type;
    llvm::json::Array traitsArray;
    for (const std::string &traitName : traitNames)
      traitsArray.push_back(
          extractLibraryInfo(traitName, &decl, &decl.getShared()));
    object["traits"] = std::move(traitsArray);
  }

  if (TypedAttr defaultValue = getReboundPogDefault(decl))
    object["default"] =
        getDefaultValueString(defaultValue, ctx.getSharedState());
  return object;
}

static llvm::json::Object serializeArgumentDecl(ASTDecl &decl,
                                                MojoParserContext &ctx) {
  PogMetadataAttr pog = decl.getPogMetadata();
  std::string type = getArgumentTypeString(decl);
  std::string description = getPogMarkdownDocString(decl);
  if (!description.empty())
    description.pop_back();
  llvm::json::Object typeMetadata =
      extractLibraryInfo(type, &decl, &decl.getShared());
  SmallVector<std::string> traitNames = parseCompoundTraitType(type);

  llvm::json::Object object{
      {"description", std::move(description)},
      {"convention", getArgumentConventionString(decl)},
      {"kind", "argument"},
      {"name", prependVariadicIdentifiers(pog.getName().getValue(),
                                          decl.getPogVariadicKind())
                   .str()},
      {"passingKind", stringifyPassingKind(pog.getPassingKind())},
  };

  if (traitNames.size() == 1) {
    for (auto &pair : typeMetadata) {
      object[pair.first] = std::move(pair.second);
    }
  } else {
    object["type"] = type;
    llvm::json::Array traitsArray;
    for (const std::string &traitName : traitNames)
      traitsArray.push_back(
          extractLibraryInfo(traitName, &decl, &decl.getShared()));
    object["traits"] = std::move(traitsArray);
  }

  if (TypedAttr defaultValue = getReboundPogDefault(decl))
    object["default"] =
        getDefaultValueString(defaultValue, ctx.getSharedState());
  return object;
}

//===----------------------------------------------------------------------===//
// Alias documentation
//===----------------------------------------------------------------------===//

static ParameterEvaluator getAliasParameterEvaluator(AliasASTDecl &decl) {
  auto generatorType = dyn_cast<GeneratorType>(decl.getOp().getType());
  if (!generatorType)
    return {};

  ParameterEvaluator evaluator;
  for (auto [index, type] :
       llvm::enumerate(generatorType.getInputParamTypes())) {
    Type reboundType = evaluator.getReboundType(type);
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(
        generatorType.getParamListAttrs().getName(index), reboundType));
  }
  return evaluator;
}

static std::string getAliasMarkdownDocString(AliasASTDecl &decl) {
  std::string summary;
  std::string description;
  if (auto docString = decl.getParsedDocString()) {
    summary = docString->getSummary();
    SmallVector<std::string> descriptionLines;
    for (size_t line = 0, end = docString->getDescription().size(); line < end;
         ++line) {
      StringRef section = docString->getDescription()[line];
      if (section == (Twine(DocString::kSectionParameters) + ":").str()) {
        skipDocumentationSection(docString->getDescription(), line, end);
      } else {
        maybeParseDocStringAdHocSection(descriptionLines,
                                        docString->getDescription(), line, end);
      }
    }
    SmallVector<StringRef> descriptionRefs;
    for (const std::string &line : descriptionLines)
      descriptionRefs.push_back(line);
    description = DocString::formatDescription(descriptionRefs);
  }

  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters,
                              decl.getUserParamDecls());
  return markdown;
}

static llvm::json::Object serializeAliasDecl(AliasASTDecl &decl,
                                             MojoParserContext &ctx) {
  ParameterEvaluator evaluator = getAliasParameterEvaluator(decl);
  std::string type;
  if (auto generatorType = dyn_cast<GeneratorType>(decl.getOp().getType()))
    type = generateTypeString(decl.getShared(),
                              evaluator.getReboundType(generatorType.getBody()),
                              VariadicKind::None);

  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);
  signatureOS << "comptime ";
  printAliasSignature(decl, signatureOS);

  std::string description;
  std::string summary;
  if (auto docString = decl.getParsedDocString()) {
    summary = docString->getSummary();
    SmallVector<std::string> descriptionLines;
    for (size_t line = 0, end = docString->getDescription().size(); line < end;
         ++line) {
      StringRef section = docString->getDescription()[line];
      if (section == (Twine(DocString::kSectionParameters) + ":").str()) {
        skipDocumentationSection(docString->getDescription(), line, end);
      } else {
        maybeParseDocStringAdHocSection(descriptionLines,
                                        docString->getDescription(), line, end);
      }
    }
    SmallVector<StringRef> descriptionRefs;
    for (const std::string &line : descriptionLines)
      descriptionRefs.push_back(line);
    description = DocString::formatDescription(descriptionRefs);
  }

  llvm::json::Array parameters;
  for (ASTDecl *parameter : decl.getUserParamDecls())
    parameters.push_back(serializeParameterDecl(*parameter, ctx));

  StableInfo stableInfo = computeStableInfo(decl.getOp());
  llvm::json::Object object{
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"description", description},
      {"kind", "alias"},
      {"name", decl.getName().value_or(StringRef())},
      {"isStable", stableInfo.isStable},
      {"isStabilityTracked", stableInfo.isStabilityTracked},
      {"sinceVersion", stableInfo.sinceVersion},
      {"summary", summary},
      {"parameters", std::move(parameters)},
      {"signature", signature},
      {"value", decl.getToolingValue()},
  };

  if (!type.empty())
    object["type"] = type;
  std::string modulePath = extractModulePathFromDecl(&decl);
  if (!modulePath.empty())
    object["path"] =
        generateDocPath(modulePath, decl.getName().value_or(StringRef()),
                        decl.getShared().getDocsBasePath(), /*isAlias=*/true);
  return object;
}

//===----------------------------------------------------------------------===//
// Function documentation
//===----------------------------------------------------------------------===//

struct FunctionDocumentation {
  std::string constraints;
  std::string description;
  std::string raisesDoc;
  std::string returnsDoc;
  std::string summary;
};

static void augmentFunctionDocumentation(FunctionDocumentation &documentation,
                                         ArrayRef<StringRef> desc,
                                         bool hasReturnType, bool raises) {
  // Process the lines of the description, looking for markers.
  SmallVector<std::string> pureDescriptionLines;
  std::string argsSectionHeader = (Twine(DocString::kSectionArgs) + ":").str();
  std::string paramsSectionHeader =
      (Twine(DocString::kSectionParameters) + ":").str();
  std::string returnsSectionHeader =
      (Twine(DocString::kSectionReturns) + ":").str();
  std::string constraintsSectionHeader =
      (Twine(DocString::kSectionConstraints) + ":").str();
  std::string raisesSectionHeader =
      (Twine(DocString::kSectionRaises) + ":").str();

  for (size_t line = 0, lineEnd = desc.size(); line < lineEnd; ++line) {
    if (desc[line] == argsSectionHeader) {
      skipDocumentationSection(desc, line, lineEnd);
    } else if (desc[line] == paramsSectionHeader) {
      skipDocumentationSection(desc, line, lineEnd);
    } else if (desc[line] == returnsSectionHeader) {
      if (hasReturnType)
        documentation.returnsDoc = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] == constraintsSectionHeader) {
      documentation.constraints = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] == raisesSectionHeader) {
      if (raises)
        documentation.raisesDoc = parseDocStringSection(desc, line, lineEnd);
    } else {
      // If this line is an ad-hoc section heading, process it to ensure
      // that it doesn't have any unexpected indentation. Otherwise, just
      // add the line to the description.
      maybeParseDocStringAdHocSection(pureDescriptionLines, desc, line,
                                      lineEnd);
    }
  }
  SmallVector<StringRef> pureDescriptionLinesRef;
  for (const auto &descLine : pureDescriptionLines) {
    pureDescriptionLinesRef.push_back(StringRef(descLine));
  }
  documentation.description =
      DocString::formatDescription(pureDescriptionLinesRef);
}

/// Return the decl for the trait default that `decl` inherits, or `decl` itself
/// if it is not an inherited default.
static ASTDecl *findDefaultImplDecl(SharedState &shared, ASTDecl *decl) {
  auto *fnDecl = dyn_cast_or_null<DefASTDecl>(decl);
  FnOp fnOp = fnDecl ? fnDecl->getOp() : nullptr;
  if (!fnOp)
    return decl;

  std::optional<SymbolRefAttr> defaultFn = fnOp.getDefaultFnRef();
  if (!defaultFn)
    return decl;

  // Trait defaults reached through a bytecode package are resolved on demand,
  // so this can be the reference that first loads one.
  ASTDecl *defaultImpl =
      shared.resolveAndGetFuncDecl(*defaultFn, decl->getLoc());
  return defaultImpl ? defaultImpl : decl;
}

static std::string getPogMarkdownDocString(ASTDecl &decl) {
  ASTDecl *owner = decl.getParentDecl();
  if (!owner)
    return {};
  owner = findDefaultImplDecl(decl.getShared(), owner);
  std::optional<DocString> docString = owner->getParsedDocString();
  if (!docString)
    return {};

  StringRef sectionName = decl.getArgNumber() >= 0
                              ? DocString::kSectionArgs
                              : DocString::kSectionParameters;
  std::string sectionHeader = (Twine(sectionName) + ":").str();
  StringRef name = decl.getPogMetadata().getName();
  ArrayRef<StringRef> lines = docString->getDescription();
  for (size_t line = 0; line < lines.size(); ++line) {
    if (lines[line] != sectionHeader)
      continue;
    for (++line; line < lines.size() && !lines[line].empty();) {
      auto [documentedName, description] = lines[line].split(':');
      documentedName = documentedName.trim();
      description = description.trim();
      size_t indentation = getIndentationLevel(lines[line]);
      std::string fullDescription(description);
      while (++line < lines.size() &&
             getIndentationLevel(lines[line]) > indentation)
        fullDescription +=
            (Twine("\n") + lines[line].drop_front(indentation).rtrim()).str();
      if (documentedName == name)
        return fullDescription.empty() ? std::string() : fullDescription + "\n";
    }
    return {};
  }
  return {};
}

static std::optional<std::string> getFunctionReturnType(DefASTDecl &decl) {
  FnOp funcOp = decl.getOp();
  FnTypeGeneratorType signature = funcOp.getFuncTypeGenerator();
  DeclResolver::DiagnosticDeclContextChanger declScopeChanger(&decl);
  auto &shared = decl.getShared();
  Type userResultType = funcOp.getUserResultType();

  // If this is a method, grab the expected "Self" type.
  std::optional<ASTType> selfType;
  if (isa<StructASTDecl>(decl.getParentDecl()))
    selfType = cast<StructASTDecl>(decl.getParentDecl())->getSelfType();

  ParameterEvaluator evaluator;
  for (auto [index, type] : llvm::enumerate(signature.getInputParamTypes())) {
    Type reboundType = evaluator.getReboundType(type);
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(
        signature.getParamListAttrs().getName(index), reboundType));
  }

  // Grab the result type, if it's non-none.
  ASTType resultType = signature.getUserResultType();
  assert(resultType && "didn't find a result type?");

  if (resultType.isNoneType())
    return std::nullopt;

  std::string str;
  std::optional<ArgConvention> convention;
  // If this is a ref result add the "ref[life, addrspace] "
  // prefix to the specifier.
  if (signature.isRefResult()) {
    convention = ArgConvention::Ref;
    str = "ref" + getRefPrefixAsString(shared, cast<RefType>(resultType),
                                       signature, /*isRefResult*/ true);
  }
  Type reboundUserResultType = evaluator.getReboundType(userResultType);
  str += generateTypeString(shared, reboundUserResultType, VariadicKind::None,
                            selfType, convention);
  return str;
}

static FunctionDocumentation getFunctionDocumentation(DefASTDecl &decl) {
  FunctionDocumentation documentation;
  FnTypeGeneratorType signature = decl.getSignature();
  // An inherited default is a copy that cannot carry a doc string of its own,
  // so its documentation comes from the trait method that declares it.
  if (auto docStr =
          findDefaultImplDecl(decl.getShared(), &decl)->getParsedDocString()) {
    documentation.summary = docStr->getSummary();
    augmentFunctionDocumentation(
        documentation, docStr->getDescription(),
        !ASTType(signature.getUserResultType()).isNoneType(),
        signature.isThrows());
  }
  return documentation;
}

static std::string getFunctionMarkdownDocString(DefASTDecl &decl) {
  FunctionDocumentation documentation = getFunctionDocumentation(decl);
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, documentation.summary);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters,
                              decl.getUserParamDecls());
  dumpMarkdownDeclListSection(os, DocString::kSectionArgs,
                              decl.getUserArgDecls());
  dumpMarkdownTextSection(os, DocString::kSectionReturns,
                          documentation.returnsDoc);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints,
                          documentation.constraints);
  dumpMarkdownTextSection(os, DocString::kSectionRaises,
                          documentation.raisesDoc);
  dumpMarkdownDocumentationDescription(os, documentation.description);
  return markdown;
}

static llvm::json::Object serializeFunctionDecl(DefASTDecl &decl,
                                                MojoParserContext &ctx) {
  FunctionDocumentation documentation = getFunctionDocumentation(decl);
  FnOp funcOp = decl.getOp();
  FnTypeGeneratorType signature = decl.getSignature();
  StableInfo stableInfo = computeStableInfo(funcOp);
  llvm::json::Array arguments;
  for (ASTDecl *argument : decl.getUserArgDecls())
    arguments.push_back(serializeArgumentDecl(*argument, ctx));
  llvm::json::Array parameters;
  for (ASTDecl *parameter : decl.getUserParamDecls())
    parameters.push_back(serializeParameterDecl(*parameter, ctx));
  llvm::json::Object result{
      {"args", std::move(arguments)},
      {"async", signature.isAsync()},
      {"constraints", documentation.constraints},
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef{})},
      {"description", documentation.description},
      {"hasDefaultImplementation", funcOp.isDefaultedTraitFn()},
      {"isStatic", funcOp.getIsStatic()},
      {"isImplicitConversion", funcOp.isImplicitConversion()},
      {"kind", "function"},
      {"name", decl.getName().value_or(StringRef{}).str()},
      {"parameters", std::move(parameters)},
      {"raises", signature.isThrows()},
      {"raisesDoc", documentation.raisesDoc},
      {"signature", getFunctionSignature(decl)},
      {"isStable", stableInfo.isStable},
      {"isStabilityTracked", stableInfo.isStabilityTracked},
      {"sinceVersion", stableInfo.sinceVersion},
      {"summary", documentation.summary},
  };
  if (std::optional<std::string> returnType = getFunctionReturnType(decl)) {
    llvm::json::Object returnObject =
        extractLibraryInfo(*returnType, &decl, &ctx.getSharedState());
    if (!documentation.returnsDoc.empty())
      returnObject["doc"] = documentation.returnsDoc;
    result["returns"] = std::move(returnObject);
  }
  return result;
}

/// Serialize name-sorted functions as one entry per overload set.
static llvm::json::Array
serializeFunctionOverloadSets(ArrayRef<DefASTDecl *> functions,
                              MojoParserContext &ctx) {
  llvm::json::Array result;
  for (size_t first = 0; first < functions.size();) {
    StringRef name = functions[first]->getName().value_or(StringRef{});
    llvm::json::Array overloads;
    size_t next = first;
    do {
      overloads.push_back(serializeFunctionDecl(*functions[next], ctx));
      ++next;
    } while (next < functions.size() &&
             functions[next]->getName().value_or(StringRef{}) == name);

    result.push_back(llvm::json::Object{{"kind", "function"},
                                        {"name", name},
                                        {"overloads", std::move(overloads)}});
    first = next;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Trait rendering
//===----------------------------------------------------------------------===//

/// Render the `where` condition of every conditional conformance in
/// `canonicalTrait` into `conditions`.
///
/// The canonical trait stores conditions in an array parallel to its symbols,
/// left empty when every conformance is unconditional. Slots whose condition
/// is trivially true are unconditional and contribute no entry; slots whose
/// condition is trivially false are already dropped from the symbol list by
/// `TraitType`'s canonicalization, so a `where False` conformance is absent
/// from the docs entirely.
///
/// Conformance conditions are stored de-short-circuited, which inverts the
/// sugar for `and`/`or`; `renderConstraintProposition` undoes that.
static void collectConformanceConditions(TraitType canonicalTrait,
                                         ParameterEvaluator &evaluator,
                                         SharedState &shared,
                                         ConformanceConditionMap &conditions) {
  ArrayRef<TraitSymbolAttr> symbols = canonicalTrait.getSymbols();
  ArrayRef<ConstraintAttr> constraints = canonicalTrait.getConstraints();
  if (constraints.empty())
    return;

  for (auto [symbol, constraint] : llvm::zip_equal(symbols, constraints)) {
    if (isTriviallyTrueConstraint(constraint))
      continue;

    conditions.try_emplace(
        symbol, renderConstraintProposition(constraint.getProposition(),
                                            &evaluator, shared));
  }
}

/// Look up the rendered `where` condition guarding conformance to `symbol`,
/// or an empty string when the conformance is unconditional. `conditions` is
/// null for decls that cannot conform conditionally (traits, extensions).
static StringRef
lookupConformanceCondition(const ConformanceConditionMap *conditions,
                           TraitSymbolAttr symbol) {
  if (!conditions)
    return {};
  auto it = conditions->find(symbol);
  return it == conditions->end() ? StringRef() : StringRef(it->second);
}

/// Collect the names of the various parent decls of a decl given its set of
/// canonical traits. Conditionally-conformed traits are rendered with their
/// `where` clause, e.g. `Writable (where conforms_to(T, Writable))`. Callers
/// join the entries with commas, so the clause is parenthesized to keep it
/// from reading as a continuation of the list.
/// TODO: Whenever we support inherited classes/structs, collect those as well.
static void collectParentTraits(ASTDecl *self,
                                SmallVectorImpl<std::string> &parentTraits,
                                TraitType canonicalTrait,
                                const ConformanceConditionMap *conditions) {
  DenseSet<TraitSymbolAttr> seenDecls;
  for (TraitSymbolAttr symbol : canonicalTrait.getSymbols()) {
    if (!seenDecls.insert(symbol).second)
      continue;
    ASTDecl *decl = self->getShared().resolveAndGetTypeDecl(symbol.getSymbol(),
                                                            self->getLoc());
    if (!decl || decl == self)
      continue;
    std::optional<StringRef> name = decl->getName();
    if (!name)
      continue;
    if (!isa<TraitASTDecl>(decl))
      continue;
    std::string entry = name->str();
    if (StringRef condition = lookupConformanceCondition(conditions, symbol);
        !condition.empty())
      entry += (" (where " + condition + ")").str();
    parentTraits.push_back(std::move(entry));
  };
  llvm::sort(parentTraits);
}

/// Collect parent traits with metadata for JSON serialization, avoiding
/// self-references
static llvm::json::Array
collectParentTraitsWithMetadata(ASTDecl *self, TraitType canonicalTrait,
                                const ConformanceConditionMap *conditions) {
  llvm::json::Array result;
  DenseSet<TraitSymbolAttr> seenDecls;

  struct TraitEntry {
    StringRef name;
    llvm::json::Object metadata;
    StringRef condition;
  };
  SmallVector<TraitEntry> traitData;

  for (TraitSymbolAttr symbol : canonicalTrait.getSymbols()) {
    if (!seenDecls.insert(symbol).second)
      continue;

    StringRef condition = lookupConformanceCondition(conditions, symbol);

    // Try to resolve the trait through AST
    ASTDecl *decl = self->getShared().resolveAndGetTypeDecl(symbol.getSymbol(),
                                                            self->getLoc());
    if (decl && decl != self) {
      std::optional<StringRef> name = decl->getName();
      if (name && isa<TraitASTDecl>(decl)) {
        traitData.push_back(
            {*name, extractLibraryInfo(*name, self, &self->getShared()),
             condition});
        continue;
      }
    }

    // Fallback: extract trait name from symbol
    SymbolRefAttr symbolRef = symbol.getSymbol();
    StringRef traitName =
        symbolRef.getNestedReferences().empty()
            ? symbolRef.getRootReference().getValue()
            : symbolRef.getNestedReferences().back().getAttr().getValue();

    if (!traitName.empty()) {
      // Check if this trait name matches the current trait (avoid
      // self-reference)
      std::optional<StringRef> selfName = self->getName();
      if (selfName && traitName == *selfName) {
        continue; // Skip self-reference
      }

      traitData.push_back(
          {traitName, extractLibraryInfo(traitName, self, &self->getShared()),
           condition});
    }
  }

  // Sort by trait name for consistency
  llvm::sort(traitData,
             [](const auto &a, const auto &b) { return a.name < b.name; });

  // Create JSON objects with flattened metadata structure
  for (const TraitEntry &entry : traitData) {
    llvm::json::Object traitObj;
    traitObj["name"] = entry.name.str();

    if (std::optional<StringRef> path = entry.metadata.getString("path"))
      traitObj["path"] = path->str();

    if (!entry.condition.empty()) {
      traitObj["condition"] = entry.condition.str();
    }

    result.push_back(std::move(traitObj));
  }

  return result;
}

static llvm::json::Object serializeTraitDecl(TraitASTDecl &decl,
                                             MojoParserContext &ctx) {
  // Ignore some inherited functions.
  auto shouldHideFn = [](FnOp decl, StringRef name) {
    return decl.getInheritedFrom() &&
           decl.getSpecialFunctionKind() == SpecialFunctionKind::kDeinit;
  };

  auto aliases = extractChildDecls<AliasASTDecl, AliasDeclOp>(decl);
  auto functions = extractChildFunctions<FnOp>(decl, shouldHideFn);

  // Collect parent traits with type metadata. A trait's own conformance list
  // cannot carry a `where` clause, so there are no conditions to report.
  llvm::json::Array parentTraitsWithMetadata =
      collectParentTraitsWithMetadata(&decl, decl.getOp().getCanonicalTrait(),
                                      /*conditions=*/nullptr);
  StableInfo stableInfo = computeStableInfo(decl.getOp());

  return llvm::json::Object{
      {"aliases", toPublicDeclJSONArray(ctx, aliases)},
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"description", decl.getDocumentationDescription()},
      {"fields", llvm::json::Array()},
      {"functions", serializeFunctionOverloadSets(functions, ctx)},
      {"kind", "trait"},
      {"name", decl.getName().value_or(StringRef())},
      {"parentTraits", std::move(parentTraitsWithMetadata)},
      {"isStable", stableInfo.isStable},
      {"isStabilityTracked", stableInfo.isStabilityTracked},
      {"sinceVersion", stableInfo.sinceVersion},
      {"summary", decl.getDocumentationSummary()},
  };
}

//===----------------------------------------------------------------------===//
// Struct rendering
//===----------------------------------------------------------------------===//

struct StructDocumentation {
  std::string constraints;
  std::string description;
};

static void augmentStructDocumentation(StructDocumentation &documentation,
                                       ArrayRef<StringRef> desc) {
  SmallVector<std::string> pureDescriptionLines;
  std::string paramsSectionHeader =
      (Twine(DocString::kSectionParameters) + ":").str();
  std::string constraintsSectionHeader =
      (Twine(DocString::kSectionConstraints) + ":").str();
  for (size_t line = 0, lineEnd = desc.size(); line < lineEnd; ++line) {
    if (desc[line] == paramsSectionHeader)
      skipDocumentationSection(desc, line, lineEnd);
    else if (desc[line] == constraintsSectionHeader)
      documentation.constraints = parseDocStringSection(desc, line, lineEnd);
    else
      maybeParseDocStringAdHocSection(pureDescriptionLines, desc, line,
                                      lineEnd);
  }

  SmallVector<StringRef> pureDescriptionLinesRef;
  for (const auto &descLine : pureDescriptionLines) {
    pureDescriptionLinesRef.push_back(StringRef(descLine));
  }
  documentation.description =
      DocString::formatDescription(pureDescriptionLinesRef);
}

static StructDocumentation getStructDocumentation(StructASTDecl &decl) {
  StructDocumentation documentation;
  if (auto docStr = decl.getParsedDocString())
    augmentStructDocumentation(documentation, docStr->getDescription());
  return documentation;
}

static ConformanceConditionMap
getStructConformanceConditions(StructASTDecl &decl) {
  StructDeclOp structOp = decl.getOp();
  TypeSignatureType signature = structOp.getSignature();
  ParameterEvaluator evaluator;
  for (auto [index, type] : llvm::enumerate(signature.getInputParamTypes())) {
    Type reboundType = evaluator.getReboundType(type);
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(
        signature.getParamListAttrs().getName(index), reboundType));
  }
  ConformanceConditionMap conditions;
  collectConformanceConditions(structOp.getCanonicalTrait(), evaluator,
                               decl.getShared(), conditions);
  return conditions;
}

std::string M::KGEN::LIT::getStructDeclarationSnippet(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << getStructSignature(decl, parameterOffsets);

  SmallVector<std::string> parentTraits;
  ConformanceConditionMap conformanceConditions =
      getStructConformanceConditions(decl);
  collectParentTraits(&decl, parentTraits, decl.getOp().getCanonicalTrait(),
                      &conformanceConditions);
  if (!parentTraits.empty()) {
    os << "\n# Traits: ";
    llvm::interleaveComma(parentTraits, os,
                          [&](StringRef token) { os << token; });
  }

  return snippet;
}

static std::string getStructMarkdownDocString(StructASTDecl &decl) {
  StructDocumentation documentation = getStructDocumentation(decl);
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, decl.getDocumentationSummary(),
                                  documentation.description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters,
                              decl.getUserParamDecls());
  dumpMarkdownTextSection(os, DocString::kSectionConstraints,
                          documentation.constraints);

  return markdown;
}

static StringRef toString(TypeConvention convention) {
  switch (convention) {
  case TypeConvention::MemoryOnly:
    return "memory_only";
  case TypeConvention::RegisterPassable:
    return "register_passable";
  case TypeConvention::RegisterPassableTrivial:
    return "register_passable_trivial";
  case TypeConvention::Unspecified:
    return "";
  }
}

static llvm::json::Object serializeStructDecl(StructASTDecl &decl,
                                              MojoParserContext &ctx) {
  StructDocumentation documentation = getStructDocumentation(decl);
  StructDeclOp structOp = decl.getOp();
  StableInfo stableInfo = computeStableInfo(structOp);
  ConformanceConditionMap conformanceConditions =
      getStructConformanceConditions(decl);
  auto aliases = extractChildDecls<AliasASTDecl, AliasDeclOp>(decl);
  auto fields = extractChildDecls<StructFieldASTDecl, StructFieldOp>(decl);
  auto functions = extractChildFunctions<FnOp>(decl);

  llvm::json::Array parentTraitsWithMetadata = collectParentTraitsWithMetadata(
      &decl, structOp.getCanonicalTrait(), &conformanceConditions);
  llvm::json::Array parameterInfos;
  for (ASTDecl *parameter : decl.getUserParamDecls())
    parameterInfos.push_back(serializeParameterDecl(*parameter, ctx));

  return llvm::json::Object{
      {"aliases", toPublicDeclJSONArray(ctx, aliases)},
      {"constraints", documentation.constraints},
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"description", documentation.description},
      {"fields", toPublicDeclJSONArray(ctx, fields)},
      {"functions", serializeFunctionOverloadSets(functions, ctx)},
      {"kind", "struct"},
      {"name", decl.getName().value_or(StringRef())},
      {"parameters", std::move(parameterInfos)},
      {"parentTraits", std::move(parentTraitsWithMetadata)},
      {"signature", getStructSignature(decl)},
      {"isStable", stableInfo.isStable},
      {"isStabilityTracked", stableInfo.isStabilityTracked},
      {"sinceVersion", stableInfo.sinceVersion},
      {"summary", decl.getDocumentationSummary()},
      {"convention", toString(structOp.getConvention())},
  };
}

//===----------------------------------------------------------------------===//
// Module and package serialization
//===----------------------------------------------------------------------===//

static llvm::json::Object serializeModuleDecl(ASTDecl &decl,
                                              MojoParserContext &ctx) {
  auto aliases = extractChildDecls<AliasASTDecl, AliasDeclOp>(decl);
  auto structs = extractChildDecls<StructASTDecl, StructDeclOp>(decl);
  auto traits = extractChildDecls<TraitASTDecl, TraitDeclOp>(decl);
  auto functions = extractChildFunctions<FnOp>(decl);

  return llvm::json::Object{
      {"aliases", toPublicDeclJSONArray(ctx, aliases)},
      {"description", decl.getDocumentationDescription()},
      {"functions", serializeFunctionOverloadSets(functions, ctx)},
      {"kind", "module"},
      {"name", decl.getName().value_or(StringRef{})},
      {"structs", toPublicDeclJSONArray(ctx, structs)},
      {"traits", toPublicDeclJSONArray(ctx, traits)},
      {"summary", decl.getDocumentationSummary()}};
}

static llvm::json::Object serializeVariableDecl(ASTDecl &decl,
                                                MojoParserContext &ctx) {
  std::string type = getVariableType(decl);
  llvm::json::Object result{
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"kind", "variable"},
      {"name", decl.getName().value_or(StringRef())},
  };

  SmallVector<std::string> traitNames = parseCompoundTraitType(type);
  if (traitNames.size() == 1) {
    llvm::json::Object metadata =
        extractLibraryInfo(type, nullptr, &ctx.getSharedState());
    for (auto &pair : metadata)
      result[pair.first] = std::move(pair.second);
  } else {
    result["type"] = type;
    llvm::json::Array traits;
    for (const std::string &traitName : traitNames)
      traits.push_back(
          extractLibraryInfo(traitName, nullptr, &ctx.getSharedState()));
    result["traits"] = std::move(traits);
  }

  return result;
}

static llvm::json::Object serializeStructFieldDecl(StructFieldASTDecl &decl) {
  return llvm::json::Object{
      {"description", decl.getDocumentationDescription()},
      {"kind", "field"},
      {"name", decl.getName().value_or(StringRef())},
      {"summary", decl.getDocumentationSummary()},
      {"type", getStructFieldType(decl)},
  };
}

static llvm::json::Array serializePublicDecls(MojoParserContext &ctx,
                                              ArrayRef<ASTDecl *> decls) {
  llvm::json::Array result;
  for (ASTDecl *decl : decls)
    if (auto *function = dyn_cast<DefASTDecl>(decl))
      result.push_back(serializeFunctionDecl(*function, ctx));
    else if (auto serialized = serializeDocDecl(*decl, ctx))
      result.push_back(std::move(*serialized));
  return result;
}

static llvm::json::Object serializePackageDecl(PackageASTDecl &packageDecl,
                                               MojoParserContext &ctx) {
  // A package's child modules and sub-packages are unlisted: they live in the
  // module-state cache (ModuleState::nestedModules) and the package IR, not in
  // declsInScope (which extractChildDecls iterates). Enumerate them from the
  // cache so the package's contents are documented. The package body has been
  // resolved by resolveAllReferencedFrom, so nestedModules is populated.
  // `__init__` is kept (it is the package's documented entry module); private
  // modules are hidden, matching extractChildDecls' filtering.
  SmallVector<ASTDecl *> modules;
  SmallVector<ASTDecl *> packages;
  for (ModuleASTDecl *child :
       ctx.getSharedState().getNestedModuleDecls(packageDecl.getOp())) {
    StringRef name = child->getName().value_or(StringRef());
    if (shouldHideDeclInDocGen(*child, name) ||
        child->getParentDecl() != &packageDecl)
      continue;
    if (isa<FileModuleASTDecl>(child))
      modules.push_back(child);
    else if (isa<PackageASTDecl>(child))
      packages.push_back(child);
  }

  auto byName = [](ASTDecl *lhs, ASTDecl *rhs) {
    return compareDeclNames(lhs->getName().value_or(StringRef{}),
                            rhs->getName().value_or(StringRef{}));
  };
  llvm::stable_sort(modules, byName);
  llvm::stable_sort(packages, byName);

  return llvm::json::Object{
      {"description", packageDecl.getDocumentationDescription()},
      {"kind", "package"},
      {"name", packageDecl.getName().value_or(StringRef{})},
      {"summary", packageDecl.getDocumentationSummary()},
      {"modules", serializePublicDecls(ctx, modules)},
      {"packages", serializePublicDecls(ctx, packages)},
  };
}

std::optional<llvm::json::Object>
M::KGEN::LIT::serializeDocDecl(ASTDecl &decl, MojoParserContext &ctx) {
  if (auto *variable = dyn_cast<VarASTDecl>(&decl)) {
    if (variable->isArgument()) {
      if (decl.isUserArgument())
        return serializeArgumentDecl(decl, ctx);
      return llvm::json::Object();
    }
    return serializeVariableDecl(decl, ctx);
  }

  if (Operation *declOp = decl.getIfOperation()) {
    if (isa<AliasDeclOp>(declOp))
      return serializeAliasDecl(cast<AliasASTDecl>(decl), ctx);
    if (isa<FileModuleOp, ImportOp>(declOp))
      return serializeModuleDecl(decl, ctx);
    if (isa<StructDeclOp>(declOp))
      return serializeStructDecl(cast<StructASTDecl>(decl), ctx);
    if (isa<StructFieldOp>(declOp))
      return serializeStructFieldDecl(cast<StructFieldASTDecl>(decl));
    if (isa<PackageOp>(declOp))
      return serializePackageDecl(cast<PackageASTDecl>(decl), ctx);
    if (isa<TraitDeclOp>(declOp))
      return serializeTraitDecl(cast<TraitASTDecl>(decl), ctx);
  }

  if (decl.isArgumentDecl()) {
    if (decl.isUserArgument())
      return serializeArgumentDecl(decl, ctx);
    return llvm::json::Object();
  }
  if (decl.isUserParameterDecl())
    return serializeParameterDecl(decl, ctx);
  return std::nullopt;
}

std::string M::KGEN::LIT::getDeclMarkdownDocString(ASTDecl &decl) {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  if (isa<DefASTDecl>(decl))
    return getFunctionMarkdownDocString(cast<DefASTDecl>(decl));
  if (isa<AliasASTDecl>(decl))
    return getAliasMarkdownDocString(cast<AliasASTDecl>(decl));
  if (auto *variable = dyn_cast<VarASTDecl>(&decl)) {
    if (!variable->isArgument())
      return {};
    return getPogMarkdownDocString(decl);
  }
  if (decl.isArgumentDecl())
    return getPogMarkdownDocString(decl);
  if (decl.isUserParameterDecl())
    return getPogMarkdownDocString(decl);
  if (isa<StructASTDecl>(decl))
    return getStructMarkdownDocString(cast<StructASTDecl>(decl));
  if (isa<StructFieldASTDecl, FileModuleASTDecl, ImportASTDecl, PackageASTDecl,
          TraitASTDecl>(decl)) {
    dumpMarkdownDocumentationHeader(os, decl.getDocumentationSummary(),
                                    decl.getDocumentationDescription());
    return markdown;
  }
  return {};
}
