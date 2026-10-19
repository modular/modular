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

#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/MojoParser/SignatureModel.h"
#include "Mojo/MojoParser/StabilityMarkers.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/TypeExtractionUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <optional>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

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

/// Parse the given docstring lines and augment the provided decls with the
/// appropriate documentation using the description.
template <typename PublicDeclT, unsigned N>
static void augmentDeclsWithDocumentation(ArrayRef<StringRef> lines,
                                          size_t &line, size_t lineEnd,
                                          SmallVector<PublicDeclT, N> &decls) {
  std::string fullArgDesc;
  llvm::raw_string_ostream fullArgDescOS(fullArgDesc);
  DenseMap<StringRef, PublicDeclT *> declMap;
  for (auto &decl : decls)
    declMap.try_emplace(decl.getName(), &decl);

  for (++line; line < lineEnd && !lines[line].empty();) {
    // Extract the argument name and description.
    auto [argName, argDesc] = lines[line].split(':');
    argName = argName.trim();
    argDesc = argDesc.trim();

    fullArgDesc.clear();
    fullArgDescOS << argDesc;

    // Merge in additional description lines that have a larger indentation.
    // Remove the initial indent but leave other whitespace intact to preserve
    // Markdown formatting.
    size_t indent = getIndentationLevel(lines[line]);
    while (++line < lineEnd && getIndentationLevel(lines[line]) > indent)
      fullArgDescOS << "\n" << lines[line].drop_front(indent).rtrim();

    // If it's a known entry, process it, otherwise skip it.
    if (auto it = declMap.find(argName); it != declMap.end()) {
      it->getSecond()->setDescription(fullArgDesc);
    }
  }
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
template <typename PublicDeclType, typename OpType>
static SmallVector<PublicDeclType, 2>
extractChildDecls(const ASTDecl &decl,
                  function_ref<bool(OpType, StringRef)> shouldHideFn = {}) {
  DenseSet<Operation *> seenOps;
  SmallVector<PublicDeclType, 2> children;

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
      children.push_back(cast<PublicDeclType>(*getPublicDecl(child)));
    }
  }

  llvm::stable_sort(children, [](auto &lhs, auto &rhs) {
    return compareDeclNames(lhs.getName(), rhs.getName());
  });

  return children;
}

template <typename JSONSerializableItems>
static llvm::json::Array toJSONArray(MojoParserContext &ctx,
                                     const JSONSerializableItems &items) {
  llvm::json::Array jsonItems;
  for (const auto &item : items)
    jsonItems.push_back(item.toJSON(ctx));
  return jsonItems;
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

/// Dump a markdown section with a list of decls. Each decl is printed with the
/// format `name: description`. Decls without description are omitted, and the
/// section title is only dumped if there is at least one decl to show.
template <typename PublicDeclList>
static void dumpMarkdownDeclListSection(llvm::raw_ostream &os,
                                        StringRef sectionTitle,
                                        const PublicDeclList &decls) {
  bool isFirst = true;
  for (const auto &decl : decls) {
    if (decl.getDescription().empty())
      continue;

    if (isFirst) {
      isFirst = false;
      /// We only show the section title if there's at least one item to show.
      dumpMarkdownSectionTitle(os, sectionTitle);
    } else {
      /// This is a special separator for unbulleted lists.
      os << "\\\n";
    }
    os << kMarkdownIndent << decl.getName() << ": " << decl.getDescription()
       << "\n";
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

static ParameterEvaluator populatePublicParameterInfos(
    ArrayRef<Type> paramTypes, PogListAttr paramListAttr,
    SmallVectorImpl<ParameterInfo> &parameters, ASTDecl &parentDecl,
    std::optional<ASTType> selfType = std::nullopt,
    ArrayRef<ConstraintAttr> bodyConstraintAttrs = {},
    std::string *bodyConstraints = nullptr) {
  DeclResolver::DiagnosticDeclContextChanger scope(&parentDecl);
  auto &shared = parentDecl.getShared();

  SmallVector<ParameterInfo, 2> infos;
  ParameterEvaluator evaluator = populateParameterInfos(
      parentDecl, paramTypes, paramListAttr, infos, selfType);

  if (!bodyConstraintAttrs.empty() && bodyConstraints) {
    *bodyConstraints = mergeConformsToConstraints(bodyConstraintAttrs,
                                                  &evaluator, shared, infos);
  }

  parameters.reserve(parameters.size() + infos.size());
  llvm::append_range(parameters, std::move(infos));
  return evaluator;
}

static std::string getVariableType(ASTDecl &decl) {
  ASTType type = decl.getToolingType().getReferenceElementType();
  return type.getAsString(ASTTypePrinterContext(&decl.getShared()));
}

static std::string getArgumentDeclarationSnippet(const ArgumentInfo &info,
                                                 MojoParserContext &ctx) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  renderArgumentInfo(info, ctx.getSharedState(), os);
  return snippet;
}

static std::string getParameterDeclarationSnippet(const ParameterInfo &info,
                                                  MojoParserContext &ctx) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  renderParameterInfo(info, ctx.getSharedState(), os);
  return snippet;
}

static std::string getStructFieldType(StructFieldASTDecl &decl) {
  std::string type;
  llvm::raw_string_ostream os(type);
  ASTType(decl.getOp().getType()).print(os, /*ctx=*/{&decl.getShared()});
  return type;
}

static std::string getAliasDeclarationSnippet(AliasASTDecl &decl,
                                              MojoParserContext &ctx);
static std::string getAliasMarkdownDocString(AliasASTDecl &decl);

//===----------------------------------------------------------------------===//
// PublicDecl
//===----------------------------------------------------------------------===//

/// Return the signature type contained by a declaration, if any.
static FnTypeGeneratorType getSignatureFromDecl(ASTDecl *decl) {
  if (!decl)
    return nullptr;
  if (auto defDecl = dyn_cast<DefASTDecl>(decl))
    if (auto func = defDecl->getOp())
      return func.getFuncTypeGenerator();
  if (auto typeValue = decl->getIfTypeValue())
    return dyn_cast_or_null<FnTypeGeneratorType>(typeValue);
  return nullptr;
}

/// Return the rendered parameter information corresponding to `decl`, if its
/// parent includes the parameter in its public signature.
static std::optional<ParameterInfo> getParameterInfo(ASTDecl &decl) {
  auto irDecl = dyn_cast<IRASTDecl>(&decl);
  if (!irDecl || irDecl->getParamNumber() < 0)
    return std::nullopt;

  size_t paramNumber = static_cast<size_t>(irDecl->getParamNumber());
  if (auto *structDecl = dyn_cast<StructASTDecl>(decl.getParentDecl())) {
    SmallVector<ParameterInfo, kTypicalParameterCount> parameters =
        getStructParameters(*structDecl);
    if (parameters.size() <= paramNumber)
      return std::nullopt;
    ParameterInfo result = parameters[paramNumber];
    result.setDecl(decl);
    return result;
  }

  auto parent = llvm::unique_dyn_cast_or_null<PublicFunctionDecl>(
      getPublicDecl(decl.getParentDecl()));
  if (!parent || parent->getParameters().size() <= paramNumber)
    return std::nullopt;
  ParameterInfo result = parent->getParameters()[paramNumber];
  result.setDecl(decl);
  return result;
}

ParameterInfo::ParameterInfo(ASTDecl &decl) {
  if (std::optional<ParameterInfo> info = getParameterInfo(decl))
    *this = std::move(*info);
  else
    setDecl(decl);
}

std::unique_ptr<PublicDecl> M::KGEN::LIT::getPublicDecl(ASTDecl *decl) {
  if (!decl)
    return nullptr;

  if (auto varDecl = dyn_cast<VarASTDecl>(decl))
    return std::unique_ptr<PublicDecl>(new PublicDecl(
        varDecl->isArgument() ? PublicDeclKind::DK_PublicArgumentDecl
                              : PublicDeclKind::DK_PublicVariableDecl,
        *decl, decl->getName().value_or(StringRef())));

  if (Operation *declOp = decl->getIfOperation()) {
    if (isa<AliasDeclOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicAliasDecl, *decl,
                         decl->getName().value_or(StringRef())));
    if (isa<FnOp>(declOp))
      return std::unique_ptr<PublicFunctionDecl>(new PublicFunctionDecl(decl));
    if (isa<FileModuleOp, ImportOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicModuleDecl, *decl,
                         decl->getName().value_or(StringRef())));
    if (isa<StructDeclOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicStructDecl, *decl,
                         decl->getName().value_or(StringRef())));
    if (isa<StructFieldOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicStructFieldDecl, *decl,
                         decl->getName().value_or(StringRef())));
    if (isa<PackageOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicPackageDecl, *decl,
                         decl->getName().value_or(StringRef())));
    if (isa<TraitDeclOp>(declOp))
      return std::unique_ptr<PublicDecl>(
          new PublicDecl(PublicDeclKind::DK_PublicTraitDecl, *decl,
                         decl->getName().value_or(StringRef())));
  }

  if (auto signature = getSignatureFromDecl(decl))
    return std::unique_ptr<PublicFunctionDecl>(
        new PublicFunctionDecl(decl, signature));

  if (auto irDecl = dyn_cast<IRASTDecl>(decl);
      irDecl && irDecl->getArgNumber() >= 0)
    return std::unique_ptr<PublicDecl>(
        new PublicDecl(PublicDeclKind::DK_PublicArgumentDecl, *decl,
                       decl->getName().value_or(StringRef())));

  if (auto parameter = getParameterInfo(*decl))
    return std::unique_ptr<PublicDecl>(new PublicDecl(
        PublicDeclKind::DK_PublicParameterDecl, *decl, parameter->getName()));

  return nullptr;
}

std::optional<PublicDeclKind>
M::KGEN::LIT::getApproximatePublicDeclKind(ASTDecl *decl) {
  if (!decl)
    return std::nullopt;

  if (auto varDecl = dyn_cast<VarASTDecl>(decl))
    return varDecl->isArgument() ? PublicDeclKind::DK_PublicArgumentDecl
                                 : PublicDeclKind::DK_PublicVariableDecl;

  if (Operation *declOp = decl->getIfOperation()) {
    if (isa<AliasDeclOp>(declOp))
      return PublicDeclKind::DK_PublicAliasDecl;
    if (isa<FnOp>(declOp))
      return PublicFunctionDecl::getKindStatic();
    if (isa<FileModuleOp, ImportOp>(declOp))
      return PublicDeclKind::DK_PublicModuleDecl;
    if (isa<StructDeclOp>(declOp))
      return PublicDeclKind::DK_PublicStructDecl;
    if (isa<StructFieldOp>(declOp))
      return PublicDeclKind::DK_PublicStructFieldDecl;
    if (isa<PackageOp>(declOp))
      return PublicDeclKind::DK_PublicPackageDecl;
    if (isa<TraitDeclOp>(declOp))
      return PublicDeclKind::DK_PublicTraitDecl;
  }

  if (getSignatureFromDecl(decl))
    return PublicFunctionDecl::getKindStatic();
  if (auto irDecl = dyn_cast<IRASTDecl>(decl)) {
    if (irDecl->getArgNumber() >= 0)
      return PublicDeclKind::DK_PublicArgumentDecl;
    if (irDecl->getParamNumber() >= 0)
      return PublicDeclKind::DK_PublicParameterDecl;
    if (irDecl->getCapturedName())
      return PublicDeclKind::DK_PublicVariableDecl;
  }
  return std::nullopt;
}

StringRef PublicDecl::getKindAsString(PublicDeclKind kind) {
  switch (kind) {
  case PublicDeclKind::DK_PublicAliasDecl:
    return "alias";
  case PublicDeclKind::DK_PublicArgumentDecl:
    return "argument";
  case PublicDeclKind::DK_PublicFunctionDecl:
    return "function";
  case PublicDeclKind::DK_PublicModuleDecl:
    return "module";
  case PublicDeclKind::DK_PublicPackageDecl:
    return "package";
  case PublicDeclKind::DK_PublicParameterDecl:
    return "parameter";
  case PublicDeclKind::DK_PublicStructDecl:
    return "struct";
  case PublicDeclKind::DK_PublicStructFieldDecl:
    return "field";
  case PublicDeclKind::DK_PublicTraitDecl:
    return "trait";
  case PublicDeclKind::DK_PublicVariableDecl:
    return "variable";
  }
  llvm_unreachable("invalid kind");
}

StringRef PublicDecl::getKindAsString() const { return getKindAsString(kind); }

std::string PublicDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);

  switch (getKind()) {
  case PublicDeclKind::DK_PublicAliasDecl:
    return getAliasDeclarationSnippet(cast<AliasASTDecl>(getDecl()), ctx);
  case PublicDeclKind::DK_PublicArgumentDecl:
    return getArgumentDeclarationSnippet(ArgumentInfo(getDecl()), ctx);
  case PublicDeclKind::DK_PublicParameterDecl:
    return getParameterDeclarationSnippet(ParameterInfo(getDecl()), ctx);
  case PublicDeclKind::DK_PublicStructDecl:
    return getStructDeclarationSnippet(cast<StructASTDecl>(getDecl()), ctx);
  case PublicDeclKind::DK_PublicTraitDecl:
    return "trait " + getName().str();
  case PublicDeclKind::DK_PublicStructFieldDecl:
    dumpIdentifierWithType(
        os, getName(), getStructFieldType(cast<StructFieldASTDecl>(getDecl())));
    return "var " + os.str();
  case PublicDeclKind::DK_PublicVariableDecl:
    dumpIdentifierWithType(os, getName(), getVariableType(getDecl()));
    return "var " + os.str();
  default:
    return {};
  }
}

std::string PublicDecl::getFullMarkdownString(MojoParserContext &ctx) const {
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

  os << formatv("### {0} `{1}`\n", getKindAsString(), getName());
  if (auto docString = getMarkdownDocString(); !docString.empty())
    os << llvm::formatv(docStringSnippet, docString);

  os << llvm::formatv(declarationSnippet, getDeclarationSnippet(ctx));
  return buff;
}

static llvm::json::Object serializeParameterInfo(const ParameterInfo &info,
                                                 MojoParserContext &ctx) {
  ASTDecl &decl = info.getDecl();
  TypeMetadata typeMetadata =
      extractLibraryInfo(info.type, &decl, &decl.getShared());
  SmallVector<std::string> traitNames = parseCompoundTraitType(info.type);

  llvm::json::Object object{
      {"kind", "parameter"},
      {"name", prependVariadicIdentifiers(info.name, info.variadicKind).str()},
      {"passingKind", stringifyPassingKind(info.passingKind)},
      {"description", info.description},
  };

  if (traitNames.size() == 1) {
    llvm::json::Object metadataJson = typeMetadata.toJSON();
    for (auto &pair : metadataJson) {
      object[pair.first] = std::move(pair.second);
    }
  } else {
    object["type"] = info.type;
    llvm::json::Array traitsArray;
    for (const std::string &traitName : traitNames)
      traitsArray.push_back(
          extractLibraryInfo(traitName, &decl, &decl.getShared()).toJSON());
    object["traits"] = std::move(traitsArray);
  }

  if (info.defaultValue)
    object["default"] =
        getDefaultValueString(info.defaultValue, ctx.getSharedState());
  if (!info.constraints.empty())
    object["constraints"] = info.constraints;
  return object;
}

/// Convert a signature argument index to its index in the documented argument
/// list, omitting compiler-synthesized ABI arguments.
static std::optional<size_t>
getDocumentedArgumentIndex(FnTypeGeneratorType signature,
                           size_t signatureArgIndex) {
  if (!signature || signature.getNumArguments() <= signatureArgIndex)
    return std::nullopt;

  size_t documentedArgIndex = 0;
  for (size_t index = 0; index <= signatureArgIndex; ++index) {
    ArgConvention convention = signature.getArgConventions()[index];
    bool isDocumented = convention != ArgConvention::ByRefError &&
                        !(convention == ArgConvention::ByRefResult &&
                          signature.getArgName(index) == "__result__");
    if (index == signatureArgIndex)
      return isDocumented ? std::optional<size_t>(documentedArgIndex)
                          : std::nullopt;
    if (isDocumented)
      ++documentedArgIndex;
  }
  llvm_unreachable("signature argument index was range checked");
}

ArgumentInfo::ArgumentInfo(ASTDecl &decl) {
  setDecl(decl);
  ssize_t argNumber = -1;
  if (auto varDecl = dyn_cast<VarASTDecl>(&decl))
    argNumber = varDecl->getArgNumber();
  else if (auto irDecl = dyn_cast<IRASTDecl>(&decl))
    argNumber = irDecl->getArgNumber();

  if (argNumber < 0)
    return;

  std::optional<size_t> documentedArgNumber =
      getDocumentedArgumentIndex(getSignatureFromDecl(decl.getParentDecl()),
                                 static_cast<size_t>(argNumber));
  if (!documentedArgNumber)
    return;

  auto functionView = llvm::unique_dyn_cast_or_null<PublicFunctionDecl>(
      getPublicDecl(decl.getParentDecl()));
  if (!functionView ||
      functionView->getArguments().size() <= *documentedArgNumber)
    return;

  *this = functionView->getArgument(*documentedArgNumber);
  setDecl(decl);
}

static llvm::json::Object serializeArgumentInfo(const ArgumentInfo &info,
                                                MojoParserContext &ctx) {
  ASTDecl &decl = info.getDecl();
  TypeMetadata typeMetadata =
      extractLibraryInfo(info.type, &decl, &decl.getShared());
  SmallVector<std::string> traitNames = parseCompoundTraitType(info.type);

  llvm::json::Object object{
      {"description", info.description},
      {"convention", getConventionString(info.convention)},
      {"kind", "argument"},
      {"name", prependVariadicIdentifiers(info.name, info.variadicKind).str()},
      {"passingKind", stringifyPassingKind(info.passingKind)},
  };

  if (traitNames.size() == 1) {
    llvm::json::Object metadataJson = typeMetadata.toJSON();
    for (auto &pair : metadataJson) {
      object[pair.first] = std::move(pair.second);
    }
  } else {
    object["type"] = info.type;
    llvm::json::Array traitsArray;
    for (const std::string &traitName : traitNames)
      traitsArray.push_back(
          extractLibraryInfo(traitName, &decl, &decl.getShared()).toJSON());
    object["traits"] = std::move(traitsArray);
  }

  if (info.defaultValue)
    object["default"] =
        getDefaultValueString(info.defaultValue, ctx.getSharedState());
  return object;
}

//===----------------------------------------------------------------------===//
// Alias documentation
//===----------------------------------------------------------------------===//

static ParameterEvaluator
populateAliasParameterInfos(AliasASTDecl &decl,
                            SmallVectorImpl<ParameterInfo> &parameters,
                            std::string *constraints = nullptr) {
  auto generatorType = dyn_cast<GeneratorType>(decl.getOp().getType());
  if (!generatorType)
    return {};

  auto &shared = decl.getShared();
  ParameterEvaluator evaluator =
      populateParameterInfos(decl, generatorType.getInputParamTypes(),
                             generatorType.getParamListAttrs(), parameters);
  if (constraints)
    *constraints = mergeConformsToConstraints(
        generatorType.getParamListAttrs().getBodyConstraints(), &evaluator,
        shared, parameters);
  return evaluator;
}

static std::string getAliasValue(AliasASTDecl &decl,
                                 ParameterEvaluator &evaluator) {
  std::optional<Attribute> value = decl.getOp().getValue();
  if (!value)
    return {};
  if (auto generator = dyn_cast<GeneratorAttr>(*value))
    value = evaluator.getReboundAttribute(generator.getBody());
  return generatePValueString(decl.getShared(), *value);
}

static std::string getAliasDeclarationSnippet(AliasASTDecl &decl,
                                              MojoParserContext &ctx) {
  SmallVector<ParameterInfo, kTypicalParameterCount> parameters;
  std::string constraints;
  ParameterEvaluator evaluator =
      populateAliasParameterInfos(decl, parameters, &constraints);
  std::string type;
  if (auto generatorType = dyn_cast<GeneratorType>(decl.getOp().getType()))
    type = generateTypeString(decl.getShared(),
                              evaluator.getReboundType(generatorType.getBody()),
                              VariadicKind::None);

  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "comptime ";
  printAliasSignatureFromInfos(decl.getName().value_or(StringRef()), type,
                               parameters, constraints, ctx.getSharedState(),
                               os, {});
  if (std::string value = getAliasValue(decl, evaluator); !value.empty())
    os << " = " << value;
  return snippet;
}

static std::string getAliasMarkdownDocString(AliasASTDecl &decl) {
  SmallVector<ParameterInfo, kTypicalParameterCount> parameters;
  populateAliasParameterInfos(decl, parameters);

  std::string summary;
  std::string description;
  if (auto docString = decl.getParsedDocString()) {
    summary = docString->getSummary();
    SmallVector<std::string> descriptionLines;
    for (size_t line = 0, end = docString->getDescription().size(); line < end;
         ++line) {
      StringRef section = docString->getDescription()[line];
      if (section == (Twine(DocString::kSectionParameters) + ":").str()) {
        augmentDeclsWithDocumentation(docString->getDescription(), line, end,
                                      parameters);
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
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  return markdown;
}

static llvm::json::Object serializeAliasDecl(AliasASTDecl &decl,
                                             MojoParserContext &ctx) {
  SmallVector<ParameterInfo, kTypicalParameterCount> parameterInfos;
  std::string constraints;
  ParameterEvaluator evaluator =
      populateAliasParameterInfos(decl, parameterInfos, &constraints);
  std::string type;
  if (auto generatorType = dyn_cast<GeneratorType>(decl.getOp().getType()))
    type = generateTypeString(decl.getShared(),
                              evaluator.getReboundType(generatorType.getBody()),
                              VariadicKind::None);

  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);
  signatureOS << "comptime ";
  printAliasSignatureFromInfos(decl.getName().value_or(StringRef()),
                               /*type=*/"", parameterInfos, constraints,
                               ctx.getSharedState(), signatureOS, {});

  std::string description;
  std::string summary;
  if (auto docString = decl.getParsedDocString()) {
    summary = docString->getSummary();
    SmallVector<std::string> descriptionLines;
    for (size_t line = 0, end = docString->getDescription().size(); line < end;
         ++line) {
      StringRef section = docString->getDescription()[line];
      if (section == (Twine(DocString::kSectionParameters) + ":").str()) {
        augmentDeclsWithDocumentation(docString->getDescription(), line, end,
                                      parameterInfos);
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
  for (const ParameterInfo &parameter : parameterInfos)
    parameters.push_back(serializeParameterInfo(parameter, ctx));

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
      {"value", getAliasValue(decl, evaluator)},
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

std::string M::KGEN::LIT::getToolingAliasValue(AliasASTDecl &decl) {
  SmallVector<ParameterInfo, kTypicalParameterCount> parameters;
  ParameterEvaluator evaluator = populateAliasParameterInfos(decl, parameters);
  return getAliasValue(decl, evaluator);
}

//===----------------------------------------------------------------------===//
// PublicFunctionDecl
//===----------------------------------------------------------------------===//

void PublicFunctionDecl::augmentWithDocumentation(ArrayRef<StringRef> desc) {
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
      augmentDeclsWithDocumentation(desc, line, lineEnd, args);
    } else if (desc[line] == paramsSectionHeader) {
      augmentDeclsWithDocumentation(desc, line, lineEnd, parameters);
    } else if (desc[line] == returnsSectionHeader) {
      if (returnType)
        returnsDoc = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] == constraintsSectionHeader) {
      constraints = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] == raisesSectionHeader) {
      if (raises())
        raisesDoc = parseDocStringSection(desc, line, lineEnd);
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
  description = DocString::formatDescription(pureDescriptionLinesRef);
}

std::string
PublicFunctionDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  return getDeclarationSnippet(ctx, /*parameterOffsets=*/nullptr,
                               /*argumentOffsets=*/nullptr);
}

std::string PublicFunctionDecl::getDeclarationSnippet(
    MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  if (isAsync())
    os << "__async ";

  unsigned returnOffset = 0;
  std::string signature =
      getSignature(ctx, parameterOffsets, argumentOffsets, &returnOffset);
  StringRef resultLessSignature(signature.data(), returnOffset);

  // Adjust the signature offsets.
  size_t signatureStart = os.str().size();
  auto adjustOffsets = [&](auto *v) {
    for (auto &offset : *v) {
      offset.first += signatureStart;
      offset.second += signatureStart;
    }
  };
  if (parameterOffsets)
    adjustOffsets(parameterOffsets);
  if (argumentOffsets)
    adjustOffsets(argumentOffsets);

  // Emit the signature.
  os << resultLessSignature;

  if (raises())
    os << " raises";

  os << StringRef(signature).drop_front(returnOffset);
  return snippet;
}

std::string PublicFunctionDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, summary);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  dumpMarkdownDeclListSection(os, DocString::kSectionArgs, args);
  dumpMarkdownTextSection(os, DocString::kSectionReturns, returnsDoc);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints, constraints);
  dumpMarkdownTextSection(os, DocString::kSectionRaises, raisesDoc);
  dumpMarkdownDocumentationDescription(os, description);

  return markdown;
}

std::string PublicFunctionDecl::getSignature(
    MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets,
    unsigned *returnOffset) const {
  // Common path: delegate to the canonical op-driven printer in MojoParser.
  // This is the same code the compiler calls when synthesizing declarations
  // for diagnostics that have no source location.
  ASTDecl &decl = getDecl();
  if (auto *fnDecl = dyn_cast<DefASTDecl>(&decl)) {
    std::string out;
    llvm::raw_string_ostream os(out);
    SignatureOffsets so;
    so.parameters = parameterOffsets;
    so.arguments = argumentOffsets;
    so.returnTypeStart = returnOffset;
    printFunctionSignature(*fnDecl, os, so);
    return out;
  }

  // Fallback path: this declaration has a derived signature without a backing
  // `FnOp`. Copy the cached parameter/arg infos into the shared renderer.
  SmallVector<ParameterInfo, kTypicalParameterCount> paramInfos;
  paramInfos.reserve(parameters.size());
  for (const auto &p : parameters)
    paramInfos.push_back(p);

  SmallVector<ArgumentInfo, kTypicalArgumentCount> argInfos;
  argInfos.reserve(args.size());
  for (const auto &a : args)
    argInfos.push_back(a);

  // `returnType` is already gated on `!hasOutArgument` at construction time:
  // `initFromSignature` only populates it when the function has no out-arg
  // result, so passing it through unconditionally is safe.
  std::string out;
  llvm::raw_string_ostream os(out);
  SignatureOffsets so;
  so.parameters = parameterOffsets;
  so.arguments = argumentOffsets;
  so.returnTypeStart = returnOffset;
  printFunctionSignatureFromInfos(
      getName().split('(').first, argInfos, paramInfos,
      returnType.value_or(std::string{}), fnConstraints, isInit, isMethodFlag,
      ctx.getSharedState(), os, so);
  return out;
}

llvm::json::Object PublicFunctionDecl::toJSON(MojoParserContext &ctx) const {
  llvm::json::Array arguments;
  for (const ArgumentInfo &argument : args)
    arguments.push_back(serializeArgumentInfo(argument, ctx));
  llvm::json::Array parameterInfos;
  for (const ParameterInfo &parameter : parameters)
    parameterInfos.push_back(serializeParameterInfo(parameter, ctx));

  llvm::json::Object result{
      {"args", std::move(arguments)},
      {"async", isAsync()},
      {"constraints", constraints},
      {"deprecated", deprecated},
      {"description", description},
      {"hasDefaultImplementation", hasDefaultImplementation()},
      {"isStatic", isStatic()},
      {"isImplicitConversion", isImplicitConversion()},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", std::move(parameterInfos)},
      {"raises", raises()},
      {"raisesDoc", raisesDoc},
      {"signature", getSignature(ctx)},
      {"isStable", stableInfo.isStable},
      {"isStabilityTracked", stableInfo.isStabilityTracked},
      {"sinceVersion", stableInfo.sinceVersion},
      {"summary", summary},
  };

  // Create unified "returns" object with type, path, and doc
  if (returnType && !returnType->empty()) {
    TypeMetadata metadata =
        extractLibraryInfo(*returnType, &getDecl(), &ctx.getSharedState());

    llvm::json::Object returnObj = metadata.toJSON();
    if (!returnsDoc.empty()) {
      returnObj["doc"] = returnsDoc;
    }

    result["returns"] = std::move(returnObj);
  }

  return result;
}

PublicFunctionDecl::PublicFunctionDecl(ASTDecl *decl)
    : PublicDecl(PublicDeclKind::DK_PublicFunctionDecl, *decl,
                 decl->getName().value_or(StringRef{})),
      deprecated(decl->getDeprecationWarning().value_or(StringRef{})) {
  auto funcOp = cast<DefASTDecl>(decl)->getOp();
  isStaticFlag = funcOp.getIsStatic();
  isImplicitConversionFlag = funcOp.isImplicitConversion();
  isMethodFlag = !isStaticFlag && isa<StructDeclOp>(funcOp->getParentOp());
  isInit = funcOp.getSpecialFunctionInfo().isInitializer();
  isDefaultImplFlag = funcOp.isDefaultedTraitFn();
  stableInfo = computeStableInfo(funcOp);

  initFromSignature(decl, funcOp.getFuncTypeGenerator(),
                    funcOp.getArgumentTypes(), funcOp.getUserResultType());
}

PublicFunctionDecl::PublicFunctionDecl(ASTDecl *decl,
                                       FnTypeGeneratorType signature)
    : PublicDecl(PublicDeclKind::DK_PublicFunctionDecl, *decl,
                 /*name=*/StringRef()) {
  initFromSignature(decl, signature, signature.getArguments(),
                    signature.getUserResultType());
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

void PublicFunctionDecl::initFromSignature(ASTDecl *decl,
                                           FnTypeGeneratorType signature,
                                           ArrayRef<Type> userArgTypes,
                                           Type userResultType) {
  DeclResolver::DiagnosticDeclContextChanger declScopeChanger(decl);
  auto &shared = decl->getShared();
  raisesFlag = signature.isThrows();
  isAsyncFlag = signature.isAsync();

  // If this is a method, grab the expected "Self" type.
  std::optional<ASTType> selfType;
  if (isa<StructASTDecl>(decl->getParentDecl()))
    selfType = cast<StructASTDecl>(decl->getParentDecl())->getSelfType();

  // Populate parameter infos (with per-param `conforms_to` merging applied
  // internally), then fold in function-level constraints. After this block
  // each `ParameterInfo` carries the final user-facing type string.
  ParameterEvaluator evaluator = populatePublicParameterInfos(
      signature.getInputParamTypes(), signature.getParamListAttrs(), parameters,
      *decl, selfType, signature.getParamListAttrs().getBodyConstraints(),
      &fnConstraints);

  // Populate argument infos.
  SmallVector<ArgumentInfo, kTypicalArgumentCount> argInfos;
  populateArgumentInfos(
      *decl, signature, userArgTypes, selfType, evaluator,
      [&]() -> bool {
        if (auto *fnDecl = dyn_cast<DefASTDecl>(decl))
          if (auto fnOp = fnDecl->getOp())
            return fnOp.getSpecialFunctionInfo().hasSelfResult();
        return false;
      },
      argInfos);
  args.reserve(argInfos.size());
  for (auto &info : argInfos) {
    info.setDecl(*decl);
    args.push_back(std::move(info));
  }

  // Grab the result type, if it's non-none.
  ASTType resultType = signature.getUserResultType();
  assert(resultType && "didn't find a result type?");

  if (!resultType.isNoneType()) {
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
    returnType = str;
  }

  // An inherited default is a copy that cannot carry a doc string of its own,
  // so its documentation comes from the trait method that declares it.
  if (auto docStr = findDefaultImplDecl(shared, decl)->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclOverloadSet
//===----------------------------------------------------------------------===//

SmallVector<FunctionDeclOverloadSet, 2>
FunctionDeclOverloadSet::fromSortedFunctions(
    SmallVector<PublicFunctionDecl, 2> &&functions) {
  SmallVector<FunctionDeclOverloadSet, 2> overloads;
  for (auto &function : functions) {
    if (overloads.empty() ||
        overloads.back().getBaseName() != function.getName())
      overloads.emplace_back(FunctionDeclOverloadSet(function.getName()));

    overloads.back().append(std::move(function));
  }
  return overloads;
}

llvm::json::Object
FunctionDeclOverloadSet::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"kind", "function"},
                            {"name", baseName},
                            {"overloads", toJSONArray(ctx, functions)}};
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
static void collectParentTraits(MojoParserContext &ctx, ASTDecl *self,
                                SmallVectorImpl<std::string> &parentTraits,
                                TraitType canonicalTrait,
                                const ConformanceConditionMap *conditions) {
  DenseSet<TraitSymbolAttr> seenDecls;
  for (TraitSymbolAttr symbol : canonicalTrait.getSymbols()) {
    if (!seenDecls.insert(symbol).second)
      continue;
    ASTDecl *decl = ctx.getTraitDecl(symbol);
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
collectParentTraitsWithMetadata(MojoParserContext &ctx, ASTDecl *self,
                                TraitType canonicalTrait,
                                const ConformanceConditionMap *conditions) {
  llvm::json::Array result;
  DenseSet<TraitSymbolAttr> seenDecls;

  struct TraitEntry {
    StringRef name;
    TypeMetadata metadata;
    StringRef condition;
  };
  SmallVector<TraitEntry> traitData;

  for (TraitSymbolAttr symbol : canonicalTrait.getSymbols()) {
    if (!seenDecls.insert(symbol).second)
      continue;

    StringRef condition = lookupConformanceCondition(conditions, symbol);

    // Try to resolve the trait through AST
    ASTDecl *decl = ctx.getTraitDecl(symbol);
    if (decl && decl != self) {
      std::optional<StringRef> name = decl->getName();
      if (name && isa<TraitASTDecl>(decl)) {
        TypeMetadata metadata =
            extractLibraryInfo(*name, self, &ctx.getSharedState());
        traitData.push_back({*name, metadata, condition});
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

      TypeMetadata metadata =
          extractLibraryInfo(traitName, self, &ctx.getSharedState());
      traitData.push_back({traitName, metadata, condition});
    }
  }

  // Sort by trait name for consistency
  llvm::sort(traitData,
             [](const auto &a, const auto &b) { return a.name < b.name; });

  // Create JSON objects with flattened metadata structure
  for (const TraitEntry &entry : traitData) {
    llvm::json::Object traitObj;
    traitObj["name"] = entry.name.str();

    if (!entry.metadata.getRelativeDocPath().empty()) {
      traitObj["path"] = entry.metadata.getRelativeDocPath().str();
    }

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

  auto aliases = extractChildDecls<PublicDecl, AliasDeclOp>(decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(decl, shouldHideFn));

  // Collect parent traits with type metadata. A trait's own conformance list
  // cannot carry a `where` clause, so there are no conditions to report.
  llvm::json::Array parentTraitsWithMetadata = collectParentTraitsWithMetadata(
      ctx, &decl, decl.getOp().getCanonicalTrait(),
      /*conditions=*/nullptr);
  StableInfo stableInfo = computeStableInfo(decl.getOp());

  return llvm::json::Object{
      {"aliases", toJSONArray(ctx, aliases)},
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"description", decl.getDocumentationDescription()},
      {"fields", llvm::json::Array()},
      {"functions", toJSONArray(ctx, functionOverloads)},
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
  SmallVector<ParameterInfo, kTypicalParameterCount> parameters;
  std::string constraints;
  std::string description;
  std::string summary;
  ConformanceConditionMap conformanceConditions;
  StableInfo stableInfo;
  TypeConvention convention;
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
      augmentDeclsWithDocumentation(desc, line, lineEnd,
                                    documentation.parameters);
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
  StructDeclOp structOp = decl.getOp();
  TypeSignatureType signature = structOp.getSignature();
  documentation.convention = structOp.getConvention();
  documentation.stableInfo = computeStableInfo(structOp);

  auto &shared = decl.getShared();
  PogListAttr paramListAttr = signature.getParamListAttrs();
  ParameterEvaluator evaluator = populatePublicParameterInfos(
      signature.getInputParamTypes(), paramListAttr, documentation.parameters,
      decl, /*selfType=*/std::nullopt, paramListAttr.getBodyConstraints(),
      /*bodyConstraints=*/nullptr);
  collectConformanceConditions(structOp.getCanonicalTrait(), evaluator, shared,
                               documentation.conformanceConditions);

  if (auto docStr = decl.getParsedDocString()) {
    documentation.summary = docStr->getSummary();
    augmentStructDocumentation(documentation, docStr->getDescription());
  }
  return documentation;
}

SmallVector<ParameterInfo, kTypicalParameterCount>
M::KGEN::LIT::getStructParameters(StructASTDecl &decl) {
  return getStructDocumentation(decl).parameters;
}

std::string M::KGEN::LIT::getStructSignature(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  SignatureOffsets offsets;
  offsets.parameters = parameterOffsets;
  printStructSignature(decl, os, offsets);
  return signature;
}

std::string M::KGEN::LIT::getStructDeclarationSnippet(
    StructASTDecl &decl, MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << getStructSignature(decl, parameterOffsets);

  SmallVector<std::string> parentTraits;
  StructDocumentation documentation = getStructDocumentation(decl);
  collectParentTraits(ctx, &decl, parentTraits,
                      decl.getOp().getCanonicalTrait(),
                      &documentation.conformanceConditions);
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

  dumpMarkdownDocumentationHeader(os, documentation.summary,
                                  documentation.description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters,
                              documentation.parameters);
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
  auto aliases = extractChildDecls<PublicDecl, AliasDeclOp>(decl);
  auto fields = extractChildDecls<PublicDecl, StructFieldOp>(decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(decl));

  llvm::json::Array parentTraitsWithMetadata = collectParentTraitsWithMetadata(
      ctx, &decl, decl.getOp().getCanonicalTrait(),
      &documentation.conformanceConditions);
  llvm::json::Array parameterInfos;
  for (const ParameterInfo &parameter : documentation.parameters)
    parameterInfos.push_back(serializeParameterInfo(parameter, ctx));

  return llvm::json::Object{
      {"aliases", toJSONArray(ctx, aliases)},
      {"constraints", documentation.constraints},
      {"deprecated", decl.getDeprecationWarning().value_or(StringRef())},
      {"description", documentation.description},
      {"fields", toJSONArray(ctx, fields)},
      {"functions", toJSONArray(ctx, functionOverloads)},
      {"kind", "struct"},
      {"name", decl.getName().value_or(StringRef())},
      {"parameters", std::move(parameterInfos)},
      {"parentTraits", std::move(parentTraitsWithMetadata)},
      {"signature", getStructSignature(decl)},
      {"isStable", documentation.stableInfo.isStable},
      {"isStabilityTracked", documentation.stableInfo.isStabilityTracked},
      {"sinceVersion", documentation.stableInfo.sinceVersion},
      {"summary", documentation.summary},
      {"convention", toString(documentation.convention)},
  };
}

//===----------------------------------------------------------------------===//
// Module and package serialization
//===----------------------------------------------------------------------===//

static llvm::json::Object serializeModuleDecl(ASTDecl &decl,
                                              MojoParserContext &ctx) {
  auto aliases = extractChildDecls<PublicDecl, AliasDeclOp>(decl);
  auto structs = extractChildDecls<PublicDecl, StructDeclOp>(decl);
  auto traits = extractChildDecls<PublicDecl, TraitDeclOp>(decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(decl));

  return llvm::json::Object{{"aliases", toJSONArray(ctx, aliases)},
                            {"description", decl.getDocumentationDescription()},
                            {"functions", toJSONArray(ctx, functionOverloads)},
                            {"kind", "module"},
                            {"name", decl.getName().value_or(StringRef{})},
                            {"structs", toJSONArray(ctx, structs)},
                            {"traits", toJSONArray(ctx, traits)},
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
    TypeMetadata metadata =
        extractLibraryInfo(type, nullptr, &ctx.getSharedState());
    llvm::json::Object metadataJSON = metadata.toJSON();
    for (auto &pair : metadataJSON)
      result[pair.first] = std::move(pair.second);
  } else {
    result["type"] = type;
    llvm::json::Array traits;
    for (const std::string &traitName : traitNames) {
      TypeMetadata metadata =
          extractLibraryInfo(traitName, nullptr, &ctx.getSharedState());
      traits.push_back(metadata.toJSON());
    }
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
    if (std::unique_ptr<PublicDecl> publicDecl = getPublicDecl(decl))
      result.push_back(publicDecl->toJSON(ctx));
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

llvm::json::Object PublicDecl::toJSON(MojoParserContext &ctx) const {
  switch (getKind()) {
  case PublicDeclKind::DK_PublicAliasDecl:
    return serializeAliasDecl(cast<AliasASTDecl>(getDecl()), ctx);
  case PublicDeclKind::DK_PublicArgumentDecl:
    return serializeArgumentInfo(ArgumentInfo(getDecl()), ctx);
  case PublicDeclKind::DK_PublicParameterDecl:
    return serializeParameterInfo(ParameterInfo(getDecl()), ctx);
  case PublicDeclKind::DK_PublicStructDecl:
    return serializeStructDecl(cast<StructASTDecl>(getDecl()), ctx);
  case PublicDeclKind::DK_PublicStructFieldDecl:
    return serializeStructFieldDecl(cast<StructFieldASTDecl>(getDecl()));
  case PublicDeclKind::DK_PublicTraitDecl:
    return serializeTraitDecl(cast<TraitASTDecl>(getDecl()), ctx);
  case PublicDeclKind::DK_PublicVariableDecl:
    return serializeVariableDecl(getDecl(), ctx);
  case PublicDeclKind::DK_PublicModuleDecl:
    return serializeModuleDecl(getDecl(), ctx);
  case PublicDeclKind::DK_PublicPackageDecl:
    return serializePackageDecl(cast<PackageASTDecl>(getDecl()), ctx);
  default:
    break;
  }
  llvm_unreachable("generic PublicDecl has no renderer");
}

std::string PublicDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  ASTDecl &decl = getDecl();
  switch (getKind()) {
  case PublicDeclKind::DK_PublicAliasDecl:
    return getAliasMarkdownDocString(cast<AliasASTDecl>(getDecl()));
  case PublicDeclKind::DK_PublicArgumentDecl:
    dumpMarkdownDocumentationHeader(os, ArgumentInfo(decl).description);
    return markdown;
  case PublicDeclKind::DK_PublicParameterDecl:
    return ParameterInfo(decl).getMarkdownDocString();
  case PublicDeclKind::DK_PublicStructDecl:
    return getStructMarkdownDocString(cast<StructASTDecl>(decl));
  case PublicDeclKind::DK_PublicStructFieldDecl:
  case PublicDeclKind::DK_PublicModuleDecl:
  case PublicDeclKind::DK_PublicPackageDecl:
  case PublicDeclKind::DK_PublicTraitDecl:
    dumpMarkdownDocumentationHeader(os, decl.getDocumentationSummary(),
                                    decl.getDocumentationDescription());
    break;
  default:
    break;
  }
  return markdown;
}
