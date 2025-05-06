//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Two spaces that are forcefully added to markdown lines that can be used for
/// indentation.
static constexpr const char *kMarkdownIndent = "&nbsp;&nbsp;";

/// Return an ordering priority number for the given decl name. Lower numbers
/// are ordered first.
static unsigned getDeclNamePriority(StringRef name) {
  // If the name is a special function, use that as the priority.
  SpecialFunctionKind specialFnKind = SpecialFunctionInfo::getKind(name);
  if (specialFnKind != SpecialFunctionKind::kNormal)
    return static_cast<unsigned>(specialFnKind);

  // Otherwise, we can't discern any priorty from the name.
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

/// Refine the convention for the given type and input convention.
static ArgConvention refineConventionForType(Type type,
                                             ArgConvention convention) {
  if (auto variadic = dyn_cast<VariadicType>(type))
    return variadic.getConvention();
  return convention;
}

/// Generate a user-readable representation of the given pvalue.
static std::string generatePValueString(SharedState &shared, PValue value) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);
  ASTType::printParam(os, value, /*forDiag=*/&shared, /*demangleParams=*/false);
  return os.str();
}

/// Unpack a origin into a printable name when it is uttered in a signature
/// position.
static std::string getSignatureOrigin(SharedState &shared, TypedAttr origin,
                                      FnTypeGeneratorType signature,
                                      bool isRefResult) {
  // Strip out extra stuff.
  origin = OriginType::stripMutCastAndFieldExtract(origin);

  // Check to see if the origin is a parameter on this signature.  If so, it
  // will have a depth of zero.
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(origin);
      indexRef && indexRef.getDepth() == 0 && signature) {
    PogListAttr paramListMetadata = signature.getParamListAttrs();

    // Handle uses of implicit origins.
    if (paramListMetadata.getPassingKind(indexRef.getIndex()) ==
        PassingKind::Implicit) {
      // If this is a "ref [_]" argument, don't print the []'s.
      if (!isRefResult)
        return "";
      // If this is a result, figure out which argument infers this and use its
      // name.
      for (auto [idx, type] : llvm::enumerate(signature.getArguments())) {
        if (auto refType = dyn_cast<RefType>(type)) {
          if (refType.getOrigin() == origin)
            return signature.getArgName(idx).str();
        }
      }
    }

    return paramListMetadata.getName(indexRef.getIndex()).str();
  }

  // Combine unions into comma separated string.
  if (auto unionAttr = dyn_cast<OriginUnionAttr>(origin)) {
    std::string result;
    llvm::interleave(
        unionAttr.getOperands(),
        [&](TypedAttr elt) {
          result += getSignatureOrigin(shared, elt, signature, isRefResult);
        },
        [&]() { result += ", "; });
    return result;
  }

  // Otherwise, just print as normal.
  return generatePValueString(shared, origin);
}

/// Unpack a "ref" argument or result type into a string that can be shown to
/// the user.
static std::string getRefPrefixAsString(SharedState &shared, RefType refType,
                                        FnTypeGeneratorType signature,
                                        bool isRefResult) {
  std::string signatureLifetime =
      getSignatureOrigin(shared, refType.getOrigin(), signature, isRefResult);

  // Include the address space if it is non-default.
  if (!refType.isDefaultAddrSpace()) {
    // It will often be two extract_elements from the inner guts of the actual
    // AddressSpace value. Remove them.
    TypedAttr addrSpace = refType.getAddressSpace();
    if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(addrSpace)) {
      addrSpace = extractAttr.getStructValue();
      if (auto extractAttr2 = dyn_cast<LIT::StructExtractAttr>(addrSpace)) {
        addrSpace = extractAttr2.getStructValue();
      }
    }
    if (!signatureLifetime.empty())
      signatureLifetime += ", ";
    signatureLifetime += generatePValueString(shared, addrSpace);
  }

  if (signatureLifetime.empty())
    return std::string();

  return "[" + signatureLifetime + "] ";
}

/// Generate a user-readable representation of the given type and variadic kind,
/// with an optional value convention, and parent struct "Self" type.
static std::string
generateTypeString(SharedState &shared, Type type, VariadicKind varKind,
                   std::optional<ASTType> selfType = std::nullopt,
                   std::optional<ArgConvention> convention = std::nullopt) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);
  ASTType astType(type);

  if (varKind == VariadicKind::PosVarArg) {
    astType = astType.getVariadicElementType();
  } else if (varKind == VariadicKind::PackVarArg && !isa<PackType>(type)) {
    // VariadicPack needs special printing, because its argument isn't a type.
    os << "*";
    if (convention && hasAddress(*convention))
      astType = astType.getReferenceElementType();

    ASTType::printParam(os, astType.getVariadicPackTypeList(),
                        /*forDiag=*/&shared, /*demangleParams=*/true);
    return os.str();
  }

  // Process the convention if present.
  if (convention && hasAddress(*convention)) {
    // In some cases variadics are passed directly (which is a hack, but okay).
    // The ABI in these cases is that we pass a variadic of refs. We leave these
    // as is, since eventually (with unpacking) this hack won't be needed.
    if (!isa<VariadicType>(astType))
      astType = astType.getReferenceElementType();
  }

  // Get the value type in a kwargs dictionary.
  if (varKind == VariadicKind::KwVarArg)
    astType = astType.getKwargsDictValueType();

  // If this type is the same as the self type, use the "Self" keyword.
  if (selfType && astType.isEqualCanon(*selfType))
    os << "Self";
  else
    os << astType.getAsString(/*forDiag=*/&shared, /*demangleParams=*/true);

  return os.str();
}

/// If the argument/parameter is variadic, we put the star (or two stars if
/// variadic keyword) before the identifier.
static Twine prependVariadicIdentifiers(const Twine &identifier,
                                        VariadicKind varKind) {
  switch (varKind) {
  case VariadicKind::PosVarArg:
  case VariadicKind::PackVarArg:
    return "*" + identifier;
  case VariadicKind::KwVarArg:
    return "**" + identifier;
  default:
    return identifier;
  }
}

// Helper function that dumps an identifier along with an optional
// type. It also takes care of varargs that need to encode * in the name.
static void dumpIdentifierWithType(raw_ostream &os, StringRef identifier,
                                   StringRef type,
                                   VariadicKind varKind = VariadicKind::None,
                                   bool elideType = false) {
  os << prependVariadicIdentifiers(identifier, varKind);
  if (!type.empty() && !elideType)
    os << ": " << type;
}

/// Parse the given docstring lines and augment the provided decls with the
/// appropriate documentation using the description.
template <typename PublicDeclT>
static void augmentDeclsWithDocumentation(ArrayRef<StringRef> lines,
                                          size_t &line, size_t lineEnd,
                                          SmallVector<PublicDeclT> &decls) {
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

  std::string paragraph;
  llvm::raw_string_ostream paragraphOS(paragraph);

  paragraphOS << lines[++line].trim();

  // Merge in additional description lines that have equal or larger
  // indentation.
  size_t indent = getIndentationLevel(lines[line]);
  while (++line < lineEnd && getIndentationLevel(lines[line]) >= indent)
    paragraphOS << "\n" << lines[line].trim();
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
    if (decls.empty() || !isa<OpType>(**decls.begin()))
      continue;

    for (ASTDecl *child : decls) {
      OpType childOp = dyn_cast<OpType>(*child);
      if (!childOp || shouldHideDeclInDocGen(*child, name))
        continue;

      // Skip declarations that were imported from other scopes.
      if (child->getParentDecl() != &decl || !seenOps.insert(childOp).second)
        continue;
      // Skip synthetic declarations that don't have accompanying documentation
      // generated with them.
      if (childOp.isSynthetic() && !childOp.getDocStringAttr())
        continue;
      if (shouldHideFn && shouldHideFn(childOp, name))
        continue;
      children.push_back(
          cast<PublicDeclType>(*MojoASTDeclRef(child).getDecl()));
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
/// format `name: description`. Decls without description are ommitted, and the
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

//===----------------------------------------------------------------------===//
// PublicDecl
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// PublicVariableDecl
//===----------------------------------------------------------------------===//

std::string
PublicVariableDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "var ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

llvm::json::Object PublicVariableDecl::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"deprecated", deprecated},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"type", type},
  };
}

PublicVariableDecl::PublicVariableDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicVariableDecl,
                 declRef.getName().value_or(StringRef{})),
      isGlobalVariable(false),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())) {
  auto &shared = *declRef.getShared();
  TypeSwitch<mlir::Operation *>(declRef.getIfOperation())
      .Case([&](VarDeclOp op) {
        type = declRef.getType().getReferenceElementType().getAsString(shared);
      })
      .Case([&](GlobalVarDeclOp op) {
        type = declRef.getType().getAsString(shared);
        isGlobalVariable = true;
      });
}

//===----------------------------------------------------------------------===//
// PublicParameterDecl
//===----------------------------------------------------------------------===//

std::string
PublicParameterDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  dumpIdentifierWithType(os, getName(), type, variadicKind);
  if (defaultValue)
    os << " = " << *defaultValue;
  return buff;
}

std::string PublicParameterDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object PublicParameterDecl::toJSON(MojoParserContext &ctx) const {
  llvm::json::Object object{
      {"kind", getKindAsString()},
      {"name", prependVariadicIdentifiers(getName(), variadicKind).str()},
      {"type", type},
      {"passingKind", stringifyPassingKind(passingKind)},
      {"description", description},
  };
  if (defaultValue)
    object["default"] = *defaultValue;
  return object;
}

//===----------------------------------------------------------------------===//
// PublicArgumentDecl
//===----------------------------------------------------------------------===//

static StringRef getConventionString(PublicArgumentDecl::Convention conv) {
  StringRef conventions[] = {"read", "mut", "owned", "ref", "out"};
  auto convIdx = static_cast<size_t>(conv);
  assert(convIdx < sizeof(conventions) / sizeof(conventions[0]) &&
         "enums added");
  return conventions[convIdx];
}

std::string
PublicArgumentDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  // Print the convention of the argument, eliding the default.
  if (convention != Convention::kBorrowed)
    os << getConventionString(convention) << ' ';

  // Include the prefix if any (eg for a ref argument).
  os << prefix;

  bool elideType = isSelf && type == "Self";
  dumpIdentifierWithType(os, getName(), type, variadicKind, elideType);
  if (defaultValue)
    os << " = " << *defaultValue;
  return buff;
}

std::string PublicArgumentDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object PublicArgumentDecl::toJSON(MojoParserContext &ctx) const {
  [[maybe_unused]] StringRef conventions[] = {"read", "mut", "owned", "ref",
                                              "out"};
  assert(static_cast<size_t>(convention) <
             sizeof(conventions) / sizeof(conventions[0]) &&
         "enums added");

  llvm::json::Object object{
      {"description", description},
      {"convention", getConventionString(convention)},
      {"kind", getKindAsString()},
      {"name", prependVariadicIdentifiers(getName(), variadicKind).str()},
      {"type", type},
      {"passingKind", stringifyPassingKind(passingKind)},
  };
  if (defaultValue)
    object["default"] = *defaultValue;
  return object;
}

//===----------------------------------------------------------------------===//
// PublicAliasDecl
//===----------------------------------------------------------------------===//

std::string
PublicAliasDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "alias " << getName();
  if (!value.empty())
    os << " = " << value;
  return snippet;
}

std::string PublicAliasDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PublicAliasDecl::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"deprecated", deprecated},  {"description", description},
      {"kind", getKindAsString()}, {"name", getName().str()},
      {"summary", summary},        {"value", value}};
}

/// Return if the given alias decl is global, i.e. nested within a module,
/// package, or struct.
static bool isGlobalAliasDecl(MojoASTDeclRef declRef) {
  return isa<FileModuleOp, PackageOp, StructDeclOp>(*declRef->getParentDecl());
}

PublicAliasDecl::PublicAliasDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicAliasDecl,
                 declRef.getName().value_or(StringRef())),
      isGlobalAlias(isGlobalAliasDecl(declRef)),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())) {
  auto aliasOp = cast<LIT::AliasDeclOp>(declRef->getIfOperation());

  auto &shared = *declRef.getShared();
  if (auto maybeValue = aliasOp.getValue())
    value = generatePValueString(shared, maybeValue.value());

  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PublicFunctionDecl
//===----------------------------------------------------------------------===//

void PublicFunctionDecl::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<std::string> pureDescriptionLines;

  for (size_t line = 0, lineEnd = desc.size(); line < lineEnd; ++line) {
    if (desc[line] == (Twine(DocString::kSectionArgs) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineEnd, args);
    } else if (desc[line] ==
               (Twine(DocString::kSectionParameters) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineEnd, parameters);
    } else if (desc[line] == (Twine(DocString::kSectionReturns) + ":").str()) {
      if (returnType)
        returnsDoc = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] ==
               (Twine(DocString::kSectionConstraints) + ":").str()) {
      constraints = parseDocStringSection(desc, line, lineEnd);
    } else if (desc[line] == (Twine(DocString::kSectionRaises) + ":").str()) {
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
    os << "async ";

  unsigned returnOffset = 0;
  std::string signature =
      getSignature(ctx, parameterOffsets, argumentOffsets, &returnOffset);
  StringRef resultLessSignature(signature.data(), returnOffset);
  os << (isDef() ? "def" : "fn") << " ";

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

/// Print the signatures for a list of arguments or parameters.
template <typename T>
static void printArgOrParameterSignature(
    MojoParserContext &ctx, ArrayRef<T> args,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets,
    llvm::raw_string_ostream &os, bool suppressSlashAfterSelf = false) {
  PassingKindPrinter passingKindPrinter(
      os, args.size(), [&](size_t idx) { return args[idx].getPassingKind(); },
      suppressSlashAfterSelf, /*slash=*/'/', /*plus=*/"//");
  size_t idx = 0;
  auto printArg = [&](const T &arg) {
    passingKindPrinter.printOptionalStarSlash(idx);

    unsigned argStart = os.str().size();
    os << arg.getDeclarationSnippet(ctx);
    if (offsets)
      offsets->push_back({argStart, os.str().size()});

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
  };
  os << (std::is_same_v<T, PublicArgumentDecl> ? "(" : "[");
  interleaveComma(args, os, printArg);
  os << (std::is_same_v<T, PublicArgumentDecl> ? ")" : "]");
}

std::string PublicFunctionDecl::getSignature(
    MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets,
    unsigned *returnOffset) const {
  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);

  // Strip off the mangled suffix from the base function name.
  signatureOS << getName().split('(').first;

  // Emit the parameters of the function.
  if (!parameters.empty())
    printArgOrParameterSignature(ctx, ArrayRef(parameters), parameterOffsets,
                                 signatureOS);

  // If this is an initializer with an out argument, we permute the out argument
  // to the start of the argument list to look more conventional.
  ArrayRef<PublicArgumentDecl> args = getArguments();
  SmallVector<PublicArgumentDecl> argTmp;

  bool hasOutArgument =
      !args.empty() &&
      args.back().getConvention() == PublicArgumentDecl::Convention::kOut;

  if (hasOutArgument && isInit) {
    // Make a mutable copy of the arguments so we can change them around.
    argTmp.append(args.begin(), args.end());
    args = argTmp;

    // Avoid weird punctuation in the signature if possible.
    auto passingKind = LIT::PassingKind::PosOrKw;
    if (args.size() != 1) {
      // Rotate arg list so output is the first argument
      std::rotate(argTmp.rbegin(), argTmp.rbegin() + 1, argTmp.rend());
      if (argTmp[1].getPassingKind() == LIT::PassingKind::PosOnly)
        passingKind = LIT::PassingKind::PosOnly;
    }
    argTmp[0].setPassingKind(passingKind);
  }

  // Emit the arguments of the function.
  printArgOrParameterSignature(ctx, args, argumentOffsets, signatureOS,
                               /*suppressSlashAfterSelf=*/isMethodFlag);

  // Emit the result type.
  if (returnOffset)
    *returnOffset = signature.size();
  if (returnType && !hasOutArgument)
    signatureOS << " -> " << *returnType;
  return signatureOS.str();
}

llvm::json::Object PublicFunctionDecl::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"args", toJSONArray(ctx, args)},
      {"async", isAsync()},
      {"constraints", constraints},
      {"deprecated", deprecated},
      {"description", description},
      {"isDef", isDef()},
      {"isStatic", isStatic()},
      {"isImplicitConversion", isImplicitConversion()},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", toJSONArray(ctx, parameters)},
      {"raises", raises()},
      {"raisesDoc", raisesDoc},
      {"returnsDoc", returnsDoc},
      {"returnType", returnType},
      {"signature", getSignature(ctx)},
      {"summary", summary},
  };
}

PublicFunctionDecl::PublicFunctionDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicFunctionDecl,
                 declRef.getName().value_or(StringRef{})),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef{})) {
  auto funcOp = cast<FnOp>(declRef.getIfOperation());
  isStaticFlag = funcOp.getIsStatic();
  isImplicitConversionFlag = funcOp.getIsImplicitConversion();
  isMethodFlag = !isStaticFlag && isa<StructDeclOp>(funcOp->getParentOp());
  isDefFlag = funcOp.isDef();
  isInit = funcOp.getSpecialFunctionInfo().isInitializer();

  initFromSignature(declRef, funcOp.getFuncTypeGenerator(),
                    funcOp.getArgumentTypes(), funcOp.getUserResultType());
}

PublicFunctionDecl::PublicFunctionDecl(MojoASTDeclRef declRef,
                                       FnTypeGeneratorType signature)
    : PublicDecl(PublicDeclKind::DK_PublicFunctionDecl,
                 /*name=*/StringRef()) {
  initFromSignature(declRef, signature, signature.getArguments(),
                    signature.getUserResultType());
}

void PublicFunctionDecl::initFromSignature(MojoASTDeclRef declRef,
                                           FnTypeGeneratorType signature,
                                           ArrayRef<Type> userArgTypes,
                                           Type userResultType) {
  auto &shared = *declRef.getShared();
  raisesFlag = signature.isThrows();
  isAsyncFlag = signature.isAsync();

  ArrayRef<PogMetadataAttr> argPogs = signature.getArgListAttrs().getPogs();
  ArrayRef<Type> sigTypes = signature.getArguments();
  ArrayRef<ArgConvention> argConventions = signature.getArgConventions();
  ArrayRef<Type> paramTypes = signature.getInputParamTypes();

  // If this is a method, grab the expected "Self" type.
  std::optional<ASTType> selfType;
  if (isa<StructDeclOp>(*declRef.getParent()))
    selfType = declRef->getParentDecl()->getTypeDeclSelf();

  // Update param / arg types with decl refs instead of index refs.
  ParameterEvaluator evaluator;

  // Grab the types of the parameters to the function.
  PogListAttr paramListAttr = signature.getParamListAttrs();
  DefaultValueHandler defaultParamHandler(paramListAttr);
  for (size_t parIdx : llvm::seq<size_t>(0, paramTypes.size())) {
    // Add input value here in case of early continue. It's ok to insert before
    // getting rebound attribute for the default value since the signature is
    // legalized separately.
    // If the parameter doesn't have a name (name is empty), it must have been
    // a synthesized parameter. Instead of using the empty string as name (which
    // will appear as missing a name), keep using index references.
    Type reboundType = evaluator.getReboundType(paramTypes[parIdx]);
    StringAttr paramName = signature.getParamName(parIdx);
    if (paramName.getValue().empty()) {
      evaluator.addInputValue(
          KGEN::ParamIndexRefAttr::get(parIdx, reboundType));
    } else {
      evaluator.addInputValue(
          KGEN::ParamDeclRefAttr::get(paramName, reboundType));
    }
    // Ignore implicitly passed parameters.
    PassingKind passingKind = paramListAttr.getPassingKind(parIdx);
    if (passingKind == PassingKind::Implicit)
      continue;
    std::optional<std::string> defaultValue;
    if (TypedAttr defaultAttr = defaultParamHandler.getDefault(parIdx)) {
      TypedAttr reboundDefaultAttr =
          cast<TypedAttr>(evaluator.getReboundAttribute(defaultAttr));
      defaultValue = generatePValueString(shared, reboundDefaultAttr);
    }
    VariadicKind variadicKind =
        signature.getParamListAttrs().getVariadicKind(parIdx);
    parameters.push_back(PublicParameterDecl(
        paramName,
        generateTypeString(shared, reboundType, variadicKind, selfType),
        passingKind, variadicKind, std::move(defaultValue)));
  }

  // Grab the types of the arguments to the function.
  DefaultValueHandler defaultArgHandler(signature.getArgListAttrs());
  for (auto [argIdx, userType, sigType, conventionX, pogAttr] :
       llvm::enumerate(userArgTypes, sigTypes, argConventions, argPogs)) {
    ArgConvention convention = refineConventionForType(userType, conventionX);
    std::optional<std::string> defaultValue;
    if (auto defaultAttr = defaultArgHandler.getDefault(argIdx)) {
      TypedAttr reboundDefaultAttr =
          cast<TypedAttr>(evaluator.getReboundAttribute(defaultAttr));
      defaultValue = generatePValueString(shared, reboundDefaultAttr);
    }

    std::string prefix;
    auto passingKind = pogAttr.getPassingKind();
    auto declConvention = PublicArgumentDecl::Convention::kBorrowed;
    switch (convention) {
    case ArgConvention::ByRefError:
      // Ignored.
      continue;
    case ArgConvention::ByRefResult:
      // by-ref result types gets modeled as the last argument. we don't want to
      // include it in the normal argument list unless it is explicitly named.
      if (pogAttr.getName() == "__result__") // Gross way to detect implicit.
        continue;
      // The passing kind is typically set to "implicit", but we don't want it
      // to print in the signature list with a ?, so switch it to the match the
      // previous or default PassingKind.
      declConvention = PublicArgumentDecl::Convention::kOut;
      if (args.empty())
        passingKind = LIT::PassingKind::PosOrKw;
      else
        passingKind = args.back().getPassingKind();
      break;
    case ArgConvention::ReadReg:
    case ArgConvention::ReadMem:
      break; // already handled.
    case ArgConvention::Mut:
      declConvention = PublicArgumentDecl::Convention::kInOut;
      break;
    case ArgConvention::Ref:
    case ArgConvention::MutRef:
      declConvention = PublicArgumentDecl::Convention::kRef;
      prefix = getRefPrefixAsString(shared, cast<RefType>(sigType), signature,
                                    /*isRefResult*/ false);
      break;
    case ArgConvention::OwnedMem:
    case ArgConvention::OwnedReg:
      declConvention = PublicArgumentDecl::Convention::kOwned;
      break;
    }
    VariadicKind variadicKind =
        signature.getArgListAttrs().getVariadicKind(argIdx);

    bool isSelf = false;
    if (selfType) {
      // Init methods like copyinit have their output argument as 'self'.
      auto fnDecl = dyn_cast_if_present<FnOp>(declRef.getIfOperation());
      if (fnDecl && fnDecl.getSpecialFunctionInfo().hasSelfResult()) {
        isSelf = declConvention == PublicArgumentDecl::Convention::kOut;
      } else {
        // Otherwise, just treat the first arg as self.
        // TODO: This is wrong for static methods.
        isSelf = argIdx == 0;
      }
    }

    Type reboundUserType = evaluator.getReboundType(userType);
    args.push_back(PublicArgumentDecl(
        pogAttr.getName(), std::move(prefix),
        generateTypeString(shared, reboundUserType, variadicKind, selfType,
                           convention),
        passingKind, variadicKind, std::move(defaultValue), declConvention,
        isSelf));
  }

  // Grab the result type, if it's non-none.
  ASTType resultType = signature.getUserResultType();
  assert(resultType && "didn't find a result type?");
  std::string resultPrefix;

  if (!resultType.isNoneType()) {
    std::string str;
    std::optional<ArgConvention> convention;
    // If this is a ref result add the "ref [life, addrspace] "
    // prefix to the specifier.
    if (signature.isRefResult()) {
      convention = ArgConvention::Ref;
      str = "ref " + getRefPrefixAsString(shared, cast<RefType>(resultType),
                                          signature, /*isRefResult*/ true);
    }
    Type reboundUserResultType = evaluator.getReboundType(userResultType);
    str += generateTypeString(shared, reboundUserResultType, VariadicKind::None,
                              selfType, convention);
    returnType = str;
  }

  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PublicStructFieldDecl
//===----------------------------------------------------------------------===//

std::string
PublicStructFieldDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "var ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

std::string PublicStructFieldDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PublicStructFieldDecl::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"summary", summary},
      {"type", type},
  };
}

PublicStructFieldDecl::PublicStructFieldDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicStructFieldDecl,
                 declRef.getName().value_or(StringRef{})) {
  auto fieldOp = cast<StructFieldOp>(declRef.getIfOperation());

  llvm::raw_string_ostream typeOS(type);
  ASTType(fieldOp.getType()).print(typeOS, /*forDiag=*/declRef.getShared());

  if (std::optional<DocString> docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
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
// PublicTraitDecl
//===----------------------------------------------------------------------===//

/// Collect the names of the various parent decls of a decl given its set of
/// canonical traits.
/// TODO: Whenever we support inherited classes/structs, collect those as well.
static void collectParentTraits(MojoParserContext &ctx, MojoASTDeclRef self,
                                SmallVectorImpl<StringRef> &parentTraits,
                                TraitType canonicalTrait) {
  DenseSet<SymbolRefAttr> seenDecls;
  for (SymbolRefAttr symbol : canonicalTrait.getSymbols()) {
    if (!seenDecls.insert(symbol).second)
      continue;
    MojoASTDeclRef decl = ctx.getDecl(TraitType::get(symbol));
    if (!decl || decl == self)
      continue;
    std::optional<StringRef> name = decl.getName();
    if (!name)
      continue;
    if (isa<TraitDeclOp>(*decl))
      parentTraits.push_back(*name);
  };
  llvm::sort(parentTraits);
}

std::string
PublicTraitDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  return "trait " + getName().str();
}

std::string PublicTraitDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PublicTraitDecl::toJSON(MojoParserContext &ctx) const {
  // Ignore some inherited functions.
  auto shouldHideFn = [](FnOp decl, StringRef name) {
    return decl.getInheritedFrom() && name == "__del__";
  };

  auto aliases = extractChildDecls<PublicAliasDecl, AliasDeclOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(*decl, shouldHideFn));

  SmallVector<StringRef> parentTraits;
  collectParentTraits(ctx, decl, parentTraits,
                      cast<TraitDeclOp>(*decl).getCanonicalTrait());

  return llvm::json::Object{
      {"aliases", toJSONArray(ctx, aliases)},
      {"deprecated", deprecated},
      {"description", description},
      {"fields", llvm::json::Array()},
      {"functions", toJSONArray(ctx, functionOverloads)},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parentTraits", llvm::json::Array(parentTraits)},
      {"summary", summary},
  };
}

PublicTraitDecl::PublicTraitDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicTraitDecl,
                 declRef.getName().value_or(StringRef())),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PublicStructDecl
//===----------------------------------------------------------------------===//

void PublicStructDecl::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<std::string>
      pureDescriptionLines; // Change to std::string to own the data
  for (size_t line = 0, lineEnd = desc.size(); line < lineEnd; ++line) {
    if (desc[line] == (Twine(DocString::kSectionParameters) + ":").str())
      augmentDeclsWithDocumentation(desc, line, lineEnd, parameters);
    else if (desc[line] == (Twine(DocString::kSectionConstraints) + ":").str())
      constraints = parseDocStringSection(desc, line, lineEnd);
    else
      // Handle any badly-indented ad-hoc sections
      maybeParseDocStringAdHocSection(pureDescriptionLines, desc, line,
                                      lineEnd);
  }

  SmallVector<StringRef> pureDescriptionLinesRef;
  for (const auto &descLine : pureDescriptionLines) {
    pureDescriptionLinesRef.push_back(StringRef(descLine));
  }
  description = DocString::formatDescription(pureDescriptionLinesRef);
}

std::string
PublicStructDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  return getDeclarationSnippet(ctx, /*parameterOffsets=*/nullptr);
}

std::string PublicStructDecl::getDeclarationSnippet(
    MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "struct " << getName();

  if (!parameters.empty())
    printArgOrParameterSignature(ctx, ArrayRef(parameters), parameterOffsets,
                                 os);

  SmallVector<StringRef> parentTraits;
  collectParentTraits(ctx, decl, parentTraits,
                      cast<StructDeclOp>(*decl).getCanonicalTrait());
  if (!parentTraits.empty()) {
    os << "\n# Traits: ";
    llvm::interleaveComma(parentTraits, os,
                          [&](StringRef token) { os << token; });
  }

  return snippet;
}

std::string PublicStructDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, summary, description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints, constraints);

  return markdown;
}

std::string PublicStructDecl::getSignature(
    MojoParserContext &ctx,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const {
  std::string output;
  llvm::raw_string_ostream os(output);
  os << "struct " << getName();
  if (!parameters.empty())
    printArgOrParameterSignature(ctx, ArrayRef(parameters), parameterOffsets,
                                 os);
  return output;
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

llvm::json::Object PublicStructDecl::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<PublicAliasDecl, AliasDeclOp>(*decl);
  auto fields = extractChildDecls<PublicStructFieldDecl, StructFieldOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(*decl));
  SmallVector<StringRef> parentTraits;
  collectParentTraits(ctx, decl, parentTraits,
                      cast<StructDeclOp>(*decl).getCanonicalTrait());
  return llvm::json::Object{
      {"aliases", toJSONArray(ctx, aliases)},
      {"constraints", constraints},
      {"deprecated", deprecated},
      {"description", description},
      {"fields", toJSONArray(ctx, fields)},
      {"functions", toJSONArray(ctx, functionOverloads)},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", toJSONArray(ctx, parameters)},
      {"parentTraits", llvm::json::Array(parentTraits)},
      {"signature", getSignature(ctx)},
      {"summary", summary},
      {"convention", toString(convention)},
  };
}

PublicStructDecl::PublicStructDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicStructDecl,
                 declRef.getName().value_or(StringRef())),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())),
      decl(declRef) {
  auto structOp = cast<StructDeclOp>(declRef.getIfOperation());
  TypeSignatureType signature = structOp.getSignature();
  convention = structOp.getConvention();

  auto &shared = *declRef.getShared();

  // Update param / arg types with decl refs instead of index refs.
  ParameterEvaluator evaluator;
  // Grab the types of the parameters to the struct.
  PogListAttr paramListAttr = signature.getParamListAttrs();
  DefaultValueHandler defaultParamHandler(paramListAttr);
  for (auto [idx, param] : llvm::enumerate(structOp.getInputParams())) {
    std::optional<std::string> defaultValue;
    if (auto defaultAttr = defaultParamHandler.getDefault(idx)) {
      TypedAttr reboundDefaultAttr =
          cast<TypedAttr>(evaluator.getReboundAttribute(defaultAttr));
      defaultValue = generatePValueString(shared, reboundDefaultAttr);
    }
    VariadicKind variadicKind =
        signature.getParamListAttrs().getVariadicKind(idx);
    StringRef paramName = demangleIfNeeded(param).getName().getValue();
    Type reboundType = evaluator.getReboundType(param.getType());
    parameters.push_back(PublicParameterDecl(
        paramName, generateTypeString(shared, param.getType(), variadicKind),
        paramListAttr.getPassingKind(idx), variadicKind,
        std::move(defaultValue)));
    evaluator.addInputValue(
        KGEN::ParamDeclRefAttr::get(paramName, reboundType));
  }

  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PublicModuleDecl
//===----------------------------------------------------------------------===//

std::string
PublicModuleDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  return {};
}

std::string PublicModuleDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PublicModuleDecl::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<PublicAliasDecl, AliasDeclOp>(*decl);
  auto structs = extractChildDecls<PublicStructDecl, StructDeclOp>(*decl);
  auto traits = extractChildDecls<PublicTraitDecl, TraitDeclOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSet::fromSortedFunctions(
      extractChildDecls<PublicFunctionDecl, FnOp>(*decl));

  return llvm::json::Object{{"aliases", toJSONArray(ctx, aliases)},
                            {"description", description},
                            {"functions", toJSONArray(ctx, functionOverloads)},
                            {"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"structs", toJSONArray(ctx, structs)},
                            {"traits", toJSONArray(ctx, traits)},
                            {"summary", summary}};
}

PublicModuleDecl::PublicModuleDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicModuleDecl,
                 declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PublicPackageDecl
//===----------------------------------------------------------------------===//

std::string
PublicPackageDecl::getDeclarationSnippet(MojoParserContext &ctx) const {
  return {};
}

std::string PublicPackageDecl::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PublicPackageDecl::toJSON(MojoParserContext &ctx) const {
  auto packages = extractChildDecls<PublicPackageDecl, PackageOp>(*decl);
  auto modules = extractChildDecls<PublicModuleDecl, FileModuleOp>(*decl);
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"summary", summary},
      {"modules", toJSONArray(ctx, modules)},
      {"packages", toJSONArray(ctx, packages)},
  };
}

PublicPackageDecl::PublicPackageDecl(MojoASTDeclRef declRef)
    : PublicDecl(PublicDeclKind::DK_PublicPackageDecl,
                 declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}
