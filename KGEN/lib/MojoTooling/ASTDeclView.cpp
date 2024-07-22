//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Two spaces that are forcefully added to markdown lines that can be used for
/// identation.
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

/// Helper to return the variadic kind of a parameter/argument.
static VariadicKind getVariadicKind(PogListAttr pogListAttr, size_t idx) {
  if (pogListAttr.isPack(idx))
    return VariadicKind::kPack;
  if (!pogListAttr.isVariadic(idx))
    return VariadicKind::kNone;
  PassingKind passingKind = pogListAttr.getPassingKind(idx);
  if (passingKind == PassingKind::KwOnly)
    return VariadicKind::kKwVar;
  assert(passingKind == PassingKind::PosOrKw);
  return VariadicKind::kPosVar;
}

/// Generate a user-readable representation of the given type and variadic kind,
/// with an optional value convention, and parent struct "Self" type.
static std::string
generateTypeString(Type type, VariadicKind varKind,
                   std::optional<ASTType> selfType = std::nullopt,
                   std::optional<ArgConvention> convention = std::nullopt) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);
  ASTType astType(type);

  if (varKind == VariadicKind::kPosVar) {
    astType = astType.getVariadicElementType();
  } else if (varKind == VariadicKind::kPack && !isa<PackType>(type)) {
    // VariadicPack needs special printing, because its argument isn't a type.
    os << "*";
    ASTType::printParam(os, astType.getVariadicPackInfo().getVariadic(),
                        /*forDiag=*/true, /*demangleParams=*/true);
    return os.str();
  }

  // Process the convention if present.
  if (convention && SignatureType::hasAddress(*convention)) {
    // In some cases variadics are passed directly (which is a hack, but okay).
    // The ABI in these cases is that we pass a variadic of refs. We leave these
    // as is, since eventually (with unpacking) this hack won't be needed.
    if (!isa<VariadicType>(astType))
      astType = astType.getReferenceElementType();
  }

  // Get the value type in a kwargs dictionary.
  if (varKind == VariadicKind::kKwVar)
    astType = astType.getKwargsDictValueType();

  // If this type is the same as the self type, use the "Self" keyword.
  if (selfType && astType.isEqualCanon(*selfType))
    os << "Self";
  else
    os << astType.getAsString(/*forDiag=*/true, /*demangleParams=*/true);

  return os.str();
}

/// Generate a user-readable representation of the given pvalue.
static std::string generatePValueString(PValue value) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);
  value.printForDiag(os);
  return os.str();
}

/// If the argument/parameter is variadic, we put the star (or two stars if
/// variadic keyword) before the identifier.
static Twine prependVariadicIdentifiers(const Twine &identifier,
                                        VariadicKind varKind) {
  switch (varKind) {
  case VariadicKind::kPosVar:
  case VariadicKind::kPack:
    return "*" + identifier;
  case VariadicKind::kKwVar:
    return "**" + identifier;
  default:
    return identifier;
  }
}

// Helper function that dumps an identifier along with an optional
// type. It also takes care of varargs that need to encode * in the name.
static void dumpIdentifierWithType(raw_ostream &os, StringRef identifier,
                                   StringRef type,
                                   VariadicKind varKind = VariadicKind::kNone) {
  os << prependVariadicIdentifiers(identifier, varKind);
  if (!type.empty())
    os << ": " << type;
}

/// Parse the given docstring lines and augment the provided decls with the
/// appropriate documentation using the description.
template <typename DeclViewT>
static void augmentDeclsWithDocumentation(ArrayRef<StringRef> lines,
                                          size_t &line, size_t lineE,
                                          SmallVector<DeclViewT> &decls) {
  std::string fullArgDesc;
  llvm::raw_string_ostream fullArgDescOS(fullArgDesc);
  DenseMap<StringRef, DeclViewT *> declMap;
  for (auto &decl : decls)
    declMap.try_emplace(decl.getName(), &decl);

  for (++line; line < lineE && !lines[line].empty();) {
    // Extract the argument name and description.
    auto [argName, argDesc] = lines[line].split(':');
    argName = argName.trim();
    argDesc = argDesc.trim();

    fullArgDesc.clear();
    fullArgDescOS << argDesc;

    // Merge in additional description lines that have a larger indentation.
    size_t indent = getIndentationLevel(lines[line]);
    while (++line < lineE && getIndentationLevel(lines[line]) > indent)
      fullArgDescOS << " " << lines[line].trim();

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
                                         size_t &line, size_t lineE) {
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
  while (++line < lineE && getIndentationLevel(lines[line]) >= indent)
    paragraphOS << " " << lines[line].trim();
  return paragraphOS.str();
}

/// Extract a list of direct children decls from a given decl. It omits
/// children whose name start with _, except for special functions that start
/// and end with __. `shouldHideFn` allows for additional filtering of decls to
/// hide.
template <typename DeclViewType, typename OpType>
static SmallVector<DeclViewType, 2>
extractChildDecls(const ASTDecl &decl,
                  function_ref<bool(OpType, StringRef)> shouldHideFn = {}) {
  DenseSet<Operation *> seenOps;
  SmallVector<DeclViewType, 2> children;

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
      children.push_back(cast<DeclViewType>(*MojoASTDeclRef(child).getView()));
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
template <typename DeclViewList>
static void dumpMarkdownDeclListSection(llvm::raw_ostream &os,
                                        StringRef sectionTitle,
                                        const DeclViewList &decls) {
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
// DeclView
//===----------------------------------------------------------------------===//

StringRef DeclView::getKindAsString(DeclViewKind kind) {
  switch (kind) {
  case DeclViewKind::DK_AliasDeclView:
    return "alias";
  case DeclViewKind::DK_ArgumentDeclView:
    return "argument";
  case DeclViewKind::DK_FunctionDeclView:
    return "function";
  case DeclViewKind::DK_ModuleDeclView:
    return "module";
  case DeclViewKind::DK_PackageDeclView:
    return "package";
  case DeclViewKind::DK_ParameterDeclView:
    return "parameter";
  case DeclViewKind::DK_StructDeclView:
    return "struct";
  case DeclViewKind::DK_StructFieldDeclView:
    return "field";
  case DeclViewKind::DK_TraitDeclView:
    return "trait";
  case DeclViewKind::DK_VariableDeclView:
    return "variable";
  }
  llvm_unreachable("invalid kind");
}

StringRef DeclView::getKindAsString() const { return getKindAsString(kind); }

std::string DeclView::getFullMarkdownString() const {
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

  os << llvm::formatv(declarationSnippet, getDeclarationSnippet());
  return buff;
}

//===----------------------------------------------------------------------===//
// VariableDeclView
//===----------------------------------------------------------------------===//

std::string VariableDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "var ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

llvm::json::Object VariableDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"deprecated", deprecated},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"type", type},
  };
}

VariableDeclView::VariableDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_VariableDeclView,
               declRef.getName().value_or(StringRef{})),
      isGlobalVariable(false),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())) {
  TypeSwitch<mlir::Operation *>(declRef.getIfOperation())
      .Case([&](VarDeclOp op) {
        type = declRef.getType().getReferenceElementType().getAsString();
      })
      .Case([&](GlobalVarDeclOp op) {
        type = declRef.getType().getAsString();
        isGlobalVariable = true;
      });
}

//===----------------------------------------------------------------------===//
// ParameterDeclView
//===----------------------------------------------------------------------===//

std::string ParameterDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  dumpIdentifierWithType(os, getName(), type, variadicKind);
  if (defaultValue)
    os << " = " << *defaultValue;
  return buff;
}

std::string ParameterDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object ParameterDeclView::toJSON(MojoParserContext &ctx) const {
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
// ArgumentDeclView
//===----------------------------------------------------------------------===//

std::string ArgumentDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  // Print the convention of the argument, eliding the defaults. In `fn`,
  // borrowed is the default convention. In `def`, owned is the default
  // convention.
  if (isInout())
    os << "inout ";
  if (isOwned())
    os << "owned ";

  dumpIdentifierWithType(os, getName(), type, variadicKind);
  if (defaultValue)
    os << " = " << *defaultValue;
  return buff;
}

std::string ArgumentDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object ArgumentDeclView::toJSON(MojoParserContext &ctx) const {
  StringRef conventions[] = {"borrowed", "inout", "owned"};

  llvm::json::Object object{
      {"description", description},
      {"convention", conventions[static_cast<int>(convention)]},
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
// AliasDeclView
//===----------------------------------------------------------------------===//

std::string AliasDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "alias " << getName();
  if (!value.empty())
    os << " = " << value;
  return snippet;
}

std::string AliasDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object AliasDeclView::toJSON(MojoParserContext &ctx) const {
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

AliasDeclView::AliasDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_AliasDeclView,
               declRef.getName().value_or(StringRef())),
      isGlobalAlias(isGlobalAliasDecl(declRef)),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())) {
  auto aliasOp = cast<LIT::AliasDeclOp>(declRef->getIfOperation());

  llvm::raw_string_ostream valueOS(value);
  PValue(aliasOp.getValue()).printForDiag(valueOS);

  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclView
//===----------------------------------------------------------------------===//

void FunctionDeclView::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<StringRef> pureDescriptionLines;
  for (size_t line = 0, lineE = desc.size(); line < lineE; ++line) {
    if (desc[line] == (Twine(DocString::kSectionArgs) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, args);
    } else if (desc[line] ==
               (Twine(DocString::kSectionParameters) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, parameters);
    } else if (desc[line] == (Twine(DocString::kSectionReturns) + ":").str()) {
      if (returnType)
        returnsDoc = parseDocStringSection(desc, line, lineE);
    } else if (desc[line] ==
               (Twine(DocString::kSectionConstraints) + ":").str()) {
      constraints = parseDocStringSection(desc, line, lineE);
    } else if (desc[line] == (Twine(DocString::kSectionRaises) + ":").str()) {
      if (raises())
        raisesDoc = parseDocStringSection(desc, line, lineE);
    } else {
      pureDescriptionLines.push_back(desc[line]);
    }
  }

  description = DocString::formatDescription(pureDescriptionLines);
}

std::string FunctionDeclView::getDeclarationSnippet() const {
  return getDeclarationSnippet(/*parameterOffsets=*/nullptr,
                               /*argumentOffsets=*/nullptr);
}

std::string FunctionDeclView::getDeclarationSnippet(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  if (isAsync())
    os << "async ";

  unsigned returnOffset = 0;
  std::string signature =
      getSignature(parameterOffsets, argumentOffsets, &returnOffset);
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

std::string FunctionDeclView::getMarkdownDocString() const {
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
    ArrayRef<T> args, SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets,
    llvm::raw_string_ostream &os, bool suppressSlashAfterSelf = false) {
  PassingKindPrinter passingKindPrinter(
      os, args.size(), [&](size_t idx) { return args[idx].getPassingKind(); },
      suppressSlashAfterSelf, /*slash=*/'/', /*plus=*/"//");
  size_t idx = 0;
  auto printArg = [&](const T &arg) {
    passingKindPrinter.printOptionalStarSlash(idx);

    unsigned argStart = os.str().size();
    os << arg.getDeclarationSnippet();
    if (offsets)
      offsets->push_back({argStart, os.str().size()});

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(idx++);
  };
  os << (std::is_same_v<T, ArgumentDeclView> ? "(" : "[");
  interleaveComma(args, os, printArg);
  os << (std::is_same_v<T, ArgumentDeclView> ? ")" : "]");
}

std::string FunctionDeclView::getSignature(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets,
    unsigned *returnOffset) const {
  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);

  // Strip off the mangled suffix from the base function name.
  signatureOS << getName().split('(').first;

  // Emit the parameters of the function.
  if (!parameters.empty())
    printArgOrParameterSignature(ArrayRef(parameters), parameterOffsets,
                                 signatureOS);

  // Emit the arguments of the function.
  printArgOrParameterSignature(ArrayRef(args), argumentOffsets, signatureOS,
                               /*suppressSlashAfterSelf=*/isMethodFlag);

  // Emit the result type.
  if (returnOffset)
    *returnOffset = signature.size();
  if (returnType)
    signatureOS << " -> " << *returnType;
  return signatureOS.str();
}

llvm::json::Object FunctionDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"args", toJSONArray(ctx, args)},
      {"async", isAsync()},
      {"constraints", constraints},
      {"deprecated", deprecated},
      {"description", description},
      {"isDef", isDef()},
      {"isStatic", isStatic()},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", toJSONArray(ctx, parameters)},
      {"raises", raises()},
      {"raisesDoc", raisesDoc},
      {"returnsDoc", returnsDoc},
      {"returnType", returnType},
      {"signature", getSignature()},
      {"summary", summary},
  };
}

FunctionDeclView::FunctionDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_FunctionDeclView,
               declRef.getName().value_or(StringRef{})),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef{})) {
  auto funcOp = cast<LIT::FuncOp>(declRef.getIfOperation());
  isStaticFlag = funcOp.getIsStatic();
  isMethodFlag = !isStaticFlag && isa<StructDeclOp>(funcOp->getParentOp());
  isDefFlag = funcOp.isDef();
  initFromSignature(declRef, funcOp.getSignature(), funcOp.getArgumentTypes());
}

FunctionDeclView::FunctionDeclView(MojoASTDeclRef declRef,
                                   KGEN::LIT::LITSignatureType signature)
    : DeclView(DeclViewKind::DK_FunctionDeclView, /*name=*/StringRef()) {
  initFromSignature(declRef, signature, signature.getArguments());
}

void FunctionDeclView::initFromSignature(MojoASTDeclRef declRef,
                                         LITSignatureType signature,
                                         ArrayRef<Type> argTypes) {
  raisesFlag = signature.isThrows();
  isAsyncFlag = signature.isAsync();

  ArrayRef<PogMetadataAttr> argPogs = signature.getArgListAttrs().getPogs();
  ArrayRef<ArgConvention> argConventions = signature.getArgConventions();
  ArrayRef<Type> paramTypes = signature.getInputParamTypes();

  // Check for a by-ref result type, which gets modeled as the last argument
  // (as it needs to be passed through memory), and we don't want to include
  // it in the normal argument list.
  if (!argConventions.empty() &&
      argConventions.back() == ArgConvention::ByRefResult) {
    argTypes = argTypes.drop_back();
    argPogs = argPogs.drop_back();
    argConventions = argConventions.drop_back();
  }
  if (!argConventions.empty() &&
      argConventions.back() == ArgConvention::ByRefError) {
    argTypes = argTypes.drop_back();
    argPogs = argPogs.drop_back();
    argConventions = argConventions.drop_back();
  }

  // If this is a method, grab the expected "Self" type.
  std::optional<ASTType> selfType;
  if (isa<StructDeclOp>(*declRef.getParentDecl()))
    selfType = declRef->getParentDecl()->getTypeDeclSelf();

  // Grab the types of the arguments to the function.
  DefaultValueHandler defaultArgHandler(signature.getArgListAttrs());
  for (auto [argIdx, type, conventionX, pogAttr] :
       llvm::enumerate(argTypes, argConventions, argPogs)) {
    ArgConvention convention = refineConventionForType(type, conventionX);
    std::optional<std::string> defaultValue;
    if (auto defaultAttr = defaultArgHandler.getDefault(argIdx))
      defaultValue = generatePValueString(defaultAttr);

    auto declConvention = ArgumentDeclView::Convention::kBorrowed;
    if (convention == ArgConvention::InOut ||
        convention == ArgConvention::InitSelf)
      declConvention = ArgumentDeclView::Convention::kInOut;
    else if (convention == ArgConvention::OwnedInMem ||
             convention == ArgConvention::OwnedInReg)
      declConvention = ArgumentDeclView::Convention::kOwned;
    VariadicKind variadicKind =
        getVariadicKind(signature.getArgListAttrs(), argIdx);
    args.push_back(ArgumentDeclView(
        pogAttr.getName(),
        generateTypeString(type, variadicKind, selfType, convention),
        pogAttr.getPassingKind(), variadicKind, std::move(defaultValue),
        declConvention));
  }

  // Grab the types of the parameters to the function.
  PogListAttr paramListAttr = signature.getParamListAttrs();
  DefaultValueHandler defaultParamHandler(paramListAttr);
  for (size_t parIdx : llvm::seq<size_t>(0, paramTypes.size())) {
    // Ignore implicitly passed parameters.
    PassingKind passingKind = paramListAttr.getPassingKind(parIdx);
    if (passingKind == PassingKind::Implicit)
      continue;
    std::optional<std::string> defaultValue;
    if (auto defaultAttr = defaultParamHandler.getDefault(parIdx))
      defaultValue = generatePValueString(defaultAttr);
    VariadicKind variadicKind =
        getVariadicKind(signature.getParamListAttrs(), parIdx);
    parameters.push_back(ParameterDeclView(
        signature.getParamName(parIdx),
        generateTypeString(paramTypes[parIdx], variadicKind, selfType),
        passingKind, variadicKind, std::move(defaultValue)));
  }

  // Grab the result type, if it's non-none.
  ASTType resultType = ASTType(signature).getSignatureUserResultType();
  assert(resultType && "didn't find a result type?");
  std::string resultPrefix;
  if (signature.isRefResult()) {
    auto refType = cast<RefType>(resultType);
    resultPrefix = "ref [" + generatePValueString(refType.getLifetime()) + "] ";
    resultType = refType.getElementType();
  }

  if (!resultType.isNoneType() || !resultPrefix.empty())
    returnType = resultPrefix +
                 generateTypeString(resultType, VariadicKind::kNone, selfType);

  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// StructFieldDeclView
//===----------------------------------------------------------------------===//

std::string StructFieldDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "var ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

std::string StructFieldDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object StructFieldDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"summary", summary},
      {"type", type},
  };
}

StructFieldDeclView::StructFieldDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_StructFieldDeclView,
               declRef.getName().value_or(StringRef{})) {
  auto fieldOp = cast<StructFieldOp>(declRef.getIfOperation());

  llvm::raw_string_ostream typeOS(type);
  ASTType(fieldOp.getType()).print(typeOS, /*forDiag=*/true);

  if (std::optional<DocString> docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclOverloadSetView
//===----------------------------------------------------------------------===//

SmallVector<FunctionDeclOverloadSetView, 2>
FunctionDeclOverloadSetView::fromSortedFunctions(
    SmallVector<FunctionDeclView, 2> &&functions) {
  SmallVector<FunctionDeclOverloadSetView, 2> overloads;
  for (auto &function : functions) {
    if (overloads.empty() ||
        overloads.back().getBaseName() != function.getName())
      overloads.emplace_back(FunctionDeclOverloadSetView(function.getName()));

    overloads.back().append(std::move(function));
  }
  return overloads;
}

llvm::json::Object
FunctionDeclOverloadSetView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"kind", "function"},
                            {"name", baseName},
                            {"overloads", toJSONArray(ctx, functions)}};
}

//===----------------------------------------------------------------------===//
// TraitDeclView
//===----------------------------------------------------------------------===//

/// Collect the names of the various parent types of the given set of type
/// lineages.
/// TODO: Whenever we support inherited classes/structs, collect those as well.
static void collectParentTypes(MojoParserContext &ctx,
                               SmallVectorImpl<StringRef> &parentTraits,
                               ArrayRef<TypeLineageAttr> parentTypes) {
  DenseSet<Type> seenTypes;
  auto addParentType = [&](Type parentType) {
    if (!seenTypes.insert(parentType).second)
      return;
    MojoASTDeclRef decl = ctx.getDecl(parentType);
    if (!decl)
      return;
    std::optional<StringRef> name = decl.getName();
    if (!name)
      return;
    if (isa<TraitDeclOp>(*decl))
      parentTraits.push_back(*name);
  };

  for (TypeLineageAttr parentType : parentTypes) {
    addParentType(parentType.getType());
    for (Type type : parentType.getInheritedFrom())
      addParentType(type);
  }

  llvm::sort(parentTraits);
}

std::string TraitDeclView::getDeclarationSnippet() const {
  return "trait " + getName().str();
}

std::string TraitDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object TraitDeclView::toJSON(MojoParserContext &ctx) const {
  // Ignore some inherited functions.
  auto shouldHideFn = [](FuncOp decl, StringRef name) {
    return decl.getIsInherited() && name == "__del__";
  };

  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl, shouldHideFn));
  SmallVector<StringRef> parentTraits;
  collectParentTypes(ctx, parentTraits,
                     cast<TraitDeclOp>(*decl).getParentTypes());
  return llvm::json::Object{
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

TraitDeclView::TraitDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_TraitDeclView,
               declRef.getName().value_or(StringRef())),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// StructDeclView
//===----------------------------------------------------------------------===//

void StructDeclView::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<StringRef> pureDescriptionLines;
  for (size_t line = 0, lineE = desc.size(); line < lineE; ++line) {
    if (desc[line] == (Twine(DocString::kSectionParameters) + ":").str())
      augmentDeclsWithDocumentation(desc, line, lineE, parameters);
    else if (desc[line] == (Twine(DocString::kSectionConstraints) + ":").str())
      constraints = parseDocStringSection(desc, line, lineE);
    else
      pureDescriptionLines.push_back(desc[line]);
  }

  description = DocString::formatDescription(pureDescriptionLines);
}

std::string StructDeclView::getDeclarationSnippet() const {
  return getDeclarationSnippet(/*parameterOffsets=*/nullptr);
}

std::string StructDeclView::getDeclarationSnippet(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "struct " << getName();

  if (!parameters.empty())
    printArgOrParameterSignature(ArrayRef(parameters), parameterOffsets, os);

  return snippet;
}

std::string StructDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, summary, description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints, constraints);

  return markdown;
}

llvm::json::Object StructDeclView::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(*decl);
  auto fields = extractChildDecls<StructFieldDeclView, StructFieldOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl));
  SmallVector<StringRef> parentTraits;
  collectParentTypes(ctx, parentTraits,
                     cast<StructDeclOp>(*decl).getParentTypes());
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
      {"summary", summary},
  };
}

StructDeclView::StructDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_StructDeclView,
               declRef.getName().value_or(StringRef())),
      deprecated(declRef.getDeprecationWarning().value_or(StringRef())),
      decl(declRef) {
  auto structOp = cast<StructDeclOp>(declRef.getIfOperation());
  TypeSignatureType signature = structOp.getSignature();

  // Grab the types of the parameters to the struct.
  PogListAttr paramListAttr = signature.getParamListAttrs();
  DefaultValueHandler defaultParamHandler(paramListAttr);
  for (auto [idx, param] : llvm::enumerate(structOp.getInputParams())) {
    std::optional<std::string> defaultValue;
    if (auto defaultAttr = defaultParamHandler.getDefault(idx))
      defaultValue = generatePValueString(defaultAttr);
    VariadicKind variadicKind =
        getVariadicKind(signature.getParamListAttrs(), idx);
    parameters.push_back(
        ParameterDeclView(demangleIfNeeded(param).getName().getValue(),
                          generateTypeString(param.getType(), variadicKind),
                          paramListAttr.getPassingKind(idx), variadicKind,
                          std::move(defaultValue)));
  }

  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// ModuleDeclView
//===----------------------------------------------------------------------===//

std::string ModuleDeclView::getDeclarationSnippet() const { return {}; }

std::string ModuleDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object ModuleDeclView::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(*decl);
  auto structs = extractChildDecls<StructDeclView, StructDeclOp>(*decl);
  auto traits = extractChildDecls<TraitDeclView, TraitDeclOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl));

  return llvm::json::Object{{"aliases", toJSONArray(ctx, aliases)},
                            {"description", description},
                            {"functions", toJSONArray(ctx, functionOverloads)},
                            {"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"structs", toJSONArray(ctx, structs)},
                            {"traits", toJSONArray(ctx, traits)},
                            {"summary", summary}};
}

ModuleDeclView::ModuleDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_ModuleDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// PackageDeclView
//===----------------------------------------------------------------------===//

std::string PackageDeclView::getDeclarationSnippet() const { return {}; }

std::string PackageDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PackageDeclView::toJSON(MojoParserContext &ctx) const {
  auto packages = extractChildDecls<PackageDeclView, PackageOp>(*decl);
  auto modules = extractChildDecls<ModuleDeclView, FileModuleOp>(*decl);
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"summary", summary},
      {"modules", toJSONArray(ctx, modules)},
      {"packages", toJSONArray(ctx, packages)},
  };
}

PackageDeclView::PackageDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_PackageDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = DocString::formatDescription(docStr->getDescription());
  }
}
