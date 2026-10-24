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

#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITInterfaces.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DocString.h"
#include "Mojo/lib/MojoParser/Traits.h"
#include "MojoUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

static FnTypeGeneratorType getToolingSignature(ASTDecl *decl) {
  if (!decl)
    return nullptr;
  if (auto functionDecl = dyn_cast<FnSignatureASTDecl>(decl))
    return functionDecl->getSignature();
  return nullptr;
}

static ParamDeclRefAttr getIfParameter(ASTDecl &decl) {
  if (auto val = decl.getIfIRValue().getIfPValue())
    return dyn_cast<ParamDeclRefAttr>(val.get());
  return {};
}

static Operation *getDefiningOpFromIR(ASTDecl &decl) {
  if (Value val = decl.getIfIRValue().getMlirValue())
    return val.getDefiningOp();
  return nullptr;
}

ASTType ASTDecl::getToolingType() {
  if (auto *varDecl = dyn_cast<VarASTDecl>(this))
    return varDecl->getOp().getType();
  if (auto *defDecl = dyn_cast<DefASTDecl>(this))
    return defDecl->getOp().getFullSignature();
  if (auto *functionTypeDecl = dyn_cast<FunctionTypeASTDecl>(this))
    return functionTypeDecl->getSignature();
  if (auto *structDecl = dyn_cast<StructASTDecl>(this))
    return StructASTDecl::computeSelfType(structDecl->getOp());
  if (auto *traitDecl = dyn_cast<TraitASTDecl>(this))
    return TraitASTDecl::computeSelfType(traitDecl->getOp());
  return {};
}

/// Returns this argument's ABI convention. Asserts if this is not an
/// argument declaration.
ArgConvention ASTDecl::getArgConvention() const {
  ssize_t argNumber = getArgNumber();
  assert(argNumber >= 0 && "argument convention requested for non-argument");
  auto *parent = cast<FnSignatureASTDecl>(getParentDecl());
  FnTypeGeneratorType signature = parent->getSignature();
  assert(signature && "function argument must have a resolved signature");
  assert(static_cast<size_t>(argNumber) < signature.getNumArguments() &&
         "argument number is out of range");
  return signature.getArgConventions()[argNumber];
}

/// Returns whether this argument appears in the user-visible signature.  This
/// filters out implicitly declared error slots and result slots, but keeps
/// user-declared "out" argument slots.
bool ASTDecl::isUserArgument() const {
  ArgConvention convention = getArgConvention();
  if (convention == ArgConvention::ByRefError)
    return false;
  // Result slots are considered user arguments if they were explicitly
  // declared.
  if (convention != ArgConvention::ByRefResult)
    return true;

  auto *parent = cast<FnSignatureASTDecl>(getParentDecl());
  return parent->getSignature().getArgName(getArgNumber()) != "__result__";
}

std::optional<StringRef> ASTDecl::getName() {
  auto getFromOp = [&](Operation *op) -> std::optional<StringRef> {
    if (!op)
      return std::nullopt;
    return TypeSwitch<Operation &, std::optional<StringRef>>(*op)
        .Case<StructDeclOp, StructFieldOp, VarDeclOp>(
            [](auto op) { return op.getName(); })
        .Case([](FnOp op) { return op.getSourceName(); })
        .Case<FileModuleOp, PackageOp>([](auto op) { return op.getDeclName(); })
        .Case([](ImportOp op) { return op.getSymName(); })
        .Case([](AliasDeclOp op) {
          return demangleParameterName(op.getParamDecl().getName(),
                                       /*forUser=*/true);
        })
        .Case([](ASTDeclInterface op) { return op.getDeclName(); })
        .Default({});
  };

  if (auto name = getFromOp(getIfOperation()))
    return name;

  if (getArgNumber() >= 0) {
    FnTypeGeneratorType signature = getToolingSignature(getParentDecl());
    size_t argNumber = static_cast<size_t>(getArgNumber());
    if (signature && argNumber < signature.getNumArguments())
      return signature.getArgName(argNumber);
  }

  if (getParamNumber() >= 0) {
    auto *parentDecl = dyn_cast<DefASTDecl>(getParentDecl());
    size_t paramNumber = static_cast<size_t>(getParamNumber());
    if (parentDecl && paramNumber < parentDecl->getOp().getParams().size())
      return demangleParameterName(
          parentDecl->getOp().getParams()[paramNumber].getName(),
          /*forUser=*/true);
  }

  if (auto irDecl = dyn_cast<IRASTDecl>(this);
      irDecl && irDecl->getCapturedName())
    return irDecl->getCapturedName().getValue();

  if (auto paramRef = getIfParameter(*this))
    return demangleParameterName(paramRef.getName(), /*forUser=*/true);

  return getFromOp(getDefiningOpFromIR(*this));
}

std::optional<StringRef> ASTDecl::getDeprecationWarning() {
  if (auto stabilityItf =
          dyn_cast<StabilityDecoratorInterface>(getIfOperation()))
    if (StringAttr attr = stabilityItf.getDeprecationWarningAttr())
      return attr.getValue();
  return {};
}

ASTDecl::ASTDecl(SharedState &shared, llvm::SMLoc loc, ASTDecl *parentDecl,
                 LexerCursor cursor, LexerCursor endCursor, ssize_t indentation)
    : ASTDecl(Kind::Base, shared, loc, parentDecl, cursor, endCursor,
              indentation) {}

ASTDecl::ASTDecl(Kind declKind, SharedState &shared, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation)
    : shared(shared), loc(loc), parentDecl(parentDecl), cursor(cursor),
      endCursorState(endCursor.getState()), indentation(indentation),
      kind(static_cast<unsigned char>(declKind)) {
  resolvedness = DeclResolvedness::unparsed;
  referencedFromBytecode = false;
  hasReferenceError = false;
  hasBodyDecorators = false;
  loadedFromBytecode = false;
  isExplicitParamScope = false;
  retired = false;
}

ASTDecl::~ASTDecl() = default;

OpASTDecl::OpASTDecl(Kind kind, SharedState &shared, Operation &operation,
                     llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation)
    : ASTDecl(kind, shared, loc, parentDecl, cursor, endCursor, indentation),
      operation(&operation) {}

IRASTDecl::IRASTDecl(SharedState &shared, CValue value, llvm::SMLoc loc,
                     ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation)
    : ASTDecl(Kind::IR, shared, loc, parentDecl, cursor, endCursor,
              indentation),
      value(std::move(value)), argOrParamNumber(-1),
      isParameter(this->value.getIfPValue()) {}

void DefASTDecl::createIRSignatureDecls(DeclResolver &resolver) {
  assert(getParamDecls().empty() && getArgDecls().empty() &&
         "signature declarations already created");
  FnOp fnOp = getOp();
  FnTypeGeneratorType signature = fnOp.getFuncTypeGenerator();

  SmallVector<ASTDecl *> newParamDecls;
  ArrayRef<ParamDeclAttr> params =
      fnOp.getParams().drop_back(signature.getNumImplicitOriginDecls());
  for (size_t paramIndex : llvm::seq(params.size())) {
    auto &paramDecl =
        resolver.addFullyResolvedDecl(CValue(), StringAttr(), getLoc(), this);
    paramDecl.setArgOrParamNumber(static_cast<ssize_t>(paramIndex),
                                  /*newIsParameter=*/true);
    newParamDecls.push_back(&paramDecl);
  }
  setParamDecls(newParamDecls);

  SmallVector<ASTDecl *> newArgDecls;
  for (size_t argIndex : llvm::seq(signature.getNumArguments())) {
    auto convention = signature.getArgConventions()[argIndex];
    auto argName = signature.getArgListAttrs().getPogs()[argIndex].getName();
    if (convention == ArgConvention::ByRefError ||
        (convention == ArgConvention::ByRefResult && argName == "__result__"))
      continue;
    auto &argDecl =
        resolver.addFullyResolvedDecl(CValue(), StringAttr(), getLoc(), this);
    argDecl.setArgOrParamNumber(static_cast<ssize_t>(argIndex),
                                /*newIsParameter=*/false);
    newArgDecls.push_back(&argDecl);
  }
  setArgDecls(newArgDecls);
}

FnSignatureASTDecl::FnSignatureASTDecl(Kind kind, SharedState &shared,
                                       llvm::SMLoc loc, ASTDecl *parentDecl,
                                       LexerCursor cursor,
                                       LexerCursor endCursor,
                                       ssize_t indentation)
    : ASTDecl(kind, shared, loc, parentDecl, cursor, endCursor, indentation) {}

void FnSignatureASTDecl::setParamDecls(ArrayRef<ASTDecl *> decls) {
  paramDecls = shared.persistentAllocator.internElements(decls);
}

void FnSignatureASTDecl::setArgDecls(ArrayRef<ASTDecl *> decls) {
  argDecls = shared.persistentAllocator.internElements(decls);
}

/// Return a list of user-declared argument declarations that we'd want to
/// show in document generation, for example.  This filters out implicitly
/// declared error slots and result slots, but keeps user-declared "out"
/// argument slots.
SmallVector<ASTDecl *> FnSignatureASTDecl::getUserArgDecls() const {
  SmallVector<ASTDecl *> result;
  for (ASTDecl *decl : getArgDecls())
    if (decl->isUserArgument())
      result.push_back(decl);
  return result;
}

FunctionTypeASTDecl::FunctionTypeASTDecl(SharedState &shared, llvm::SMLoc loc,
                                         ASTDecl *parentDecl,
                                         LexerCursor cursor,
                                         LexerCursor endCursor,
                                         ssize_t indentation)
    : FnSignatureASTDecl(Kind::FnSignature, shared, loc, parentDecl, cursor,
                         endCursor, indentation) {}

void FunctionTypeASTDecl::setSignature(FnTypeGeneratorType newSignature,
                                       ArrayRef<ASTDecl *> params,
                                       ArrayRef<ASTDecl *> args) {
  signature = newSignature;
  setParamDecls(params);
  setArgDecls(args);
}

OtherOpASTDecl::OtherOpASTDecl(SharedState &shared, Operation &op,
                               llvm::SMLoc loc, ASTDecl *parentDecl,
                               LexerCursor cursor, LexerCursor endCursor,
                               ssize_t indentation)
    : OpASTDecl(Kind::OtherOp, shared, op, loc, parentDecl, cursor, endCursor,
                indentation) {}

VarASTDecl::VarASTDecl(SharedState &shared, VarDeclOp varOp, llvm::SMLoc loc,
                       ASTDecl *parentDecl, LexerCursor cursor,
                       LexerCursor endCursor, ssize_t indentation)
    : OpASTDecl(Kind::Var, shared, *varOp.getOperation(), loc, parentDecl,
                cursor, endCursor, indentation),
      argNumber(-1) {}

ModuleASTDecl::ModuleASTDecl(Kind kind, SharedState &shared,
                             Operation &moduleOp, llvm::SMLoc loc,
                             ASTDecl *parentDecl, LexerCursor cursor,
                             LexerCursor endCursor, ssize_t indentation)
    : OpASTDecl(kind, shared, moduleOp, loc, parentDecl, cursor, endCursor,
                indentation) {}

LogicalResult ModuleASTDecl::resolveSignature(Lexer &) { return success(); }

LogicalResult ModuleASTDecl::resolveSyntheticSignature() { return success(); }

LogicalResult ModuleASTDecl::resolveBody(Lexer &) { return success(); }

LogicalResult ModuleASTDecl::resolveSyntheticBody() { return success(); }

ImportPathAttr ModuleASTDecl::getAbsoluteModuleName() const {
  SymbolRefAttr ref = getSymbolRef();
  if (!ref)
    return {};
  SmallVector<StringRef> components;
  components.push_back(ref.getRootReference().getValue());
  for (FlatSymbolRefAttr nested : ref.getNestedReferences())
    components.push_back(nested.getValue());
  return ImportPathAttr::get(getContext(), /*relativeLevel=*/0, components);
}

TopLevelModuleASTDecl::TopLevelModuleASTDecl(SharedState &shared,
                                             ModuleOp moduleOp, llvm::SMLoc loc,
                                             ASTDecl *parentDecl,
                                             LexerCursor cursor,
                                             LexerCursor endCursor,
                                             ssize_t indentation)
    : ModuleASTDecl(Kind::TopLevelModule, shared, *moduleOp.getOperation(), loc,
                    parentDecl, cursor, endCursor, indentation) {}

FileModuleASTDecl::FileModuleASTDecl(SharedState &shared, FileModuleOp moduleOp,
                                     llvm::SMLoc loc, ASTDecl *parentDecl,
                                     LexerCursor cursor, LexerCursor endCursor,
                                     ssize_t indentation)
    : ModuleASTDecl(Kind::FileModule, shared, *moduleOp.getOperation(), loc,
                    parentDecl, cursor, endCursor, indentation) {}

PackageASTDecl::PackageASTDecl(SharedState &shared, PackageOp packageOp,
                               llvm::SMLoc loc, ASTDecl *parentDecl,
                               LexerCursor cursor, LexerCursor endCursor,
                               ssize_t indentation)
    : ModuleASTDecl(Kind::Package, shared, *packageOp.getOperation(), loc,
                    parentDecl, cursor, endCursor, indentation) {}

DefASTDecl::DefASTDecl(SharedState &shared, FnOp fnOp, llvm::SMLoc loc,
                       ASTDecl *parentDecl, LexerCursor cursor,
                       LexerCursor endCursor, ssize_t indentation)
    : FnSignatureASTDecl(Kind::Def, shared, loc, parentDecl, cursor, endCursor,
                         indentation),
      operation(fnOp) {}

TypeScopeASTDecl::TypeScopeASTDecl(Kind kind, SharedState &shared,
                                   Operation &op, llvm::SMLoc loc,
                                   ASTDecl *parentDecl, LexerCursor cursor,
                                   LexerCursor endCursor, ssize_t indentation)
    : OpASTDecl(kind, shared, op, loc, parentDecl, cursor, endCursor,
                indentation) {}

TypeASTDecl::TypeASTDecl(Kind kind, SharedState &shared, Operation &op,
                         llvm::SMLoc loc, ASTDecl *parentDecl,
                         LexerCursor cursor, LexerCursor endCursor,
                         ssize_t indentation)
    : TypeScopeASTDecl(kind, shared, op, loc, parentDecl, cursor, endCursor,
                       indentation) {}

ExtensionASTDecl::ExtensionASTDecl(SharedState &shared,
                                   ExtensionDeclOp extensionOp, llvm::SMLoc loc,
                                   ASTDecl *parentDecl, LexerCursor cursor,
                                   LexerCursor endCursor, ssize_t indentation)
    : TypeScopeASTDecl(Kind::Extension, shared, *extensionOp.getOperation(),
                       loc, parentDecl, cursor, endCursor, indentation) {}

StructASTDecl::StructASTDecl(SharedState &shared, StructDeclOp structOp,
                             llvm::SMLoc loc, ASTDecl *parentDecl,
                             LexerCursor cursor, LexerCursor endCursor,
                             ssize_t indentation)
    : TypeASTDecl(Kind::Struct, shared, *structOp.getOperation(), loc,
                  parentDecl, cursor, endCursor, indentation) {}

TraitASTDecl::TraitASTDecl(SharedState &shared, TraitDeclOp traitOp,
                           llvm::SMLoc loc, ASTDecl *parentDecl,
                           LexerCursor cursor, LexerCursor endCursor,
                           ssize_t indentation)
    : TypeASTDecl(Kind::Trait, shared, *traitOp.getOperation(), loc, parentDecl,
                  cursor, endCursor, indentation) {}

AliasASTDecl::AliasASTDecl(SharedState &shared, AliasDeclOp aliasOp,
                           llvm::SMLoc loc, ASTDecl *parentDecl,
                           LexerCursor cursor, LexerCursor endCursor,
                           ssize_t indentation)
    : OpASTDecl(Kind::Alias, shared, *aliasOp.getOperation(), loc, parentDecl,
                cursor, endCursor, indentation) {}

ConformanceASTDecl::ConformanceASTDecl(SharedState &shared,
                                       ConformanceOp conformanceOp,
                                       llvm::SMLoc loc, ASTDecl *parentDecl,
                                       LexerCursor cursor,
                                       LexerCursor endCursor,
                                       ssize_t indentation)
    : OpASTDecl(Kind::Conformance, shared, *conformanceOp.getOperation(), loc,
                parentDecl, cursor, endCursor, indentation) {}

ImportASTDecl::ImportASTDecl(SharedState &shared, Operation &importOp,
                             llvm::SMLoc loc, ASTDecl *parentDecl,
                             LexerCursor cursor, LexerCursor endCursor,
                             ssize_t indentation)
    : OpASTDecl(Kind::Import, shared, importOp, loc, parentDecl, cursor,
                endCursor, indentation) {
  assert((isa<ImportOp, UnresolvedImportOp>(importOp)) &&
         "expected an import operation");
}

EnumCaseASTDecl::EnumCaseASTDecl(SharedState &shared, EnumCaseOp caseOp,
                                 llvm::SMLoc loc, ASTDecl *parentDecl,
                                 LexerCursor cursor, LexerCursor endCursor,
                                 ssize_t indentation)
    : OpASTDecl(Kind::EnumCase, shared, *caseOp.getOperation(), loc, parentDecl,
                cursor, endCursor, indentation) {}

StructFieldASTDecl::StructFieldASTDecl(SharedState &shared,
                                       StructFieldOp fieldOp, llvm::SMLoc loc,
                                       ASTDecl *parentDecl, LexerCursor cursor,
                                       LexerCursor endCursor,
                                       ssize_t indentation)
    : OpASTDecl(Kind::StructField, shared, *fieldOp.getOperation(), loc,
                parentDecl, cursor, endCursor, indentation) {}

WitnessASTDecl::WitnessASTDecl(SharedState &shared,
                               UnresolvedDecls unresolvedDecls,
                               TraitSymbolAttr traitSymbol, llvm::SMLoc loc,
                               ASTDecl *parentDecl, LexerCursor cursor,
                               LexerCursor endCursor, ssize_t indentation)
    : ASTDecl(Kind::Witness, shared, loc, parentDecl, cursor, endCursor,
              indentation),
      storage(std::move(unresolvedDecls)), traitSymbol(traitSymbol) {}

LogicalResult ASTDecl::resolveSignature(Lexer &lexer) {
  auto &resolver = shared.getDeclResolver();
  if (getIfOperation()) {
    llvm_unreachable("do not know how to resolve this decl signature");
  }

  if (auto traitType = dyn_cast_or_null<TraitType>(getIfTypeValue()))
    return success(!failed(resolver.resolveSignature(traitType, *this)));

  llvm_unreachable("do not know how to resolve this decl signature");
}

LogicalResult ASTDecl::resolveSyntheticSignature() {
  auto &resolver = shared.getDeclResolver();
  if (!getIfOperation())
    if (auto traitType = dyn_cast_or_null<TraitType>(getIfTypeValue()))
      return success(!failed(resolver.resolveSignature(traitType, *this)));

  Lexer lexer(getShared().diags, getCursor());
  LogicalResult result = resolveSignature(lexer);
  getCursor() = lexer.getCursor();
  return result;
}

LogicalResult ASTDecl::resolveBody(Lexer &lexer) {
  auto &resolver = shared.getDeclResolver();
  if (getIfOperation()) {
    llvm_unreachable("do not know how to resolve this decl body");
  }

  if (auto traitType = dyn_cast_or_null<TraitType>(getIfTypeValue()))
    return success(!failed(resolver.resolveBody(traitType, *this)));

  llvm_unreachable("do not know how to resolve this decl body");
}

LogicalResult ASTDecl::resolveSyntheticBody() {
  auto &resolver = shared.getDeclResolver();
  if (auto traitType = dyn_cast_or_null<TraitType>(getIfTypeValue())) {
    return success(!failed(resolver.resolveBody(traitType, *this)));
  }

  llvm_unreachable("do not know how to resolve this synthetic decl body");
}

bool ASTDecl::isCallableDecl() const {
  if (auto *witness = dyn_cast<WitnessASTDecl>(this))
    return isa<FnTypeGeneratorType>(witness->getWitnessEntry().witnessType);
  return isa<DefASTDecl>(this);
}

FnTypeGeneratorType
ASTDecl::getDeclFullSignature(ASTType contextualType) const {
  if (auto *witness = dyn_cast<WitnessASTDecl>(this)) // already expanded.
    return cast<FnTypeGeneratorType>(witness->getWitnessEntry().witnessType);
  FnOp fnOp = cast<DefASTDecl>(this)->getOp();
  auto genAttr = sugarDynCastIfPresent<GeneratorAttr>(PValue(contextualType));
  if (!genAttr)
    return fnOp.getFullSignature();

  // The struct's parameters come from the contextual generator, e.g. `A` in
  // `comptime A[T: AnyType] = S[0, T]`. Build the full signature with the
  // generator's inputs parameter.
  auto structBody = sugarCast<StructType>(ASTType(genAttr.getBody()));
  // Name the generator's inputs, the name does not matter, it is shortlived.
  ParamRefRemapper nameRemapper;
  SmallVector<ParamDeclAttr> genDecls;
  for (auto [i, type] : llvm::enumerate(genAttr.getInputParamTypes())) {
    auto decl = ParamDeclAttr::get(("Gen#" + Twine(i)).str(),
                                   nameRemapper.replace(type));
    nameRemapper.appendParamDecl(decl);
    genDecls.push_back(decl);
  }

  // The method's own signature refers to the struct's parameters by name;
  // replace them with the struct body's values.
  auto structDecl = cast<DeclInterface>(fnOp->getParentOp());
  ArrayRef<ParamDeclAttr> structParams = structDecl.getInputParams();
  ArrayRef<TypedAttr> structValues = structBody.getParamValues();
  assert(structParams.size() == structValues.size() &&
         "struct body doesn't bind every struct parameter");
  SmallVector<TypedAttr> namedStructValues;
  for (TypedAttr value : structValues)
    namedStructValues.push_back(cast<TypedAttr>(nameRemapper.replace(value)));
  ParameterEvaluator evaluator(structParams, namedStructValues);

  // Get the fnSig name, but with name remapped to the generator's parameter.
  auto fnSig = cast<FnTypeGeneratorType>(
      evaluator.getReboundType(fnOp.getFuncTypeGenerator()));

  SmallVector<StringAttr> genPogNames;
  SmallVector<TypedAttr> genDefaults;
  for (PogMetadataAttr pog : genAttr.getMetadata().getPogs()) {
    genPogNames.push_back(pog.getName());
    genDefaults.push_back(pog.getDefaultValue());
  }

  return FnTypeGeneratorType::prependParams(fnSig, genDecls, genPogNames,
                                            genDefaults);
}

bool ASTDecl::isStaticMethodDecl() const {
  if (auto *witness = dyn_cast<WitnessASTDecl>(this))
    return witness->getWitnessEntry().isStaticMethod;
  return cast<DefASTDecl>(this)->getOp().getIsStatic();
}

ImplicitConversionKind ASTDecl::getDeclImplicitConversionKind() const {
  if (auto *witness = dyn_cast<WitnessASTDecl>(this))
    return witness->getWitnessEntry().implicitConversion;
  return cast<DefASTDecl>(this)->getOp().getImplicitConversion();
}

DocStringAttr ASTDecl::getDocString() const {
  if (auto astDeclOp = dyn_cast_or_null<ASTDeclInterface>(getIfOperation()))
    return astDeclOp.getDocStringAttr();
  return {};
}

void ASTDecl::setErroneous() { hasReferenceError = true; }

std::optional<DocString> ASTDecl::getParsedDocString() const {
  if (auto rawDocStr = getDocString())
    return DocString(rawDocStr);
  return {};
}

std::string ASTDecl::getDocumentationSummary() const {
  if (auto docString = getParsedDocString())
    return docString->getSummary().str();
  return {};
}

std::string ASTDecl::getDocumentationDescription() const {
  if (auto docString = getParsedDocString())
    return DocString::formatDescription(docString->getDescription());
  return {};
}

/// Given a reference to a parameter, look at this declaration and enclosing
/// scopes to find the ASTDecl that defines it (e.g. the enclosing function,
/// struct or trait).  This can return null if not found.
std::tuple<const ASTDecl *, ArrayRef<ParamDeclAttr>, size_t>
ASTDecl::lookupParamReference(ParamDeclRefAttr paramRef) const {
  auto paramDecl = ParamDeclAttr::get(paramRef);

  const ASTDecl *current = this;
  for (; current; current = current->getParentDecl()) {
    if (auto op = current->getIfOperation()) {
      if (auto declIntf = dyn_cast<DeclInterface>(op)) {
        for (auto [idx, param] : llvm::enumerate(declIntf.getInputParams())) {
          if (param.getName() == paramDecl.getName() &&
              param.getType() == paramDecl.getType())
            return {current, declIntf.getInputParams(), idx};
        }
      }
    }
  }
  return {nullptr, {}, ~size_t(0)}; // Not found.
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringRef name) const {
  return lookupInCurrentScope(StringAttr::get(getContext(), name));
}

ArrayRef<ASTDecl *> ASTDecl::lookupInCurrentScope(StringAttr name) const {
  assert(resolvedness == DeclResolvedness::body &&
         "cannot perform lookup in a decl that isn't fully resolved");
  if (!declsInScope)
    return {};

  auto it = declsInScope->find(name);
  if (it != declsInScope->end() && !it->second.empty())
    return it->second;
  return {};
}

/// If this is a method of a struct, trait, or extension, return its parent.
TypeScopeASTDecl *DefASTDecl::tryGetMethodParentDecl() const {
  // Don't return non-null for nested functions or module-level functions.
  return dyn_cast_if_present<TypeScopeASTDecl>(getParentDecl());
}

void StructASTDecl::setParamDecls(ArrayRef<ASTDecl *> decls) {
  paramDecls = shared.persistentAllocator.internElements(decls);
}

ASTDecl *StructASTDecl::getParamByName(StringAttr name) const {
  for (ASTDecl *paramDecl : paramDecls) {
    auto paramValue = paramDecl->getIfIRValue().getIfPValue();
    if (paramValue && isa<ParamDeclRefAttr>(paramValue.get()) &&
        cast<ParamDeclRefAttr>(paramValue.get()).getName() == name)
      return paramDecl;
  }
  return nullptr;
}

/// Collect the struct/trait declaration and all visible extension declarations
/// for the given type from this use-site context.
llvm::SmallVector<ASTDecl *, 4>
ASTDecl::collectTypeAndExtensions(ASTType type, llvm::SMLoc callLoc) {
  SharedState &shared = getShared();
  auto astDecl = type.getDecl(shared);

  SmallVector<ASTDecl *, 4> result;
  if (astDecl)
    result.push_back(astDecl);

  // Handle both struct and trait types for extension lookup.
  if (!isa_and_nonnull<TypeASTDecl>(astDecl) ||
      !astDecl->getUserNameIfOperation())
    return result;

  // Now find all extensions that target this struct/trait.
  StringAttr extensionName =
      shared.getExtensionName(astDecl->getUserNameIfOperation().value());
  LookupAllResult lookupResult = shared.lookupAllDeclsWithName(
      extensionName.getValue(), callLoc, *this, true);

  // Only consider results from successful lookups. Lookups with isErroneous()
  // means the error was already diagnosed. Lookups with isFailure() should
  // be rare since we expect to find at least the original type declaration,
  // but we gracefully handle it by skipping extension lookup.
  if (!lookupResult.isSuccess())
    return result;

  ArrayRef<ASTDecl *> foundAstDecls = lookupResult.getIfSuccess();
  SymbolRefAttr typeSymbol = astDecl->getSymbolRef();
  for (ASTDecl *foundAstDecl : foundAstDecls) {
    if (failed(shared.declResolver->resolveBody(*foundAstDecl, callLoc))) {
      // Do nothing, skip it. Errors were already printed out, and we don't
      // mind missing call candidates from erroneous extensions.
      continue;
    }

    // The bucket is keyed by the target's leaf name only, so extensions of
    // distinct types that share a name land together; keep only those whose
    // resolved target is this type's own decl, as
    // findExtensionsInScopeForStruct does. An import re-exports every
    // extension visible in its source module, so one extension can be found in
    // several enclosing scopes; count it once, or its methods tie with
    // themselves in overload resolution.
    auto *extensionDecl = dyn_cast<ExtensionASTDecl>(foundAstDecl);
    if (extensionDecl &&
        extensionDecl->getOp().getTargetStructAttr() == typeSymbol &&
        !llvm::is_contained(result, foundAstDecl))
      result.push_back(foundAstDecl);
  }

  return result;
}

void ASTDecl::takeDecls(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  for (auto &[name, children] : src.getDeclsInScope())
    for (ASTDecl *child : children)
      child->parentDecl = this;
  declsInScope = std::move(src.declsInScope);
  counter = src.counter;
  assert(!knownAssumptions && "assumptions should be empty before overwriting");
  knownAssumptions = std::move(src.knownAssumptions);
}

/// Append the children decls of `src` into this scope (updating their
/// `parentDecl`), without replacing existing names already in this scope.
/// Used when promoting temporary pattern-binding scopes into a case scope.
void ASTDecl::mergeDeclsFrom(ASTDecl &src) {
  if (src.isErroneous())
    setErroneous();
  if (!src.declsInScope)
    return;
  if (!declsInScope)
    declsInScope.reset(new DeclInScopeType());
  for (auto &[name, children] : *src.declsInScope) {
    TinyPtrVector<ASTDecl *> &entries = (*declsInScope)[name];
    for (ASTDecl *child : children) {
      child->parentDecl = this;
      entries.push_back(child);
    }
  }
  src.declsInScope.reset();
}

DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>> *
TypeScopeASTDecl::getTraitConformanceLineage(bool createIfMissing) {
  if (!traitConformanceLineage && createIfMissing)
    traitConformanceLineage.reset(new TraitConformanceLineageType());
  return traitConformanceLineage.get();
}

void ASTDecl::getKnownAssumptionsIncludingParents(
    SmallVectorImpl<ConstraintAttr> &assumptions) const {
  const ASTDecl *decl = this;
  while (decl) {
    if (decl->knownAssumptions)
      assumptions.append(decl->knownAssumptions->begin(),
                         decl->knownAssumptions->end());
    decl = decl->getParentDecl();
  }
}

llvm::SmallVector<ConstraintAttr>
ASTDecl::getAssumptionsFromScope(const ASTDecl *scope) {
  llvm::SmallVector<ConstraintAttr> assumptions;
  if (scope)
    scope->getKnownAssumptionsIncludingParents(assumptions);
  return assumptions;
}

void ASTDecl::insertKnownAssumptions(ArrayRef<ConstraintAttr> assumptions) {
  if (!knownAssumptions)
    knownAssumptions.reset(new llvm::SetVector<ConstraintAttr>());
  knownAssumptions->insert(assumptions.begin(), assumptions.end());
}

/// Return the nearest parameter scope (i.e. DeclInterface) for the given decl,
/// as well as the total depth from the nearest file module.
static std::pair<ASTDecl *, size_t> getNearestParamScopeAndDepth(
    ASTDecl *decl, function_ref<void(const ASTDecl *)> checkForCollision) {
  ASTDecl *paramScope = nullptr;
  size_t depth = 0;
  while (decl) {
    checkForCollision(decl);

    bool isParamScope =
        isa_and_nonnull<DeclInterface>(decl->getIfOperation()) ||
        decl->getIsExplicitParamScope();
    if (!paramScope && isParamScope)
      paramScope = decl;

    if (isParamScope) {
      ++depth;
      if (isa<FileModuleASTDecl>(decl))
        break;
    }

    decl = decl->getParentDecl();
  }

  return {paramScope, --depth}; // Adjust so depth starts at 0.
}

void ASTDecl::addRecursivelyStableName(StringAttr name) {
  if (!recursivelyStableNames)
    recursivelyStableNames = std::make_unique<llvm::DenseSet<StringAttr>>();
  recursivelyStableNames->insert(name);
}

bool ASTDecl::hasRecursivelyStableName(StringAttr name) const {
  for (const ASTDecl *scope = this; scope; scope = scope->getParentDecl()) {
    if (scope->recursivelyStableNames &&
        scope->recursivelyStableNames->contains(name))
      return true;
  }
  return false;
}

void ASTDecl::addDocInlineName(StringAttr name) {
  if (!docInlineNames)
    docInlineNames = std::make_unique<llvm::DenseSet<StringAttr>>();
  docInlineNames->insert(name);
}

bool ASTDecl::isDocInlineName(StringAttr name) const {
  return docInlineNames && docInlineNames->contains(name);
}

bool ASTDecl::hasRecursivelyStableType(const ASTDecl *typeDecl) const {
  if (!typeDecl)
    return false;
  if (auto name = typeDecl->getUserNameIfOperation())
    return hasRecursivelyStableName(StringAttr::get(getContext(), *name));
  return false;
}

bool UnresolvedWildcardImport::markSearched(StringRef name) {
  if (!searchedNames)
    searchedNames.reset(new llvm::StringSet<>());
  return searchedNames->insert(name).second;
}

/// Add an unresolved wild card import into this scope.
void ASTDecl::addUnresolvedWildcardImport(
    UnresolvedWildcardImport unresolvedImport) {
  // Lazy allocate the storage.
  if (!unresolvedWildcardImports)
    unresolvedWildcardImports.reset(new UnresolvedWildcardImportsType());
  else {
    // If we are already tracking this entity, mark it as superseded so we can
    // place it last. The last unresolved wildcard statement always wins:
    //   from a import *
    //   from b import *
    //   from a import *
    for (UnresolvedWildcardImport &import : *unresolvedWildcardImports)
      if (import.moduleName == unresolvedImport.moduleName)
        import.isSuperseded = true;
  }
  unresolvedWildcardImports->emplace_back(std::move(unresolvedImport));
}

/// Mangle a parameter name for the given parameter scope and scope depth. Due
/// to the use of depth, the mangling doesn't change when the order of function
/// declarations change, so we have hash stability.
static StringAttr mangleParamNameImpl(const Twine &name, size_t depth,
                                      ASTDecl *paramScope) {
  MLIRContext *ctx = paramScope->getContext();

  // Top level funcs/structs are the most common, so we want to simplify the
  // mangling for that case. Many tests (and real world code too) has a single
  // parameter in a scope, so we also try to make that case nicer.
  std::string suffix = "`";
  if (depth != 1)
    suffix.append(llvm::utostr(depth) + 'x');
  if (size_t id = paramScope->getNextUniqueID(); id != 0)
    suffix.append(llvm::utostr(id));

  return StringAttr::get(ctx, name + suffix);
}

StringAttr ASTDecl::mangleUserDefinedParamName(StringAttr name) {
  bool hasCollision = false;
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *curScope) {
        hasCollision =
            hasCollision || !curScope->lookupInCurrentScope(name).empty();
      });
  if (!hasCollision)
    return name;

  return mangleParamNameImpl(name.strref(), depth, paramScope);
}

StringAttr ASTDecl::mangleParamName(const Twine &name) {
  // This function always mangles, so no need to check for collisions.
  auto [paramScope, depth] =
      getNearestParamScopeAndDepth(this, [&](const ASTDecl *) {});
  return mangleParamNameImpl(name, depth, paramScope);
}

void ASTDecl::dump() const {
  // The value is either an operation or a type of MLIR `Value`.
  if (auto *op = getIfOperation()) {
    // Print without verifying, since IR could be in an invalid state.
    op->print(llvm::errs(), mlir::OpPrintingFlags().printGenericOpForm());
    llvm::errs() << "\n";
  } else if (auto cv = getIfIRValue()) {
    cv.dump();
  } else if (auto *witness = dyn_cast<WitnessASTDecl>(this)) {
    llvm::errs() << "witness ";
    if (resolvedness < DeclResolvedness::signature) {
      llvm::errs() << "<unresolved>";
    } else {
      auto resolvedType = witness->getWitnessEntry();
      resolvedType.witnessName.print(llvm::errs());
      llvm::errs() << ": ";
      resolvedType.witnessType.print(llvm::errs());
    }
    llvm::errs() << "\n";
  } else {
    llvm::errs() << "<null decl>\n";
  }
}

ASTType ASTDecl::getIfTypeValue() const {
  if (auto cv = getIfIRValue().getIfPValue())
    return cv.getIfTypeValue();
  return {};
}

std::optional<StringRef> ASTDecl::getUserNameIfOperation() const {
  if (Operation *op = getIfOperation())
    if (auto decl = dyn_cast<ASTDeclInterface>(op))
      return decl.getDeclName().getValue();
  return {};
}

/// Return the SymbolRefAttr for a declaration, including all scoping that may
/// be needed, making it unique for every declaration.  This returns null for
/// named values that do not have a declaration.
SymbolRefAttr ASTDecl::getSymbolRef() const {
  if (auto traitType = dyn_cast_if_present<TraitType>(getIfTypeValue())) {
    auto reducedSymbols =
        LIT::reduceTraitCompositionSymbols(getShared(), traitType.getSymbols());
    // If this is a single trait, return it.
    if (reducedSymbols.size() == 1)
      return reducedSymbols[0].getSymbol();
    return {};
  }

  auto op = dyn_cast_if_present<mlir::SymbolOpInterface>(getIfOperation());
  if (!op)
    return {};
  assert((!isa<FnOp>(op) || resolvedness >= DeclResolvedness::signature) &&
         "Functions don't have a symbol until their signatures are resolved");
  return getFullyResolvedSymbolRef(op);
}

/// Given an MLIR op for a struct declaration, return the self type.
Type StructASTDecl::computeSelfType(StructDeclOp structOp) {
  SmallVector<TypedAttr> parameters;
  for (auto decl : structOp.getParams()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    parameters.push_back(ParamDeclRefAttr::get(decl));
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return structOp.bindReference(parameters);
}

Type TraitASTDecl::computeSelfType(TraitDeclOp traitOp) {
  // The last parameter to the trait is the 'T' parameter which (when everything
  // gets instantiated) resolves to the final type the trait is instantiated on.
  return ASTType(ParamDeclRefAttr::get(traitOp.getParamsAttr().back()));
}

void ASTDecl::findExtensionsInScopeForStruct(
    SymbolRefAttr targetStruct, llvm::SmallPtrSetImpl<ASTDecl *> &results,
    std::optional<TraitSymbolAttr> filterTrait) {
  if (!declsInScope)
    return;

  // Fast path: any scope that has extensions also registers them under the
  // aggregate name "extension:" (see the ExtensionDeclOp case in
  // SharedState::addDeclsForOp and the module-import paths in DeclResolver).
  // This is a hot conformance-check query and the vast majority of scopes have
  // no extensions at all, so bail out with a single pointer-keyed lookup on the
  // pre-interned marker before building any per-struct name.
  if (declsInScope->find(shared.extensionsScopeMarker) == declsInScope->end())
    return;

  // The exact-symbol check below filters out other structs sharing the leaf
  // name.
  auto it = declsInScope->find(
      shared.getExtensionName(targetStruct.getLeafReference().getValue()));
  if (it == declsInScope->end())
    return;

  for (ASTDecl *decl : it->second) {
    auto *extensionDecl = dyn_cast<ExtensionASTDecl>(decl);
    if (!extensionDecl ||
        extensionDecl->getOp().getTargetStructAttr() != targetStruct)
      continue;
    ExtensionDeclOp extOp = extensionDecl->getOp();

    // If no trait filter specified, add this extension.
    if (!filterTrait.has_value()) {
      results.insert(decl);
      continue;
    }

    // Extension doesn't have canonicalTrait computed yet - skip it. This
    // happens during error conditions or early parsing phases.
    if (!extOp.getCanonicalTrait())
      continue;

    // Use the extension's canonicalTrait (flattened hierarchy) to check
    // whether it implements the filter trait.
    for (TraitSymbolAttr symbol :
         extOp.getCanonicalTrait().value().getSymbols()) {
      if (symbol == filterTrait.value()) {
        results.insert(decl);
        break; // Found it, no need to check more symbols.
      }
    }
  }
}
