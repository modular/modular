//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoTypeSystem.h"
#include "../ExpressionParser/MojoDiagnostic.h"
#include "../ExpressionParser/MojoExpressionParser.h"
#include "../ExpressionParser/MojoExpressionVariable.h"
#include "../ExpressionParser/MojoUserExpression.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Runtime/Runtime.h"
#include "MojoTypeDataLayout.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Process.h"
#include <mlir/AsmParser/AsmParser.h>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;
using namespace mlir;

/// Convert a KGENDType, which is an extension to the regular MLIR DType, into
/// MLIR types that can be understood by the typesystem.
static std::optional<mlir::Type>
getMLIRTypeForDType(MLIRContext *ctx, KGENDType dtype, size_t indexBitwidth) {
  // `address` and `index` are extensions to the regular dtype.
  if (dtype.isAddress())
    return LLVM::LLVMPointerType::get(ctx);

  if (dtype.isIndex())
    return IntegerType::get(ctx, indexBitwidth);

  // This checks for `bool` and `int` types.
  if (IntegerType intType = getEquivalentIntegerType(ctx, dtype))
    return intType;

  if (FloatType fpType = getEquivalentFloatType(ctx, dtype))
    return fpType;

  return {};
}

//===----------------------------------------------------------------------===//
// MojoTypeSystem::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeSystem::Impl {
  Impl(Target *target, const ArchSpec &archSpec)
      : target(target), archSpec(archSpec) {
    // Register all of the various dialect state.
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    registerKGENToLLVMTranslation(registry);

    // Set up the dialects in the context.
    mlirContext.appendDialectRegistry(registry);

    // Configure the runtime.
    runtime = std::make_unique<LLCL::Runtime>(
        LLCL::createMallocAllocator(),
        LLCL::createThreadPoolWorkQueue(0, /*mainWillDonate=*/false));

    // Compute the target information for the expression.
    compilationOptions.targetTriple = archSpec.GetTriple().str();

    // TODO: Populate cpu information properly here.
    if (archSpec.IsValid()) {
      compilationOptions.targetTriple = archSpec.GetTriple().str();
      compilationOptions.relocModel = archSpec.GetTriple().isOSBinFormatMachO()
                                          ? llvm::Reloc::PIC_
                                          : llvm::Reloc::Static;
    }
    compilationOptions.targetCpu = llvm::sys::getHostCPUName();

    // Configure the parser context.
    LIT::ParserConfig parserConfig(&mlirContext, *runtime, compilationOptions);
    parserConfig.moduleCachingLevel = LIT::ParserConfig::kCacheNone;
    parserContext =
        std::make_unique<MojoParserContext>(sourceMgr, parserConfig);

    auto targetInfoOr = M::getTargetInfoFor(
        &mlirContext, compilationOptions.targetTriple,
        compilationOptions.targetCpu, compilationOptions.targetFeatures,
        /*tuneCpu=*/"", compilationOptions.relocModel);
    if (succeeded(targetInfoOr))
      targetInfo = *targetInfoOr;

    dataLayoutContext =
        std::make_unique<MojoTypeDataLayoutContext>(*parserContext, targetInfo);
  }

  /// Utility that returns a StructDeclOp if the given astType corresponds to a
  /// struct, otherwise an invalid object is returned.
  LIT::StructDeclOp getIfStructDecl(MojoASTTypeRef astType) {
    if (auto declRef = parserContext->getDecl(astType)) {
      if (LIT::StructDeclOp structDeclOp =
              dyn_cast_if_present<LIT::StructDeclOp>(
                  declRef.getIfOperation())) {
        return structDeclOp;
      }
    }
    return {};
  }

  /// The MLIR context to use for compilation/processing associated with this
  /// typesystem.
  MLIRContext mlirContext;

  /// The LLCL runtime to use for compilation/processing associated with this
  /// typesystem.
  std::unique_ptr<LLCL::Runtime> runtime;

  /// The compilation options to use when compiling.
  KGEN::CompilationOptions compilationOptions;

  /// The source manager used for expression compilation.
  llvm::SourceMgr sourceMgr;

  /// The current stack of working directories.
  SmallVector<std::string> expressionWorkingDirectories;

  /// The main parser context used for compilation.
  std::unique_ptr<MojoParserContext> parserContext;

  /// The target that this typesystem is associated with. It's available only
  /// for expression evaluation.
  lldb_private::Target *target;

  lldb_private::ArchSpec archSpec;

  /// The persistent state for this typesystem.
  MojoPersistentExpressionState persistentState;

  /// The target info of the current LLDB Target.
  TargetInfoAttr targetInfo;

  /// The cache to be used for querying data layouts.
  std::unique_ptr<MojoTypeDataLayoutContext> dataLayoutContext;

  std::unique_ptr<MojoDWARFParser> dwarfParser;
};

//===----------------------------------------------------------------------===//
// MojoTypeSystem
//===----------------------------------------------------------------------===//

MojoTypeSystem::MojoTypeSystem(Target *target, const ArchSpec &archSpec)
    : impl(std::make_unique<Impl>(target, archSpec)) {}

MojoTypeSystem::~MojoTypeSystem() = default;
char MojoTypeSystem::ID = 0;

MLIRContext *MojoTypeSystem::getMLIRContext() { return &impl->mlirContext; }

LIT::SharedState &MojoTypeSystem::getSharedState() {
  return impl->parserContext->getSharedState();
}

MojoParserContext &MojoTypeSystem::getParserContext() {
  return *impl->parserContext;
}

TargetInfoAttr MojoTypeSystem::GetTargetInfo() const {
  return impl->targetInfo;
}

LLCL::Runtime &MojoTypeSystem::getRuntime() { return *impl->runtime; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Create a MojoTypeSystem instance from the given module and target.
static lldb::TypeSystemSP createInstance(lldb::LanguageType language,
                                         Module *module, Target *target) {
  if (language != lldb::eLanguageTypeMojo)
    return {};

  ArchSpec arch;
  if (module)
    arch = module->GetArchitecture();
  else if (target)
    arch = target->GetArchitecture();

  if (!arch.IsValid())
    return {};

  return std::make_shared<MojoTypeSystem>(target, arch);
}

void MojoTypeSystem::Initialize() {
  LanguageSet languages;
  languages.Insert(lldb::eLanguageTypeMojo);
  PluginManager::RegisterPlugin(getPluginNameStatic(), "Mojo TypeSystem",
                                createInstance, languages, languages);
}

void MojoTypeSystem::Terminate() {
  PluginManager::UnregisterPlugin(createInstance);
}

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

void MojoTypeSystem::pushWorkingDirectory(StringRef workingDirectory) {
  std::vector<std::string> currentDirs = impl->sourceMgr.getIncludeDirs();

  // Update the include directories to include this new directory.
  if (!impl->expressionWorkingDirectories.empty()) {
    auto it =
        llvm::find(currentDirs, impl->expressionWorkingDirectories.back());
    assert(it != currentDirs.end() &&
           "working directory not found in include directories");
    *it = currentDirs.back();
  } else {
    currentDirs.insert(currentDirs.begin(), workingDirectory.str());
  }

  impl->expressionWorkingDirectories.push_back(workingDirectory.str());
  impl->sourceMgr.setIncludeDirs(currentDirs);
}

void MojoTypeSystem::popWorkingDirectory() {
  if (impl->expressionWorkingDirectories.empty())
    return;
  std::string dir = impl->expressionWorkingDirectories.pop_back_val();

  // Update the include directories to remove this directory.
  std::vector<std::string> currentDirs = impl->sourceMgr.getIncludeDirs();
  auto it = llvm::find(currentDirs, dir);
  assert(it != currentDirs.end() &&
         "working directory not found in include directories");
  if (impl->expressionWorkingDirectories.empty())
    currentDirs.erase(it);
  else
    *it = impl->expressionWorkingDirectories.back();

  impl->sourceMgr.setIncludeDirs(currentDirs);
}

//===----------------------------------------------------------------------===//
// Type Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::IsPointerOrReferenceType(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeType) {
  return IsReferenceType(type, pointeeType, /*isRValue=*/nullptr) ||
         IsPointerType(type, pointeeType);
}

bool MojoTypeSystem::IsAggregateType(lldb::opaque_compiler_type_t type) {
  switch (GetTypeClass(type)) {
  case lldb::eTypeClassArray:
  case lldb::eTypeClassStruct:
    return true;
  default:
    return false;
  }
}

bool MojoTypeSystem::IsPointerType(lldb::opaque_compiler_type_t type,
                                   lldb_private::CompilerType *pointeeType) {
  if (!type)
    return false;

  if (auto pointerType = dyn_cast<KGEN::PointerType>(MojoASTTypeRef(type))) {
    if (pointeeType)
      *pointeeType = createCompilerType(pointerType.getElementType());
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsReferenceType(lldb::opaque_compiler_type_t type,
                                     lldb_private::CompilerType *pointeeType,
                                     bool *isRValue) {
  return false;
}

lldb_private::CompilerType
MojoTypeSystem::GetPointeeType(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};
  MojoASTTypeRef astType(type);
  return createCompilerType(astType.getPointerElementType());
}

lldb_private::CompilerType
MojoTypeSystem::GetPointerType(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};
  MojoASTTypeRef astType(type);
  return createCompilerType(KGEN::PointerType::get(astType.getMLIRType()));
}

lldb::TypeClass
MojoTypeSystem::GetTypeClass(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};

  MojoASTTypeRef astType(type);

  if (auto ptrType = dyn_cast<PointerType>(astType))
    return lldb::eTypeClassPointer;

  if (isa<POP::SIMDType>(astType))
    return lldb::eTypeClassVector;

  if (impl->getIfStructDecl(astType))
    return lldb::eTypeClassStruct;

  return lldb::eTypeClassOther;
}

uint32_t MojoTypeSystem::GetTypeInfo(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeOrElementCompilerType) {
  if (!type)
    return 0;

  if (pointeeOrElementCompilerType)
    pointeeOrElementCompilerType->Clear();

  MojoASTTypeRef astType(type);

  if (auto ptrType = dyn_cast<PointerType>(astType)) {
    if (pointeeOrElementCompilerType) {
      *pointeeOrElementCompilerType =
          createCompilerType(ptrType.getElementType());
    }
    return lldb::eTypeIsPointer | lldb::eTypeHasChildren | lldb::eTypeHasValue;
  }

  if (isa<IndexType>(astType))
    return lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar;

  if (auto intType = dyn_cast<IntegerType>(astType)) {
    auto result =
        lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar;
    if (intType.isSignedInteger())
      return result | lldb::eTypeIsSigned;
    return result;
  }

  if (isa<FloatType>(astType))
    return lldb::eTypeIsFloat | lldb::eTypeHasValue | lldb::eTypeIsScalar;

  if (isa<POP::SIMDType>(astType))
    return lldb::eTypeHasChildren | lldb::eTypeIsVector;

  if (isa<KGEN::StringType>(astType))
    return lldb::eTypeIsPointer | lldb::eTypeHasChildren | lldb::eTypeHasValue;

  if (impl->getIfStructDecl(astType))
    return lldb::eTypeHasChildren | lldb::eTypeIsClass;

  if (isa<LIT::REPLResultRefType>(astType))
    return lldb::eTypeHasChildren;

  return {};
}

lldb::Format MojoTypeSystem::GetFormat(lldb::opaque_compiler_type_t type) {
  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger)
    return lldb::eFormatDecimal;
  if (flags & lldb::eTypeIsFloat)
    return lldb::eFormatFloat;
  if (flags & lldb::eTypeIsPointer) {
    if (isa<KGEN::StringType>(MojoASTTypeRef(type)))
      return lldb::eFormatCString;
    return lldb::eFormatHex;
  }
  if (flags & lldb::eTypeIsClass)
    return lldb::eFormatHex;
  if (flags & lldb::eTypeIsFuncPrototype || flags & lldb::eTypeIsBlock)
    return lldb::eFormatAddressInfo;
  return lldb::eFormatBytes;
}

lldb_private::CompilerType
MojoTypeSystem::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(type);
}

lldb_private::CompilerType
MojoTypeSystem::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(type);
}

lldb_private::CompilerType
MojoTypeSystem::GetCanonicalType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(type);
}

uint32_t MojoTypeSystem::GetPointerByteSize() {
  return impl->targetInfo.getDataLayout().getPointerSize();
}

std::optional<uint64_t>
MojoTypeSystem::GetBitSize(lldb::opaque_compiler_type_t type,
                           lldb_private::ExecutionContextScope *exeScope) {
  if (!type)
    return {};

  if (auto &layout = impl->dataLayoutContext->getOrCalculate(type))
    return layout->getByteSize() * CHAR_BIT;

  return {};
}

std::optional<size_t>
MojoTypeSystem::GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                                lldb_private::ExecutionContextScope *exeScope) {
  if (!type)
    return {};

  if (auto &layout = impl->dataLayoutContext->getOrCalculate(type))
    return layout->getAlignment() * CHAR_BIT;

  return {};
}

lldb::Encoding MojoTypeSystem::GetEncoding(lldb::opaque_compiler_type_t type,
                                           uint64_t &count) {
  if (!type)
    return lldb::eEncodingInvalid;

  // Count is the number of elements encoded in the type.
  count = 1;

  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger) {
    if (flags & lldb::eTypeIsSigned)
      return lldb::eEncodingSint;
    return lldb::eEncodingUint;
  }

  if (flags & lldb::eTypeIsFloat)
    return lldb::eEncodingIEEE754;

  if (flags & lldb::eTypeIsPointer)
    return lldb::eEncodingUint;

  count = 0;
  return lldb::eEncodingInvalid;
}

ConstString MojoTypeSystem::GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) {
  if (!type)
    return {};

  std::string name;
  llvm::raw_string_ostream os(name);
  mlir::Type::getFromOpaquePointer(type).print(os);
  return ConstString(name);
}

ConstString
MojoTypeSystem::GetDisplayTypeName(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};

  MojoASTTypeRef astType(type);

  // There's no way to change the display type name using synthetic formatters,
  // so we have to do it here for REPLResultRefType.
  if (auto replType = dyn_cast<LIT::REPLResultRefType>(astType))
    return createCompilerType(replType.getElementType()).GetDisplayTypeName();

  std::string name = astType.getAsString();

  auto mangledOr =
      LIT::MangledSymbol::demangle(StringAttr::get(&impl->mlirContext, name));

  // We need to delete the artificial module we use for expression evaluations
  // to avoid confusing the user.
  if (succeeded(mangledOr) && !mangledOr->moduleNames.empty() &&
      MojoPersistentExpressionState::isExpressionModuleName(
          mangledOr->moduleNames.back()))
    return ConstString(mangledOr->symName);

  return ConstString(name);
}

//===----------------------------------------------------------------------===//
// IsType Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::IsFloatingPointType(lldb::opaque_compiler_type_t type,
                                         uint32_t &count, bool &isComplex) {
  count = 0;
  isComplex = false;
  if (GetTypeInfo(type) & lldb::eTypeIsFloat) {
    count = 1;
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsIntegerType(lldb::opaque_compiler_type_t type,
                                   bool &isSigned) {
  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger) {
    isSigned = flags & lldb::eTypeIsSigned;
    return true;
  }
  return false;
}

bool MojoTypeSystem::IsScalarType(lldb::opaque_compiler_type_t type) {
  return GetTypeInfo(type) & lldb::eTypeIsScalar;
}

//===--------------------------------------------------------------------===//
// Type Navigation
//===--------------------------------------------------------------------===//

uint32_t
MojoTypeSystem::GetNumChildren(lldb::opaque_compiler_type_t type,
                               bool omitEmptyBaseClasses,
                               const lldb_private::ExecutionContext *exeCtx) {
  if (!type)
    return 0;

  MojoASTTypeRef astType(type);

  if (isa<KGEN::PointerType>(astType) || isa<LIT::REPLResultRefType>(astType))
    return 1;

  if (auto simdTy = dyn_cast<POP::SIMDType>(astType)) {
    if (simdTy.isScalar())
      return 1;
    return simdTy.getResolvedSize().value_or(0);
  }

  if (LIT::StructDeclOp structDecl = impl->getIfStructDecl(astType))
    return llvm::range_size(structDecl.getFieldDecls());

  if (auto packType = dyn_cast<PackType>(astType)) {
    if (auto attr = dyn_cast<VariadicAttr>(packType.getVariadic()))
      return llvm::range_size(attr.getValues());
    return 0;
  }

  return 0;
}

lldb_private::CompilerType MojoTypeSystem::GetChildCompilerTypeAtIndex(
    lldb::opaque_compiler_type_t type, lldb_private::ExecutionContext *exeCtx,
    size_t idx, bool transparent_pointers, bool omitEmptyBaseClasses,
    bool ignoreArrayBounds, std::string &childName, uint32_t &childByteSize,
    int32_t &childByteOffset, uint32_t &childBitfieldBitSize,
    uint32_t &childBitfieldBitOffset, bool &childIsBaseClass,
    bool &childIsDerefOfParent, lldb_private::ValueObject *valobj,
    uint64_t &languageFlags) {
  if (!type)
    return lldb_private::CompilerType();

  if (!ignoreArrayBounds &&
      idx >= GetNumChildren(type, omitEmptyBaseClasses, exeCtx))
    return {};

  MojoASTTypeRef astType(type);

  // Pointer only has one child, so just return the unwrapped pointer type
  if (isa<KGEN::PointerType>(astType)) {
    MojoASTTypeRef eltType(astType.getPointerElementType());
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(eltType)) {
      childByteSize = layout->getByteSize();
      childByteOffset = 0;
      childIsDerefOfParent = true;
      const char *parentName =
          valobj ? valobj->GetName().GetCString() : nullptr;
      if (parentName) {
        childName.assign(1, '*');
        childName += parentName;
      }
      return createCompilerType(astType.getPointerElementType());
    }
    return {};
  }

  // REPLResultRefType owns a pointer, so we return it as its child.
  if (auto replType = dyn_cast<LIT::REPLResultRefType>(astType)) {
    MojoASTTypeRef eltType(replType.getElementType());
    auto compilerType = GetPointerType(
        const_cast<void *>(replType.getElementType().getAsOpaquePointer()));
    childByteSize = GetPointerByteSize();
    childByteOffset = 0;
    return compilerType;
  }

  if (auto simdType = dyn_cast<POP::SIMDType>(astType)) {
    if (std::optional<KGENDType> kgenDTypeOpt = simdType.getResolvedDType()) {
      if (kgenDTypeOpt.has_value()) {
        std::optional<mlir::Type> eltMlirType = getMLIRTypeForDType(
            getMLIRContext(), *kgenDTypeOpt, 8 * GetPointerByteSize());
        if (!eltMlirType)
          return {};
        MojoASTTypeRef eltType(*eltMlirType);

        if (const std::optional<MojoTypeDataLayout> &layout =
                impl->dataLayoutContext->getOrCalculate(eltType)) {
          childName = std::string(llvm::formatv("[{0}]", idx));
          childByteSize = layout->getByteSize();
          childByteOffset = (int32_t)idx * (int32_t)childByteSize;
          return createCompilerType(eltType);
        } else {
          return {};
        }
      }
    }
    return {};
  }

  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      auto fieldDecl = *std::next(structDeclOp.getFieldDecls().begin(), idx);
      childName.assign(fieldDecl.getName());
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return {};
  }

  if (auto packType = dyn_cast<PackType>(astType)) {
    if (const std::optional<MojoTypeDataLayout> &layout =
            impl->dataLayoutContext->getOrCalculate(astType)) {
      childName = std::string(llvm::formatv("[{0}]", idx));
      const auto &field = layout->getFields()[idx];
      childByteOffset = field.getByteOffset();
      childByteSize = field.getByteSize();
      return createCompilerType(field.getConcreteType());
    }
    return {};
  }
  return {};
}

size_t MojoTypeSystem::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, llvm::StringRef name,
    bool omitEmptyBaseClasses, std::vector<uint32_t> &childIndices) {
  // This method should return the total number of indices in `childIndices`
  // in the case of success. As a remark, the `childIndices` vector passed in
  // might not be empty.
  MojoASTTypeRef astType(type);

  // Check if the name is an index of a SIMD or of a pack, which are 0-indexed.
  if (isa<PackType, POP::SIMDType>(astType)) {
    unsigned long index;
    if (name.consume_front("[") && !name.consumeInteger(10, index) &&
        name.consume_front("]") && name.empty()) {
      childIndices.push_back(index);
      return childIndices.size();
    }
    return 0;
  }

  // Check if it's a field of a struct.
  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType)) {
    for (auto field : llvm::enumerate(structDeclOp.getFieldDecls())) {
      if (field.value().getName() == name) {
        childIndices.push_back(field.index());
        return childIndices.size();
      }
    }
    return 0;
  }

  return 0;
}

//===--------------------------------------------------------------------===//
// Mojo-specific Type Queries
//===--------------------------------------------------------------------===//

llvm::ArrayRef<TypedAttr>
MojoTypeSystem::getStructDecorators(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};
  MojoASTTypeRef astType(type);
  if (LIT::StructDeclOp structDeclOp = impl->getIfStructDecl(astType))
    return structDeclOp.getDecorators();
  return {};
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

UserExpression *MojoTypeSystem::GetUserExpression(
    StringRef expr, StringRef prefix, lldb::LanguageType language,
    Expression::ResultType desiredType,
    const EvaluateExpressionOptions &options, ValueObject *ctxObj) {
  if (!impl->target || ctxObj)
    return nullptr;
  return new MojoUserExpression(*impl->target, expr, prefix, language,
                                desiredType, options);
}

PersistentExpressionState *MojoTypeSystem::GetPersistentExpressionState() {
  return &impl->persistentState;
}

//===--------------------------------------------------------------------===//
// Debug info parsing
//===--------------------------------------------------------------------===//

DWARFASTParser *MojoTypeSystem::GetDWARFParser() {
  if (!impl->dwarfParser)
    impl->dwarfParser = std::make_unique<MojoDWARFParser>(*this);
  return impl->dwarfParser.get();
}

CompilerType
MojoTypeSystem::getBuiltinTypeFromMLIRTypeName(llvm::StringRef typeName) {
  if (typeName.empty())
    return {};
  ScopedDiagnosticHandler diagHandler(getMLIRContext(), [&](Diagnostic &diag) {
    // These logs can get extremely noisy when attempting to parse the DWARF
    // of builtin types, so we only enable them if `verbose` is on.
    if (Log *log = GetLog(LLDBLog::Types); log && log->GetVerbose()) {
      LLDB_LOG(log,
               "[MojoTypeSystem::getBuiltinTypeFromMLIRTypeName] MLIR "
               "diagnostic: {0}",
               diag.str());
    }
  });
  if (auto type = mlir::parseType(typeName, getMLIRContext()))
    return createCompilerType(type);
  return {};
}

CompilerType MojoTypeSystem::getBuiltinScalarType(llvm::StringRef typeName,
                                                  uint32_t dwarfEncoding,
                                                  uint32_t byteSize) {
  if (dwarfEncoding == DW_ATE_unsigned)
    return createCompilerType(IntegerType::get(getMLIRContext(), byteSize * 8,
                                               IntegerType::Unsigned));

  if (dwarfEncoding == DW_ATE_signed)
    return createCompilerType(
        IntegerType::get(getMLIRContext(), byteSize * 8, IntegerType::Signed));

  // Fortunately MLIR DTypes have the same name as KGEN DTypes, so we can use
  // the common translator.
  if (dwarfEncoding == DW_ATE_float || dwarfEncoding == DW_ATE_boolean)
    return createCompilerTypeFromDType(typeName);

  return {};
}

lldb_private::CompilerType
MojoTypeSystem::createCompilerTypeFromDType(StringRef dtype) {
  auto dTypeOr = KGENDType::getFromString(dtype);
  if (failed(dTypeOr))
    return {};
  return createCompilerType(*getMLIRTypeForDType(getMLIRContext(), *dTypeOr,
                                                 8 * GetPointerByteSize()));
}

lldb_private::CompilerType MojoTypeSystem::createSIMDType(StringRef dtype,
                                                          size_t numElements) {
  if (llvm::popcount(numElements) != 1)
    return {};
  auto dTypeOr = KGENDType::getFromString(dtype);
  if (failed(dTypeOr))
    return {};
  return createCompilerType(
      KGEN::POP::SIMDType::get(getMLIRContext(), numElements, *dTypeOr));
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateModuleDecl(StringRef moduleName,
                                      MojoASTDeclRef parentDeclRef) {
  LIT::SharedState &sharedState = impl->parserContext->getSharedState();
  LIT::ASTDecl &parentDecl =
      parentDeclRef ? *parentDeclRef
                    : impl->parserContext->getSharedState().getTopLevelDecl();

  // We first check if the module already exists, in which case we just return
  // its decl.
  StringAttr mangledName =
      sharedState.getMangledModuleName(getMLIRContext(), moduleName);
  auto &declsInScope = parentDecl.getDeclsInScope();
  if (auto it = declsInScope.find(mangledName); it != declsInScope.end()) {
    assert(it->second.size() == 1 &&
           "We expect one single module decl with a given name.");
    return it->second[0];
  }

  // We create a fake empty file so that parser diagnostics can be emitted if
  // we are doing somethig wrong when creating the decls. Otherwise, we hit
  // asserts and LLDB aborts.
  auto loc = FileLineColLoc::get(getMLIRContext(), moduleName, /*line=*/0,
                                 /*column=*/0);
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBufferCopy("", loc.getFilename().getValue());
  auto &sourceMgr = impl->parserContext->getSourceMgr();
  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(
      sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc()));
  LIT::Lexer lexer(impl->parserContext->getSharedState().diags, sourceBuf);

  Operation *fileOp = parentDecl.getDeclEndBuilder().create<LIT::FileModuleOp>(
      sharedState.translateLocation(parentDecl.getLoc()), mangledName,
      StringAttr::get(sharedState.getContext(), moduleName));
  return &sharedState.declResolver->addFullyResolvedDecl(
      fileOp, mangledName, lexer.getToken().getLoc(), &parentDecl);
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateFunctionDecl(llvm::StringRef mangledName) {
  auto mangled = StringAttr::get(getMLIRContext(), mangledName);
  FailureOr<LIT::MangledSymbol> mangledSymbol =
      LIT::MangledSymbol::demangle(mangled);
  if (failed(mangledSymbol))
    return {};

  // We traverse modules and structs creating them as needed.
  LIT::ASTDecl *parentDecl = nullptr;
  LIT::SharedState &sharedState = impl->parserContext->getSharedState();

  for (StringAttr moduleName : mangledSymbol->moduleNames)
    parentDecl = &*getOrCreateModuleDecl(moduleName, parentDecl);

  for (StringAttr structName : mangledSymbol->structNames)
    parentDecl = &*getOrCreateStructDecl(structName, parentDecl);

  assert(parentDecl != nullptr && "All functions must have a parent decl.");

  StringAttr name = mangledSymbol->symName;

  // We check if the function already exists, in which case we just return
  // its decl.
  auto &declsInScope = parentDecl->getDeclsInScope();
  if (auto it = declsInScope.find(name); it != declsInScope.end()) {
    assert(it->second.size() == 1 &&
           "We expect one single function decl with a given name.");
    return it->second[0];
  }

  auto builder = parentDecl->getDeclEndBuilder();
  auto fnType = builder.getFunctionType({}, {NoneType::get(getMLIRContext())});
  // We might need to fill in the full signature when expression evaluation is
  // needed. We don't need it for now.
  auto metadata = LIT::FnMetadataAttr::get(getMLIRContext(), {}, {});
  auto signature = LIT::LITSignatureType::get(fnType, {}, {}, {}, {}, metadata);

  // FIXME(23810): We need to support nested functions.

  StringAttr nameAttr = LIT::DeclResolver::getMangledName(name, signature);
  auto newFunction = builder.create<LIT::FuncOp>(
      sharedState.translateLocation(parentDecl->getLoc()), nameAttr, name,
      signature);
  return MojoASTDeclRef(&sharedState.declResolver->addDecl(
      newFunction, parentDecl->getLoc(), name, parentDecl, {}, {}, -1));
}

MojoASTDeclRef
MojoTypeSystem::getOrCreateStructDecl(StringRef structName,
                                      MojoASTDeclRef parentDecl) {
  StringAttr name = StringAttr::get(getMLIRContext(), structName);

  // We first check if the struct already exists, in which case we just return
  // its decl.
  auto &declsInScope = parentDecl->getDeclsInScope();
  if (auto it = declsInScope.find(name); it != declsInScope.end()) {
    assert(it->second.size() == 1 &&
           "We expect one single struct decl with a given name.");
    return it->second[0];
  }

  auto newStruct = parentDecl->getDeclEndBuilder().create<LIT::StructDeclOp>(
      getSharedState().translateLocation(parentDecl->getLoc()), name);
  return MojoASTDeclRef(&getSharedState().declResolver->addDecl(
      newStruct, parentDecl->getLoc(), name, &*parentDecl, {}, {}, -1));
}

MojoASTDeclRef MojoTypeSystem::getOrCreateStructDecl(StringRef mangledName,
                                                     const DWARFDIE &die) {
  FailureOr<LIT::MangledSymbol> mangledSymbol = LIT::MangledSymbol::demangle(
      StringAttr::get(getMLIRContext(), mangledName));
  if (failed(mangledSymbol) || mangledSymbol->moduleNames.empty()) {
    // Builtin structs might not have a parent module, so we can just put them
    // in an anonymous one. Besides that, as multiple definitions of different
    // structs with the same name might exist in the same compilation unit, we
    // create a different anonymous module for each definition using the offset
    // of the corresponding die. This happens with pack, for example, where
    // different instances of !kgen.pack have different inner data, but they
    // turn out to have the same name, therefore they cannot live under the same
    // module.
    // Another advantage of crating this anonymous module is that we don't need
    // to modify the type name to have it in a unique scope.
    return getOrCreateStructDecl(
        mangledName, &*getOrCreateModuleDecl("anonymous_" +
                                             std::to_string(die.GetOffset())));
  }

  // Note: if we ever have in the same DWARF file two structs with the exact
  // same mangled name but different implementation, we might need to create a
  // top level unique module for each one of them (see the solution for the case
  // above), but that's not the case now.
  LIT::ASTDecl *parentDecl = nullptr;
  for (StringAttr moduleName : mangledSymbol->moduleNames)
    parentDecl = &*getOrCreateModuleDecl(moduleName, parentDecl);

  return getOrCreateStructDecl(mangledSymbol->symName, parentDecl);
}

MojoASTDeclRef
MojoTypeSystem::addFieldToStruct(MojoASTDeclRef structDecl, StringRef fieldName,
                                 lldb::opaque_compiler_type_t type) {
  StringAttr name = StringAttr::get(getMLIRContext(), fieldName);
  auto newField = structDecl->getDeclEndBuilder().create<LIT::StructFieldOp>(
      getSharedState().translateLocation(structDecl->getLoc()), name,
      mlir::Type::getFromOpaquePointer(type), LIT::DocStringAttr());
  return MojoASTDeclRef(&getSharedState().declResolver->addDecl(
      newField, structDecl->getLoc(), name, &*structDecl, {}, {}, -1));
}

ConstString
MojoTypeSystem::DeclContextGetScopeQualifiedName(void *opaqueDeclCtx) {
  if (!opaqueDeclCtx)
    return {};
  return ConstString(MojoASTDeclRef(static_cast<LIT::ASTDecl *>(opaqueDeclCtx))
                         .getType()
                         .getAsString());
}

ConstString MojoTypeSystem::DeclContextGetName(void *opaqueDeclCtx) {
  if (!opaqueDeclCtx) {
    if (std::optional<StringRef> name =
            MojoASTDeclRef(static_cast<LIT::ASTDecl *>(opaqueDeclCtx))
                .getName()) {
      return ConstString(*name);
    }
  }
  return {};
}

//===--------------------------------------------------------------------===//
// Dumping
//===--------------------------------------------------------------------===//

void MojoTypeSystem::Dump(llvm::raw_ostream &output) {
  impl->parserContext->getModule()->dump();
}

bool MojoTypeSystem::DumpTypeValue(
    lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
    lldb::Format format, const lldb_private::DataExtractor &data,
    lldb::offset_t dataOffset, size_t dataByteSize, uint32_t bitfieldBitSize,
    uint32_t bitfieldBitOffset, lldb_private::ExecutionContextScope *exeScope) {
  if (!type)
    return false;
  return lldb_private::DumpDataExtractor(
      data, &s, dataOffset, format, dataByteSize,
      /*itemCount=*/1, UINT32_MAX, LLDB_INVALID_ADDRESS, bitfieldBitSize,
      bitfieldBitOffset, exeScope);
}

void MojoTypeSystem::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                         lldb::DescriptionLevel level) {
  StreamFile s(stdout, false);
  DumpTypeDescription(type, s, level);
}

void MojoTypeSystem::DumpTypeDescription(lldb::opaque_compiler_type_t type,
                                         Stream &s,
                                         lldb::DescriptionLevel level) {
  if (!type)
    return;
  // TODO: complete the implementation. This should dump the type in a way that
  // resembles the source code.
  s << GetDisplayTypeName(type);
}

//===--------------------------------------------------------------------===//
// Utils
//===--------------------------------------------------------------------===//

lldb_private::CompilerType MojoTypeSystem::createCompilerType(mlir::Type type) {
  return lldb_private::CompilerType(
      weak_from_this(), const_cast<void *>(type.getAsOpaquePointer()));
}

lldb_private::CompilerType
MojoTypeSystem::createCompilerType(MojoASTTypeRef astType) {
  return createCompilerType(astType.getMLIRType());
}
