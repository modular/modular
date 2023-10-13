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
#include "MojoTypeDataLayout.h"

#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// MojoTypeSystem::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeSystem::Impl {
  Impl(Target &target) : target(target.shared_from_this()) {
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
    // TODO: Populate cpu information properly here.
    ArchSpec targetArch = target.GetArchitecture();
    if (targetArch.IsValid()) {
      compilationOptions.targetTriple = targetArch.GetTriple().str();
      compilationOptions.relocModel =
          targetArch.GetTriple().isOSBinFormatMachO() ? llvm::Reloc::PIC_
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

  /// The main parser context used for compilation.
  std::unique_ptr<MojoParserContext> parserContext;

  /// The target that this typesystem is associated with.
  lldb::TargetWP target;

  /// The persistent state for this typesystem.
  MojoPersistentExpressionState persistentState;

  /// The target info of the current LLDB Target.
  TargetInfoAttr targetInfo;

  /// The cache to be used for querying data layouts.
  std::unique_ptr<MojoTypeDataLayoutContext> dataLayoutContext;
};

//===----------------------------------------------------------------------===//
// MojoTypeSystem
//===----------------------------------------------------------------------===//

MojoTypeSystem::MojoTypeSystem(Target &target)
    : impl(std::make_unique<Impl>(target)) {}
MojoTypeSystem::~MojoTypeSystem() = default;
char MojoTypeSystem::ID = 0;

MLIRContext *MojoTypeSystem::getMLIRContext() { return &impl->mlirContext; }

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
  // TODO: Support creating a type system from a module.
  if (language != lldb::eLanguageTypeMojo || !target)
    return nullptr;
  return std::make_shared<MojoTypeSystem>(*target);
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
// Type Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::IsPointerOrReferenceType(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeType) {
  return IsReferenceType(type, pointeeType, /*isRValue=*/nullptr) ||
         IsPointerType(type, pointeeType);
}

bool MojoTypeSystem::IsPointerType(lldb::opaque_compiler_type_t type,
                                   lldb_private::CompilerType *pointeeType) {
  if (!type)
    return false;

  if (auto pointerType = dyn_cast<KGEN::PointerType>(MojoASTTypeRef(type))) {
    if (pointeeType)
      *pointeeType = createCompilerType(pointerType.getElementAsType());
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
          createCompilerType(ptrType.getElementAsType());
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
    return lldb::eTypeHasChildren | lldb::eTypeIsArray;

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
        MojoASTTypeRef eltType;
        if (auto intType = getEquivalentIntegerType(getMLIRContext(),
                                                    kgenDTypeOpt.value()))
          eltType = intType;
        else if (auto floatType = getEquivalentFloatType(getMLIRContext(),
                                                         kgenDTypeOpt.value()))
          eltType = floatType;
        else
          return {};

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
  return {};
}

size_t MojoTypeSystem::GetIndexOfChildMemberWithName(
    lldb::opaque_compiler_type_t type, llvm::StringRef name,
    bool omitEmptyBaseClasses, std::vector<uint32_t> &childIndices) {
  // This method should return the total number of indices in `childIndices`
  // in the case of success. As a remark, the `childIndices` vector passed in
  // might not be empty.
  MojoASTTypeRef astType(type);

  // Check if the name is an index of a SIMD.
  if (isa<POP::SIMDType>(astType)) {
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
MojoTypeSystem::GetStructDecorators(lldb::opaque_compiler_type_t type) {
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
  lldb::TargetSP target = impl->target.lock();
  if (!target || ctxObj)
    return nullptr;
  return new MojoUserExpression(*target.get(), expr, prefix, language,
                                desiredType, options);
}

PersistentExpressionState *MojoTypeSystem::GetPersistentExpressionState() {
  return &impl->persistentState;
}

//===--------------------------------------------------------------------===//
// Dumping
//===--------------------------------------------------------------------===//

bool MojoTypeSystem::DumpTypeValue(
    lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
    lldb::Format format, const lldb_private::DataExtractor &data,
    lldb::offset_t dataOffset, size_t dataByteSize, uint32_t bitfieldBitSize,
    uint32_t bitfieldBitOffset, lldb_private::ExecutionContextScope *exeScope) {
  return lldb_private::DumpDataExtractor(
      data, &s, dataOffset, format, dataByteSize,
      /*itemCount=*/1, UINT32_MAX, LLDB_INVALID_ADDRESS, bitfieldBitSize,
      bitfieldBitOffset, exeScope);
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
