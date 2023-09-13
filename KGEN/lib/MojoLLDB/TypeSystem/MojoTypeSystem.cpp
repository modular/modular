//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoTypeSystem.h"
#include "../../MojoParser/ASTType.h"
#include "../ExpressionParser/MojoDiagnostic.h"
#include "../ExpressionParser/MojoExpressionParser.h"
#include "../ExpressionParser/MojoExpressionVariable.h"
#include "../ExpressionParser/MojoUserExpression.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLCL/Runtime/Runtime.h"
#include "MojoTypeDataLayout.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

/// Method used to get the correct MojoASTTypeRef of a given opaque type. It
/// also dereferences REPLResultRefType if present.
static MojoASTTypeRef dereferenceType(lldb::opaque_compiler_type_t type) {
  auto mlirType = mlir::Type::getFromOpaquePointer(type);
  if (auto refType = dyn_cast<LIT::REPLResultRefType>(mlirType))
    return MojoASTTypeRef(refType.getElementType());
  return MojoASTTypeRef(mlirType);
}

//===----------------------------------------------------------------------===//
// MojoTypeSystem::Impl
//===----------------------------------------------------------------------===//

struct MojoTypeSystem::Impl {
  Impl(Target &target) : target(target.shared_from_this()) {
    // Register all of the various dialect state.
    DialectRegistry registry;
    registerAllKGENDialects(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    // Set up the dialects in the context.
    mlirContext.appendDialectRegistry(registry);

    // Allow unregistered dialects, we will verify we know what to do with it
    // later.
    mlirContext.allowUnregisteredDialects();

    // Configure the runtime.
    runtime = std::make_unique<LLCL::Runtime>(
        LLCL::createMallocAllocator(), LLCL::createThreadPoolWorkQueue());

    // Compute the target information for the expression.
    // TODO: Populate cpu information properly here.
    ArchSpec targetArch = target.GetArchitecture();
    if (targetArch.IsValid())
      compilationOptions.targetTriple = targetArch.GetTriple().str();
    compilationOptions.targetCpu = llvm::sys::getHostCPUName();

    // Configure the parser context.
    MojoParserConfig parserConfig(&mlirContext, *runtime, compilationOptions);
    parserConfig.moduleCachingLevel = MojoParserConfig::kCacheNone;
    parserContext =
        std::make_unique<MojoParserContext>(sourceMgr, parserConfig);

    auto targetInfoOr = M::getTargetInfoFor(
        &mlirContext, compilationOptions.targetTriple,
        compilationOptions.targetCpu, compilationOptions.targetFeatures);
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
    : Broadcaster(target.GetDebugger().GetBroadcasterManager(),
                  "mojo-type-system.broadcaster"),
      impl(std::make_unique<Impl>(target)) {}
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
// Logging
//===----------------------------------------------------------------------===//

void MojoTypeSystem::broadcastUserMessage(StringRef message) {
  lldb::EventSP event = std::make_shared<Event>(eBroadcastUserMessage,
                                                new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoTypeSystem::dumpIR(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eDumpIR, new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoTypeSystem::debugLog(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eDebugLog, new EventDataBytes(message));
  BroadcastEvent(event);
}

void MojoTypeSystem::flushIRDumpAndDebugLog() {
  lldb::EventSP event = std::make_shared<Event>(eFlushIRAndDebugLog);
  BroadcastEvent(event);
}

void MojoTypeSystem::errorLog(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eErrorLog, new EventDataBytes(message));
  BroadcastEvent(event);
  // When we hit an error, we want to flush the debug logs as well.
  flushIRDumpAndDebugLog();
}

void MojoTypeSystem::crashLog(StringRef message) {
  lldb::EventSP event =
      std::make_shared<Event>(eCrashLog, new EventDataBytes(message));
  BroadcastEvent(event);
  // When we hit an error, we want to flush the debug logs as well.
  flushIRDumpAndDebugLog();
}

void MojoTypeSystem::broadcastDiagnostics(
    DiagnosticManager &diagnosticManager,
    function_ref<bool(MojoDiagnostic &)> filter) {
  debugLog("Emitted diagnostics");

  std::string msg;
  llvm::raw_string_ostream msgOS(msg);
  for (const auto &diag : diagnosticManager.Diagnostics()) {
    if (auto *mojoDiag = dyn_cast<MojoDiagnostic>(diag.get())) {
      if (filter && !filter(*mojoDiag))
        continue;
    }

    switch (diag->GetSeverity()) {
    case eDiagnosticSeverityError:
      msgOS << "error: ";

      // Log error diagnostics explicitly so they get captured in the error log,
      // the full diagnostic message will be available in the debug logs.
      errorLog(diag->GetMessage());
      break;
    case eDiagnosticSeverityWarning:
      msgOS << "warning: ";
      break;
    case eDiagnosticSeverityRemark:
      break;
    }
    msgOS << diag->GetMessage() << "\n";
  }
  if (!msg.empty())
    broadcastUserMessage(msg);
}

//===----------------------------------------------------------------------===//
// Listener Support
//===----------------------------------------------------------------------===//

/// Get a null-terminated string from an event.
static std::string getStringFromEvent(const lldb::EventSP &event) {
  size_t readLen = EventDataBytes::GetByteSizeFromEvent(event.get());
  const char *rawData =
      static_cast<const char *>(EventDataBytes::GetBytesFromEvent(event.get()));
  return {rawData, readLen};
}

/// Stringify the event type.
static std::string stringifyType(MojoTypeSystem::MessageKind type) {
  SmallVector<std::string, 1> typeStrs;
  if (type & MojoTypeSystem::eBroadcastUserMessage)
    typeStrs.push_back("BroadcastUser");
  if (type & MojoTypeSystem::eDumpIR)
    typeStrs.push_back("DumpIR");
  if (type & MojoTypeSystem::eDebugLog)
    typeStrs.push_back("DebugLog");
  if (type & MojoTypeSystem::eErrorLog)
    typeStrs.push_back("ErrorLog");
  if (type & MojoTypeSystem::eCrashLog)
    typeStrs.push_back("CrashLog");

  std::string out;
  llvm::raw_string_ostream outStream(out);
  llvm::interleave(typeStrs, outStream, "|");
  return out;
}

void MojoTypeSystem::handleEvent(
    const lldb::EventSP &event,
    std::deque<std::pair<MessageKind, std::string>> &debugMessageCache,
    function_ref<void(StringRef, StringRef)> reportMessage,
    function_ref<void(StringRef)> sendUserOutput) {
  assert(llvm::popcount(event->GetType()) == 1 &&
         "a message must contain one single type");

  auto addEventToDebugMessageCache = [&] {
    debugMessageCache.emplace_back(MessageKind(event->GetType()),
                                   getStringFromEvent(event));
    // Pop the front message if we've exceeded 40 items in the deque.
    if (debugMessageCache.size() > 40)
      debugMessageCache.pop_front();
  };

  if (event->GetType() & MojoTypeSystem::eBroadcastUserMessage) {
    // If it's a user message broadcast, send that output.
    sendUserOutput(getStringFromEvent(event));
    addEventToDebugMessageCache();
  } else if (event->GetType() &
             (MojoTypeSystem::eErrorLog | MojoTypeSystem::eCrashLog)) {
    // If it's an error log, send that output as well.
    reportMessage(stringifyType(MessageKind(event->GetType())),
                  getStringFromEvent(event));
  } else if (event->GetType() & (eDumpIR | eDebugLog)) {
    // These logs are only displayed right away if the LLDB expr logs are
    // enabled.
    if (Log *log = GetLog(LLDBLog::Expressions)) {
      LLDB_LOG(log, "[{0}] {1}", stringifyType(MessageKind(event->GetType())),
               getStringFromEvent(event));
    }

    // These messages are extremely noisy, so we don't want to add them to the
    // cache by default.
    if (llvm::sys::Process::GetEnv("MOJO_REPL_VERBOSE_LOG"))
      addEventToDebugMessageCache();
  } else if (event->GetType() & MojoTypeSystem::eFlushIRAndDebugLog) {
    for (const auto &message : debugMessageCache)
      reportMessage(stringifyType(message.first), message.second);

    // Clear out the message cache.
    debugMessageCache.clear();
  } else {
    llvm_unreachable("Unexpected message type");
  }
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
  if (!type)
    return false;

  if (auto refType = dyn_cast<LIT::REPLResultRefType>(MojoASTTypeRef(type))) {
    if (pointeeType)
      *pointeeType = createCompilerType(refType.getElementType());
    return true;
  }
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

  MojoASTTypeRef astType = dereferenceType(type);

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
    return lldb::eFormatAddressInfo;
  }
  if (flags & lldb::eTypeIsClass)
    return lldb::eFormatHex;
  if (flags & lldb::eTypeIsFuncPrototype || flags & lldb::eTypeIsBlock)
    return lldb::eFormatAddressInfo;
  return lldb::eFormatBytes;
}

lldb_private::CompilerType
MojoTypeSystem::GetNonReferenceType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(dereferenceType(type));
}

lldb_private::CompilerType
MojoTypeSystem::GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) {
  return createCompilerType(dereferenceType(type));
}

uint32_t MojoTypeSystem::GetPointerByteSize() {
  return impl->targetInfo.getDataLayout().getPointerSize();
}

std::optional<uint64_t>
MojoTypeSystem::GetBitSize(lldb::opaque_compiler_type_t type,
                           lldb_private::ExecutionContextScope *exeScope) {
  if (!type)
    return {};

  // We need this to handle correctly REPLResultRefType.
  if (IsPointerOrReferenceType(type, /*pointeeType=*/nullptr))
    return GetPointerByteSize() * CHAR_BIT;

  if (auto &layout =
          impl->dataLayoutContext->getOrCalculate(dereferenceType(type))) {
    return layout->getByteSize() * CHAR_BIT;
  }

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
  dereferenceType(type).getMLIRType().print(os);
  return ConstString(name);
}

ConstString
MojoTypeSystem::GetDisplayTypeName(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};

  std::string name = dereferenceType(type).getAsString();

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

  MojoASTTypeRef astType = dereferenceType(type);

  if (isa<KGEN::PointerType>(astType))
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

  MojoASTTypeRef astType = dereferenceType(type);

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
  // This method should return the total number of indices in `childIndices` in
  // the case of success. As a remark, the `childIndices` vector passed in might
  // not be empty.
  MojoASTTypeRef astType = dereferenceType(type);

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
