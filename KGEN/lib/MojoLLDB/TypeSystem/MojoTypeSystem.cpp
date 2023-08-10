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
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/POPDialect/POPTypes.h"
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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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

lldb_private::CompilerType
MojoTypeSystem::getCompilerTypeFromType(mlir::Type type) {
  return lldb_private::CompilerType(
      weak_from_this(), const_cast<void *>(type.getAsOpaquePointer()));
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
  return IsReferenceType(type, pointeeType, nullptr) ||
         IsPointerType(type, pointeeType);
}

bool MojoTypeSystem::IsReferenceType(lldb::opaque_compiler_type_t type,
                                     lldb_private::CompilerType *pointeeType,
                                     bool *isRValue) {
  MojoASTTypeRef refType(type);
  return (isa<POP::PointerType, LIT::REPLResultRefType>(refType.getMLIRType()));
}

uint32_t MojoTypeSystem::GetTypeInfo(
    lldb::opaque_compiler_type_t type,
    lldb_private::CompilerType *pointeeOrElementCompilerType) {
  if (!type)
    return 0;

  if (pointeeOrElementCompilerType)
    pointeeOrElementCompilerType->Clear();

  MojoASTTypeRef refType(type);
  Type mlirType = refType.getMLIRType();

  if (auto refType = dyn_cast<LIT::REPLResultRefType>(mlirType)) {
    return GetTypeInfo(
        getCompilerTypeFromType(refType.getElementType()).GetOpaqueQualType(),
        pointeeOrElementCompilerType);
  }

  if (auto ptrType = dyn_cast<POP::PointerType>(mlirType)) {
    if (pointeeOrElementCompilerType) {
      pointeeOrElementCompilerType->SetCompilerType(
          weak_from_this(),
          const_cast<void *>(ptrType.getElementAsType().getAsOpaquePointer()));
    }
    return lldb::eTypeIsPointer | lldb::eTypeHasChildren | lldb::eTypeHasValue;
  }

  if (isa<IndexType>(mlirType))
    return lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar;

  if (isa<IntegerType>(mlirType)) {
    auto result =
        lldb::eTypeIsInteger | lldb::eTypeHasValue | lldb::eTypeIsScalar;
    if (mlirType.isSignedInteger())
      return result | lldb::eTypeIsSigned;
    return result;
  }

  if (isa<FloatType>(mlirType))
    return lldb::eTypeIsFloat | lldb::eTypeHasValue | lldb::eTypeIsScalar;

  if (auto simd = dyn_cast<POP::SIMDType>(mlirType))
    return lldb::eTypeHasChildren | lldb::eTypeIsArray;

  if (auto declRef = getParserContext().getDecl(refType)) {
    if (llvm::isa_and_present<LIT::StructDeclOp>(declRef.getIfOperation()))
      return lldb::eTypeHasChildren | lldb::eTypeIsClass;
  }
  return {};
}

lldb::Format MojoTypeSystem::GetFormat(lldb::opaque_compiler_type_t type) {
  if (auto replRefType =
          dyn_cast<LIT::REPLResultRefType>(MojoASTTypeRef(type).getMLIRType()))
    return GetFormat(getCompilerTypeFromType(replRefType.getElementType())
                         .GetOpaqueQualType());

  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger)
    return lldb::eFormatDecimal;
  if (flags & lldb::eTypeIsFloat)
    return lldb::eFormatFloat;
  if (flags & lldb::eTypeIsPointer)
    return lldb::eFormatAddressInfo;
  if (flags & lldb::eTypeIsClass)
    return lldb::eFormatHex;
  if (flags & lldb::eTypeIsFuncPrototype || flags & lldb::eTypeIsBlock)
    return lldb::eFormatAddressInfo;
  return lldb::eFormatBytes;
}

std::optional<uint64_t>
MojoTypeSystem::GetBitSize(lldb::opaque_compiler_type_t type,
                           lldb_private::ExecutionContextScope *exeScope) {
  MojoASTTypeRef astType(type);
  Type mlirTy = astType.getMLIRType();
  if (mlirTy.isIntOrFloat())
    return mlirTy.getIntOrFloatBitWidth();
  if (mlirTy.isIndex())
    return GetPointerByteSize() * CHAR_BIT;
  // TODO: Use the target info to get the current target width and return that.
  return GetPointerByteSize() * CHAR_BIT;
}

ConstString MojoTypeSystem::GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) {
  if (!type)
    return {};

  MojoASTTypeRef astType(type);
  Type typeToPrint;
  // We don't want to expose the REPLResultRefType to the user, so we'll only
  // print the element type.
  if (auto replRef = dyn_cast<LIT::REPLResultRefType>(astType.getMLIRType()))
    typeToPrint = Type::getFromOpaquePointer(
        replRef.getElementType().getAsOpaquePointer());
  else
    typeToPrint = Type::getFromOpaquePointer(type);

  std::string name;
  llvm::raw_string_ostream os(name);
  typeToPrint.print(os);
  return ConstString(name);
}

ConstString
MojoTypeSystem::GetDisplayTypeName(lldb::opaque_compiler_type_t type) {
  if (!type)
    return {};

  MojoASTTypeRef astType(type);
  std::string name;

  // We don't want to expose the REPLResultRefType to the user, so we'll only
  // print the element type.
  if (auto replRef = dyn_cast<LIT::REPLResultRefType>(astType.getMLIRType()))
    name = MojoASTTypeRef(Type::getFromOpaquePointer(
                              replRef.getElementType().getAsOpaquePointer()))
               .getAsString();
  else
    name = MojoASTTypeRef(Type::getFromOpaquePointer(type)).getAsString();

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
  return (GetTypeInfo(type) & lldb::eTypeIsInteger);
}

bool MojoTypeSystem::IsScalarType(lldb::opaque_compiler_type_t type) {
  return (GetTypeInfo(type) & lldb::eTypeIsScalar);
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

  MojoASTTypeRef refType(type);
  Type mlirType = refType.getMLIRType();

  if (auto replRefType = dyn_cast<LIT::REPLResultRefType>(mlirType))
    return GetNumChildren(getCompilerTypeFromType(replRefType.getElementType())
                              .GetOpaqueQualType(),
                          omitEmptyBaseClasses, exeCtx);

  if (isa<POP::PointerType>(mlirType))
    return 1;

  if (auto simdTy = dyn_cast<POP::SIMDType>(mlirType)) {
    if (simdTy.isScalar())
      return 1;
    getParserContext().getDecl(refType);
    return simdTy.getResolvedSize().value_or(0);
  }

  if (auto declRef = getParserContext().getDecl(refType)) {
    if (Operation *op = declRef.getIfOperation()) {
      if (LIT::StructDeclOp structDeclOp = dyn_cast<LIT::StructDeclOp>(op)) {
        auto range = structDeclOp.getFieldDecls();
        return std::distance(range.begin(), range.end());
      }
    }
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

  if (idx >= GetNumChildren(type, omitEmptyBaseClasses, exeCtx))
    return {};

  MojoASTTypeRef refType(type);
  Type mlirType = refType.getMLIRType();

  // Unwrap the REPLResultRefType and return the child type of it's element type
  if (auto replRefType = dyn_cast<LIT::REPLResultRefType>(mlirType)) {
    return GetChildCompilerTypeAtIndex(
        getCompilerTypeFromType(replRefType.getElementType())
            .GetOpaqueQualType(),
        exeCtx, idx, transparent_pointers, omitEmptyBaseClasses,
        ignoreArrayBounds, childName, childByteSize, childByteOffset,
        childBitfieldBitSize, childBitfieldBitOffset, childIsBaseClass,
        childIsDerefOfParent, valobj, languageFlags);
  }

  // Pointer only has one child, so just return the unwrapped pointer type
  if (isa<POP::PointerType>(mlirType))
    return getCompilerTypeFromType(
        refType.getPointerElementType().getMLIRType());

  if (auto simdType = dyn_cast<POP::SIMDType>(mlirType)) {
    if (std::optional<KGENDType> kgenDTypeOpt = simdType.getResolvedDType()) {
      if (kgenDTypeOpt.has_value()) {
        Type eltType;
        if (auto intType = getEquivalentIntegerType(getMLIRContext(),
                                                    kgenDTypeOpt.value()))
          eltType = intType;
        else if (auto floatType = getEquivalentFloatType(getMLIRContext(),
                                                         kgenDTypeOpt.value()))
          eltType = floatType;
        else
          return {};

        if (!simdType.isScalar()) {
          // Set stuff required for printing array elements.
          childName = std::string(llvm::formatv("[{0}]", idx));
          childByteSize =
              GetBitSize(getCompilerTypeFromType(eltType).GetOpaqueQualType(),
                         exeCtx->GetBestExecutionContextScope())
                  .value() /
              8;
          childByteOffset = (int32_t)idx * (int32_t)childByteSize;
        }
        return getCompilerTypeFromType(eltType);
      }
    }
    return {};
  }
  auto declRef = getParserContext().getDecl(refType);
  if (LIT::StructDeclOp structDeclOp =
          dyn_cast_if_present<LIT::StructDeclOp>(declRef.getIfOperation())) {
    auto field = *std::next(structDeclOp.getFieldDecls().begin(), idx);
    childName.assign(field.getName());
    auto refType = cast<DeclRefType>(mlirType);
    MojoASTTypeRef fieldType = MojoASTTypeRef(field.getTypeAttr().getValue());

    // If the DeclRefType has parameters, try to evaluate and substitute them
    // into the type. This is needed for displaying SIMDs-- we have to resolve
    // their size here.
    if (!refType.getParamValues().empty())
      fieldType = getParserContext().concretizeType(refType.getParamValues(),
                                                    fieldType);

    return lldb_private::CompilerType(weak_from_this(),
                                      fieldType.getAsVoidPointer());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

UserExpression *MojoTypeSystem::GetUserExpression(
    StringRef expr, StringRef prefix, lldb::LanguageType language,
    Expression::ResultType desiredType, const EvaluateExpressionOptions &opts,
    ValueObject *ctxObj) {
  lldb::TargetSP target = impl->target.lock();
  if (!target || ctxObj)
    return nullptr;
  auto options = opts;
  // Disable timeout because the expression could take a very long time.
  options.SetTimeout(Timeout<std::micro>(std::nullopt));
  options.SetOneThreadTimeout(Timeout<std::micro>(std::nullopt));
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
