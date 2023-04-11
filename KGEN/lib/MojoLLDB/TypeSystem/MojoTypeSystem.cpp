//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoTypeSystem.h"
#include "../ExpressionParser/MojoExpressionVariable.h"
#include "../ExpressionParser/MojoUserExpression.h"
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/SymbolExport.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

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
    registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                    mlir::index::IndexDialect, mlir::LLVM::LLVMDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    // Set up the dialects in the context.
    mlirContext.appendDialectRegistry(registry);
    mlirContext.loadAllAvailableDialects();

    // Allow unregistered dialects, we will verify we know what to do with it
    // later.
    mlirContext.allowUnregisteredDialects();

    // Configure the runtime.
    runtime = std::make_unique<LLCL::Runtime>(
        LLCL::createMallocAllocator(), LLCL::createThreadPoolWorkQueue());
  }

  /// The MLIR context to use for compilation/processing associated with this
  /// typesystem.
  MLIRContext mlirContext;

  /// The LLCL runtime to use for compilation/processing associated with this
  /// typesystem.
  std::unique_ptr<LLCL::Runtime> runtime;

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
LLCL::Runtime &MojoTypeSystem::getRuntime() { return *impl->runtime; }

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Create a MojoTypeSystem instance from the given module and target.
static lldb::TypeSystemSP createInstance(lldb::LanguageType language,
                                         Module *module, Target *target) {
  // TODO: Support creating a type system from a module.
  if (language != eLanguageTypeMojo || !target)
    return nullptr;
  return std::make_shared<MojoTypeSystem>(*target);
}

void MojoTypeSystem::Initialize() {
  LanguageSet languages;
  languages.Insert(eLanguageTypeMojo);
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
  lldb::EventSP event = std::make_shared<Event>(
      eBroadcastUserMessage | eDebugLog, new EventDataBytes(message));
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
  // If it's a user message broadcast, send that output.
  if (event->GetType() & MojoTypeSystem::eBroadcastUserMessage)
    sendUserOutput(getStringFromEvent(event));

  // It may (also) be one of `eDumpIR`, `eDebugLog`, `eFlushIRAndDebugLog`, or
  // `eErrorLog`. Flush that correctly.
  if (event->GetType() & MojoTypeSystem::eDumpIR ||
      event->GetType() & MojoTypeSystem::eDebugLog) {
    debugMessageCache.emplace_back(MessageKind(event->GetType()),
                                   getStringFromEvent(event));
  } else if (event->GetType() & MojoTypeSystem::eFlushIRAndDebugLog) {
    for (const auto &message : debugMessageCache)
      reportMessage(stringifyType(message.first), message.second);

    // Clear out the message cache.
    debugMessageCache.clear();
  } else if (event->GetType() & MojoTypeSystem::eErrorLog) {
    reportMessage(stringifyType(MessageKind(event->GetType())),
                  getStringFromEvent(event));
  } else {
    llvm::report_fatal_error("unknown message type");
  }

  // Pop the front message if we've exceeded 10 items in the deque.
  if (debugMessageCache.size() > 10)
    debugMessageCache.pop_front();
}

//===----------------------------------------------------------------------===//
// Type Queries
//===----------------------------------------------------------------------===//

bool MojoTypeSystem::ShouldTreatScalarValueAsAddress(
    lldb::opaque_compiler_type_t type) {
  return Flags(GetTypeInfo(type))
      .AnySet(lldb::eTypeInstanceIsPointer | lldb::eTypeIsReference);
}

lldb::Format MojoTypeSystem::GetFormat(lldb::opaque_compiler_type_t type) {
  auto flags = GetTypeInfo(type);
  if (flags & lldb::eTypeIsInteger)
    return lldb::eFormatDecimal;
  if (flags & lldb::eTypeIsFloat)
    return lldb::eFormatFloat;
  if (flags & lldb::eTypeIsPointer || flags & lldb::eTypeIsClass)
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
  // TODO: Realistically we should generically introspect the type and perform
  // whatever necessary transformations to determine what the size should be
  // when compiled. For now we just explicitly check for the single case that
  // we ever generate variables for, i.e., Pointers.
  return GetPointerByteSize() * CHAR_BIT;
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
