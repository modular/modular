//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/Broadcaster.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private.h"

namespace M {
class MojoASTTypeRef;
class MojoParserContext;
class TargetInfoAttr;
} // namespace M

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::Mojo {
/// Forward declaration for use below.
class MojoDiagnostic;

class MojoTypeSystem : public lldb_private::TypeSystem,
                       public lldb_private::Broadcaster {
  static char ID;

public:
  MojoTypeSystem(lldb_private::Target &target);
  ~MojoTypeSystem() override;

  /// Return the MLIR context for this type system.
  MLIRContext *getMLIRContext();

  /// Return the LLCL runtime for this type system.
  LLCL::Runtime &getRuntime();

  /// Return the Mojo parser context attached to this type system.
  MojoParserContext &getParserContext();

  /// Return the target info that corresponds to the current LLDB target, it
  /// might be invalid if it couldn't be computed.
  TargetInfoAttr GetTargetInfo() const;

  /// Return if the given language is supported by this type system.
  bool SupportsLanguage(lldb::LanguageType language) override {
    return language == lldb::eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  static void Initialize();
  static void Terminate();

  llvm::StringRef GetPluginName() override { return getPluginNameStatic(); }
  static llvm::StringRef getPluginNameStatic() { return "Mojo"; }

  //===--------------------------------------------------------------------===//
  // Broadcaster
  //===--------------------------------------------------------------------===//

  /// The convention for message naming is that a `Message` suffix means
  /// something we should display to the user, while other suffixes are used for
  /// various kinds of logging.
  enum MessageKind : uint32_t {
    /// Informational messages related to Mojo targets that are not part of
    /// the inferior's stderr or stdout but should still be displayed to the
    /// users when not using the CLI.
    eBroadcastUserMessage = (1u << 0),
    /// An IR dump that we are emitting for debug purposes. This will not be
    /// flushed to stderr unless `eFlushToStderr` is produced.
    eDumpIR = (1u << 1),
    /// A debug log message. This will not be flushed to stderr unless
    /// `eFlushToStderr` is produced.
    eDebugLog = (1u << 2),
    /// A signal that the plugin should flush its IR and log buffers to the
    /// stderr.
    eFlushIRAndDebugLog = (1u << 3),
    /// A log message that we should always flush to the stderr.
    eErrorLog = (1u << 4),
    /// A log message that we show on a crash.
    eCrashLog = (1u << 5),
    /// A mask that we can use to listen for all MojoTypeSystem messages.
    eAllMessagesMask = (1u << 6) - 1,
  };

  void broadcastUserMessage(StringRef message);

  /// Log the provided IR, copying the underlying bytes into the Event object
  /// (to avoid lifetime issues).
  void dumpIR(StringRef message);
  /// Use llvm::formatv to log an IR.
  template <typename... Args>
  void dumpIR(StringRef fmt, Args &&...args) {
    dumpIR(llvm::formatv(fmt.data(), std::forward<Args>(args)...).str());
  }

  /// Log the provided message, copying the underlying bytes into the Event
  /// object (to avoid lifetime issues).
  void debugLog(StringRef message);
  /// Use llvm::formatv to log a message.
  template <typename... Args>
  void debugLog(StringRef fmt, Args &&...args) {
    debugLog(llvm::formatv(fmt.data(), std::forward<Args>(args)...).str());
  }

  /// Flush the debug logs.
  void flushIRDumpAndDebugLog();

  /// Log an error message, copying the underlying bytes into the Event object
  /// (to avoid lifetime issues).
  void errorLog(StringRef message);
  /// Use llvm::formatv to log a message.
  template <typename... Args>
  void errorLog(StringRef fmt, Args &&...args) {
    errorLog(llvm::formatv(fmt.data(), std::forward<Args>(args)...).str());
  }

  /// Log an error message, copying the underlying bytes into the Event object
  /// (to avoid lifetime issues).
  void crashLog(StringRef message);
  /// Use llvm::formatv to log a message.
  template <typename... Args>
  void crashLog(StringRef fmt, Args &&...args) {
    crashLog(llvm::formatv(fmt.data(), std::forward<Args>(args)...).str());
  }

  /// Broadcast the diagnostics within the given diagnostic manager. An optional
  /// filter function can be provided to determine which diagnostics should be
  /// included in the output.
  void broadcastDiagnostics(lldb_private::DiagnosticManager &diagnosticManager,
                            function_ref<bool(MojoDiagnostic &)> filter = {});

  /// This function provides a reasonable default message handling policy. Users
  /// that want different behavior are encouraged to provide their own handler.
  /// The behavior of this function is to:
  ///  - Send eBroadcastUserMessage to `sendUserOutput`
  ///  - Send eIRMessage to `debugMessageCache`
  ///  - Treat eDebugMessage same as eIRMessage
  ///  - On eFlushToStderr flush `debugMessageCache` to `reportMessage`
  ///  - Send eErrorMessage to `reportMessage`
  /// `debugMessageCache` is capped at a static number of the most recent items.
  /// The first argument to `reportMessage` is the string-ified version of the
  /// message kind.
  static void handleEvent(
      const lldb::EventSP &event,
      std::deque<std::pair<MessageKind, std::string>> &debugMessageCache,
      function_ref<void(StringRef, StringRef)> reportMessage,
      function_ref<void(StringRef)> sendUserOutput);

  //===--------------------------------------------------------------------===//
  // Dumping
  //===--------------------------------------------------------------------===//

  void Dump(llvm::raw_ostream &output) override {}

  void DumpValue(lldb::opaque_compiler_type_t type,
                 lldb_private::ExecutionContext *exeCtx,
                 lldb_private::Stream &s, lldb::Format format,
                 const lldb_private::DataExtractor &data,
                 lldb::offset_t dataOffset, size_t dataByteSize,
                 uint32_t bitfieldBitSize, uint32_t bitfieldBitOffset,
                 bool showTypes, bool showSummary, bool verbose,
                 uint32_t depth) override {}

  void DumpSummary(lldb::opaque_compiler_type_t type,
                   lldb_private::ExecutionContext *exeCtx,
                   lldb_private::Stream &s,
                   const lldb_private::DataExtractor &data,
                   lldb::offset_t dataOffset, size_t dataByteSize) override {}

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
                     lldb::Format format,
                     const lldb_private::DataExtractor &data,
                     lldb::offset_t dataOffset, size_t dataByteSize,
                     uint32_t bitfieldBitSize, uint32_t bitfieldBitOffset,
                     lldb_private::ExecutionContextScope *exeScope) override;

#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump(lldb::opaque_compiler_type_t type) const override {
  }
#endif

  /// Dump the type to stdout.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  /// Print a description of the type to a stream. The exact implementation
  /// varies, but the expectation is that eDescriptionLevelFull returns a
  /// source-like representation of the type, whereas eDescriptionLevelVerbose
  /// does a dump of the underlying AST if applicable.
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, lldb_private::Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) override {}

  //===--------------------------------------------------------------------===//
  // Type Queries
  //===--------------------------------------------------------------------===//

#ifndef NDEBUG
  /// Verify the integrity of the type to catch CompilerTypes that mix
  /// and match invalid TypeSystem/Opaque type pairs.
  bool Verify(lldb::opaque_compiler_type_t type) override {
    // MLIR type construction should already handle verifying the necessary
    // invariants here.
    return true;
  }
#endif

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override {
    return lldb::eLanguageTypeMojo;
  }

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }

  bool CanPassInRegisters(const lldb_private::CompilerType &type) override {
    return false;
  }

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  const llvm::fltSemantics &GetFloatTypeSemantics(size_t byteSize) override {
    return llvm::APFloatBase::Bogus();
  }

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType
  GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                             const size_t index) override {
    return {};
  }

  uint32_t GetPointerByteSize() override { return 8; }

  lldb_private::ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                                        bool baseOnly) override;

  lldb_private::ConstString
  GetDisplayTypeName(lldb::opaque_compiler_type_t type) override;

  uint32_t GetTypeInfo(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *pointeeOrElementCompilerType) override;

  /// An overload of GetTypeInfo that uses a null
  /// `pointeeOrElementCompilerType`.
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type) {
    return GetTypeInfo(type, /*pointeeOrElementCompilerType=*/nullptr);
  }

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  std::optional<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             lldb_private::ExecutionContextScope *exeScope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override {
    return {};
  }

  //===--------------------------------------------------------------------===//
  // DeclContext
  //===--------------------------------------------------------------------===//

  lldb_private::ConstString DeclGetName(void *opaqueDecl) override {
    return {};
  }
  lldb_private::CompilerType GetTypeForDecl(void *opaqueDecl) override {
    return {};
  }
  lldb_private::ConstString DeclContextGetName(void *opaqueDeclCtx) override {
    return {};
  }
  lldb_private::ConstString
  DeclContextGetScopeQualifiedName(void *opaqueDeclCtx) override {
    return {};
  }
  bool DeclContextIsClassMethod(void *opaqueDeclCtx) override { return {}; }

  bool DeclContextIsContainedInLookup(void *opaqueDeclCtx,
                                      void *otherOpaqueDeclCtx) override {
    return {};
  }

  lldb::LanguageType DeclContextGetLanguage(void *) override {
    return lldb::eLanguageTypeMojo;
  }

  //===--------------------------------------------------------------------===//
  // IsType Queries
  //===--------------------------------------------------------------------===//

  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsCharType(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsCompleteType(lldb::opaque_compiler_type_t type) override {
    return true;
  }
  bool IsConst(lldb::opaque_compiler_type_t type) override { return false; }
  bool IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count,
                           bool &isComplex) override;
  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &isSigned) override;
  bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsScalarType(lldb::opaque_compiler_type_t type) override;
  bool IsCStringType(lldb::opaque_compiler_type_t type,
                     uint32_t &length) override {
    return false;
  }
  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    lldb_private::CompilerType *elementType,
                    uint64_t *size) override {
    return false;
  }
  uint32_t
  IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                         lldb_private::CompilerType *baseTypePtr) override {
    return 0;
  }
  bool IsBlockPointerType(
      lldb::opaque_compiler_type_t type,
      lldb_private::CompilerType *functionPointerTypePtr) override {
    return false;
  }
  bool IsMemberFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override {
    return false;
  }
  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool
  IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                           lldb_private::CompilerType *pointeeType) override;

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       lldb_private::CompilerType *pointeeType,
                       bool *isRValue) override;

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   lldb_private::CompilerType *elementType, uint64_t *size,
                   bool *isIncomplete) override {
    return false;
  }

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsDefined(lldb::opaque_compiler_type_t type) override { return false; }

  bool IsFunctionType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return false;
  }

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             lldb_private::CompilerType *targetType,
                             bool checkCplusplus, bool checkObjc) override {
    return false;
  }

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     lldb_private::CompilerType *pointeeType) override {
    return false;
  }

  bool IsVoidType(lldb::opaque_compiler_type_t type) override { return false; }

  //===--------------------------------------------------------------------===//
  // GetType Queries
  //===--------------------------------------------------------------------===//

  lldb_private::CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) override {
    return {};
  }
  lldb_private::CompilerType
  GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    return {};
  }
  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override {
    return lldb::eBasicTypeInvalid;
  }

  lldb_private::CompilerType
  GetLValueReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetRValueReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetNonReferenceType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  lldb_private::ExecutionContextScope *exeScope) override {
    return {};
  }

  lldb_private::CompilerType
  GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                      size_t bitSize) override {
    return {};
  }

  lldb_private::CompilerType
  GetTypedefedType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetArrayElementType(lldb::opaque_compiler_type_t type,
                      lldb_private::ExecutionContextScope *exeScope) override {
    return {};
  }

  lldb_private::CompilerType GetArrayType(lldb::opaque_compiler_type_t type,
                                          uint64_t size) override {
    return {};
  }

  lldb_private::CompilerType
  GetCanonicalType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override {
    return -1;
  }

  lldb_private::CompilerType
  GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                 size_t idx) override {
    return {};
  }

  lldb_private::CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override {
    return {};
  }

  lldb_private::CompilerType
  GetPointeeType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  lldb_private::CompilerType
  GetPointerType(lldb::opaque_compiler_type_t type) override {
    return {};
  }

  //===--------------------------------------------------------------------===//
  // Type Navigation
  //===--------------------------------------------------------------------===//

  uint32_t
  GetNumChildren(lldb::opaque_compiler_type_t type, bool omitEmptyBaseClasses,
                 const lldb_private::ExecutionContext *exeCtx) override;

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                             size_t idx, std::string &name,
                                             uint64_t *bitOffsetPtr,
                                             uint32_t *bitfieldBitSizePtr,
                                             bool *isBitfieldPtr) override {
    return {};
  }

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  lldb_private::CompilerType
  GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                            uint32_t *bitOffsetPtr) override {
    return {};
  }

  lldb_private::CompilerType
  GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                             uint32_t *bitOffsetPtr) override {
    return {};
  }

  lldb_private::CompilerType GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, lldb_private::ExecutionContext *exeCtx,
      size_t idx, bool transparent_pointers, bool omitEmptyBaseClasses,
      bool ignoreArrayBounds, std::string &childName, uint32_t &childByteSize,
      int32_t &childByteOffset, uint32_t &childBitfieldBitSize,
      uint32_t &childBitfieldBitOffset, bool &childIsBaseClass,
      bool &childIsDerefOfParent, lldb_private::ValueObject *valobj,
      uint64_t &languageFlags) override;

  uint32_t GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                   StringRef name,
                                   bool omitEmptyBaseClasses) override {
    return 0;
  }

  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name, bool omitEmptyBaseClasses,
                                std::vector<uint32_t> &childIndices) override;

  //===--------------------------------------------------------------------===//
  // Expressions
  //===--------------------------------------------------------------------===//

  /// Return a new user expression for the given expression text, or nullptr in
  /// the case of an error.
  lldb_private::UserExpression *
  GetUserExpression(StringRef expr, StringRef prefix,
                    lldb::LanguageType language,
                    lldb_private::Expression::ResultType desiredType,
                    const lldb_private::EvaluateExpressionOptions &options,
                    lldb_private::ValueObject *ctxObj) override;

  /// Return a pointer to the persistent expression state for this type system.
  lldb_private::PersistentExpressionState *
  GetPersistentExpressionState() override;

  //===--------------------------------------------------------------------===//
  // Utils
  //===--------------------------------------------------------------------===//

  /// Create a CompilerType for the given MLIR type.
  lldb_private::CompilerType createCompilerType(mlir::Type type);

  /// Create a CompilerType for the given MojoASTTypeRef.
  lldb_private::CompilerType createCompilerType(MojoASTTypeRef type);

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  bool isA(const void *classID) const override { return classID == &ID; }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }

protected:
  struct Impl;

  std::unique_ptr<Impl> impl;
};
} // namespace M::KGEN::Mojo

/// Allow cast<MojoTypeSystem>(lldb::TypeSystemSP) ->
/// std::shared_ptr<MojoTypeSystem>. This is necessary because the standard LLVM
/// infra does not support std::shared_ptr.
namespace llvm {
template <>
struct CastInfo<M::KGEN::Mojo::MojoTypeSystem, lldb::TypeSystemSP> {
  using To = std::shared_ptr<M::KGEN::Mojo::MojoTypeSystem>;
  using From = lldb::TypeSystemSP;
  static inline bool isPossible(From &f) {
    return llvm::isa<M::KGEN::Mojo::MojoTypeSystem>(&*f);
  }

  static To doCast(From &f) {
    return std::static_pointer_cast<M::KGEN::Mojo::MojoTypeSystem>(f);
  }

  static inline To castFailed() { return nullptr; }

  static To doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return doCast(f);
  }
};

template <>
struct CastInfo<M::KGEN::Mojo::MojoTypeSystem, const lldb::TypeSystemSP>
    : public ConstStrippingForwardingCast<
          M::KGEN::Mojo::MojoTypeSystem, const lldb::TypeSystemSP,
          CastInfo<M::KGEN::Mojo::MojoTypeSystem, lldb::TypeSystemSP>> {};
} // namespace llvm

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOTYPESYSTEM_H
