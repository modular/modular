//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOEXPRESSIONLOGGER_H
#define KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOEXPRESSIONLOGGER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "lldb/Utility/Broadcaster.h"
#include "llvm/Support/FormatVariadic.h"

namespace M::KGEN::Mojo {
class MojoDiagnostic;

/// Utility used to listen and emit logs related to expression evaluation of
/// mojo Targets. More information is available in the `Logging` section of
/// `MojoREPL.md`.
class MojoExpressionLogger : public lldb_private::Broadcaster {
public:
  /// The convention for message naming is that a `Message` suffix means
  /// something we should display to the user, while other suffixes are used for
  /// various kinds of logging.
  enum MessageKind : uint32_t {
    /// Informational messages related to Mojo targets that are not part of
    /// the inferior's stderr or stdout but should still be displayed to the
    /// users when not using the CLI.
    eBroadcastUserMessage = (1u << 0),
    /// An IR dump that we are emitting for debug purposes. This will not be
    /// emitted unless verbose logs are enabled.
    eDumpIR = (1u << 1),
    /// A debug log message. This will not be flushed to stderr unless
    /// `eFlushToStderr` is produced.
    eDebugLog = (1u << 2),
    /// A log message that we should always flush to the stderr.
    eErrorLog = (1u << 4),
    /// A mask that we can use to listen for all MojoTypeSystem messages.
    eAllMessagesMask = (1u << 6) - 1,
  };

  MojoExpressionLogger(lldb_private::Target &target);

  // Move-only type.
  MojoExpressionLogger(const MojoExpressionLogger &) = delete;
  MojoExpressionLogger &operator=(const MojoExpressionLogger &) = delete;

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

  /// Log an error message, copying the underlying bytes into the Event object
  /// (to avoid lifetime issues).
  void errorLog(StringRef message);
  /// Use llvm::formatv to log a message.
  template <typename... Args>
  void errorLog(StringRef fmt, Args &&...args) {
    errorLog(llvm::formatv(fmt.data(), std::forward<Args>(args)...).str());
  }

  /// Broadcast the diagnostics within the given diagnostic manager. An optional
  /// filter function can be provided to determine which diagnostics should be
  /// included in the output.
  void broadcastDiagnostics(lldb_private::DiagnosticManager &diagnosticManager,
                            function_ref<bool(MojoDiagnostic &)> filter = {});

  /// This function provides a reasonable default message handling policy. Users
  /// that want different behavior are encouraged to provide their own handler.
  /// The provided `sendUserOutput` function is used for user broadcast events
  /// and error logs. The first argument is the message kind, and the second is
  /// the message itself. If the message kind is `eErrorLog`, the type is
  /// "error". For `eBroadcastUserMessage`, the type is "user".
  MODULAR_VISIBILITY_EXPORT static void
  handleEvent(const lldb::EventSP &event,
              function_ref<void(StringRef, StringRef)> sendUserOutput);

  /// Get or create a unique expression logger for the given target.
  MODULAR_VISIBILITY_EXPORT static MojoExpressionLogger &
  getLoggerForTarget(lldb_private::Target &target);
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_TYPESYSTEM_MOJOEXPRESSIONLOGGER_H
